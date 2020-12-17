from typing import Iterable
from typing import Dict, List, Optional, Union

import pandas as pd
import pyarrow.parquet as pq

import ray.util.iter as para_iter
from .dataset import MLDataset
from .reader import divide_data_pieces, DataPiece, OutOfIndexException, SourceReader


class ParquetFileDataPiece(DataPiece):
    def __init__(self,
                 piece: pq.ParquetDatasetPiece,
                 columns: Optional[List[str]],
                 partitions: Optional[pq.ParquetPartitions]):
        super(ParquetFileDataPiece, self).__init__()

        self._row_groups = []
        if piece.row_group is None:
            for number in piece.get_metadata().to_dict()["num_row_groups"]:
                self._row_groups.append(
                    pq.ParquetDatasetPiece(piece.path, piece.open_file_func,
                                           piece.file_options, number,
                                           piece.partition_keys))
        else:
            self._row_groups.append(piece)
        self._num_rows = piece.get_metadata().to_dict()["num_rows"]
        self._columns = columns
        self._partitions = partitions

    def __iter__(self) -> Iterable[pd.DataFrame]:
        return []

    def read(self, batch_index: int) -> pd.DataFrame:
        if batch_index < len(self._row_groups):
            return self._row_groups[batch_index].read(
                columns=self._columns,
                use_threads=False,
                partitions=self._partitions).to_pandas()
        else:
            raise OutOfIndexException

    @property
    def num_records(self) -> int:
        return self._num_rows


class ParquetSourceShard(SourceReader):
    def __init__(self,
                 shard_id: int,
                 data_pieces: List[ParquetFileDataPiece],
                 max_parallel: int = 1,
                 resources: Dict = None):
        super(ParquetSourceShard, self).__init__(
            shard_id, max_parallel, resources)
        self._data_pieces = data_pieces

    def prefix(self) -> str:
        return "Parquet"

    @property
    def shard_id(self) -> int:
        return self._shard_id

    def get_data_pieces(self) -> List[DataPiece]:
        return self._data_pieces


def read_parquet(paths: Union[str, List[str]],
                 num_shards: int,
                 rowgroup_split: bool = True,
                 shuffle: bool = False,
                 columns: Optional[List[str]] = None,
                 max_parallel: int = 1,
                 resources: Dict = None,
                 **kwargs) -> MLDataset:
    """Read parquet format data from hdfs like filesystem into a MLDataset.

    .. code-block:: python

        # create dummy data
        spark.range(...).write.parquet(...)
        # create MLDataset
        data = ray.util.data.read_parquet(...)
        # convert to TorchMLDataset
        ds = data.to_torch(feature_columns=..., label_column=...)
        # get the given shard data
        shard = ds.get_shard(0)
        # create the DataLoader from the shard data and this can be used for
        # the TorchTrainer data creator as well
        data = DataLoader(shard, batch_size=32)

    Args:
        paths (Union[str, List[str]): a single file path or a list of file path
        num_shards (int): the number of shards
        rowgroup_split (bool): whether split the files into shards based on
            rowgroup. If set False, each shard will have a list of files.
        shuffle (bool): whether shuffle the ParquetDatasetPiece order when
            divide into shards
        columns (Optional[List[str]]): a list of column names to read
        max_parallel (int): the maximum parallelisms to support read in
            concurrent for each shard
        resources (Dict): the remote function resources for read data in
            parallel
        kwargs: the other parquet read options
    Returns:
        A MLDataset
    """
    pq_ds = pq.ParquetDataset(paths, **kwargs)
    pieces = pq_ds.pieces
    file_pieces = []
    if rowgroup_split:
        # split base on rowgroup
        for piece in pieces:
            metadata = piece.get_metadata().to_dict()
            num_row_groups = metadata["num_row_groups"]
            for i in range(num_row_groups):
                data_piece = pq.ParquetDatasetPiece(piece.path,
                                                    piece.open_file_func,
                                                    piece.file_options, i,
                                                    piece.partition_keys)
                file_pieces.append(ParquetFileDataPiece(
                    data_piece, columns, pq_ds.partitions))
    else:
        # split base on file pieces
        for piece in pieces:
            file_pieces.append(
                ParquetFileDataPiece(piece, columns, pq_ds.partitions))

    if len(file_pieces) < num_shards:
        raise ValueError(f"number of data pieces: {len(file_pieces)} should "
                         f"larger than num_shards: {num_shards}")

    rank_to_pieces = divide_data_pieces(file_pieces, num_shards)
    shards = []
    for i, pieces in rank_to_pieces.items():
        shards.append(ParquetSourceShard(i, pieces, max_parallel, resources))
    it = para_iter.from_iterators(shards, False, "parquet")
    return MLDataset.from_parallel_it(it, batch_size=0, repeated=False)
