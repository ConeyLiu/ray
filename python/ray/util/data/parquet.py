from typing import Callable, Dict, List, Optional, Union
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq

from .dataset import MLDataset
from .reader import Reader, SourceReader


class OutOfIndexException(Exception):
    pass


class ParquetFileDataPiece:
    def __init__(self,
                 piece: pq.ParquetDatasetPiece,
                 columns: Optional[List[str]],
                 partitions: Optional[pq.ParquetPartitions]):
        self._batch_index = 0
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

    def read(self, batch_index: int) -> pd.DataFrame:
        if batch_index < len(self._row_groups):
            return self._row_groups[batch_index].read(
                columns=self._columns,
                use_threads=False,
                partitions=self._partitions).to_pandas()
        else:
            raise OutOfIndexException

    def read_remote(self, batch_index: int) -> Callable:
        def f():
            return self.read(batch_index)
        return f

    @property
    def num_records(self) -> int:
        return self._num_rows

    def __iter__(self) -> Iterable[pd.DataFrame]:
        while True:
            try:
                value = self.read(self._batch_index)
                yield value
                self._batch_index += 1
            except OutOfIndexException:
                break


class ParquetSourceShard(SourceReader):
    def __init__(self,
                 shard_id: int,
                 data_pieces: List[ParquetFileDataPiece],
                 max_parallel: int = 1,
                 resources: Dict = None,
                 balance_mode: bool = True):
        super(ParquetSourceShard, self).__init__(
            shard_id, max_parallel, resources)
        self._data_pieces = data_pieces
        self._balance_mode = balance_mode

    def prefix(self) -> str:
        return "Parquet"

    @property
    def shard_id(self) -> int:
        return self._shard_id

    @property
    def num_records(self) -> int:
        return sum(p.num_records for p in self._data_pieces)

    def read(self) -> Iterable[pd.DataFrame]:
        for piece in self._data_pieces:
            for pdf in iter(piece):
                yield pdf

    def read_parallel(self) -> Iterable[Callable]:
        if not self._balance_mode:
            for piece in self._data_pieces:
                batch_index = 0
                while True:
                    try:
                        yield piece.read_remote(batch_index)
                        batch_index += 1
                    except OutOfIndexException:
                        break
        else:
            data_pieces = self._data_pieces.copy()
            batch_indexes = [0] * len(data_pieces)
            index = 0
            while True:
                piece = data_pieces[index]
                try:
                    yield piece.read_remote(batch_indexes[index])
                    batch_indexes[index] += 1
                    index = (index + 1) % len(data_pieces)
                except OutOfIndexException:
                    data_pieces.pop(index)
                    batch_indexes.pop(index)
                    if len(data_pieces) == 0:
                        break
                    else:
                        index = index % len(data_pieces)


class ParquetReader(Reader):
    def __init__(self,
                 paths: Union[str, List[str]],
                 num_shards: int,
                 batch_size: int,
                 rowgroup_split: bool = True,
                 columns: Optional[List[str]] = None,
                 max_parallel: int = 1,
                 resources: Dict = None,
                 **read_options):
        self._paths = paths
        self._num_shards = num_shards
        self._batch_size = batch_size
        self._rowgroup_split = rowgroup_split
        self._columns = columns
        self._max_parallel = max_parallel
        self._resources = resources
        self._read_options = read_options

        pq_ds = pq.ParquetDataset(paths, **read_options)
        file_pieces = self._list_file_pieces(pq_ds, rowgroup_split, columns)
        self._source_shards = self._create_source_shards(file_pieces)

    def _list_file_pieces(self, pq_ds, rowgroup_split, columns):
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
        return file_pieces

    def _create_source_shards(self,
                              file_pieces: List[ParquetFileDataPiece]):
        num_shards = self._num_shards
        if len(file_pieces) < num_shards:
            raise ValueError(f"number of data pieces: {len(file_pieces)} should "
                             f"larger than num_shards: {num_shards}")
        rank_to_pieces = divide_data_pieces(file_pieces, num_shards)
        shards = []
        for i, pieces in rank_to_pieces.items():
            shards.append(
                ParquetSourceShard(i, pieces, self.max_parallel(), self.resources()))
        return shards

    def batch_size(self) -> int:
        return self._batch_size

    def num_shards(self) -> int:
        return self._num_shards

    def repartition(self, num_partitions: int):
        if self._num_shards != num_partitions:
            return ParquetReader(
                self._paths, num_partitions, self._batch_size, self._rowgroup_split, self._columns,
                self._max_parallel, self._resources, **self._read_options)

    def get_shard(self, shard_id) -> "SourceReader":
        assert shard_id < self._num_shards
        return self._source_shards[shard_id]

    def repeated(self) -> bool:
        return False

    def max_parallel(self) -> int:
        return self._max_parallel

    def resources(self) -> Dict:
        return self._resources


def divide_data_pieces(data_pieces: List[ParquetFileDataPiece],
                       world_size: int) -> Dict:
    """Divide the data pieces into world_size partitions

    Divide the data pieces into world_size partitions and return a word rank
    to a list of data pieces mapping.
    Args:
        data_pieces (List[DataPiece]): the data pieces to be divided
        world_size (int): the data pieces will be divided into world_size
                          partitions
    Returns:
        a dict, the key is the world rank, and the value the data pieces
    """
    if len(data_pieces) < world_size:
        raise Exception("do not have enough data pieces to divide")
    results = {}
    tmp_queue = {}
    for i in range(world_size):
        results[i] = []
        tmp_queue[i] = 0
    sorted_pieces = sorted(data_pieces,
                           key=lambda item: item.num_records,
                           reverse=True)
    for piece in sorted_pieces:
        rank = sorted(tmp_queue, key=lambda x: tmp_queue[x])[0]
        results[rank].append(piece)
        tmp_queue[rank] = tmp_queue[rank] + piece.num_records

    return results


def read_parquet(paths: Union[str, List[str]],
                 num_shards: int,
                 batch_size: int = 128,
                 rowgroup_split: bool = True,
                 columns: Optional[List[str]] = None,
                 max_parallel: int = 1,
                 resources: Dict = None,
                 **read_options) -> MLDataset:
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
        columns (Optional[List[str]]): a list of column names to read
        max_parallel (int): the maximum parallelisms to support read in
            concurrent for each shard
        resources (Dict): the remote function resources for read data in
            parallel
        read_options: the other parquet read options
    Returns:
        A MLDataset
    """
    reader = ParquetReader(paths, num_shards, batch_size, rowgroup_split, columns,
                           max_parallel, resources, **read_options)
    return MLDataset.from_reader("parquet", reader)
