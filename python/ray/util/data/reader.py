from typing import Callable, Dict, List, Iterable, Union

import pandas as pd


class OutOfIndexException(Exception):
    pass


class DataPiece:
    def __init__(self):
        self._batch_index = 0

    @property
    def num_records(self) -> int:
        """The number of records for this data piece.

        This should be equal with sum of all pandas DataFrame rows.
        """
        raise NotImplementedError

    def setup(self, epoch: int):
        self._batch_index = 0

    def read(self, batch_index: int) -> pd.DataFrame:
        raise OutOfIndexException

    def read_remote(self, batch_index: int) -> Callable:
        def f():
            return self.read(batch_index)
        return f

    def __iter__(self) -> Iterable[pd.DataFrame]:
        while True:
            try:
                value = self.read(self._batch_index)
                yield value
                self._batch_index += 1
            except OutOfIndexException:
                break


class ShardReader:
    """A interface for source data reading"""

    def __init__(self,
                 shard_id: int,
                 max_parallel: int = 1,
                 resources: Dict = None,
                 balance_mode: bool = True):
        self._shard_id = shard_id
        self._max_parallel = max_parallel
        self._resources = resources or {}
        self._balance_mode = balance_mode
        self.epoch = 0

    @property
    def shard_id(self) -> int:
        return self._shard_id

    @property
    def num_records(self) -> int:
        """The number of records for this source shard.

        This should be equal with sum of all pandas DataFrame rows.
        """
        return sum([p.num_records for p in self.get_data_pieces()])

    def set_epoch(self, epoch):
        self.epoch = epoch

    def prefix(self) -> str:
        raise NotImplementedError

    def get_data_pieces(self) -> List[DataPiece]:
        raise NotImplementedError

    def max_parallel(self) -> int:
        return self._max_parallel

    def resources(self):
        return self._resources

    def read(self) -> Iterable[pd.DataFrame]:
        for piece in self.get_data_pieces():
            for pdf in iter(piece):
                yield pdf

    def read_parallel(self) -> Iterable[Callable]:
        if not self._balance_mode:
            for piece in self.get_data_pieces():
                batch_index = 0
                while True:
                    try:
                        yield piece.read_remote(batch_index)
                        batch_index += 1
                    except OutOfIndexException:
                        break
        else:
            data_pieces = self.get_data_pieces().copy()
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

    def __iter__(self) -> Union[Iterable[pd.DataFrame], Iterable[Callable]]:
        # set up
        [piece.setup(self.epoch) for piece in self.get_data_pieces()]
        self.epoch += 1
        if self.max_parallel() > 1:
            return iter(self.read_parallel())
        else:
            return iter(self.read())

    def __str__(self):
        return repr(self)

    def __repr__(self):
        if self.max_parallel() > 1:
            suffix = f"_parallel[{self._max_parallel}]"
        else:
            suffix = ""
        return f"{self.prefix()}SourceShard[{self.shard_id}]" + suffix


class Reader:
    def repartition(self, num_partitions: int):
        raise NotImplementedError

    def get_shard(self, shard_id) -> "ShardReader":
        raise NotImplementedError


def divide_data_pieces(data_pieces: List[DataPiece],
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
