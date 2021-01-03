from typing import Callable, Dict, List, Iterable, Union, TypeVar

import pandas as pd

from ray.util.iter import ParallelIterator


class OutOfIndexException(Exception):
    pass


class SourceReader:
    """A interface for given shard source data reading

    Args:
        shard_id (int): the shard id
        max_parallel (int): the maximum parallelism for this source data
            reading. This is should be same as Reader.max_parallel()
        resources (Dict): the resources required for the parallel data
            reading. This is should be same as Reader.resources()
    """

    def __init__(self,
                 shard_id: int,
                 max_parallel: int = 1,
                 resources: Dict = None):
        self._shard_id = shard_id
        self._max_parallel = max_parallel
        self._resources = resources or {}
        self.epoch = 0

    @property
    def shard_id(self) -> int:
        return self._shard_id

    @property
    def num_records(self) -> int:
        """The number of records for this source shard.

        This should be equal with sum of all pandas DataFrame rows.
        """
        raise NotImplementedError

    def set_epoch(self, epoch):
        self.epoch = epoch

    def prefix(self) -> str:
        raise NotImplementedError

    def max_parallel(self) -> int:
        return self._max_parallel

    def resources(self):
        return self._resources

    def read(self) -> Iterable[pd.DataFrame]:
        """Read the source data in serial mode"""
        raise NotImplementedError

    def read_parallel(self) -> Iterable[Callable]:
        """Read the source data in parallel mode."""
        raise NotImplementedError

    def __iter__(self) -> Union[Iterable[pd.DataFrame], Iterable[Callable]]:
        if self.max_parallel() > 1:
            return self.read_parallel()
        else:
            return self.read()

    def __str__(self):
        return repr(self)

    def __repr__(self):
        if self.max_parallel() > 1:
            suffix = f"_parallel[{self._max_parallel}]"
        else:
            suffix = ""
        return f"{self.prefix()}SourceShard[{self.shard_id}]" + suffix


class Reader:
    """A interface for source data reading.

    This is a common interface for data reading of MLDataset. See the
    following: ParallelIteratorReader as a example.
    """
    def batch_size(self) -> int:
        """This means the each record(pandas.DataFrame) will have batch size
        of rows. 0 means known."""
        return 0

    def num_shards(self) -> int:
        """Return the number of shards"""
        raise NotImplementedError

    def repartition(self, num_partitions: int) -> "Reader":
        """Repartition the shards"""
        raise NotImplementedError

    def get_shard(self, shard_id) -> "SourceReader":
        raise NotImplementedError

    def repeated(self) -> bool:
        return False

    def max_parallel(self) -> int:
        """The maximum parallelism

        It means whether we support read the source data parallel. The
        SourceReader must support read_parallel if this is larger than one.
        """
        return 1

    def resources(self) -> Dict:
        """The resources required for the parallel reading task"""
        return None


ReaderVar = TypeVar("ReaderVar", bound=Reader)
SourceReaderVar = TypeVar("SourceReaderVar", bound=SourceReader)


class ParallelIteratorSourceReader(SourceReader):

    def __init__(self,
                 shard_id,
                 it: ParallelIterator):
        super(ParallelIteratorSourceReader, self).__init__(shard_id, 1, None)
        self._it = it
        self._num_records = None

    @property
    def num_records(self) -> int:
        if not self._num_records:
            self._num_records = sum(self._it.get_shard(self.shard_id))
        return self._num_records

    def prefix(self) -> str:
        return "ParallelIterator"

    def read(self) -> Iterable[pd.DataFrame]:
        return self._it.get_shard(self.shard_id)

    def read_parallel(self) -> Iterable[Callable]:
        raise Exception("Not supported operation")


class ParallelIteratorReader(Reader):
    def __init__(self, it: ParallelIterator, batch_size):
        self._it = it
        self._batch_size = batch_size

    def batch_size(self) -> int:
        return self._batch_size

    def num_shards(self) -> int:
        return self._it.num_shards()

    def repartition(self, num_partitions: int) -> "Reader":
        return ParallelIteratorReader(self._it.repartition(num_partitions))

    def get_shard(self, shard_id) -> "SourceReader":
        return ParallelIteratorSourceReader(shard_id, self._it)
