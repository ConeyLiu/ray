from typing import Dict, List, Iterable, Iterator

from ray.util.data import MLDataset
from ray.util.placement_group import PlacementGroup
from collections import namedtuple
import pandas as pd
from enum import Enum
from ray.util.iter import LocalIterator, T, _NextValueNotReady
import random
import logging

logger = logging.getLogger(__name__)

WorkerMetaData = namedtuple("WorkerMetaData",
                            ["placementGroup", "placement_group_bundle_index"])


class ShuffleMode(Enum):
    NO_SHUFFLE = 1
    RECORD_SHUFFLE = 1
    LOCAL_BATCH_SHUFFLE = 2


class SGDMLDataset:
    def __init__(self,
                 ds: MLDataset,
                 num_workers: int,
                 actor_aggregated: bool,
                 metadata_mapping: Dict[int, WorkerMetaData],
                 shuffle_mode: ShuffleMode,
                 shuffle_buffer_size: int = 1,
                 seed: int = None):
        if ds.num_shards() % num_workers != 0:
            logger.warning(f"MLDataset number of shards({ds.num_shards()}) is "
                           f"not multiple of num_workers({num_workers})")
            num_shards = (ds.num_shards() // num_workers) + 1
            ds = ds.repartition(num_shards)

        self._ds = ds
        self._num_workers = num_workers
        self._actor_aggregated = actor_aggregated
        self._metadata_mapping = metadata_mapping
        self._shuffle_mode = shuffle_mode
        self._shuffle_buffer_size = shuffle_buffer_size
        self._seed = seed

    def get_repeatable_shard(self,
                             indexes: List[int],
                             batch_ms: int = 0,
                             num_async: int = 1,
                             shuffle_mode: ShuffleMode = ShuffleMode.NO_SHUFFLE,
                             shuffle_buffer_size: int = 1,
                             seed: int = None) -> Iterator:
        """Get the given shard of the current dataset.

        The return is a iterator. Each call iter on the returned iterator will
        get the shard data from beginning. And it support shuffle the return
        iterator when each call iter on the return.
        Args:
            indexes (List[int]): the shard index ids, None means collect all
                data.
            batch_ms (int): Batches items for batch_ms milliseconds
                before retrieving it. Increasing batch_ms increases latency
                but improves throughput. If this value is 0, then items are
                returned immediately.
            num_async (int): The max number of requests in flight. Increasing
                this improves the amount of pipeline parallelism in the
                iterator.
            shuffle_mode (ShuffleMode): the shuffle mode
            shuffle_buffer_size (int): same as ParallelIterator.local_shuffle
            seed (int): the random seed
        Returns:
            The given shard iterator. If the shuffle is True, each call iter
            will return a different ordered iterator.
        """
        return _RepeatableIterator(self._ds, indexes, batch_ms, num_async,
                                   shuffle_mode, shuffle_buffer_size, seed)

    def _get_shard_ids(self, worker_id):
        shard_ids = []
        i = worker_id
        step = self._ds.num_shards() // self._num_workers
        while i <= self._ds.num_shards():
            shard_ids.append(i)
            i += step
        return shard_ids

    def _read_data(self,
                   worker_id: int,
                   batch_ms: int = 0,
                   num_async: int = 1) -> Iterable[pd.DataFrame]:
        shard_ids = self._get_shard_ids(worker_id)
        return self.get_repeatable_shard(
            shard_ids, batch_ms, num_async, self._shuffle_mode,
            self._shuffle_buffer_size, self._seed)

    def _read_data_in_actor(self, work_id) -> Iterable[pd.DataFrame]:
        pass

    def get_data(self, worker_id):
        return self._read_data(worker_id)


class _RepeatableIterator(Iterator[T]):
    """A repeatable iterator for the given shard index data.

    Each call iter(_RepeatableIterator instance) will fetch the data from
    beginning and will return a different order or data if set shuffle
    Args:
        ds (MLDataset): a MLDataset
        shard_indexes (List[int]): the shard index ids. -1 means collect all data.
        batch_ms (int): Batches items for batch_ms milliseconds
            before retrieving it. Increasing batch_ms increases latency
            but improves throughput. If this value is 0, then items are
            returned immediately.
        num_async (int): The max number of requests in flight. Increasing this
            improves the amount of pipeline parallelism in the iterator.
        shuffle_mode (ShuffleMode): the shuffle mode
        shuffle_buffer_size (int): same as ParallelIterator.local_shuffle
        seed (int): the random seed
    """

    def __init__(self,
                 ds: MLDataset,
                 shard_indexes: List[int],
                 batch_ms: int = 0,
                 num_async: int = 1,
                 shuffle_mode: ShuffleMode = ShuffleMode.NO_SHUFFLE,
                 shuffle_buffer_size: int = 1,
                 seed: int = None):
        super(_RepeatableIterator, self).__init__()
        self._ds = ds
        self._shard_indexes = shard_indexes
        self._batch_ms = batch_ms
        self._num_async = num_async
        self._shuffle_mode = shuffle_mode
        self._shuffle_buffer_size = shuffle_buffer_size
        self._seed = seed
        self._local_it: LocalIterator[T] = None

        self._i = 0

    def __next__(self) -> T:
        assert self._local_it is not None
        return next(self._local_it)

    def __iter__(self) -> Iterator[T]:
        if self._shard_indexes:
            shards = self._ds.select_shards(self._shard_indexes)
            it = shards.gather_async(self._batch_ms, self._num_async)
        else:
            if self._num_async > 0:
                it = self._ds.gather_async(
                    batch_ms=self._batch_ms, num_async=self._num_async)
            else:
                it = self._ds.gather_sync()
        if self._shuffle_mode:
            it = self.shuffle(it)

        self._local_it = it
        return self

    def shuffle(self,
                shuffle_mode: ShuffleMode,
                local_it: LocalIterator[T]) -> LocalIterator[pd.DataFrame]:
        if shuffle_mode == ShuffleMode.RECORD_SHUFFLE:
            def apply_record_shuffle(it):
                for item in it:
                    if isinstance(item, _NextValueNotReady):
                        yield item
                    else:
                        yield item.sample(frac=1, random_state=self._seed)

            return LocalIterator(
                local_it.base_iterator,
                local_it.shared_metrics,
                local_it.local_transforms + [apply_record_shuffle],
                name=local_it.name +
                     ".shuffle(shuffle_mode={}, seed={})".format(
                         shuffle_mode,
                         str(self._seed) if self._seed is not None else "None"))

        else:
            shuffle_random = random.Random(self._seed)

            def apply_shuffle(it):
                buffer = []
                for item in it:
                    if isinstance(item, _NextValueNotReady):
                        yield item
                    else:
                        buffer.append(item)
                        if len(buffer) >= self._shuffle_buffer_size:
                            item = buffer.pop(
                                shuffle_random.randint(0,
                                                       len(buffer) - 1))
                            item = item.sample(frac=1, random_state=self._seed)
                            yield item
                while len(buffer) > 0:
                    item = buffer.pop(shuffle_random.randint(0, len(buffer) - 1))
                    item = item.sample(frac=1, random_state=self._seed)
                    yield item

            return LocalIterator(
                local_it.base_iterator,
                local_it.shared_metrics,
                local_it.local_transforms + [apply_shuffle],
                name=local_it.name +
                     ".shuffle(shuffle_mode={}, shuffle_buffer_size={}, seed={})".format(
                         shuffle_mode,
                         self._shuffle_buffer_size,
                         str(self._seed) if self._seed is not None else "None"))
