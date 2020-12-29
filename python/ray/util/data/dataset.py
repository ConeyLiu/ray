import random
from typing import Any, Callable, Dict, List, Iterable, Iterator, Optional, Union

import pandas as pd

from ray.util.iter import (from_items, _NextValueNotReady, LocalIterator, ParallelIterator,
                           T, U)
from .reader import ReaderVar, ParallelIteratorReader
from .task import SerialTask, ParallelTask, TaskQueue, UnresolvedTask

from enum import Enum


class ShuffleMode(Enum):
    record_level = 1
    local_batch_level = 2


class MLDataset:
    """A distributed ML dataset implemented based on ParallelIterator

    Args:
        task_queues (List[TaskQueue]): which hold the tasks to execute.
        batch_size (int): The batch size of the current dataset. It should be
            larger than zero, and 0 means unknown.
    """

    def __init__(self,
                 name: str,
                 task_queues: List[TaskQueue],
                 batch_size: int):
        self._name = name
        self._task_queues = task_queues
        self._batch_size = batch_size

    @classmethod
    def from_parallel_it(cls, it: ParallelIterator, batch_size: int = 0) -> "MLDataset":
        """Create MLDataset from a existed ParallelIterator

        Args:
            it (ParallelIterator): the existed ParallelIterator, the item
                should be pandas.DataFrame
            batch_size (int): the batch size
        """
        reader = ParallelIteratorReader(it)
        task_queue = TaskQueue(reader, [])
        return MLDataset(it.name, [task_queue], batch_size)

    @classmethod
    def from_reader(cls, name: str, reader: ReaderVar, batch_size: int = 0) -> "MLDataset":
        """Create MLDataset from a Reader"""
        task_queue = TaskQueue(reader, [])
        return MLDataset(name, [task_queue], batch_size)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def num_shards(self) -> int:
        return sum(q.num_shards() for q in self._task_queues)

    def __iter__(self):
        raise TypeError("Unsupported operation")

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"MLDataset[{self._name}]"

    def _execute(self, action_fn: Optional[Callable[[ParallelIterator], Any]] = None) -> "ParallelIterator":
        tasks = []
        for q in self._task_queues:
            tasks += q.create_execution_task().values()
        tasks = [lambda:t.execute(s) for s, t in tasks]
        it = from_items(tasks, self.num_shards, repeat=False)
        if action_fn is not None:
            it = action_fn(it)
        return it

    def _with_transform(self,
                        fn,
                        local_it_fn,
                        fn_name, max_parallel: Union[int, str] = 1,
                        resources: Dict = None) -> "MLDataset":
        """Helper function to create new MLDataset

        Args:
            fn (Callable[[pd.DataFrame], pd.DataFrame]): the transform function
                which will apply to each record. This function will be used for
                create a parallel task.
            local_it_fn (Callable[[Iterable[pd.DataFrame]],
                                  Iterable[pd.DataFrame]]) the transform
                function accept a iterable of pd.DataFrame as input and
                output a iterable of pd.DataFrame. This function will be used
                for create a serial task.
            fn_name (str), the name of the transform function
            max_parallel (int): the maximum parallelism of this transform
                function to execute
            resources (Dict): remote function resources, this is only needes
                when max_parallel is larger than 1
        """
        if max_parallel == "auto":
            task = UnresolvedTask(fn, local_it_fn)
        elif max_parallel > 1:
            task = ParallelTask(fn, max_parallel, resources)
        else:
            task = SerialTask(local_it_fn)
        task_queues = [q.with_task(task) for q in self._task_queues]
        return MLDataset(self._name + fn_name, task_queues, self._batch_size)

    def map(self,
            map_fn: Callable[[pd.DataFrame], pd.DataFrame],
            max_parallel: Union[int, str] = "auto",
            resources: Dict = None) -> "MLDataset":
        return self._with_transform(
            map_fn, lambda it: it.for_each(map_fn), ".map()", max_parallel,
            resources)

    def filter(self, filter_fn: Callable[[pd.DataFrame], bool],
               max_parallel: Union[int, str] = "auto",
               resources: Dict = None) -> "MLDataset":
        def filter_fn_parallel(df: pd.DataFrame) -> pd.DataFrame:
            if filter_fn(df):
                return df
            else:
                return _NextValueNotReady()
        return self._with_transform(
            filter_fn_parallel, lambda it: it.filter(filter_fn), ".filter()",
            max_parallel, resources)

    def transform(
            self,
            fn: Callable[[Iterable[pd.DataFrame]], Iterable[pd.DataFrame]]
    ) -> "MLDataset":
        """Apply the fn function to the MLDataset

        Args:
            fn (Callable[[Iterable[DataFrame]], Iterable[DataFrame]]):
                The function to applied. The input is a iterator of
                pandas.DataFrame, and the output should also be a iterator of
                pandas.DataFrame.
        Returns:
            A new MLDataset
        """
        return self._with_transform(
            None, lambda local_it: local_it.transform(fn), ".transform()",
            max_parallel=1, resources=None)

    def batch(self, batch_size: int) -> "MLDataset":
        """Rebatch the number of rows for each pandas.DataFrame record

        Unlike the ParallelIterator.batch. This method rebatch the underlying
        the pandas DataFrame, and each pandas DataFrame will have batch_size
        rows.
        """
        if batch_size == self._batch_size:
            return self

        def batch_fn(it: Iterable[pd.DataFrame]) -> Iterable[pd.DataFrame]:
            it = iter(it)
            return_df = None
            while True:
                try:
                    cur_df = next(it)
                    cur_index = 0
                    cur_size = cur_df.shape[0]
                    while cur_df is not None or (
                            cur_index + batch_size) < cur_size:
                        if cur_df is None or cur_index == cur_size:
                            cur_df = next(it)
                            cur_index = 0
                            cur_size = cur_df.shape[0]
                        if return_df is not None:
                            ri = cur_index + batch_size - return_df.shape[0]
                            ri = min(ri, cur_size)
                            tmp = cur_df.iloc[cur_index:ri]
                            return_df = pd.concat([return_df, tmp])
                            cur_index = ri
                        else:
                            ri = cur_index + batch_size
                            ri = min(ri, cur_size)
                            return_df = cur_df.iloc[cur_index:ri]
                            cur_index = ri
                        if return_df.shape[0] == batch_size:
                            return_df.index = range(return_df.shape[0])
                            yield return_df
                            return_df = None
                except StopIteration:
                    break

            if return_df is not None:
                return_df.index = range(return_df.shape[0])
                yield return_df

        new_ds = self._with_transform(
            None, lambda local_it: local_it.transform(batch_fn),
            f".batch({batch_size})")
        new_ds._batch_size = batch_size
        return new_ds

    def local_shuffle(self,
                      shuffle_mode: ShuffleMode = ShuffleMode.record_level,
                      max_parallel: Union[int, str] = "auto",
                      resources: Dict = None,
                      shuffle_buffer_size: int = 1,
                      seed: int = None) -> "MLDataset":
        """Applying local shuffle

        Unlike the ParallelIterator.local_shuffle. This shuffle will first
        apply the local_shuffle for each shards and then shuffle the each
        pandas DataFrame.
        """
        shuffle_random = random.Random(seed)

        def apply_shuffle(it):
            buffer = []
            for item in it:
                if isinstance(item, _NextValueNotReady):
                    yield item
                else:
                    buffer.append(item)
                    if len(buffer) >= shuffle_buffer_size:

                        df = buffer.pop(
                            shuffle_random.randint(0,
                                                   len(buffer) - 1))
                        df = df.sample(frac=1, random_state=seed)
                        yield df
            while len(buffer) > 0:
                yield buffer.pop(shuffle_random.randint(0, len(buffer) - 1))

        if shuffle_mode == ShuffleMode.record_level:
            def parallel_shuffle(df):
                return df.sample(frac=1, random_state=seed)
            name = ".local_shuffle(record_level)"
            return self._with_transform(
                parallel_shuffle,
                lambda local_it: local_it.transform(apply_shuffle),
                name, max_parallel, resources)
        else:
            name = ".local_shuffle(local_batch_level)"
            return self._with_transform(
                None, lambda local_it: local_it.transform(apply_shuffle), name,
                max_parallel=1, resources=None)

    def repartition(self, num_partitions: int) -> "MLDataset":
        """see ParallelIterator.repartition"""
        if num_partitions == self.num_shards:
            return self

        if len(self._task_queues) == 1:
            reader = self._task_queues[0].reader
            reader = reader.repartition(num_partitions)
            task_queue = self._task_queues[0].with_reader(reader)
            return MLDataset(self._name + f".repartition({num_partitions})", [task_queue], self._batch_size)
        else:
            it = self._execute()
            it = it.repartition(num_partitions)
            return MLDataset.from_parallel_it(it, self._batch_size)

    def union(self, other: "MLDataset") -> "MLDataset":
        """Return an iterator that is the union of this and the other."""
        if not isinstance(other, MLDataset):
            raise TypeError(
                f"other must be of type MLDataset, got {type(other)}")

        batch_size = 0
        if self._batch_size == other._batch_size:
            batch_size = self._batch_size

        return MLDataset(
            f"ParallelUnion[{self}, {other}]",
            self._task_queues + other._task_queues,
            batch_size)

    def to_parallel_it(self) -> "ParallelIterator":
        return self._execute()

    def get_shard(self,
                  shard_index: int,
                  batch_ms: int = 0,
                  num_async: int = 1) -> "LocalIterator[T]":
        return self.to_parallel_it().get_shard(shard_index, batch_ms, num_async)

    def gather_sync(self) -> "LocalIterator[T]":
        return self.to_parallel_it().gather_sync()

    def gather_async(self, batch_ms=0, num_async=1) -> "LocalIterator[T]":
        return self.to_parallel_it().gather_async(batch_ms, num_async)

    def to_torch(self,
                 feature_columns=None,
                 feature_shapes=None,
                 feature_types=None,
                 label_column=None,
                 label_shape=None,
                 label_type=None,
                 batch_size: int = 128,
                 shuffle=False,
                 shuffle_buffer_size=None,
                 seed=None):
        """Create a TorchMLDataset from the current MLDataset.

        Args:
            feature_columns (List[Any]): the column indexes name.
            feature_shapes (Optional[List[Any]]): the feature shapes should
               match the feature columns if provided.
            feature_types (Optional[List["torch.dtype"]]): the feature types
               should match the feature columns if provided. All feature will
               be cast into torch.float by default. Otherwise, cast into the
               provided type.
            label_column (Any): the label name.
            label_shape (Optional[int]): the label shape.
            label_type (Optional["torch.dtype"]): the label type, this will be
               cast into torch.float by default
            batch_size (int): the expected batch size
            shuffle (bool): whether shuffle the data
            shuffle_buffer_size (int): The algorithm fills a buffer with
                shuffle_buffer_size elements and randomly samples elements from
                this buffer, replacing the selected elements with new elements.
                For perfect shuffling, this argument should be greater than or
                equal to the largest iterator size.
            seed (int): Seed to use for
                randomness. Default value is None.

        Returns:
            A TorchMLDataset
        """
        if batch_size != self.batch_size:
            ds = self.batch(batch_size)
        else:
            ds = self
        from ray.util.sgd.torch.torch_dataset import TorchMLDataset
        return TorchMLDataset(ds, feature_columns, feature_shapes,
                              feature_types, label_column, label_shape,
                              label_type, shuffle, shuffle_buffer_size, seed)

    def to_tf(self,
              feature_columns=None,
              feature_shapes=None,
              feature_types=None,
              label_column=None,
              label_shape=None,
              label_type=None,
              batch_size: int = 128,
              shuffle=False,
              buffer_size: int = 128,
              seed=None):
        """Create a TFMLDataset from the current MLDataset.

        Args:
            feature_columns (List[Any]): the column names.
            feature_shapes (Optional[List[tf.TensorShape]]): the feature shapes
                should match the feature columns if provided.
            feature_types (Optional[List["tf.DType"]]): the feature types
               should match the feature columns if provided. All feature will
               be cast into tf.float by default. Otherwise, cast into the
               provided type.
            label_column (Any): the label name.
            label_shape (Optional[tf.TensorShape]): the label shape.
            label_type (Optional["tf.DType"]): the label type, this will be
               cast into tf.float by default
            batch_size (int): the expected batch size
            shuffle (bool): whether shuffle the data
            buffer_size (int): representing the number of elements from this
               dataset from which the new dataset will sample.
            seed (int): Seed to use for
                randomness. Default value is None.
        Returns:
            A TFMLDataset
        """
        from ray.util.sgd.tf.tf_dataset import TFMLDataset
        return TFMLDataset(self, feature_columns, feature_shapes,
                           feature_types, label_column, label_shape,
                           label_type, batch_size, shuffle, buffer_size, seed)


class _RepeatableIterator(Iterator[T]):
    """A repeatable iterator for the given shard index data.

    Each call iter(_RepeatableIterator instance) will fetch the data from
    beginning and will return a different order or data if set shuffle
    Args:
        it (ParallelIterator): a ParallelIterator
        shard_index (int): the shard index id. -1 means collect all data.
        num_async (int): The max number of requests in flight. Increasing this
            improves the amount of pipeline parallelism in the iterator.
        shuffle (bool): whether shuffle the given shard data
        shuffle_buffer_size (int): same as ParallelIterator.local_shuffle
        seed (int): the random seed
    """

    def __init__(self,
                 it: ParallelIterator,
                 shard_index: int,
                 batch_ms: int = 0,
                 num_async: int = 1,
                 shuffle: bool = False,
                 shuffle_buffer_size: int = 1,
                 item_shuffle_fn: Callable = None,
                 seed: int = None):
        super(_RepeatableIterator, self).__init__()
        self._it = it
        self._shard_index = shard_index
        self._num_async = num_async
        self._batch_ms = batch_ms
        self._shuffle = shuffle
        self._shuffle_buffer_size = shuffle_buffer_size
        self._item_shuffle_fn = item_shuffle_fn or (lambda x: x)
        self._seed = seed
        self._local_it: LocalIterator[T] = None

        self._i = 0

    def __next__(self) -> T:
        assert self._local_it is not None
        return next(self._local_it)

    def __iter__(self) -> Iterator[T]:
        if self._shard_index >= 0:
            it = self._it.get_shard(self._shard_index, self._batch_ms, self._num_async)
        else:
            if self._num_async > 0:
                it = self._it.gather_async(self._batch_ms, num_async=self._num_async)
            else:
                it = self._it.gather_sync()
        if self._shuffle:
            it = self.shuffle(it)

        self._local_it = it
        return self

    def shuffle(self,
                local_it: LocalIterator[T]) -> LocalIterator[pd.DataFrame]:
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
                        item = self._item_shuffle_fn(item)
                        yield item
            while len(buffer) > 0:
                item = buffer.pop(shuffle_random.randint(0, len(buffer) - 1))
                item = self._item_shuffle_fn(item)
                yield item

        return LocalIterator(
            local_it.base_iterator,
            local_it.shared_metrics,
            local_it.local_transforms + [apply_shuffle],
            name=local_it.name +
            ".shuffle(shuffle_buffer_size={}, seed={})".format(
                self._shuffle_buffer_size,
                str(self._seed) if self._seed is not None else "None"))
