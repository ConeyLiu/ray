import random
from typing import Callable, List, Iterable, Iterator

import pandas as pd

from ray.util.iter import (_NextValueNotReady, LocalIterator, ParallelIterator,
                           T, U)


class MLDataset(ParallelIterator[pd.DataFrame]):
    """A distributed ML dataset implemented based on ParallelIterator

    All item should be a list like object or dataclass instance.

    Args:
        batch_size (int): The batch size of the current dataset. It should be
            larger than zero, and 0 means unknown.
    """

    def __init__(self, actor_sets: List["_ActorSet"], name: str,
                 parent_iterators: List[ParallelIterator[pd.DataFrame]],
                 batch_size: int, repeated: bool):
        super(MLDataset, self).__init__(actor_sets, name, parent_iterators)
        self._batch_size = batch_size
        self._repeated = repeated

    @staticmethod
    def from_parallel_it(para_it: ParallelIterator[pd.DataFrame],
                         batch_size: int,
                         repeated: bool = False) -> "MLDataset":
        """Create a MLDataset from an parallel iterator

        The record of ParallelIterator should be pandas.DataFrame.

        Args:
            para_it (ParallelIterator[T]): An existing parallel iterator,
                and each should be a list like object or dataclass instance
            batch_size (int): The batch size of the current dataset. It
                should be larger than zero, and 0 means unknown.
            repeated (bool): whether the para_it is repeated.
        Returns:
            A MLDataset
        """
        return MLDataset(para_it.actor_sets, para_it.name,
                         para_it.parent_iterators, batch_size, repeated)

    def __iter__(self):
        raise TypeError(
            "You must use it.gather_sync() or it.gather_async() to "
            "iterate over the results of a MLDataset.")

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"MLDataset[{self.name}]"

    def _with_transform(self, local_it_fn, name) -> "MLDataset":
        """Helper function to create new MLDataset"""
        para_it = super()._with_transform(local_it_fn, name)
        return MLDataset.from_parallel_it(para_it, self._batch_size,
                                          self._repeated)

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
        return self._with_transform(lambda local_it: local_it.transform(fn),
                                    ".transform()")

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

        self._batch_size = batch_size
        return self._with_transform(
            lambda local_it: local_it.transform(batch_fn),
            f".batch({batch_size})")

    def flatten(self) -> "MLDataset":
        raise Exception("Unsupported operation")

    def combine(self, fn: Callable[[T], List[U]]) -> "MLDataset":
        raise Exception("Unsupported operation")

    @property
    def repeated(self) -> bool:
        return self._repeated

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def local_shuffle(self, shuffle_buffer_size: int,
                      seed: int = None) -> "MLDataset":
        """Applying local shuffle

        Unlike the ParallelIterator.local_shuffle. This shuffle will first
        apply the local_shuffle for each shards and then shuffle the each
        pandas DataFrame.
        """
        ds = super().local_shuffle(shuffle_buffer_size, seed)

        def shuffle_fn(it: Iterable[pd.DataFrame]) -> Iterable[pd.DataFrame]:
            for df in it:
                df = df.sample(frac=1, random_state=seed)
                yield df

        ds = ds._with_transform(
            lambda local_it: local_it.transform(shuffle_fn),
            ".inner_pandas_shuffle()")

        return ds

    def repartition(self, num_partitions: int,
                    batch_ms: int = 0) -> "MLDataset":
        """see ParallelIterator.repartition"""
        if num_partitions == self.num_shards():
            return self
        para_it = super().repartition(num_partitions, batch_ms)
        return MLDataset.from_parallel_it(para_it, self._batch_size)

    def union(self, other: "MLDataset") -> "MLDataset":
        """Return an iterator that is the union of this and the other."""
        if not isinstance(other, MLDataset):
            raise TypeError(
                f"other must be of type MLDataset, got {type(other)}")

        if self._repeated != other.repeated:
            raise TypeError(
                f"want to union two MLDataset which have different repeated "
                f"type, self repeated: {self._repeated}, other repeated: "
                f"{other.repeated}")

        batch_size = 0
        if self._batch_size == other._batch_size:
            batch_size = self._batch_size

        actor_sets = []
        actor_sets.extend(self.actor_sets)
        actor_sets.extend(other.actor_sets)
        # if one of these iterators is a result of a repartition, we need to
        # keep an explicit reference to its parent iterator
        return MLDataset(
            actor_sets,
            f"ParallelUnion[{self}, {other}]",
            parent_iterators=self.parent_iterators + other.parent_iterators,
            batch_size=batch_size,
            repeated=self._repeated)

    def select_shards(self, shards_to_keep: List[int]) -> "MLDataset":
        para_it = super().select_shards(shards_to_keep)
        return MLDataset.from_parallel_it(para_it, self._batch_size,
                                          self._repeated)

    def to_torch(self,
                 feature_columns=None,
                 feature_shapes=None,
                 feature_types=None,
                 label_column=None,
                 label_shape=None,
                 label_type=None):
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
        Returns:
            A TorchMLDataset
        """
        from ray.util.sgd.torch.torch_dataset import TorchMLDataset
        return TorchMLDataset(self, feature_columns, feature_shapes,
                              feature_types, label_column, label_shape,
                              label_type)

    def to_tf(self,
              feature_columns=None,
              feature_shapes=None,
              feature_types=None,
              label_column=None,
              label_shape=None,
              label_type=None):
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
        Returns:
            A TFMLDataset
        """
        from ray.util.sgd.tf.tf_dataset import TFMLDataset
        return TFMLDataset(self, feature_columns, feature_shapes,
                           feature_types, label_column, label_shape,
                           label_type)
