import functools
import logging
from collections import Iterator
from collections.abc import Iterable
from typing import Any, Callable, List, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader

from ray.util.data import MLDataset
from ray.util.data.dataset import _RepeatableIterator
from ray.util.iter import ParallelIterator


def convert_to_tensor(df, feature_columns: List[Any],
                      feature_shapes: List[Any],
                      feature_types: List[torch.dtype], label_column: Any,
                      label_shape: Optional[int], label_type: torch.dtype):
    feature_tensor = []
    for col, shape, dtype in zip(feature_columns, feature_shapes,
                                 feature_types):
        column = df[col].values
        if column.dtype == np.object:
            if isinstance(column[0], np.ndarray):
                column = np.stack(column)
            elif isinstance(column[0], (list, tuple)):
                column = list(column)
            else:
                raise Exception(
                    f"Column {col}'s type: {type(column[0])} is not supported."
                    " It must be numpy built in type or numpy object of "
                    "(ndarray, list, tuple)")

        t = torch.as_tensor(column, dtype=dtype)
        if shape is not None:
            t = t.view(*(-1, *shape))
        else:
            t = t.view(-1, 1)
        feature_tensor.append(t)

    label_df = df[label_column].values
    label_tensor = torch.as_tensor(label_df, dtype=label_type)
    if label_shape:
        label_tensor = label_tensor.view(-1, label_shape)
    else:
        label_tensor = label_tensor.view(-1, 1)
    return feature_tensor, label_tensor


class TorchMLDataset:
    """A TorchMLDataset which converted from MLDataset

    .. code-block:: python

        ds = ml_dataset.to_torch(feature_columns=["x"], label_column="y")
        shard = ds.get_shard(0)
        data = DataLoader(shard, batch_size=32)
        batch_tensor_x, batch_tensor_y = next(iter(data))

        ds = ml_dataset.to_torch(feature_columns=["x", "y"], label_column="z")
        shard = ds.get_shard(0)
        data = DataLoader(shard, batch_size=32)
        batch_tensor_x, batch_tensor_y, batch_tensor_z = next(iter(data))


    Args:
        ds (MLDataset): a MLDataset
        feature_columns (List[Any]): the feature columns' name
        feature_shapes (Optional[List[Any]]): the shape for each
            feature. If provide, it should match the size of feature_columns.
        feature_types (Optional[List[torch.dtype]]): the data type for each
            feature. If provide, it should match the size of feature_columns
        label_column (Any): the label column name
        label_shape (Optional[int]): the shape for the label data
        label_type (Optional[torch.dtype]): the data type for the label data
    """

    def __init__(self,
                 ds: MLDataset = None,
                 feature_columns: List[Any] = None,
                 feature_shapes: Optional[List[Any]] = None,
                 feature_types: Optional[List[torch.dtype]] = None,
                 label_column: Any = None,
                 label_shape: Optional[int] = None,
                 label_type: Optional[torch.dtype] = None,
                 shuffle: bool = False,
                 shuffle_buffer_size: int = 1,
                 seed: int = 0):

        self._feature_columns = feature_columns
        self._feature_shapes = feature_shapes
        self._feature_types = feature_types
        self._label_column = label_column
        self._label_shape = label_shape
        self._label_type = label_type

        self._type_check_and_convert()

        self._ds = ds
        self._it = None

        self._shuffle = shuffle
        self._shuffle_buffer_size = shuffle_buffer_size
        self._seed = seed

    def _type_check_and_convert(self):
        # convert to list for convenience
        if not isinstance(self._feature_columns, list):
            self._feature_columns = [self._feature_columns]

        if self._feature_shapes:
            if not isinstance(self._feature_shapes, list):
                self._feature_shapes = [self._feature_shapes]

            assert len(self._feature_columns) == len(self._feature_shapes), \
                "The feature_shapes size must match the feature_columns"
            for i in range(len(self._feature_shapes)):
                if not isinstance(self._feature_shapes[i], Iterable):
                    self._feature_shapes[i] = [self._feature_shapes[i]]
        else:
            self._feature_shapes = [None] * len(self._feature_columns)

        if self._feature_types:
            if not isinstance(self._feature_types, list):
                self._feature_types = [self._feature_types]

            assert len(self._feature_columns) == len(self._feature_types), \
                "The feature_types size must match the feature_columns"
            for i in range(len(self._feature_types)):
                assert (all(isinstance(dtype, torch.dtype)
                            for dtype in self._feature_types)), \
                    "All value in feature_types should be torch.dtype instance"
        else:
            self._feature_types = [torch.float] * len(self._feature_columns)

        if not self._label_type:
            self._label_type = torch.float

    def _to_tensor(self, ds: MLDataset) -> ParallelIterator:
        convert_fn = functools.partial(
            convert_to_tensor,
            feature_columns=self._feature_columns,
            feature_shapes=self._feature_shapes,
            feature_types=self._feature_types,
            label_column=self._label_column,
            label_shape=self._label_shape,
            label_type=self._label_type)
        it = ds._with_transform(
            lambda it: it.for_each(convert_fn), ".to_torch_tensor()",
            "auto", None)._execute()
        return it

    def set_num_shards(self, num_shards):
        """Reshards the iterator if necessary"""
        if num_shards != self._ds.num_shards():
            logging.info("Setting num shards", num_shards)
            self._ds = self._ds.repartition(num_shards)
            if self._it is not None:
                self._it = self._to_tensor(self._ds)

    def get_shard(self,
                  shard_index: int,
                  num_async: int = 1,
                  batch_ms: int = 0,
                  epoch: int = 0,
                  data_loader_fn: Callable = None) -> DataLoader:
        if self._it is None:
            self._it = self._to_tensor(self._ds)

        seed = self._seed + epoch

        def shuffle_torch(tensors):
            feature_tensor, label_tensor = tensors
            feature_tensor = [t[torch.randperm(t.size()[0])] for t in feature_tensor]
            label_tensor = label_tensor[torch.randperm(label_tensor.size()[0])]
            return feature_tensor, label_tensor
        shard_it = _RepeatableIterator(self._it, shard_index, batch_ms, num_async,
                                       self._shuffle, self._shuffle_buffer_size,
                                       shuffle_torch, seed)
        dataset = TorchIterableDataset(shard_it)
        if data_loader_fn is None:
            return DefaultDataLoader(dataset, self._ds.batch_size)
        else:
            return data_loader_fn(dataset)


class TorchIterableDataset(IterableDataset):
    def __init__(self, it: Iterator, flatten: bool = False):
        super().__init__()
        self._it = it
        self._flatten = flatten

    def __iter__(self):
        if not self._flatten:
            for tensors in iter(self._it):
                feature_tensor, label_tensor = tensors
                yield (*feature_tensor, label_tensor)
        else:
            for tensors in iter(self._it):
                feature_tensor, label_tensor = tensors
                num_rows = label_tensor.size()[0]
                for i in range(num_rows):
                    features = [tensor[i] for tensor in feature_tensor]
                    label = label_tensor[i]
                    yield (*features, label)


class DefaultDataLoader(DataLoader):
    def __init__(self, dataset, batch_size):
        assert isinstance(dataset, TorchIterableDataset)
        super(DefaultDataLoader, self).__init__(dataset, batch_size=1, shuffle=False)
        self.batch_size = batch_size
        self._it = None

    def __iter__(self):
        self._it = iter(self.dataset)
        return self

    def __next__(self):
        data = next(self._it)
        return data
