# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for dealing with tf.data.Dataset."""

import collections.abc
import functools
from typing import Any, Callable, Iterable, Iterator, Union

import pandas as pd
import numpy as np

import tensorflow.compat.v2 as tf
from tensorflow_datasets.core import tf_compat
from tensorflow_datasets.core import utils
from tensorflow_datasets.core.utils import type_utils

Tree = type_utils.Tree
Tensor = type_utils.Tensor

TensorflowElem = Union[Tensor, tf.data.Dataset]
NumpyValue = Union[tf.RaggedTensor, np.ndarray, np.generic, bytes]
NumpyElem = Union[NumpyValue, Iterable[NumpyValue]]


class _IterableDataset(collections.abc.Iterable):
  """Iterable which can be called multiple times."""

  def __init__(
      self,
      make_iterator_fn: Callable[..., Iterator[NumpyElem]],
      *args: Any,
      **kwargs: Any,
  ):
    self._make_iterator_fn = functools.partial(
        make_iterator_fn, *args, **kwargs
    )

  def __iter__(self) ->  Iterator[NumpyElem]:
    """Calling `iter(ds)` multiple times recreates a new iterator."""
    return self._make_iterator_fn()


def _eager_dataset_iterator(ds: tf.data.Dataset) -> Iterator[NumpyElem]:
  for elem in ds:
    yield tf.nest.map_structure(
        lambda t: t if isinstance(t, tf.RaggedTensor) else t.numpy(),
        elem
    )


def _graph_dataset_iterator(ds_iter, graph: tf.Graph) -> Iterator[NumpyElem]:
  """Constructs a Python generator from a tf.data.Iterator."""
  with graph.as_default():
    init = ds_iter.initializer
    ds_item = ds_iter.get_next()
    with utils.nogpu_session() as sess:
      sess.run(init)
      while True:
        try:
          yield sess.run(ds_item)
        except tf.errors.OutOfRangeError:
          break


def _assert_ds_types(nested_ds: Tree[TensorflowElem]) -> None:
  """Assert all inputs are from valid types."""
  for el in tf.nest.flatten(nested_ds):
    if not (
        isinstance(el, (tf.Tensor, tf.RaggedTensor))
        or tf_compat.is_dataset(el)
    ):
      nested_types = tf.nest.map_structure(type, nested_ds)
      raise TypeError(
          'Arguments to as_numpy must be tf.Tensors or tf.data.Datasets. '
          f'Got: {nested_types}.'
      )


def _elem_to_numpy_eager(tf_el: TensorflowElem) -> NumpyElem:
  """Converts a single element from tf to numpy."""
  if isinstance(tf_el, tf.Tensor):
    return tf_el.numpy()
  elif isinstance(tf_el, tf.RaggedTensor):
    return tf_el
  elif tf_compat.is_dataset(tf_el):
    return _IterableDataset(_eager_dataset_iterator, tf_el)
  else:
    raise AssertionError(f'Unexpected element: {type(tf_el)}: {tf_el}')


def _nested_to_numpy_graph(ds_nested: Tree[TensorflowElem]) -> Tree[NumpyElem]:
  """Convert the nested structure of TF element to numpy."""
  all_ds = []
  all_arrays = []
  flat_ds = tf.nest.flatten(ds_nested)
  for elem in flat_ds:
    # Create an iterator for all datasets
    if tf_compat.is_dataset(elem):
      # Capture the current graph, so calling `iter(ds)` twice will reuse the
      # graph in which `as_numpy` was created.
      graph = tf.compat.v1.get_default_graph()
      ds_iter = tf.compat.v1.data.make_initializable_iterator(elem)
      all_ds.append(_IterableDataset(_graph_dataset_iterator, ds_iter, graph))
    else:
      all_arrays.append(elem)

  # Then create numpy arrays for all tensors
  if all_arrays:
    with utils.nogpu_session() as sess:  # Shared session for tf.Tensor
      all_arrays = sess.run(all_arrays)

  # Merge the dataset iterators and np arrays
  iter_ds = iter(all_ds)
  iter_array = iter(all_arrays)
  return tf.nest.pack_sequence_as(ds_nested, [
      next(iter_ds) if tf_compat.is_dataset(ds_el) else next(iter_array)
      for ds_el in flat_ds
  ])


def pd_as_numpy(dataset: Tree[TensorflowElem]) -> Tree[NumpyElem]:
  """Converts a `tf.data.Dataset` to an iterable of Pandas dataFrame.

  `pd_as_numpy` converts a possibly nested structure of `tf.data.Dataset`s
  and `tf.Tensor`s to pandas dataFrame.

  Example:

  ```
  ds = tfds.load(name="mnist", split="train")
  ds_numpy = tfds.as_numpy(ds)  # Convert `tf.data.Dataset` to Python generator
  for ex in ds_numpy:
    # `{'image': np.array(shape=(28, 28, 1)), 'labels': np.array(shape=())}`
    print(ex)
  ```

  Args:
    dataset: a possibly nested structure of `tf.data.Dataset`s and/or
      `tf.Tensor`s.

  Returns:
    pandas DataFrame
  """
  _assert_ds_types(dataset)
  if tf.executing_eagerly():
    local_np = tf.nest.map_structure(_elem_to_numpy_eager, dataset)
    local_dataframe = pd.DataFrame(local_np)
    return local_dataframe
  else:
    local_np = _nested_to_numpy_graph(dataset)
    local_dataframe = pd.DataFrame(local_np)
    return local_dataframe


def dataset_shape_is_fully_defined(ds):
  output_shapes = tf.compat.v1.data.get_output_shapes(ds)
  return all([ts.is_fully_defined() for ts in tf.nest.flatten(output_shapes)])


def features_shape_is_fully_defined(features):
  return all([tf.TensorShape(info.shape).is_fully_defined() for info in
              tf.nest.flatten(features.get_tensor_info())])
