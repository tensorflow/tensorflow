# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Grouping dataset transformations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import sparse
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops


def group_by_window(key_func,
                    reduce_func,
                    window_size=None,
                    window_size_func=None):
  """A transformation that groups windows of elements by key and reduces them.

  This transformation maps each consecutive element in a dataset to a key
  using `key_func` and groups the elements by key. It then applies
  `reduce_func` to at most `window_size_func(key)` elements matching the same
  key. All execpt the final window for each key will contain
  `window_size_func(key)` elements; the final window may be smaller.

  You may provide either a constant `window_size` or a window size determined by
  the key through `window_size_func`.

  Args:
    key_func: A function mapping a nested structure of tensors
      (having shapes and types defined by `self.output_shapes` and
      `self.output_types`) to a scalar `tf.int64` tensor.
    reduce_func: A function mapping a key and a dataset of up to `window_size`
      consecutive elements matching that key to another dataset.
    window_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
      consecutive elements matching the same key to combine in a single
      batch, which will be passed to `reduce_func`. Mutually exclusive with
      `window_size_func`.
    window_size_func: A function mapping a key to a `tf.int64` scalar
      `tf.Tensor`, representing the number of consecutive elements matching
      the same key to combine in a single batch, which will be passed to
      `reduce_func`. Mutually exclusive with `window_size`.

  Returns:
    A `Dataset` transformation function, which can be passed to
    @{tf.data.Dataset.apply}.

  Raises:
    ValueError: if neither or both of {`window_size`, `window_size_func`} are
      passed.
  """
  if (window_size is not None and window_size_func or
      not (window_size is not None or window_size_func)):
    raise ValueError("Must pass either window_size or window_size_func.")

  if window_size is not None:

    def constant_window_func(unused_key):
      return ops.convert_to_tensor(window_size, dtype=dtypes.int64)

    window_size_func = constant_window_func

  assert window_size_func is not None

  def _apply_fn(dataset):
    """Function from `Dataset` to `Dataset` that applies the transformation."""
    return GroupByWindowDataset(dataset, key_func, reduce_func,
                                window_size_func)

  return _apply_fn


class _VariantDataset(dataset_ops.Dataset):
  """A Dataset wrapper for a tf.variant-typed function argument."""

  def __init__(self, dataset_variant, output_types, output_shapes,
               output_classes):
    super(_VariantDataset, self).__init__()
    self._dataset_variant = dataset_variant
    self._output_types = output_types
    self._output_shapes = output_shapes
    self._output_classes = output_classes

  def _as_variant_tensor(self):
    return self._dataset_variant

  @property
  def output_classes(self):
    return self._output_classes

  @property
  def output_shapes(self):
    return self._output_shapes

  @property
  def output_types(self):
    return self._output_types


class GroupByWindowDataset(dataset_ops.Dataset):
  """A `Dataset` that groups its input and performs a windowed reduction."""

  def __init__(self, input_dataset, key_func, reduce_func, window_size_func):
    """See `group_by_window()` for details."""
    super(GroupByWindowDataset, self).__init__()

    self._input_dataset = input_dataset

    self._make_key_func(key_func, input_dataset)
    self._make_reduce_func(reduce_func, input_dataset)
    self._make_window_size_func(window_size_func)

  def _make_window_size_func(self, window_size_func):
    """Make wrapping Defun for window_size_func."""

    @function.Defun(dtypes.int64)
    def tf_window_size_func(key):
      key.set_shape([])
      window_size = ops.convert_to_tensor(
          window_size_func(key), dtype=dtypes.int64)
      if window_size.dtype != dtypes.int64:
        raise ValueError(
            "`window_size_func` must return a single tf.int64 tensor.")
      return window_size

    self._window_size_func = tf_window_size_func
    self._window_size_func.add_to_graph(ops.get_default_graph())

  def _make_key_func(self, key_func, input_dataset):
    """Make wrapping Defun for key_func."""

    @function.Defun(*nest.flatten(
        sparse.as_dense_types(input_dataset.output_types,
                              input_dataset.output_classes)))
    def tf_key_func(*args):
      """A wrapper for Defun that facilitates shape inference."""
      # Pass in shape information from the input_dataset.
      dense_shapes = sparse.as_dense_shapes(input_dataset.output_shapes,
                                            input_dataset.output_classes)
      for arg, shape in zip(args, nest.flatten(dense_shapes)):
        arg.set_shape(shape)

      nested_args = nest.pack_sequence_as(input_dataset.output_types, args)
      nested_args = sparse.deserialize_sparse_tensors(
          nested_args, input_dataset.output_types, input_dataset.output_shapes,
          input_dataset.output_classes)
      # pylint: disable=protected-access
      if dataset_ops._should_unpack_args(nested_args):
        ret = key_func(*nested_args)
      # pylint: enable=protected-access
      else:
        ret = key_func(nested_args)
      ret = ops.convert_to_tensor(ret, dtype=dtypes.int64)
      if ret.dtype != dtypes.int64:
        raise ValueError("`key_func` must return a single tf.int64 tensor.")
      return ret

    self._key_func = tf_key_func
    self._key_func.add_to_graph(ops.get_default_graph())

  def _make_reduce_func(self, reduce_func, input_dataset):
    """Make wrapping Defun for reduce_func."""

    @function.Defun(dtypes.int64, dtypes.variant)
    def tf_reduce_func(key, window_dataset_variant):
      """A wrapper for Defun that facilitates shape inference."""
      key.set_shape([])
      window_dataset = _VariantDataset(
          window_dataset_variant, input_dataset.output_types,
          input_dataset.output_shapes, input_dataset.output_classes)
      if not isinstance(window_dataset, dataset_ops.Dataset):
        raise TypeError("`window_dataset` must return a `Dataset` object.")
      output_dataset = reduce_func(key, window_dataset)
      if not isinstance(output_dataset, dataset_ops.Dataset):
        raise TypeError("`reduce_func` must return a `Dataset` object.")
      self._output_classes = output_dataset.output_classes
      self._output_types = output_dataset.output_types
      self._output_shapes = output_dataset.output_shapes
      return output_dataset._as_variant_tensor()  # pylint: disable=protected-access

    self._reduce_func = tf_reduce_func
    self._reduce_func.add_to_graph(ops.get_default_graph())

  @property
  def output_classes(self):
    return self._output_classes

  @property
  def output_shapes(self):
    return self._output_shapes

  @property
  def output_types(self):
    return self._output_types

  def _as_variant_tensor(self):
    return gen_dataset_ops.group_by_window_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
        self._key_func.captured_inputs,
        self._reduce_func.captured_inputs,
        self._window_size_func.captured_inputs,
        key_func=self._key_func,
        reduce_func=self._reduce_func,
        window_size_func=self._window_size_func,
        output_types=nest.flatten(
            sparse.as_dense_types(self.output_types, self.output_classes)),
        output_shapes=nest.flatten(
            sparse.as_dense_shapes(self.output_shapes, self.output_classes)))
