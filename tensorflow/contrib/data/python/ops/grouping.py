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

import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import sparse
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import math_ops


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


def bucket_by_sequence_length(element_length_func,
                              bucket_boundaries,
                              bucket_batch_sizes,
                              padded_shapes=None,
                              padding_values=None,
                              pad_to_bucket_boundary=False):
  """A transformation that buckets elements in a `Dataset` by length.

  Elements of the `Dataset` are grouped together by length and then are padded
  and batched.

  This is useful for sequence tasks in which the elements have variable length.
  Grouping together elements that have similar lengths reduces the total
  fraction of padding in a batch which increases training step efficiency.

  Args:
    element_length_func: function from element in `Dataset` to `tf.int64`,
      determines the length of the element, which will determine the bucket it
      goes into.
    bucket_boundaries: `list<int>`, upper length boundaries of the buckets.
    bucket_batch_sizes: `list<int>`, batch size per bucket. Length should be
      `len(bucket_boundaries) + 1`.
    padded_shapes: Nested structure of `tf.TensorShape` to pass to
      @{tf.data.Dataset.padded_batch}. If not provided, will use
      `dataset.output_shapes`, which will result in variable length dimensions
      being padded out to the maximum length in each batch.
    padding_values: Values to pad with, passed to
      @{tf.data.Dataset.padded_batch}. Defaults to padding with 0.
    pad_to_bucket_boundary: bool, if `False`, will pad dimensions with unknown
      size to maximum length in batch. If `True`, will pad dimensions with
      unknown size to bucket boundary, and caller must ensure that the source
      `Dataset` does not contain any elements with length longer than
      `max(bucket_boundaries)`.

  Returns:
    A `Dataset` transformation function, which can be passed to
    @{tf.data.Dataset.apply}.

  Raises:
    ValueError: if `len(bucket_batch_sizes) != len(bucket_boundaries) + 1`.
  """
  with ops.name_scope("bucket_by_seq_length"):
    if len(bucket_batch_sizes) != (len(bucket_boundaries) + 1):
      raise ValueError(
          "len(bucket_batch_sizes) must equal len(bucket_boundaries) + 1")

    batch_sizes = constant_op.constant(bucket_batch_sizes, dtype=dtypes.int64)

    def element_to_bucket_id(element):
      """Return int64 id of the length bucket for this element."""
      seq_length = element_length_func(element)

      boundaries = list(bucket_boundaries)
      buckets_min = [np.iinfo(np.int32).min] + boundaries
      buckets_max = boundaries + [np.iinfo(np.int32).max]
      conditions_c = math_ops.logical_and(
          math_ops.less_equal(buckets_min, seq_length),
          math_ops.less(seq_length, buckets_max))
      bucket_id = math_ops.reduce_min(array_ops.where(conditions_c))

      return bucket_id

    def window_size_fn(bucket_id):
      # The window size is set to the batch size for this bucket
      window_size = batch_sizes[bucket_id]
      return window_size

    def make_padded_shapes(shapes, none_filler=None):
      padded = []
      for shape in nest.flatten(shapes):
        shape = tensor_shape.TensorShape(shape)
        shape = [
            none_filler if d.value is None else d
            for d in shape
        ]
        padded.append(shape)
      return nest.pack_sequence_as(shapes, padded)

    def batching_fn(bucket_id, grouped_dataset):
      """Batch elements in dataset."""
      batch_size = batch_sizes[bucket_id]
      none_filler = None
      if pad_to_bucket_boundary:
        err_msg = ("When pad_to_bucket_boundary=True, elements must have "
                   "length <= max(bucket_boundaries).")
        check = check_ops.assert_less(
            bucket_id,
            constant_op.constant(len(bucket_batch_sizes) - 1,
                                 dtype=dtypes.int64),
            message=err_msg)
        with ops.control_dependencies([check]):
          boundaries = constant_op.constant(bucket_boundaries,
                                            dtype=dtypes.int64)
          bucket_boundary = boundaries[bucket_id]
          none_filler = bucket_boundary
      shapes = make_padded_shapes(
          padded_shapes or grouped_dataset.output_shapes,
          none_filler=none_filler)
      return grouped_dataset.padded_batch(batch_size, shapes, padding_values)

    def _apply_fn(dataset):
      return dataset.apply(
          group_by_window(element_to_bucket_id, batching_fn,
                          window_size_func=window_size_fn))

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
