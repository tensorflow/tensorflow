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
from tensorflow.python.data.util import structure
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export


@tf_export("data.experimental.group_by_reducer")
def group_by_reducer(key_func, reducer):
  """A transformation that groups elements and performs a reduction.

  This transformation maps element of a dataset to a key using `key_func` and
  groups the elements by key. The `reducer` is used to process each group; its
  `init_func` is used to initialize state for each group when it is created, the
  `reduce_func` is used to update the state every time an element is mapped to
  the matching group, and the `finalize_func` is used to map the final state to
  an output value.

  Args:
    key_func: A function mapping a nested structure of tensors
      (having shapes and types defined by `self.output_shapes` and
      `self.output_types`) to a scalar `tf.int64` tensor.
    reducer: An instance of `Reducer`, which captures the reduction logic using
      the `init_func`, `reduce_func`, and `finalize_func` functions.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):
    """Function from `Dataset` to `Dataset` that applies the transformation."""
    return _GroupByReducerDataset(dataset, key_func, reducer)

  return _apply_fn


@tf_export("data.experimental.group_by_window")
def group_by_window(key_func,
                    reduce_func,
                    window_size=None,
                    window_size_func=None):
  """A transformation that groups windows of elements by key and reduces them.

  This transformation maps each consecutive element in a dataset to a key
  using `key_func` and groups the elements by key. It then applies
  `reduce_func` to at most `window_size_func(key)` elements matching the same
  key. All except the final window for each key will contain
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
    `tf.data.Dataset.apply`.

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
    return _GroupByWindowDataset(dataset, key_func, reduce_func,
                                 window_size_func)

  return _apply_fn


@tf_export("data.experimental.bucket_by_sequence_length")
def bucket_by_sequence_length(element_length_func,
                              bucket_boundaries,
                              bucket_batch_sizes,
                              padded_shapes=None,
                              padding_values=None,
                              pad_to_bucket_boundary=False,
                              no_padding=False):
  """A transformation that buckets elements in a `Dataset` by length.

  Elements of the `Dataset` are grouped together by length and then are padded
  and batched.

  This is useful for sequence tasks in which the elements have variable length.
  Grouping together elements that have similar lengths reduces the total
  fraction of padding in a batch which increases training step efficiency.

  Args:
    element_length_func: function from element in `Dataset` to `tf.int32`,
      determines the length of the element, which will determine the bucket it
      goes into.
    bucket_boundaries: `list<int>`, upper length boundaries of the buckets.
    bucket_batch_sizes: `list<int>`, batch size per bucket. Length should be
      `len(bucket_boundaries) + 1`.
    padded_shapes: Nested structure of `tf.TensorShape` to pass to
      `tf.data.Dataset.padded_batch`. If not provided, will use
      `dataset.output_shapes`, which will result in variable length dimensions
      being padded out to the maximum length in each batch.
    padding_values: Values to pad with, passed to
      `tf.data.Dataset.padded_batch`. Defaults to padding with 0.
    pad_to_bucket_boundary: bool, if `False`, will pad dimensions with unknown
      size to maximum length in batch. If `True`, will pad dimensions with
      unknown size to bucket boundary minus 1 (i.e., the maximum length in each
      bucket), and caller must ensure that the source `Dataset` does not contain
      any elements with length longer than `max(bucket_boundaries)`.
    no_padding: `bool`, indicates whether to pad the batch features (features
      need to be either of type `tf.SparseTensor` or of same shape).

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.

  Raises:
    ValueError: if `len(bucket_batch_sizes) != len(bucket_boundaries) + 1`.
  """
  with ops.name_scope("bucket_by_seq_length"):
    if len(bucket_batch_sizes) != (len(bucket_boundaries) + 1):
      raise ValueError(
          "len(bucket_batch_sizes) must equal len(bucket_boundaries) + 1")

    batch_sizes = constant_op.constant(bucket_batch_sizes, dtype=dtypes.int64)

    def element_to_bucket_id(*args):
      """Return int64 id of the length bucket for this element."""
      seq_length = element_length_func(*args)

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
            none_filler if tensor_shape.dimension_value(d) is None else d
            for d in shape
        ]
        padded.append(shape)
      return nest.pack_sequence_as(shapes, padded)

    def batching_fn(bucket_id, grouped_dataset):
      """Batch elements in dataset."""
      batch_size = window_size_fn(bucket_id)
      if no_padding:
        return grouped_dataset.batch(batch_size)
      none_filler = None
      if pad_to_bucket_boundary:
        err_msg = ("When pad_to_bucket_boundary=True, elements must have "
                   "length < max(bucket_boundaries).")
        check = check_ops.assert_less(
            bucket_id,
            constant_op.constant(len(bucket_batch_sizes) - 1,
                                 dtype=dtypes.int64),
            message=err_msg)
        with ops.control_dependencies([check]):
          boundaries = constant_op.constant(bucket_boundaries,
                                            dtype=dtypes.int64)
          bucket_boundary = boundaries[bucket_id]
          none_filler = bucket_boundary - 1
      shapes = make_padded_shapes(
          padded_shapes or grouped_dataset.output_shapes,
          none_filler=none_filler)
      return grouped_dataset.padded_batch(batch_size, shapes, padding_values)

    def _apply_fn(dataset):
      return dataset.apply(
          group_by_window(element_to_bucket_id, batching_fn,
                          window_size_func=window_size_fn))

    return _apply_fn


class _GroupByReducerDataset(dataset_ops.UnaryDataset):
  """A `Dataset` that groups its input and performs a reduction."""

  def __init__(self, input_dataset, key_func, reducer):
    """See `group_by_reducer()` for details."""
    super(_GroupByReducerDataset, self).__init__(input_dataset)

    self._input_dataset = input_dataset

    self._make_key_func(key_func, input_dataset)
    self._make_init_func(reducer.init_func)
    self._make_reduce_func(reducer.reduce_func, input_dataset)
    self._make_finalize_func(reducer.finalize_func)

  def _make_key_func(self, key_func, input_dataset):
    """Make wrapping defun for key_func."""
    wrapped_func = dataset_ops.StructuredFunctionWrapper(
        key_func, self._transformation_name(), dataset=input_dataset)
    if not (
        wrapped_func.output_types == dtypes.int64 and
        wrapped_func.output_shapes.is_compatible_with(tensor_shape.scalar())):
      raise ValueError(
          "`key_func` must return a single tf.int64 tensor. "
          "Got type=%s and shape=%s"
          % (wrapped_func.output_types, wrapped_func.output_shapes))
    self._key_func = wrapped_func

  def _make_init_func(self, init_func):
    """Make wrapping defun for init_func."""
    wrapped_func = dataset_ops.StructuredFunctionWrapper(
        init_func,
        self._transformation_name(),
        input_classes=ops.Tensor,
        input_shapes=tensor_shape.scalar(),
        input_types=dtypes.int64)
    self._init_func = wrapped_func
    self._state_classes = wrapped_func.output_classes
    self._state_shapes = wrapped_func.output_shapes
    self._state_types = wrapped_func.output_types

  def _make_reduce_func(self, reduce_func, input_dataset):
    """Make wrapping defun for reduce_func."""

    # Iteratively rerun the reduce function until reaching a fixed point on
    # `self._state_shapes`.
    need_to_rerun = True
    while need_to_rerun:

      wrapped_func = dataset_ops.StructuredFunctionWrapper(
          reduce_func,
          self._transformation_name(),
          input_classes=(self._state_classes, input_dataset.output_classes),
          input_shapes=(self._state_shapes, input_dataset.output_shapes),
          input_types=(self._state_types, input_dataset.output_types),
          add_to_graph=False)

      # Extract and validate class information from the returned values.
      for new_state_class, state_class in zip(
          nest.flatten(wrapped_func.output_classes),
          nest.flatten(self._state_classes)):
        if not issubclass(new_state_class, state_class):
          raise TypeError(
              "The element classes for the new state must match the initial "
              "state. Expected %s; got %s." %
              (self._state_classes, wrapped_func.output_classes))

      # Extract and validate type information from the returned values.
      for new_state_type, state_type in zip(
          nest.flatten(wrapped_func.output_types),
          nest.flatten(self._state_types)):
        if new_state_type != state_type:
          raise TypeError(
              "The element types for the new state must match the initial "
              "state. Expected %s; got %s." %
              (self._state_types, wrapped_func.output_types))

      # Extract shape information from the returned values.
      flat_state_shapes = nest.flatten(self._state_shapes)
      flat_new_state_shapes = nest.flatten(wrapped_func.output_shapes)
      weakened_state_shapes = [
          original.most_specific_compatible_shape(new)
          for original, new in zip(flat_state_shapes, flat_new_state_shapes)
      ]

      need_to_rerun = False
      for original_shape, weakened_shape in zip(flat_state_shapes,
                                                weakened_state_shapes):
        if original_shape.ndims is not None and (
            weakened_shape.ndims is None or
            original_shape.as_list() != weakened_shape.as_list()):
          need_to_rerun = True
          break

      if need_to_rerun:
        self._state_shapes = nest.pack_sequence_as(self._state_shapes,
                                                   weakened_state_shapes)

    self._reduce_func = wrapped_func
    self._reduce_func.function.add_to_graph(ops.get_default_graph())

  def _make_finalize_func(self, finalize_func):
    """Make wrapping defun for finalize_func."""
    wrapped_func = dataset_ops.StructuredFunctionWrapper(
        finalize_func,
        self._transformation_name(),
        input_classes=self._state_classes,
        input_shapes=self._state_shapes,
        input_types=self._state_types)
    self._finalize_func = wrapped_func
    self._output_classes = wrapped_func.output_classes
    self._output_shapes = wrapped_func.output_shapes
    self._output_types = wrapped_func.output_types

  @property
  def output_classes(self):
    return self._output_classes

  @property
  def output_shapes(self):
    return self._output_shapes

  @property
  def output_types(self):
    return self._output_types

  def _functions(self):
    return [
        self._key_func, self._init_func, self._reduce_func, self._finalize_func
    ]

  def _as_variant_tensor(self):
    return ged_ops.experimental_group_by_reducer_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
        self._key_func.function.captured_inputs,
        self._init_func.function.captured_inputs,
        self._reduce_func.function.captured_inputs,
        self._finalize_func.function.captured_inputs,
        key_func=self._key_func.function,
        init_func=self._init_func.function,
        reduce_func=self._reduce_func.function,
        finalize_func=self._finalize_func.function,
        **dataset_ops.flat_structure(self))

  def _transformation_name(self):
    return "tf.data.experimental.group_by_reducer()"


class _GroupByWindowDataset(dataset_ops.UnaryDataset):
  """A `Dataset` that groups its input and performs a windowed reduction."""

  def __init__(self, input_dataset, key_func, reduce_func, window_size_func):
    """See `group_by_window()` for details."""
    super(_GroupByWindowDataset, self).__init__(input_dataset)

    self._input_dataset = input_dataset

    self._make_key_func(key_func, input_dataset)
    self._make_reduce_func(reduce_func, input_dataset)
    self._make_window_size_func(window_size_func)

  def _make_window_size_func(self, window_size_func):
    """Make wrapping defun for window_size_func."""

    def window_size_func_wrapper(key):
      return ops.convert_to_tensor(window_size_func(key), dtype=dtypes.int64)
    wrapped_func = dataset_ops.StructuredFunctionWrapper(
        window_size_func_wrapper,
        self._transformation_name(),
        input_classes=ops.Tensor,
        input_shapes=tensor_shape.scalar(),
        input_types=dtypes.int64)
    if not (
        wrapped_func.output_types == dtypes.int64 and
        wrapped_func.output_shapes.is_compatible_with(tensor_shape.scalar())):
      raise ValueError(
          "`window_size_func` must return a single tf.int64 scalar tensor.")
    self._window_size_func = wrapped_func

  def _make_key_func(self, key_func, input_dataset):
    """Make wrapping defun for key_func."""

    def key_func_wrapper(*args):
      return ops.convert_to_tensor(key_func(*args), dtype=dtypes.int64)
    wrapped_func = dataset_ops.StructuredFunctionWrapper(
        key_func_wrapper, self._transformation_name(), dataset=input_dataset)
    if not (
        wrapped_func.output_types == dtypes.int64 and
        wrapped_func.output_shapes.is_compatible_with(tensor_shape.scalar())):
      raise ValueError(
          "`key_func` must return a single tf.int64 scalar tensor.")
    self._key_func = wrapped_func

  def _make_reduce_func(self, reduce_func, input_dataset):
    """Make wrapping defun for reduce_func."""
    nested_dataset = dataset_ops.DatasetStructure(
        structure.Structure._from_legacy_structure(  # pylint: disable=protected-access
            input_dataset.output_types, input_dataset.output_shapes,
            input_dataset.output_classes))
    wrapped_func = dataset_ops.StructuredFunctionWrapper(
        reduce_func,
        self._transformation_name(),
        input_classes=(ops.Tensor, nested_dataset),
        input_shapes=(tensor_shape.scalar(), nested_dataset),
        input_types=(dtypes.int64, nested_dataset))
    if not isinstance(
        wrapped_func.output_structure, dataset_ops.DatasetStructure):
      raise TypeError("`reduce_func` must return a `Dataset` object.")
    # pylint: disable=protected-access
    element_structure = wrapped_func.output_structure._element_structure
    self._output_classes = element_structure._to_legacy_output_classes()
    self._output_types = element_structure._to_legacy_output_types()
    self._output_shapes = element_structure._to_legacy_output_shapes()
    self._reduce_func = wrapped_func

  @property
  def output_classes(self):
    return self._output_classes

  @property
  def output_shapes(self):
    return self._output_shapes

  @property
  def output_types(self):
    return self._output_types

  def _functions(self):
    return [self._key_func, self._reduce_func, self._window_size_func]

  def _as_variant_tensor(self):
    return ged_ops.experimental_group_by_window_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
        self._key_func.function.captured_inputs,
        self._reduce_func.function.captured_inputs,
        self._window_size_func.function.captured_inputs,
        key_func=self._key_func.function,
        reduce_func=self._reduce_func.function,
        window_size_func=self._window_size_func.function,
        **dataset_ops.flat_structure(self))

  def _transformation_name(self):
    return "tf.data.experimental.group_by_window()"


@tf_export("data.experimental.Reducer")
class Reducer(object):
  """A reducer is used for reducing a set of elements.

  A reducer is represented as a tuple of the three functions:
    1) initialization function: key => initial state
    2) reduce function: (old state, input) => new state
    3) finalization function: state => result
  """

  def __init__(self, init_func, reduce_func, finalize_func):
    self._init_func = init_func
    self._reduce_func = reduce_func
    self._finalize_func = finalize_func

  @property
  def init_func(self):
    return self._init_func

  @property
  def reduce_func(self):
    return self._reduce_func

  @property
  def finalize_func(self):
    return self._finalize_func
