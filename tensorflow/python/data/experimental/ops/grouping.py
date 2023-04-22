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
from tensorflow.python.data.util import structure
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.util import deprecation
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


@deprecation.deprecated(None, "Use `tf.data.Dataset.group_by_window(...)`.")
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

  def _apply_fn(dataset):
    """Function from `Dataset` to `Dataset` that applies the transformation."""
    return dataset.group_by_window(
        key_func=key_func,
        reduce_func=reduce_func,
        window_size=window_size,
        window_size_func=window_size_func)

  return _apply_fn


@deprecation.deprecated(None,
                        "Use `tf.data.Dataset.bucket_by_sequence_length(...)`.")
@tf_export("data.experimental.bucket_by_sequence_length")
def bucket_by_sequence_length(element_length_func,
                              bucket_boundaries,
                              bucket_batch_sizes,
                              padded_shapes=None,
                              padding_values=None,
                              pad_to_bucket_boundary=False,
                              no_padding=False,
                              drop_remainder=False):
  """A transformation that buckets elements in a `Dataset` by length.

  Elements of the `Dataset` are grouped together by length and then are padded
  and batched.

  This is useful for sequence tasks in which the elements have variable length.
  Grouping together elements that have similar lengths reduces the total
  fraction of padding in a batch which increases training step efficiency.

  Below is an example to bucketize the input data to the 3 buckets
  "[0, 3), [3, 5), [5, inf)" based on sequence length, with batch size 2.

  >>> elements = [
  ...   [0], [1, 2, 3, 4], [5, 6, 7],
  ...   [7, 8, 9, 10, 11], [13, 14, 15, 16, 19, 20], [21, 22]]

  >>> dataset = tf.data.Dataset.from_generator(
  ...     lambda: elements, tf.int64, output_shapes=[None])

  >>> dataset = dataset.apply(
  ...     tf.data.experimental.bucket_by_sequence_length(
  ...         element_length_func=lambda elem: tf.shape(elem)[0],
  ...         bucket_boundaries=[3, 5],
  ...         bucket_batch_sizes=[2, 2, 2]))

  >>> for elem in dataset.as_numpy_iterator():
  ...   print(elem)
  [[1 2 3 4]
   [5 6 7 0]]
  [[ 7  8  9 10 11  0]
   [13 14 15 16 19 20]]
  [[ 0  0]
   [21 22]]

  There is also a possibility to pad the dataset till the bucket boundary.
  You can also provide which value to be used while padding the data.
  Below example uses `-1` as padding and it also shows the input data
  being bucketizied to two buckets "[0,3], [4,6]".

  >>> elements = [
  ...   [0], [1, 2, 3, 4], [5, 6, 7],
  ...   [7, 8, 9, 10, 11], [13, 14, 15, 16, 19, 20], [21, 22]]

  >>> dataset = tf.data.Dataset.from_generator(
  ...   lambda: elements, tf.int32, output_shapes=[None])

  >>> dataset = dataset.apply(
  ...     tf.data.experimental.bucket_by_sequence_length(
  ...         element_length_func=lambda elem: tf.shape(elem)[0],
  ...         bucket_boundaries=[4, 7],
  ...         bucket_batch_sizes=[2, 2, 2],
  ...         pad_to_bucket_boundary=True,
  ...         padding_values=-1))

  >>> for elem in dataset.as_numpy_iterator():
  ...   print(elem)
  [[ 0 -1 -1]
   [ 5  6  7]]
  [[ 1  2  3  4 -1 -1]
   [ 7  8  9 10 11 -1]]
  [[21 22 -1]]
  [[13 14 15 16 19 20]]

  When using `pad_to_bucket_boundary` option, it can be seen that it is
  not always possible to maintain the bucket batch size.
  You can drop the batches that do not maintain the bucket batch size by
  using the option `drop_remainder`. Using the same input data as in the
  above example you get the following result.

  >>> elements = [
  ...   [0], [1, 2, 3, 4], [5, 6, 7],
  ...   [7, 8, 9, 10, 11], [13, 14, 15, 16, 19, 20], [21, 22]]

  >>> dataset = tf.data.Dataset.from_generator(
  ...   lambda: elements, tf.int32, output_shapes=[None])

  >>> dataset = dataset.apply(
  ...     tf.data.experimental.bucket_by_sequence_length(
  ...         element_length_func=lambda elem: tf.shape(elem)[0],
  ...         bucket_boundaries=[4, 7],
  ...         bucket_batch_sizes=[2, 2, 2],
  ...         pad_to_bucket_boundary=True,
  ...         padding_values=-1,
  ...         drop_remainder=True))

  >>> for elem in dataset.as_numpy_iterator():
  ...   print(elem)
  [[ 0 -1 -1]
   [ 5  6  7]]
  [[ 1  2  3  4 -1 -1]
   [ 7  8  9 10 11 -1]]

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
      need to be either of type `tf.sparse.SparseTensor` or of same shape).
    drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
      whether the last batch should be dropped in the case it has fewer than
      `batch_size` elements; the default behavior is not to drop the smaller
      batch.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.

  Raises:
    ValueError: if `len(bucket_batch_sizes) != len(bucket_boundaries) + 1`.
  """

  def _apply_fn(dataset):
    return dataset.bucket_by_sequence_length(
        element_length_func=element_length_func,
        bucket_boundaries=bucket_boundaries,
        bucket_batch_sizes=bucket_batch_sizes,
        padded_shapes=padded_shapes,
        padding_values=padding_values,
        pad_to_bucket_boundary=pad_to_bucket_boundary,
        no_padding=no_padding,
        drop_remainder=drop_remainder)

  return _apply_fn


class _GroupByReducerDataset(dataset_ops.UnaryDataset):
  """A `Dataset` that groups its input and performs a reduction."""

  def __init__(self, input_dataset, key_func, reducer):
    """See `group_by_reducer()` for details."""
    self._input_dataset = input_dataset
    self._make_key_func(key_func, input_dataset)
    self._make_init_func(reducer.init_func)
    self._make_reduce_func(reducer.reduce_func, input_dataset)
    self._make_finalize_func(reducer.finalize_func)
    variant_tensor = ged_ops.experimental_group_by_reducer_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._key_func.function.captured_inputs,
        self._init_func.function.captured_inputs,
        self._reduce_func.function.captured_inputs,
        self._finalize_func.function.captured_inputs,
        key_func=self._key_func.function,
        init_func=self._init_func.function,
        reduce_func=self._reduce_func.function,
        finalize_func=self._finalize_func.function,
        **self._flat_structure)
    super(_GroupByReducerDataset, self).__init__(input_dataset, variant_tensor)

  def _make_key_func(self, key_func, input_dataset):
    """Make wrapping defun for key_func."""
    self._key_func = dataset_ops.StructuredFunctionWrapper(
        key_func, self._transformation_name(), dataset=input_dataset)
    if not self._key_func.output_structure.is_compatible_with(
        tensor_spec.TensorSpec([], dtypes.int64)):
      raise ValueError(
          "`key_func` must return a single tf.int64 tensor. "
          "Got type=%s and shape=%s"
          % (self._key_func.output_types, self._key_func.output_shapes))

  def _make_init_func(self, init_func):
    """Make wrapping defun for init_func."""
    self._init_func = dataset_ops.StructuredFunctionWrapper(
        init_func,
        self._transformation_name(),
        input_structure=tensor_spec.TensorSpec([], dtypes.int64))

  def _make_reduce_func(self, reduce_func, input_dataset):
    """Make wrapping defun for reduce_func."""

    # Iteratively rerun the reduce function until reaching a fixed point on
    # `self._state_structure`.
    self._state_structure = self._init_func.output_structure
    state_types = self._init_func.output_types
    state_shapes = self._init_func.output_shapes
    state_classes = self._init_func.output_classes
    need_to_rerun = True
    while need_to_rerun:

      wrapped_func = dataset_ops.StructuredFunctionWrapper(
          reduce_func,
          self._transformation_name(),
          input_structure=(self._state_structure, input_dataset.element_spec),
          add_to_graph=False)

      # Extract and validate class information from the returned values.
      for new_state_class, state_class in zip(
          nest.flatten(wrapped_func.output_classes),
          nest.flatten(state_classes)):
        if not issubclass(new_state_class, state_class):
          raise TypeError(
              "The element classes for the new state must match the initial "
              "state. Expected %s; got %s." %
              (self._state_classes, wrapped_func.output_classes))

      # Extract and validate type information from the returned values.
      for new_state_type, state_type in zip(
          nest.flatten(wrapped_func.output_types), nest.flatten(state_types)):
        if new_state_type != state_type:
          raise TypeError(
              "The element types for the new state must match the initial "
              "state. Expected %s; got %s." %
              (self._init_func.output_types, wrapped_func.output_types))

      # Extract shape information from the returned values.
      flat_state_shapes = nest.flatten(state_shapes)
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
        state_shapes = nest.pack_sequence_as(
            self._init_func.output_shapes, weakened_state_shapes)
        self._state_structure = structure.convert_legacy_structure(
            state_types, state_shapes, state_classes)

    self._reduce_func = wrapped_func
    self._reduce_func.function.add_to_graph(ops.get_default_graph())

  def _make_finalize_func(self, finalize_func):
    """Make wrapping defun for finalize_func."""
    self._finalize_func = dataset_ops.StructuredFunctionWrapper(
        finalize_func, self._transformation_name(),
        input_structure=self._state_structure)

  @property
  def element_spec(self):
    return self._finalize_func.output_structure

  def _functions(self):
    return [
        self._key_func, self._init_func, self._reduce_func, self._finalize_func
    ]

  def _transformation_name(self):
    return "tf.data.experimental.group_by_reducer()"


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
