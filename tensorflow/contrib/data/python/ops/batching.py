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
"""Batching dataset transformations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import math_ops


def dense_to_sparse_batch(batch_size, row_shape):
  """A transformation that batches ragged elements into `tf.SparseTensor`s.

  Like `Dataset.padded_batch()`, this transformation combines multiple
  consecutive elements of the dataset, which might have different
  shapes, into a single element. The resulting element has three
  components (`indices`, `values`, and `dense_shape`), which
  comprise a `tf.SparseTensor` that represents the same data. The
  `row_shape` represents the dense shape of each row in the
  resulting `tf.SparseTensor`, to which the effective batch size is
  prepended. For example:

  ```python
  # NOTE: The following examples use `{ ... }` to represent the
  # contents of a dataset.
  a = { ['a', 'b', 'c'], ['a', 'b'], ['a', 'b', 'c', 'd'] }

  a.apply(tf.contrib.data.dense_to_sparse_batch(batch_size=2, row_shape=[6])) ==
  {
      ([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]],  # indices
       ['a', 'b', 'c', 'a', 'b'],                 # values
       [2, 6]),                                   # dense_shape
      ([[2, 0], [2, 1], [2, 2], [2, 3]],
       ['a', 'b', 'c', 'd'],
       [1, 6])
  }
  ```

  Args:
    batch_size: A `tf.int64` scalar `tf.Tensor`, representing the
      number of consecutive elements of this dataset to combine in a
      single batch.
    row_shape: A `tf.TensorShape` or `tf.int64` vector tensor-like
      object representing the equivalent dense shape of a row in the
      resulting `tf.SparseTensor`. Each element of this dataset must
      have the same rank as `row_shape`, and must have size less
      than or equal to `row_shape` in each dimension.

  Returns:
    A `Dataset` transformation function, which can be passed to
    @{tf.contrib.data.Dataset.apply}.
  """

  def _apply_fn(dataset):
    return DenseToSparseBatchDataset(dataset, batch_size, row_shape)

  return _apply_fn


def unbatch():
  """A Transformation which splits the elements of a dataset.

  For example, if elements of the dataset are shaped `[B, a0, a1, ...]`,
  where `B` may vary from element to element, then for each element in
  the dataset, the unbatched dataset will contain `B` consecutive elements
  of shape `[a0, a1, ...]`.

  Returns:
    A `Dataset` transformation function, which can be passed to
    @{tf.contrib.data.Dataset.apply}.
  """

  def _apply_fn(dataset):

    def unbatch_map(arg, *rest):
      if rest:
        return dataset_ops.Dataset.from_tensor_slices((arg,) + rest)
      else:
        return dataset_ops.Dataset.from_tensor_slices(arg)

    return dataset.flat_map(map_func=unbatch_map)

  return _apply_fn


def batch_and_drop_remainder(batch_size):
  """A batching transformation that omits the final small batch (if present).

  Like @{tf.contrib.data.Dataset.batch}, this transformation combines
  consecutive elements of this dataset into batches. However, if the batch
  size does not evenly divide the input dataset size, this transformation will
  drop the final smaller element.

  The following example illustrates the difference between this
  transformation and `Dataset.batch()`:

  ```python
  dataset = tf.contrib.data.Dataset.range(200)
  batched = dataset.apply(tf.contrib.data.batch_and_drop_remainder(128))
  print(batched.output_shapes)  # ==> "(128,)" (the batch dimension is known)
  ```

  By contrast, `dataset.batch(128)` would yield a two-element dataset with
  shapes `(128,)` and `(72,)`, so the batch dimension would not be statically
  known.

  Args:
    batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
        consecutive elements of this dataset to combine in a single batch.

  Returns:
    A `Dataset` transformation function, which can be passed to
    @{tf.contrib.data.Dataset.apply}
  """

  def _apply_fn(dataset):
    """Function from `Dataset` to `Dataset` that applies the transformation."""
    tensor_batch_size = ops.convert_to_tensor(
        batch_size, dtype=dtypes.int64, name="batch_size")

    batched = dataset.batch(tensor_batch_size)
    flattened = _RestructuredDataset(batched,
                                     tuple(nest.flatten(batched.output_types)))

    def _predicate(*xs):
      """Return `True` if this element is a full batch."""
      # Extract the dynamic batch size from the first component of the flattened
      # batched element.
      first_component = xs[0]
      first_component_batch_size = array_ops.shape(
          first_component, out_type=dtypes.int64)[0]

      return math_ops.equal(first_component_batch_size, tensor_batch_size)

    filtered = flattened.filter(_predicate)

    maybe_constant_batch_size = tensor_util.constant_value(tensor_batch_size)

    def _set_first_dimension(shape):
      return shape.merge_with(
          tensor_shape.vector(maybe_constant_batch_size).concatenate(shape[1:]))

    known_shapes = nest.map_structure(_set_first_dimension,
                                      batched.output_shapes)
    return _RestructuredDataset(filtered, batched.output_types, known_shapes)

  return _apply_fn


class DenseToSparseBatchDataset(dataset_ops.Dataset):
  """A `Dataset` that batches ragged dense elements into `tf.SparseTensor`s."""

  def __init__(self, input_dataset, batch_size, row_shape):
    """See `Dataset.dense_to_sparse_batch()` for more details."""
    super(DenseToSparseBatchDataset, self).__init__()
    if not isinstance(input_dataset.output_types, dtypes.DType):
      raise TypeError("DenseToSparseDataset requires an input whose elements "
                      "have a single component, whereas the input has %r." %
                      input_dataset.output_types)
    self._input_dataset = input_dataset
    self._batch_size = batch_size
    # pylint: disable=protected-access
    self._row_shape = dataset_ops._partial_shape_to_tensor(row_shape)
    # pylint: enable=protected-access

  def _as_variant_tensor(self):
    return gen_dataset_ops.dense_to_sparse_batch_dataset(
        self._input_dataset._as_variant_tensor(),  # pylint: disable=protected-access
        self._batch_size,
        self._row_shape,
        output_shapes=self.output_shapes,
        output_types=self.output_types)

  @property
  def output_shapes(self):
    num_elements = tensor_shape.Dimension(None)
    return (tensor_shape.matrix(num_elements, self._row_shape.shape[0] + 1),
            tensor_shape.vector(num_elements),
            tensor_shape.vector(self._row_shape.shape[0] + 1))

  @property
  def output_types(self):
    return (dtypes.int64, self._input_dataset.output_types, dtypes.int64)


class _RestructuredDataset(dataset_ops.Dataset):
  """An internal helper for changing the structure and shape of a dataset."""

  def __init__(self, dataset, output_types, output_shapes=None):
    """Creates a new dataset with the given output types and shapes.

    The given `dataset` must have a structure that is convertible:
    * `dataset.output_types` must be the same as `output_types` module nesting.
    * Each shape in `dataset.output_shapes` must be compatible with each shape
      in `output_shapes` (if given).

    Note: This helper permits "unsafe casts" for shapes, equivalent to using
    `tf.Tensor.set_shape()` where domain-specific knowledge is available.

    Args:
      dataset: A `Dataset` object.
      output_types: A nested structure of `tf.DType` objects.
      output_shapes: (Optional.) A nested structure of `tf.TensorShape` objects.
        If omitted, the shapes will be inherited from `dataset`.

    Raises:
      ValueError: If either `output_types` or `output_shapes` is not compatible
        with the structure of `dataset`.
    """
    super(_RestructuredDataset, self).__init__()
    self._dataset = dataset

    # Validate that the types are compatible.
    output_types = nest.map_structure(dtypes.as_dtype, output_types)
    flat_original_types = nest.flatten(dataset.output_types)
    flat_new_types = nest.flatten(output_types)
    if flat_original_types != flat_new_types:
      raise ValueError(
          "Dataset with output types %r cannot be restructured to have output "
          "types %r" % (dataset.output_types, output_types))

    self._output_types = output_types

    if output_shapes is None:
      # Inherit shapes from the original `dataset`.
      self._output_shapes = nest.pack_sequence_as(output_types,
                                                  nest.flatten(
                                                      dataset.output_shapes))
    else:
      # Validate that the shapes are compatible.
      nest.assert_same_structure(output_types, output_shapes)
      flat_original_shapes = nest.flatten(dataset.output_shapes)
      flat_new_shapes = nest.flatten_up_to(output_types, output_shapes)

      for original_shape, new_shape in zip(flat_original_shapes,
                                           flat_new_shapes):
        if not original_shape.is_compatible_with(new_shape):
          raise ValueError(
              "Dataset with output shapes %r cannot be restructured to have "
              "incompatible output shapes %r" % (dataset.output_shapes,
                                                 output_shapes))
      self._output_shapes = nest.map_structure_up_to(
          output_types, tensor_shape.as_shape, output_shapes)

  def _as_variant_tensor(self):
    return self._dataset._as_variant_tensor()  # pylint: disable=protected-access

  @property
  def output_types(self):
    return self._output_types

  @property
  def output_shapes(self):
    return self._output_shapes
