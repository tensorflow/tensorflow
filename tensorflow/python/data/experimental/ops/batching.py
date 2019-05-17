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
from tensorflow.python.data.util import convert
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


@tf_export("data.experimental.dense_to_sparse_batch")
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

  a.apply(tf.data.experimental.dense_to_sparse_batch(
      batch_size=2, row_shape=[6])) ==
  {
      ([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]],  # indices
       ['a', 'b', 'c', 'a', 'b'],                 # values
       [2, 6]),                                   # dense_shape
      ([[0, 0], [0, 1], [0, 2], [0, 3]],
       ['a', 'b', 'c', 'd'],
       [1, 6])
  }
  ```

  Args:
    batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
      consecutive elements of this dataset to combine in a single batch.
    row_shape: A `tf.TensorShape` or `tf.int64` vector tensor-like object
      representing the equivalent dense shape of a row in the resulting
      `tf.SparseTensor`. Each element of this dataset must have the same rank as
      `row_shape`, and must have size less than or equal to `row_shape` in each
      dimension.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):
    return _DenseToSparseBatchDataset(dataset, batch_size, row_shape)

  return _apply_fn


@deprecation.deprecated(None, "Use `tf.data.experimental.map_and_batch()")
@tf_export(v1=["data.experimental.map_and_batch_with_legacy_function"])
def map_and_batch_with_legacy_function(map_func,
                                       batch_size,
                                       num_parallel_batches=None,
                                       drop_remainder=False,
                                       num_parallel_calls=None):
  """Fused implementation of `map` and `batch`.

  NOTE: This is an escape hatch for existing uses of `map_and_batch` that do not
  work with V2 functions. New uses are strongly discouraged and existing uses
  should migrate to `map_and_batch` as this method will not be removed in V2.

  Args:
    map_func: A function mapping a nested structure of tensors to another
      nested structure of tensors.
    batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
      consecutive elements of this dataset to combine in a single batch.
    num_parallel_batches: (Optional.) A `tf.int64` scalar `tf.Tensor`,
      representing the number of batches to create in parallel. On one hand,
      higher values can help mitigate the effect of stragglers. On the other
      hand, higher values can increase contention if CPU is scarce.
    drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
      whether the last batch should be dropped in case its size is smaller than
      desired; the default behavior is not to drop the smaller batch.
    num_parallel_calls: (Optional.) A `tf.int32` scalar `tf.Tensor`,
      representing the number of elements to process in parallel. If not
      specified, `batch_size * num_parallel_batches` elements will be processed
      in parallel. If the value `tf.data.experimental.AUTOTUNE` is used, then
      the number of parallel calls is set dynamically based on available CPU.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.

  Raises:
    ValueError: If both `num_parallel_batches` and `num_parallel_calls` are
      specified.
  """

  if num_parallel_batches is None and num_parallel_calls is None:
    num_parallel_calls = batch_size
  elif num_parallel_batches is not None and num_parallel_calls is None:
    num_parallel_calls = batch_size * num_parallel_batches
  elif num_parallel_batches is not None and num_parallel_calls is not None:
    raise ValueError("The `num_parallel_batches` and `num_parallel_calls` "
                     "arguments are mutually exclusive.")

  def _apply_fn(dataset):
    return _MapAndBatchDataset(dataset, map_func, batch_size,
                               num_parallel_calls, drop_remainder,
                               use_legacy_function=True)

  return _apply_fn


@deprecation.deprecated(
    None,
    "Use `tf.data.Dataset.map(map_func, num_parallel_calls)` followed by "
    "`tf.data.Dataset.batch(batch_size, drop_remainder)`. Static tf.data "
    "optimizations will take care of using the fused implementation.")
@tf_export("data.experimental.map_and_batch")
def map_and_batch(map_func,
                  batch_size,
                  num_parallel_batches=None,
                  drop_remainder=False,
                  num_parallel_calls=None):
  """Fused implementation of `map` and `batch`.

  Maps `map_func` across `batch_size` consecutive elements of this dataset
  and then combines them into a batch. Functionally, it is equivalent to `map`
  followed by `batch`. However, by fusing the two transformations together, the
  implementation can be more efficient. Surfacing this transformation in the API
  is temporary. Once automatic input pipeline optimization is implemented,
  the fusing of `map` and `batch` will happen automatically and this API will be
  deprecated.

  Args:
    map_func: A function mapping a nested structure of tensors to another
      nested structure of tensors.
    batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
      consecutive elements of this dataset to combine in a single batch.
    num_parallel_batches: (Optional.) A `tf.int64` scalar `tf.Tensor`,
      representing the number of batches to create in parallel. On one hand,
      higher values can help mitigate the effect of stragglers. On the other
      hand, higher values can increase contention if CPU is scarce.
    drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
      whether the last batch should be dropped in case its size is smaller than
      desired; the default behavior is not to drop the smaller batch.
    num_parallel_calls: (Optional.) A `tf.int32` scalar `tf.Tensor`,
      representing the number of elements to process in parallel. If not
      specified, `batch_size * num_parallel_batches` elements will be processed
      in parallel. If the value `tf.data.experimental.AUTOTUNE` is used, then
      the number of parallel calls is set dynamically based on available CPU.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.

  Raises:
    ValueError: If both `num_parallel_batches` and `num_parallel_calls` are
      specified.
  """

  if num_parallel_batches is None and num_parallel_calls is None:
    num_parallel_calls = batch_size
  elif num_parallel_batches is not None and num_parallel_calls is None:
    num_parallel_calls = batch_size * num_parallel_batches
  elif num_parallel_batches is not None and num_parallel_calls is not None:
    raise ValueError("The `num_parallel_batches` and `num_parallel_calls` "
                     "arguments are mutually exclusive.")

  def _apply_fn(dataset):
    return _MapAndBatchDataset(dataset, map_func, batch_size,
                               num_parallel_calls, drop_remainder)

  return _apply_fn


@tf_export("data.experimental.unbatch")
def unbatch():
  """Splits elements of a dataset into multiple elements on the batch dimension.

  For example, if elements of the dataset are shaped `[B, a0, a1, ...]`,
  where `B` may vary for each input element, then for each element in the
  dataset, the unbatched dataset will contain `B` consecutive elements
  of shape `[a0, a1, ...]`.

  ```python
  # NOTE: The following example uses `{ ... }` to represent the contents
  # of a dataset.
  a = { ['a', 'b', 'c'], ['a', 'b'], ['a', 'b', 'c', 'd'] }

  a.apply(tf.data.experimental.unbatch()) == {
      'a', 'b', 'c', 'a', 'b', 'a', 'b', 'c', 'd'}
  ```

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):
    """Function from `Dataset` to `Dataset` that applies the transformation."""

    # NOTE(mrry): We must ensure that any SparseTensors in `dataset`
    # are normalized to the rank-1 dense representation, so that the
    # sparse-oblivious unbatching logic will slice them
    # appropriately. This leads to a somewhat inefficient re-encoding step
    # for all SparseTensor components.
    # TODO(mrry): Consider optimizing this in future if it turns out to be
    # a bottleneck.
    def normalize(arg, *rest):
      # pylint: disable=protected-access
      if rest:
        return dataset._element_structure._to_batched_tensor_list((arg,) + rest)
      else:
        return dataset._element_structure._to_batched_tensor_list(arg)

    normalized_dataset = dataset.map(normalize)

    # NOTE(mrry): Our `map()` has lost information about the sparseness
    # of any SparseTensor components, so re-apply the structure of the
    # original dataset.
    restructured_dataset = _RestructuredDataset(
        normalized_dataset,
        dataset_ops.get_legacy_output_types(dataset),
        dataset_ops.get_legacy_output_shapes(dataset),
        dataset_ops.get_legacy_output_classes(dataset),
        allow_unsafe_cast=True)
    return _UnbatchDataset(restructured_dataset)

  return _apply_fn


class _DenseToSparseBatchDataset(dataset_ops.UnaryDataset):
  """A `Dataset` that batches ragged dense elements into `tf.SparseTensor`s."""

  def __init__(self, input_dataset, batch_size, row_shape):
    """See `Dataset.dense_to_sparse_batch()` for more details."""
    if not isinstance(
        dataset_ops.get_legacy_output_types(input_dataset), dtypes.DType):
      raise TypeError("DenseToSparseDataset requires an input whose elements "
                      "have a single component, whereas the input has %r." %
                      dataset_ops.get_legacy_output_types(input_dataset))
    self._input_dataset = input_dataset
    self._batch_size = batch_size
    self._row_shape = row_shape
    self._structure = structure.SparseTensorStructure(
        dataset_ops.get_legacy_output_types(input_dataset),
        tensor_shape.vector(None).concatenate(self._row_shape))

    variant_tensor = ged_ops.experimental_dense_to_sparse_batch_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._batch_size,
        row_shape=convert.partial_shape_to_tensor(self._row_shape),
        **dataset_ops.flat_structure(self))
    super(_DenseToSparseBatchDataset, self).__init__(input_dataset,
                                                     variant_tensor)

  @property
  def _element_structure(self):
    return self._structure


class _MapAndBatchDataset(dataset_ops.UnaryDataset):
  """A `Dataset` that maps a function over a batch of elements."""

  def __init__(self, input_dataset, map_func, batch_size, num_parallel_calls,
               drop_remainder, use_legacy_function=False):
    """See `Dataset.map()` for details."""
    self._input_dataset = input_dataset

    self._map_func = dataset_ops.StructuredFunctionWrapper(
        map_func,
        "tf.data.experimental.map_and_batch()",
        dataset=input_dataset,
        use_legacy_function=use_legacy_function)
    self._batch_size_t = ops.convert_to_tensor(
        batch_size, dtype=dtypes.int64, name="batch_size")
    self._num_parallel_calls_t = ops.convert_to_tensor(
        num_parallel_calls, dtype=dtypes.int64, name="num_parallel_calls")
    self._drop_remainder_t = ops.convert_to_tensor(
        drop_remainder, dtype=dtypes.bool, name="drop_remainder")

    constant_drop_remainder = tensor_util.constant_value(self._drop_remainder_t)
    if constant_drop_remainder:
      # NOTE(mrry): `constant_drop_remainder` may be `None` (unknown statically)
      # or `False` (explicitly retaining the remainder).
      self._structure = self._map_func.output_structure._batch(  # pylint: disable=protected-access
          tensor_util.constant_value(self._batch_size_t))
    else:
      self._structure = self._map_func.output_structure._batch(None)  # pylint: disable=protected-access
    variant_tensor = ged_ops.experimental_map_and_batch_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._map_func.function.captured_inputs,
        f=self._map_func.function,
        batch_size=self._batch_size_t,
        num_parallel_calls=self._num_parallel_calls_t,
        drop_remainder=self._drop_remainder_t,
        preserve_cardinality=True,
        **dataset_ops.flat_structure(self))
    super(_MapAndBatchDataset, self).__init__(input_dataset, variant_tensor)

  def _functions(self):
    return [self._map_func]

  @property
  def _element_structure(self):
    return self._structure


class _RestructuredDataset(dataset_ops.UnaryDataset):
  """An internal helper for changing the structure and shape of a dataset."""

  def __init__(self,
               dataset,
               output_types,
               output_shapes=None,
               output_classes=None,
               allow_unsafe_cast=False):
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
      output_classes: (Optional.) A nested structure of class types. If omitted,
        the class types will be inherited from `dataset`.
      allow_unsafe_cast: (Optional.) If `True`, the caller may switch the
        reported output types and shapes of the restructured dataset, e.g. to
        switch a sparse tensor represented as `tf.variant` to its user-visible
        type and shape.

    Raises:
      ValueError: If either `output_types` or `output_shapes` is not compatible
        with the structure of `dataset`.
    """
    self._input_dataset = dataset

    input_types = dataset_ops.get_legacy_output_types(dataset)
    if not allow_unsafe_cast:
      # Validate that the types are compatible.
      output_types = nest.map_structure(dtypes.as_dtype, output_types)
      flat_original_types = nest.flatten(input_types)
      flat_new_types = nest.flatten(output_types)
      if flat_original_types != flat_new_types:
        raise ValueError(
            "Dataset with output types %r cannot be restructured to have "
            "output types %r" %
            (dataset_ops.get_legacy_output_types(dataset), output_types))

    input_shapes = dataset_ops.get_legacy_output_shapes(dataset)
    if output_shapes is None:
      # Inherit shapes from the original `dataset`.
      output_shapes = nest.pack_sequence_as(
          output_types, nest.flatten(input_shapes))
    else:
      if not allow_unsafe_cast:
        # Validate that the shapes are compatible.
        nest.assert_same_structure(output_types, output_shapes)
        flat_original_shapes = nest.flatten(input_shapes)
        flat_new_shapes = nest.flatten_up_to(output_types, output_shapes)

        for original_shape, new_shape in zip(flat_original_shapes,
                                             flat_new_shapes):
          if not original_shape.is_compatible_with(new_shape):
            raise ValueError(
                "Dataset with output shapes %r cannot be restructured to have "
                "incompatible output shapes %r" % (input_shapes,
                                                   output_shapes))
      output_shapes = nest.map_structure_up_to(
          output_types, tensor_shape.as_shape, output_shapes)

    input_classes = dataset_ops.get_legacy_output_classes(dataset)
    if output_classes is None:
      # Inherit class types from the original `dataset`.
      output_classes = nest.pack_sequence_as(
          output_types, nest.flatten(input_classes))

    self._structure = structure.convert_legacy_structure(
        output_types, output_shapes, output_classes)
    variant_tensor = self._input_dataset._variant_tensor  # pylint: disable=protected-access
    super(_RestructuredDataset, self).__init__(dataset, variant_tensor)

  @property
  def _element_structure(self):
    return self._structure


class _UnbatchDataset(dataset_ops.UnaryDataset):
  """A dataset that splits the elements of its input into multiple elements."""

  def __init__(self, input_dataset):
    """See `unbatch()` for more details."""
    input_shapes = dataset_ops.get_legacy_output_shapes(input_dataset)
    flat_shapes = nest.flatten(input_shapes)
    if any(s.ndims == 0 for s in flat_shapes):
      raise ValueError("Cannot unbatch an input with scalar components.")
    known_batch_dim = tensor_shape.Dimension(None)
    for s in flat_shapes:
      try:
        known_batch_dim = known_batch_dim.merge_with(s[0])
      except ValueError:
        raise ValueError("Cannot unbatch an input whose components have "
                         "different batch sizes.")
    self._input_dataset = input_dataset

    self._structure = dataset_ops.get_structure(input_dataset)._unbatch()  # pylint: disable=protected-access

    variant_tensor = ged_ops.experimental_unbatch_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        **dataset_ops.flat_structure(self))
    super(_UnbatchDataset, self).__init__(input_dataset, variant_tensor)

  @property
  def _element_structure(self):
    return self._structure
