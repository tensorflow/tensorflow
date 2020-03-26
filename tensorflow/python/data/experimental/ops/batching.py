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
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


@tf_export("data.experimental.dense_to_ragged_batch")
def dense_to_ragged_batch(batch_size,
                          drop_remainder=False,
                          row_splits_dtype=dtypes.int64):
  """A transformation that batches ragged elements into `tf.RaggedTensor`s.

  This transformation combines multiple consecutive elements of the input
  dataset into a single element.

  Like `tf.data.Dataset.batch`, the components of the resulting element will
  have an additional outer dimension, which will be `batch_size` (or
  `N % batch_size` for the last element if `batch_size` does not divide the
  number of input elements `N` evenly and `drop_remainder` is `False`). If
  your program depends on the batches having the same outer dimension, you
  should set the `drop_remainder` argument to `True` to prevent the smaller
  batch from being produced.

  Unlike `tf.data.Dataset.batch`, the input elements to be batched may have
  different shapes, and each batch will be encoded as a `tf.RaggedTensor`.
  Example:

  >>> dataset = tf.data.Dataset.from_tensor_slices(np.arange(6))
  >>> dataset = dataset.map(lambda x: tf.range(x))
  >>> dataset = dataset.apply(
  ...     tf.data.experimental.dense_to_ragged_batch(batch_size=2))
  >>> for batch in dataset:
  ...   print(batch)
  <tf.RaggedTensor [[], [0]]>
  <tf.RaggedTensor [[0, 1], [0, 1, 2]]>
  <tf.RaggedTensor [[0, 1, 2, 3], [0, 1, 2, 3, 4]]>

  Args:
    batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
      consecutive elements of this dataset to combine in a single batch.
    drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
      whether the last batch should be dropped in the case it has fewer than
      `batch_size` elements; the default behavior is not to drop the smaller
      batch.
    row_splits_dtype: The dtype that should be used for the `row_splits` of any
      new ragged tensors.  Existing `tf.RaggedTensor` elements do not have their
      row_splits dtype changed.

  Returns:
    Dataset: A `Dataset`.
  """

  def _apply_fn(dataset):
    ragged_dataset = _DenseToRaggedDataset(dataset, row_splits_dtype)
    return dataset_ops.BatchDataset(
        ragged_dataset, batch_size=batch_size, drop_remainder=drop_remainder)

  return _apply_fn


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
  followed by `batch`. This API is temporary and deprecated since input pipeline
  optimization now fuses consecutive `map` and `batch` operations automatically.

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


@deprecation.deprecated(None, "Use `tf.data.Dataset.unbatch()`.")
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

  a.unbatch() == {
      'a', 'b', 'c', 'a', 'b', 'a', 'b', 'c', 'd'}
  ```

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):
    return dataset.unbatch()

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
    self._element_spec = sparse_tensor.SparseTensorSpec(
        tensor_shape.TensorShape([None]).concatenate(self._row_shape),
        dataset_ops.get_legacy_output_types(input_dataset))

    variant_tensor = ged_ops.dense_to_sparse_batch_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._batch_size,
        row_shape=convert.partial_shape_to_tensor(self._row_shape),
        **self._flat_structure)
    super(_DenseToSparseBatchDataset, self).__init__(input_dataset,
                                                     variant_tensor)

  @property
  def element_spec(self):
    return self._element_spec


class _MapAndBatchDataset(dataset_ops.UnaryDataset):
  """A `Dataset` that maps a function over a batch of elements."""

  def __init__(self, input_dataset, map_func, batch_size, num_parallel_calls,
               drop_remainder, use_legacy_function=False):
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
    # pylint: disable=protected-access
    if constant_drop_remainder:
      # NOTE(mrry): `constant_drop_remainder` may be `None` (unknown statically)
      # or `False` (explicitly retaining the remainder).
      # pylint: disable=g-long-lambda
      self._element_spec = nest.map_structure(
          lambda component_spec: component_spec._batch(
              tensor_util.constant_value(self._batch_size_t)),
          self._map_func.output_structure)
    else:
      self._element_spec = nest.map_structure(
          lambda component_spec: component_spec._batch(None),
          self._map_func.output_structure)
    # pylint: enable=protected-access
    variant_tensor = ged_ops.map_and_batch_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._map_func.function.captured_inputs,
        f=self._map_func.function,
        batch_size=self._batch_size_t,
        num_parallel_calls=self._num_parallel_calls_t,
        drop_remainder=self._drop_remainder_t,
        preserve_cardinality=True,
        **self._flat_structure)
    super(_MapAndBatchDataset, self).__init__(input_dataset, variant_tensor)

  def _functions(self):
    return [self._map_func]

  @property
  def element_spec(self):
    return self._element_spec


class _DenseToRaggedDataset(dataset_ops.UnaryDataset):
  """A `Dataset` that encodes dense inputs as ragged (w/ ragged_rank=0).

  In particular:

  * Any tf.Tensor elements with rank>0 are encoded as ragged tensors with
    ragged_rank=0.  This allows tensors with varying shape to be batched
    together.
  * Any other elements are left as-is.
  """

  def __init__(self, input_dataset, row_splits_dtype):
    """Constructs a new _DenseToRaggedDataset.

    Args:
      input_dataset: The dataset whose tf.Tensor elements should be made ragged.
      row_splits_dtype: The dtype that should be used for the `row_splits` of
        any new ragged tensors.  Existing `tf.RaggedTensor` elements do *not*
        have their row_splits dtype changed.
    """

    # Replace each TensorSpec in the input dataset's structure with a
    # corresponding RaggedTensorSpec.
    def to_ragged_spec(spec):
      if isinstance(spec, tensor_spec.TensorSpec) and spec.shape.ndims != 0:
        return ragged_tensor.RaggedTensorSpec(
            shape=spec.shape,
            dtype=spec.dtype,
            ragged_rank=0,
            row_splits_dtype=row_splits_dtype)
      else:
        return spec

    self._structure = nest.map_structure(to_ragged_spec,
                                         input_dataset.element_spec)

    # Replace each tf.Tensor value in the input dataset with a variant-encoded
    # RaggedTensor.  Since we're updating the corresponding structure to be
    # a RaggedTensorSpec, this variant-encoded tensor will be decoded with
    # RaggedTensorSpec._from_tensor_list.
    def to_ragged_variant(value):
      if isinstance(value, ops.Tensor) and value.shape.ndims != 0:
        spec = to_ragged_spec(tensor_spec.TensorSpec.from_tensor(value))
        return spec._to_tensor_list(value)[0]  # pylint: disable=protected-access
      else:
        return value

    self._mapped_dataset = input_dataset.map(
        lambda value: nest.map_structure(to_ragged_variant, value))

    variant = self._mapped_dataset._variant_tensor  # pylint: disable=protected-access
    super(_DenseToRaggedDataset, self).__init__(input_dataset, variant)

  @property
  def element_spec(self):
    return self._structure
