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

from tensorflow.contrib.framework import with_shape
from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.util import nest
from tensorflow.python.util import deprecation


@deprecation.deprecated(
    None, "Use `tf.data.experimental.dense_to_sparse_batch(...)`.")
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
      ([[0, 0], [0, 1], [0, 2], [0, 3]],
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
    `tf.data.Dataset.apply`.
  """
  return batching.dense_to_sparse_batch(batch_size, row_shape)


@deprecation.deprecated(None, "Use `tf.data.experimental.unbatch()`.")
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

  a.apply(tf.contrib.data.unbatch()) == {
      'a', 'b', 'c', 'a', 'b', 'a', 'b', 'c', 'd'}
  ```

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """
  return batching.unbatch()


@deprecation.deprecated(
    None, "Use `tf.data.Dataset.batch(..., drop_remainder=True)`.")
def batch_and_drop_remainder(batch_size):
  """A batching transformation that omits the final small batch (if present).

  Like `tf.data.Dataset.batch`, this transformation combines
  consecutive elements of this dataset into batches. However, if the batch
  size does not evenly divide the input dataset size, this transformation will
  drop the final smaller element.

  The following example illustrates the difference between this
  transformation and `Dataset.batch()`:

  ```python
  dataset = tf.data.Dataset.range(200)
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
    `tf.data.Dataset.apply`
  """

  def _apply_fn(dataset):
    """Function from `Dataset` to `Dataset` that applies the transformation."""
    return dataset.batch(batch_size, drop_remainder=True)

  return _apply_fn


@deprecation.deprecated(
    None, "Use `tf.data.Dataset.padded_batch(..., drop_remainder=True)`.")
def padded_batch_and_drop_remainder(batch_size,
                                    padded_shapes,
                                    padding_values=None):
  """A batching and padding transformation that omits the final small batch.

  Like `tf.data.Dataset.padded_batch`, this transformation combines
  consecutive elements of this dataset into batches. However, if the batch
  size does not evenly divide the input dataset size, this transformation will
  drop the final smaller element.

  See `tf.contrib.data.batch_and_drop_remainder` for more details.

  Args:
    batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
      consecutive elements of this dataset to combine in a single batch.
    padded_shapes: A nested structure of `tf.TensorShape` or
      `tf.int64` vector tensor-like objects. See
      `tf.data.Dataset.padded_batch` for details.
    padding_values: (Optional.) A nested structure of scalar-shaped
      `tf.Tensor`. See `tf.data.Dataset.padded_batch` for details.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`
  """

  def _apply_fn(dataset):
    """Function from `Dataset` to `Dataset` that applies the transformation."""
    return dataset.padded_batch(
        batch_size, padded_shapes=padded_shapes, padding_values=padding_values,
        drop_remainder=True)

  return _apply_fn


# TODO(b/116817045): Move this to `tf.data.experimental` when the `with_shape()`
# function is available in the core.
def assert_element_shape(expected_shapes):
  """Assert the shape of this `Dataset`.

  ```python
  shapes = [tf.TensorShape([16, 256]), tf.TensorShape([None, 2])]
  result = dataset.apply(tf.contrib.data.assert_element_shape(shapes))
  print(result.output_shapes)  # ==> "((16, 256), (<unknown>, 2))"
  ```

  If dataset shapes and expected_shape, are fully defined, assert they match.
  Otherwise, add assert op that will validate the shapes when tensors are
  evaluated, and set shapes on tensors, respectively.

  Note that unknown dimension in `expected_shapes` will be ignored.

  Args:
    expected_shapes: A nested structure of `tf.TensorShape` objects.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`
  """

  def _merge_output_shapes(original_shapes, expected_shapes):
    flat_original_shapes = nest.flatten(original_shapes)
    flat_new_shapes = nest.flatten_up_to(original_shapes, expected_shapes)
    flat_merged_output_shapes = [
        original_shape.merge_with(new_shape)
        for original_shape, new_shape in zip(flat_original_shapes,
                                             flat_new_shapes)]
    return nest.pack_sequence_as(original_shapes, flat_merged_output_shapes)

  def _check_shape(*elements):
    flatten_tensors = nest.flatten(elements)
    flatten_shapes = nest.flatten(expected_shapes)
    checked_tensors = [
        with_shape(shape, tensor) if shape else tensor  # Ignore unknown shape
        for shape, tensor in zip(flatten_shapes, flatten_tensors)
    ]
    return nest.pack_sequence_as(elements, checked_tensors)

  def _apply_fn(dataset):
    output_shapes = _merge_output_shapes(dataset.output_shapes,
                                         expected_shapes)
    # pylint: disable=protected-access
    return batching._RestructuredDataset(
        dataset.map(_check_shape),
        dataset.output_types,
        output_shapes=output_shapes,
        output_classes=dataset.output_classes)

  return _apply_fn


@deprecation.deprecated(None, "Use `tf.data.experimental.map_and_batch(...)`.")
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
        specified, `batch_size * num_parallel_batches` elements will be
        processed in parallel.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.

  Raises:
    ValueError: If both `num_parallel_batches` and `num_parallel_calls` are
      specified.
  """
  return batching.map_and_batch(map_func, batch_size, num_parallel_batches,
                                drop_remainder, num_parallel_calls)
