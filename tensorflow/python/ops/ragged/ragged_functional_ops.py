# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Support for ragged tensors."""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops.ragged import ragged_config
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export


@tf_export("ragged.map_flat_values")
@dispatch.add_dispatch_support
def map_flat_values(op, *args, **kwargs):
  """Applies `op` to the `flat_values` of one or more RaggedTensors.

  Replaces any `RaggedTensor` in `args` or `kwargs` with its `flat_values`
  tensor (which collapses all ragged dimensions), and then calls `op`.  Returns
  a `RaggedTensor` that is constructed from the input `RaggedTensor`s'
  `nested_row_splits` and the value returned by the `op`.

  If the input arguments contain multiple `RaggedTensor`s, then they must have
  identical `nested_row_splits`.

  This operation is generally used to apply elementwise operations to each value
  in a `RaggedTensor`.

  Warning: `tf.ragged.map_flat_values` does *not* apply `op` to each row of a
  ragged tensor.  This difference is important for non-elementwise operations,
  such as `tf.reduce_sum`.  If you wish to apply a non-elementwise operation to
  each row of a ragged tensor, use `tf.map_fn` instead.  (You may need to
  specify an `output_signature` when using `tf.map_fn` with ragged tensors.)

  Examples:

  >>> rt = tf.ragged.constant([[1, 2, 3], [], [4, 5], [6]])
  >>> tf.ragged.map_flat_values(tf.ones_like, rt)
  <tf.RaggedTensor [[1, 1, 1], [], [1, 1], [1]]>
  >>> tf.ragged.map_flat_values(tf.multiply, rt, rt)
  <tf.RaggedTensor [[1, 4, 9], [], [16, 25], [36]]>
  >>> tf.ragged.map_flat_values(tf.add, rt, 5)
  <tf.RaggedTensor [[6, 7, 8], [], [9, 10], [11]]>

  Example with a non-elementwise operation (note that `map_flat_values` and
  `map_fn` return different results):

  >>> rt = tf.ragged.constant([[1.0, 3.0], [], [3.0, 6.0, 3.0]])
  >>> def normalized(x):
  ...   return x / tf.reduce_sum(x)
  >>> tf.ragged.map_flat_values(normalized, rt)
  <tf.RaggedTensor [[0.0625, 0.1875], [], [0.1875, 0.375, 0.1875]]>
  >>> tf.map_fn(normalized, rt)
  <tf.RaggedTensor [[0.25, 0.75], [], [0.25, 0.5, 0.25]]>

  Args:
    op: The operation that should be applied to the RaggedTensor `flat_values`.
      `op` is typically an element-wise operation (such as math_ops.add), but
      any operation that preserves the size of the outermost dimension can be
      used.  I.e., `shape[0]` of the value returned by `op` must match
      `shape[0]` of the `RaggedTensor`s' `flat_values` tensors.
    *args: Arguments for `op`.
    **kwargs: Keyword arguments for `op`.

  Returns:
    A `RaggedTensor` whose `ragged_rank` matches the `ragged_rank` of all
    input `RaggedTensor`s.
  Raises:
    ValueError: If args contains no `RaggedTensors`, or if the `nested_splits`
      of the input `RaggedTensor`s are not identical.
  """
  # Replace RaggedTensors with their values; and collect the partitions tensors
  # from each RaggedTensor.
  partition_lists = []
  flat_values_nrows = []
  inner_args = _replace_ragged_with_flat_values(args, partition_lists,
                                                flat_values_nrows)
  inner_kwargs = _replace_ragged_with_flat_values(kwargs, partition_lists,
                                                  flat_values_nrows)
  if not partition_lists:
    return op(*args, **kwargs)

  # If we can statically determine that the inputs are incompatible, then raise
  # an error.  (We can't guarantee full compatibility statically, so we need to
  # perform some runtime checks too; but this allows us to fail sooner in some
  # cases.)
  if flat_values_nrows:
    flat_values_nrows = set(flat_values_nrows)
    if len(flat_values_nrows) != 1:
      raise ValueError("Input RaggedTensors' flat_values must all have the "
                       "same outer-dimension size.  Got sizes: %s" %
                       flat_values_nrows)
    flat_values_nrows = flat_values_nrows.pop()  # Get the single element
  else:
    flat_values_nrows = None

  partition_dtypes = set(p[0].dtype for p in partition_lists)
  if len(partition_dtypes) > 1:
    if not ragged_config.auto_cast_partition_dtype():
      raise ValueError("Input RaggedTensors have mismatched row partition "
                       "dtypes; use RaggedTensor.with_row_splits_dtype() to "
                       "convert them to compatible dtypes.")

    partition_lists = [
        [p.with_dtype(dtypes.int64)
         for p in partition_list]  # pylint: disable=g-complex-comprehension
        for partition_list in partition_lists
    ]

  # Delegate to `op`
  op_output = op(*inner_args, **inner_kwargs)
  # Check that the result has the expected shape (if known).
  if flat_values_nrows is not None:
    if not op_output.shape[:1].is_compatible_with([flat_values_nrows]):
      raise ValueError(
          "tf.ragged.map_flat_values requires that the output of `op` have "
          "the same outer-dimension size as flat_values of any ragged "
          "inputs. (output shape: %s; expected outer dimension size: %s)" %
          (op_output.shape, flat_values_nrows))
  # Compose the result from the transformed values and the partitions.
  return ragged_tensor.RaggedTensor._from_nested_row_partitions(  # pylint: disable=protected-access
      op_output,
      _merge_partition_lists(partition_lists),
      validate=False)


def _replace_ragged_with_flat_values(value, partition_lists, flat_values_nrows):
  """Replace RaggedTensors with their flat_values, and record their partitions.

  Returns a copy of `value`, with any nested `RaggedTensor`s replaced by their
  `flat_values` tensor.  Looks inside lists, tuples, and dicts.

  Appends each `RaggedTensor`'s `RowPartition`s to `partition_lists`.

  Args:
    value: The value that should be transformed by replacing `RaggedTensors`.
    partition_lists: An output parameter used to record the row partitions
      for any `RaggedTensors` that were replaced.
    flat_values_nrows: An output parameter used to record the outer dimension
      size for each replacement `flat_values` (when known).  Contains a list of
      int.

  Returns:
    A copy of `value` with nested `RaggedTensors` replaced by their `values`.
  """
  # Base case
  if ragged_tensor.is_ragged(value):
    value = ragged_tensor.convert_to_tensor_or_ragged_tensor(value)
    partition_lists.append(value._nested_row_partitions)  # pylint: disable=protected-access
    nrows = tensor_shape.dimension_at_index(value.flat_values.shape, 0).value
    if nrows is not None:
      flat_values_nrows.append(nrows)
    return value.flat_values

  # Recursion cases
  def recurse(v):
    return _replace_ragged_with_flat_values(v, partition_lists,
                                            flat_values_nrows)

  if isinstance(value, list):
    return [recurse(v) for v in value]
  elif isinstance(value, tuple):
    return tuple(recurse(v) for v in value)
  elif isinstance(value, dict):
    return dict((k, recurse(v)) for (k, v) in value.items())
  else:
    return value


def _merge_partition_lists(partition_lists):
  """Merges the given list of lists of RowPartitions.

  Args:
    partition_lists: A list of lists of RowPartition.

  Returns:
    A list of RowPartitions, where `result[i]` is formed by merging
    `partition_lists[j][i]` for all `j`, using
    `RowPartition._merge_precomputed_encodings`.
  """
  dst = list(partition_lists[0])
  for src in partition_lists[1:]:
    if len(src) != len(dst):
      raise ValueError("All ragged inputs must have the same ragged_rank.")
    for i in range(len(dst)):
      # pylint: disable=protected-access
      dst[i] = dst[i]._merge_precomputed_encodings(src[i])
  return dst
