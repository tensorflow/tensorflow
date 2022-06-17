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
"""Classes for storing ragged tensors and their values."""

import functools
import operator

import typing
import numpy as np

from tensorflow.python import tf2
from tensorflow.python.client import session
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_ragged_conversion_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_config
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.types import core as core_types
from tensorflow.python.types import internal as internal_types
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls

# pylint: disable=protected-access
_convert_row_partition = RowPartition._convert_row_partition
# pylint: enable=protected-access

#===============================================================================
# RaggedTensor
#===============================================================================


@tf_export("RaggedTensor")
class RaggedTensor(composite_tensor.CompositeTensor,
                   internal_types.NativeObject):
  """Represents a ragged tensor.

  A `RaggedTensor` is a tensor with one or more *ragged dimensions*, which are
  dimensions whose slices may have different lengths.  For example, the inner
  (column) dimension of `rt=[[3, 1, 4, 1], [], [5, 9, 2], [6], []]` is ragged,
  since the column slices (`rt[0, :]`, ..., `rt[4, :]`) have different lengths.
  Dimensions whose slices all have the same length are called *uniform
  dimensions*.  The outermost dimension of a `RaggedTensor` is always uniform,
  since it consists of a single slice (and so there is no possibility for
  differing slice lengths).

  The total number of dimensions in a `RaggedTensor` is called its *rank*,
  and the number of ragged dimensions in a `RaggedTensor` is called its
  *ragged-rank*.  A `RaggedTensor`'s ragged-rank is fixed at graph creation
  time: it can't depend on the runtime values of `Tensor`s, and can't vary
  dynamically for different session runs.

  Note that the `__init__` constructor is private. Please use one of the
  following methods to construct a `RaggedTensor`:

  * `tf.RaggedTensor.from_row_lengths`
  * `tf.RaggedTensor.from_value_rowids`
  * `tf.RaggedTensor.from_row_splits`
  * `tf.RaggedTensor.from_row_starts`
  * `tf.RaggedTensor.from_row_limits`
  * `tf.RaggedTensor.from_nested_row_splits`
  * `tf.RaggedTensor.from_nested_row_lengths`
  * `tf.RaggedTensor.from_nested_value_rowids`

  ### Potentially Ragged Tensors

  Many ops support both `Tensor`s and `RaggedTensor`s
  (see [tf.ragged](https://www.tensorflow.org/api_docs/python/tf/ragged) for a
  full listing). The term "potentially ragged tensor" may be used to refer to a
  tensor that might be either a `Tensor` or a `RaggedTensor`.  The ragged-rank
  of a `Tensor` is zero.

  ### Documenting RaggedTensor Shapes

  When documenting the shape of a RaggedTensor, ragged dimensions can be
  indicated by enclosing them in parentheses.  For example, the shape of
  a 3-D `RaggedTensor` that stores the fixed-size word embedding for each
  word in a sentence, for each sentence in a batch, could be written as
  `[num_sentences, (num_words), embedding_size]`.  The parentheses around
  `(num_words)` indicate that dimension is ragged, and that the length
  of each element list in that dimension may vary for each item.

  ### Component Tensors

  Internally, a `RaggedTensor` consists of a concatenated list of values that
  are partitioned into variable-length rows.  In particular, each `RaggedTensor`
  consists of:

    * A `values` tensor, which concatenates the variable-length rows into a
      flattened list.  For example, the `values` tensor for
      `[[3, 1, 4, 1], [], [5, 9, 2], [6], []]` is `[3, 1, 4, 1, 5, 9, 2, 6]`.

    * A `row_splits` vector, which indicates how those flattened values are
      divided into rows.  In particular, the values for row `rt[i]` are stored
      in the slice `rt.values[rt.row_splits[i]:rt.row_splits[i+1]]`.

  Example:

  >>> print(tf.RaggedTensor.from_row_splits(
  ...       values=[3, 1, 4, 1, 5, 9, 2, 6],
  ...       row_splits=[0, 4, 4, 7, 8, 8]))
  <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2], [6], []]>

  ### Alternative Row-Partitioning Schemes

  In addition to `row_splits`, ragged tensors provide support for five other
  row-partitioning schemes:

    * `row_lengths`: a vector with shape `[nrows]`, which specifies the length
      of each row.

    * `value_rowids` and `nrows`: `value_rowids` is a vector with shape
      `[nvals]`, corresponding one-to-one with `values`, which specifies
      each value's row index.  In particular, the row `rt[row]` consists of the
      values `rt.values[j]` where `value_rowids[j]==row`.  `nrows` is an
      integer scalar that specifies the number of rows in the
      `RaggedTensor`. (`nrows` is used to indicate trailing empty rows.)

    * `row_starts`: a vector with shape `[nrows]`, which specifies the start
      offset of each row.  Equivalent to `row_splits[:-1]`.

    * `row_limits`: a vector with shape `[nrows]`, which specifies the stop
      offset of each row.  Equivalent to `row_splits[1:]`.

    * `uniform_row_length`: A scalar tensor, specifying the length of every
      row.  This row-partitioning scheme may only be used if all rows have
      the same length.

  Example: The following ragged tensors are equivalent, and all represent the
  nested list `[[3, 1, 4, 1], [], [5, 9, 2], [6], []]`.

  >>> values = [3, 1, 4, 1, 5, 9, 2, 6]
  >>> RaggedTensor.from_row_splits(values, row_splits=[0, 4, 4, 7, 8, 8])
  <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2], [6], []]>
  >>> RaggedTensor.from_row_lengths(values, row_lengths=[4, 0, 3, 1, 0])
  <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2], [6], []]>
  >>> RaggedTensor.from_value_rowids(
  ...     values, value_rowids=[0, 0, 0, 0, 2, 2, 2, 3], nrows=5)
  <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2], [6], []]>
  >>> RaggedTensor.from_row_starts(values, row_starts=[0, 4, 4, 7, 8])
  <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2], [6], []]>
  >>> RaggedTensor.from_row_limits(values, row_limits=[4, 4, 7, 8, 8])
  <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2], [6], []]>
  >>> RaggedTensor.from_uniform_row_length(values, uniform_row_length=2)
  <tf.RaggedTensor [[3, 1], [4, 1], [5, 9], [2, 6]]>

  ### Multiple Ragged Dimensions

  `RaggedTensor`s with multiple ragged dimensions can be defined by using
  a nested `RaggedTensor` for the `values` tensor.  Each nested `RaggedTensor`
  adds a single ragged dimension.

  >>> inner_rt = RaggedTensor.from_row_splits(  # =rt1 from above
  ...     values=[3, 1, 4, 1, 5, 9, 2, 6], row_splits=[0, 4, 4, 7, 8, 8])
  >>> outer_rt = RaggedTensor.from_row_splits(
  ...     values=inner_rt, row_splits=[0, 3, 3, 5])
  >>> print(outer_rt.to_list())
  [[[3, 1, 4, 1], [], [5, 9, 2]], [], [[6], []]]
  >>> print(outer_rt.ragged_rank)
  2

  The factory function `RaggedTensor.from_nested_row_splits` may be used to
  construct a `RaggedTensor` with multiple ragged dimensions directly, by
  providing a list of `row_splits` tensors:

  >>> RaggedTensor.from_nested_row_splits(
  ...     flat_values=[3, 1, 4, 1, 5, 9, 2, 6],
  ...     nested_row_splits=([0, 3, 3, 5], [0, 4, 4, 7, 8, 8])).to_list()
  [[[3, 1, 4, 1], [], [5, 9, 2]], [], [[6], []]]

  ### Uniform Inner Dimensions

  `RaggedTensor`s with uniform inner dimensions can be defined
  by using a multidimensional `Tensor` for `values`.

  >>> rt = RaggedTensor.from_row_splits(values=tf.ones([5, 3], tf.int32),
  ...                                   row_splits=[0, 2, 5])
  >>> print(rt.to_list())
  [[[1, 1, 1], [1, 1, 1]],
   [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]
  >>> print(rt.shape)
  (2, None, 3)

  ### Uniform Outer Dimensions

  `RaggedTensor`s with uniform outer dimensions can be defined by using
  one or more `RaggedTensor` with a `uniform_row_length` row-partitioning
  tensor.  For example, a `RaggedTensor` with shape `[2, 2, None]` can be
  constructed with this method from a `RaggedTensor` values with shape
  `[4, None]`:

  >>> values = tf.ragged.constant([[1, 2, 3], [4], [5, 6], [7, 8, 9, 10]])
  >>> print(values.shape)
  (4, None)
  >>> rt6 = tf.RaggedTensor.from_uniform_row_length(values, 2)
  >>> print(rt6)
  <tf.RaggedTensor [[[1, 2, 3], [4]], [[5, 6], [7, 8, 9, 10]]]>
  >>> print(rt6.shape)
  (2, 2, None)

  Note that `rt6` only contains one ragged dimension (the innermost
  dimension). In contrast, if `from_row_splits` is used to construct a similar
  `RaggedTensor`, then that `RaggedTensor` will have two ragged dimensions:

  >>> rt7 = tf.RaggedTensor.from_row_splits(values, [0, 2, 4])
  >>> print(rt7.shape)
  (2, None, None)

  Uniform and ragged outer dimensions may be interleaved, meaning that a
  tensor with any combination of ragged and uniform dimensions may be created.
  For example, a RaggedTensor `t4` with shape `[3, None, 4, 8, None, 2]` could
  be constructed as follows:

  ```python
  t0 = tf.zeros([1000, 2])                           # Shape:         [1000, 2]
  t1 = RaggedTensor.from_row_lengths(t0, [...])      #           [160, None, 2]
  t2 = RaggedTensor.from_uniform_row_length(t1, 8)   #         [20, 8, None, 2]
  t3 = RaggedTensor.from_uniform_row_length(t2, 4)   #       [5, 4, 8, None, 2]
  t4 = RaggedTensor.from_row_lengths(t3, [...])      # [3, None, 4, 8, None, 2]
  ```

  """

  #=============================================================================
  # Constructor (private)
  #=============================================================================
  @doc_controls.do_not_generate_docs
  def __init__(self, values, row_partition, internal=False):
    """Creates a `RaggedTensor` with a specified partitioning for `values`.

    This constructor is private -- please use one of the following ops to
    build `RaggedTensor`s:

      * `tf.RaggedTensor.from_row_lengths`
      * `tf.RaggedTensor.from_value_rowids`
      * `tf.RaggedTensor.from_row_splits`
      * `tf.RaggedTensor.from_row_starts`
      * `tf.RaggedTensor.from_row_limits`
      * `tf.RaggedTensor.from_nested_row_splits`
      * `tf.RaggedTensor.from_nested_row_lengths`
      * `tf.RaggedTensor.from_nested_value_rowids`

    Args:
      values: A potentially ragged tensor of any dtype and shape `[nvals, ...]`.
      row_partition: A `RowPartition` object, representing the arrangement of
        the lists at the top level.
      internal: True if the constructor is being called by one of the factory
        methods.  If false, an exception will be raised.

    Raises:
      ValueError: If internal = False. Note that this method is intended only
                 for internal use.
      TypeError: If values is not a `RaggedTensor` or `Tensor`, or
                 row_partition is not a `RowPartition`.
    """

    if not internal:
      raise ValueError("RaggedTensor constructor is private; please use one "
                       "of the factory methods instead (e.g., "
                       "RaggedTensor.from_row_lengths())")
    _assert_is_supported_ragged_values_type(values)
    if not isinstance(row_partition, RowPartition):
      raise TypeError(f"Argument `row_partition` must be a RowPartition. "
                      f"Received {row_partition}.")

    # Validate shapes.
    values.shape.with_rank_at_least(1)
    if isinstance(values, RaggedTensor):
      # pylint: disable=protected-access
      assert row_partition.dtype == values._row_partition.dtype

    self._values = values
    self._row_partition = row_partition

  #=============================================================================
  # Factory Methods
  #=============================================================================

  @classmethod
  def _from_row_partition(cls, values, row_partition, validate=True):
    """Creates a `RaggedTensor` with a row partition.

    This is used as a way for RaggedTensors to share row partitions.

    The outer dimension of values must be equal to `partition.nvals()`.

    Args:
      values: A potentially ragged tensor.
      row_partition: a `RowPartition`: can be shared between tensors.
      validate: If true, then use assertions to check that the arguments form a
        valid `RaggedTensor`.

    Returns:
      A `RaggedTensor`.  `result.rank = values.rank + 1`.
      `result.ragged_rank = values.ragged_rank + 1`.

    Raises:
      ValueError: If partition.nvals() != _nrows(values)
    """
    if not isinstance(row_partition, RowPartition):
      raise TypeError(f"Argument `row_partition` must be a RowPartition. "
                      f"Received {row_partition}.")
    if not isinstance(validate, bool):
      raise TypeError(f"Argument `validate` must have type bool. "
                      f"Received {validate}.")
    values, row_partition = cls._convert_values_and_partition(
        values, row_partition, "partition")
    if row_partition._has_precomputed_value_rowids():  # pylint: disable=protected-access
      value_rowids_shape = row_partition.value_rowids().shape
      values.shape[:1].assert_is_compatible_with(value_rowids_shape)
    if validate:
      msg = "Arguments to _from_row_partition do not form a valid RaggedTensor"
      nvals = _nrows(values, row_partition.dtype)
      checks = [
          check_ops.assert_equal(
              math_ops.cast(row_partition.nvals(), row_partition.dtype),
              nvals,
              message=msg),
      ]
      if not isinstance(values, RaggedTensor):
        checks.append(check_ops.assert_rank_at_least(values, 1))
      row_partition = row_partition._with_dependencies(checks)  # pylint: disable=protected-access
    return cls(values=values, internal=True, row_partition=row_partition)

  @classmethod
  @dispatch.add_dispatch_support
  def from_value_rowids(cls,
                        values,
                        value_rowids,
                        nrows=None,
                        name=None,
                        validate=True):
    """Creates a `RaggedTensor` with rows partitioned by `value_rowids`.

    The returned `RaggedTensor` corresponds with the python list defined by:

    ```python
    result = [[values[i] for i in range(len(values)) if value_rowids[i] == row]
              for row in range(nrows)]
    ```

    Args:
      values: A potentially ragged tensor with shape `[nvals, ...]`.
      value_rowids: A 1-D integer tensor with shape `[nvals]`, which corresponds
        one-to-one with `values`, and specifies each value's row index.  Must be
        nonnegative, and must be sorted in ascending order.
      nrows: An integer scalar specifying the number of rows.  This should be
        specified if the `RaggedTensor` may containing empty training rows. Must
        be greater than `value_rowids[-1]` (or zero if `value_rowids` is empty).
        Defaults to `value_rowids[-1] + 1` (or zero if `value_rowids` is empty).
      name: A name prefix for the RaggedTensor (optional).
      validate: If true, then use assertions to check that the arguments form
        a valid `RaggedTensor`.  Note: these assertions incur a runtime cost,
          since they must be checked for each tensor value.

    Returns:
      A `RaggedTensor`.  `result.rank = values.rank + 1`.
      `result.ragged_rank = values.ragged_rank + 1`.

    Raises:
      ValueError: If `nrows` is incompatible with `value_rowids`.

    #### Example:

    >>> print(tf.RaggedTensor.from_value_rowids(
    ...     values=[3, 1, 4, 1, 5, 9, 2, 6],
    ...     value_rowids=[0, 0, 0, 0, 2, 2, 2, 3],
    ...     nrows=5))
    <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2], [6], []]>

    """
    if not isinstance(validate, bool):
      raise TypeError(f"Argument `validate` must have type bool. "
                      f"Received {validate}.")

    with ops.name_scope(name, "RaggedFromValueRowIds",
                        [values, value_rowids, nrows]):
      row_partition = RowPartition.from_value_rowids(
          value_rowids=value_rowids,
          nrows=nrows,
          validate=validate,
          dtype_hint=_get_optional_partition_dtype(values))
      return cls._from_row_partition(values, row_partition, validate=validate)

  @classmethod
  @dispatch.add_dispatch_support
  def from_row_splits(cls, values, row_splits, name=None, validate=True):
    """Creates a `RaggedTensor` with rows partitioned by `row_splits`.

    The returned `RaggedTensor` corresponds with the python list defined by:

    ```python
    result = [values[row_splits[i]:row_splits[i + 1]]
              for i in range(len(row_splits) - 1)]
    ```

    Args:
      values: A potentially ragged tensor with shape `[nvals, ...]`.
      row_splits: A 1-D integer tensor with shape `[nrows+1]`.  Must not be
        empty, and must be sorted in ascending order.  `row_splits[0]` must be
        zero and `row_splits[-1]` must be `nvals`.
      name: A name prefix for the RaggedTensor (optional).
      validate: If true, then use assertions to check that the arguments form
        a valid `RaggedTensor`.  Note: these assertions incur a runtime cost,
          since they must be checked for each tensor value.

    Returns:
      A `RaggedTensor`.  `result.rank = values.rank + 1`.
      `result.ragged_rank = values.ragged_rank + 1`.

    Raises:
      ValueError: If `row_splits` is an empty list.

    #### Example:

    >>> print(tf.RaggedTensor.from_row_splits(
    ...     values=[3, 1, 4, 1, 5, 9, 2, 6],
    ...     row_splits=[0, 4, 4, 7, 8, 8]))
    <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2], [6], []]>

    """
    if not isinstance(validate, bool):
      raise TypeError(f"Argument `validate` must have type bool. "
                      f"Received {validate}.")

    with ops.name_scope(name, "RaggedFromRowSplits", [values, row_splits]):
      row_partition = RowPartition.from_row_splits(
          row_splits=row_splits,
          validate=validate,
          dtype_hint=_get_optional_partition_dtype(values))
      return cls._from_row_partition(values, row_partition, validate=validate)

  @classmethod
  @dispatch.add_dispatch_support
  def from_row_lengths(cls, values, row_lengths, name=None, validate=True):
    """Creates a `RaggedTensor` with rows partitioned by `row_lengths`.

    The returned `RaggedTensor` corresponds with the python list defined by:

    ```python
    result = [[values.pop(0) for i in range(length)]
              for length in row_lengths]
    ```

    Args:
      values: A potentially ragged tensor with shape `[nvals, ...]`.
      row_lengths: A 1-D integer tensor with shape `[nrows]`.  Must be
        nonnegative.  `sum(row_lengths)` must be `nvals`.
      name: A name prefix for the RaggedTensor (optional).
      validate: If true, then use assertions to check that the arguments form
        a valid `RaggedTensor`.  Note: these assertions incur a runtime cost,
          since they must be checked for each tensor value.

    Returns:
      A `RaggedTensor`.  `result.rank = values.rank + 1`.
      `result.ragged_rank = values.ragged_rank + 1`.

    #### Example:

    >>> print(tf.RaggedTensor.from_row_lengths(
    ...     values=[3, 1, 4, 1, 5, 9, 2, 6],
    ...     row_lengths=[4, 0, 3, 1, 0]))
    <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2], [6], []]>

    """
    if not isinstance(validate, bool):
      raise TypeError(f"Argument `validate` must have type bool. "
                      f"Received {validate}.")

    with ops.name_scope(name, "RaggedFromRowLengths", [values, row_lengths]):
      row_partition = RowPartition.from_row_lengths(
          row_lengths=row_lengths,
          validate=validate,
          dtype_hint=_get_optional_partition_dtype(values))
      return cls._from_row_partition(values, row_partition, validate=validate)

  @classmethod
  @dispatch.add_dispatch_support
  def from_row_starts(cls, values, row_starts, name=None, validate=True):
    """Creates a `RaggedTensor` with rows partitioned by `row_starts`.

    Equivalent to: `from_row_splits(values, concat([row_starts, nvals]))`.

    Args:
      values: A potentially ragged tensor with shape `[nvals, ...]`.
      row_starts: A 1-D integer tensor with shape `[nrows]`.  Must be
        nonnegative and sorted in ascending order.  If `nrows>0`, then
        `row_starts[0]` must be zero.
      name: A name prefix for the RaggedTensor (optional).
      validate: If true, then use assertions to check that the arguments form
        a valid `RaggedTensor`.  Note: these assertions incur a runtime cost,
          since they must be checked for each tensor value.

    Returns:
      A `RaggedTensor`.  `result.rank = values.rank + 1`.
      `result.ragged_rank = values.ragged_rank + 1`.

    #### Example:

    >>> print(tf.RaggedTensor.from_row_starts(
    ...     values=[3, 1, 4, 1, 5, 9, 2, 6],
    ...     row_starts=[0, 4, 4, 7, 8]))
    <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2], [6], []]>

    """
    if not isinstance(validate, bool):
      raise TypeError(f"Argument `validate` must have type bool. "
                      f"Received {validate}.")
    with ops.name_scope(name, "RaggedFromRowStarts", [values, row_starts]):
      values = _convert_to_ragged_tensor_values(values)
      row_partition = RowPartition.from_row_starts(
          row_starts=row_starts,
          nvals=_nrows(values),
          validate=validate,
          dtype_hint=_get_optional_partition_dtype(values))
      return cls._from_row_partition(values, row_partition, validate=validate)

  @classmethod
  @dispatch.add_dispatch_support
  def from_row_limits(cls, values, row_limits, name=None, validate=True):
    """Creates a `RaggedTensor` with rows partitioned by `row_limits`.

    Equivalent to: `from_row_splits(values, concat([0, row_limits]))`.

    Args:
      values: A potentially ragged tensor with shape `[nvals, ...]`.
      row_limits: A 1-D integer tensor with shape `[nrows]`.  Must be sorted in
        ascending order.  If `nrows>0`, then `row_limits[-1]` must be `nvals`.
      name: A name prefix for the RaggedTensor (optional).
      validate: If true, then use assertions to check that the arguments form
        a valid `RaggedTensor`.  Note: these assertions incur a runtime cost,
          since they must be checked for each tensor value.

    Returns:
      A `RaggedTensor`.  `result.rank = values.rank + 1`.
      `result.ragged_rank = values.ragged_rank + 1`.

    #### Example:

    >>> print(tf.RaggedTensor.from_row_limits(
    ...     values=[3, 1, 4, 1, 5, 9, 2, 6],
    ...     row_limits=[4, 4, 7, 8, 8]))
    <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2], [6], []]>

    """
    if not isinstance(validate, bool):
      raise TypeError(f"Argument `validate` must have type bool. "
                      f"Received {validate}.")
    with ops.name_scope(name, "RaggedFromRowLimits", [values, row_limits]):
      values = _convert_to_ragged_tensor_values(values)
      row_partition = RowPartition.from_row_limits(
          row_limits=row_limits,
          validate=validate,
          dtype_hint=_get_optional_partition_dtype(values))
      return cls._from_row_partition(values, row_partition, validate=validate)

  @classmethod
  @dispatch.add_dispatch_support
  def from_uniform_row_length(cls,
                              values,
                              uniform_row_length,
                              nrows=None,
                              validate=True,
                              name=None):
    """Creates a `RaggedTensor` with rows partitioned by `uniform_row_length`.

    This method can be used to create `RaggedTensor`s with multiple uniform
    outer dimensions.  For example, a `RaggedTensor` with shape `[2, 2, None]`
    can be constructed with this method from a `RaggedTensor` values with shape
    `[4, None]`:

    >>> values = tf.ragged.constant([[1, 2, 3], [4], [5, 6], [7, 8, 9, 10]])
    >>> print(values.shape)
    (4, None)
    >>> rt1 = tf.RaggedTensor.from_uniform_row_length(values, 2)
    >>> print(rt1)
    <tf.RaggedTensor [[[1, 2, 3], [4]], [[5, 6], [7, 8, 9, 10]]]>
    >>> print(rt1.shape)
    (2, 2, None)

    Note that `rt1` only contains one ragged dimension (the innermost
    dimension). In contrast, if `from_row_splits` is used to construct a similar
    `RaggedTensor`, then that `RaggedTensor` will have two ragged dimensions:

    >>> rt2 = tf.RaggedTensor.from_row_splits(values, [0, 2, 4])
    >>> print(rt2.shape)
    (2, None, None)

    Args:
      values: A potentially ragged tensor with shape `[nvals, ...]`.
      uniform_row_length: A scalar integer tensor.  Must be nonnegative. The
        size of the outer axis of `values` must be evenly divisible by
        `uniform_row_length`.
      nrows: The number of rows in the constructed RaggedTensor.  If not
        specified, then it defaults to `nvals/uniform_row_length` (or `0` if
        `uniform_row_length==0`).  `nrows` only needs to be specified if
        `uniform_row_length` might be zero.  `uniform_row_length*nrows` must be
        `nvals`.
      validate: If true, then use assertions to check that the arguments form
        a valid `RaggedTensor`.  Note: these assertions incur a runtime cost,
          since they must be checked for each tensor value.
      name: A name prefix for the RaggedTensor (optional).

    Returns:
      A `RaggedTensor` that corresponds with the python list defined by:

      ```python
      result = [[values.pop(0) for i in range(uniform_row_length)]
                for _ in range(nrows)]
      ```

      `result.rank = values.rank + 1`.
      `result.ragged_rank = values.ragged_rank + 1`.
    """
    if not isinstance(validate, bool):
      raise TypeError(f"Argument `validate` must have type bool. "
                      f"Received {validate}.")
    with ops.name_scope(name, "RaggedFromUniformRowLength",
                        [values, uniform_row_length, nrows]):
      values = _convert_to_ragged_tensor_values(values)
      uniform_row_length = _convert_row_partition(
          uniform_row_length, "UniformRowLength",
          _get_optional_partition_dtype(values))
      nvals = _nvals_uniform_row_length(values, uniform_row_length)
      row_partition = RowPartition.from_uniform_row_length(
          uniform_row_length=uniform_row_length,
          nvals=nvals,
          nrows=nrows,
          validate=validate,
          dtype_hint=_get_optional_partition_dtype(values))
      return cls._from_row_partition(values, row_partition, validate=validate)

  @classmethod
  @dispatch.add_dispatch_support
  def from_nested_value_rowids(cls,
                               flat_values,
                               nested_value_rowids,
                               nested_nrows=None,
                               name=None,
                               validate=True):
    """Creates a `RaggedTensor` from a nested list of `value_rowids` tensors.

    Equivalent to:

    ```python
    result = flat_values
    for (rowids, nrows) in reversed(zip(nested_value_rowids, nested_nrows)):
      result = from_value_rowids(result, rowids, nrows)
    ```

    Args:
      flat_values: A potentially ragged tensor.
      nested_value_rowids: A list of 1-D integer tensors.  The `i`th tensor is
        used as the `value_rowids` for the `i`th ragged dimension.
      nested_nrows: A list of integer scalars.  The `i`th scalar is used as the
        `nrows` for the `i`th ragged dimension.
      name: A name prefix for the RaggedTensor (optional).
      validate: If true, then use assertions to check that the arguments form
        a valid `RaggedTensor`.  Note: these assertions incur a runtime cost,
          since they must be checked for each tensor value.

    Returns:
      A `RaggedTensor` (or `flat_values` if `nested_value_rowids` is empty).

    Raises:
      ValueError: If `len(nested_values_rowids) != len(nested_nrows)`.
    """
    if not isinstance(validate, bool):
      raise TypeError(f"Argument `validate` must have type bool. "
                      f"Received {validate}.")
    if isinstance(nested_value_rowids, ops.Tensor):
      raise TypeError(f"Argument `nested_value_rowids` must be a list of "
                      f"Tensors. Received {nested_value_rowids}.")
    if nested_nrows is None:
      nested_nrows = [None] * len(nested_value_rowids)
    else:
      if isinstance(nested_nrows, ops.Tensor):
        raise TypeError(f"Argument `nested_nrows` must be a list of "
                        f"Tensors. Received {nested_nrows}.")
      if len(nested_nrows) != len(nested_value_rowids):
        raise ValueError(
            f"Argument `nested_nrows` must have the same length as "
            f"argument `nested_value_rowids`. len(nested_nrows) = "
            f"{len(nested_nrows)} vs. len(nested_values_rowids) = "
            f"{len(nested_value_rowids)}.")

    with ops.name_scope(name, "RaggedFromNestedValueRowIds", [flat_values] +
                        list(nested_value_rowids) + list(nested_nrows)):
      result = flat_values
      for value_rowids, nrows in reversed(
          list(zip(nested_value_rowids, nested_nrows))):
        result = cls.from_value_rowids(
            result, value_rowids, nrows, validate=validate)
      return result

  @classmethod
  @dispatch.add_dispatch_support
  def from_nested_row_splits(cls,
                             flat_values,
                             nested_row_splits,
                             name=None,
                             validate=True):
    """Creates a `RaggedTensor` from a nested list of `row_splits` tensors.

    Equivalent to:

    ```python
    result = flat_values
    for row_splits in reversed(nested_row_splits):
      result = from_row_splits(result, row_splits)
    ```

    Args:
      flat_values: A potentially ragged tensor.
      nested_row_splits: A list of 1-D integer tensors.  The `i`th tensor is
        used as the `row_splits` for the `i`th ragged dimension.
      name: A name prefix for the RaggedTensor (optional).
      validate: If true, then use assertions to check that the arguments form
        a valid `RaggedTensor`.  Note: these assertions incur a runtime cost,
          since they must be checked for each tensor value.

    Returns:
      A `RaggedTensor` (or `flat_values` if `nested_row_splits` is empty).
    """
    if not isinstance(validate, bool):
      raise TypeError(f"Argument `validate` must have type bool. "
                      f"Received {validate}.")
    if isinstance(nested_row_splits, ops.Tensor):
      raise TypeError(f"Argument `nested_row_splits` must be a list of "
                      f"Tensors. Received {nested_row_splits}.")
    with ops.name_scope(name, "RaggedFromNestedRowSplits",
                        [flat_values] + list(nested_row_splits)):
      result = flat_values
      for splits in reversed(nested_row_splits):
        result = cls.from_row_splits(result, splits, validate=validate)
      return result

  @classmethod
  @dispatch.add_dispatch_support
  def from_nested_row_lengths(cls,
                              flat_values,
                              nested_row_lengths,
                              name=None,
                              validate=True):
    """Creates a `RaggedTensor` from a nested list of `row_lengths` tensors.

    Equivalent to:

    ```python
    result = flat_values
    for row_lengths in reversed(nested_row_lengths):
      result = from_row_lengths(result, row_lengths)
    ```

    Args:
      flat_values: A potentially ragged tensor.
      nested_row_lengths: A list of 1-D integer tensors.  The `i`th tensor is
        used as the `row_lengths` for the `i`th ragged dimension.
      name: A name prefix for the RaggedTensor (optional).
      validate: If true, then use assertions to check that the arguments form
        a valid `RaggedTensor`.  Note: these assertions incur a runtime cost,
          since they must be checked for each tensor value.

    Returns:
      A `RaggedTensor` (or `flat_values` if `nested_row_lengths` is empty).
    """
    if not isinstance(validate, bool):
      raise TypeError(f"Argument `validate` must have type bool. "
                      f"Received {validate}.")
    if isinstance(nested_row_lengths, ops.Tensor):
      raise TypeError(f"Argument `nested_row_lengths` must be a list of "
                      f"Tensors. Received {nested_row_lengths}.")
    with ops.name_scope(name, "RaggedFromNestedRowlengths",
                        [flat_values] + list(nested_row_lengths)):
      result = flat_values
      for lengths in reversed(nested_row_lengths):
        result = cls.from_row_lengths(result, lengths, validate=validate)
      return result

  @classmethod
  def _from_nested_row_partitions(cls,
                                  flat_values,
                                  nested_row_partitions,
                                  name=None,
                                  validate=True):
    """Creates a `RaggedTensor` from a nested list of row partitions.

    Equivalent to:

    ```python
    result = flat_values
    for row_partition in reversed(nested_row_partitions):
      result = _from_row_partition(result, row_partition)
    ```

    Args:
      flat_values: A potentially ragged tensor.
      nested_row_partitions: A list of row partitions.  The `i`th element is
        used as the row partition for the `i`th ragged dimension.
      name: A name prefix for the RaggedTensor (optional).
      validate: If true, then use assertions to check that the arguments form
        a valid `RaggedTensor`.  Note: these assertions incur a runtime cost,
          since they must be checked for each tensor value.

    Returns:
      A `RaggedTensor` (or `flat_values` if `nested_row_lengths` is empty).
    """
    if not isinstance(validate, bool):
      raise TypeError(f"Argument `validate` must have type bool. "
                      f"Received {validate}.")
    if isinstance(nested_row_partitions, RowPartition):
      raise TypeError(f"Argument `nested_row_partitions` must be a list of "
                      f"RowPartitions. Received {nested_row_partitions}.")
    if isinstance(nested_row_partitions, ops.Tensor):
      raise TypeError(f"Argument `nested_row_partitions` must be a list of "
                      f"RowPartitions. Received {nested_row_partitions}.")
    with ops.name_scope(name, "RaggedFromNestedRowPartitions",
                        [flat_values] + list(nested_row_partitions)):
      result = flat_values
      for partition in reversed(nested_row_partitions):
        result = cls._from_row_partition(result, partition, validate=validate)
      return result

  @classmethod
  def _convert_values_and_partition(cls, values, row_partition, name):
    """Converts `values` and `partition` to Tensors.

    If `values` is a `RaggedTensor`, then converts `values` and `partition`
    to have compatible row-partitioning dtypes.  In particular, if any of the
    row partitioning tensors are `int64`, then all of the other row
    partitioning tensors wil be cast to `int64` (if auto_cast_partition_dtype()
    is true) or an error will be raised (if auto_cast_partition_dtype() is
    false).

    Args:
      values: The `values` for the `RaggedTensor` being constructed.
      row_partition: A RowPartition object for the `RaggedTensor` being
        constructed.
      name: The name of the RowPartition object.

    Returns:
      A tuple (values, partition).
    """
    if not isinstance(row_partition, RowPartition):
      raise TypeError(f"Argument `row_partition` must be a RowPartition. "
                      f"Received {row_partition}.")
    if isinstance(values, RaggedTensor):
      # pylint: disable=protected-access
      if values._row_partition.dtype != row_partition.dtype:
        if not ragged_config.auto_cast_partition_dtype():
          # pylint: disable=protected-access
          # TODO(edloper): get rid of the `name` parameter.
          raise ValueError(
              f"Argument `row_partition` of RaggedTensor with name: {name} "
              f"must have same dtype as Argument `values`. "
              f"({row_partition.dtype} vs. {values._row_partition.dtype}).")
        values = values.with_row_splits_dtype(row_partition.dtype)
    else:
      values = _convert_to_ragged_tensor_values(values)

    return (values, row_partition)

  #=============================================================================
  # Accessors
  #=============================================================================

  @property
  def dtype(self):
    """The `DType` of values in this tensor."""
    return self._values.dtype

  @property
  def shape(self):
    """The statically known shape of this ragged tensor.

    Returns:
      A `TensorShape` containing the statically known shape of this ragged
      tensor.  Ragged dimensions have a size of `None`.

    Examples:

    >>> tf.ragged.constant([[0], [1, 2]]).shape
    TensorShape([2, None])

    >>> tf.ragged.constant([[[0, 1]], [[1, 2], [3, 4]]], ragged_rank=1).shape
    TensorShape([2, None, 2])

    """
    nrows = self._row_partition.static_nrows
    ncols = self._row_partition.static_uniform_row_length
    value_shape = self._values.shape[1:]
    return tensor_shape.TensorShape([nrows, ncols]).concatenate(value_shape)

  def get_shape(self):
    """The statically known shape of this ragged tensor.

    Returns:
      A `TensorShape` containing the statically known shape of this ragged
      tensor.  Ragged dimensions have a size of `None`.

    Alias for `shape` property.

    Examples:

    >>> tf.ragged.constant([[0], [1, 2]]).get_shape()
    TensorShape([2, None])

    >>> tf.ragged.constant(
    ...    [[[0, 1]], [[1, 2], [3, 4]]], ragged_rank=1).get_shape()
    TensorShape([2, None, 2])

    """
    return self.shape

  @property
  def ragged_rank(self):
    """The number of times the RaggedTensor's flat_values is partitioned.

    Examples:

    >>> values = tf.ragged.constant([[1, 2, 3], [4], [5, 6], [7, 8, 9, 10]])
    >>> values.ragged_rank
    1

    >>> rt = tf.RaggedTensor.from_uniform_row_length(values, 2)
    >>> rt.ragged_rank
    2

    Returns:
      A Python `int` indicating the number of times the underlying `flat_values`
      Tensor has been partitioned to add a new dimension.
      I.e., `tf.rank(rt) = tf.rank(rt.flat_values) + rt.ragged_rank`.
    """
    values_is_ragged = isinstance(self._values, RaggedTensor)
    return self._values.ragged_rank + 1 if values_is_ragged else 1

  @property
  def values(self):
    """The concatenated rows for this ragged tensor.

    `rt.values` is a potentially ragged tensor formed by flattening the two
    outermost dimensions of `rt` into a single dimension.

    `rt.values.shape = [nvals] + rt.shape[2:]` (where `nvals` is the
    number of items in the outer two dimensions of `rt`).

    `rt.ragged_rank = self.ragged_rank - 1`

    Returns:
      A potentially ragged tensor.

    #### Example:

    >>> rt = tf.ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
    >>> print(rt.values)
    tf.Tensor([3 1 4 1 5 9 2 6], shape=(8,), dtype=int32)

    """
    return self._values

  @property
  def _nested_row_partitions(self):
    """Returns the row partitions for this `RaggedTensor`."""
    partitions = [self._row_partition]
    rt_values = self.values
    while isinstance(rt_values, RaggedTensor):
      # pylint: disable=protected-access
      partitions.append(rt_values._row_partition)
      rt_values = rt_values.values
    return tuple(partitions)

  @property
  def row_splits(self):
    """The row-split indices for this ragged tensor's `values`.

    `rt.row_splits` specifies where the values for each row begin and end in
    `rt.values`.  In particular, the values for row `rt[i]` are stored in
    the slice `rt.values[rt.row_splits[i]:rt.row_splits[i+1]]`.

    Returns:
      A 1-D integer `Tensor` with shape `[self.nrows+1]`.
      The returned tensor is non-empty, and is sorted in ascending order.
      `self.row_splits[0]` is zero, and `self.row_splits[-1]` is equal to
      `self.values.shape[0]`.

    #### Example:

    >>> rt = tf.ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
    >>> print(rt.row_splits)  # indices of row splits in rt.values
    tf.Tensor([0 4 4 7 8 8], shape=(6,), dtype=int64)

    """
    return self._row_partition.row_splits()

  @property
  def uniform_row_length(self):
    """The length of each row in this ragged tensor, or None if rows are ragged.

    >>> rt1 = tf.ragged.constant([[1, 2, 3], [4], [5, 6], [7, 8, 9, 10]])
    >>> print(rt1.uniform_row_length)  # rows are ragged.
    None

    >>> rt2 = tf.RaggedTensor.from_uniform_row_length(
    ...     values=rt1, uniform_row_length=2)
    >>> print(rt2)
    <tf.RaggedTensor [[[1, 2, 3], [4]], [[5, 6], [7, 8, 9, 10]]]>
    >>> print(rt2.uniform_row_length)  # rows are not ragged (all have size 2).
    tf.Tensor(2, shape=(), dtype=int64)

    A RaggedTensor's rows are only considered to be uniform (i.e. non-ragged)
    if it can be determined statically (at graph construction time) that the
    rows all have the same length.

    Returns:
      A scalar integer `Tensor`, specifying the length of every row in this
      ragged tensor (for ragged tensors whose rows are uniform); or `None`
      (for ragged tensors whose rows are ragged).
    """
    return self._row_partition.uniform_row_length()

  @property
  def flat_values(self):
    """The innermost `values` tensor for this ragged tensor.

    Concretely, if `rt.values` is a `Tensor`, then `rt.flat_values` is
    `rt.values`; otherwise, `rt.flat_values` is `rt.values.flat_values`.

    Conceptually, `flat_values` is the tensor formed by flattening the
    outermost dimension and all of the ragged dimensions into a single
    dimension.

    `rt.flat_values.shape = [nvals] + rt.shape[rt.ragged_rank + 1:]`
    (where `nvals` is the number of items in the flattened dimensions).

    Returns:
      A `Tensor`.

    #### Example:

    >>> rt = tf.ragged.constant([[[3, 1, 4, 1], [], [5, 9, 2]], [], [[6], []]])
    >>> print(rt.flat_values)
    tf.Tensor([3 1 4 1 5 9 2 6], shape=(8,), dtype=int32)

    """
    rt_values = self.values
    while isinstance(rt_values, RaggedTensor):
      rt_values = rt_values.values
    return rt_values

  @property
  def nested_row_splits(self):
    """A tuple containing the row_splits for all ragged dimensions.

    `rt.nested_row_splits` is a tuple containing the `row_splits` tensors for
    all ragged dimensions in `rt`, ordered from outermost to innermost.  In
    particular, `rt.nested_row_splits = (rt.row_splits,) + value_splits` where:

        * `value_splits = ()` if `rt.values` is a `Tensor`.
        * `value_splits = rt.values.nested_row_splits` otherwise.

    Returns:
      A `tuple` of 1-D integer `Tensor`s.

    #### Example:

    >>> rt = tf.ragged.constant(
    ...     [[[[3, 1, 4, 1], [], [5, 9, 2]], [], [[6], []]]])
    >>> for i, splits in enumerate(rt.nested_row_splits):
    ...   print('Splits for dimension %d: %s' % (i+1, splits.numpy()))
    Splits for dimension 1: [0 3]
    Splits for dimension 2: [0 3 3 5]
    Splits for dimension 3: [0 4 4 7 8 8]

    """
    rt_nested_splits = [self.row_splits]
    rt_values = self.values
    while isinstance(rt_values, RaggedTensor):
      rt_nested_splits.append(rt_values.row_splits)
      rt_values = rt_values.values
    return tuple(rt_nested_splits)

  def value_rowids(self, name=None):
    """Returns the row indices for the `values` in this ragged tensor.

    `rt.value_rowids()` corresponds one-to-one with the outermost dimension of
    `rt.values`, and specifies the row containing each value.  In particular,
    the row `rt[row]` consists of the values `rt.values[j]` where
    `rt.value_rowids()[j] == row`.

    Args:
      name: A name prefix for the returned tensor (optional).

    Returns:
      A 1-D integer `Tensor` with shape `self.values.shape[:1]`.
      The returned tensor is nonnegative, and is sorted in ascending order.

    #### Example:

    >>> rt = tf.ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
    >>> print(rt.values)
    tf.Tensor([3 1 4 1 5 9 2 6], shape=(8,), dtype=int32)
    >>> print(rt.value_rowids())  # corresponds 1:1 with rt.values
    tf.Tensor([0 0 0 0 2 2 2 3], shape=(8,), dtype=int64)

    """
    with ops.name_scope(name, "RaggedValueRowIds", [self]):
      return self._row_partition.value_rowids()

  def nested_value_rowids(self, name=None):
    """Returns a tuple containing the value_rowids for all ragged dimensions.

    `rt.nested_value_rowids` is a tuple containing the `value_rowids` tensors
    for
    all ragged dimensions in `rt`, ordered from outermost to innermost.  In
    particular, `rt.nested_value_rowids = (rt.value_rowids(),) + value_ids`
    where:

    * `value_ids = ()` if `rt.values` is a `Tensor`.
    * `value_ids = rt.values.nested_value_rowids` otherwise.

    Args:
      name: A name prefix for the returned tensors (optional).

    Returns:
      A `tuple` of 1-D integer `Tensor`s.

    #### Example:

    >>> rt = tf.ragged.constant(
    ...     [[[[3, 1, 4, 1], [], [5, 9, 2]], [], [[6], []]]])
    >>> for i, ids in enumerate(rt.nested_value_rowids()):
    ...   print('row ids for dimension %d: %s' % (i+1, ids.numpy()))
    row ids for dimension 1: [0 0 0]
    row ids for dimension 2: [0 0 0 2 2]
    row ids for dimension 3: [0 0 0 0 2 2 2 3]

    """
    with ops.name_scope(name, "RaggedNestedValueRowIds", [self]):
      rt_nested_ids = [self.value_rowids()]
      rt_values = self.values
      while isinstance(rt_values, RaggedTensor):
        rt_nested_ids.append(rt_values.value_rowids())
        rt_values = rt_values.values
      return tuple(rt_nested_ids)

  def nrows(self, out_type=None, name=None):
    """Returns the number of rows in this ragged tensor.

    I.e., the size of the outermost dimension of the tensor.

    Args:
      out_type: `dtype` for the returned tensor.  Defaults to
        `self.row_splits.dtype`.
      name: A name prefix for the returned tensor (optional).

    Returns:
      A scalar `Tensor` with dtype `out_type`.

    #### Example:

    >>> rt = tf.ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
    >>> print(rt.nrows())  # rt has 5 rows.
    tf.Tensor(5, shape=(), dtype=int64)

    """
    with ops.name_scope(name, "RaggedNRows", [self]):
      if out_type is None:
        return self._row_partition.nrows()
      else:
        return math_ops.cast(self._row_partition.nrows(), dtype=out_type)

  def row_starts(self, name=None):
    """Returns the start indices for rows in this ragged tensor.

    These indices specify where the values for each row begin in
    `self.values`.  `rt.row_starts()` is equal to `rt.row_splits[:-1]`.

    Args:
      name: A name prefix for the returned tensor (optional).

    Returns:
      A 1-D integer Tensor with shape `[nrows]`.
      The returned tensor is nonnegative, and is sorted in ascending order.

    #### Example:

    >>> rt = tf.ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
    >>> print(rt.values)
    tf.Tensor([3 1 4 1 5 9 2 6], shape=(8,), dtype=int32)
    >>> print(rt.row_starts())  # indices of row starts in rt.values
    tf.Tensor([0 4 4 7 8], shape=(5,), dtype=int64)

    """
    with ops.name_scope(name, "RaggedRowStarts", [self]):
      return self._row_partition.row_starts()

  def row_limits(self, name=None):
    """Returns the limit indices for rows in this ragged tensor.

    These indices specify where the values for each row end in
    `self.values`.  `rt.row_limits(self)` is equal to `rt.row_splits[:-1]`.

    Args:
      name: A name prefix for the returned tensor (optional).

    Returns:
      A 1-D integer Tensor with shape `[nrows]`.
      The returned tensor is nonnegative, and is sorted in ascending order.

    #### Example:

    >>> rt = tf.ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
    >>> print(rt.values)
    tf.Tensor([3 1 4 1 5 9 2 6], shape=(8,), dtype=int32)
    >>> print(rt.row_limits())  # indices of row limits in rt.values
    tf.Tensor([4 4 7 8 8], shape=(5,), dtype=int64)

    """
    with ops.name_scope(name, "RaggedRowLimits", [self]):
      return self._row_partition.row_limits()

  def row_lengths(self, axis=1, name=None):
    """Returns the lengths of the rows in this ragged tensor.

    `rt.row_lengths()[i]` indicates the number of values in the
    `i`th row of `rt`.

    Args:
      axis: An integer constant indicating the axis whose row lengths should be
        returned.
      name: A name prefix for the returned tensor (optional).

    Returns:
      A potentially ragged integer Tensor with shape `self.shape[:axis]`.

    Raises:
      ValueError: If `axis` is out of bounds.

    #### Example:

    >>> rt = tf.ragged.constant(
    ...     [[[3, 1, 4], [1]], [], [[5, 9], [2]], [[6]], []])
    >>> print(rt.row_lengths())  # lengths of rows in rt
    tf.Tensor([2 0 2 1 0], shape=(5,), dtype=int64)
    >>> print(rt.row_lengths(axis=2))  # lengths of axis=2 rows.
    <tf.RaggedTensor [[3, 1], [], [2, 1], [1], []]>

    """
    if axis == 0:
      return self._row_partition.nrows()

    if axis == 1:
      return self._row_partition.row_lengths()

    with ops.name_scope(name, "RaggedRowLengths", [self]):
      axis = array_ops.get_positive_axis(
          axis, self.shape.rank, ndims_name="rank(self)")
      if axis == 0:
        return self.nrows()
      elif axis == 1:
        splits = self.row_splits
        return splits[1:] - splits[:-1]
      elif isinstance(self.values, RaggedTensor):
        return self.with_values(self.values.row_lengths(axis - 1))
      else:
        shape = array_ops.shape(self.values, out_type=self._row_partition.dtype)
        return self.with_values(
            array_ops.ones(shape[:axis - 1], self._row_partition.dtype) *
            shape[axis - 1])

  def nested_row_lengths(self, name=None):
    """Returns a tuple containing the row_lengths for all ragged dimensions.

    `rt.nested_row_lengths()` is a tuple containing the `row_lengths` tensors
    for all ragged dimensions in `rt`, ordered from outermost to innermost.

    Args:
      name: A name prefix for the returned tensors (optional).

    Returns:
      A `tuple` of 1-D integer `Tensors`.  The length of the tuple is equal to
      `self.ragged_rank`.
    """
    with ops.name_scope(name, "RaggedNestedRowLengths", [self]):
      rt_nested_row_lengths = []
      rt = self
      while isinstance(rt, RaggedTensor):
        rt_nested_row_lengths.append(rt.row_lengths())
        rt = rt.values
      return tuple(rt_nested_row_lengths)

  def bounding_shape(self, axis=None, name=None, out_type=None):
    """Returns the tight bounding box shape for this `RaggedTensor`.

    Args:
      axis: An integer scalar or vector indicating which axes to return the
        bounding box for.  If not specified, then the full bounding box is
        returned.
      name: A name prefix for the returned tensor (optional).
      out_type: `dtype` for the returned tensor.  Defaults to
        `self.row_splits.dtype`.

    Returns:
      An integer `Tensor` (`dtype=self.row_splits.dtype`).  If `axis` is not
      specified, then `output` is a vector with
      `output.shape=[self.shape.ndims]`.  If `axis` is a scalar, then the
      `output` is a scalar.  If `axis` is a vector, then `output` is a vector,
      where `output[i]` is the bounding size for dimension `axis[i]`.

    #### Example:

    >>> rt = tf.ragged.constant([[1, 2, 3, 4], [5], [], [6, 7, 8, 9], [10]])
    >>> rt.bounding_shape().numpy()
    array([5, 4])

    """
    if out_type is None:
      out_type = self._row_partition.dtype
    else:
      out_type = dtypes.as_dtype(out_type)
    with ops.name_scope(name, "RaggedBoundingBox", [self, axis]):
      nested_splits = self.nested_row_splits
      rt_flat_values = self.flat_values

      # Optimized special cases for when axis=0 or axis=1:
      if isinstance(axis, int):
        if axis == 0:
          return array_ops.shape(nested_splits[0], out_type=out_type)[0] - 1
        elif axis == 1:
          result = math_ops.maximum(math_ops.reduce_max(self.row_lengths()), 0)
          if out_type != self._row_partition.dtype:
            result = math_ops.cast(result, out_type)
          return result

      splits_shape = array_ops.shape(self.row_splits, out_type=out_type)
      flat_values_shape = array_ops.shape(rt_flat_values, out_type=out_type)

      ragged_dimensions = [splits_shape[0] - 1] + [
          math_ops.maximum(math_ops.reduce_max(splits[1:] - splits[:-1]), 0)
          for splits in nested_splits
      ]
      inner_dimensions = flat_values_shape[1:]

      if out_type != self._row_partition.dtype:
        ragged_dimensions = [
            math_ops.cast(d, out_type) for d in ragged_dimensions
        ]
      bbox = array_ops.concat(
          [array_ops.stack(ragged_dimensions), inner_dimensions], axis=0)
      return bbox if axis is None else array_ops.gather(bbox, axis)

  #=============================================================================
  # Transformation
  #=============================================================================

  def with_values(self, new_values):
    """Returns a copy of `self` with `values` replaced by `new_value`.

    Preserves cached row-partitioning tensors such as `self.cached_nrows` and
    `self.cached_value_rowids` if they have values.

    Args:
      new_values: Potentially ragged tensor to use as the `values` for the
        returned `RaggedTensor`.  Must have `rank > 0`, and must have the same
        number of rows as `self.values`.

    Returns:
      A `RaggedTensor`.  `result.rank = 1 + new_values.rank`.
      `result.ragged_rank = 1 + new_values.ragged_rank`
    """
    new_values = _convert_to_ragged_tensor_values(new_values)
    new_values.shape.with_rank_at_least(1)
    self.values.shape[:1].assert_is_compatible_with(new_values.shape[:1])
    if (isinstance(new_values, RaggedTensor) and
        self._row_partition.dtype != new_values.row_splits.dtype):
      if not ragged_config.auto_cast_partition_dtype():
        raise ValueError("self and new_values have mismatched row_splits "
                         "dtypes; use RaggedTensor.with_row_splits_dtype() to "
                         "convert them to compatible dtypes.")
      new_values = new_values.with_row_splits_dtype(dtypes.int64)
      return self.with_row_splits_dtype(dtypes.int64).with_values(new_values)
    return RaggedTensor(
        values=new_values, row_partition=self._row_partition, internal=True)

  def with_flat_values(self, new_values):
    """Returns a copy of `self` with `flat_values` replaced by `new_value`.

    Preserves cached row-partitioning tensors such as `self.cached_nrows` and
    `self.cached_value_rowids` if they have values.

    Args:
      new_values: Potentially ragged tensor that should replace
        `self.flat_values`.  Must have `rank > 0`, and must have the same number
        of rows as `self.flat_values`.

    Returns:
      A `RaggedTensor`.
      `result.rank = self.ragged_rank + new_values.rank`.
      `result.ragged_rank = self.ragged_rank + new_values.ragged_rank`.
    """
    if isinstance(self._values, RaggedTensor):
      return self.with_values(self.values.with_flat_values(new_values))
    else:
      new_values = _convert_to_ragged_tensor_values(new_values)
    return self.with_values(new_values)

  def with_row_splits_dtype(self, dtype):
    """Returns a copy of this RaggedTensor with the given `row_splits` dtype.

    For RaggedTensors with multiple ragged dimensions, the `row_splits` for all
    nested `RaggedTensor` objects are cast to the given dtype.

    Args:
      dtype: The dtype for `row_splits`.  One of `tf.int32` or `tf.int64`.

    Returns:
      A copy of this RaggedTensor, with the `row_splits` cast to the given
      type.
    """
    dtype = dtypes.as_dtype(dtype)
    if dtype not in (dtypes.int32, dtypes.int64):
      raise ValueError(f"Argument `row_splits` dtype must be int32 or int64. "
                       f"Received {dtype}.")
    if self._row_partition.dtype == dtype:
      return self
    current_values = self._values
    if isinstance(current_values, RaggedTensor):
      return RaggedTensor(
          values=current_values.with_row_splits_dtype(dtype),
          row_partition=self._row_partition.with_dtype(dtype),
          internal=True)
    else:
      return RaggedTensor(
          values=current_values,
          row_partition=self._row_partition.with_dtype(dtype),
          internal=True)

  def merge_dims(self, outer_axis, inner_axis):
    """Merges outer_axis...inner_axis into a single dimension.

    Returns a copy of this RaggedTensor with the specified range of dimensions
    flattened into a single dimension, with elements in row-major order.

    #### Examples:

    >>> rt = tf.ragged.constant([[[1, 2], [3]], [[4, 5, 6]]])
    >>> print(rt.merge_dims(0, 1))
    <tf.RaggedTensor [[1, 2], [3], [4, 5, 6]]>
    >>> print(rt.merge_dims(1, 2))
    <tf.RaggedTensor [[1, 2, 3], [4, 5, 6]]>
    >>> print(rt.merge_dims(0, 2))
    tf.Tensor([1 2 3 4 5 6], shape=(6,), dtype=int32)

    To mimic the behavior of `np.flatten` (which flattens all dimensions), use
    `rt.merge_dims(0, -1).  To mimic the behavior of `tf.layers.Flatten` (which
    flattens all dimensions except the outermost batch dimension), use
    `rt.merge_dims(1, -1)`.

    Args:
      outer_axis: `int`: The first dimension in the range of dimensions to
        merge. May be negative if `self.shape.rank` is statically known.
      inner_axis: `int`: The last dimension in the range of dimensions to merge.
        May be negative if `self.shape.rank` is statically known.

    Returns:
      A copy of this tensor, with the specified dimensions merged into a
      single dimension.  The shape of the returned tensor will be
      `self.shape[:outer_axis] + [N] + self.shape[inner_axis + 1:]`, where `N`
      is the total number of slices in the merged dimensions.
    """
    outer_axis = array_ops.get_positive_axis(
        outer_axis,
        self.shape.rank,
        axis_name="outer_axis",
        ndims_name="rank(self)")
    inner_axis = array_ops.get_positive_axis(
        inner_axis,
        self.shape.rank,
        axis_name="inner_axis",
        ndims_name="rank(self)")
    if not outer_axis <= inner_axis:
      raise ValueError(f"Expected outer_axis ({outer_axis}) to be less than or "
                       f"equal to inner_axis ({inner_axis}).")
    return merge_dims(self, outer_axis, inner_axis)

  def _set_shape(self, shape):
    """Updates the static shape of `self` to be `shape`.

    * If a dimension of `shape` has known rank, and is encoded via
      partitioning, then this will update the corresponding partition to
      define `_uniform_row_length` and `nrows`.
    * If a dimension of `shape` has a known rank, and is encoded as one
      of the `flat_values` dimensions, then `flat_values.set_shape()` will
      be used to update its shape.

    Warning: Using this method to assert an incorrect shape for a RaggedTensor
    (i.e., one that's not consistent with its actual shape) can cause
    segmentation faults and very difficult-to-diagnose behavior.  Only use this
    method if you are certain that the shape is correct.

    Args:
      shape: `tf.TensorShape` specifying the shape for this `RaggedTensor`.
    """
    # TODO(edloper): Refactor this to not directly access private members
    # of RowPartition.
    # pylint: disable=protected-access

    shape = tensor_shape.as_shape(shape)
    if shape.rank is None:
      return  # Nothing to do.

    shape = shape.as_list()

    # Outermost dimension
    if shape[0] is not None:
      self._row_partition._row_splits.set_shape(shape[0] + 1)

    # Partitioned dimensions
    dtype = self._row_partition.dtype
    for i, partition in enumerate(self._nested_row_partitions):
      size = shape[i + 1]
      if size is not None:
        if partition._uniform_row_length is not None:
          old_row_length = tensor_util.constant_value(
              partition._uniform_row_length)
          if old_row_length is not None:
            if size == old_row_length:
              continue  # already have shape info for this axis.
            else:
              raise ValueError(f"Inconsistent size for axis {i + 1}: "
                               f"{old_row_length} vs. {size}.")
        partition._uniform_row_length = ops.convert_to_tensor(size, dtype)
        if partition._nrows is None:
          partition._nrows = array_ops.size(
              partition._row_splits, out_type=dtype) - 1

    # self.flat_values could be a CompositeTensor and doesn't have set_shape.
    if hasattr(self.flat_values, "set_shape"):
      # Inner dimensions
      flat_shape = tensor_shape.as_shape([None] + shape[self.ragged_rank + 1:])
      self.flat_values.set_shape(flat_shape)

  #=============================================================================
  # Tensor Type Conversions
  #=============================================================================

  @classmethod
  @dispatch.add_dispatch_support
  def from_tensor(cls,
                  tensor,
                  lengths=None,
                  padding=None,
                  ragged_rank=1,
                  name=None,
                  row_splits_dtype=dtypes.int64):
    """Converts a `tf.Tensor` into a `RaggedTensor`.

    The set of absent/default values may be specified using a vector of lengths
    or a padding value (but not both).  If `lengths` is specified, then the
    output tensor will satisfy `output[row] = tensor[row][:lengths[row]]`. If
    'lengths' is a list of lists or tuple of lists, those lists will be used
    as nested row lengths. If `padding` is specified, then any row *suffix*
    consisting entirely of `padding` will be excluded from the returned
    `RaggedTensor`.  If neither `lengths` nor `padding` is specified, then the
    returned `RaggedTensor` will have no absent/default values.

    Examples:

    >>> dt = tf.constant([[5, 7, 0], [0, 3, 0], [6, 0, 0]])
    >>> tf.RaggedTensor.from_tensor(dt)
    <tf.RaggedTensor [[5, 7, 0], [0, 3, 0], [6, 0, 0]]>
    >>> tf.RaggedTensor.from_tensor(dt, lengths=[1, 0, 3])
    <tf.RaggedTensor [[5], [], [6, 0, 0]]>

    >>> tf.RaggedTensor.from_tensor(dt, padding=0)
    <tf.RaggedTensor [[5, 7], [0, 3], [6]]>

    >>> dt = tf.constant([[[5, 0], [7, 0], [0, 0]],
    ...                   [[0, 0], [3, 0], [0, 0]],
    ...                   [[6, 0], [0, 0], [0, 0]]])
    >>> tf.RaggedTensor.from_tensor(dt, lengths=([2, 0, 3], [1, 1, 2, 0, 1]))
    <tf.RaggedTensor [[[5], [7]], [], [[6, 0], [], [0]]]>

    Args:
      tensor: The `Tensor` to convert.  Must have rank `ragged_rank + 1` or
        higher.
      lengths: An optional set of row lengths, specified using a 1-D integer
        `Tensor` whose length is equal to `tensor.shape[0]` (the number of rows
        in `tensor`).  If specified, then `output[row]` will contain
        `tensor[row][:lengths[row]]`.  Negative lengths are treated as zero. You
          may optionally pass a list or tuple of lengths to this argument, which
          will be used as nested row lengths to construct a ragged tensor with
          multiple ragged dimensions.
      padding: An optional padding value.  If specified, then any row suffix
        consisting entirely of `padding` will be excluded from the returned
        RaggedTensor.  `padding` is a `Tensor` with the same dtype as `tensor`
        and with `shape=tensor.shape[ragged_rank + 1:]`.
      ragged_rank: Integer specifying the ragged rank for the returned
        `RaggedTensor`.  Must be greater than zero.
      name: A name prefix for the returned tensors (optional).
      row_splits_dtype: `dtype` for the returned `RaggedTensor`'s `row_splits`
        tensor.  One of `tf.int32` or `tf.int64`.

    Returns:
      A `RaggedTensor` with the specified `ragged_rank`.  The shape of the
      returned ragged tensor is compatible with the shape of `tensor`.

    Raises:
      ValueError: If both `lengths` and `padding` are specified.
      ValueError: If the rank of `tensor` is 0 or 1.
    """
    row_splits_dtype = dtypes.as_dtype(row_splits_dtype)
    if lengths is not None and padding is not None:
      raise ValueError("Specify argument `lengths` or `padding`, but not both.")
    if not isinstance(ragged_rank, int):
      raise TypeError(f"Argument `ragged_rank` must be an int. "
                      f"Received {ragged_rank}.")
    if ragged_rank <= 0:
      raise ValueError(f"Argument `ragged_rank` must be greater than 0. "
                       f"Received {ragged_rank}.")

    with ops.name_scope(name, "RaggedFromTensor", [tensor, lengths, padding]):
      tensor = ops.convert_to_tensor(tensor, name="tensor")
      if tensor.shape.rank is not None and tensor.shape.rank < 2:
        raise ValueError(f"The rank of a RaggedTensor must be greater than 1, "
                         f"i.e., a list of scalars won't have ragged "
                         f"dimensions. Received argument `tensor` with rank "
                         f"{tensor.shape.rank}.")
      tensor.shape.with_rank_at_least(ragged_rank + 1)
      input_shape = array_ops.shape(tensor, out_type=row_splits_dtype)
      ncols = input_shape[1]

      # Handle nested row lengths.
      if (lengths is not None and isinstance(lengths, (list, tuple)) and
          len(lengths) and not isinstance(lengths[0], (int, float))):
        if ragged_rank not in (1, len(lengths)):
          # Note: we accept `ragged_rank=1` here because it's the default value;
          # i.e., if the user passes in a tuple of lengths, but doesn't specify
          # ragged_rank, then we should use that tuple to determine ragged_rank.
          # We only want to complain if they pass in an explicit ragged_rank
          # that doesn't match len(lengths).
          raise ValueError(f"If Argument `lengths` is a tuple of row_lengths, "
                           f"argument `ragged_rank` must be "
                           f"len(lengths): {len(lengths)}. Received "
                           f"ragged_rank: {ragged_rank}.")
        # Rather than reconstructing the tensor mask directly, we can
        # recreate it as a boolean RaggedTensor, then densify that and use
        # that as the mask to clear out the unused data in the passed tensor.
        tensor.shape.with_rank_at_least(len(lengths) + 1)
        num_tokens = math_ops.reduce_sum(lengths[-1])
        ones_mask = array_ops.ones([num_tokens], dtype=dtypes.bool)
        ragged_mask = cls.from_nested_row_lengths(
            ones_mask, lengths, validate=False)
        dense_ragged_mask = ragged_mask.to_tensor(default_value=False)
        masked_data = array_ops.boolean_mask(tensor, dense_ragged_mask)
        return cls.from_nested_row_lengths(masked_data, lengths, validate=False)

      # Handle ragged_rank>1 via recursion:
      # If the output should have multiple ragged dimensions, then first
      # flatten the tensor to eliminate all but the last ragged dimension,
      # and recursively convert that flattened tensor.  Then add on the splits
      # for the dimensions that we flattened out.
      if ragged_rank > 1:
        if tensor.shape.is_fully_defined():
          input_shape = tensor.shape.as_list()
          # The total number of elements in each  dimension.  E.g., if
          # input_shape=[3, 4, 5, 6], then dim[2] has 3*4*5 elements in total.
          dim_size = np.cumprod(input_shape)
          new_shape = [dim_size[ragged_rank - 1]] + input_shape[ragged_rank:]
        else:
          dim_size = math_ops.cumprod(input_shape)
          new_shape = array_ops.concat(
              [[dim_size[ragged_rank - 1]], input_shape[ragged_rank:]], axis=0)
        flattened = array_ops.reshape(tensor, new_shape)
        result = cls.from_tensor(
            flattened, lengths, padding, row_splits_dtype=row_splits_dtype)

        for axis in range(ragged_rank - 1, 0, -1):
          dim_len = tensor_shape.dimension_at_index(tensor.shape, axis).value
          if dim_len is None:
            dim_len = input_shape[axis]
          else:
            dim_len = constant_op.constant(dim_len, row_splits_dtype)
          result = RaggedTensor.from_uniform_row_length(
              values=result,
              uniform_row_length=dim_len,
              nrows=dim_size[axis - 1],
              validate=False)
        return result

      # If padding was specified, then use it to find row lengths.
      if padding is not None:
        padding = ops.convert_to_tensor(
            padding, name="padding", dtype=tensor.dtype)
        padding.shape.assert_is_compatible_with(tensor.shape[2:])

        # Find places where the padding is equal to the tensor.  (This will
        # broadcast `padding` across the outermost 2 dimensions of `tensor`,
        # so `has_default_value.shape = tensor.shape`.)
        has_default_value = math_ops.equal(padding, tensor)

        # If the padding isn't a scalar, then require that all values in the
        # padding match each item in the tensor.  After this block of code,
        # `has_default.shape = tensor.shape[:2]`.  (Unfortunately, we can't just
        # use reduce_all for both cases, becaue when you pass an empty `axis`
        # list to reduce_all, it reduces all axes; but we want it to reduce no
        # axes -- i.e., to be a no-op.)
        tensor_rank = array_ops.rank(tensor)
        reduce_axis = math_ops.range(2, tensor_rank)
        has_default = control_flow_ops.cond(
            tensor_rank > 2,
            lambda: math_ops.reduce_all(has_default_value, axis=reduce_axis),
            lambda: has_default_value)
        has_default.set_shape(tensor_shape.TensorShape([None, None]))
        has_default.set_shape(tensor.shape[:2])

        # Use has_default to find the length of each row: for each
        # non-default item in a row, calculate the length that the row needs to
        # have to include that item; and then take the max of those values
        # (across each row).
        has_nondefault = math_ops.logical_not(has_default)
        has_nondefault = math_ops.cast(has_nondefault, row_splits_dtype)
        length_for_nondefault_value = (
            has_nondefault *
            array_ops.expand_dims(math_ops.range(1, ncols + 1), 0))
        lengths = math_ops.reduce_max(length_for_nondefault_value, axis=1)

      if lengths is not None:
        # If we have lengths (either directly supplied, or computed from
        # paddings), then use those to construct splits; and then use masking
        # to get the corresponding values.
        lengths = ragged_util.convert_to_int_tensor(lengths, "lengths",
                                                    row_splits_dtype)
        lengths.shape.assert_has_rank(1)
        lengths = math_ops.minimum(lengths, ncols)
        lengths = math_ops.maximum(lengths, 0)
        limits = math_ops.cumsum(lengths)
        splits = array_ops.concat(
            [array_ops.zeros([1], row_splits_dtype), limits], axis=0)
        mask = array_ops.sequence_mask(lengths, maxlen=ncols)
        values = array_ops.boolean_mask(tensor, mask)
        return cls.from_row_splits(values, splits, validate=False)

      # If neither padding nor lengths were specified, then create a splits
      # vector that contains no default values, and reshape the input tensor
      # to form the values for the RaggedTensor.
      values_shape = array_ops.concat(
          [[input_shape[0] * input_shape[1]], input_shape[2:]], axis=0)
      values = array_ops.reshape(tensor, values_shape)
      const_nrows = tensor_shape.dimension_at_index(tensor.shape, 0).value
      const_ncols = tensor_shape.dimension_at_index(tensor.shape, 1).value
      if const_nrows is not None:
        nrows = constant_op.constant(const_nrows, row_splits_dtype)
      else:
        nrows = input_shape[0]
      if const_ncols is not None:
        ncols = constant_op.constant(const_ncols, row_splits_dtype)
      else:
        ncols = input_shape[1]
      return RaggedTensor.from_uniform_row_length(
          values=values, uniform_row_length=ncols, nrows=nrows, validate=False)

  def to_tensor(self, default_value=None, name=None, shape=None):
    """Converts this `RaggedTensor` into a `tf.Tensor`.

    If `shape` is specified, then the result is padded and/or truncated to
    the specified shape.

    Examples:

    >>> rt = tf.ragged.constant([[9, 8, 7], [], [6, 5], [4]])
    >>> print(rt.to_tensor())
    tf.Tensor(
        [[9 8 7] [0 0 0] [6 5 0] [4 0 0]], shape=(4, 3), dtype=int32)
    >>> print(rt.to_tensor(shape=[5, 2]))
    tf.Tensor(
        [[9 8] [0 0] [6 5] [4 0] [0 0]], shape=(5, 2), dtype=int32)

    Args:
      default_value: Value to set for indices not specified in `self`. Defaults
        to zero.  `default_value` must be broadcastable to
        `self.shape[self.ragged_rank + 1:]`.
      name: A name prefix for the returned tensors (optional).
      shape: The shape of the resulting dense tensor.  In particular,
        `result.shape[i]` is `shape[i]` (if `shape[i]` is not None), or
        `self.bounding_shape(i)` (otherwise).`shape.rank` must be `None` or
        equal to `self.rank`.

    Returns:
      A `Tensor` with shape `ragged.bounding_shape(self)` and the
      values specified by the non-empty values in `self`.  Empty values are
      assigned `default_value`.
    """
    with ops.name_scope(name, "RaggedToTensor", [self, default_value, shape]):
      if default_value is not None:
        default_value = ops.convert_to_tensor(
            default_value, name="default_value", dtype=self.dtype)
      type_tensor_pairs = _get_row_partition_type_tensor_pairs(self)
      row_partition_types = [x[0] for x in type_tensor_pairs]
      row_partition_tensors = [x[1] for x in type_tensor_pairs]
      if default_value is None:
        default_value = array_ops.zeros((), self.dtype)

      if (isinstance(shape, (list, tuple)) and
          any(isinstance(v, ops.Tensor) for v in shape) and
          all(isinstance(v, (int, ops.Tensor)) for v in shape)):
        shape = array_ops.stack(shape)

      shape_tensor = _shape_as_tensor(shape, row_partition_tensors[0].dtype)
      tensor = gen_ragged_conversion_ops.ragged_tensor_to_tensor(
          shape=shape_tensor,
          values=self.flat_values,
          default_value=default_value,
          row_partition_types=row_partition_types,
          row_partition_tensors=row_partition_tensors)

      ragged_shape = self.shape

      if ragged_shape.rank is not None and not isinstance(shape, ops.Tensor):
        # Merged self.shape and shape, favoring the second one as it takes
        # into account potential padding added to the output.
        shape = tensor_shape.as_shape(shape)
        if shape.rank is None:
          output_shape = ragged_shape
        else:
          # At this point we can assume that hshape.rank == ragged_shape.rank
          # because otherwise it would have failed earlier.
          output_shape = [
              s1 if s1 is not None else s2
              for (s1, s2) in zip(shape.as_list(), ragged_shape.as_list())
          ]
        tensor.set_shape(output_shape)

      return tensor

  @classmethod
  @dispatch.add_dispatch_support
  def from_sparse(cls, st_input, name=None, row_splits_dtype=dtypes.int64):
    """Converts a 2D `tf.sparse.SparseTensor` to a `RaggedTensor`.

    Each row of the `output` `RaggedTensor` will contain the explicit values
    from the same row in `st_input`.  `st_input` must be ragged-right.  If not
    it is not ragged-right, then an error will be generated.

    Example:

    >>> indices = [[0, 0], [0, 1], [0, 2], [1, 0], [3, 0]]
    >>> st = tf.sparse.SparseTensor(indices=indices,
    ...                             values=[1, 2, 3, 4, 5],
    ...                             dense_shape=[4, 3])
    >>> tf.RaggedTensor.from_sparse(st).to_list()
    [[1, 2, 3], [4], [], [5]]

    Currently, only two-dimensional `SparseTensors` are supported.

    Args:
      st_input: The sparse tensor to convert.  Must have rank 2.
      name: A name prefix for the returned tensors (optional).
      row_splits_dtype: `dtype` for the returned `RaggedTensor`'s `row_splits`
        tensor.  One of `tf.int32` or `tf.int64`.

    Returns:
      A `RaggedTensor` with the same values as `st_input`.
      `output.ragged_rank = rank(st_input) - 1`.
      `output.shape = [st_input.dense_shape[0], None]`.
    Raises:
      ValueError: If the number of dimensions in `st_input` is not known
        statically, or is not two.
    """
    row_splits_dtype = dtypes.as_dtype(row_splits_dtype)
    if not sparse_tensor.is_sparse(st_input):
      raise TypeError(f"Argument `st_input` must be of type SparseTensor, but "
                      f"is of type {type(st_input).__name__}.")
    with ops.name_scope(name, "RaggedFromSparse", [st_input]):
      st_input = sparse_tensor.convert_to_tensor_or_sparse_tensor(
          st_input, name="st_input")

      if st_input.dense_shape.shape.ndims is None:
        static_rank_from_dense_shape = None
      else:
        static_rank_from_dense_shape = st_input.dense_shape.shape.dims[0].value

      if st_input.indices.shape.ndims is None:
        static_rank_from_indices = None
      else:
        static_rank_from_indices = st_input.indices.shape.dims[1].value

      if static_rank_from_dense_shape != 2 and static_rank_from_indices != 2:
        raise ValueError("rank(st_input) must be 2.")

      with ops.control_dependencies(
          _assert_sparse_indices_are_ragged_right(st_input.indices)):
        # Treat sparse row indices as segment ids to generate a splits tensor
        # thta we can pair with the sparse tensor values.  (Ignore sparse column
        # indices.)
        segment_ids = math_ops.cast(st_input.indices[:, 0], row_splits_dtype)
        num_segments = math_ops.cast(st_input.dense_shape[0], row_splits_dtype)
        return cls.from_value_rowids(
            st_input.values, segment_ids, num_segments, validate=False)

  def to_sparse(self, name=None):
    """Converts this `RaggedTensor` into a `tf.sparse.SparseTensor`.

    Example:

    >>> rt = tf.ragged.constant([[1, 2, 3], [4], [], [5, 6]])
    >>> print(rt.to_sparse())
    SparseTensor(indices=tf.Tensor(
                     [[0 0] [0 1] [0 2] [1 0] [3 0] [3 1]],
                     shape=(6, 2), dtype=int64),
                 values=tf.Tensor([1 2 3 4 5 6], shape=(6,), dtype=int32),
                 dense_shape=tf.Tensor([4 3], shape=(2,), dtype=int64))

    Args:
      name: A name prefix for the returned tensors (optional).

    Returns:
      A SparseTensor with the same values as `self`.
    """
    with ops.name_scope(name, "RaggedToSparse", [self]):
      result = gen_ragged_conversion_ops.ragged_tensor_to_sparse(
          self.nested_row_splits, self.flat_values, name=name)
      return sparse_tensor.SparseTensor(result.sparse_indices,
                                        result.sparse_values,
                                        result.sparse_dense_shape)

  @classmethod
  def _from_variant(cls,
                    variant,
                    dtype,
                    output_ragged_rank,
                    input_ragged_rank=None,
                    row_splits_dtype=dtypes.int64,
                    name=None):
    """Converts a `variant` Tensor into a `RaggedTensor`.

    The input `variant` could be a scalar, meaning it encodes a single
    `RaggedTensor` with ragged_rank `output_ragged_rank`. Alternatively it could
    have an arbitrary rank, in which case each element is decoded into a
    `RaggedTensor` with ragged_rank `input_ragged_rank` and these are then
    stacked according to the input shape to output a single `RaggedTensor`
    with ragged_rank `output_ragged_rank`. If `input_ragged_rank` is not
    provided, it is inferred dynamically as `output_ragged_rank` -
    `rank(variant)`. If `input_ragged_rank` is provided, the following must be
    true: `output_ragged_rank` = `input_ragged_rank` + `rank(variant)`.

    Example:

    >>> rt = tf.ragged.constant([[0], [1, 2]])
    >>> et = rt._to_variant()
    >>> stacked_et = tf.stack([et, et])
    >>> tf.RaggedTensor._from_variant(  # scalar input.
    ...     et, dtype=tf.int32, output_ragged_rank=1).to_list()
    [[0], [1, 2]]
    >>> tf.RaggedTensor._from_variant(  # batched input.
    ...     stacked_et, dtype=tf.int32, output_ragged_rank=2).to_list()
    [[[0], [1, 2]], [[0], [1, 2]]]

    Args:
      variant: A `variant` Tensor representing an encoded (possibly
        nested-batched) `RaggedTensor`.
      dtype: The dtype of the encoded `RaggedTensor`.
      output_ragged_rank: The expected ragged rank of the output `RaggedTensor`.
      input_ragged_rank: The ragged rank of each encoded `RaggedTensor`. This is
        optional and inferred dynamically if not provided.
      row_splits_dtype: `dtype` for the RaggedTensor's `row_splits` tensor. One
        of `tf.int32` or `tf.int64`.
      name: A name prefix for the returned tensors (optional).

    Returns:
      A `RaggedTensor` of dtype `dtype` and ragged rank `output_ragged_rank`.

    Raises:
      ValueError: If the input rank is known, `input_ragged_rank` is provided
          and `output_ragged_rank` = `input_ragged_rank` + `rank(variant)` does
          not hold.
    """
    variant = ops.convert_to_tensor(
        variant, name="variant", dtype=dtypes.variant)
    if (variant.shape.ndims is not None and input_ragged_rank is not None and
        output_ragged_rank != input_ragged_rank + variant.shape.ndims):
      raise ValueError(
          f"Argument `output_ragged_rank` ({output_ragged_rank}) must be equal "
          f"to `input_ragged_rank` + `variant.shape.ndims` "
          f"({input_ragged_rank} + {variant.shape.ndims}).")
    input_ragged_rank = -1 if input_ragged_rank is None else input_ragged_rank
    with ops.name_scope(
        name, "RaggedFromVariant",
        [variant, dtype, input_ragged_rank, output_ragged_rank]):
      result = gen_ragged_conversion_ops.ragged_tensor_from_variant(
          variant, input_ragged_rank, max(output_ragged_rank, 0), dtype,
          row_splits_dtype, name)
      return cls.from_nested_row_splits(
          result.output_dense_values,
          result.output_nested_splits,
          validate=False)

  def _to_variant(self, batched_input=False, name=None):
    """Converts this `RaggedTensor` into a `variant` Tensor.

    If `batched_input` is `True`, then the `RaggedTensor` is unbatched along the
    zero-th dimension, each component `RaggedTensor` is encoded into a scalar
    `variant` Tensor, and these are stacked to return a 1-D `variant` Tensor.
    If `batched_input` is `False`, then the `RaggedTensor` is encoded as is and
    a scalar `variant` Tensor is returned.

    Example:
    >>> rt = tf.ragged.constant([[[0]], [[1]], [[2]]])
    >>> rt._to_variant().shape.as_list()
    []
    >>> rt._to_variant(batched_input=True).shape.as_list()
    [3]

    Args:
      batched_input: If `True`, the `RaggedTensor` is unbatched and converted to
        a `variant` vector. Set to `False` by default.
      name: A name prefix for the returned tensors (optional).

    Returns:
      A `variant` Tensor that encodes this `RaggedTensor`.
    """
    with ops.name_scope(name, "RaggedToVariant", [self, batched_input]):
      return gen_ragged_conversion_ops.ragged_tensor_to_variant(
          self.nested_row_splits, self.flat_values, batched_input, name)

  #=============================================================================
  # String Encoding
  #=============================================================================
  def __repr__(self):
    if self._is_eager():
      # The np.array2string in _formatter provides a separator argument, but
      # doesn't handle recursive calls correctly. The np.printoptions handles
      # recursive calls correctly, but doesn't provide a separator argument.
      # Combines them together to print elements separated by comma, while
      # avoiding the redundant array prefixes and dtypes. For example,
      # the value of tf.ragged.constant([[1, 2], [3, 4]]) will look like
      #
      # [[1, 2],
      #  [3, 4]]
      with np.printoptions(formatter={"all": _formatter}):
        value_text = _formatter(self.numpy())
      return f"<tf.RaggedTensor {value_text}>"
    else:
      return "tf.RaggedTensor(values=%s, row_splits=%s)" % (self.values,
                                                            self.row_splits)

  #=============================================================================
  # Eager Execution Mode
  #=============================================================================

  def numpy(self):
    """Returns a numpy `array` with the values for this `RaggedTensor`.

    Requires that this `RaggedTensor` was constructed in eager execution mode.

    Ragged dimensions are encoded using numpy `arrays` with `dtype=object` and
    `rank=1`, where each element is a single row.

    #### Examples

    In the following example, the value returned by `RaggedTensor.numpy()`
    contains three numpy `array` objects: one for each row (with `rank=1` and
    `dtype=int64`), and one to combine them (with `rank=1` and `dtype=object`):

    >>> tf.ragged.constant([[1, 2, 3], [4, 5]], dtype=tf.int64).numpy()
    array([array([1, 2, 3]), array([4, 5])], dtype=object)

    Uniform dimensions are encoded using multidimensional numpy `array`s.  In
    the following example, the value returned by `RaggedTensor.numpy()` contains
    a single numpy `array` object, with `rank=2` and `dtype=int64`:

    >>> tf.ragged.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int64).numpy()
    array([[1, 2, 3], [4, 5, 6]])

    Returns:
      A numpy `array`.
    """
    if not self._is_eager():
      raise ValueError("RaggedTensor.numpy() is only supported in eager mode.")
    values = self.values.numpy()
    splits = self.row_splits.numpy()
    rows = [values[splits[i]:splits[i + 1]] for i in range(len(splits) - 1)]
    if not rows:
      return np.zeros((0, 0) + values.shape[1:], dtype=values.dtype)
    # Note: if `rows` have ragged lengths, then they will be stored in a
    # np.ndarray with dtype=object and rank=1.  If they have uniform lengths,
    # they will be combined into a single np.ndarray with dtype=row.dtype and
    # rank=row.rank+1.
    #
    # Manually set dtype as numpy now complains when given ragged rows.
    has_variable_length_rows = any(len(row) != len(rows[0]) for row in rows)
    dtype = np.object_ if has_variable_length_rows else None
    return np.array(rows, dtype=dtype)

  def to_list(self):
    """Returns a nested Python `list` with the values for this `RaggedTensor`.

    Requires that `rt` was constructed in eager execution mode.

    Returns:
      A nested Python `list`.
    """
    if not isinstance(self.row_splits, ops.EagerTensor):
      raise ValueError("to_list can only be used in eager mode.")
    row_splits = self.row_splits.numpy().tolist()
    values = self.values

    if isinstance(values, RaggedTensor):
      return [
          values[row_splits[i]:row_splits[i + 1]].to_list()
          for i in range(len(row_splits) - 1)
      ]
    else:
      # Convert values to a Python list.
      if hasattr(values, "numpy"):
        values_as_list = values.numpy().tolist()
      elif hasattr(values, "to_list"):
        values_as_list = values.to_list()
      else:
        raise ValueError("values must be convertible to a list")

      return [
          values_as_list[row_splits[i]:row_splits[i + 1]]
          for i in range(len(row_splits) - 1)
      ]

  def _eager_value(self):
    """Returns a RaggedTensorValue for self.  Requires self._is_eager()=true."""
    value = self.flat_values.numpy()
    for row_splits in reversed(self.nested_row_splits):
      value = ragged_tensor_value.RaggedTensorValue(value, row_splits.numpy())
    return value

  def _is_eager(self):
    """Returns True if values & row_splits Tensors are all `EagerTensor`s."""
    rt = self
    while isinstance(rt, RaggedTensor):
      if not isinstance(rt.row_splits, ops.EagerTensor):
        return False
      rt = rt.values
    return isinstance(rt, ops.EagerTensor)

  #=============================================================================
  # Operators
  #=============================================================================
  # To avoid circular dependencies, we define stub methods for operators here,
  # and then override them when the ragged_operators module is imported.

  def _overloaded_operator(name):  # pylint: disable=no-self-argument

    def stub(*args, **kwargs):
      del args, kwargs
      raise ValueError(
          f"You must import 'tensorflow.python.ops.ragged.ragged_ops' "
          f"before using RaggedTensor.{name}.")

    return stub

  __getitem__ = _overloaded_operator("__getitem__")
  __ge__ = _overloaded_operator("__ge__")
  __gt__ = _overloaded_operator("__gt__")
  __le__ = _overloaded_operator("__le__")
  __lt__ = _overloaded_operator("__lt__")
  __and__ = _overloaded_operator("__and__")
  __rand__ = _overloaded_operator("__rand__")
  __invert__ = _overloaded_operator("__invert__")
  __ror__ = _overloaded_operator("__ror__")
  __or__ = _overloaded_operator("__or__")
  __xor__ = _overloaded_operator("__xor__")
  __rxor__ = _overloaded_operator("__rxor__")
  __abs__ = _overloaded_operator("__abs__")
  __add__ = _overloaded_operator("__add__")
  __radd__ = _overloaded_operator("__radd__")
  __div__ = _overloaded_operator("__div__")
  __rdiv__ = _overloaded_operator("__rdiv__")
  __floordiv__ = _overloaded_operator("__floordiv__")
  __rfloordiv__ = _overloaded_operator("__rfloordiv__")
  __mod__ = _overloaded_operator("__mod__")
  __rmod__ = _overloaded_operator("__rmod__")
  __mul__ = _overloaded_operator("__mul__")
  __rmul__ = _overloaded_operator("__rmul__")
  __neg__ = _overloaded_operator("__neg__")
  __pow__ = _overloaded_operator("__pow__")
  __rpow__ = _overloaded_operator("__rpow__")
  __sub__ = _overloaded_operator("__sub__")
  __rsub__ = _overloaded_operator("__rsub__")
  __truediv__ = _overloaded_operator("__truediv__")
  __rtruediv__ = _overloaded_operator("__rtruediv__")
  del _overloaded_operator

  #=============================================================================
  # Name Scope
  #=============================================================================

  # This private function is used by ops.name_scope to ensure that all of the
  # input tensors for the scope belong to the same graph.  Defining this means
  # that you may include `RaggedTensor` objects in the name_scope `values`
  # list.
  def _as_graph_element(self):
    """Convert `self` to a graph element."""
    values = self.values
    while isinstance(values, RaggedTensor):
      values = values.values
    return values

  #=============================================================================
  # Composite Tensor
  #=============================================================================

  @property
  def _type_spec(self):
    return RaggedTensorSpec.from_value(self)

  def _shape_invariant_to_type_spec(self, shape):
    return RaggedTensorSpec(shape, self.dtype, self.ragged_rank,
                            self.row_splits.dtype)

  def consumers(self):
    return self._consumers()

  __composite_gradient__ = (
      composite_tensor_gradient.WithValuesCompositeTensorGradient())


def is_ragged(value):
  """Returns true if `value` is a ragged tensor or ragged tensor value."""
  return isinstance(value,
                    (RaggedTensor, ragged_tensor_value.RaggedTensorValue))


def match_row_splits_dtypes(*tensors, **kwargs):
  """Return a copy of `tensors` with row_splits all having the same dtype.

  Args:
    *tensors: A list of Tensors or RaggedTensors.
    **kwargs: If 'return_dtype=True', then return a tuple (dtype, tensors),
      where `dtype` is the data type used by row-splits, and `tensors` is the
      converted list of `Tensors` and `RaggedTensors`.

  Returns:
    The converted list of `Tensors` and `RaggedTensors`.
  """
  return_dtype = kwargs.pop("return_dtype", False)
  if kwargs:
    raise ValueError(f"Unexpected keyword args {kwargs}.")

  has_int32 = False
  has_int64 = False
  for tensor in tensors:
    if isinstance(tensor, RaggedTensor):
      if tensor.row_splits.dtype == dtypes.int32:
        has_int32 = True
      else:
        has_int64 = True

  if has_int32 and has_int64:
    if not ragged_config.auto_cast_partition_dtype():
      raise ValueError("Input RaggedTensors have mismatched row_splits dtypes; "
                       "use RaggedTensor.with_row_splits_dtype() to convert "
                       "them to compatible dtypes.")
    dtype = dtypes.int64
    tensors = tuple(
        t.with_row_splits_dtype(dtypes.int64) if isinstance(t, RaggedTensor
                                                           ) else t
        for t in tensors)

  elif has_int32:
    dtype = dtypes.int32
  else:
    dtype = dtypes.int64

  if return_dtype:
    return (dtype, tensors)
  else:
    return tensors


#===============================================================================
# RaggedTensorSpec
#===============================================================================
@tf_export("RaggedTensorSpec")
@type_spec.register("tf.RaggedTensorSpec")
class RaggedTensorSpec(type_spec.BatchableTypeSpec):
  """Type specification for a `tf.RaggedTensor`."""

  __slots__ = [
      "_shape", "_dtype", "_ragged_rank", "_row_splits_dtype",
      "_flat_values_spec"
  ]

  @property
  def dtype(self):
    """The `tf.dtypes.DType` specified by this type for the RaggedTensor.

    Examples:

    >>> rt = tf.ragged.constant([["a"], ["b", "c"]], dtype=tf.string)
    >>> tf.type_spec_from_value(rt).dtype
    tf.string

    Returns:
      A `tf.dtypes.DType` of the values in the RaggedTensor.
    """
    return self._dtype

  @property
  def shape(self):
    """The statically known shape of the RaggedTensor.

    Examples:

    >>> rt = tf.ragged.constant([[0], [1, 2]])
    >>> tf.type_spec_from_value(rt).shape
    TensorShape([2, None])

    >>> rt = tf.ragged.constant([[[0, 1]], [[1, 2], [3, 4]]], ragged_rank=1)
    >>> tf.type_spec_from_value(rt).shape
    TensorShape([2, None, 2])

    Returns:
      A `tf.TensorShape` containing the statically known shape of the
      RaggedTensor. Ragged dimensions have a size of `None`.
    """
    return self._shape

  @property
  def ragged_rank(self):
    """The number of times the RaggedTensor's flat_values is partitioned.

    Defaults to `shape.ndims - 1`.

    Examples:

    >>> values = tf.ragged.constant([[1, 2, 3], [4], [5, 6], [7, 8, 9, 10]])
    >>> tf.type_spec_from_value(values).ragged_rank
    1

    >>> rt1 = tf.RaggedTensor.from_uniform_row_length(values, 2)
    >>> tf.type_spec_from_value(rt1).ragged_rank
    2

    Returns:
      A Python `int` indicating the number of times the underlying `flat_values`
      Tensor has been partitioned to add a new dimension.
      I.e., `tf.rank(rt) = tf.rank(rt.flat_values) + rt.ragged_rank`.
    """
    return self._ragged_rank

  @property
  def row_splits_dtype(self):
    """The `tf.dtypes.DType` of the RaggedTensor's `row_splits`.

    Examples:

    >>> rt = tf.ragged.constant([[1, 2, 3], [4]], row_splits_dtype=tf.int64)
    >>> tf.type_spec_from_value(rt).row_splits_dtype
    tf.int64

    Returns:
      A `tf.dtypes.DType` for the RaggedTensor's `row_splits` tensor. One
      of `tf.int32` or `tf.int64`.
    """
    return self._row_splits_dtype

  @property
  def flat_values_spec(self):
    """The `TypeSpec` of the flat_values of RaggedTensor.

    Returns:
      - The TypeSpec of flat_values.
      - None when the flat_values is a Tensor.
    """
    return self._flat_values_spec

  @property
  def value_type(self):
    return RaggedTensor if self._ragged_rank > 0 else ops.Tensor

  def __init__(self,
               shape=None,
               dtype=dtypes.float32,
               ragged_rank=None,
               row_splits_dtype=dtypes.int64,
               flat_values_spec=None):
    """Constructs a type specification for a `tf.RaggedTensor`.

    Args:
      shape: The shape of the RaggedTensor, or `None` to allow any shape.  If a
        shape is specified, then all ragged dimensions must have size `None`.
      dtype: `tf.DType` of values in the RaggedTensor.
      ragged_rank: Python integer, the number of times the RaggedTensor's
        flat_values is partitioned.  Defaults to `shape.ndims - 1`.
      row_splits_dtype: `dtype` for the RaggedTensor's `row_splits` tensor. One
        of `tf.int32` or `tf.int64`.
      flat_values_spec: TypeSpec for flat_value of the RaggedTensor. It shall be
        provided when the flat_values is a CompositeTensor rather then Tensor.
        If both `dtype` and `flat_values_spec` and  are provided, `dtype` must
        be the same as `flat_values_spec.dtype`. (experimental)
    """
    self._shape = tensor_shape.as_shape(shape)
    self._row_splits_dtype = dtypes.as_dtype(row_splits_dtype)
    if flat_values_spec is not None:
      if dtype is None:
        dtype = flat_values_spec.dtype
      elif dtype != flat_values_spec.dtype:
        raise ValueError("dtype must be the same as flat_values_spec.dtype")
    elif dtype is None:
      raise ValueError(
          "At least one of dtype or flat_values_spec must be provided")
    self._dtype = dtypes.as_dtype(dtype)
    self._flat_values_spec = flat_values_spec

    rank = self._shape.ndims
    if ragged_rank is None:
      if rank is None:
        raise ValueError("Must specify ragged_rank or "
                         "a shape with a known rank.")
      ragged_rank = rank - 1
    self._ragged_rank = ragged_rank
    if not isinstance(self._ragged_rank, int):
      raise TypeError(f"Argument `ragged_rank` must be an int. "
                      f"Received {ragged_rank}.")

    if rank is not None:
      if ragged_rank >= rank:
        raise ValueError(f"Argument `ragged_rank` ({ragged_rank}) must be less "
                         f"than rank ({rank}).")

  def is_compatible_with(self, spec_or_value):
    # RaggedTensor with ragged_rank 0 can be compatible with raw flat_values.
    if self._ragged_rank == 0:
      if self._flat_values_spec is None:
        if isinstance(spec_or_value, (ops.Tensor, tensor_spec.TensorSpec)):
          return tensor_spec.TensorSpec(
              self._shape, self._dtype).is_compatible_with(spec_or_value)
      elif not isinstance(spec_or_value, (RaggedTensor, RaggedTensorSpec)):
        return self._flat_values_spec.is_compatible_with(spec_or_value)
    return super(RaggedTensorSpec, self).is_compatible_with(spec_or_value)

  def _serialize(self):
    if self._flat_values_spec is None:
      return (self._shape, self._dtype, self._ragged_rank,
              self._row_splits_dtype)
    else:
      return (self._shape, self._dtype, self._ragged_rank,
              self._row_splits_dtype, self._flat_values_spec)

  @property
  def _component_specs(self):
    if self._ragged_rank <= 0:
      if self._flat_values_spec is not None:
        return [self._flat_values_spec]
      else:
        return [tensor_spec.TensorSpec(self._shape, self._dtype)]

    flat_values_spec = self._flat_values_spec
    if flat_values_spec is None:
      flat_values_shape = tensor_shape.TensorShape([None]).concatenate(
          self._shape[self._ragged_rank + 1:])
      flat_values_spec = tensor_spec.TensorSpec(flat_values_shape, self._dtype)
    outer_dim = tensor_shape.dimension_at_index(self._shape, 0)
    outer_splits_shape = [None if outer_dim is None else outer_dim + 1]
    inner_splits_spec = tensor_spec.TensorSpec([None], self._row_splits_dtype)

    specs = ([
        flat_values_spec,
        tensor_spec.TensorSpec(outer_splits_shape, self._row_splits_dtype)
    ] + [inner_splits_spec for _ in range(self._ragged_rank - 1)])
    return specs

  def _to_components(self, value):
    if is_ragged(value):
      return [value.flat_values] + list(value.nested_row_splits)
    else:
      return [value]

  def _from_components(self, tensor_list):
    result = tensor_list[0]
    if (all(isinstance(t, np.ndarray) for t in tensor_list) and
        not tf2.enabled()):
      for row_splits in reversed(tensor_list[1:]):
        result = ragged_tensor_value.RaggedTensorValue(result, row_splits)
    else:
      if isinstance(tensor_list[0], np.ndarray):
        tensor_list = [ops.convert_to_tensor(t) for t in tensor_list]
        result = tensor_list[0]
      for row_splits in reversed(tensor_list[1:]):
        result = RaggedTensor(
            result,
            RowPartition.from_row_splits(row_splits, validate=False),
            internal=True)
    if self._shape.ndims is not None:
      if isinstance(result, RaggedTensor):
        result._set_shape(self._shape)  # pylint: disable=protected-access
        # TODO(xjun): MaskedTensor doesn't implement set_shape.
        if self.flat_values_spec is not None and hasattr(result.flat_values,
                                                         "set_shape"):
          result.flat_values.set_shape(self.flat_values_spec.shape)
      elif isinstance(result, ops.Tensor):
        result.set_shape(self._shape)
    return result

  # The RaggedTensorSpec tensor_list encoding uses to/from_variant ops
  # to (un)box the component tensors in a way that allows for batching &
  # unbatching.
  @property
  def _flat_tensor_specs(self):
    # NOTE(mishragaurav): The default flat shape of a boxed `RaggedTensor` is
    # `[]` (scalar), but a `RaggedTensorSpec` can also represent a batch of
    # boxed `RaggedTensor` objects with shape `(...)` (and batches of batches,
    # etc.), so the flat shape must be unknown.
    return [tensor_spec.TensorSpec(None, dtypes.variant)]

  def _to_tensor_list(self, value):
    # TODO(edloper): Update gen_ragged_conversion_ops that convert to and
    # from variant to include all of the row-partitioning tensors.
    if self._flat_values_spec is not None:
      raise ValueError("Customized value_type is not supported.")
    if isinstance(value, RaggedTensor):
      if value.ragged_rank != self._ragged_rank:
        raise ValueError(
            f"Ragged rank of value {value.ragged_rank} does not match "
            f"ragged rank of type {self._ragged_rank}.")
      # pylint: disable=protected-access
      return [value._to_variant(batched_input=False)]
    else:
      if self._ragged_rank > 0:
        raise ValueError(
            f"Expected a RaggedTensor if ragged rank={self._ragged_rank}"
            f" but got {type(value).__name__}."
        )
      return [
          gen_ragged_conversion_ops.ragged_tensor_to_variant(
              (), value, batched_input=False)
      ]

  def _to_batched_tensor_list(self, value):
    if self._flat_values_spec is not None:
      raise ValueError("Customized value_type is not supported.")
    if isinstance(value, RaggedTensor):
      if value.ragged_rank != self._ragged_rank:
        raise ValueError(
            f"Ragged rank of value {value.ragged_rank} does not match "
            f"ragged rank of type {self._ragged_rank}.")
      # pylint: disable=protected-access
      return [value._to_variant(batched_input=True)]
    else:
      if self._ragged_rank > 0:
        raise ValueError(
            f"Expected a RaggedTensor if ragged rank={self._ragged_rank}"
            f" but got {type(value).__name__}."
        )
      return [
          gen_ragged_conversion_ops.ragged_tensor_to_variant(
              rt_nested_splits=(), rt_dense_values=value, batched_input=True)
      ]

  def _from_compatible_tensor_list(self, tensor_list):
    if self._flat_values_spec is not None:
      raise ValueError("Customized value_type is not supported.")
    result = RaggedTensor._from_variant(  # pylint: disable=protected-access
        tensor_list[0],
        dtype=self._dtype,
        row_splits_dtype=self._row_splits_dtype,
        output_ragged_rank=self._ragged_rank)
    if self._shape.ndims is not None:
      if isinstance(result, RaggedTensor):
        result._set_shape(self._shape)  # pylint: disable=protected-access
        # TODO(xjun): MaskedTensor doesn't implement set_shape.
        if self.flat_values_spec is not None and hasattr(self.flat_values,
                                                         "set_shape"):
          result.flat_values.set_shape(self.flat_values_spec.shape)
      else:
        result.set_shape(self._shape)
    return result

  def _batch(self, batch_size):
    if self._flat_values_spec is not None:
      raise ValueError("Customized value_type is not supported.")
    return RaggedTensorSpec(
        tensor_shape.TensorShape([batch_size]).concatenate(self._shape),
        self._dtype, self._ragged_rank + 1, self._row_splits_dtype)

  def _unbatch(self):
    if self._flat_values_spec is not None:
      raise ValueError("Customized value_type is not supported.")
    # Note: Negative ragged_rank is allowed here because the dataset could be
    # subsequently batched again. If ragged_rank > 1, assume row_splits_dtype is
    # consistent. Errors are handled in
    # RaggedTensorSpec._from_compatible_tensor_list()
    return RaggedTensorSpec(self._shape[1:], self._dtype, self._ragged_rank - 1,
                            self._row_splits_dtype)

  def _to_legacy_output_types(self):
    return self._dtype

  def _to_legacy_output_shapes(self):
    return self._shape

  def _to_legacy_output_classes(self):
    return self

  @classmethod
  def from_value(cls, value):
    if (isinstance(value, ragged_tensor_value.RaggedTensorValue) or
        isinstance(value.flat_values, ops.Tensor)):
      return cls(
          shape=value.shape,
          dtype=value.values.dtype,
          ragged_rank=value.ragged_rank,
          row_splits_dtype=value.row_splits.dtype)
    else:
      flat_values_spec = type_spec.type_spec_from_value(value.flat_values)
      # Relax shape[0] to None, as it is connected to dynamic ragged shapes.
      flat_values_spec = flat_values_spec._unbatch()._batch(None)  # pylint: disable=protected-access
      return cls(
          shape=value.shape,
          dtype=value.values.dtype,
          ragged_rank=value.ragged_rank,
          row_splits_dtype=value.row_splits.dtype,
          flat_values_spec=flat_values_spec)


type_spec.register_type_spec_from_value_converter(
    ragged_tensor_value.RaggedTensorValue, RaggedTensorSpec.from_value)


#===============================================================================
# Convert value -> tensor
#===============================================================================
def convert_to_tensor_or_ragged_tensor(value,
                                       dtype=None,
                                       preferred_dtype=None,
                                       name=None):
  """Converts value to a `RaggedTensor` or `Tensor`.

  * If `value` is a `RaggedTensor`, then return it as-is.
  * If `value` is a `RaggedTensorValue`, return a corresponding constant
    `RaggedTensor`.
  * Otherwise, use `convert_to_tensor` to convert `value` to a `Tensor`.

  Args:
    value: A `RaggedTensor`, a `RaggedTensorValue`, or an object whose type has
      a registered `Tensor` conversion function.
    dtype: Optional element type for the returned tensor.  If missing the type
      is inferred from the type of `value`.
    preferred_dtype: Optional element type for the returned tensor, used when
      dtype is None.  This argument has no effect if `value` is already a
      tensor, or when conversion is not possible.
    name: Optional name to use if a new `Tensor` is created.

  Returns:
    A `Tensor` or `RaggedTensor`.
  """
  if isinstance(value, RaggedTensor):
    if dtype and not dtype.is_compatible_with(value.dtype):
      raise ValueError(f"Tensor conversion requested dtype {dtype.name} for "
                       f"RaggedTensor with dtype {value.dtype.name}: {value}.")
    return value
  elif isinstance(value, ragged_tensor_value.RaggedTensorValue):
    with ops.name_scope(name, "ConvertToTensorOrRaggedTensor", []):
      flat_values = ops.convert_to_tensor(
          value=value.flat_values,
          dtype=dtype,
          dtype_hint=preferred_dtype,
          name="flat_values")
      return RaggedTensor.from_nested_row_splits(
          flat_values, value.nested_row_splits, validate=False)
  else:
    return ops.convert_to_tensor_v2_with_dispatch(
        value=value, dtype=dtype, dtype_hint=preferred_dtype, name=name)


def _convert_to_ragged_tensor_values(value):
  """Converts value to supported RaggedTensor value.

  * If `value` is an object of supported value type, then return it as-is.
  * Otherwise convert it to Tensor or RaggedTensor.

  Args:
    value: An object of `Tensor`, `RaggedTensor` or registerred RaggedTensor
      value types, or an object whose type has a registered `Tensor` conversion
      function.

  Returns:
    An object of `Tensor`, `RaggedTensor` or registerred RaggedTensor
    value types
  """
  if _is_supported_ragged_values_type(value):
    return value
  else:
    return convert_to_tensor_or_ragged_tensor(value, name="values")


#===============================================================================
# Register RaggedTensor for use with session.run.
#===============================================================================
def _ragged_tensor_value_from_components(components):
  components = list(components)
  value = components.pop()
  while components:
    value = ragged_tensor_value.RaggedTensorValue(value, components.pop())
  return value


def _ragged_tensor_session_fetch(rt):
  components = rt.nested_row_splits + (rt.flat_values,)
  return (components, _ragged_tensor_value_from_components)


def _ragged_tensor_session_feed(feed_key, feed_val):
  key_components = feed_key.nested_row_splits + (feed_key.flat_values,)
  val_components = feed_val.nested_row_splits + (feed_val.flat_values,)
  return zip(key_components, val_components)


def _ragged_tensor_session_feed_for_partial_run(feed_key):
  return feed_key.nested_row_splits + (feed_key.flat_values,)


session.register_session_run_conversion_functions(
    RaggedTensor, _ragged_tensor_session_fetch, _ragged_tensor_session_feed,
    _ragged_tensor_session_feed_for_partial_run)


#===============================================================================
# RaggedTensorType
#===============================================================================
class RaggedTensorType:
  """Encoding of a static type for a `RaggedTensor`.

  Use this type to express/declare that an output must have the type of
  `RaggedTensor`.
  """

  def __init__(self, dtype, ragged_rank, row_splits_dtype=dtypes.int64):
    """Initializes a RaggedTensorType object.

    Args:
      dtype: data type of the `RaggedTensor`'s inner values.
      ragged_rank: ragged_rank of the declared `RaggedTensor`.
      row_splits_dtype: data type for the `RaggedTensor`'s row splits.
        One of: `tf.int32` or `tf.int64`.
    """
    row_splits_dtype = dtypes.as_dtype(row_splits_dtype)
    self._dtype = dtype
    self._ragged_rank = ragged_rank
    self._row_splits_dtype = row_splits_dtype

  dtype = property(lambda self: self._dtype)
  ragged_rank = property(lambda self: self._ragged_rank)
  row_splits_dtype = property(lambda self: self._row_splits_dtype)

  def __repr__(self):
    return "RaggedTensorType(%r, %r, %r)" % (self.dtype, self.ragged_rank,
                                             self.row_splits_dtype)


#===============================================================================
# Helper Functions
#===============================================================================
def _assert_sparse_indices_are_ragged_right(indices):
  """Checks that the given SparseTensor.indices tensor is ragged-right.

  Example: `indices = [[0, 0], [0, 1], [2, 0], [3, 1]]` is not ragged right
  because the entry `[3, 1]` skips a cell.

  Args:
    indices: The SparseTensor indices to check.

  Returns:
    A list of control dependency op tensors.
  """
  index_prefix = indices[:, :-1]
  index_suffix = indices[:, -1]

  # Check whether each index is starting a new row in the innermost dimension
  # (prefix[i] != prefix[i-1]) or continuing a row (prefix[i] == prefix[i-1]).
  # (Note: this skips the first index; we will check that separately below.)
  index_prefix_changed = math_ops.reduce_any(
      math_ops.not_equal(index_prefix[1:], index_prefix[:-1]), axis=1)

  # Check two cases:
  #   * For indices that start a new row: index_suffix[i] must be zero.
  #   * For indices that continue a row: index_suffix[i] must be equal to
  #     index_suffix[i-1]+1.
  index_ok = array_ops.where(
      index_prefix_changed, math_ops.equal(index_suffix[1:], 0),
      math_ops.equal(index_suffix[1:], index_suffix[:-1] + 1))

  # Also check that the very first index didn't skip any cells.  The first
  # index starts a new row (by definition), so its suffix should be zero.
  sparse_indices_are_ragged_right = math_ops.logical_and(
      math_ops.reduce_all(math_ops.equal(index_suffix[:1], 0)),
      math_ops.reduce_all(index_ok))

  message = [
      "SparseTensor is not right-ragged", "SparseTensor.indices =", indices
  ]
  return [control_flow_ops.Assert(sparse_indices_are_ragged_right, message)]


@ops.RegisterGradient("RaggedTensorToSparse")
def _ragged_tensor_to_sparse_gradient(op, unused_sparse_indices_grad,
                                      sparse_values_grad,
                                      unused_sparse_shape_grad):
  """Gradient for RaggedTensorToSparse."""
  op_inputs_nested_row_splits = op.inputs[:-1]
  op_inputs_flat_values = op.inputs[-1]

  # No gradient for the RaggedTensor's nested_row_splits.
  nested_row_splits_gradient = [None] * len(op_inputs_nested_row_splits)

  # Gradient for the RaggedTensor's flat_values is formed by reshaping
  # the gradient for the SparseTensor's values.
  flat_values_shape = array_ops.shape(op_inputs_flat_values)
  flat_values_gradient = array_ops.reshape(sparse_values_grad,
                                           flat_values_shape)

  return nested_row_splits_gradient + [flat_values_gradient]


def _assert_monotonic_increasing(tensor, message=None):
  return check_ops.assert_non_negative(
      tensor[1:] - tensor[:-1], message=message)


def _assert_zero(tensor, message=None):
  return check_ops.assert_equal(
      tensor, constant_op.constant(0, dtype=tensor.dtype), message=message)


def _nrows(tensor, out_type=dtypes.int32):
  if isinstance(tensor, RaggedTensor):
    return tensor.nrows(out_type=out_type)
  else:
    return array_ops.shape(tensor, out_type=out_type)[0]


def merge_dims(value, outer_axis, inner_axis):
  """Merges value[outer_axis...inner_axis] into a single dimension.

  See `RaggedTensor.merge_dims()` for more details.  This helper differs from
  `RaggedTensor.merge_dims()` in that `value` may be a dense or ragged tensor.

  Args:
    value: A `RaggedTensor` or `Tensor`
    outer_axis: `int`
    inner_axis: `int`

  Returns:
    A flattened `RaggedTensor` or `Tensor`.
  """
  if outer_axis == inner_axis:
    return value

  # Flatten outer dimensions of a RaggedTensor by just taking its values.
  while outer_axis == 0 and isinstance(value, RaggedTensor):
    value = value.values
    inner_axis -= 1
    if inner_axis == 0:
      return value

  # Flatten non-Ragged tensors using tf.reshape().
  if not isinstance(value, RaggedTensor):
    if value.shape.is_fully_defined():
      old_shape = value.shape.as_list()
      new_shape = old_shape[:outer_axis] + [-1] + old_shape[inner_axis + 1:]
    else:
      old_shape = array_ops.shape(value)
      new_shape = array_ops.concat(
          [old_shape[:outer_axis], [-1], old_shape[inner_axis + 1:]], axis=0)
    return array_ops.reshape(value, new_shape)

  # Handle outer_axis>1 via recursion.
  if outer_axis > 1:
    return value.with_values(
        merge_dims(value.values, outer_axis - 1, inner_axis - 1))

  # At this point, we know outer_axis == 1, and value is a RaggedTensor.
  # So we need to flatten the values and build a corresponding splits tensor.
  new_values = value.values
  new_splits = value.row_splits
  for axis in range(outer_axis, inner_axis):
    if isinstance(new_values, RaggedTensor):
      # Flatten a single ragged dimension.
      new_splits = array_ops.gather(new_values.row_splits, new_splits)
      new_values = new_values.values
    else:
      # Flatten all remaining dense dimensions.
      shape_split = inner_axis - axis + 1
      if new_values.shape.is_fully_defined():
        old_shape = new_values.shape.as_list()
        new_shape = [-1] + old_shape[shape_split:]
        flat_size = _prod(old_shape[1:shape_split])
      else:
        old_shape = array_ops.shape(new_values)
        new_shape = array_ops.concat([[-1], old_shape[shape_split:]], axis=0)
        flat_size = math_ops.cast(
            math_ops.reduce_prod(old_shape[1:shape_split]), new_splits.dtype)
      new_values = array_ops.reshape(new_values, new_shape)
      new_splits = new_splits * flat_size
      break
  return RaggedTensor.from_row_splits(new_values, new_splits)


def _prod(lst):
  """Returns the product of the numbers in a list."""
  return functools.reduce(operator.mul, lst, 1)


def _get_row_partition_type_tensor_pairs_tail(partition):
  """Gets a row partition type tensor pair for the tail.

  If value_rowid is defined, then it is used. Otherwise, row_splits
  are used.

  Args:
    partition: a RowPartition.

  Returns:
    A list of (row_partition_type, row_partition_tensor) pairs.
  """
  if partition._has_precomputed_value_rowids():  # pylint: disable=protected-access
    return ("VALUE_ROWIDS", partition.value_rowids())
  else:
    return ("ROW_SPLITS", partition.row_splits())


def _get_row_partition_type_tensor_pairs(rt_input):
  """Gets a list of the row partitions for rt_input.

  If value_rowids are defined, then they are used. Otherwise, row_splits
  are used. If the outermost level has value_rowids defind, then nrows is
  also added.

  Args:
    rt_input: a ragged tensor.

  Returns:
    A list of (row_partition_type, row_partition_tensor) pairs.
  """
  partitions = rt_input._nested_row_partitions  # pylint: disable=protected-access
  tail = [_get_row_partition_type_tensor_pairs_tail(x) for x in partitions[1:]]

  if partitions[0]._value_rowids is not None:  # pylint: disable=protected-access
    return [("FIRST_DIM_SIZE", partitions[0].nrows()),
            ("VALUE_ROWIDS", partitions[0].value_rowids())] + tail
  else:
    return [("ROW_SPLITS", partitions[0].row_splits())] + tail


def _shape_as_tensor(shape, dtype):
  """Takes shape and coerces it to a shape as a tensor.

  If the object is already a tensor, simply passes it on (result is guaranteed
  to be int64 or int32, but not necessarily dtype).
  If not, creates a tensor of type dtype.

  Result is either a scalar equal to -1 if the shape is unknown_rank.
  Otherwise, it is a vector, where unknown dimensions are represented with a
  value of -1.

  In C++, see TensorShapeFromTensor for parsing shapes in kernels, and
  InferenceContext::MakeShapeFromShapeTensorTreatScalarAsUnknownShape, for
  use in the shape inference function.

  Args:
    shape: input to coerce from TensorShape, Tensor, None, List[Optional[Int]],
      Tuple[Optional[Int]].
    dtype: tf.int64 or tf.int32

  Returns:
    a scalar or vector tensor of dtype tf.int32 or tf.int64.
  """
  if dtype != dtypes.int64 and dtype != dtypes.int32:
    raise ValueError(f"Expected int64 or int32 for dtype: got {dtype}.")

  if isinstance(shape, ops.Tensor):
    if shape.dtype != dtypes.int64 and shape.dtype != dtypes.int32:
      return math_ops.cast(shape, dtype)
    return shape
  shape = tensor_shape.as_shape(shape)
  if not shape:
    # Imply rank is unknown using a -1 scalar.
    return constant_op.constant(-1, dtype=dtype)
  shape = [(-1 if x is None else x) for x in shape.as_list()]
  # At this point, shape is List[Int].
  return constant_op.constant(shape, dtype=dtype)


def _nvals_uniform_row_length(values, uniform_row_length):
  """Get the number of values for uniform row length constructor."""
  const_nvals = tensor_shape.dimension_at_index(values.shape, 0).value
  if const_nvals is not None:
    nvals = constant_op.constant(const_nvals, uniform_row_length.dtype)
  elif isinstance(values, RaggedTensor):
    nvals = values.nrows(out_type=uniform_row_length.dtype)
  else:
    nvals = array_ops.shape(values, out_type=uniform_row_length.dtype)[0]
  return nvals


def _get_optional_partition_dtype(values):
  """Returns the partition dtype, or None if None exists."""
  if isinstance(values, RaggedTensor):
    # pylint: disable=protected-access
    return values._row_partition.dtype
  return None


_SUPPORTED_RAGGED_VALUE_TYPES = (ops.Tensor, RaggedTensor)


# TODO(edloper): Consider whether we should change the registry to be on
# TypeSpecs rather than ValueTypes.
def _add_supported_value_type(cls):
  """Register the `cls` as supported value type of RaggedTenosr.

  The cls must be a subclass of CompositeTensor, and must support:
   - Spec:
     The Spec must be a `BatchableTypeSpec`
   - Properties:
     - x.shape
     - x.dtype
   - Methods:
     - x.__getitem__(idx) (method: returns a supported value type)
     - x.set_shape(shape)
   - Ops:
     - tf.shape(x) -- tf.shape(x)[0] must be a tf.Tensor.
     - tf.tile(x)
     - assert_rank_at_least(x)
     - tf.ones_like(x)
     - tf.gather(params=x, indices=Tensor)
     - tf.add(x, y)
     - tf.boolean_mask(x, ...)
     - @TODO(edloper): Complete this list

   Note: the following RaggedTensor, RaggedTensorSpec methods & ops are not
   currently supported unless `rt.values` is a RaggedTensor or a tf.Tensor:
     - rt.to_tensor()
     - rt.to_sparse_tensor()
     - rt._to_variant()
     - rt._from_variant()
     - tf.ragged.cross([rt])
     - tf.gather(params=x, indices=rt)  # rt used for indices
     - RaggedTensorSpec methods:
       - _batch
       - _unbatch
       - _to_tensor_list
       - _to_batched_tensor_list
       - _from_compatible_tensor_list

  Args:
    cls: The type to be added to supported value types.
  """
  if not issubclass(cls, composite_tensor.CompositeTensor):
    raise ValueError(f"cls ({cls}) must be a subclass of CompositeTensor.")
  if not hasattr(cls, "shape"):
    raise ValueError("cls must support the `shape` property.")
  if not hasattr(cls, "dtype"):
    raise ValueError("cls must support the `dtype` property.")
  global _SUPPORTED_RAGGED_VALUE_TYPES
  _SUPPORTED_RAGGED_VALUE_TYPES += (cls,)


def _is_supported_ragged_values_type(value):
  return isinstance(value, _SUPPORTED_RAGGED_VALUE_TYPES)


def _assert_is_supported_ragged_values_type(value):
  if not _is_supported_ragged_values_type(value):
    ok_types = ", ".join(cls.__name__ for cls in _SUPPORTED_RAGGED_VALUE_TYPES)
    raise TypeError(f"type(values) must be one of: {ok_types}, got {value}.")


def _formatter(x):
  """Separate Numpy array elements with comma."""
  if isinstance(x, np.ndarray):
    if x.size != 0:
      return np.array2string(x, separator=", ")
    else:
      # When x.size==0, np.array2string always returns `[]`.  This isn't always
      # what we want.  E.g., if `x.shape=[0, 3]`, then we want `[[], [], []]`.
      return repr(x.tolist())
  else:
    return str(x)

# Type annotation indicating that a value is ragged.  Includes RaggedTensor
# as well as the (deprecated) RaggedTensorValue class from TF 1.x.
Ragged = typing.Union[RaggedTensor, ragged_tensor_value.RaggedTensorValue]

# Type annotation indicating that a value is a ragged tensor, a dense tensor,
# or a value that can be converted to a tensor (e.g. np.array).
# TODO(edloper): Add Variable to TensorLike, and remove it from here.
RaggedOrDense = typing.Union[Ragged, core_types.TensorLike]
