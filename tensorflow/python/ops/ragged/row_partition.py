# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""A class used to partition a sequence into contiguous subsequences ("rows").
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import segment_id_ops

# pylint: disable=protected-access
_eval_using_default_session = ops._eval_using_default_session
# pylint: enable=protected-access

#===============================================================================
# RowPartition
#===============================================================================


class RowPartition(object):
  """Partitioning of a sequence of values into contiguous subsequences ("rows").

  A `RowPartition` describes how a sequence with `nvals` items should be
  divided into `nrows` contiguous subsequences ("rows").  For example, a
  `RowPartition` could be used to partition the vector `[1, 2, 3, 4, 5]` into
  subsequences `[[1, 2], [3], [], [4, 5]]`.  Note that `RowPartition` stores
  information about how values are partitioned, but does not include the
  partitioned values themselves.  `tf.RaggedTensor` is used to pair a `values`
  tensor with one or more `RowPartition`s, providing a complete encoding for a
  ragged tensor (i.e. a tensor with variable-length dimensions).

  `RowPartition`s may be defined using several different schemes:

    * `row_lengths`: an integer vector with shape `[nrows]`, which specifies
      the length of each row.

    * `row_splits`: an integer vector with shape `[nrows+1]`, specifying the
      "split points" between each row.

    * `row_starts`: an integer vector with shape `[nrows]`, which specifies
      the start offset for each row.  Equivalent to `row_splits[:-1]`.

    * `row_limits`: an integer vector with shape `[nrows]`, which specifies
      the stop offset for each row.  Equivalent to `row_splits[1:]`.

    * `value_rowids` is an integer vector with shape `[nvals]`, corresponding
      one-to-one with sequence values, which specifies the row that each value
      belongs to.  If the partition has empty trailing rows, then `nrows`
      must also be specified.

    * `uniform_row_length` is an integer scalar, specifying the length of every
      row.  This scheme may only be used if all rows have the same length.

  For example, the following `RowPartition`s all represent the partitioning of
  8 values into 5 sublists as follows: `[[*, *, *, *], [], [*, *, *], [*], []]`.

  >>> p1 = RowPartition.from_row_lengths([4, 0, 3, 1, 0])
  >>> p2 = RowPartition.from_row_splits([0, 4, 4, 7, 8, 8])
  >>> p3 = RowPartition.from_row_starts([0, 4, 4, 7, 8], nvals=8)
  >>> p4 = RowPartition.from_row_limits([4, 4, 7, 8, 8])
  >>> p5 = RowPartition.from_value_rowids([0, 0, 0, 0, 2, 2, 2, 3], nrows=5)

  For more information about each scheme, see the documentation for the
  its factory method.  For additional examples, see the documentation on
  `tf.RaggedTensor`.

  ### Precomputed Encodings

  `RowPartition` always stores at least one encoding of the partitioning, but
  it can be configured to cache additional encodings as well.  This can
  avoid unnecessary recomputation in eager mode.  (In graph mode, optimizations
  such as common subexpression elimination will typically prevent these
  unnecessary recomputations.)  To check which encodings are precomputed, use
  `RowPartition.has_precomputed_<encoding>`.
  """

  #=============================================================================
  # Constructor (private)
  #=============================================================================
  def __init__(self,
               row_splits,
               cached_row_lengths=None,
               cached_value_rowids=None,
               cached_nrows=None,
               internal=False,
               uniform_row_length=None):
    """Creates a `RowPartition` from the specified encoding tensor(s).

    This constructor is private -- please use one of the following ops to
    build `RowPartition`s:

      * `RowPartition.from_row_lengths`
      * `RowPartition.from_value_rowids`
      * `RowPartition.from_row_splits`
      * `RowPartition.from_row_starts`
      * `RowPartition.from_row_limits`

    Args:
      row_splits: A 1-D integer tensor with shape `[nrows+1]`.
      cached_row_lengths: A 1-D integer tensor with shape `[nrows]`
      cached_value_rowids: A 1-D integer tensor with shape `[nvals]`.
      cached_nrows: A 1-D integer scalar tensor.
      internal: True if the constructor is being called by one of the factory
        methods.  If false, an exception will be raised.
      uniform_row_length: A scalar tensor.

    Raises:
      TypeError: If a row partitioning tensor has an inappropriate dtype.
      TypeError: If exactly one row partitioning argument was not specified.
      ValueError: If a row partitioning tensor has an inappropriate shape.
      ValueError: If multiple partitioning arguments are specified.
      ValueError: If nrows is specified but value_rowids is not None.
    """
    if not internal:
      raise ValueError("RaggedTensor constructor is private; please use one "
                       "of the factory methods instead (e.g., "
                       "RaggedTensor.from_row_lengths())")

    # Validate the arguments.
    if not isinstance(row_splits, ops.Tensor):
      raise TypeError("Row-partitioning argument must be a Tensor, got %r" %
                      row_splits)
    if row_splits.dtype not in (dtypes.int32, dtypes.int64):
      raise ValueError("Row-partitioning argument must be int32 or int64")

    # Validate shapes & dtypes.
    row_splits.shape.assert_has_rank(1)
    row_splits.set_shape([None])
    self._row_splits = row_splits

    # Store any cached tensors.  These are used to avoid unnecessary
    # round-trip conversions when a RaggedTensor is constructed from
    # lengths or rowids, and we later want those lengths/rowids back.
    for tensor in [cached_row_lengths, cached_value_rowids, cached_nrows]:
      if tensor is not None:
        if not isinstance(tensor, ops.Tensor):
          raise TypeError("Cached value must be a Tensor or None.")
        elif tensor.dtype not in (dtypes.int32, dtypes.int64):
          raise TypeError("Cached value must be int32 or int64.")
    self._cached_row_lengths = cached_row_lengths
    self._cached_value_rowids = cached_value_rowids
    self._cached_nrows = cached_nrows

    if uniform_row_length is not None:
      if not isinstance(uniform_row_length, ops.Tensor):
        raise TypeError("uniform_row_length must be a Tensor or None.")
      elif uniform_row_length.dtype not in (dtypes.int32, dtypes.int64):
        raise TypeError("uniform_row_length must be int32 or int64.")
    self._uniform_row_length = uniform_row_length

  #=============================================================================
  # Factory Methods
  #=============================================================================

  @classmethod
  def from_value_rowids(cls,
                        value_rowids,
                        nrows=None,
                        name=None,
                        validate=True,
                        preferred_dtype=None):
    """Creates a `RowPartition` with rows partitioned by `value_rowids`.

    This `RowPartition` divides a sequence `values` into rows by specifying
    which row each value should be added to:

    ```python
    rows = [[] for _ in nrows]
    for (value, rowid) in zip(values, value_rowids):
      rows[rowid].append(value)
    ``

    Args:
      value_rowids: A 1-D integer tensor with shape `[nvals]`, which corresponds
        one-to-one with `values`, and specifies each value's row index.  Must be
        nonnegative, and must be sorted in ascending order.
      nrows: An integer scalar specifying the number of rows.  This should be
        specified if the `RowPartition` may containing empty training rows. Must
        be greater than `value_rowids[-1]` (or greater than or equal to zero if
        `value_rowids` is empty). Defaults to `value_rowids[-1]` (or zero if
        `value_rowids` is empty).
      name: A name prefix for the RaggedTensor (optional).
      validate: If true, then use assertions to check that the arguments form a
        valid `RowPartition`.
      preferred_dtype: The dtype to encode value_rowids if it doesn't already
        have one. The default is tf.int64.

    Returns:
      A `RowPartition`.

    Raises:
      ValueError: If `nrows` is incompatible with `value_rowids`.

    #### Example:

    >>> print(RowPartition.from_value_rowids(
    ...     value_rowids=[0, 0, 0, 0, 2, 2, 2, 3],
    ...     nrows=4))
    tf.RowPartition(row_splits=tf.Tensor([0 4 4 7 8], shape=(5,), dtype=int64))
    """
    if not isinstance(validate, bool):
      raise TypeError("validate must have type bool")
    with ops.name_scope(name, "RowPartitionFromValueRowIds",
                        [value_rowids, nrows]):
      value_rowids = cls._convert_row_partition(value_rowids, "value_rowids",
                                                preferred_dtype)
      if nrows is None:
        const_rowids = tensor_util.constant_value(value_rowids)
        if const_rowids is None:
          nrows = array_ops.concat([value_rowids[-1:], [-1]], axis=0)[0] + 1
          const_nrows = None
        else:
          const_nrows = const_rowids[-1] + 1 if const_rowids.size > 0 else 0
          nrows = ops.convert_to_tensor(
              const_nrows, value_rowids.dtype, name="nrows")
      else:
        nrows = ops.convert_to_tensor(nrows, value_rowids.dtype, "nrows")
        const_nrows = tensor_util.constant_value(nrows)
        if const_nrows is not None:
          if const_nrows < 0:
            raise ValueError("Expected nrows >= 0; got %d" % const_nrows)
          const_rowids = tensor_util.constant_value(value_rowids)
          if const_rowids is not None and const_rowids.size > 0:
            if not const_nrows >= const_rowids[-1] + 1:
              raise ValueError(
                  "Expected nrows >= value_rowids[-1] + 1; got nrows=%d, "
                  "value_rowids[-1]=%d" % (const_nrows, const_rowids[-1]))

      value_rowids.shape.assert_has_rank(1)
      nrows.shape.assert_has_rank(0)

      if validate:
        msg = ("Arguments to from_value_rowids do not form a valid "
               "RowPartition")
        checks = [
            check_ops.assert_rank(value_rowids, 1, message=msg),
            check_ops.assert_rank(nrows, 0, message=msg),
            check_ops.assert_non_negative(value_rowids[:1], message=msg),
            _assert_monotonic_increasing(value_rowids, message=msg),
            check_ops.assert_less(value_rowids[-1:], nrows, message=msg),
        ]
        value_rowids = control_flow_ops.with_dependencies(checks, value_rowids)

      # Convert value_rowids & nrows to row_splits.
      # Note: we don't use segment_ids_to_row_splits() here because we want
      # to save the intermediate value `row_lengths`, so we can cache it.
      # TODO(b/116708836) Upgrade bincount to accept int64 so we can skip the
      # cast.
      value_rowids_int32 = math_ops.cast(value_rowids, dtypes.int32)
      nrows_int32 = math_ops.cast(nrows, dtypes.int32)
      row_lengths = math_ops.bincount(
          value_rowids_int32,
          minlength=nrows_int32,
          maxlength=nrows_int32,
          dtype=value_rowids.dtype)
      row_splits = array_ops.concat([[0], math_ops.cumsum(row_lengths)], axis=0)
      if const_nrows is not None:
        row_lengths.set_shape([const_nrows])
        row_splits.set_shape([const_nrows + 1])

      return cls(
          row_splits,
          cached_row_lengths=row_lengths,
          cached_value_rowids=value_rowids,
          cached_nrows=nrows,
          internal=True)

  @classmethod
  def from_row_splits(cls,
                      row_splits,
                      name=None,
                      validate=True,
                      preferred_dtype=None):
    """Creates a `RowPartition` with rows partitioned by `row_splits`.

    This `RowPartition` divides a sequence `values` into rows by indicating
    where each row begins and ends:

    ```python
    rows = []
    for i in range(len(row_splits) - 1):
      row_start = row_splits[i]
      row_end = row_splits[i + 1]
      rows.append(values[row_start:row_end])
    ```

    Args:
      row_splits: A 1-D integer tensor with shape `[nrows+1]`.  Must not be
        empty, and must be sorted in ascending order.  `row_splits[0]` must be
        zero.
      name: A name prefix for the RaggedTensor (optional).
      validate: If true, then use assertions to check that the arguments form a
        valid `RowPartition`.
      preferred_dtype: If row_splits has an unspecified type, use this one. If
        preferred_dtype is None, defaults to dtypes.int64.

    Returns:
      A `RowPartition`.

    Raises:
      ValueError: If `row_splits` is an empty list.
    """
    if not isinstance(validate, bool):
      raise TypeError("validate must have type bool")
    if isinstance(row_splits, (list, tuple)) and not row_splits:
      raise ValueError("row_splits tensor may not be empty.")
    if isinstance(row_splits, tensor_spec.TensorSpec):
      return cls(row_splits=row_splits, internal=True)

    with ops.name_scope(name, "RowPartitionFromRowSplits", [row_splits]):
      row_splits = cls._convert_row_partition(row_splits, "row_splits",
                                              preferred_dtype)
      row_splits.shape.assert_has_rank(1)

      if validate:
        msg = "Arguments to from_row_splits do not form a valid RaggedTensor:"
        checks = [
            check_ops.assert_rank(row_splits, 1, message=(msg + "rank")),
            _assert_zero(row_splits[0], message=(msg + "zero")),
            _assert_monotonic_increasing(
                row_splits, message=(msg + "monotonic")),
        ]
        row_splits = control_flow_ops.with_dependencies(checks, row_splits)

      return cls(row_splits=row_splits, internal=True)

  @classmethod
  def from_row_lengths(cls,
                       row_lengths,
                       name=None,
                       validate=True,
                       preferred_dtype=None):
    """Creates a `RowPartition` with rows partitioned by `row_lengths`.

    This `RowPartition` divides a sequence `values` into rows by indicating
    the length of each row:

    ```python
    rows = [[values.pop(0) for _ in range(length)]
            for length in row_lengths]
    ```

    Args:
      row_lengths: A 1-D integer tensor with shape `[nrows]`.  Must be
        nonnegative.
      name: A name prefix for the RowPartition (optional).
      validate: If true, then use assertions to check that the arguments form a
        valid `RowPartition`.
      preferred_dtype: If row_lengths has an unspecified type, use this one. If
        preferred_dtype is None, defaults to dtypes.int64.

    Returns:
      A `RowPartition`.
    """
    if not isinstance(validate, bool):
      raise TypeError("validate must have type bool")
    with ops.name_scope(name, "RowPartitionFromRowLengths", [row_lengths]):
      row_lengths = cls._convert_row_partition(row_lengths, "row_lengths",
                                               preferred_dtype)
      row_lengths.shape.assert_has_rank(1)

      if validate:
        msg = "Arguments to from_row_lengths do not form a valid RowPartition"
        checks = [
            check_ops.assert_rank(row_lengths, 1, message=msg),
            check_ops.assert_non_negative(row_lengths, message=msg),
        ]
        row_lengths = control_flow_ops.with_dependencies(checks, row_lengths)

      row_limits = math_ops.cumsum(row_lengths)
      row_splits = array_ops.concat([[0], row_limits], axis=0)
      return cls(
          row_splits=row_splits, cached_row_lengths=row_lengths, internal=True)

  @classmethod
  def from_row_starts(cls,
                      row_starts,
                      nvals,
                      name=None,
                      validate=True,
                      preferred_dtype=None):
    """Creates a `RowPartition` with rows partitioned by `row_starts`.

    Equivalent to: `from_row_splits(concat([row_starts, nvals], axis=0))`.

    Args:
      row_starts: A 1-D integer tensor with shape `[nrows]`.  Must be
        nonnegative and sorted in ascending order.  If `nrows>0`, then
        `row_starts[0]` must be zero.
      nvals: A scalar tensor indicating the number of values.
      name: A name prefix for the RowPartition (optional).
      validate: If true, then use assertions to check that the arguments form a
        valid `RowPartition`.
      preferred_dtype: If row_limits has an unspecified type, use this one. If
        preferred_dtype is None, defaults to dtypes.int64.

    Returns:
      A `RowPartition`.
    """
    if not isinstance(validate, bool):
      raise TypeError("validate must have type bool")
    with ops.name_scope(name, "RowPartitionFromRowStarts", [row_starts]):
      row_starts = cls._convert_row_partition(row_starts, "row_starts",
                                              preferred_dtype)
      row_starts.shape.assert_has_rank(1)
      nvals = math_ops.cast(nvals, row_starts.dtype)
      if validate:
        msg = "Arguments to from_row_starts do not form a valid RaggedTensor"
        checks = [
            check_ops.assert_rank(row_starts, 1, message=msg),
            _assert_zero(row_starts[:1], message=msg),
            _assert_monotonic_increasing(row_starts, message=msg),
            check_ops.assert_less_equal(row_starts[-1:], nvals, message=msg),
        ]
        row_starts = control_flow_ops.with_dependencies(checks, row_starts)

      row_splits = array_ops.concat([row_starts, [nvals]], axis=0)
      return cls(row_splits=row_splits, internal=True)

  @classmethod
  def from_row_limits(cls,
                      row_limits,
                      name=None,
                      validate=True,
                      preferred_dtype=None):
    """Creates a `RowPartition` with rows partitioned by `row_limits`.

    Equivalent to: `from_row_splits(values, concat([0, row_limits], axis=0))`.

    Args:
      row_limits: A 1-D integer tensor with shape `[nrows]`.  Must be sorted in
        ascending order.
      name: A name prefix for the RaggedTensor (optional).
      validate: If true, then use assertions to check that the arguments form a
        valid `RowPartition`.
      preferred_dtype: If row_limits has an unspecified type, use this one. If
        preferred_dtype is None, defaults to dtypes.int64.

    Returns:
      A `RowPartition`.
    """
    if not isinstance(validate, bool):
      raise TypeError("validate must have type bool")
    with ops.name_scope(name, "RowPartitionFromRowLimits", [row_limits]):
      row_limits = cls._convert_row_partition(row_limits, "row_limits",
                                              preferred_dtype)
      row_limits.shape.assert_has_rank(1)

      if validate:
        msg = "Arguments to from_row_limits do not form a valid RaggedTensor"
        checks = [
            check_ops.assert_rank(row_limits, 1, message=msg),
            check_ops.assert_non_negative(row_limits[:1], message=msg),
            _assert_monotonic_increasing(row_limits, message=msg),
        ]
        row_limits = control_flow_ops.with_dependencies(checks, row_limits)

      zero = array_ops.zeros([1], row_limits.dtype)
      row_splits = array_ops.concat([zero, row_limits], axis=0)
      return cls(row_splits=row_splits, internal=True)

  @classmethod
  def from_uniform_row_length(cls,
                              nvals,
                              uniform_row_length,
                              nrows=None,
                              validate=True,
                              name=None,
                              preferred_dtype=None):
    """Creates a `RowPartition` with rows partitioned by `uniform_row_length`.

    This `RowPartition` divides a sequence `values` into rows that all have
    the same length:

    ```python
    nrows = [[values.pop(0) for _ in range(uniform_row_length)]
             for _ in range(nrows)]
    ```

    Args:
      nvals: a non-negative scalar integer tensor for the number of values.
      uniform_row_length: A scalar integer tensor.  Must be nonnegative. The
        size of the outer axis of `values` must be evenly divisible by
        `uniform_row_length`.
      nrows: The number of rows in the constructed RowPartition.  If not
        specified, then it defaults to `nvals/uniform_row_length` (or `0` if
        `uniform_row_length==0`).  `nrows` only needs to be specified if
        `uniform_row_length` might be zero.  `uniform_row_length*nrows` must be
        `nvals`.
      validate: If true, then use assertions to check that the arguments form a
        valid `RowPartition`.
      name: A name prefix for the RaggedTensor (optional)
      preferred_dtype: if uniform_row_length has no dtype, use this one.

    Returns:
      A `RowPartition`.
    """
    if not isinstance(validate, bool):
      raise TypeError("validate must have type bool")
    with ops.name_scope(name, "RowPartitionFromUniformRowLength",
                        [uniform_row_length, nrows]):
      uniform_row_length = cls._convert_row_partition(uniform_row_length,
                                                      "uniform_row_length",
                                                      preferred_dtype)
      uniform_row_length.shape.assert_has_rank(0)

      # Find nrows.
      const_row_length = tensor_util.constant_value(uniform_row_length)
      if nrows is None:
        if const_row_length is None:
          # Avoid division by zero if uniform_row_length==0 (and nvals==0).
          rowlen_or_1 = math_ops.maximum(
              uniform_row_length,
              constant_op.constant(1, uniform_row_length.dtype))
          nrows = nvals // rowlen_or_1
        elif const_row_length == 0:
          nrows = 0
        else:
          nrows = nvals // const_row_length
      nrows = ops.convert_to_tensor(
          nrows, uniform_row_length.dtype, name="nrows")
      const_nrows = tensor_util.constant_value(nrows)
      const_nvals = tensor_util.constant_value(nvals)

      # Find row_splits.
      if const_nrows is not None and const_row_length is not None:
        row_splits = [v * const_row_length for v in range(const_nrows + 1)]
        row_splits = constant_op.constant(row_splits, uniform_row_length.dtype)
      else:
        row_splits = math_ops.range(nrows + 1) * uniform_row_length

      if validate:
        checks = []

        if (const_nrows is None or const_row_length is None or
            const_nvals is None):
          checks.append(
              check_ops.assert_equal(
                  nrows * uniform_row_length, nvals,
                  ("uniform_row_length", uniform_row_length, "times nrows",
                   nrows, "must equal nvals", nvals)))
        else:
          if const_nrows * const_row_length != const_nvals:
            raise ValueError(
                "uniform_row_length=%d times nrows=%d must equal nvals=%d" %
                (const_row_length, const_nrows, const_nvals))

        if uniform_row_length.shape.rank is None:
          checks.append(
              check_ops.assert_rank(
                  uniform_row_length,
                  0,
                  message="uniform_row_length must be a scalar."))

        const_row_length = tensor_util.constant_value(uniform_row_length)
        if const_row_length is None:
          checks.append(
              check_ops.assert_greater_equal(
                  uniform_row_length,
                  constant_op.constant(0, uniform_row_length.dtype),
                  message="uniform_row_length must be >= 0."))
        else:
          if const_row_length < 0:
            raise ValueError("uniform_row_length must be >= 0.")

        row_splits = control_flow_ops.with_dependencies(checks, row_splits)

      return cls(
          row_splits=row_splits,
          uniform_row_length=uniform_row_length,
          cached_nrows=nrows,
          internal=True)

  @classmethod
  def _convert_row_partition(cls, partition, name, preferred_dtype):
    """Converts `partition` to Tensors.

    Args:
      partition: A row-partitioning tensor for the `RowPartition` being
        constructed.  I.e., one of: row_splits, row_lengths, row_starts,
          row_limits, value_rowids.
      name: The name of the row-partitioning tensor.
      preferred_dtype: If partition has no dtype, give it this one. If
      no dtype is specified, use dtypes.int64.

    Returns:
      A tensor equivalent to partition.

    Raises:
      ValueError: if dtype is not int32 or int64.
    """
    if preferred_dtype is None:
      preferred_dtype = dtypes.int64
    if isinstance(partition, np.ndarray) and partition.dtype == np.int32:
      partition = ops.convert_to_tensor(partition, name=name)
    else:
      partition = ops.convert_to_tensor(
          partition, preferred_dtype=preferred_dtype, name=name)
    if partition.dtype not in (dtypes.int32, dtypes.int64):
      raise ValueError("%s must have dtype int32 or int64" % name)

    return partition

  def with_dependencies(self, dependencies):
    """Returns a new RowPartition equal to self with control dependencies.

    Specifically, self._row_splits is gated by the given control dependencies.
    Used to add sanity checks to the constructors.

    Args:
      dependencies: a list of tensors to use as dependencies.

    Returns:
      A new RowPartition object.
    """
    new_row_splits = control_flow_ops.with_dependencies(dependencies,
                                                        self._row_splits)
    return RowPartition(
        row_splits=new_row_splits,
        cached_row_lengths=self._cached_row_lengths,
        cached_value_rowids=self._cached_value_rowids,
        cached_nrows=self._cached_nrows,
        internal=True,
        uniform_row_length=self._uniform_row_length)

  #=============================================================================
  # Accessors
  #=============================================================================

  @property
  def dtype(self):
    """The `DType` used to encode the row partition (either int32 or int64)."""
    return self._row_splits.dtype

  # TODO(edloper): Change this to a method (not property) for consistency
  # with the other accessors.
  @property
  def row_splits(self):
    """Returns the row-split indices for this row partition.

    `row_splits` specifies where the values for each row begin and end.
    In particular, the values for row `i` are stored in the slice
    `values[row_splits[i]:row_splits[i+1]]`.

    Returns:
      A 1-D integer `Tensor` with shape `[self.nrows+1]`.
      The returned tensor is non-empty, and is sorted in ascending order.
      `self.row_splits()[0] == 0`.
      `self.row_splits()[-1] == self.nvals()`.
    """
    return self._row_splits

  def value_rowids(self, name=None):
    """Returns the row indices for this row partition.

    `value_rowids` specifies the row index fo reach value.  In particular,
    `value_rowids[i]` is the row index for `values[i]`.

    Args:
      name: A name prefix for the returned tensor (optional).

    Returns:
      A 1-D integer `Tensor` with shape `[self.nvals()]`.
      The returned tensor is nonnegative, and is sorted in ascending order.
    """
    if self._cached_value_rowids is not None:
      return self._cached_value_rowids

    with ops.name_scope(name, "RaggedValueRowIds", [self]):
      return segment_id_ops.row_splits_to_segment_ids(self.row_splits)

  def nrows_as_dimension(self):
    """Returns the first dimension of the shape as a `tf.Dimension`."""
    return tensor_shape.dimension_at_index(self._row_splits.shape, 0) - 1

  def nvals(self, out_type=None, name=None):
    """Returns the number of values partitioned by this `RowPartition`.

    If the sequence partitioned by this `RowPartition` is a tensor, then
    `nvals` is the size of that tensor's outermost dimension -- i.e.,
    `nvals == values.shape[0]`.

    Args:
      out_type: `dtype` for the returned tensor.  Defaults to
        `self.dtype`.
      name: A name prefix for the returned tensor (optional).

    Returns:
      scalar integer Tensor
    """
    if out_type is None:
      return self.row_splits[-1]
    else:
      out_type = dtypes.as_dtype(out_type)
      return math_ops.cast(self.row_splits[-1], name=name, dtype=out_type)

  def nrows(self, out_type=None, name=None):
    """Returns the number of rows created by this `RowPartition`.

    Args:
      out_type: `dtype` for the returned tensor.  Defaults to
        `self.dtype`.
      name: A name prefix for the returned tensor (optional).

    Returns:
      scalar integer Tensor
    """
    if out_type is None:
      out_type = self.dtype
    else:
      out_type = dtypes.as_dtype(out_type)
    if self._cached_nrows is not None:
      return math_ops.cast(self._cached_nrows, out_type)
    with ops.name_scope(name, "RaggedNRows", [self]):
      nsplits = tensor_shape.dimension_at_index(self.row_splits.shape, 0)
      if nsplits.value is None:
        return array_ops.shape(self.row_splits, out_type=out_type)[0] - 1
      else:
        return constant_op.constant(nsplits.value - 1, dtype=out_type)

  def uniform_row_length(self):
    """Returns the length of each row in this partition, if rows are uniform.

    If all rows in this `RowPartition` have the same length, then this returns
    that length as a scalar integer `Tensor`.  Otherwise, it returns `None`.

    Returns:
      scalar Tensor with `type=self.dtype`, or `None`.
    """
    return self._uniform_row_length

  def row_starts(self, name=None):
    """Returns the start indices for rows in this row partition.

    These indices specify where the values for each row begin.
    `partition.row_starts()` is equal to `partition.row_splits()[:-1]`.

    Args:
      name: A name prefix for the returned tensor (optional).

    Returns:
      A 1-D integer Tensor with shape `[self.nrows()]`.
      The returned tensor is nonnegative, and is sorted in ascending order.
      `self.row_starts()[0] == 0`.
      `self.row_starts()[-1] <= self.nvals()`.
    """
    with ops.name_scope(name, "RaggedRowStarts", [self]):
      return self.row_splits[:-1]

  def row_limits(self, name=None):
    """Returns the limit indices for rows in this row partition.

    These indices specify where the values for each row end.
    `partition.row_limits()` is equal to `partition.row_splits()[:-1]`.

    Args:
      name: A name prefix for the returned tensor (optional).

    Returns:
      A 1-D integer Tensor with shape `[self.nrows]`.
      The returned tensor is nonnegative, and is sorted in ascending order.
      `self.row_limits()[-1] == self.nvals()`.
    """
    with ops.name_scope(name, "RaggedRowLimits", [self]):
      return self.row_splits[1:]

  def row_lengths(self, name=None):
    """Returns the lengths of the rows in this `RowPartition`.

    Args:
      name: A name prefix for the returned tensor (optional).

    Returns:
      A 1-D integer Tensor with shape `[self.nrows]`.
      The returned tensor is nonnegative.
      `tf.reduce_sum(self.row_lengths) == self.nvals()`.
    """
    if self._cached_row_lengths is not None:
      return self._cached_row_lengths
    splits = self.row_splits
    with ops.name_scope(name, "RaggedRowLengths", [self]):
      return splits[1:] - splits[:-1]

  #=============================================================================
  # Transformation
  #=============================================================================

  def with_row_splits_dtype(self, dtype):
    """Returns a copy of this RowPartition with the given `row_splits` dtype.

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
      raise ValueError("dtype must be int32 or int64")
    if self.dtype == dtype:
      return self

    row_splits = math_ops.cast(self._row_splits, dtype)

    cached_row_lengths = self._cached_row_lengths
    if cached_row_lengths is not None:
      cached_row_lengths = math_ops.cast(cached_row_lengths, dtype)
    cached_value_rowids = self._cached_value_rowids
    if cached_value_rowids is not None:
      cached_value_rowids = math_ops.cast(cached_value_rowids, dtype)
    cached_nrows = self._cached_nrows
    if cached_value_rowids is not None:
      cached_value_rowids = math_ops.cast(cached_value_rowids, dtype)
    uniform_row_length = self._uniform_row_length
    if uniform_row_length is not None:
      uniform_row_length = math_ops.cast(uniform_row_length, dtype)

    return RowPartition(
        row_splits,
        cached_row_lengths,
        cached_value_rowids,
        cached_nrows,
        internal=True,
        uniform_row_length=uniform_row_length)

  #=============================================================================
  # String Encoding
  #=============================================================================

  def __repr__(self):
    return "tf.RowPartition(row_splits=%s)" % (self._row_splits)

  #=============================================================================
  # Precomputed Encodings
  #=============================================================================

  def has_precomputed_row_splits(self):
    """Returns true if `row_splits` has already been computed.

    If true, then `self.row_splits()` will return its value without calling
    any TensorFlow ops.
    """
    return self._row_splits is not None

  def has_precomputed_row_lengths(self):
    """Returns true if `row_lengths` has already been computed.

    If true, then `self.row_lengths()` will return its value without calling
    any TensorFlow ops.
    """
    return self._cached_row_lengths is not None

  def has_precomputed_value_rowids(self):
    """Returns true if `value_rowids` has already been computed.

    If true, then `self.value_rowids()` will return its value without calling
    any TensorFlow ops.
    """
    return self._cached_value_rowids is not None

  def has_precomputed_nrows(self):
    """Returns true if `nrows` has already been computed.

    If true, then `self.nrows()` will return its value without calling
    any TensorFlow ops.
    """
    return self._cached_nrows is not None


#===============================================================================
# Helper Functions
#===============================================================================


def _assert_monotonic_increasing(tensor, message=None):
  return check_ops.assert_non_negative(
      tensor[1:] - tensor[:-1], message=message)


def _assert_zero(tensor, message=None):
  return check_ops.assert_equal(
      tensor, constant_op.constant(0, dtype=tensor.dtype), message=message)
