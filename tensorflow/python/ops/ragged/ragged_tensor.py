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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_ragged_conversion_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.util.tf_export import tf_export

# pylint: disable=protected-access
_eval_using_default_session = ops._eval_using_default_session

# pylint: enable=protected-access

#===============================================================================
# RaggedTensor
#===============================================================================


@tf_export("RaggedTensor")
class RaggedTensor(object):
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

  ### Potentially Ragged Tensors

  Many ops support both `Tensor`s and `RaggedTensor`s.  The term "potentially
  ragged tensor" may be used to refer to a tensor that might be either a
  `Tensor` or a `RaggedTensor`.  The ragged-rank of a `Tensor` is zero.

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

  ```python
  >>> print(tf.RaggedTensor.from_row_splits(
  ...     values=[3, 1, 4, 1, 5, 9, 2, 6],
  ...     row_splits=[0, 4, 4, 7, 8, 8]))
  <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2], [6], []]>
  ```

  ### Alternative Row-Partitioning Schemes

  In addition to `row_splits`, ragged tensors provide support for four other
  row-partitioning schemes:

    * `row_lengths`: a vector with shape `[nrows]`, which specifies the length
      of each row.

    * `value_rowids` and `nrows`: `value_rowids` is a vector with shape
      `[nvals]`, corresponding one-to-one with `values`, which specifies
      each value's row index.  In particular, the row `rt[row]` consists of the
      values `rt.values[j]` where `value_rowids[j]==row`.  `nrows` is an
      int64 scalar that specifies the number of rows in the `RaggedTensor`.
      (`nrows` is used to indicate trailing empty rows.)

    * `row_starts`: a vector with shape `[nrows]`, which specifies the start
      offset of each row.  Equivalent to `row_splits[:-1]`.

    * `row_limits`: a vector with shape `[nrows]`, which specifies the stop
      offset of each row.  Equivalent to `row_splits[1:]`.

  Example: The following ragged tensors are equivalent, and all represent the
  nested list `[[3, 1, 4, 1], [], [5, 9, 2], [6], []]`.

  ```python
  >>> values = [3, 1, 4, 1, 5, 9, 2, 6]
  >>> rt1 = RaggedTensor.from_row_splits(values, row_splits=[0, 4, 4, 7, 8, 8])
  >>> rt2 = RaggedTensor.from_row_lengths(values, row_lengths=[4, 0, 3, 1, 0])
  >>> rt3 = RaggedTensor.from_value_rowids(
  ...     values, value_rowids=[0, 0, 0, 0, 2, 2, 2, 3], nrows=5)
  >>> rt4 = RaggedTensor.from_row_starts(values, row_starts=[0, 4, 4, 7, 8])
  >>> rt5 = RaggedTensor.from_row_limits(values, row_limits=[4, 4, 7, 8, 8])
  ```

  ### Multiple Ragged Dimensions

  `RaggedTensor`s with multiple ragged dimensions can be defined by using
  a nested `RaggedTensor` for the `values` tensor.  Each nested `RaggedTensor`
  adds a single ragged dimension.

  ```python
  >>> inner_rt = RaggedTensor.from_row_splits(  # =rt1 from above
  ...     values=[3, 1, 4, 1, 5, 9, 2, 6], row_splits=[0, 4, 4, 7, 8, 8])
  >>> outer_rt = RaggedTensor.from_row_splits(
  ...     values=inner_rt, row_splits=[0, 3, 3, 5])
  >>> print outer_rt.to_list()
  [[[3, 1, 4, 1], [], [5, 9, 2]], [], [[6], []]]
  >>> print outer_rt.ragged_rank
  2
  ```

  The factory function `RaggedTensor.from_nested_row_splits` may be used to
  construct a `RaggedTensor` with multiple ragged dimensions directly, by
  providing a list of `row_splits` tensors:

  ```python
  >>> RaggedTensor.from_nested_row_splits(
  ...     flat_values=[3, 1, 4, 1, 5, 9, 2, 6],
  ...     nested_row_splits=([0, 3, 3, 5], [0, 4, 4, 7, 8, 8])).to_list()
  [[[3, 1, 4, 1], [], [5, 9, 2]], [], [[6], []]]
  ```

  ### Uniform Inner Dimensions

  `RaggedTensor`s with uniform inner dimensions can be defined
  by using a multidimensional `Tensor` for `values`.

  ```python
  >>> rt = RaggedTensor.from_row_splits(values=tf.ones([5, 3]),
  ..                                    row_splits=[0, 2, 5])
  >>> print rt.to_list()
  [[[1, 1, 1], [1, 1, 1]],
   [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]
   >>> print rt.shape
   (2, ?, 3)
  ```

  ### RaggedTensor Shape Restrictions

  The shape of a RaggedTensor is currently restricted to have the following
  form:

    * A single uniform dimension
    * Followed by one or more ragged dimensions
    * Followed by zero or more uniform dimensions.

  This restriction follows from the fact that each nested `RaggedTensor`
  replaces the uniform outermost dimension of its `values` with a uniform
  dimension followed by a ragged dimension.
  """

  #=============================================================================
  # Constructor (private)
  #=============================================================================
  def __init__(self,
               values,
               row_splits,
               cached_row_lengths=None,
               cached_value_rowids=None,
               cached_nrows=None,
               internal=False):
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
      row_splits: A 1-D int64 tensor with shape `[nrows+1]`.
      cached_row_lengths: A 1-D int64 tensor with shape `[nrows]`
      cached_value_rowids: A 1-D int64 tensor with shape `[nvals]`.
      cached_nrows: A 1-D int64 scalar tensor.
      internal: True if the constructor is being called by one of the factory
        methods.  If false, an exception will be raised.

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
    if not isinstance(values, (RaggedTensor, ops.Tensor)):
      raise TypeError("values must be a Tensor or RaggedTensor.")
    if not isinstance(row_splits, ops.Tensor):
      raise TypeError("Row-partitioning argument must be a Tensor.")
    values.shape.with_rank_at_least(1)
    row_splits.shape.assert_has_rank(1)
    row_splits.set_shape([None])

    self._values = values
    self._row_splits = row_splits

    # Store any cached tensors.  These are used to avoid unnecessary
    # round-trip conversions when a RaggedTensor is constructed from
    # lengths or rowids, and we later want those lengths/rowids back.
    for tensor in [cached_row_lengths, cached_value_rowids, cached_nrows]:
      if tensor is not None and not isinstance(tensor, ops.Tensor):
        raise TypeError("Cached value must be a Tensor or None.")
    self._cached_row_lengths = cached_row_lengths
    self._cached_value_rowids = cached_value_rowids
    self._cached_nrows = cached_nrows

  #=============================================================================
  # Factory Methods
  #=============================================================================

  @classmethod
  def from_value_rowids(cls, values, value_rowids, nrows=None, name=None):
    """Creates a `RaggedTensor` with rows partitioned by `value_rowids`.

    The returned `RaggedTensor` corresponds with the python list defined by:

    ```python
    result = [[values[i] for i in range(len(values)) if value_rowids[i] == row]
              for row in range(nrows)]
    ```

    Warning: currently, this needs to cast value_rowids to int64 before
    converting, since `tf.bincount` only supports `int32`.

    Args:
      values: A potentially ragged tensor with shape `[nvals, ...]`.
      value_rowids: A 1-D int64 tensor with shape `[nvals]`, which corresponds
        one-to-one with `values`, and specifies each value's row index.  Must be
        nonnegative, and must be sorted in ascending order.
      nrows: An int64 scalar specifying the number of rows.  This should be
        specified if the `RaggedTensor` may containing empty training rows. Must
        be greater than `value_rowids[-1]` (or zero if `value_rowids` is empty).
        Defaults to `value_rowids[-1]` (or zero if `value_rowids` is empty).
      name: A name prefix for the RaggedTensor (optional).

    Returns:
      A `RaggedTensor`.  `result.rank = values.rank + 1`.
      `result.ragged_rank = values.ragged_rank + 1`.

    Raises:
      ValueError: If `nrows` is incompatible with `value_rowids`.

    #### Example:
      ```python
      >>> print(tf.RaggedTensor.from_value_rowids(
      ...     values=[3, 1, 4, 1, 5, 9, 2, 6],
      ...     value_rowids=[0, 0, 0, 0, 2, 2, 2, 3],
      ...     nrows=5))
      <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2], [6], []]>
      ```
    """
    with ops.name_scope(name, "RaggedFromValueRowIds",
                        [values, value_rowids, nrows]):
      values = convert_to_tensor_or_ragged_tensor(values, name="values")
      value_rowids = ops.convert_to_tensor(
          value_rowids, dtypes.int64, name="value_rowids")
      if nrows is None:
        const_rowids = tensor_util.constant_value(value_rowids)
        if const_rowids is None:
          nrows = array_ops.concat([value_rowids[-1:], [-1]], axis=0)[0] + 1
          const_nrows = None
        else:
          const_nrows = const_rowids[-1] + 1 if const_rowids.size > 0 else 0
          nrows = ops.convert_to_tensor(const_nrows, dtypes.int64, name="nrows")
      else:
        nrows = ops.convert_to_tensor(nrows, dtypes.int64, "nrows")
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
      values.shape[:1].assert_is_compatible_with(value_rowids.shape)

      # Convert value_rowids & nrows to row_splits.
      # Note: we don't use segment_ids_to_row_splits() here because we want
      # to save the intermediate value `row_lengths`, so we can cache it.
      # TODO(b/116708836) Upgrade bincount to accept int64 so we can skip the
      # cast (Remove the warning in the docstring when we do.)
      value_rowids_int32 = math_ops.cast(value_rowids, dtypes.int32)
      nrows_int32 = math_ops.cast(nrows, dtypes.int32)
      row_lengths = math_ops.bincount(
          value_rowids_int32,
          minlength=nrows_int32,
          maxlength=nrows_int32,
          dtype=dtypes.int64)
      row_splits = array_ops.concat([[0], math_ops.cumsum(row_lengths)], axis=0)
      if const_nrows is not None:
        row_lengths.set_shape([const_nrows])
        row_splits.set_shape([const_nrows + 1])

      return cls(
          values,
          row_splits,
          cached_row_lengths=row_lengths,
          cached_value_rowids=value_rowids,
          cached_nrows=nrows,
          internal=True)

  @classmethod
  def from_row_splits(cls, values, row_splits, name=None):
    """Creates a `RaggedTensor` with rows partitioned by `row_splits`.

    The returned `RaggedTensor` corresponds with the python list defined by:

    ```python
    result = [values[row_splits[i]:row_splits[i + 1]]
              for i in range(len(row_splits) - 1)]
    ```

    Args:
      values: A potentially ragged tensor with shape `[nvals, ...]`.
      row_splits: A 1-D int64 tensor with shape `[nrows+1]`.  Must not be empty,
        and must be sorted in ascending order.  `row_splits[0]` must be zero and
        `row_splits[-1]` must be `nvals`.
      name: A name prefix for the RaggedTensor (optional).

    Returns:
      A `RaggedTensor`.  `result.rank = values.rank + 1`.
      `result.ragged_rank = values.ragged_rank + 1`.

    Raises:
      ValueError: If `row_splits` is an empty list.

    #### Example:
      ```python
      >>> print(tf.RaggedTensor.from_row_splits(
      ...     values=[3, 1, 4, 1, 5, 9, 2, 6],
      ...     row_splits=[0, 4, 4, 7, 8, 8]))
      <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2], [6], []]>
      ```
    """
    if isinstance(row_splits, (list, tuple)) and not row_splits:
      raise ValueError("row_splits tensor may not be empty.")
    with ops.name_scope(name, "RaggedFromRowSplits", [values, row_splits]):
      values = convert_to_tensor_or_ragged_tensor(values, name="values")
      row_splits = ops.convert_to_tensor(row_splits, dtypes.int64, "row_splits")
      row_splits.shape.assert_has_rank(1)
      return cls(values=values, row_splits=row_splits, internal=True)

  @classmethod
  def from_row_lengths(cls, values, row_lengths, name=None):
    """Creates a `RaggedTensor` with rows partitioned by `row_lengths`.

    The returned `RaggedTensor` corresponds with the python list defined by:

    ```python
    result = [[values.pop(0) for i in range(length)]
              for length in row_lengths]
    ```

    Args:
      values: A potentially ragged tensor with shape `[nvals, ...]`.
      row_lengths: A 1-D int64 tensor with shape `[nrows]`.  Must be
        nonnegative.  `sum(row_lengths)` must be `nvals`.
      name: A name prefix for the RaggedTensor (optional).

    Returns:
      A `RaggedTensor`.  `result.rank = values.rank + 1`.
      `result.ragged_rank = values.ragged_rank + 1`.

    #### Example:
      ```python
      >>> print(tf.RaggedTensor.from_row_lengths(
      ...     values=[3, 1, 4, 1, 5, 9, 2, 6],
      ...     row_lengths=[4, 0, 3, 1, 0]))
      <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2], [6], []])>
      ```
    """
    with ops.name_scope(name, "RaggedFromRowLengths", [values, row_lengths]):
      values = convert_to_tensor_or_ragged_tensor(values, name="values")
      row_lengths = ops.convert_to_tensor(row_lengths, dtypes.int64,
                                          "row_lengths")
      row_lengths.shape.assert_has_rank(1)
      row_limits = math_ops.cumsum(row_lengths)
      row_splits = array_ops.concat([[0], row_limits], axis=0)
      return cls(
          values=values,
          row_splits=row_splits,
          cached_row_lengths=row_lengths,
          internal=True)

  @classmethod
  def from_row_starts(cls, values, row_starts, name=None):
    """Creates a `RaggedTensor` with rows partitioned by `row_starts`.

    Equivalent to: `from_row_splits(values, concat([row_starts, nvals]))`.

    Args:
      values: A potentially ragged tensor with shape `[nvals, ...]`.
      row_starts: A 1-D int64 tensor with shape `[nrows]`.  Must be nonnegative
        and sorted in ascending order.  If `nrows>0`, then `row_starts[0]` must
        be zero.
      name: A name prefix for the RaggedTensor (optional).

    Returns:
      A `RaggedTensor`.  `result.rank = values.rank + 1`.
      `result.ragged_rank = values.ragged_rank + 1`.

    #### Example:
      ```python
      >>> print(tf.RaggedTensor.from_row_starts(
      ...     values=[3, 1, 4, 1, 5, 9, 2, 6],
      ...     row_starts=[0, 4, 4, 7, 8]))
      <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2], [6], []]>
      ```
    """
    with ops.name_scope(name, "RaggedFromRowStarts", [values, row_starts]):
      values = convert_to_tensor_or_ragged_tensor(values, name="values")
      row_starts = ops.convert_to_tensor(row_starts, dtypes.int64, "row_starts")
      row_starts.shape.assert_has_rank(1)
      nvals = array_ops.shape(values, out_type=dtypes.int64)[:1]
      row_splits = array_ops.concat([row_starts, nvals], axis=0)
      return cls(values=values, row_splits=row_splits, internal=True)

  @classmethod
  def from_row_limits(cls, values, row_limits, name=None):
    """Creates a `RaggedTensor` with rows partitioned by `row_limits`.

    Equivalent to: `from_row_splits(values, concat([0, row_limits]))`.

    Args:
      values: A potentially ragged tensor with shape `[nvals, ...]`.
      row_limits: A 1-D int64 tensor with shape `[nrows]`.  Must be sorted in
        ascending order.  If `nrows>0`, then `row_limits[-1]` must be `nvals`.
      name: A name prefix for the RaggedTensor (optional).

    Returns:
      A `RaggedTensor`.  `result.rank = values.rank + 1`.
      `result.ragged_rank = values.ragged_rank + 1`.

    #### Example:
      ```python
      >>> print(tf.RaggedTensor.from_row_limits(
      ...     values=[3, 1, 4, 1, 5, 9, 2, 6],
      ...     row_limits=[4, 4, 7, 8, 8]))
      <tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2], [6], []]>
      ```
    """
    with ops.name_scope(name, "RaggedFromRowLimits", [values, row_limits]):
      values = convert_to_tensor_or_ragged_tensor(values, name="values")
      row_limits = ops.convert_to_tensor(row_limits, dtypes.int64, "row_limits")
      row_limits.shape.assert_has_rank(1)
      zero = array_ops.zeros([1], dtypes.int64)
      row_splits = array_ops.concat([zero, row_limits], axis=0)
      return cls(values=values, row_splits=row_splits, internal=True)

  @classmethod
  def from_nested_value_rowids(cls,
                               flat_values,
                               nested_value_rowids,
                               nested_nrows=None,
                               name=None):
    """Creates a `RaggedTensor` from a nested list of `value_rowids` tensors.

    Equivalent to:

    ```python
    result = flat_values
    for (rowids, nrows) in reversed(zip(nested_value_rowids, nested_nrows)):
      result = from_value_rowids(result, rowids, nrows)
    ```

    Args:
      flat_values: A potentially ragged tensor.
      nested_value_rowids: A list of 1-D int64 tensors.  The `i`th tensor is
        used as the `value_rowids` for the `i`th ragged dimension.
      nested_nrows: A list of int64 scalars.  The `i`th scalar is used as the
        `nrows` for the `i`th ragged dimension.
      name: A name prefix for the RaggedTensor (optional).

    Returns:
      A `RaggedTensor` (or `flat_values` if `nested_value_rowids` is empty).

    Raises:
      ValueError: If `len(nested_values_rowids) != len(nested_nrows)`.
    """
    if isinstance(nested_value_rowids, ops.Tensor):
      raise TypeError("nested_value_rowids must be a list of Tensors")
    if nested_nrows is None:
      nested_nrows = [None] * len(nested_value_rowids)
    else:
      if isinstance(nested_nrows, ops.Tensor):
        raise TypeError("nested_nrows must be a list of Tensors")
      if len(nested_nrows) != len(nested_value_rowids):
        raise ValueError("nested_nrows must have the same length as "
                         "nested_value_rowids")

    with ops.name_scope(
        name, "RaggedFromNestedValueRowIds",
        [flat_values] + list(nested_value_rowids) + list(nested_nrows)):
      result = flat_values
      for value_rowids, nrows in reversed(
          list(zip(nested_value_rowids, nested_nrows))):
        result = cls.from_value_rowids(result, value_rowids, nrows)
      return result

  @classmethod
  def from_nested_row_splits(cls, flat_values, nested_row_splits, name=None):
    """Creates a `RaggedTensor` from a nested list of `row_splits` tensors.

    Equivalent to:

    ```python
    result = flat_values
    for row_splits in reversed(nested_row_splits):
      result = from_row_splits(result, row_splits)
    ```

    Args:
      flat_values: A potentially ragged tensor.
      nested_row_splits: A list of 1-D int64 tensors.  The `i`th tensor is used
        as the `row_splits` for the `i`th ragged dimension.
      name: A name prefix for the RaggedTensor (optional).

    Returns:
      A `RaggedTensor` (or `flat_values` if `nested_row_splits` is empty).
    """
    if isinstance(nested_row_splits, ops.Tensor):
      raise TypeError("nested_row_splits must be a list of Tensors")
    with ops.name_scope(name, "RaggedFromNestedRowSplits",
                        [flat_values] + list(nested_row_splits)):
      result = flat_values
      for splits in reversed(nested_row_splits):
        result = cls.from_row_splits(result, splits)
      return result

  @classmethod
  def from_nested_row_lengths(cls, flat_values, nested_row_lengths, name=None):
    """Creates a `RaggedTensor` from a nested list of `row_lengths` tensors.

    Equivalent to:

    ```python
    result = flat_values
    for row_lengths in reversed(nested_row_lengths):
      result = from_row_lengths(result, row_lengths)
    ```

    Args:
      flat_values: A potentially ragged tensor.
      nested_row_lengths: A list of 1-D int64 tensors.  The `i`th tensor is used
        as the `row_lengths` for the `i`th ragged dimension.
      name: A name prefix for the RaggedTensor (optional).

    Returns:
      A `RaggedTensor` (or `flat_values` if `nested_row_lengths` is empty).
    """
    if isinstance(nested_row_lengths, ops.Tensor):
      raise TypeError("nested_row_lengths must be a list of Tensors")
    with ops.name_scope(name, "RaggedFromNestedRowlengths",
                        [flat_values] + list(nested_row_lengths)):
      result = flat_values
      for lengths in reversed(nested_row_lengths):
        result = cls.from_row_lengths(result, lengths)
      return result

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

      ```python
      >>> ragged.constant([[0], [1, 2]]).shape
      TensorShape([Dimension(2), Dimension(None)])

      >>> ragged.constant([[[0, 1]], [[1, 2], [3, 4]]], ragged_rank=1).shape
      TensorShape([Dimension(2), Dimension(None), Dimension(2)
      ```
    """
    nrows = tensor_shape.dimension_at_index(self._row_splits.shape, 0) - 1

    values_shape = self._values.shape
    value_shape = values_shape[1:]
    return tensor_shape.TensorShape([nrows, None]).concatenate(value_shape)

  @property
  def ragged_rank(self):
    """The number of ragged dimensions in this ragged tensor.

    Returns:
      A Python `int` indicating the number of ragged dimensions in this ragged
      tensor.  The outermost dimension is not considered ragged.
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
      ```python
      >>> rt = ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
      >>> print rt.values
      tf.Tensor([3, 1, 4, 1, 5, 9, 2, 6])
      ```
    """
    return self._values

  @property
  def row_splits(self):
    """The row-split indices for this ragged tensor's `values`.

    `rt.row_splits` specifies where the values for each row begin and end in
    `rt.values`.  In particular, the values for row `rt[i]` are stored in
    the slice `rt.values[rt.row_splits[i]:rt.row_splits[i+1]]`.

    Returns:
      A 1-D `int64` `Tensor` with shape `[self.nrows+1]`.
      The returned tensor is non-empty, and is sorted in ascending order.
      `self.row_splits[0]` is zero, and `self.row_splits[-1]` is equal to
      `self.values.shape[0]`.

    #### Example:
      ```python
      >>> rt = ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
      >>> print rt.row_splits  # indices of row splits in rt.values
      tf.Tensor([0, 4, 4, 7, 8, 8])
      ```
    """
    return self._row_splits

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

      ```python
      >>> rt = ragged.constant([[[3, 1, 4, 1], [], [5, 9, 2]], [], [[6], []]])
      >>> print rt.flat_values()
      tf.Tensor([3, 1, 4, 1, 5, 9, 2, 6])
      ```
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
      A `tuple` of 1-D `int64` `Tensor`s.

    #### Example:

      ```python
      >>> rt = ragged.constant([[[[3, 1, 4, 1], [], [5, 9, 2]], [], [[6], []]]])
      >>> for i, splits in enumerate(rt.nested_row_splits()):
      ...   print('Splits for dimension %d: %s' % (i+1, splits))
      Splits for dimension 1: [0, 1]
      Splits for dimension 2: [0, 3, 3, 5]
      Splits for dimension 3: [0, 4, 4, 7, 8, 8]
      ```

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
      A 1-D `int64` `Tensor` with shape `self.values.shape[:1]`.
      The returned tensor is nonnegative, and is sorted in ascending order.

    #### Example:
      ```python
      >>> rt = ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
      >>> rt.values
      tf.Tensor([3, 1, 4, 1, 5, 9, 2, 6])
      >>> rt.value_rowids()
      tf.Tensor([0, 0, 0, 0, 2, 2, 2, 3])  # corresponds 1:1 with rt.values
      ```
    """
    if self._cached_value_rowids is not None:
      return self._cached_value_rowids

    with ops.name_scope(name, "RaggedValueRowIds", [self]):
      return segment_id_ops.row_splits_to_segment_ids(self.row_splits)

  def nrows(self, out_type=dtypes.int64, name=None):
    """Returns the number of rows in this ragged tensor.

    I.e., the size of the outermost dimension of the tensor.

    Args:
      out_type: `dtype` for the returned tensor.
      name: A name prefix for the returned tensor (optional).

    Returns:
      A scalar `Tensor` with dtype `out_type`.

    #### Example:
      ```python
      >>> rt = ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
      >>> rt.nrows()  # rt has 5 rows.
      5
      ```
    """
    if self._cached_nrows is not None:
      return self._cached_nrows

    with ops.name_scope(name, "RaggedNRows", [self]):
      return array_ops.shape(self.row_splits, out_type=out_type)[0] - 1

  def row_starts(self, name=None):
    """Returns the start indices for rows in this ragged tensor.

    These indices specify where the values for each row begin in
    `self.values`.  `rt.row_starts()` is equal to `rt.row_splits[:-1]`.

    Args:
      name: A name prefix for the returned tensor (optional).

    Returns:
      A 1-D Tensor of int64 with shape `[nrows]`.
      The returned tensor is nonnegative, and is sorted in ascending order.

    #### Example:
      ```python
      >>> rt = ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
      >>> rt.values
      tf.Tensor([3, 1, 4, 1, 5, 9, 2, 6])
      >>> rt.row_starts()  # indices of row starts in rt.values
      tf.Tensor([0, 4, 4, 7, 8])
      ```
    """
    with ops.name_scope(name, "RaggedRowStarts", [self]):
      return self.row_splits[:-1]

  def row_limits(self, name=None):
    """Returns the limit indices for rows in this ragged tensor.

    These indices specify where the values for each row end in
    `self.values`.  `rt.row_limits(self)` is equal to `rt.row_splits[:-1]`.

    Args:
      name: A name prefix for the returned tensor (optional).

    Returns:
      A 1-D Tensor of int64 with shape `[nrows]`.
      The returned tensor is nonnegative, and is sorted in ascending order.

    #### Example:
      ```python
      >>> rt = ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
      >>> rt.values
      tf.Tensor([3, 1, 4, 1, 5, 9, 2, 6])
      >>> rt.row_limits()  # indices of row limits in rt.values
      tf.Tensor([4, 4, 7, 8, 8])
      ```
    """
    with ops.name_scope(name, "RaggedRowLimits", [self]):
      return self.row_splits[1:]

  def row_lengths(self, axis=1, name=None):
    """Returns the lengths of the rows in this ragged tensor.

    `rt.row_lengths()[i]` indicates the number of values in the
    `i`th row of `rt`.

    Args:
      axis: An integer constant indicating the axis whose row lengths should be
        returned.
      name: A name prefix for the returned tensor (optional).

    Returns:
      A potentially ragged Tensor of int64 with shape `self.shape[:axis]`.

    Raises:
      ValueError: If `axis` is out of bounds.

    #### Example:
      ```python
      >>> rt = ragged.constant([[[3, 1, 4], [1]], [], [[5, 9], [2]], [[6]], []])
      >>> rt.row_lengths(rt)  # lengths of rows in rt
      tf.Tensor([2, 0, 2, 1, 0])
      >>> rt.row_lengths(axis=2)  # lengths of axis=2 rows.
      <tf.RaggedTensor [[3, 1], [], [2, 1], [1], []]>
      ```
    """
    if self._cached_row_lengths is not None:
      return self._cached_row_lengths

    with ops.name_scope(name, "RaggedRowLengths", [self]):
      axis = ragged_util.get_positive_axis(axis, self.shape.ndims)
      if axis == 0:
        return self.nrows()
      elif axis == 1:
        splits = self.row_splits
        return splits[1:] - splits[:-1]
      elif isinstance(self.values, RaggedTensor):
        return self.with_values(self.values.row_lengths(axis - 1))
      else:
        shape = array_ops.shape(self.values, out_type=dtypes.int64)
        return self.with_values(
            array_ops.ones(shape[:axis - 1], dtypes.int64) * shape[axis - 1])

  def nested_row_lengths(self, name=None):
    """Returns a tuple containing the row_lengths for all ragged dimensions.

    `rtnested_row_lengths()` is a tuple containing the `row_lengths` tensors for
    all ragged dimensions in `rt`, ordered from outermost to innermost.

    Args:
      name: A name prefix for the returned tensors (optional).

    Returns:
      A `tuple` of 1-D `int64` `Tensors`.  The length of the tuple is equal to
      `self.ragged_rank`.
    """
    with ops.name_scope(name, "RaggedNestedRowLengths", [self]):
      rt_nested_row_lengths = []
      rt = self
      while isinstance(rt, RaggedTensor):
        rt_nested_row_lengths.append(rt.row_lengths())
        rt = rt.values
      return tuple(rt_nested_row_lengths)

  def bounding_shape(self, axis=None, name=None):
    """Returns the tight bounding box shape for this `RaggedTensor`.

    Args:
      axis: An integer scalar or vector indicating which axes to return the
        bounding box for.  If not specified, then the full bounding box is
        returned.
      name: A name prefix for the returned tensor (optional).

    Returns:
      An int64 `Tensor`.  If `axis` is not specified, then `output`
      is a vector with `output.shape=[self.shape.ndims]`.  If `axis` is a
      scalar, then the `output` is a scalar.  If `axis` is a vector, then
      `output` is a vector, where `output[i]` is the bounding size for
      dimension `axis[i]`.

    #### Example:
      ```python
      >>> rt = ragged.constant([[1, 2, 3, 4], [5], [], [6, 7, 8, 9], [10]])
      >>> rt.bounding_shape()
      [5, 4]
      ```
    """
    with ops.name_scope(name, "RaggedBoundingBox", [self, axis]):
      nested_splits = self.nested_row_splits
      rt_flat_values = self.flat_values

      # Optimized special cases for when axis=0 or axis=1:
      if isinstance(axis, int):
        if axis == 0:
          return array_ops.shape(nested_splits[0], out_type=dtypes.int64)[0] - 1
        elif axis == 1:
          return math_ops.maximum(math_ops.reduce_max(self.row_lengths()), 0)

      splits_shape = array_ops.shape(self.row_splits, out_type=dtypes.int64)
      flat_values_shape = array_ops.shape(rt_flat_values, out_type=dtypes.int64)

      ragged_dimensions = array_ops.stack([splits_shape[0] - 1] + [
          math_ops.maximum(math_ops.reduce_max(splits[1:] - splits[:-1]), 0)
          for splits in nested_splits
      ])
      inner_dimensions = flat_values_shape[1:]

      bbox = array_ops.concat([ragged_dimensions, inner_dimensions], axis=0)
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
    new_values.shape.with_rank_at_least(1)
    self.values.shape[:1].assert_is_compatible_with(new_values.shape[:1])
    return RaggedTensor(
        new_values,
        self._row_splits,
        self._cached_row_lengths,
        self._cached_value_rowids,
        self._cached_nrows,
        internal=True)

  def with_flat_values(self, new_values):
    """Returns a copy of `self` with `flat_values` replaced by `new_value`.

    Preserves cached row-partitioning tensors such as `self.cached_nrows` and
    `self.cached_value_rowids` if they have values.

    Args:
      new_values: Potentially ragged tensor that should replace
      `self.flat_values`.  Must have `rank > 0`, and must have the same
      number of rows as `self.flat_values`.

    Returns:
      A `RaggedTensor`.
      `result.rank = self.ragged_rank + new_values.rank`.
      `result.ragged_rank = self.ragged_rank + new_values.ragged_rank`.
    """
    if isinstance(self._values, ops.Tensor):
      return self.with_values(new_values)
    else:
      return self.with_values(self.values.with_flat_values(new_values))

  #=============================================================================
  # Tensor Type Conversions
  #=============================================================================

  @classmethod
  def from_tensor(cls,
                  tensor,
                  lengths=None,
                  padding=None,
                  ragged_rank=1,
                  name=None):
    """Converts a `tf.Tensor` into a `RaggedTensor`.

    The set of absent/default values may be specified using a vector of lengths
    or a padding value (but not both).  If `lengths` is specified, then the
    output tensor will satisfy `output[row] = tensor[row][:lengths[row]]`.
    If `padding` is specified, then any row *suffix* consisting entirely of
    `padding` will be excluded from the returned `RaggedTensor`.  If neither
    `lengths` nor `padding` is specified, then the returned `RaggedTensor` will
    have no absent/default values.

    Examples:

    ```python
    >>> dt = tf.constant([[5, 7, 0], [0, 3, 0], [6, 0, 0]])
    >>> tf.RaggedTensor.from_tensor(dt)
    <tf.RaggedTensor [[5, 7, 0], [0, 3, 0], [6, 0, 0]]>
    >>> tf.RaggedTensor.from_tensor(dt, lengths=[2, 0, 3])
    <tf.RaggedTensor [[5, 7], [], [6, 0, 0]]>
    >>> tf.RaggedTensor.from_tensor(dt, padding=0)
    <tf.RaggedTensor [[5, 7], [0, 3], [6]]>
    ```

    Args:
      tensor: The `Tensor` to convert.  Must have rank `ragged_rank + 1` or
        higher.
      lengths: An optional set of row lengths, specified using a 1-D integer
        `Tensor` whose length is equal to `tensor.shape[0]` (the number of rows
        in `tensor`).  If specified, then `output[row]` will contain
        `tensor[row][:lengths[row]]`.  Negative lengths are treated as zero.
      padding: An optional padding value.  If specified, then any row suffix
        consisting entirely of `padding` will be excluded from the returned
        RaggedTensor.  `padding` is a `Tensor` with the same dtype as `tensor`
        and with `shape=tensor.shape[ragged_rank + 1:]`.
      ragged_rank: Integer specifying the ragged rank for the returned
        `RaggedTensor`.  Must be greater than zero.
      name: A name prefix for the returned tensors (optional).

    Returns:
      A `RaggedTensor` with the specified `ragged_rank`.  The shape of the
      returned ragged tensor is compatible with the shape of `tensor`.
    Raises:
      ValueError: If both `lengths` and `padding` are specified.
    """
    if lengths is not None and padding is not None:
      raise ValueError("Specify lengths or padding, but not both")
    if not isinstance(ragged_rank, int):
      raise TypeError("ragged_rank expected int, got %r" % ragged_rank)
    if ragged_rank <= 0:
      raise ValueError(
          "ragged_rank must be greater than 0; got %s" % ragged_rank)

    with ops.name_scope(name, "RaggedFromTensor", [tensor, lengths, padding]):
      tensor = ops.convert_to_tensor(tensor, name="tensor")
      tensor.shape.with_rank_at_least(ragged_rank + 1)
      input_shape = array_ops.shape(tensor, out_type=dtypes.int64)
      ncols = input_shape[1]

      # Handle ragged_rank>1 via recursion:
      # If the output should have multiple ragged dimensions, then first
      # flatten the tensor to eliminate all but the last ragged dimension,
      # and recursively convert that flattened tensor.  Then add on the splits
      # for the dimensions that we flattened out.
      if ragged_rank > 1:
        # Flatten `tensor` to eliminate all but the last ragged dimension.
        new_shape = array_ops.concat([
            constant_op.constant([-1], dtypes.int64), input_shape[ragged_rank:]
        ],
                                     axis=0)
        flattened = array_ops.reshape(tensor, new_shape)
        # Recursively convert the flattened tensor.
        values = cls.from_tensor(flattened, lengths, padding)
        # The total number of elements in each  dimension.  E.g., if
        # input_shape=[3, 4, 5, 6], then dim[2] has 3*4*5 elements in total.
        dim_size = math_ops.cumprod(input_shape)
        # Construct splits tensors for the dimensions that were flattened.
        new_splits = [
            math_ops.range(0, dim_size[dim - 1] + 1) * input_shape[dim]
            for dim in range(1, ragged_rank)
        ]
        return cls.from_nested_row_splits(values, new_splits)

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

        # Use has_default it to find the length of each row: for each
        # non-default item in a row, calculate the length that the row needs to
        # have to include that item; and then take the max of those values
        # (across each row).
        has_nondefault = math_ops.logical_not(has_default)
        has_nondefault = math_ops.cast(has_nondefault, dtypes.int64)
        length_for_nondefault_value = (
            has_nondefault * array_ops.expand_dims(
                math_ops.range(1, ncols + 1), 0))
        lengths = math_ops.reduce_max(length_for_nondefault_value, axis=1)

      # If we have lengths (either directly supplied, or computed from
      # paddings), then use those to construct splits; and then use masking
      # to get the corresponding values.
      if lengths is not None:
        lengths = ragged_util.convert_to_int_tensor(lengths, "lengths",
                                                    dtypes.int64)
        lengths.shape.assert_has_rank(1)
        lengths = math_ops.minimum(lengths, ncols)
        lengths = math_ops.maximum(lengths, 0)
        limits = math_ops.cumsum(lengths)
        splits = array_ops.concat([array_ops.zeros([1], dtypes.int64), limits],
                                  axis=0)
        mask = array_ops.sequence_mask(lengths, maxlen=ncols)
        values = array_ops.boolean_mask(tensor, mask)
        return cls.from_row_splits(values, splits)

      # If neither padding nor lengths were specified, then create a splits
      # vector that contains no default values, and reshape the input tensor
      # to form the values for the RaggedTensor.
      nrows = input_shape[0]
      nvals = nrows * ncols
      splits = math_ops.range(nrows + 1) * ncols
      values_shape = array_ops.concat([[nvals], input_shape[2:]], axis=0)
      values = array_ops.reshape(tensor, values_shape)
      return cls.from_row_splits(values, splits)

  def to_tensor(self, default_value=None, name=None):
    """Converts this `RaggedTensor` into a `tf.Tensor`.

    Example:

    ```python
    >>> rt = ragged.constant([[9, 8, 7], [], [6, 5], [4]])
    >>> print rt.to_tensor()
    [[9 8 7]
     [0 0 0]
     [6 5 0]
     [4 0 0]]
    ```

    Args:
      default_value: Value to set for indices not specified in `self`. Defaults
        to zero.  `default_value` must be broadcastable to
        `self.shape[self.ragged_rank + 1:]`.
      name: A name prefix for the returned tensors (optional).

    Returns:
      A `Tensor` with shape `ragged.bounding_shape(self)` and the
      values specified by the non-empty values in `self`.  Empty values are
      assigned `default_value`.
    """
    with ops.name_scope(name, "RaggedToTensor", [self, default_value]):
      if default_value is not None:
        default_value = ops.convert_to_tensor(
            default_value, name="default_value", dtype=self.dtype)

      # If ragged_rank > 1, then recursively convert the ragged values into a
      # `Tensor` before we proceed.
      values = self.values
      if is_ragged(values):
        values = values.to_tensor(default_value)

      # Tile the default value, if necessary.
      if default_value is not None:
        if values.shape.ndims is not None:
          default_value.shape.with_rank_at_most(values.shape.ndims - 1)
        if (values.shape.ndims is None or default_value.shape.ndims is None or
            values.shape.ndims != default_value.shape.ndims + 1):
          value_shape = array_ops.shape(values)[1:]
          default_value = array_ops.broadcast_to(default_value, value_shape)
        default_value.shape.assert_is_compatible_with(values.shape[1:])

      # Get the expected dense shape ([nrows, ncols] + value_shape).
      rt_row_lengths = [self.row_splits[1:] - self.row_splits[:-1]]
      nrows = array_ops.shape(self.row_splits, out_type=dtypes.int64)[0] - 1
      ncols = math_ops.maximum(math_ops.reduce_max(rt_row_lengths), 0)
      values_shape = array_ops.shape(values, out_type=dtypes.int64)
      value_shape = values_shape[1:]
      nvals = values_shape[0]

      # Build a default value if none was supplied.
      if default_value is None:
        default_value = array_ops.zeros(value_shape, dtype=values.dtype)
      default_value.shape.assert_is_compatible_with(values.shape[1:])
      default_value.set_shape(values.shape[1:])

      # Get the row start indices, and expand to shape=[nrows, 1].
      starts = array_ops.expand_dims(self.row_splits[:-1], 1)

      # Get the row limit indices, and expand to shape=[nrows, 1].
      limits = array_ops.expand_dims(self.row_splits[1:], 1)

      # Get the column indices, and expand to shape=[1, ncols].
      columns = array_ops.expand_dims(math_ops.range(0, ncols), 0)

      # Build a list containing the values plus the default value.  We will use
      # tf.gather to collect values from this list for the `Tensor` (using
      # nvals as the index for the default value).
      values_and_default = array_ops.concat(
          [values, array_ops.stack([default_value])], axis=0)

      # Construct a matrix "indices" pointing into values_and_default.  I.e.,
      # output[r, c] = values_and_default[indices[r, c].
      nondefault_index = starts + columns
      has_value = nondefault_index < limits
      default_index = array_ops.fill(array_ops.stack([nrows, ncols]), nvals)
      indices = array_ops.where(has_value, nondefault_index, default_index)

      # Gather the results into a `Tensor`.
      return array_ops.gather(values_and_default, indices)

  @classmethod
  def from_sparse(cls, st_input, name=None):
    """Converts a 2D `tf.SparseTensor` to a `RaggedTensor`.

    Each row of the `output` `RaggedTensor` will contain the explicit values
    from the same row in `st_input`.  `st_input` must be ragged-right.  If not
    it is not ragged-right, then an error will be generated.

    Example:

    ```python
    >>> st = SparseTensor(indices=[[0, 1], [0, 2], [0, 3], [1, 0], [3, 0]],
    ...                   values=[1, 2, 3, 4, 5],
    ...                   dense_shape=[4, 3])
    >>> rt.RaggedTensor.from_sparse(st).eval().tolist()
    [[1, 2, 3], [4], [], [5]]
    ```

    Currently, only two-dimensional `SparseTensors` are supported.

    Args:
      st_input: The sparse tensor to convert.  Must have rank 2.
      name: A name prefix for the returned tensors (optional).

    Returns:
      A `RaggedTensor` with the same values as `st_input`.
      `output.ragged_rank = rank(st_input) - 1`.
      `output.shape = [st_input.dense_shape[0], None]`.
    Raises:
      ValueError: If the number of dimensions in `st_input` is not known
        statically, or is not two.
    """
    if not sparse_tensor.is_sparse(st_input):
      raise TypeError("Expected SparseTensor, got %s" % type(st_input).__name__)
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
        raise ValueError("rank(st_input) must be 2")

      with ops.control_dependencies(
          _assert_sparse_indices_are_ragged_right(st_input.indices)):
        # Treat sparse row indices as segment ids to generate a splits tensor
        # thta we can pair with the sparse tensor values.  (Ignore sparse column
        # indices.)
        segment_ids = st_input.indices[:, 0]
        num_segments = st_input.dense_shape[0]
        return cls.from_value_rowids(st_input.values, segment_ids, num_segments)

  def to_sparse(self, name=None):
    """Converts this `RaggedTensor` into a `tf.SparseTensor`.

    Example:

    ```python
    >>> rt = ragged.constant([[1, 2, 3], [4], [], [5, 6]])
    >>> rt.to_sparse().eval()
    SparseTensorValue(indices=[[0, 0], [0, 1], [0, 2], [1, 0], [3, 0], [3, 1]],
                      values=[1, 2, 3, 4, 5, 6],
                      dense_shape=[4, 3])
    ```

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

  #=============================================================================
  # String Encoding
  #=============================================================================
  def __str__(self):
    if self._is_eager():
      return "<tf.RaggedTensor %s>" % self.to_list()
    else:
      return self.__repr__()

  def __repr__(self):
    return "tf.RaggedTensor(values=%s, row_splits=%s)" % (self._values,
                                                          self._row_splits)

  #=============================================================================
  # Eager Execution Mode
  #=============================================================================

  def to_list(self):
    """Returns a nested Python `list` with the values for this `RaggedTensor`.

    Requires that `rt` was constructed in eager execution mode.

    Returns:
      A nested Python `list`.
    """
    if self._is_eager():
      return self._eager_value().to_list()
    else:
      raise ValueError("RaggedTensor.to_list() is only supported in eager "
                       "mode; in graph mode, evaluate the RaggedTensor first "
                       "and then use RaggedTensorValue.to_list().")

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
  # Indexing & Slicing
  #=============================================================================
  def __getitem__(self, key):
    """Returns the specified piece of this RaggedTensor."""
    # See ragged_getitem.py for the documentation and implementation of this
    # method.
    #
    # Note: the imports in ragged/__init__.py ensure that this method always
    # gets overridden before it is called.

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


def is_ragged(value):
  """Returns true if `value` is a ragged tensor or ragged tensor value."""
  return isinstance(value,
                    (RaggedTensor, ragged_tensor_value.RaggedTensorValue))


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
      raise ValueError("Tensor conversion requested dtype %s for "
                       "RaggedTensor with dtype %s: %r" %
                       (dtype.name, value.dtype.name, value))
    return value
  elif isinstance(value, ragged_tensor_value.RaggedTensorValue):
    with ops.name_scope(name, "ConvertToTensorOrRaggedTensor", []):
      flat_values = ops.convert_to_tensor(
          value=value.flat_values,
          dtype=dtype,
          preferred_dtype=preferred_dtype,
          name="flat_values")
      return RaggedTensor.from_nested_row_splits(flat_values,
                                                 value.nested_row_splits)
  else:
    return ops.convert_to_tensor(
        value=value, dtype=dtype, preferred_dtype=preferred_dtype, name=name)


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
class RaggedTensorType(object):
  """Encoding of a static type for a `RaggedTensor`.

  Use this type to express/declare that an output must have the type of
  `RaggedTensor`.
  """

  def __init__(self, dtype, ragged_rank):
    """Initializes a RaggedTensorType object.

    Args:
      dtype: data type of the `RaggedTensor`'s inner values.
      ragged_rank: ragged_rank of the declared `RaggedTensor`.
    """
    self._dtype = dtype
    self._ragged_rank = ragged_rank

  dtype = property(lambda self: self._dtype)
  ragged_rank = property(lambda self: self._ragged_rank)


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
