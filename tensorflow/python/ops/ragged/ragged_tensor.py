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
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops.ragged import ragged_tensor_value

# pylint: disable=protected-access
_eval_using_default_session = ops._eval_using_default_session

# pylint: enable=protected-access

#===============================================================================
# RaggedTensor
#===============================================================================


class RaggedTensor(object):
  """Represents a ragged tensor (go/ragged).

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
  >>> rt = ragged.from_row_splits(values=[3, 1, 4, 1, 5, 9, 2, 6],
  ...                             row_splits=[0, 4, 4, 7, 8, 8])
  >>> rt.tolist()
  [[3, 1, 4, 1], [], [5, 9, 2], [6], []]
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
  >>> rt1 = ragged.from_row_splits(values, row_splits=[0, 4, 4, 7, 8, 8])
  >>> rt2 = ragged.from_row_lengths(values, row_lengths=[4, 0, 3, 1, 0])
  >>> rt3 = ragged.from_value_rowids(values,
  ...                                value_rowids=[0, 0, 0, 0, 2, 2, 2, 3],
  ...                                nrows=5)
  >>> rt4 = ragged.from_row_starts(values, row_starts=[0, 4, 4, 7, 8])
  >>> rt5 = ragged.from_row_limits(values, row_limits=[4, 4, 7, 8, 8])
  ```

  ### Multiple Ragged Dimensions

  `RaggedTensor`s with multiple ragged dimensions can be defined by using
  a nested `RaggedTensor` for the `values` tensor.  Each nested `RaggedTensor`
  adds a single ragged dimension.

  ```python
  >>> inner_rt = ragged.from_row_splits(  # =rt1 from above
  ...     values=[3, 1, 4, 1, 5, 9, 2, 6], row_splits=[0, 4, 4, 7, 8, 8])
  >>> outer_rt = ragged.from_row_splits(
  ...     values=inner_rt, row_splits=[0, 3, 3, 5])
  >>> print outer_rt.tolist()
  [[[3, 1, 4, 1], [], [5, 9, 2]], [], [[6], []]]
  >>> print outer_rt.ragged_rank
  2
  ```

  The factory function `ragged.from_nested_row_splits` may be used to
  construct a `RaggedTensor` with multiple ragged dimensions directly, by
  providing a list of `row_splits` tensors:

  ```python
  >>> ragged.from_nested_row_splits(
  ...     inner_values=[3, 1, 4, 1, 5, 9, 2, 6],
  ...     nested_row_splits=([0, 3, 3, 5], [0, 4, 4, 7, 8, 8])).tolist()
  [[[3, 1, 4, 1], [], [5, 9, 2]], [], [[6], []]]
  ```

  ### Uniform Inner Dimensions

  `RaggedTensor`s with uniform inner dimensions can be defined
  by using a multidimensional `Tensor` for `values`.

  ```python
  >>> rt = ragged.from_row_splits(values=tf.ones([5, 3]), row_splits=[0, 2, 5])
  >>> print rt.tolist()
  [[[1, 1, 1], [1, 1, 1]],
   [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]
   >>> print rt.shape.as_list()
   [2, None, 3]
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
  # Implementation notes
  #=============================================================================
  # Currently, the RaggedTensor class uses a single row-partitioning scheme
  # (row_splits).
  #
  # We are considering adding value_rowids+nvals as a secondary
  # row-partitioning scheme.  This change would not impact the functional
  # interface of the RaggedTensor class, but it would impact the efficiency
  # of several operations.  In particular:
  #
  #   * The functions `ragged.value_rowids` and `ragged.nrows` would always
  #     return pre-existing tensors; they would not need to add any ops to
  #     the graph.
  #
  #   * The `RaggedTensor` constructor would construct all row-partitioning
  #     tensors (row_splits, value_rowids, and nvals).  In eager mode, this
  #     would mean that conversion operations would occur whenever a
  #     `RaggedTensor` is constructed.  But in graph mode, the converted
  #     row-partitioning tensors would only be evaluated if they are used.
  #
  # Since this change impacts efficiency but not functionality, we would like
  # to perform additional profiling with real-world use cases before we
  # decide whether to make this change.

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

      * [`ragged.from_row_lengths()`](from_row_lengths.md)
      * [`ragged.from_value_rowids()`](from_value_rowids.md)
      * [`ragged.from_row_splits()`](from_row_splits.md)
      * [`ragged.from_row_starts()`](from_row_starts.md)
      * [`ragged.from_row_limits()`](from_row_limits.md)
      * [`ragged.from_nested_row_splits()`](from_nested_row_splits.md)
      * [`ragged.from_nested_value_rowids()`](from_nested_value_rowids.md)

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
                       "ragged.from_row_lengths())")

    # Validate the arguments.
    if not isinstance(values, (RaggedTensor, ops.Tensor)):
      raise TypeError("values must be a Tensor or RaggedTensor.")
    if not isinstance(row_splits, ops.Tensor):
      raise TypeError("Row-partitioning argument must be a Tensor.")
    values.shape.with_rank_at_least(1)
    row_splits.shape.assert_has_rank(1)

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
      >>> rt.values.eval()
      [3, 1, 4, 1, 5, 9, 2, 6]
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
      >>> rt.values.eval()
      [3, 1, 4, 1, 5, 9, 2, 6]
      >>> rt.row_splits.eval()  # indices of row splits in ragged.values
      [0, 4, 4, 7, 8, 8]
      ```
    """
    return self._row_splits

  @property
  def inner_values(self):
    """The innermost `values` tensor for this ragged tensor.

    Concretely, if `rt.values` is a `Tensor`, then `rt.inner_values` is
    `rt.values`; otherwise, `rt.inner_values` is `rt.values.inner_values`.

    Conceptually, `inner_values` is the tensor formed by flattening the
    outermost dimension and all of the ragged dimensions into a single
    dimension.

    `rt.inner_values.shape = [nvals] + rt.shape[rt.ragged_rank + 1:]`
    (where `nvals` is the number of items in the flattened dimensions).

    Returns:
      A `Tensor`.

    #### Example:

      ```python
      >>> rt = ragged.constant([[[3, 1, 4, 1], [], [5, 9, 2]], [], [[6], []]])
      >>> ragged.inner_values(rt).eval()
      [3, 1, 4, 1, 5, 9, 2, 6]
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
      >>> for i, splits in enumerate(ragged.nested_row_splits(rt)):
      ...   print('Splits for dimension %d: %s' % (i+1, splits.eval()))
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

  @property
  def cached_value_rowids(self):
    """The row lengths for this `RaggedTensor`, or `None`.

    Returns:
      The `value_rowids` tensor that was used to construct this `RaggedTensor`
      if it was constructed using
      [`ragged.from_value_rowids`](from_value_rowids.md); or `None` otherwise.
    """
    return self._cached_value_rowids

  @property
  def cached_nrows(self):
    """The row lengths for this `RaggedTensor`, or `None`.

    Returns:
      The `nrows` tensor that was used to construct this `RaggedTensor`
      if it was constructed using
      [`ragged.from_value_rowids`](from_value_rowids.md); or `None` otherwise.
    """
    return self._cached_nrows

  @property
  def cached_row_lengths(self):
    """The row lengths for this `RaggedTensor`, or `None`.

    Returns:
      The `row_lengths` tensor that was used to construct this `RaggedTensor`
      if it was constructed using
      [`ragged.from_row_lengths`](from_row_lengths.md); or `None` otherwise.
    """
    return self._cached_row_lengths

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
    self.values.shape[0].assert_is_compatible_with(new_values.shape[0])
    return RaggedTensor(
        new_values,
        self._row_splits,
        self._cached_row_lengths,
        self._cached_value_rowids,
        self._cached_nrows,
        internal=True)

  def with_inner_values(self, new_values):
    """Returns a copy of `self` with `inner_values` replaced by `new_value`.

    Preserves cached row-partitioning tensors such as `self.cached_nrows` and
    `self.cached_value_rowids` if they have values.

    Args:
      new_values: Potentially ragged tensor that should replace
      `self.inner_values`.  Must have `rank > 0`, and must have the same
      number of rows as `self.inner_values`.

    Returns:
      A `RaggedTensor`.
      `result.rank = self.ragged_rank + new_values.rank`.
      `result.ragged_rank = self.ragged_rank + new_values.ragged_rank`.
    """
    if isinstance(self._values, ops.Tensor):
      return self.with_values(new_values)
    else:
      return self.with_values(self.values.with_inner_values(new_values))

  #=============================================================================
  # String Encoding
  #=============================================================================
  def __str__(self):
    if self._is_eager():
      return "RaggedTensor(%s)" % self.tolist()
    else:
      return self.__repr__()

  def __repr__(self):
    return "RaggedTensor(values=%s, row_splits=%s)" % (self._values,
                                                       self._row_splits)

  #=============================================================================
  # Eager Execution Mode
  #=============================================================================

  def tolist(self):
    """Returns a nested Python `list` with the values for this `RaggedTensor`.

    If a `RaggedTensor` `rt` was constructed in graph execution mode, then
    `rt.tolist()` is equivalent to `rt.eval().tolist()`.

    If a `RaggedTensor` `rt` was constructed in eager execution mode, then
    `rt.tolist()` builds the Python list based on `rt`'s `EagerTensor`
    components.

    Returns:
      A nested Python `list`.
    """
    if self._is_eager():
      return self._eager_value().tolist()
    else:
      return self.eval().tolist()

  def _eager_value(self):
    """Returns a RaggedTensorValue for self.  Requires self._is_eager()=true."""
    value = self.inner_values.numpy()
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
  # Evaluation
  #=============================================================================
  def eval(self, feed_dict=None, session=None):  # pylint: disable=redefined-outer-name
    """Evaluates this ragged tensor in a `Session`.

    Args:
      feed_dict: A dictionary that maps `Tensor` objects to feed values. See
        `tf.Session.run` for a description of the valid feed values.
      session: The `Session` to be used to evaluate this ragged tensor. If none,
        the default session will be used.

    Returns:
      A `RaggedTensorValue` object.
    """
    return _eval_using_default_session(self, feed_dict,
                                       self._as_graph_element().graph, session)

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
# Register RaggedTensor for use with session.run.
#===============================================================================
def _ragged_tensor_value_from_components(components):
  components = list(components)
  value = components.pop()
  while components:
    value = ragged_tensor_value.RaggedTensorValue(value, components.pop())
  return value


def _ragged_tensor_session_fetch(rt):
  components = rt.nested_row_splits + (rt.inner_values,)
  return (components, _ragged_tensor_value_from_components)


def _ragged_tensor_session_feed(feed_key, feed_val):
  key_components = feed_key.nested_row_splits + (feed_key.inner_values,)
  val_components = feed_val.nested_row_splits + (feed_val.inner_values,)
  return zip(key_components, val_components)


def _ragged_tensor_session_feed_for_partial_run(feed_key):
  return feed_key.nested_row_splits + (feed_key.inner_values,)


session.register_session_run_conversion_functions(
    RaggedTensor, _ragged_tensor_session_fetch, _ragged_tensor_session_feed,
    _ragged_tensor_session_feed_for_partial_run)


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
