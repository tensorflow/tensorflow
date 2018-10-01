# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Classes and functions used to construct graphs."""
# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.util.tf_export import tf_export

# pylint: disable=protected-access
_TensorLike = ops._TensorLike
_eval_using_default_session = ops._eval_using_default_session
_override_helper = ops._override_helper
# pylint: enable=protected-access


@tf_export("sparse.SparseTensor", "SparseTensor")
class SparseTensor(_TensorLike):
  """Represents a sparse tensor.

  TensorFlow represents a sparse tensor as three separate dense tensors:
  `indices`, `values`, and `dense_shape`.  In Python, the three tensors are
  collected into a `SparseTensor` class for ease of use.  If you have separate
  `indices`, `values`, and `dense_shape` tensors, wrap them in a `SparseTensor`
  object before passing to the ops below.

  Concretely, the sparse tensor `SparseTensor(indices, values, dense_shape)`
  comprises the following components, where `N` and `ndims` are the number
  of values and number of dimensions in the `SparseTensor`, respectively:

  * `indices`: A 2-D int64 tensor of dense_shape `[N, ndims]`, which specifies
    the indices of the elements in the sparse tensor that contain nonzero
    values (elements are zero-indexed). For example, `indices=[[1,3], [2,4]]`
    specifies that the elements with indexes of [1,3] and [2,4] have
    nonzero values.

  * `values`: A 1-D tensor of any type and dense_shape `[N]`, which supplies the
    values for each element in `indices`. For example, given
    `indices=[[1,3], [2,4]]`, the parameter `values=[18, 3.6]` specifies
    that element [1,3] of the sparse tensor has a value of 18, and element
    [2,4] of the tensor has a value of 3.6.

  * `dense_shape`: A 1-D int64 tensor of dense_shape `[ndims]`, which specifies
    the dense_shape of the sparse tensor. Takes a list indicating the number of
    elements in each dimension. For example, `dense_shape=[3,6]` specifies a
    two-dimensional 3x6 tensor, `dense_shape=[2,3,4]` specifies a
    three-dimensional 2x3x4 tensor, and `dense_shape=[9]` specifies a
    one-dimensional tensor with 9 elements.

  The corresponding dense tensor satisfies:

  ```python
  dense.shape = dense_shape
  dense[tuple(indices[i])] = values[i]
  ```

  By convention, `indices` should be sorted in row-major order (or equivalently
  lexicographic order on the tuples `indices[i]`). This is not enforced when
  `SparseTensor` objects are constructed, but most ops assume correct ordering.
  If the ordering of sparse tensor `st` is wrong, a fixed version can be
  obtained by calling `tf.sparse_reorder(st)`.

  Example: The sparse tensor

  ```python
  SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
  ```

  represents the dense tensor

  ```python
  [[1, 0, 0, 0]
   [0, 0, 2, 0]
   [0, 0, 0, 0]]
  ```
  """

  @classmethod
  def from_value(cls, sparse_tensor_value):
    if not is_sparse(sparse_tensor_value):
      raise TypeError("Neither a SparseTensor nor SparseTensorValue: %s." %
                      sparse_tensor_value)
    return SparseTensor(
        indices=sparse_tensor_value.indices,
        values=sparse_tensor_value.values,
        dense_shape=sparse_tensor_value.dense_shape)

  def __init__(self, indices, values, dense_shape):
    """Creates a `SparseTensor`.

    Args:
      indices: A 2-D int64 tensor of shape `[N, ndims]`.
      values: A 1-D tensor of any type and shape `[N]`.
      dense_shape: A 1-D int64 tensor of shape `[ndims]`.

    """
    with ops.name_scope(None, "SparseTensor",
                        [indices, values, dense_shape]):
      indices = ops.convert_to_tensor(
          indices, name="indices", dtype=dtypes.int64)
      # Always pass as_ref=True because we want to be able to update
      # values later if it is a VariableOp.
      # TODO(touts): Consider adding mutable_values() when 'values'
      # is a VariableOp and updating users of SparseTensor.
      values = ops.internal_convert_to_tensor(
          values, name="values", as_ref=True)
      dense_shape = ops.convert_to_tensor(
          dense_shape, name="dense_shape", dtype=dtypes.int64)
    self._indices = indices
    self._values = values
    self._dense_shape = dense_shape

    indices_shape = indices.get_shape().with_rank(2)
    values_shape = values.get_shape().with_rank(1)
    dense_shape_shape = dense_shape.get_shape().with_rank(1)

    # Assert number of rows in indices match the number of elements in values.
    indices_shape[0].merge_with(values_shape[0])
    # Assert number of columns in indices matches the number of elements in
    # dense_shape.
    indices_shape[1].merge_with(dense_shape_shape[0])

  def get_shape(self):
    """Get the `TensorShape` representing the shape of the dense tensor.

    Returns:
      A `TensorShape` object.
    """
    return tensor_util.constant_value_as_shape(self._dense_shape)

  @property
  def indices(self):
    """The indices of non-zero values in the represented dense tensor.

    Returns:
      A 2-D Tensor of int64 with dense_shape `[N, ndims]`, where `N` is the
        number of non-zero values in the tensor, and `ndims` is the rank.
    """
    return self._indices

  @property
  def values(self):
    """The non-zero values in the represented dense tensor.

    Returns:
      A 1-D Tensor of any data type.
    """
    return self._values

  @property
  def op(self):
    """The `Operation` that produces `values` as an output."""
    return self.values.op

  @property
  def dtype(self):
    """The `DType` of elements in this tensor."""
    return self._values.dtype

  @property
  def dense_shape(self):
    """A 1-D Tensor of int64 representing the shape of the dense tensor."""
    return self._dense_shape

  @property
  def shape(self):
    """Get the `TensorShape` representing the shape of the dense tensor.

    Returns:
      A `TensorShape` object.
    """
    return tensor_util.constant_value_as_shape(self._dense_shape)

  @property
  def graph(self):
    """The `Graph` that contains the index, value, and dense_shape tensors."""
    return self._indices.graph

  def consumers(self):
    """Returns a list of `Operation`s that consume this `SparseTensor`.

    Returns:
      A list of `Operation`s.
    """
    values_consumers = set(self._values.consumers())
    indices_consumers = set(self._indices.consumers())
    dense_shape_consumers = set(self._dense_shape.consumers())
    return list(values_consumers \
                .union(indices_consumers, dense_shape_consumers))

  def __str__(self):
    return "SparseTensor(indices=%s, values=%s, dense_shape=%s)" % (
        self._indices, self._values, self._dense_shape)

  def eval(self, feed_dict=None, session=None):
    """Evaluates this sparse tensor in a `Session`.

    Calling this method will execute all preceding operations that
    produce the inputs needed for the operation that produces this
    tensor.

    *N.B.* Before invoking `SparseTensor.eval()`, its graph must have been
    launched in a session, and either a default session must be
    available, or `session` must be specified explicitly.

    Args:
      feed_dict: A dictionary that maps `Tensor` objects to feed values.
        See `tf.Session.run` for a
        description of the valid feed values.
      session: (Optional.) The `Session` to be used to evaluate this sparse
        tensor. If none, the default session will be used.

    Returns:
      A `SparseTensorValue` object.
    """
    indices, values, dense_shape = _eval_using_default_session(
        [self.indices, self.values, self.dense_shape], feed_dict, self.graph,
        session)
    return SparseTensorValue(indices, values, dense_shape)

  @staticmethod
  def _override_operator(operator, func):
    _override_helper(SparseTensor, operator, func)


SparseTensorValue = collections.namedtuple(
    "SparseTensorValue", ["indices", "values", "dense_shape"])
tf_export("SparseTensorValue")(SparseTensorValue)
pywrap_tensorflow.RegisterType("SparseTensorValue", SparseTensorValue)


@tf_export("convert_to_tensor_or_sparse_tensor")
def convert_to_tensor_or_sparse_tensor(value, dtype=None, name=None):
  """Converts value to a `SparseTensor` or `Tensor`.

  Args:
    value: A `SparseTensor`, `SparseTensorValue`, or an object whose type has a
      registered `Tensor` conversion function.
    dtype: Optional element type for the returned tensor. If missing, the
      type is inferred from the type of `value`.
    name: Optional name to use if a new `Tensor` is created.

  Returns:
    A `SparseTensor` or `Tensor` based on `value`.

  Raises:
    RuntimeError: If result type is incompatible with `dtype`.
  """
  if dtype is not None:
    dtype = dtypes.as_dtype(dtype)
  if isinstance(value, SparseTensorValue):
    value = SparseTensor.from_value(value)
  if isinstance(value, SparseTensor):
    if dtype and not dtype.is_compatible_with(value.dtype):
      raise RuntimeError(
          "Sparse dtype: requested = %s, actual = %s" % (
              dtype.name, value.dtype.name))
    return value
  return ops.internal_convert_to_tensor(
      value, dtype=dtype, name=name)


def is_sparse(x):
  """Check whether `x` is sparse.

  Check whether an object is a `tf.SparseTensor` or `tf.SparseTensorValue`.

  Args:
    x: A python object to check.

  Returns:
    `True` iff `x` is a `tf.SparseTensor` or `tf.SparseTensorValue`.
  """
  return isinstance(x, (SparseTensor, SparseTensorValue))
