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
"""TensorNode for autograd tracing of computations with Tensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from autograd import core as ag_core

from tensorflow.python.eager import context
from tensorflow.python.eager import custom_gradient
from tensorflow.python.eager import tape
from tensorflow.python.eager import tensor
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


@ag_core.primitive
def _tensor_numpy(t):
  return t.numpy()


@ag_core.primitive
def _as_gpu_tensor(t, index=0):
  return t.as_gpu_tensor(gpu_index=index)


_as_gpu_tensor.defvjp(
    lambda g, ans, vs, gvs, t, index: g.as_cpu_tensor(), argnum=0)


@custom_gradient.custom_gradient
def _tensor_copy(t, ctx=None, device_name=None):

  def grad(dresult):
    return dresult._copy(device_name=t.device)  # pylint: disable=protected-access

  return t.value._copy(ctx=ctx, device_name=device_name), grad  # pylint: disable=protected-access


@ag_core.primitive
def _as_cpu_tensor(t):
  return t.as_cpu_tensor()


_as_cpu_tensor.defvjp(lambda g, ans, vs, gvs, t: g.as_gpu_tensor(), argnum=0)


# TODO(apassos,ashankar): The operator overrides here need to be kept in sync
# with the overrides for ops.Tensor and ops.EagerTensor.
#
# Note that we cannot use self.value.__op__() because that would result
# in an ops.EagerTensor instead of a TensorNode being returned.
#
# We need to figure out a way to ensure that the two are in sync.
class TensorNode(ag_core.Node):
  """A TensorFlow Tensor."""

  __slots__ = []

  def __getitem__(self, idx):
    return array_ops._SliceHelper(self, idx)  # pylint: disable=protected-access

  shape = property(lambda self: self.value.shape)
  dtype = property(lambda self: self.value.dtype)
  device = property(lambda self: self.value.device)

  def get_shape(self):
    return self.shape

  def numpy(self):
    return _tensor_numpy(self)

  def _shape_tuple(self):
    return self.value._shape_tuple  # pylint: disable=protected-access

  def as_cpu_tensor(self):
    return _as_cpu_tensor(self)

  def as_gpu_tensor(self, gpu_index=0):
    return _as_gpu_tensor(self, gpu_index)

  def _copy(self, ctx=None, device_name=None):
    return _tensor_copy(self, ctx, device_name)

  def __neg__(self):
    return math_ops.negative(self)

  def __abs__(self):
    return math_ops.abs(self)  # pylint: disable=protected-access

  def __invert__(self):
    # ops.Tensor used math_ops.logical_not as of August 2017.
    # Now that bitwise_ops.invert exists, it might make sense
    # for both ops.Tensor and TensorNode to use that if the
    # type is compatible.
    return math_ops.logical_not(self)

  def __hash__(self):
    return id(self)

  def __add__(self, other):
    if isinstance(self.value, tensor.LazyZero):
      return other
    if isinstance(other, tensor.LazyZero):
      return self
    return math_ops.add(self, other)

  def __radd__(self, other):
    if isinstance(self.value, tensor.LazyZero):
      return other
    if isinstance(ag_core.getval(other), tensor.LazyZero):
      return self
    return math_ops.add(other, self)

  def __sub__(self, other):
    return math_ops.subtract(self, other)

  def __rsub__(self, other):
    return math_ops.subtract(other, self)

  def __mul__(self, other):
    return math_ops.multiply(self, other)

  def __rmul__(self, other):
    return math_ops.multiply(other, self)

  def __mod__(self, other):
    return math_ops.floormod(self, other)

  def __rmod__(self, other):
    return math_ops.floormod(other, self)

  def __pow__(self, other):
    return math_ops.pow(self, other)

  def __rpow__(self, other):
    return math_ops.pow(other, self)

  def __div__(self, other):
    return math_ops._div_python2(self, other)  # pylint: disable=protected-access

  def __rdiv__(self, other):
    return math_ops._div_python2(other, self)  # pylint: disable=protected-access

  def __truediv__(self, other):
    return math_ops._truediv_python3(self, other)  # pylint: disable=protected-access

  def __rtruediv__(self, other):
    return math_ops._truediv_python3(other, self)  # pylint: disable=protected-access

  def __floordiv__(self, other):
    return math_ops.floordiv(self, other)

  def __rfloordiv__(self, other):
    return math_ops.floordiv(other, self)

  def __eq__(self, other):
    # math_ops.equal raises an error if shapes are not compatible, so check that
    # explicitly first.
    if common_shapes.is_broadcast_compatible(
        self.shape, ops.convert_to_tensor(other).shape):
      return math_ops.equal(self, other)
    return False

  def __gt__(self, other):
    return math_ops.greater(self, other)

  def __ge__(self, other):
    return math_ops.greater_equal(self, other)

  def __lt__(self, other):
    return math_ops.less(self, other)

  def __le__(self, other):
    return math_ops.less_equal(self, other)


ag_core.register_node(TensorNode, tensor.Tensor)
ag_core.register_node(TensorNode, ops.Tensor)


def _zeros(shape, dtype):
  with context.device("cpu:0"):
    shape = tensor.Tensor(shape, dtype=dtypes.int32)
  return array_ops.fill(shape, tensor.Tensor(0, dtype=dtype))


def _ones(shape, dtype):
  return array_ops.fill(
      tensor.Tensor(shape, dtype=dtypes.int32), tensor.Tensor(1, dtype=dtype))


def _lazy_zero_tensor(zero):
  return _zeros(zero.shape, zero.dtype)


tensor.LazyZero.tensor = _lazy_zero_tensor


def _lazy_zero_to_tensor(lazy_zero, dtype=None, name=None, as_ref=False):
  del as_ref, name, dtype
  return _zeros(lazy_zero.shape, lazy_zero.dtype)


ops.register_tensor_conversion_function(tensor.LazyZero, _lazy_zero_to_tensor)


def _indexed_slices_to_tensor(value):
  """Converts an IndexedSlices object `value` to a Tensor.

  Args:
    value: An ops.IndexedSlices object.

  Returns:
    A dense Tensor representing the values in the given IndexedSlices.

  Raises:
    ValueError: If the IndexedSlices does not have the same dtype.
  """
  if value.dense_shape is None:
    raise ValueError(
        "Tensor conversion requested for IndexedSlices without dense_shape: %s"
        % str(value))
  return math_ops.unsorted_segment_sum(value.values, value.indices,
                                       value.dense_shape[0])


class TensorVSpace(ag_core.VSpace):
  """VSpace for tf/tfe Tensors in autograd."""

  def __init__(self, value):
    if isinstance(value, ops.IndexedSlices):
      self.shape = tensor_shape.TensorShape(value.dense_shape.numpy())
      self.dtype = value.values.dtype
    else:
      self.shape = value.shape
      self.dtype = value.dtype
    self.size = self.shape.num_elements()
    # TODO(apassos) put gradients on the same device as ops.

  def __eq__(self, other):
    if isinstance(other, tape.NoneVSpace):
      return True
    if self.dtype == dtypes.resource or other.dtype == dtypes.resource:
      return True
    return (type(self) == type(other)  # pylint: disable=unidiomatic-typecheck
            and self.dtype == other.dtype)

  def __ne__(self, other):
    return not self.__eq__(other)

  def zeros(self):
    return tensor.LazyZero(self.shape, self.dtype)

  def ones(self):
    return _ones(self.shape, self.dtype)

  def standard_basis(self):
    raise NotImplementedError

  def flatten(self, value):
    return array_ops.reshape(value, tensor.Tensor(-1))

  def unflatten(self, value):
    return array_ops.reshape(value, tensor.Tensor(self.shape))

  def mut_add(self, x, y):
    """Add wrapper safe for IndexedSlices and LazyZero."""
    if isinstance(ag_core.getval(x), tensor.LazyZero):
      return y
    if isinstance(ag_core.getval(y), tensor.LazyZero):
      return x
    if isinstance(x, ops.IndexedSlices):
      x = _indexed_slices_to_tensor(x)
    if isinstance(y, ops.IndexedSlices):
      y = _indexed_slices_to_tensor(y)
    return math_ops.add(x, y)


ag_core.register_vspace(TensorVSpace, tensor.Tensor)
ag_core.register_vspace(TensorVSpace, ops.Tensor)
ag_core.register_vspace(TensorVSpace, ops.IndexedSlices)
ag_core.register_vspace(TensorVSpace, tensor.LazyZero)
ag_core.register_node(TensorNode, tensor.LazyZero)
