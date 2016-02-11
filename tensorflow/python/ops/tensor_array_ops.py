# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Data Flow Operations."""
# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import gen_data_flow_ops


# pylint: disable=protected-access
class TensorArray(object):
  """Class wrapping dynamic-sized, per-time-step, write-once Tensor arrays.

  This class is meant to be used with dynamic iteration primitives such as
  `While` loops, and supports gradient back-propagation via special "flow"
  control flow dependencies.

  @@handle
  @@flow

  @@read
  @@unpack

  @@write
  @@pack

  @@grad
  """

  def __init__(self, dtype, size=None, dynamic_size=None,
               tensor_array_name=None,
               handle=None, flow=None, name=None):
    """Construct a new TensorArray or wrap an existing TensorArray handle.

    Args:
      dtype: (required) data type of the TensorArray.
      size: (optional) int32 scalar `Tensor`: the size of the TensorArray.
        Required if handle is not provided.
      dynamic_size: (optional) Python bool: If true, writes to the TensorArray
        can grow the TensorArray past its initial size.  Default: False.
      tensor_array_name: (optional) Python string: the name of the TensorArray.
        This is used when creating the TensorArray handle.  If this value is
        set, handle should be None.
      handle: (optional) A `Tensor` handle to an existing TensorArray.  If this
        is set, tensor_array_name should be None.
      flow: (optional) A float `Tensor` scalar coming from an existing
        TensorArray.flow.
      name: A name for the operation (optional).

    Raises:
      ValueError: if both handle and tensor_array_name are provided.
      TypeError: if handle is provided but is not a Tensor.
    """
    if handle and tensor_array_name:
      raise ValueError(
          "Cannot construct with both handle and tensor_array_name")
    if handle and not isinstance(handle, ops.Tensor):
      raise TypeError("Handle must be a Tensor")
    if handle is None and size is None:
      raise ValueError("Size must be provided if handle is not provided")
    if handle and size is not None:
      raise ValueError("Cannot provide both a handle and size "
                       "at the same time")
    if handle and dynamic_size is not None:
      raise ValueError("Cannot provide both a handle and dynamic_size "
                       "at the same time")

    dynamic_size = dynamic_size or False

    self._dtype = dtype
    with ops.op_scope([handle, size, flow], name, "TensorArray") as scope:
      if handle:
        self._handle = handle
      else:
        self._handle = gen_data_flow_ops._tensor_array(
            dtype=dtype, size=size, dynamic_size=dynamic_size,
            tensor_array_name=tensor_array_name, name=scope)
    self._flow = flow or constant_op.constant(0, dtype=_dtypes.float32)

  @property
  def flow(self):
    """The flow `Tensor` forcing ops leading to this TensorArray state."""
    return self._flow

  @property
  def dtype(self):
    """The data type of this TensorArray."""
    return self._dtype

  @property
  def handle(self):
    """The reference to the TensorArray."""
    return self._handle

  def grad(self, source, flow=None):
    # tensor_array_grad requires a flow input when forward
    # TensorArrays are dynamically sized.  This forces the creation
    # of the grad TensorArray only once the final forward array's size
    # is fixed.
    g_handle = gen_data_flow_ops._tensor_array_grad(
        handle=self._handle, source=source, flow_in=flow or self.flow)
    g = TensorArray(dtype=self._dtype, handle=g_handle, flow=flow or self.flow)
    return g

  def read(self, index, name=None):
    """Read the value at location `index` in the TensorArray."""
    value = gen_data_flow_ops._tensor_array_read(
        handle=self._handle, index=index, flow_in=self._flow, dtype=self._dtype,
        name=name)
    return value

  def write(self, index, value, name=None):
    """Write `value` into index `index` of the TensorArray."""
    flow_out = gen_data_flow_ops._tensor_array_write(
        handle=self._handle, index=index, value=value, flow_in=self._flow,
        name=name)
    # Size below is ignored
    ta = TensorArray(dtype=self._dtype, handle=self._handle)
    ta._flow = flow_out
    return ta

  def pack(self, name=None):
    """Return the values in the TensorArray as a packed `Tensor`."""
    value = gen_data_flow_ops._tensor_array_pack(
        handle=self._handle, flow_in=self._flow, dtype=self._dtype,
        name=name)

    return value

  def unpack(self, value, name=None):
    """Packs the values of a `Tensor` in the TensorArray."""
    flow_out = gen_data_flow_ops._tensor_array_unpack(
        handle=self._handle, value=value, flow_in=self._flow,
        name=name)
    ta = TensorArray(dtype=self._dtype, handle=self._handle)
    ta._flow = flow_out
    return ta

  def size(self, name=None):
    """Returns the size of the TensorArray."""
    return gen_data_flow_ops._tensor_array_size(
        handle=self._handle, flow_in=self.flow, name=name)

  def close(self, name=None):
    """Close the current TensorArray."""
    return gen_data_flow_ops._tensor_array_close(
        handle=self._handle, name=name)


@ops.RegisterShape("TensorArray")
def _TensorArrayShape(op):
  # size is a scalar
  op.inputs[0].get_shape().merge_with(tensor_shape.scalar())
  return [tensor_shape.vector(2)]


@ops.RegisterShape("TensorArrayRead")
def _TensorArrayReadShape(op):
  # handle, index, flow_in
  op.inputs[0].get_shape().merge_with(tensor_shape.vector(2))
  op.inputs[1].get_shape().merge_with(tensor_shape.scalar())
  op.inputs[2].get_shape().merge_with(tensor_shape.scalar())
  # value
  return [tensor_shape.unknown_shape()]


@ops.RegisterShape("TensorArrayWrite")
def _TensorArrayWriteShape(op):
  # handle, index, value, flow_in
  op.inputs[0].get_shape().merge_with(tensor_shape.vector(2))
  op.inputs[1].get_shape().merge_with(tensor_shape.scalar())
  op.inputs[3].get_shape().merge_with(tensor_shape.scalar())
  # flow_out
  return [tensor_shape.scalar()]


@ops.RegisterShape("TensorArraySize")
def _TensorArraySizeShape(op):
  op.inputs[0].get_shape().merge_with(tensor_shape.vector(2))
  return [tensor_shape.scalar()]


@ops.RegisterShape("TensorArrayClose")
def _TensorArrayCloseShape(op):
  """Shape function for ops that take a scalar and produce no outputs."""
  op.inputs[0].get_shape().merge_with(tensor_shape.vector(2))
  return []


@ops.RegisterShape("TensorArrayGrad")
def _TensorArrayGradShape(op):
  """Shape function for ops that take a scalar and produce no outputs."""
  op.inputs[0].get_shape().merge_with(tensor_shape.vector(2))
  return [tensor_shape.vector(2)]


@ops.RegisterShape("TensorArrayPack")
def _TensorArrayPackShape(op):
  # handle, flow_in
  op.inputs[0].get_shape().merge_with(tensor_shape.vector(2))
  op.inputs[1].get_shape().merge_with(tensor_shape.scalar())
  # value
  return [tensor_shape.unknown_shape()]


@ops.RegisterShape("TensorArrayUnpack")
def _TensorArrayUnpackShape(op):
  # handle, value, flow_in
  op.inputs[0].get_shape().merge_with(tensor_shape.vector(2))
  op.inputs[2].get_shape().merge_with(tensor_shape.scalar())
  # flow_out
  return [tensor_shape.scalar()]
# pylint: enable=protected-access
