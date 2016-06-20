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

"""TensorArray operations.

## Classes containing dynamically sized arrays of Tensors.

@@TensorArray
"""
# Mixture of pep8 and non-pep8 names, so disable pylint bad-name
# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import math_ops


# TensorArray object accesses many of the hidden generated ops, but is
# in fact built to wrap these methods.
# pylint: disable=protected-access
class TensorArray(object):
  """Class wrapping dynamic-sized, per-time-step, write-once Tensor arrays.

  This class is meant to be used with dynamic iteration primitives such as
  `while_loop` and `map_fn`.  It supports gradient back-propagation via special
  "flow" control flow dependencies.

  @@handle
  @@flow

  @@read
  @@unpack
  @@split

  @@write
  @@pack
  @@concat

  @@grad
  """

  def __init__(self, dtype, size=None, dynamic_size=None,
               clear_after_read=None, tensor_array_name=None, handle=None,
               flow=None, infer_shape=True, name=None):
    """Construct a new TensorArray or wrap an existing TensorArray handle.

    A note about the parameter `name`:

    The name of the `TensorArray` (even if passed in) is uniquified: each time
    a new `TensorArray` is created at runtime it is assigned its own name for
    the duration of the run.  This avoids name collissions if a `TensorArray`
    is created within a `while_loop`.

    Args:
      dtype: (required) data type of the TensorArray.
      size: (optional) int32 scalar `Tensor`: the size of the TensorArray.
        Required if handle is not provided.
      dynamic_size: (optional) Python bool: If true, writes to the TensorArray
        can grow the TensorArray past its initial size.  Default: False.
      clear_after_read: Boolean (optional, default: True).  If True, clear
        TensorArray values after reading them.  This disables read-many
        semantics, but allows early release of memory.
      tensor_array_name: (optional) Python string: the name of the TensorArray.
        This is used when creating the TensorArray handle.  If this value is
        set, handle should be None.
      handle: (optional) A `Tensor` handle to an existing TensorArray.  If this
        is set, tensor_array_name should be None.
      flow: (optional) A float `Tensor` scalar coming from an existing
        `TensorArray.flow`.
      infer_shape: (optional, default: True) If True, shape inference
        is enabled.  In this case, all elements must have the same shape.
      name: A name for the operation (optional).

    Raises:
      ValueError: if both handle and tensor_array_name are provided.
      TypeError: if handle is provided but is not a Tensor.
    """
    if handle is not None and tensor_array_name:
      raise ValueError(
          "Cannot construct with both handle and tensor_array_name")
    if handle is not None and not isinstance(handle, ops.Tensor):
      raise TypeError("Handle must be a Tensor")
    if handle is None and size is None:
      raise ValueError("Size must be provided if handle is not provided")
    if handle is not None and size is not None:
      raise ValueError("Cannot provide both a handle and size "
                       "at the same time")
    if handle is not None and dynamic_size is not None:
      raise ValueError("Cannot provide both a handle and dynamic_size "
                       "at the same time")
    if handle is not None and clear_after_read is not None:
      raise ValueError("Cannot provide both a handle and clear_after_read "
                       "at the same time")

    if clear_after_read is None:
      clear_after_read = True
    dynamic_size = dynamic_size or False

    self._dtype = dtype
    self._infer_shape = infer_shape
    # Record the current static shape for the array elements. The first
    # write adds the shape of the tensor it writes, and all subsequent
    # writes checks for shape equality.
    self._elem_shape = []
    with ops.op_scope([handle, size, flow], name, "TensorArray") as scope:
      if handle is not None:
        self._handle = handle
      else:
        if flow is not None:
          with ops.colocate_with(flow):
            self._handle = gen_data_flow_ops._tensor_array(
                dtype=dtype, size=size, dynamic_size=dynamic_size,
                clear_after_read=clear_after_read,
                tensor_array_name=tensor_array_name, name=scope)
        else:
          self._handle = gen_data_flow_ops._tensor_array(
              dtype=dtype, size=size, dynamic_size=dynamic_size,
              clear_after_read=clear_after_read,
              tensor_array_name=tensor_array_name, name=scope)
      if flow is not None:
        self._flow = flow
      else:
        with ops.colocate_with(self._handle):
          self._flow = constant_op.constant(0, dtype=_dtypes.float32)

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

  def grad(self, source, flow=None, name=None):
    # tensor_array_grad requires a flow input when forward
    # TensorArrays are dynamically sized.  This forces the creation
    # of the grad TensorArray only once the final forward array's size
    # is fixed.
    if flow is None:
      flow = self.flow
    with ops.op_scope([self._handle], name, "TensorArrayGrad"):
      with ops.colocate_with(self._handle):
        g_handle = gen_data_flow_ops._tensor_array_grad(
            handle=self._handle, source=source, flow_in=flow, name=name)
        with ops.control_dependencies([g_handle]):
          flow = array_ops.identity(flow, name="gradient_flow")
        g = TensorArray(dtype=self._dtype, handle=g_handle, flow=flow,
                        infer_shape=self._infer_shape)
        return g

  def read(self, index, name=None):
    """Read the value at location `index` in the TensorArray.

    Args:
      index: 0-D.  int32 tensor with the index to read from.
      name: A name for the operation (optional).

    Returns:
      The tensor at index `index`.
    """
    with ops.colocate_with(self._handle):
      value = gen_data_flow_ops._tensor_array_read(
          handle=self._handle, index=index, flow_in=self._flow,
          dtype=self._dtype, name=name)
      if self._elem_shape:
        value.set_shape(self._elem_shape[0].dims)
      return value

  def write(self, index, value, name=None):
    """Write `value` into index `index` of the TensorArray.

    Args:
      index: 0-D.  int32 scalar with the index to write to.
      value: N-D.  Tensor of type `dtype`.  The Tensor to write to this index.
      name: A name for the operation (optional).

    Returns:
      A new TensorArray object with flow that ensures the write occurs.
      Use this object all for subsequent operations.

    Raises:
      ValueError: if there are more writers than specified.
    """
    with ops.colocate_with(self._handle):
      flow_out = gen_data_flow_ops._tensor_array_write(
          handle=self._handle, index=index, value=value, flow_in=self._flow,
          name=name)
      ta = TensorArray(dtype=self._dtype, handle=self._handle)
      ta._flow = flow_out
      ta._infer_shape = self._infer_shape
      ta._elem_shape = self._elem_shape
      if ta._infer_shape:
        val_shape = flow_out.op.inputs[2].get_shape()
        if ta._elem_shape:
          if not val_shape == ta._elem_shape[0]:
            raise ValueError(
                "Inconsistent shapes: saw %s but expected %s "
                "(and infer_shape=True)" % (val_shape, ta._elem_shape[0]))
        else:
          ta._elem_shape.append(val_shape)
      return ta

  def pack(self, name=None):
    """Return the values in the TensorArray as a packed `Tensor`.

    All of the values must have been written and their shapes must all match.

    Args:
      name: A name for the operation (optional).

    Returns:
      All the tensors in the TensorArray packed into one tensor.
    """
    with ops.colocate_with(self._handle):
      value = gen_data_flow_ops._tensor_array_pack(
          handle=self._handle, flow_in=self._flow, dtype=self._dtype,
          name=name)
      if self._elem_shape and self._elem_shape[0].dims is not None:
        value.set_shape([None] + self._elem_shape[0].dims)
      return value

  def concat(self, name=None):
    """Return the values in the TensorArray as a concatenated `Tensor`.

    All of the values must have been written, their ranks must match, and
    and their shapes must all match for all dimensions except the first.

    Args:
      name: A name for the operation (optional).

    Returns:
      All the tensors in the TensorArray concatenated into one tensor.
    """
    with ops.colocate_with(self._handle):
      value, _ = gen_data_flow_ops._tensor_array_concat(
          handle=self._handle, flow_in=self._flow, dtype=self._dtype,
          name=name)
      if self._elem_shape and self._elem_shape[0].dims is not None:
        value.set_shape([None] + self._elem_shape[0].dims[1:])
      return value

  def unpack(self, value, name=None):
    """Pack the values of a `Tensor` in the TensorArray.

    Args:
      value: (N+1)-D.  Tensor of type `dtype`.  The Tensor to unpack.
      name: A name for the operation (optional).

    Returns:
      A new TensorArray object with flow that ensures the unpack occurs.
      Use this object all for subsequent operations.

    Raises:
      ValueError: if the shape inference fails.
    """
    with ops.colocate_with(self._handle):
      flow_out = gen_data_flow_ops._tensor_array_unpack(
          handle=self._handle, value=value, flow_in=self._flow,
          name=name)
      ta = TensorArray(dtype=self._dtype, handle=self._handle)
      ta._flow = flow_out
      ta._infer_shape = self._infer_shape
      ta._elem_shape = self._elem_shape
      if ta._infer_shape:
        val_shape = flow_out.op.inputs[1].get_shape()
        elem_shape = tensor_shape.unknown_shape()
        if val_shape.dims is not None:
          elem_shape = tensor_shape.TensorShape(val_shape.dims[1:])
        if ta._elem_shape:
          if not elem_shape == ta._elem_shape[0]:
            raise ValueError(
                "Inconsistent shapes: saw %s but expected %s "
                "(and infer_shape=True)" % (elem_shape, ta._elem_shape[0]))
        else:
          ta._elem_shape.append(elem_shape)
      return ta

  def split(self, value, lengths, name=None):
    """Split the values of a `Tensor` into the TensorArray.

    Args:
      value: (N+1)-D.  Tensor of type `dtype`.  The Tensor to split.
      lengths: 1-D.  int32 vector with the lengths to use when splitting
        `value` along its first dimension.
      name: A name for the operation (optional).

    Returns:
      A new TensorArray object with flow that ensures the split occurs.
      Use this object all for subsequent operations.

    Raises:
      ValueError: if the shape inference fails.
    """
    with ops.colocate_with(self._handle):
      with ops.op_scope(
          [self._handle, value, lengths], name, "TensorArraySplit"):
        lengths_64 = math_ops.to_int64(lengths)
      flow_out = gen_data_flow_ops._tensor_array_split(
          handle=self._handle, value=value, lengths=lengths_64,
          flow_in=self._flow, name=name)
      ta = TensorArray(dtype=self._dtype, handle=self._handle)
      ta._flow = flow_out
      ta._infer_shape = self._infer_shape
      ta._elem_shape = self._elem_shape
      if ta._infer_shape:
        val_shape = flow_out.op.inputs[1].get_shape()
        clengths = tensor_util.constant_value(flow_out.op.inputs[2])
        elem_shape = tensor_shape.unknown_shape()
        if val_shape.dims is not None:
          if clengths is not None and clengths.max() == clengths.min():
            elem_shape = tensor_shape.TensorShape(
                [clengths[0]] + val_shape.dims[1:])
        if ta._elem_shape:
          if not elem_shape == ta._elem_shape[0]:
            raise ValueError(
                "Inconsistent shapes: saw %s but expected %s "
                "(and infer_shape=True)" % (elem_shape, ta._elem_shape[0]))
        else:
          ta._elem_shape.append(elem_shape)
      return ta

  def size(self, name=None):
    """Return the size of the TensorArray."""
    with ops.colocate_with(self._handle):
      return gen_data_flow_ops._tensor_array_size(
          handle=self._handle, flow_in=self.flow, name=name)

  def close(self, name=None):
    """Close the current TensorArray."""
    with ops.colocate_with(self._handle):
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


@ops.RegisterShape("TensorArrayConcat")
def _TensorArrayConcatShape(op):
  # handle, flow_in
  op.inputs[0].get_shape().merge_with(tensor_shape.vector(2))
  op.inputs[1].get_shape().merge_with(tensor_shape.scalar())
  # value, lengths
  return [tensor_shape.unknown_shape(), tensor_shape.vector(None)]


@ops.RegisterShape("TensorArraySplit")
def _TensorArraySplitShape(op):
  # handle, value, lengths, flow_in
  op.inputs[0].get_shape().merge_with(tensor_shape.vector(2))
  op.inputs[2].get_shape().merge_with(tensor_shape.vector(None))
  op.inputs[3].get_shape().merge_with(tensor_shape.scalar())
  # flow_out
  return [tensor_shape.scalar()]


@ops.RegisterShape("TensorArrayUnpack")
def _TensorArrayUnpackShape(op):
  # handle, value, flow_in
  op.inputs[0].get_shape().merge_with(tensor_shape.vector(2))
  op.inputs[2].get_shape().merge_with(tensor_shape.scalar())
  # flow_out
  return [tensor_shape.scalar()]

# pylint: enable=protected-access
