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
"""TensorArray: a dynamically sized array of Tensors.

@@TensorArray
"""
# Mixture of pep8 and non-pep8 names, so disable pylint bad-name
# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import math_ops


def _maybe_set_device(handle_op, value_t):
  # NOTE(ebrevdo): Do not try this at home, kids
  # _______________________________________________
  # | I WILL NOT ACCESS PRIVATE METHODS ^^^^^^^^\ |
  # | I WILL NOT ACCESS PRIVATE METHODS |       | |
  # | I WILL NOT ACCESS PRIVATE METHODS |_ __   | |
  # | I WILL NOT ACCESS PRIVATE METHODS (.(. )  | |
  # | I WILL NOT ACCESS PRIVATE         (_      ) |
  # |                           \\      /___/' /  |
  # |                           _\\_      \    |  |
  # |                          ((   )     /====|  |
  # |                           \  <.__._-      \ |
  # |___________________________ <//___.         ||
  #
  if not handle_op.device and value_t.device:
    handle_op._set_device(value_t.device)  # pylint: disable=protected-access


# TensorArray object accesses many of the hidden generated ops, but is
# in fact built to wrap these methods.
# pylint: disable=protected-access
class TensorArray(object):
  """Class wrapping dynamic-sized, per-time-step, write-once Tensor arrays.

  This class is meant to be used with dynamic iteration primitives such as
  `while_loop` and `map_fn`.  It supports gradient back-propagation via special
  "flow" control flow dependencies.
  """

  def __init__(self,
               dtype,
               size=None,
               dynamic_size=None,
               clear_after_read=None,
               tensor_array_name=None,
               handle=None,
               flow=None,
               infer_shape=True,
               element_shape=None,
               name=None):
    """Construct a new TensorArray or wrap an existing TensorArray handle.

    A note about the parameter `name`:

    The name of the `TensorArray` (even if passed in) is uniquified: each time
    a new `TensorArray` is created at runtime it is assigned its own name for
    the duration of the run.  This avoids name collisions if a `TensorArray`
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
      element_shape: (optional, default: None) A `TensorShape` object specifying
        the shape constraints of each of the elements of the TensorArray.
        Need not be fully defined.
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
    if handle is not None and element_shape is not None:
      raise ValueError("Cannot provide both a handle and element_shape "
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
    # Record the current static shape for the array elements. The element
    # shape is defined either by `element_shape` or the shape of the tensor
    # of the first write. If `infer_shape` is true, all writes checks for
    # shape equality.
    if element_shape is None:
      self._infer_shape = infer_shape
      self._element_shape = []
    else:
      self._infer_shape = True
      self._element_shape = [tensor_shape.TensorShape(element_shape)]
    with ops.name_scope(name, "TensorArray", [handle, size, flow]) as scope:
      if handle is not None:
        self._handle = handle
        if flow is None:
          raise ValueError("flow must not be None if handle is not None.")
        self._flow = flow
      else:
        # Construct the TensorArray with an empty device.  The first
        # write into the TensorArray from a Tensor with a set device
        # will retroactively set the device value of this op.
        with ops.device(None), ops.colocate_with(None, ignore_existing=True):
          self._handle, self._flow = gen_data_flow_ops._tensor_array_v3(
              dtype=dtype,
              size=size,
              element_shape=element_shape,
              dynamic_size=dynamic_size,
              clear_after_read=clear_after_read,
              tensor_array_name=tensor_array_name,
              name=scope)

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

  def _merge_element_shape(self, shape):
    """Changes the element shape of the array given a shape to merge with.

    Args:
      shape: A `TensorShape` object to merge with.

    Raises:
      ValueError: if the provided shape is incompatible with the current
          element shape of the `TensorArray`.
    """

    if self._element_shape:
      if not shape.is_compatible_with(self._element_shape[0]):
        raise ValueError(
            "Inconsistent shapes: saw %s but expected %s "
            "(and infer_shape=True)" % (shape, self._element_shape[0]))
      self._element_shape[0] = self._element_shape[0].merge_with(shape)
    else:
      self._element_shape.append(shape)

  def identity(self):
    """Returns a TensorArray with the same content and properties.

    Returns:
      A new TensorArray object with flow that ensures the control dependencies
      from the contexts will become control dependencies for writes, reads, etc.
      Use this object all for subsequent operations.
    """
    flow = array_ops.identity(self._flow)
    ta = TensorArray(dtype=self._dtype, handle=self._handle, flow=flow,
                     infer_shape=self._infer_shape)
    ta._element_shape = self._element_shape
    return ta

  def grad(self, source, flow=None, name=None):
    # tensor_array_grad requires a flow input when forward
    # TensorArrays are dynamically sized.  This forces the creation
    # of the grad TensorArray only once the final forward array's size
    # is fixed.
    if flow is None:
      flow = self.flow
    with ops.name_scope(name, "TensorArrayGrad", [self._handle]):
      with ops.colocate_with(self._handle):
        g_handle, unused_flow = gen_data_flow_ops._tensor_array_grad_v3(
            handle=self._handle, source=source, flow_in=flow, name=name)
        with ops.control_dependencies([g_handle]):
          flow = array_ops.identity(flow, name="gradient_flow")
        g = TensorArray(
            dtype=self._dtype,
            handle=g_handle,
            flow=flow,
            infer_shape=self._infer_shape)
        g._element_shape = self._element_shape
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
      value = gen_data_flow_ops._tensor_array_read_v3(
          handle=self._handle,
          index=index,
          flow_in=self._flow,
          dtype=self._dtype,
          name=name)
      if self._element_shape:
        value.set_shape(self._element_shape[0].dims)
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
    with ops.name_scope(name, "TensorArrayWrite", [self._handle, index, value]):
      value = ops.convert_to_tensor(value, name="value")
      _maybe_set_device(self._handle.op, value)
      with ops.colocate_with(self._handle):
        flow_out = gen_data_flow_ops._tensor_array_write_v3(
            handle=self._handle,
            index=index,
            value=value,
            flow_in=self._flow,
            name=name)
      ta = TensorArray(dtype=self._dtype, handle=self._handle, flow=flow_out)
      ta._infer_shape = self._infer_shape
      ta._element_shape = self._element_shape
      if ta._infer_shape:
        ta._merge_element_shape(value.get_shape())
      return ta

  def stack(self, name=None):
    """Return the values in the TensorArray as a stacked `Tensor`.

    All of the values must have been written and their shapes must all match.
    If input shapes have rank-`R`, then output shape will have rank-`(R+1)`.

    Args:
      name: A name for the operation (optional).

    Returns:
      All the tensors in the TensorArray stacked into one tensor.
    """
    with ops.colocate_with(self._handle):
      with ops.name_scope(name, "TensorArrayStack", [self._handle]):
        return self.gather(math_ops.range(0, self.size()), name=name)

  def gather(self, indices, name=None):
    """Return selected values in the TensorArray as a packed `Tensor`.

    All of selected values must have been written and their shapes
    must all match.

    Args:
      indices: A `1-D` `Tensor` taking values in `[0, max_value)`.  If
        the `TensorArray` is not dynamic, `max_value=size()`.
      name: A name for the operation (optional).

    Returns:
      The in the `TensorArray` selected by `indices`, packed into one tensor.
    """
    with ops.colocate_with(self._handle):
      if self._element_shape:
        element_shape = self._element_shape[0]
      else:
        element_shape = tensor_shape.TensorShape(None)
      value = gen_data_flow_ops._tensor_array_gather_v3(
          handle=self._handle,
          indices=indices,
          flow_in=self._flow,
          dtype=self._dtype,
          name=name,
          element_shape=element_shape)
      if self._element_shape and self._element_shape[0].dims is not None:
        value.set_shape([None] + self._element_shape[0].dims)
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
    if self._element_shape and self._element_shape[0].dims is not None:
      element_shape_except0 = (
          tensor_shape.TensorShape(self._element_shape[0].dims[1:]))
    else:
      element_shape_except0 = tensor_shape.TensorShape(None)
    with ops.colocate_with(self._handle):
      value, _ = gen_data_flow_ops._tensor_array_concat_v3(
          handle=self._handle,
          flow_in=self._flow,
          dtype=self._dtype,
          name=name,
          element_shape_except0=element_shape_except0)
      if self._element_shape and self._element_shape[0].dims is not None:
        value.set_shape([None] + self._element_shape[0].dims[1:])
      return value

  def unstack(self, value, name=None):
    """Unstack the values of a `Tensor` in the TensorArray.

    If input value shapes have rank-`R`, then the output TensorArray will
    contain elements whose shapes are rank-`(R-1)`.
    Args:
      value: (N+1)-D.  Tensor of type `dtype`.  The Tensor to unstack.
      name: A name for the operation (optional).

    Returns:
      A new TensorArray object with flow that ensures the unstack occurs.
      Use this object all for subsequent operations.

    Raises:
      ValueError: if the shape inference fails.
    """
    with ops.name_scope(name, "TensorArrayUnstack", [self._handle, value]):
      num_elements = array_ops.shape(value)[0]
      return self.scatter(
          indices=math_ops.range(0, num_elements), value=value, name=name)

  def scatter(self, indices, value, name=None):
    """Scatter the values of a `Tensor` in specific indices of a `TensorArray`.

    Args:
      indices: A `1-D` `Tensor` taking values in `[0, max_value)`.  If
        the `TensorArray` is not dynamic, `max_value=size()`.
      value: (N+1)-D.  Tensor of type `dtype`.  The Tensor to unpack.
      name: A name for the operation (optional).

    Returns:
      A new TensorArray object with flow that ensures the scatter occurs.
      Use this object all for subsequent operations.

    Raises:
      ValueError: if the shape inference fails.
    """
    with ops.name_scope(name, "TensorArrayScatter",
                        [self._handle, value, indices]):
      value = ops.convert_to_tensor(value, name="value")
      _maybe_set_device(self._handle.op, value)
      with ops.colocate_with(self._handle):
        flow_out = gen_data_flow_ops._tensor_array_scatter_v3(
            handle=self._handle,
            indices=indices,
            value=value,
            flow_in=self._flow,
            name=name)
      ta = TensorArray(dtype=self._dtype, handle=self._handle, flow=flow_out)
      ta._infer_shape = self._infer_shape
      ta._element_shape = self._element_shape
      if ta._infer_shape:
        val_shape = flow_out.op.inputs[2].get_shape()
        element_shape = tensor_shape.unknown_shape()
        if val_shape.dims is not None:
          element_shape = tensor_shape.TensorShape(val_shape.dims[1:])
        ta._merge_element_shape(element_shape)
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
    with ops.name_scope(name, "TensorArraySplit",
                        [self._handle, value, lengths]):
      value = ops.convert_to_tensor(value, name="value")
      _maybe_set_device(self._handle.op, value)
      lengths_64 = math_ops.to_int64(lengths)
      with ops.colocate_with(self._handle):
        flow_out = gen_data_flow_ops._tensor_array_split_v3(
            handle=self._handle,
            value=value,
            lengths=lengths_64,
            flow_in=self._flow,
            name=name)
      ta = TensorArray(dtype=self._dtype, handle=self._handle, flow=flow_out)
      ta._infer_shape = self._infer_shape
      ta._element_shape = self._element_shape
      if ta._infer_shape:
        val_shape = flow_out.op.inputs[1].get_shape()
        clengths = tensor_util.constant_value(flow_out.op.inputs[2])
        element_shape = tensor_shape.unknown_shape()
        if val_shape.dims is not None:
          if clengths is not None and clengths.max() == clengths.min():
            element_shape = tensor_shape.TensorShape([clengths[0]] +
                                                     val_shape.dims[1:])
        ta._merge_element_shape(element_shape)
      return ta

  def size(self, name=None):
    """Return the size of the TensorArray."""
    with ops.colocate_with(self._handle):
      return gen_data_flow_ops._tensor_array_size_v3(
          handle=self._handle, flow_in=self.flow, name=name)

  def close(self, name=None):
    """Close the current TensorArray."""
    with ops.colocate_with(self._handle):
      return gen_data_flow_ops._tensor_array_close_v3(
          handle=self._handle, name=name)

# pylint: enable=protected-access
