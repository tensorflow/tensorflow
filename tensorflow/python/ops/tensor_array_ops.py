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
"""TensorArray: a dynamically sized array of Tensors."""
# Mixture of pep8 and non-pep8 names, so disable pylint bad-name
# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import weakref

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import tf_should_use
from tensorflow.python.util.tf_export import tf_export


# _GraphTensorArray accesses many of the hidden generated ops, but is in
# fact built to wrap these methods.
# pylint: disable=protected-access
class _GraphTensorArray(object):
  """Graph-mode implementation of TensorArray.
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
               colocate_with_first_write_call=True,
               name=None):
    """Constructs a graph mode TensorArray.

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
        is set, tensor_array_name should be None. Only supported in graph mode.
      flow: (optional) A float `Tensor` scalar coming from an existing
        `TensorArray.flow`. Only supported in graph mode.
      infer_shape: (optional, default: True) If True, shape inference
        is enabled.  In this case, all elements must have the same shape.
      element_shape: (optional, default: None) A `TensorShape` object specifying
        the shape constraints of each of the elements of the TensorArray.
        Need not be fully defined.
      colocate_with_first_write_call: If `True`, the TensorArray will be
        colocated on the same device as the Tensor used on its first write
        (write operations include `write`, `unstack`, and `split`).  If `False`,
        the TensorArray will be placed on the device determined by the
        device context available during its initialization.
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
    self._dynamic_size = None
    dynamic_size = dynamic_size or False

    self._dtype = dtype

    # Used to keep track of what tensors the TensorArray should be
    # colocated with.  We choose to colocate the TensorArray with the
    # first tensor written to it.
    self._colocate_with_first_write_call = colocate_with_first_write_call
    if colocate_with_first_write_call:
      self._colocate_with = []
    else:
      self._colocate_with = None

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
        def create():
          """Create the TensorArray op."""
          return gen_data_flow_ops.tensor_array_v3(
              dtype=dtype,
              size=size,
              element_shape=element_shape,
              identical_element_shapes=infer_shape,
              dynamic_size=dynamic_size,
              clear_after_read=clear_after_read,
              tensor_array_name=tensor_array_name,
              name=scope)
        if colocate_with_first_write_call:
          with ops.device(None), ops.colocate_with(None, ignore_existing=True):
            self._handle, self._flow = create()
        else:
          self._handle, self._flow = create()

  @property
  def flow(self):
    return self._flow

  @property
  def dtype(self):
    return self._dtype

  @property
  def handle(self):
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

  @contextlib.contextmanager
  def _maybe_colocate_with(self, value):
    """Colocate operations with an internal colocation group or `value`.

    Args:
      value: `Tensor`, the tensor to try to colocate with.

    Yields:
      Does not yield anything, but the new context is a colocation context.

    If no internal colocation group is set, colocate with `value` and set
    the internal colocation group to be value.
    """
    if not self._colocate_with_first_write_call:
      yield
    else:
      if not self._colocate_with:
        self._colocate_with.append(value)
      with ops.colocate_with(self._colocate_with[0]):
        yield

  def identity(self):
    """See TensorArray."""
    flow = array_ops.identity(self._flow)
    ta = TensorArray(
        dtype=self._dtype,
        handle=self._handle,
        flow=flow,
        infer_shape=self._infer_shape,
        colocate_with_first_write_call=self._colocate_with_first_write_call)
    ta._element_shape = self._element_shape
    ta._colocate_with = self._colocate_with
    return ta

  def grad(self, source, flow=None, name=None):
    """See TensorArray."""
    # tensor_array_grad requires a flow input when forward
    # TensorArrays are dynamically sized.  This forces the creation
    # of the grad TensorArray only once the final forward array's size
    # is fixed.
    if flow is None:
      flow = self.flow
    with ops.name_scope(name, "TensorArrayGrad", [self._handle]):
      with ops.colocate_with(self._handle):
        g_handle, unused_flow = gen_data_flow_ops.tensor_array_grad_v3(
            handle=self._handle, source=source, flow_in=flow, name=name)
        with ops.control_dependencies([g_handle]):
          flow = array_ops.identity(flow, name="gradient_flow")
        g = TensorArray(
            dtype=self._dtype,
            handle=g_handle,
            flow=flow,
            infer_shape=self._infer_shape,
            colocate_with_first_write_call=False)
        g._element_shape = self._element_shape
        return g

  def read(self, index, name=None):
    """See TensorArray."""
    value = gen_data_flow_ops.tensor_array_read_v3(
        handle=self._handle,
        index=index,
        flow_in=self._flow,
        dtype=self._dtype,
        name=name)
    if self._element_shape:
      value.set_shape(self._element_shape[0].dims)
    return value

  @tf_should_use.should_use_result
  def write(self, index, value, name=None):
    """See TensorArray."""
    with ops.name_scope(name, "TensorArrayWrite", [self._handle, index, value]):
      value = ops.convert_to_tensor(value, name="value")
      if self._infer_shape:
        self._merge_element_shape(value.shape)
      with self._maybe_colocate_with(value):
        flow_out = gen_data_flow_ops.tensor_array_write_v3(
            handle=self._handle,
            index=index,
            value=value,
            flow_in=self._flow,
            name=name)
      ta = TensorArray(
          dtype=self._dtype,
          handle=self._handle,
          flow=flow_out,
          colocate_with_first_write_call=self._colocate_with_first_write_call)
      ta._infer_shape = self._infer_shape
      ta._element_shape = self._element_shape
      ta._colocate_with = self._colocate_with
      return ta

  def stack(self, name=None):
    """See TensorArray."""
    with ops.colocate_with(self._handle):
      with ops.name_scope(name, "TensorArrayStack", [self._handle]):
        return self.gather(math_ops.range(0, self.size()), name=name)

  def gather(self, indices, name=None):
    """See TensorArray."""
    if self._element_shape:
      element_shape = self._element_shape[0]
    else:
      element_shape = tensor_shape.TensorShape(None)
    value = gen_data_flow_ops.tensor_array_gather_v3(
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
    """See TensorArray."""
    if self._element_shape and self._element_shape[0].dims is not None:
      element_shape_except0 = (
          tensor_shape.TensorShape(self._element_shape[0].dims[1:]))
    else:
      element_shape_except0 = tensor_shape.TensorShape(None)
    value, _ = gen_data_flow_ops.tensor_array_concat_v3(
        handle=self._handle,
        flow_in=self._flow,
        dtype=self._dtype,
        name=name,
        element_shape_except0=element_shape_except0)
    if self._element_shape and self._element_shape[0].dims is not None:
      value.set_shape([None] + self._element_shape[0].dims[1:])
    return value

  @tf_should_use.should_use_result
  def unstack(self, value, name=None):
    """See TensorArray."""
    with ops.name_scope(name, "TensorArrayUnstack", [self._handle, value]):
      num_elements = array_ops.shape(value)[0]
      return self.scatter(
          indices=math_ops.range(0, num_elements), value=value, name=name)

  @tf_should_use.should_use_result
  def scatter(self, indices, value, name=None):
    """See TensorArray."""
    with ops.name_scope(name, "TensorArrayScatter",
                        [self._handle, value, indices]):
      value = ops.convert_to_tensor(value, name="value")
      if self._infer_shape and not context.executing_eagerly():
        self._merge_element_shape(value.shape[1:])
      with self._maybe_colocate_with(value):
        flow_out = gen_data_flow_ops.tensor_array_scatter_v3(
            handle=self._handle,
            indices=indices,
            value=value,
            flow_in=self._flow,
            name=name)
      ta = TensorArray(
          dtype=self._dtype,
          handle=self._handle,
          flow=flow_out,
          colocate_with_first_write_call=self._colocate_with_first_write_call)
      ta._infer_shape = self._infer_shape
      ta._element_shape = self._element_shape
      ta._colocate_with = self._colocate_with
      return ta

  @tf_should_use.should_use_result
  def split(self, value, lengths, name=None):
    """See TensorArray."""
    with ops.name_scope(name, "TensorArraySplit",
                        [self._handle, value, lengths]):
      value = ops.convert_to_tensor(value, name="value")
      with self._maybe_colocate_with(value):
        lengths_64 = math_ops.to_int64(lengths)
        if self._infer_shape and not context.executing_eagerly():
          clengths = tensor_util.constant_value(lengths_64)
          if value.shape.dims is not None:
            if clengths is not None and clengths.max() == clengths.min():
              self._merge_element_shape(
                  tensor_shape.TensorShape([clengths[0]]).concatenate(
                      value.shape[1:]))
        flow_out = gen_data_flow_ops.tensor_array_split_v3(
            handle=self._handle,
            value=value,
            lengths=lengths_64,
            flow_in=self._flow,
            name=name)
      ta = TensorArray(
          dtype=self._dtype,
          handle=self._handle,
          flow=flow_out,
          colocate_with_first_write_call=self._colocate_with_first_write_call)
      ta._infer_shape = self._infer_shape
      ta._element_shape = self._element_shape
      ta._colocate_with = self._colocate_with
      return ta

  def size(self, name=None):
    """See TensorArray."""
    return gen_data_flow_ops.tensor_array_size_v3(
        handle=self._handle, flow_in=self.flow, name=name)

  @tf_should_use.should_use_result
  def close(self, name=None):
    """See TensorArray."""
    return gen_data_flow_ops.tensor_array_close_v3(
        handle=self._handle, name=name)


class _GraphTensorArrayV2(object):
  """Graph-mode implementation of TensorArray backed by TensorLists.

  The backing tensor of this TensorArray is a TensorList variant tensor which is
  stored in the `flow`. The `handle` is always none here. The reason we use the
  `flow` field and not the `handle` field is to ensure backwards compatibility
  with legacy control flow.
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
               colocate_with_first_write_call=True,
               name=None):
    """Constructs a graph mode TensorArray.

    Args:
      dtype: (required) data type of the TensorArray.
      size: (optional) int32 scalar `Tensor`: the size of the TensorArray.
        Required if flow is not provided.
      dynamic_size: (optional) Python bool: If true, writes to the TensorArray
        can grow the TensorArray past its initial size.  Default: False.
      clear_after_read: (optional) unused. Not supported in TensorLists.
      tensor_array_name: (optional) unused.
      handle: (optional) Must always be None.
      flow: (optional) A variant `Tensor` scalar for a TensorList.
      infer_shape: (optional, default: True) If True, shape inference is
        enabled.  In this case, all elements must have the same shape.
      element_shape: (optional, default: None) A `TensorShape` object specifying
        the shape constraints of each of the elements of the TensorArray. Need
        not be fully defined.
      colocate_with_first_write_call: (optional). unused.
      name: (optional) A name for the operation.

    Raises:
      ValueError: if both handle and tensor_array_name are provided.
      TypeError: if handle is provided but is not a Tensor.
    """
    assert handle is None
    del handle
    del clear_after_read
    del tensor_array_name
    del colocate_with_first_write_call

    self._dynamic_size = dynamic_size

    if (flow is not None and
        (not isinstance(flow, ops.Tensor) or flow.dtype != dtypes.variant)):
      raise TypeError("flow must be a variant tensor")
    if flow is None and size is None:
      raise ValueError("Size must be provided if flow is not provided")
    if flow is not None and size is not None:
      raise ValueError("Cannot provide both a flow and size "
                       "at the same time")
    if flow is not None and element_shape is not None:
      raise ValueError("Cannot provide both a flow and element_shape "
                       "at the same time")

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
    with ops.name_scope(name, "TensorArrayV2", [size, flow]) as scope:
      if flow is None:
        self._flow = list_ops.tensor_list_reserve(
            element_shape=element_shape,
            num_elements=size,
            element_dtype=dtype,
            name=scope)
      else:
        self._flow = flow

    # For backwards compatibility.
    self._colocate_with_first_write_call = None
    self._colocate_with = None

  @property
  def flow(self):
    return self._flow

  @property
  def dtype(self):
    return self._dtype

  @property
  def handle(self):
    # We intentionally do not raise an error so that legacy while_loop does not
    # complain.
    return None

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
    """See TensorArray."""
    flow = array_ops.identity(self._flow)
    return build_ta_with_new_flow(self, flow)

  def grad(self, source, flow=None, name=None):
    """Not supported."""
    raise NotImplementedError()

  def read(self, index, name=None):
    """See TensorArray."""
    value = list_ops.tensor_list_get_item(
        input_handle=self._flow,
        index=index,
        element_dtype=self._dtype,
        name=name)
    if self._element_shape:
      value.set_shape(self._element_shape[0].dims)
    return value

  @tf_should_use.should_use_result
  def write(self, index, value, name=None):
    """See TensorArray."""
    with ops.name_scope(name, "TensorArrayV2Write", [self._flow, index, value]):
      value = ops.convert_to_tensor(value, name="value")
      if self._infer_shape:
        self._merge_element_shape(value.shape)
      flow_out = list_ops.tensor_list_set_item(
          input_handle=self._flow,
          index=index,
          item=value,
          resize_if_index_out_of_bounds=self._dynamic_size,
          name=name)
      return build_ta_with_new_flow(self, flow_out)

  def stack(self, name=None):
    """See TensorArray."""
    with ops.name_scope(name, "TensorArrayV2Stack", [self._flow]):
      value = list_ops.tensor_list_stack(
          input_handle=self._flow, element_dtype=self._dtype)
      if self._element_shape and self._element_shape[0].dims is not None:
        value.set_shape([None] + self._element_shape[0].dims)
      return value

  def gather(self, indices, name=None):
    """See TensorArray."""
    value = list_ops.tensor_list_gather(
        input_handle=self._flow,
        indices=indices,
        element_dtype=self._dtype,
        name=name)
    if self._element_shape and self._element_shape[0].dims is not None:
      value.set_shape([None] + self._element_shape[0].dims)
    return value

  def concat(self, name=None):
    """See TensorArray."""
    if self._element_shape and self._element_shape[0].dims is not None:
      element_shape = [None] + self._element_shape[0].dims[1:]
    else:
      element_shape = None

    value = list_ops.tensor_list_concat(
        input_handle=self._flow,
        element_dtype=self._dtype,
        element_shape=element_shape,
        name=name)
    return value

  @tf_should_use.should_use_result
  def unstack(self, value, name=None):
    """See TensorArray."""
    with ops.name_scope(name, "TensorArrayUnstack", [self._flow, value]):
      value = ops.convert_to_tensor(value, name="value")
      if self._infer_shape and not context.executing_eagerly():
        self._merge_element_shape(value.shape[1:])
      flow_out = list_ops.tensor_list_from_tensor(
          tensor=value, element_shape=value.shape[1:])
      return build_ta_with_new_flow(self, flow_out)

  @tf_should_use.should_use_result
  def scatter(self, indices, value, name=None):
    """See TensorArray."""
    with ops.name_scope(name, "TensorArrayScatter",
                        [self._flow, value, indices]):
      value = ops.convert_to_tensor(value, name="value")
      if self._infer_shape and not context.executing_eagerly():
        self._merge_element_shape(value.shape[1:])
      flow_out = list_ops.tensor_list_scatter(
          tensor=value, indices=indices, element_shape=-1)
      return build_ta_with_new_flow(self, flow_out)

  @tf_should_use.should_use_result
  def split(self, value, lengths, name=None):
    """See TensorArray."""
    with ops.name_scope(name, "TensorArraySplit", [self._flow, value, lengths]):
      value = ops.convert_to_tensor(value, name="value")
      lengths_64 = math_ops.to_int64(lengths)
      if self._infer_shape and not context.executing_eagerly():
        clengths = tensor_util.constant_value(lengths_64)
        if value.shape.dims is not None:
          if clengths is not None and clengths.max() == clengths.min():
            self._merge_element_shape(
                tensor_shape.TensorShape([clengths[0]]).concatenate(
                    value.shape[1:]))
      flow_out = list_ops.tensor_list_split(
          tensor=value,
          lengths=lengths_64,
          element_shape=self._element_shape[0] if self._element_shape else None,
          name=name)
      return build_ta_with_new_flow(self, flow_out)

  def size(self, name=None):
    """See TensorArray."""
    return list_ops.tensor_list_length(input_handle=self._flow, name=name)

  @tf_should_use.should_use_result
  def close(self, name=None):
    """See TensorArray."""
    return gen_control_flow_ops.no_op(name=name)

# pylint: enable=protected-access


class _EagerTensorArray(object):
  """Eager-compatible implementation of TensorArray.
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
               colocate_with_first_write_call=True,
               name=None):
    """Constructs a TensorArray compatible with eager execution.

    Args:
      dtype: (required) data type of the TensorArray.
      size: (optional) int32 scalar `Tensor`: the size of the TensorArray.
        Required if handle is not provided.
      dynamic_size: (optional) Python bool: If true, writes to the TensorArray
        can grow the TensorArray past its initial size.  Default: False.
      clear_after_read: Boolean (optional, default: True).  If True, clear
        TensorArray values after reading them.  This disables read-many
        semantics, but allows early release of memory.
      tensor_array_name: unused.
      handle: unsupported.
      flow: unsupported.
      infer_shape: used for error checking, same semantics as TensorArray.
      element_shape: used for error checking, same semantics as TensorArray.
      colocate_with_first_write_call: unsupported.
      name: unsupported.

    Raises:
      ValueError: handle or flow are supplied, or if size is not supplied.
    """

    del (flow, tensor_array_name, name)  # Unused.

    if handle is not None:
      raise ValueError("TensorArray handles are not supported when eager "
                       "execution is enabled.")
    if size is None:
      raise ValueError("Size must be declared for TensorArrays when eager "
                       "execution is enabled.")

    # These attributes are not meaningful when eager is enabled, but some
    # library functions (e.g., those in control_flow_ops.py) access them to
    # create new tensor arrays; as such, we define them for the sake of
    # compatibility.
    self._handle = None
    # we assign a dummy value to _flow in case other code assumes it to be
    # a Tensor
    self._flow = constant_op.constant(0, dtype=dtypes.int32)
    self._infer_shape = infer_shape
    self._element_shape = element_shape
    self._colocate_with_first_write_call = colocate_with_first_write_call

    self._dtype = dtype
    self._dynamic_size = dynamic_size or False
    self._clear_after_read = (
        True if clear_after_read is None else clear_after_read)
    self._previously_read_indices = []

    if isinstance(size, ops.EagerTensor):
      size = size.numpy()
    self._tensor_array = [None for _ in range(size)]

  @property
  def flow(self):
    """For compatibility; flows are not meaningful when eager is enabled."""
    return self._flow

  @property
  def dtype(self):
    return self._dtype

  @property
  def handle(self):
    """For compatibility; handles are not meaningful when eager is enabled."""
    return self._handle

  def identity(self):
    """See TensorArray."""
    return self.parent()

  def grad(self, source, flow=None, name=None):
    raise NotImplementedError(
        "TensorArray.grad is not supported when executing eagerly; eager's "
        "gradient implementation does not use/need this function to compute "
        "gradients of operations that use TensorArrays.")

  def read(self, index, name=None):
    """See TensorArray."""
    del name  # not meaningful when executing eagerly.

    if isinstance(index, ops.EagerTensor):
      index = index.numpy()

    if index < 0:
      raise errors_impl.OutOfRangeError(
          None, None,
          "Reading from negative indices (index %d) is not allowed." % index)

    if index >= len(self._tensor_array):
      raise errors_impl.OutOfRangeError(
          None, None, "Tried to read from index %d but array size is: %d" %
          (index, len(self._tensor_array)))

    tensor = self._tensor_array[index]
    if tensor is None:
      if index in self._previously_read_indices:
        raise errors_impl.InvalidArgumentError(
            None, None,
            "Could not read index %d twice because it was cleared after "
            "a previous read (perhaps try setting clear_after_read = false?)" %
            index)
      else:
        tensor = self._maybe_zero(index)

    if self._clear_after_read:
      self._tensor_array[index] = None
      self._previously_read_indices.append(index)
    return tensor

  def _write(self, index, value):
    """Writes `value` into index named by `index`.

    Args:
      index: 0-D.  int32 scalar with the index to write to.
      value: N-D.  Tensor of type `dtype`.  The `Tensor` to write to `index`.

    Raises:
      errors_impl.InvalidArgumentError: `value` dtype does not match dtype.
      errors_impl.OutOfRangeError: `index` is out of bounds.
      ValueError: shape of `value` is not consistent with inferred shape.
    """

    if isinstance(index, ops.EagerTensor):
      index = index.numpy()

    if index < 0:
      raise errors_impl.OutOfRangeError(
          None, None,
          "Writing to negative indices (index %d) is not allowed." % index)

    size = len(self._tensor_array)
    if index >= size:
      if not self._dynamic_size:
        raise errors_impl.OutOfRangeError(
            None, None,
            "Tried to write to index %d but array is not resizeable and size "
            "is: %d" % (index, size))
      self._tensor_array.extend([None for _ in range(index - size + 1)])

    if not isinstance(value, ops.EagerTensor):
      value = ops.convert_to_tensor(value)

    if self._infer_shape:
      if self._element_shape is None:
        self._element_shape = value.shape
      elif self._element_shape != value.shape:
        raise ValueError("Incompatible shape for value (%s), expected (%s)" %
                         (value.shape.as_list(), self._element_shape.as_list()))

    if self._dtype != value.dtype:
      raise errors_impl.InvalidArgumentError(
          None, None,
          "TensorArray dtype is %s but Op is trying to write dtype %s" %
          (self._dtype.name, value.dtype.name))
    self._tensor_array[index] = value

  def write(self, index, value, name=None):
    """See TensorArray."""
    del name  # not meaningful when executing eagerly.
    self._write(index, value)
    return self.parent()

  def _maybe_zero(self, ix):
    val = self._tensor_array[ix]
    if val is None:
      val = self._tensor_array[ix] = array_ops.zeros(
          shape=self._element_shape, dtype=self._dtype)
    return val

  def stack(self, name=None):
    """See TensorArray."""
    if self._tensor_array:
      for ix in range(len(self._tensor_array)):
        self._maybe_zero(ix)
    return ops.convert_to_tensor(
        self._tensor_array, name=name, dtype=self._dtype)

  def gather(self, indices, name=None):
    """See TensorArray."""
    del name  # not meaningful when executing eagerly.
    return array_ops.stack([self._maybe_zero(i) for i in indices.numpy()])

  def concat(self, name=None):
    """See TensorArray."""
    try:
      return array_ops.concat(
          [self._maybe_zero(ix) for ix in range(len(self._tensor_array))],
          0, name=name)
    except errors_impl.OpError:
      # Reproduce a subset of the error-handling for graph-mode TensorArrays.
      shapes = [t.shape for t in self._tensor_array]
      ndims = [s.ndims for s in shapes]
      if 0 in ndims:
        idx = ndims.index(0)
        raise errors_impl.InvalidArgumentError(
            None, None, "Concat saw a scalar shape at index %d but requires "
            "at least vectors." % idx)
      else:
        raise

  def unstack(self, value, name=None):
    """See TensorArray."""
    tensors = array_ops.unstack(value, name=name)
    if len(tensors) > len(self._tensor_array) and not self._dynamic_size:
      raise ValueError(
          "Cannot unstack %d tensors into a TensorArray of static size %d" %
          (len(tensors), len(self._tensor_array)))
    self._tensor_array = tensors
    return self.parent()

  def scatter(self, indices, value, name=None):
    """See TensorArray."""
    del name  # not meaningful when executing eagerly.
    for index, val in zip(indices.numpy(), array_ops.unstack(value)):
      self._write(index, val)  # pylint: disable=protected-access
    return self.parent()

  def split(self, value, lengths, name=None):
    """See TensorArray."""
    # error checking to match graph-mode errors
    value = ops.convert_to_tensor(value)
    lengths = ops.convert_to_tensor(lengths)
    sum_lengths = math_ops.reduce_sum(lengths)
    if lengths.shape.ndims != 1:
      raise errors_impl.InvalidArgumentError(
          None, None, "Expected lengths to be a vector, received shape: %s" %
          lengths.shape.as_list())
    elif value.shape.ndims == 0:
      raise errors_impl.InvalidArgumentError(
          None, None, "Expected value to be at least a vector, "
          "but received shape: %s" % value.shape.as_list())
    elif sum_lengths.numpy() != value.shape.as_list()[0]:
      raise errors_impl.InvalidArgumentError(
          None, None, "Expected sum of lengths to be equal to "
          "values.shape[0], but sum of lengths is %d and "
          "value's shape is: %s " % (sum_lengths.numpy(),
                                     value.shape.as_list()))
    elif not self._dynamic_size and lengths.shape[0] != len(self._tensor_array):
      raise errors_impl.InvalidArgumentError(
          None, None, "TensorArray's size is not equal to the size of "
          "lengths (%d vs. %d), and the TensorArray is not marked as "
          "dynamically resizeable" % (len(self._tensor_array),
                                      lengths.shape[0]))
    else:
      self._tensor_array = array_ops.split(value, lengths, name=name)
      return self.parent()

  def size(self, name=None):
    """See TensorArray."""
    del name  # not meaningful when executing eagerly.
    return constant_op.constant(len(self._tensor_array))

  def close(self, name=None):
    del name  # not meaningful when executing eagerly.
    del self._tensor_array[:]


# TensorArray is designed to hide an underlying implementation object
# and as such accesses many of that object's hidden fields.
# pylint: disable=protected-access
@tf_export("TensorArray")
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
               colocate_with_first_write_call=True,
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
        is set, tensor_array_name should be None. Only supported in graph mode.
      flow: (optional) A float `Tensor` scalar coming from an existing
        `TensorArray.flow`. Only supported in graph mode.
      infer_shape: (optional, default: True) If True, shape inference
        is enabled.  In this case, all elements must have the same shape.
      element_shape: (optional, default: None) A `TensorShape` object specifying
        the shape constraints of each of the elements of the TensorArray.
        Need not be fully defined.
      colocate_with_first_write_call: If `True`, the TensorArray will be
        colocated on the same device as the Tensor used on its first write
        (write operations include `write`, `unstack`, and `split`).  If `False`,
        the TensorArray will be placed on the device determined by the
        device context available during its initialization.
      name: A name for the operation (optional).

    Raises:
      ValueError: if both handle and tensor_array_name are provided.
      TypeError: if handle is provided but is not a Tensor.
    """
    if context.executing_eagerly():
      implementation = _EagerTensorArray
    else:
      if control_flow_util.EnableControlFlowV2(ops.get_default_graph()):
        implementation = _GraphTensorArrayV2
      else:
        implementation = _GraphTensorArray
    self._implementation = implementation(
        dtype,
        size=size,
        dynamic_size=dynamic_size,
        clear_after_read=clear_after_read,
        tensor_array_name=tensor_array_name,
        handle=handle,
        flow=flow,
        infer_shape=infer_shape,
        element_shape=element_shape,
        colocate_with_first_write_call=colocate_with_first_write_call,
        name=name)

    self._implementation.parent = weakref.ref(self)

  @property
  def flow(self):
    """The flow `Tensor` forcing ops leading to this TensorArray state."""
    return self._implementation._flow

  @property
  def dtype(self):
    """The data type of this TensorArray."""
    return self._implementation._dtype

  @property
  def handle(self):
    """The reference to the TensorArray."""
    return self._implementation.handle

  @property
  def _dynamic_size(self):
    return self._implementation._dynamic_size

  @property
  def _infer_shape(self):
    return self._implementation._infer_shape

  @_infer_shape.setter
  def _infer_shape(self, infer_shape):
    self._implementation._infer_shape = infer_shape

  @property
  def _element_shape(self):
    return self._implementation._element_shape

  @_element_shape.setter
  def _element_shape(self, element_shape):
    self._implementation._element_shape = element_shape

  @property
  def _colocate_with_first_write_call(self):
    return self._implementation._colocate_with_first_write_call

  @property
  def _colocate_with(self):
    return self._implementation._colocate_with

  @_colocate_with.setter
  def _colocate_with(self, colocate_with):
    self._implementation._colocate_with = colocate_with

  def identity(self):
    """Returns a TensorArray with the same content and properties.

    Returns:
      A new TensorArray object with flow that ensures the control dependencies
      from the contexts will become control dependencies for writes, reads, etc.
      Use this object all for subsequent operations.
    """
    return self._implementation.identity()

  def grad(self, source, flow=None, name=None):
    return self._implementation.grad(source, flow=flow, name=name)

  def read(self, index, name=None):
    """Read the value at location `index` in the TensorArray.

    Args:
      index: 0-D.  int32 tensor with the index to read from.
      name: A name for the operation (optional).

    Returns:
      The tensor at index `index`.
    """
    return self._implementation.read(index, name=name)

  @tf_should_use.should_use_result
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
    return self._implementation.write(index, value, name=name)

  def stack(self, name=None):
    """Return the values in the TensorArray as a stacked `Tensor`.

    All of the values must have been written and their shapes must all match.
    If input shapes have rank-`R`, then output shape will have rank-`(R+1)`.

    Args:
      name: A name for the operation (optional).

    Returns:
      All the tensors in the TensorArray stacked into one tensor.
    """
    return self._implementation.stack(name=name)

  def gather(self, indices, name=None):
    """Return selected values in the TensorArray as a packed `Tensor`.

    All of selected values must have been written and their shapes
    must all match.

    Args:
      indices: A `1-D` `Tensor` taking values in `[0, max_value)`.  If
        the `TensorArray` is not dynamic, `max_value=size()`.
      name: A name for the operation (optional).

    Returns:
      The tensors in the `TensorArray` selected by `indices`, packed into one
      tensor.
    """
    return self._implementation.gather(indices, name=name)

  def concat(self, name=None):
    """Return the values in the TensorArray as a concatenated `Tensor`.

    All of the values must have been written, their ranks must match, and
    and their shapes must all match for all dimensions except the first.

    Args:
      name: A name for the operation (optional).

    Returns:
      All the tensors in the TensorArray concatenated into one tensor.
    """
    return self._implementation.concat(name=name)

  @tf_should_use.should_use_result
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
    return self._implementation.unstack(value, name=name)

  @tf_should_use.should_use_result
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
    return self._implementation.scatter(indices, value, name=name)

  @tf_should_use.should_use_result
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
    return self._implementation.split(value, lengths, name=name)

  def size(self, name=None):
    """Return the size of the TensorArray."""
    return self._implementation.size(name=name)

  @tf_should_use.should_use_result
  def close(self, name=None):
    """Close the current TensorArray."""
    return self._implementation.close(name=name)


def build_ta_with_new_flow(old_ta, flow):
  """Builds a TensorArray with a new `flow` tensor."""
  ta = TensorArray(
      dtype=old_ta.dtype,
      dynamic_size=old_ta._dynamic_size,
      handle=old_ta.handle,
      flow=flow,
      infer_shape=old_ta._infer_shape,
      colocate_with_first_write_call=old_ta._colocate_with_first_write_call)
  ta._colocate_with = old_ta._colocate_with
  ta._element_shape = old_ta._element_shape
  return ta

# pylint: enable=protected-access
