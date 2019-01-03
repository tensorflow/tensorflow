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
"""Ops to manipulate lists of tensors."""

# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_list_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_list_ops import *
# pylint: enable=wildcard-import
from tensorflow.python.util.lazy_loader import LazyLoader

# list_ops -> control_flow_ops -> tensor_array_ops -> list_ops
control_flow_ops = LazyLoader(
    "control_flow_ops", globals(),
    "tensorflow.python.ops.control_flow_ops")


ops.NotDifferentiable("TensorListConcatLists")
ops.NotDifferentiable("TensorListElementShape")
ops.NotDifferentiable("TensorListLength")
ops.NotDifferentiable("TensorListPushBackBatch")


def empty_tensor_list(element_shape,
                      element_dtype,
                      max_num_elements=None,
                      name=None):
  if max_num_elements is None:
    max_num_elements = -1

  return gen_list_ops.empty_tensor_list(
      element_shape=_build_element_shape(element_shape),
      element_dtype=element_dtype,
      max_num_elements=max_num_elements,
      name=name)


def tensor_list_reserve(element_shape, num_elements, element_dtype, name=None):
  return gen_list_ops.tensor_list_reserve(
      element_shape=_build_element_shape(element_shape),
      num_elements=num_elements,
      element_dtype=element_dtype,
      name=name)


def tensor_list_from_tensor(tensor, element_shape, name=None):
  return gen_list_ops.tensor_list_from_tensor(
      tensor=tensor,
      element_shape=_build_element_shape(element_shape),
      name=name)


def tensor_list_concat(input_handle, element_dtype, element_shape=None,
                       name=None):
  # Ignore the lengths output of TensorListConcat. It is only used during
  # gradient computation.
  return gen_list_ops.tensor_list_concat(
      input_handle=input_handle, element_dtype=element_dtype,
      element_shape=element_shape, name=name)[0]


def tensor_list_split(tensor, element_shape, lengths, name=None):
  return gen_list_ops.tensor_list_split(
      tensor=tensor,
      element_shape=_build_element_shape(element_shape),
      lengths=lengths,
      name=name)


def tensor_list_set_item(input_handle,
                         index,
                         item,
                         resize_if_index_out_of_bounds=False,
                         name=None):
  """Sets `item` at `index` in input list."""
  if resize_if_index_out_of_bounds:
    input_list_size = gen_list_ops.tensor_list_length(input_handle)
    # TODO(srbs): This could cause some slowdown. Consider fusing resize
    # functionality in the SetItem op.
    input_handle = control_flow_ops.cond(
        index >= input_list_size,
        lambda: gen_list_ops.tensor_list_resize(  # pylint: disable=g-long-lambda
            input_handle, index + 1),
        lambda: input_handle)
  return gen_list_ops.tensor_list_set_item(
      input_handle=input_handle, index=index, item=item, name=name)


@ops.RegisterGradient("TensorListPushBack")
def _PushBackGrad(op, dresult):
  return gen_list_ops.tensor_list_pop_back(
      dresult, element_dtype=op.get_attr("element_dtype"))


@ops.RegisterGradient("TensorListPopBack")
def _PopBackGrad(op, dlist, delement):
  if dlist is None:
    dlist = empty_tensor_list(
        element_dtype=delement.dtype,
        element_shape=gen_list_ops.tensor_list_element_shape(
            op.outputs[0], shape_type=dtypes.int32))
  return gen_list_ops.tensor_list_push_back(dlist, delement)


@ops.RegisterGradient("TensorListStack")
def _TensorListStackGrad(unused_op, dtensor):
  return tensor_list_from_tensor(dtensor, element_shape=dtensor.shape[1:])


@ops.RegisterGradient("TensorListConcat")
def _TensorListConcatGrad(op, dtensor, unused_dlengths):
  # TODO(srbs): We lose the element_shape information in tensor_list_concat.
  # Consider providing that as an output of TensorListConcat?
  if dtensor.shape.rank is None:
    element_shape = None
  else:
    element_shape = [None] + dtensor.shape.as_list()[1:]
  return tensor_list_split(
      dtensor,
      element_shape=_build_element_shape(element_shape),
      lengths=op.outputs[1])


@ops.RegisterGradient("TensorListSplit")
def _TensorListSplitGrad(op, dlist):
  return tensor_list_concat(dlist, element_dtype=op.inputs[0].dtype), None, None


@ops.RegisterGradient("TensorListFromTensor")
def _TensorListFromTensorGrad(op, dlist):
  """Gradient for TensorListFromTensor."""
  if op.inputs[0].shape.dims and op.inputs[0].shape.dims[0].value is not None:
    num_elements = op.inputs[0].shape.dims[0].value
  else:
    num_elements = None
  if dlist is None:
    dlist = empty_tensor_list(
        element_dtype=op.inputs[0].dtype,
        element_shape=gen_list_ops.tensor_list_element_shape(
            op.outputs[0], shape_type=dtypes.int32))
  tensor_grad = gen_list_ops.tensor_list_stack(
      dlist, element_dtype=op.inputs[0].dtype, num_elements=num_elements)
  shape_grad = None
  return tensor_grad, shape_grad


@ops.RegisterGradient("TensorListGetItem")
def _TensorListGetItemGrad(op, ditem):
  """Gradient for TensorListGetItem."""
  list_size = gen_list_ops.tensor_list_length(op.inputs[0])
  list_grad = gen_list_ops.tensor_list_set_item(
      gen_list_ops.tensor_list_reserve(
          gen_list_ops.tensor_list_element_shape(op.inputs[0],
                                                 shape_type=dtypes.int32),
          list_size, element_dtype=ditem.dtype),
      index=op.inputs[1],
      item=ditem)
  index_grad = None
  return list_grad, index_grad


@ops.RegisterGradient("TensorListSetItem")
def _TensorListSetItemGrad(op, dlist):
  _, index, item = op.inputs
  list_grad = gen_list_ops.tensor_list_set_item(
      dlist, index=index, item=array_ops.zeros_like(item))
  index_grad = None
  element_grad = gen_list_ops.tensor_list_get_item(
      dlist, index, element_dtype=item.dtype)
  return list_grad, index_grad, element_grad


@ops.RegisterGradient("TensorListResize")
def _TensorListResizeGrad(op, dlist):
  input_list, _ = op.inputs
  input_list_size = gen_list_ops.tensor_list_length(input_list)
  return gen_list_ops.tensor_list_resize(dlist, input_list_size), None


@ops.RegisterGradient("TensorListGather")
def _TensorListGatherGrad(op, dtensor):
  _, indices = op.inputs
  return gen_list_ops.tensor_list_scatter(
      tensor=dtensor, indices=indices,
      element_shape=ops.convert_to_tensor(-1, dtype=dtypes.int32)), None


@ops.RegisterGradient("TensorListScatter")
def _TensorListScatterGrad(op, dlist):
  t, indices, _ = op.inputs
  return gen_list_ops.tensor_list_gather(
      dlist, indices, element_dtype=t.dtype), None, None


def _build_element_shape(shape):
  """Converts shape to a format understood by list_ops for element_shape.

  If `shape` is already a `Tensor` it is returned as-is. We do not perform a
  type check here.

  If shape is None or a TensorShape with unknown rank, -1 is returned.

  If shape is a scalar, an int32 tensor with empty list is returned. Note we
  do directly return an empty list since ops.convert_to_tensor would conver it
  to a float32 which is not a valid type for element_shape.

  If shape is a sequence of dims, None's in the list are replaced with -1. We
  do not check the dtype of the other dims.

  Args:
    shape: Could be None, Tensor, TensorShape or a list of dims (each dim could
      be a None, scalar or Tensor).

  Returns:
    A None-free shape that can be converted to a tensor.
  """
  if isinstance(shape, ops.Tensor):
    return shape
  if isinstance(shape, tensor_shape.TensorShape):
    # `TensorShape.as_list` requires rank to be known.
    shape = shape.as_list() if shape else None
  # Shape is unknown.
  if shape is None:
    return -1
  # Shape is a scalar.
  if not shape:
    return ops.convert_to_tensor(shape, dtype=dtypes.int32)
  # Shape is a sequence of dimensions. Convert None dims to -1.
  return [d if d is not None else -1 for d in shape]
