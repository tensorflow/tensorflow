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
import numpy as np

from tensorflow.core.framework import full_type_pb2
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_list_ops
from tensorflow.python.ops import handle_data_util
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


def _set_handle_data(list_handle, element_shape, element_dtype):
  """Sets type information on `list_handle` for consistency with graphs."""
  # TODO(b/169968286): It would be better if we had a consistent story for
  # creating handle data from eager operations (shared with VarHandleOp).
  if isinstance(list_handle, ops.EagerTensor):
    if tensor_util.is_tf_type(element_shape):
      element_shape = tensor_shape.TensorShape(None)
    elif not isinstance(element_shape, tensor_shape.TensorShape):
      element_shape = tensor_shape.TensorShape(element_shape)
    handle_data = cpp_shape_inference_pb2.CppShapeInferenceResult.HandleData()
    handle_data.is_set = True
    # TODO(b/191472076): This duplicates type inference. Clean up.
    handle_data.shape_and_type.append(
        cpp_shape_inference_pb2.CppShapeInferenceResult.HandleShapeAndType(
            shape=element_shape.as_proto(),
            dtype=element_dtype.as_datatype_enum,
            type=full_type_pb2.FullTypeDef(type_id=full_type_pb2.TFT_ARRAY)))
    list_handle._handle_data = handle_data  # pylint: disable=protected-access


def tensor_list_reserve(element_shape, num_elements, element_dtype, name=None):
  result = gen_list_ops.tensor_list_reserve(
      element_shape=_build_element_shape(element_shape),
      num_elements=num_elements,
      element_dtype=element_dtype,
      name=name)
  # TODO(b/169968286): gen_ops needs to ensure the metadata is properly
  # populated for eager operations.
  _set_handle_data(result, element_shape, element_dtype)
  return result


def tensor_list_from_tensor(tensor, element_shape, name=None):
  tensor = ops.convert_to_tensor(tensor)
  result = gen_list_ops.tensor_list_from_tensor(
      tensor=tensor,
      element_shape=_build_element_shape(element_shape),
      name=name)
  _set_handle_data(result, tensor.shape, tensor.dtype)
  return result


def tensor_list_get_item(input_handle, index, element_dtype, element_shape=None,
                         name=None):
  return gen_list_ops.tensor_list_get_item(
      input_handle=input_handle,
      index=index,
      element_shape=_build_element_shape(element_shape),
      element_dtype=element_dtype,
      name=name)


def tensor_list_pop_back(input_handle, element_dtype, name=None):
  return gen_list_ops.tensor_list_pop_back(
      input_handle=input_handle,
      element_shape=-1,
      element_dtype=element_dtype,
      name=name)


def tensor_list_gather(input_handle,
                       indices,
                       element_dtype,
                       element_shape=None,
                       name=None):
  return gen_list_ops.tensor_list_gather(
      input_handle=input_handle,
      indices=indices,
      element_shape=_build_element_shape(element_shape),
      element_dtype=element_dtype,
      name=name)


def tensor_list_scatter(tensor,
                        indices,
                        element_shape=None,
                        input_handle=None,
                        name=None):
  """Returns a TensorList created or updated by scattering `tensor`."""
  tensor = ops.convert_to_tensor(tensor)
  if input_handle is not None:
    output_handle = gen_list_ops.tensor_list_scatter_into_existing_list(
        input_handle=input_handle, tensor=tensor, indices=indices, name=name)
    handle_data_util.copy_handle_data(input_handle, output_handle)
    return output_handle
  else:
    output_handle = gen_list_ops.tensor_list_scatter_v2(
        tensor=tensor,
        indices=indices,
        element_shape=_build_element_shape(element_shape),
        num_elements=-1,
        name=name)
    _set_handle_data(output_handle, element_shape, tensor.dtype)
    return output_handle


def tensor_list_stack(input_handle,
                      element_dtype,
                      num_elements=-1,
                      element_shape=None,
                      name=None):
  return gen_list_ops.tensor_list_stack(
      input_handle=input_handle,
      element_shape=_build_element_shape(element_shape),
      element_dtype=element_dtype,
      num_elements=num_elements,
      name=name)


def tensor_list_concat(input_handle, element_dtype, element_shape=None,
                       name=None):
  # Ignore the lengths output of TensorListConcat. It is only used during
  # gradient computation.
  return gen_list_ops.tensor_list_concat_v2(
      input_handle=input_handle,
      element_dtype=element_dtype,
      element_shape=_build_element_shape(element_shape),
      leading_dims=ops.convert_to_tensor([], dtype=dtypes.int64),
      name=name)[0]


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
  output_handle = gen_list_ops.tensor_list_set_item(
      input_handle=input_handle, index=index, item=item, name=name)
  handle_data_util.copy_handle_data(input_handle, output_handle)
  return output_handle


@ops.RegisterGradient("TensorListPushBack")
def _PushBackGrad(op, dresult):
  return gen_list_ops.tensor_list_pop_back(
      dresult,
      element_shape=array_ops.shape(op.inputs[1]),
      element_dtype=op.get_attr("element_dtype"))


@ops.RegisterGradient("TensorListPopBack")
def _PopBackGrad(op, dlist, delement):
  if dlist is None:
    dlist = empty_tensor_list(
        element_dtype=delement.dtype,
        element_shape=gen_list_ops.tensor_list_element_shape(
            op.outputs[0], shape_type=dtypes.int32))
  if delement is None:
    delement = array_ops.zeros_like(op.outputs[1])
  return gen_list_ops.tensor_list_push_back(dlist, delement), None


@ops.RegisterGradient("TensorListStack")
def _TensorListStackGrad(unused_op, dtensor):
  return tensor_list_from_tensor(dtensor, element_shape=dtensor.shape[1:]), None


@ops.RegisterGradient("TensorListConcat")
@ops.RegisterGradient("TensorListConcatV2")
def _TensorListConcatGrad(op, dtensor, unused_dlengths):
  """Gradient function for TensorListConcat."""
  dlist = tensor_list_split(
      dtensor,
      element_shape=gen_list_ops.tensor_list_element_shape(
          op.inputs[0], shape_type=dtypes.int32),
      lengths=op.outputs[1])
  if op.type == "TensorListConcatV2":
    return dlist, None, None
  else:
    return dlist


@ops.RegisterGradient("TensorListSplit")
def _TensorListSplitGrad(op, dlist):
  tensor, _, lengths = op.inputs
  element_shape = array_ops.slice(array_ops.shape(tensor), [1], [-1])
  element_shape = array_ops.concat([[-1], element_shape], axis=0)
  return gen_list_ops.tensor_list_concat_v2(
      dlist,
      element_shape=element_shape,
      leading_dims=lengths,
      element_dtype=op.inputs[0].dtype)[0], None, None


@ops.RegisterGradient("TensorListFromTensor")
def _TensorListFromTensorGrad(op, dlist):
  """Gradient for TensorListFromTensor."""
  t = op.inputs[0]
  if t.shape.dims and t.shape.dims[0].value is not None:
    num_elements = t.shape.dims[0].value
  else:
    num_elements = None
  if dlist is None:
    dlist = empty_tensor_list(
        element_dtype=t.dtype,
        element_shape=gen_list_ops.tensor_list_element_shape(
            op.outputs[0], shape_type=dtypes.int32))
  tensor_grad = gen_list_ops.tensor_list_stack(
      dlist,
      element_shape=array_ops.slice(array_ops.shape(t), [1], [-1]),
      element_dtype=t.dtype,
      num_elements=num_elements)
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
  element_shape_grad = None
  return list_grad, index_grad, element_shape_grad


@ops.RegisterGradient("TensorListSetItem")
def _TensorListSetItemGrad(op, dlist):
  """Gradient function for TensorListSetItem."""
  _, index, item = op.inputs
  list_grad = gen_list_ops.tensor_list_set_item(
      dlist, index=index, item=array_ops.zeros_like(item))
  index_grad = None
  element_grad = tensor_list_get_item(
      dlist,
      index,
      element_shape=array_ops.shape(item),
      element_dtype=item.dtype)
  return list_grad, index_grad, element_grad


@ops.RegisterGradient("TensorListResize")
def _TensorListResizeGrad(op, dlist):
  input_list, _ = op.inputs
  input_list_size = gen_list_ops.tensor_list_length(input_list)
  return gen_list_ops.tensor_list_resize(dlist, input_list_size), None


@ops.RegisterGradient("TensorListGather")
def _TensorListGatherGrad(op, dtensor):
  """Gradient function for TensorListGather."""
  input_list, indices, _ = op.inputs
  element_shape = gen_list_ops.tensor_list_element_shape(
      input_list, shape_type=dtypes.int32)
  num_elements = gen_list_ops.tensor_list_length(input_list)
  dlist = tensor_list_reserve(element_shape, num_elements, dtensor.dtype)
  dlist = tensor_list_scatter(
      tensor=dtensor, indices=indices, input_handle=dlist)
  return dlist, None, None


@ops.RegisterGradient("TensorListScatter")
@ops.RegisterGradient("TensorListScatterV2")
def _TensorListScatterGrad(op, dlist):
  """Gradient function for TensorListScatter."""
  tensor = op.inputs[0]
  indices = op.inputs[1]
  dtensor = gen_list_ops.tensor_list_gather(
      dlist,
      indices,
      element_shape=array_ops.slice(array_ops.shape(tensor), [1], [-1]),
      element_dtype=tensor.dtype)
  if op.type == "TensorListScatterV2":
    return dtensor, None, None, None
  else:
    return dtensor, None, None


@ops.RegisterGradient("TensorListScatterIntoExistingList")
def _TensorListScatterIntoExistingListGrad(op, dlist):
  """Gradient function for TensorListScatterIntoExistingList."""
  _, tensor, indices = op.inputs
  dtensor = gen_list_ops.tensor_list_gather(
      dlist,
      indices,
      element_shape=array_ops.slice(array_ops.shape(tensor), [1], [-1]),
      element_dtype=tensor.dtype)
  zeros = array_ops.zeros_like(tensor)
  dlist = tensor_list_scatter(zeros, indices, indices, input_handle=dlist)
  return dlist, dtensor, None


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
  # Shape is numpy array or a scalar.
  if isinstance(shape, (np.ndarray, np.generic)) or not shape:
    return ops.convert_to_tensor(shape, dtype=dtypes.int32)
  # Shape is a sequence of dimensions. Convert None dims to -1.
  def convert(val):
    if val is None:
      return -1
    if isinstance(val, ops.Tensor):
      return val
    if isinstance(val, tensor_shape.Dimension):
      return val.value if val.value is not None else -1
    return val

  return [convert(d) for d in shape]
