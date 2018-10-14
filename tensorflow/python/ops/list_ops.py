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
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_list_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_list_ops import *
# pylint: enable=wildcard-import


ops.NotDifferentiable("TensorListConcat")
ops.NotDifferentiable("TensorListPushBackBatch")


@ops.RegisterGradient("TensorListPushBack")
def _PushBackGrad(op, dresult):
  return gen_list_ops.tensor_list_pop_back(
      dresult, element_dtype=op.get_attr("element_dtype"))


@ops.RegisterGradient("TensorListPopBack")
def _PopBackGrad(op, dlist, delement):
  if dlist is None:
    dlist = gen_list_ops.empty_tensor_list(
        element_dtype=delement.dtype,
        element_shape=gen_list_ops.tensor_list_element_shape(
            op.outputs[0], shape_type=dtypes.int32))
  return gen_list_ops.tensor_list_push_back(dlist, delement)


@ops.RegisterGradient("TensorListStack")
def _TensorListStackGrad(unused_op, dtensor):
  return gen_list_ops.tensor_list_from_tensor(dtensor,
                                              element_shape=dtensor.shape[1:])


@ops.RegisterGradient("TensorListFromTensor")
def _TensorListFromTensorGrad(op, dlist):
  """Gradient for TensorListFromTensor."""
  if op.inputs[0].shape[0].value is not None:
    num_elements = op.inputs[0].shape[0].value
  else:
    num_elements = None
  if dlist is None:
    dlist = gen_list_ops.empty_tensor_list(
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
      dlist, indices, element_dtype=t.dtype), None
