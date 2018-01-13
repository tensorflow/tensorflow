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

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_list_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_list_ops import *
# pylint: enable=wildcard-import


@ops.RegisterGradient("TensorListPushBack")
def _PushBackGradient(op, dresult):
  return gen_list_ops.tensor_list_pop_back(
      dresult, element_dtype=op.get_attr("element_dtype"))


@ops.RegisterGradient("TensorListPopBack")
def _PopBackGradient(unused_op, dlist, delement):
  if dlist is None:
    dlist = gen_list_ops.empty_tensor_list(
        element_dtype=delement.dtype,
        element_shape=-1)
  return gen_list_ops.tensor_list_push_back(dlist, delement)


@ops.RegisterGradient("TensorListStack")
def _TensorListStack(unused_op, dtensor):
  return gen_list_ops.tensor_list_from_tensor(dtensor,
                                              element_shape=dtensor.shape[1:])


@ops.RegisterGradient("TensorListFromTensor")
def _TensorListFromTensor(op, dlist):
  if op.inputs[0].shape[0] is not None:
    num_elements = op.inputs[0].shape[0]
  else:
    num_elements = None
  if dlist is None:
    dlist = gen_list_ops.empty_tensor_list(
        element_dtype=op.inputs[0].dtype,
        element_shape=-1)
  return gen_list_ops.tensor_list_stack(
      dlist, element_dtype=op.inputs[0].dtype,
      num_elements=num_elements)
