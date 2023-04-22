# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Ops to manipulate hashmap of tensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_map_ops
from tensorflow.python.ops.gen_map_ops import *

ops.NotDifferentiable("EmptyTensorMap")

def empty_tensor_map():
  return gen_map_ops.empty_tensor_map()

def tensor_map_size(input_handle):
  return gen_map_ops.tensor_map_size(input_handle)

def tensor_map_insert(input_handle, key, value):
  return gen_map_ops.tensor_map_insert(input_handle, key, value)

def tensor_map_lookup(input_handle, key, value_dtype):
  return gen_map_ops.tensor_map_lookup(input_handle, key, value_dtype)

def tensor_map_erase(input_handle, key, value_dtype):
  return gen_map_ops.tensor_map_erase(input_handle, key, value_dtype)

def tensor_map_has_key(input_handle, key):
  return gen_map_ops.tensor_map_has_key(input_handle, key)


def tensor_map_stack_keys(input_handle, key_dtype):
  return gen_map_ops.tensor_map_stack_keys(input_handle, key_dtype)


@ops.RegisterGradient("TensorMapLookup")
def LookupGrad(op, dval):
  _, k = op.inputs
  map_grad = empty_tensor_map()
  map_grad = tensor_map_insert(map_grad, k, dval)
  key_grad = None
  return map_grad, key_grad

@ops.RegisterGradient("TensorMapInsert")
def InsertGrad(op, dmap):
  _, k, v = op.inputs
  key_grad = None
  (value_grad, map_grad) = control_flow_ops.cond(
      tensor_map_has_key(dmap, k), lambda:
      (tensor_map_lookup(dmap, k, v.dtype), tensor_map_erase(dmap, k, v.dtype)),
      lambda: (array_ops.zeros_like(v), dmap))
  return map_grad, key_grad, value_grad

@ops.RegisterGradient("TensorMapErase")
def EraseGrad(op, dmap):
  key_grad = None
  map_grad = dmap
  return map_grad, key_grad
