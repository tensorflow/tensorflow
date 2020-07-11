# Copyright 2018 The Sonnet Authors. All Rights Reserved.
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
"""Use zero_out ops in python."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_map_ops
from tensorflow.python.ops.gen_map_ops import *
from tensorflow.python.framework import constant_op


#zero_out_ops = load_library.load_op_library(
#    resource_loader.get_path_to_datafile('_zero_out_ops.so'))
#zero_out = zero_out_ops.zero_out 

ops.NotDifferentiable("EmptyTensorMap")

def empty_tensor_map():
  return gen_map_ops.empty_tensor_map()

def tensor_map_size(input_handle):
  return gen_map_ops.tensor_map_size(input_handle)

def tensor_map_insert(input_handle, key, value):
  return gen_map_ops.tensor_map_insert(input_handle, key, value)

def tensor_map_lookup(input_handle, key):
  return gen_map_ops.tensor_map_lookup(input_handle, key)

def tensor_map_erase(input_handle, key):
  return gen_map_ops.tensor_map_erase(input_handle, key)

def tensor_map_replace(input_handle, key, value):
  return gen_map_ops.tensor_map_replace(input_handle, key, value)

@ops.RegisterGradient("TensorMapLookup")
def LookupGrad(op, dval):
  # map grad should be a map that is 0 everywhere except 1 @key k
  m, k = op.inputs
  #m = gen_map_ops.tensor_map_zeros(m)
  map_grad = empty_tensor_map()
  map_grad = tensor_map_insert(map_grad, k, dval)
  key = op.inputs[1]
  key_grad = None
  return map_grad, key_grad

@ops.RegisterGradient("TensorMapInsert")
def InsertGrad(op, dmap):
  _, key, val = op.inputs
  map_grad, _ = gen_map_ops.tensor_map_erase(dmap, key)
  key_grad = None
  value_grad = tensor_map_lookup(dmap, key)
  return map_grad, key_grad, value_grad
