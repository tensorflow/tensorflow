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
"""Model script to test TF-TensorRT integration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

from tensorflow.contrib import tensorrt as trt
from tensorflow.core.protobuf import config_pb2 as cpb2
from tensorflow.core.protobuf import rewriter_config_pb2 as rwpb2
from tensorflow.python.client import session as csess
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import importer as importer
from tensorflow.python.framework import ops as ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.layers import core
from tensorflow.python.training import training
from base_unit_test import BaseUnitTest
from utilities import get_all_variables

class BiasaddMatMulTest(BaseUnitTest):
  """Testing BiasAdd MatMul in TF-TRT conversion"""

  def __init__(self, log_file='log.txt'):
    super(BiasaddMatMulTest, self).__init__()
    self.static_mode_list = {"FP32", "FP16"}
    self.debug=True
    self.dynamic_mode_list = {}
    self.inp_dims = (48, 12)
    self.dummy_input = np.random.random_sample(self.inp_dims)
    self.get_network = self.matmul_test
    self.expect_nb_nodes = 53
    self.log_file = log_file
    self.test_name = self.__class__.__name__ 

  def matmul_test(self):
    g = ops.Graph()
    gpu_options = cpb2.GPUOptions(per_process_gpu_memory_fraction=0.50)
    sessconfig = cpb2.ConfigProto(gpu_options=gpu_options)
    with g.as_default():
      x = array_ops.placeholder(
          dtype=dtypes.float32, shape=self.inp_dims, name="input")

      b = constant_op.constant(
          np.random.randn(12, 4), dtype=dtypes.float32)
      x1 = math_ops.matmul(x, b)
      b = constant_op.constant(
          np.random.randn(1, 4), dtype=dtypes.float32)
      x1 = x1 + b

      b = constant_op.constant(
          np.random.randn(48, 4), dtype=dtypes.float32)
      x2 = math_ops.matmul(x, b, transpose_a=True)
      x2 = gen_array_ops.reshape(x2, [48, 1])

      b = constant_op.constant(
          np.random.randn(4, 12), dtype=dtypes.float32)
      x3 = math_ops.matmul(x, b, transpose_b=True)

      b = constant_op.constant(
          np.random.randn(16, 48), dtype=dtypes.float32)
      x4 = math_ops.matmul(x, b, transpose_b=True, transpose_a=True)
      x4 = gen_array_ops.reshape(x4, [48, 4])

      x5 = gen_array_ops.reshape(x, [4, 12, 12])
      x5 = core.flatten(x5)
      b = constant_op.constant(
          np.random.randn(144, 48), dtype=dtypes.float32)
      x5 = math_ops.matmul(x5, b)
      b = constant_op.constant(
          np.random.randn(48), dtype=dtypes.float32)
      x5 = nn.bias_add(x5, b)
      x5 = gen_array_ops.reshape(x5, [48, 4])

      x6 = gen_array_ops.reshape(x, [4, 12, 12])
      b = constant_op.constant(
          np.random.randn(12), dtype=dtypes.float32)
      x6 = nn.bias_add(x6, b, data_format="NHWC")
      x6 = gen_array_ops.reshape(x6, [48, -1])

      x7 = gen_array_ops.reshape(x, [4, 12, 3, 4])
      b = constant_op.constant(
          np.random.randn(4), dtype=dtypes.float32)
      x7 = nn.bias_add(x7, b, data_format="NHWC")
      x7 = gen_array_ops.reshape(x7, [48, -1])

      x8 = gen_array_ops.reshape(x, [4, 12, 3, 2, 2])
      b = constant_op.constant(
          np.random.randn(2), dtype=dtypes.float32)
      x8 = nn.bias_add(x8, b, data_format="NHWC")
      x8 = gen_array_ops.reshape(x8, [48, -1])

      x9 = gen_array_ops.reshape(x, [4, 12, 3, 2, 2])
      b = constant_op.constant(
          np.random.randn(3), dtype=dtypes.float32)
      x9 = nn.bias_add(x9, b, data_format="NCHW")
      x9 = gen_array_ops.reshape(x9, [48, -1])

      x10 = gen_array_ops.reshape(x, [4, 12, 3, 4])
      b = constant_op.constant(
          np.random.randn(12), dtype=dtypes.float32)
      x10 = nn.bias_add(x10, b, data_format="NCHW")
      x10 = gen_array_ops.reshape(x10, [48, -1])

      x11 = gen_array_ops.reshape(x, [4, 12, 12])
      b = constant_op.constant(
          np.random.randn(4), dtype=dtypes.float32)
      x11 = nn.bias_add(x11, b, data_format="NCHW")
      x11 = gen_array_ops.reshape(x11, [48, -1])

      out = array_ops.concat([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11], axis=-1)
      out = array_ops.squeeze(out, name="output")

    return g.as_graph_def()
