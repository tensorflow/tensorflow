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

class BatchMatMulTest(BaseUnitTest):
  """Testing BatchMatMul in TF-TRT conversion"""

  def __init__(self, log_file='log.txt'):
    super(BatchMatMulTest, self).__init__()
    self.static_mode_list = {"FP32", "FP16"}
    self.debug=True
    self.dynamic_mode_list = {}
    self.inp_dims = (12, 5, 8, 12)
    self.dummy_input = np.random.random_sample(self.inp_dims)
    self.get_network = self.matmul_test
    self.expect_nb_nodes = 16
    self.log_file = log_file
    self.test_name = self.__class__.__name__ 
    self.ckpt = "./tmp.ckpt"
    sess = csess.Session()

  def matmul_test(self):
    g = ops.Graph()
    gpu_options = cpb2.GPUOptions()
    sessconfig = cpb2.ConfigProto(gpu_options=gpu_options)
    with g.as_default():
      x = array_ops.placeholder(
          dtype=dtypes.float32, shape=self.inp_dims, name="input")

      b = constant_op.constant(
          np.random.randn(12, 5, 12, 7), dtype=dtypes.float32)
      x1 = math_ops.matmul(x, b)
      b = constant_op.constant(
          np.random.randn(5, 1, 1), dtype=dtypes.float32)
      x1 = x1 + b

      var = variable_scope.get_variable("test", [12, 5, 12, 7], dtype=dtypes.float32, initializer=init_ops.truncated_normal_initializer)
      x2 = math_ops.matmul(x, var)
      b = constant_op.constant(
          np.random.randn(5, 1, 1), dtype=dtypes.float32)
      x2 = x2 * b

      var = variable_scope.get_variable("test2", [12, 84], dtype=dtypes.float32, initializer=init_ops.truncated_normal_initializer)
      c = gen_array_ops.reshape(x, [12, 40, 12])
      b = gen_array_ops.reshape(var, [12, 12, 7])
      x3 = math_ops.matmul(c, b)
      b = constant_op.constant(
          np.random.randn(40, 1), dtype=dtypes.float32)
      x3 = x3 + b
      x3 = gen_array_ops.reshape(x3, [12, 5, 8, 7])

      out = x3 + x1
      array_ops.squeeze(out, name="output")

      with csess.Session(config=sessconfig, graph=g) as sess:
        names_var_list = get_all_variables(sess)
        saver = training.Saver(names_var_list)
        sess.run(variables.global_variables_initializer())
        saver.save(sess, self.ckpt)
    return g.as_graph_def()
