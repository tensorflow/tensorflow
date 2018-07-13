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

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import training
from tensorflow.contrib.tensorrt.test.base_unit_test import BaseUnitTest
from tensorflow.contrib.tensorrt.test.utilities import get_all_variables


class UnaryTest(BaseUnitTest):
  """Unit tests for unary operations in TF-TRT"""

  def __init__(self, log_file='log.txt'):
    super(UnaryTest, self).__init__()
    self.static_mode_list = {"FP32", "FP16"}
    self.debug = True
    self.dynamic_mode_list = {}
    self.inp_dims = (12, 5, 8, 1, 1, 12)
    self.dummy_input = np.random.random_sample(self.inp_dims)
    self.get_network = self.unary_test
    self.expect_nb_nodes = 17
    self.log_file = log_file
    self.test_name = self.__class__.__name__
    self.ckpt = "./tmp.ckpt"

  def unary_test(self):
    g = ops.Graph()
    gpu_options = config_pb2.GPUOptions(per_process_gpu_memory_fraction=0.50)
    sessconfig = config_pb2.ConfigProto(gpu_options=gpu_options)
    with g.as_default():
      x = array_ops.placeholder(
          dtype=dtypes.float32, shape=self.inp_dims, name="input")
      q = math_ops.abs(x)
      q = q + 1.0
      q = gen_math_ops.exp(q)
      q = gen_math_ops.log(q)
      q = array_ops.squeeze(q, axis=-2)
      q = math_ops.abs(q)
      q = q + 2.2
      q = gen_math_ops.sqrt(q)
      q = gen_math_ops.rsqrt(q)
      q = math_ops.negative(q)
      q = array_ops.squeeze(q, axis=3)
      q = math_ops.abs(q)
      q = q + 3.0
      a = gen_math_ops.reciprocal(q)

      x = constant_op.constant(np.random.randn(5, 8, 12), dtype=dtypes.float32)
      q = math_ops.abs(x)
      q = q + 2.0
      q = gen_math_ops.exp(q)
      q = gen_math_ops.log(q)
      q = math_ops.abs(q)
      q = q + 2.1
      q = gen_math_ops.sqrt(q)
      q = gen_math_ops.rsqrt(q)
      q = math_ops.negative(q)
      q = math_ops.abs(q)
      q = q + 4.0
      b = gen_math_ops.reciprocal(q)

      # TODO(jie): this one will break, broadcasting on batch.
      x = variable_scope.get_variable(
          "test", [12, 40, 12],
          dtype=dtypes.float32,
          initializer=init_ops.truncated_normal_initializer)
      x = gen_array_ops.reshape(x, [12, 5, 8, 1, 12, 1, 1])
      q = math_ops.abs(x)
      q = q + 5.0
      q = gen_math_ops.exp(q)
      q = array_ops.squeeze(q, axis=[-1, -2, 3])
      q = gen_math_ops.log(q)
      q = math_ops.abs(q)
      q = q + 5.1
      q = gen_array_ops.reshape(q, [12, 5, 1, 1, 8, 1, 12])
      q = array_ops.squeeze(q, axis=[5, 2, 3])
      q = gen_math_ops.sqrt(q)
      q = math_ops.abs(q)
      q = q + 5.2
      q = gen_math_ops.rsqrt(q)
      q = math_ops.negative(q)
      q = math_ops.abs(q)
      q = q + 5.3
      c = gen_math_ops.reciprocal(q)

      q = a * b
      q = q / c
      array_ops.squeeze(q, name="output")

      with session.Session(config=sessconfig, graph=g) as sess:
        names_var_list = get_all_variables(sess)
        saver = training.Saver(names_var_list)
        sess.run(variables.global_variables_initializer())
        saver.save(sess, self.ckpt)
    return g.as_graph_def()
