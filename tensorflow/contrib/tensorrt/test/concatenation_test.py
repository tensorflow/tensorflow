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
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.contrib.tensorrt.test.base_unit_test import BaseUnitTest


class ConcatenationTest(BaseUnitTest):
  """Testing Concatenation in TF-TRT conversion"""

  def __init__(self, log_file='log.txt'):
    super(ConcatenationTest, self).__init__()
    self.static_mode_list = {"FP32", "FP16"}
    self.debug = True
    self.dynamic_mode_list = {}
    self.inp_dims = (2, 3, 3, 1)
    self.dummy_input = np.random.random_sample(self.inp_dims)
    self.get_network = self.get_simple_graph_def
    self.expect_nb_nodes = 4
    self.log_file = log_file
    self.test_name = self.__class__.__name__

  def get_simple_graph_def(self):
    g = ops.Graph()
    gpu_options = config_pb2.GPUOptions(per_process_gpu_memory_fraction=0.50)
    sessconfig = config_pb2.ConfigProto(gpu_options=gpu_options)
    with g.as_default():
      x = array_ops.placeholder(
          dtype=dtypes.float32, shape=self.inp_dims, name="input")

      # scale
      a = constant_op.constant(np.random.randn(3, 1, 1), dtype=dtypes.float32)
      r1 = x / a
      a = constant_op.constant(np.random.randn(3, 1, 1), dtype=dtypes.float32)
      r2 = a / x
      a = constant_op.constant(np.random.randn(1, 3, 1), dtype=dtypes.float32)
      r3 = a + x
      a = constant_op.constant(np.random.randn(1, 3, 1), dtype=dtypes.float32)
      r4 = x * a
      a = constant_op.constant(np.random.randn(3, 1, 1), dtype=dtypes.float32)
      r5 = x - a
      a = constant_op.constant(np.random.randn(3, 1, 1), dtype=dtypes.float32)
      r6 = a - x
      a = constant_op.constant(np.random.randn(3, 1), dtype=dtypes.float32)
      r7 = x - a
      a = constant_op.constant(np.random.randn(3, 1), dtype=dtypes.float32)
      r8 = a - x
      a = constant_op.constant(np.random.randn(3, 1, 1), dtype=dtypes.float32)
      r9 = gen_math_ops.maximum(x, a)
      a = constant_op.constant(np.random.randn(3, 1), dtype=dtypes.float32)
      r10 = gen_math_ops.minimum(a, x)
      a = constant_op.constant(np.random.randn(3), dtype=dtypes.float32)
      r11 = x * a
      a = constant_op.constant(np.random.randn(1), dtype=dtypes.float32)
      r12 = a * x
      concat1 = array_ops.concat([r1, r2, r3, r4, r5, r6], axis=-1)
      concat2 = array_ops.concat([r7, r8, r9, r10, r11, r12], axis=3)
      x = array_ops.concat([concat1, concat2], axis=-1)

      gen_array_ops.reshape(x, [2, -1], name="output")

    return g.as_graph_def()
