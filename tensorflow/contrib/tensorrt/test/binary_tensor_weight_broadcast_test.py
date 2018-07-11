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
from tensorflow.python.ops import math_ops
from tensorflow.contrib.tensorrt.test.base_unit_test import BaseUnitTest


class BinaryTensorWeightBroadcastTest(BaseUnitTest):
  """unit tests for scale & elementwise layers in TF-TRT"""

  def __init__(self, log_file='log.txt'):
    super(BinaryTensorWeightBroadcastTest, self).__init__()
    self.static_mode_list = {"FP32", "FP16"}
    self.debug = True
    self.dynamic_mode_list = {}
    self.inp_dims = (10, 24, 24, 20)
    self.dummy_input = np.random.random_sample(self.inp_dims)
    self.get_network = self.get_simple_graph_def
    self.expect_nb_nodes = 35
    self.log_file = log_file
    self.test_name = self.__class__.__name__
    self.allclose_rtol = 0.1
    self.allclose_atol = 0.05

  def get_simple_graph_def(self):
    g = ops.Graph()
    gpu_options = config_pb2.GPUOptions(per_process_gpu_memory_fraction=0.50)
    sessconfig = config_pb2.ConfigProto(gpu_options=gpu_options)
    with g.as_default():
      x = array_ops.placeholder(
          dtype=dtypes.float32, shape=self.inp_dims, name="input")

      # scale
      a = constant_op.constant(np.random.randn(1), dtype=dtypes.float32)
      f = x + a
      x = math_ops.sigmoid(f)

      # scale
      a = constant_op.constant(np.random.randn(1), dtype=dtypes.float32)
      f = a + x
      x = math_ops.sigmoid(f)

      # scale
      a = constant_op.constant(np.random.randn(24, 1, 1), dtype=dtypes.float32)
      f = x + a
      x = math_ops.sigmoid(f)

      # scale
      a = constant_op.constant(np.random.randn(24, 1, 1), dtype=dtypes.float32)
      f = a + x
      x = math_ops.sigmoid(f)

      # scale
      a = constant_op.constant(
          np.random.randn(24, 24, 20), dtype=dtypes.float32)
      f = a + x
      x = math_ops.sigmoid(f)

      # scale
      a = constant_op.constant(
          np.random.randn(24, 24, 20), dtype=dtypes.float32)
      f = x + a
      x = math_ops.sigmoid(f)

      # elementwise
      a = constant_op.constant(np.random.randn(20), dtype=dtypes.float32)
      f = x + a
      x = math_ops.sigmoid(f)

      # elementwise
      a = constant_op.constant(np.random.randn(20), dtype=dtypes.float32)
      f = a + x
      x = math_ops.sigmoid(f)

      # elementwise
      a = constant_op.constant(
          np.random.randn(1, 24, 1, 1), dtype=dtypes.float32)
      f = a + x
      x = math_ops.sigmoid(f)

      # elementwise
      a = constant_op.constant(
          np.random.randn(1, 24, 1, 1), dtype=dtypes.float32)
      f = x + a
      x = math_ops.sigmoid(f)

      # elementwise
      a = constant_op.constant(
          np.random.randn(1, 24, 24, 1), dtype=dtypes.float32)
      f = a + x
      x = math_ops.sigmoid(f)

      # elementwise
      a = constant_op.constant(
          np.random.randn(1, 24, 24, 1), dtype=dtypes.float32)
      f = x + a
      x = math_ops.sigmoid(f)

      # elementwise
      a = constant_op.constant(
          np.random.randn(1, 24, 24, 20), dtype=dtypes.float32)
      f = a + x
      x = math_ops.sigmoid(f)

      # elementwise
      a = constant_op.constant(
          np.random.randn(1, 24, 24, 20), dtype=dtypes.float32)
      f = x + a
      x = math_ops.sigmoid(f)

      # elementwise
      a = constant_op.constant(np.random.randn(24, 20), dtype=dtypes.float32)
      f = a + x
      x = math_ops.sigmoid(f)

      # elementwise
      a = constant_op.constant(np.random.randn(24, 20), dtype=dtypes.float32)
      f = x + a
      x = math_ops.sigmoid(f)

      gen_array_ops.reshape(x, [5, -1], name="output")

    return g.as_graph_def()
