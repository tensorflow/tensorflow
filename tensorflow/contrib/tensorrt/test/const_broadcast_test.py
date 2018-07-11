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
"""Script to test TF-TensorRT integration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.contrib.tensorrt.test.base_unit_test import BaseUnitTest


class ConstBroadcastTest(BaseUnitTest):
  """Testing Constant broadcasting in TF-TRT"""

  def __init__(self, log_file='log.txt'):
    super(ConstBroadcastTest, self).__init__()
    self.static_mode_list = {"FP32", "FP16"}
    self.debug = True
    self.dynamic_mode_list = {}
    self.inp_dims = (5, 12, 12, 2)
    self.dummy_input = np.random.random_sample(self.inp_dims)
    self.get_network = self.conv_broadcast
    self.expect_nb_nodes = 7
    self.log_file = log_file
    self.test_name = self.__class__.__name__
    self.allclose_rtol = 0.05
    self.allclose_atol = 0.05

  def conv_broadcast(self):
    g = ops.Graph()
    gpu_options = config_pb2.GPUOptions(per_process_gpu_memory_fraction=0.50)
    sessconfig = config_pb2.ConfigProto(gpu_options=gpu_options)
    with g.as_default():
      x = array_ops.placeholder(
          dtype=dtypes.float32, shape=self.inp_dims, name="input")
      filt1 = constant_op.constant(
          1, shape=(3, 3, 2, 1), dtype=dtypes.float32, name='filt1')
      y1 = nn.conv2d(x, filt1, strides=[1, 1, 1, 1], padding='SAME', name='y1')
      z1 = nn.relu(y1, name='z1')
      filt2 = constant_op.constant(
          np.random.randn(9),
          shape=(3, 3, 1, 1),
          dtype=dtypes.float32,
          name='filt2')
      y2 = nn.conv2d(z1, filt2, strides=[1, 1, 1, 1], padding='SAME', name='y2')
      z2 = nn.relu(y2, name='z')
      filt3 = constant_op.constant(
          np.random.randn(3, 3, 1, 1),
          shape=(3, 3, 1, 1),
          dtype=dtypes.float32,
          name='filt3')
      y3 = nn.conv2d(z2, filt3, strides=[1, 1, 1, 1], padding='SAME', name='y3')
      z = nn.relu(y3, name='output')

    return g.as_graph_def()
