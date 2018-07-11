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
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn_impl
from tensorflow.contrib.tensorrt.test.base_unit_test import BaseUnitTest


class VGGBlockTest(BaseUnitTest):
  """single vgg layer test in TF-TRT conversion"""

  def __init__(self, log_file='log.txt'):
    super(VGGBlockTest, self).__init__()
    self.static_mode_list = {"FP32", "FP16"}
    self.debug = True
    self.dynamic_mode_list = {}
    self.inp_dims = (5, 8, 8, 2)
    self.dummy_input = np.random.random_sample(self.inp_dims)
    self.get_network = self.get_simple_graph_def
    self.expect_nb_nodes = 7
    self.log_file = log_file
    self.test_name = self.__class__.__name__

  def get_simple_graph_def(self):
    g = ops.Graph()
    gpu_options = config_pb2.GPUOptions(per_process_gpu_memory_fraction=0.50)
    sessconfig = config_pb2.ConfigProto(gpu_options=gpu_options)
    with g.as_default():
      x = array_ops.placeholder(
          dtype=dtypes.float32, shape=self.inp_dims, name="input")
      x, mean_x, var_x = nn_impl.fused_batch_norm(
          x,
          np.random.randn(2).astype(np.float32),
          np.random.randn(2).astype(np.float32),
          mean=np.random.randn(2).astype(np.float32),
          variance=np.random.randn(2).astype(np.float32),
          is_training=False)
      e = constant_op.constant(
          np.random.randn(1, 1, 2, 6), name="weights", dtype=dtypes.float32)
      conv = nn.conv2d(
          input=x, filter=e, strides=[1, 2, 2, 1], padding="SAME", name="conv")
      b = constant_op.constant(
          np.random.randn(6), name="bias", dtype=dtypes.float32)
      t = nn.bias_add(conv, b, name="biasAdd")
      relu = nn.relu(t, "relu")
      idty = array_ops.identity(relu, "ID")
      v = nn_ops.max_pool(
          idty, [1, 2, 2, 1], [1, 2, 2, 1], "VALID", name="max_pool")
      array_ops.squeeze(v, name="output")

    return g.as_graph_def()
