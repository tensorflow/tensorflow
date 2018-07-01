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

import argparse
import numpy as np
import tensorflow as tf

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

class ConstBroadcastTest(BaseUnitTest):
  """Testing Constant broadcasting in TF-TRT"""

  def __init__(self, log_file='log.txt'):
    super(ConstBroadcastTest, self).__init__()
    self.static_mode_list = {"FP32", "FP16"}
    self.debug=True
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
    gpu_options = cpb2.GPUOptions(per_process_gpu_memory_fraction=0.50)
    sessconfig = cpb2.ConfigProto(gpu_options=gpu_options)
    with g.as_default():
      x = array_ops.placeholder(
          dtype=dtypes.float32, shape=self.inp_dims, name="input")
      filt1 = tf.constant(1, shape=(3,3,2,1), dtype=tf.float32, name='filt1')
      y1 = tf.nn.conv2d(x, filt1, strides=[1,1, 1, 1], padding='SAME', name='y1')
      z1 = tf.nn.relu(y1, name='z1')
      filt2 = tf.constant(np.random.randn(9), shape=(3,3,1,1), dtype=tf.float32, name='filt2')
      y2 = tf.nn.conv2d(z1, filt2, strides=[1,1, 1, 1], padding='SAME', name='y2')
      z2 = tf.nn.relu(y2, name='z')
      filt3 = tf.constant(np.random.randn(3,3,1,1), shape=(3,3,1,1), dtype=tf.float32, name='filt3')
      y3 = tf.nn.conv2d(z2, filt3, strides=[1,1, 1, 1], padding='SAME', name='y3')
      z = tf.nn.relu(y3, name='output')

    return g.as_graph_def()
