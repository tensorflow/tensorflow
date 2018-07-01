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
from tensorflow.python.training import training
from base_unit_test import BaseUnitTest
from utilities import get_all_variables

class NeighboringEngineTest(BaseUnitTest):
  """Neighboring node wiring tests in TF-TRT conversion"""

  def __init__(self, log_file='log.txt'):
    super(NeighboringEngineTest, self).__init__()
    self.static_mode_list = {"FP32", "FP16"}
    self.debug=True
    self.dynamic_mode_list = {}
    self.inp_dims = (2, 3, 7, 5)
    self.dummy_input = np.random.random_sample(self.inp_dims)
    self.get_network = self.neighboring_tensor_test
    self.expect_nb_nodes = 5
    self.log_file = log_file
    self.test_name = self.__class__.__name__ 
    self.allclose_rtol = 0.05
    self.allclose_atol = 0.05

  def neighboring_tensor_test(self):
    g = ops.Graph()
    gpu_options = cpb2.GPUOptions(per_process_gpu_memory_fraction=0.50)
    sessconfig = cpb2.ConfigProto(gpu_options=gpu_options)
    with g.as_default():
      x = array_ops.placeholder(
          dtype=dtypes.float32, shape=self.inp_dims, name="input")
      e = constant_op.constant(
          np.random.normal(.3, 0.05, [3,2,3,4]),
          name="weights",
          dtype=dtypes.float32)
      conv = nn.conv2d(
          input=x, filter=e, data_format="NCHW",strides=[1, 1, 1, 1], padding="VALID", name="conv")
      b = constant_op.constant(
          np.random.normal(1.0, 1.0, [1,4,1,1]), name="bias", dtype=dtypes.float32)
      t = conv*b

      e = gen_math_ops.tan(conv)
      t = t - e
      array_ops.squeeze(t, name="output")

    return g.as_graph_def()
