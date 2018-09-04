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

from tensorflow.contrib.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import test


class ConstBroadcastTest(trt_test.TfTrtIntegrationTestBase):

  def GetParams(self):
    """Test for Constant broadcasting in TF-TRT."""
    dtype = dtypes.float32
    input_name = 'input'
    input_dims = [5, 12, 12, 2]
    output_name = 'output'
    g = ops.Graph()
    with g.as_default():
      x = array_ops.placeholder(dtype=dtype, shape=input_dims, name=input_name)
      filt1 = constant_op.constant(
          0.3, shape=(3, 3, 2, 1), dtype=dtype, name='filt1')
      y1 = nn.conv2d(x, filt1, strides=[1, 1, 1, 1], padding='SAME', name='y1')
      z1 = nn.relu(y1, name='z1')
      filt2 = constant_op.constant(
          np.random.randn(9), shape=(3, 3, 1, 1), dtype=dtype, name='filt2')
      y2 = nn.conv2d(z1, filt2, strides=[1, 1, 1, 1], padding='SAME', name='y2')
      z2 = nn.relu(y2, name='z')
      filt3 = constant_op.constant(
          np.random.randn(3, 3, 1, 1),
          shape=(3, 3, 1, 1),
          dtype=dtype,
          name='filt3')
      y3 = nn.conv2d(z2, filt3, strides=[1, 1, 1, 1], padding='SAME', name='y3')
      nn.relu(y3, name=output_name)
    return trt_test.TfTrtIntegrationTestParams(
        gdef=g.as_graph_def(),
        input_names=[input_name],
        input_dims=[input_dims],
        output_names=[output_name],
        expected_output_dims=[(5, 12, 12, 1)])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ['my_trt_op_0']

  def ExpectedAbsoluteTolerance(self, run_params):
    """The absolute tolerance to compare floating point results."""
    return 1.e-04 if run_params.precision_mode == 'FP32' else 1.e-02

  def ExpectedRelativeTolerance(self, run_params):
    """The relative tolerance to compare floating point results."""
    return 1.e-04 if run_params.precision_mode == 'FP32' else 1.e-02


if __name__ == '__main__':
  test.main()
