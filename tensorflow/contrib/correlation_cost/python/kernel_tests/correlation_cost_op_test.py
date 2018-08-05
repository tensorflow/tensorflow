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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


from tensorflow.contrib.correlation_cost.python.ops import correlation_cost_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.framework import constant_op


class CorrelationCostTest(test.TestCase):

  def _forward(self, input_a, input_b,
               kernel_size,
               max_displacement,
               stride_1,
               stride_2,
               pad,
               data_format,
               use_gpu=False):
    with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu) as sess:

      input_a_op = ops.convert_to_tensor(input_a)
      input_b_op = ops.convert_to_tensor(input_b)

      kernel_size = 1
      max_displacement = 2
      stride_1 = 1
      stride_2 = 2
      pad = 4

      call_op = correlation_cost_op.correlation_cost
      actual_op = call_op(input_a_op, input_b_op,
                          kernel_size=kernel_size,
                          max_displacement=max_displacement,
                          stride_1=stride_1,
                          stride_2=stride_2,
                          pad=pad,
                          data_format=data_format)

      return sess.run(actual_op)

  def _forward_both(self, shape, data_format='NCHW', dtype=dtypes.float32):
    # some shape to test uneven number of channels
    input_a = np.random.randn(*shape)
    input_b = np.random.randn(*shape)

    input_a = constant_op.constant(input_a, dtype=dtype)
    input_b = constant_op.constant(input_b, dtype=dtype)

    kernel_size = 1
    max_displacement = 2
    stride_1 = 1
    stride_2 = 2
    pad = 4

    if data_format == 'NHWC':
      input_a = array_ops.transpose(input_a, [0, 2, 3, 1])
      input_b = array_ops.transpose(input_b, [0, 2, 3, 1])

    actual_cpu = self._forward(input_a, input_b,
                               kernel_size=kernel_size,
                               max_displacement=max_displacement,
                               stride_1=stride_1,
                               stride_2=stride_2,
                               pad=pad,
                               data_format=data_format,
                               use_gpu=False)

    actual_gpu = self._forward(input_a, input_b,
                               kernel_size=kernel_size,
                               max_displacement=max_displacement,
                               stride_1=stride_1,
                               stride_2=stride_2,
                               pad=pad,
                               data_format=data_format,
                               use_gpu=True)

    self.assertEqual(actual_cpu.shape, actual_gpu.shape)
    self.assertAllClose(actual_cpu, actual_gpu)

  def _gradients(self, data_format='NCHW', use_gpu=False):

    batch, channels, height, width = 2, 3, 5, 6
    input_a = np.random.randn(batch, channels, height, width)
    input_b = np.random.randn(batch, channels, height, width)

    kernel_size = 1
    max_displacement = 2
    stride_1 = 1
    stride_2 = 2
    pad = 4

    if data_format == 'NHWC':
      input_a = input_a.transpose(0, 2, 3, 1)
      input_b = input_b.transpose(0, 2, 3, 1)

    with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu):

      input_a_op = ops.convert_to_tensor(input_a, dtype=dtypes.float32)
      input_b_op = ops.convert_to_tensor(input_b, dtype=dtypes.float32)

      call_op = correlation_cost_op.correlation_cost
      actual_op = call_op(input_a_op, input_b_op,
                          kernel_size=kernel_size,
                          max_displacement=max_displacement,
                          stride_1=stride_1,
                          stride_2=stride_2,
                          pad=pad,
                          data_format=data_format)

      err_a = test.compute_gradient_error(
          [input_a_op, input_b_op],
          [input_a.shape, input_b.shape],
          actual_op, actual_op.shape.as_list())

      self.assertLess(err_a, 1e-4)

  def testForwardSameFloatLarge(self):
    # to test channel_num larger than 1 warp
    self._forward_both((1, 65, 3, 4), data_format='NCHW', dtype=dtypes.float32)
    self._forward_both((1, 65, 3, 4), data_format='NHWC', dtype=dtypes.float32)

  def testForwardSameDoubleLarge(self):
    # to test channel_num larger than 1 warp
    self._forward_both((1, 65, 3, 4), data_format='NCHW', dtype=dtypes.float64)
    self._forward_both((1, 65, 3, 4), data_format='NHWC', dtype=dtypes.float64)

  def testForwardSameFloatSmall(self):
    # to test channel_num smaller than 1 warp
    self._forward_both((1, 15, 3, 4), data_format='NCHW', dtype=dtypes.float32)
    self._forward_both((1, 15, 3, 4), data_format='NHWC', dtype=dtypes.float32)

  def testForwardSameDoubleSmall(self):
    # to test channel_num smaller than 1 warp
    self._forward_both((1, 15, 3, 4), data_format='NCHW', dtype=dtypes.float64)
    self._forward_both((1, 15, 3, 4), data_format='NHWC', dtype=dtypes.float64)

  def testBackwardNCHW(self):
    self._gradients(data_format='NCHW', use_gpu=False)
    self._gradients(data_format='NCHW', use_gpu=True)

  def testBackwardNHWC(self):
    self._gradients(data_format='NHWC', use_gpu=False)
    self._gradients(data_format='NHWC', use_gpu=True)


if __name__ == "__main__":
  test.main()
