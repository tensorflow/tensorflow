# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Functional tests for deterministic image op gradient functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from absl.testing import parameterized

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import image_grad_test_base as test_base
from tensorflow.python.ops import image_ops
from tensorflow.python.platform import test


class ResizeBilinearOpDeterministicTest(test_base.ResizeBilinearOpTestBase):

  def _randomNDArray(self, shape):
    return 2 * np.random.random_sample(shape) - 1

  def _randomDataOp(self, shape, data_type):
    return constant_op.constant(self._randomNDArray(shape), dtype=data_type)

  @parameterized.parameters(
      # Note that there is no 16-bit floating point format registered for GPU
      {
          'align_corners': False,
          'half_pixel_centers': False,
          'data_type': dtypes.float32
      },
      {
          'align_corners': False,
          'half_pixel_centers': False,
          'data_type': dtypes.float64
      },
      {
          'align_corners': True,
          'half_pixel_centers': False,
          'data_type': dtypes.float32
      },
      {
          'align_corners': False,
          'half_pixel_centers': True,
          'data_type': dtypes.float32
      })
  @test_util.run_in_graph_and_eager_modes
  @test_util.run_cuda_only
  def testDeterministicGradients(self, align_corners, half_pixel_centers,
                                 data_type):
    if not align_corners and test_util.is_xla_enabled():
      # Align corners is deprecated in TF2.0, but align_corners==False is not
      # supported by XLA.
      self.skipTest('align_corners==False not currently supported by XLA')
    with self.session(force_gpu=True):
      seed = (
          hash(align_corners) % 256 + hash(half_pixel_centers) % 256 +
          hash(data_type) % 256)
      np.random.seed(seed)
      input_shape = (1, 25, 12, 3)  # NHWC
      output_shape = (1, 200, 250, 3)
      input_image = self._randomDataOp(input_shape, data_type)
      repeat_count = 3
      if context.executing_eagerly():

        def resize_bilinear_gradients(local_seed):
          np.random.seed(local_seed)
          upstream_gradients = self._randomDataOp(output_shape, dtypes.float32)
          with backprop.GradientTape(persistent=True) as tape:
            tape.watch(input_image)
            output_image = image_ops.resize_bilinear(
                input_image,
                output_shape[1:3],
                align_corners=align_corners,
                half_pixel_centers=half_pixel_centers)
            gradient_injector_output = output_image * upstream_gradients
          return tape.gradient(gradient_injector_output, input_image)

        for i in range(repeat_count):
          local_seed = seed + i  # select different upstream gradients
          result_a = resize_bilinear_gradients(local_seed)
          result_b = resize_bilinear_gradients(local_seed)
          self.assertAllEqual(result_a, result_b)
      else:  # graph mode
        upstream_gradients = array_ops.placeholder(
            dtypes.float32, shape=output_shape, name='upstream_gradients')
        output_image = image_ops.resize_bilinear(
            input_image,
            output_shape[1:3],
            align_corners=align_corners,
            half_pixel_centers=half_pixel_centers)
        gradient_injector_output = output_image * upstream_gradients
        # The gradient function behaves as if grad_ys is multiplied by the op
        # gradient result, not passing the upstram gradients through the op's
        # gradient generation graph. This is the reason for using the
        # gradient injector
        resize_bilinear_gradients = gradients_impl.gradients(
            gradient_injector_output,
            input_image,
            grad_ys=None,
            colocate_gradients_with_ops=True)[0]
        for i in range(repeat_count):
          feed_dict = {upstream_gradients: self._randomNDArray(output_shape)}
          result_a = resize_bilinear_gradients.eval(feed_dict=feed_dict)
          result_b = resize_bilinear_gradients.eval(feed_dict=feed_dict)
          self.assertAllEqual(result_a, result_b)


if __name__ == '__main__':
  # Note that the effect of setting the following environment variable to
  # 'true' is not tested. Unless we can find a simpler pattern for testing these
  # environment variables, it would require this file to be made into a base
  # and then two more test files to be created.
  #
  # When deterministic op functionality can be enabled and disabled between test
  # cases in the same process, then the tests for deterministic op
  # functionality, for this op and for other ops, will be able to be included in
  # the same file with the regular tests, simplifying the organization of tests
  # and test files.
  os.environ['TF_DETERMINISTIC_OPS'] = '1'
  test.main()
