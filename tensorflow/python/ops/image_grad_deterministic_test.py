# Copyright 2020-2021 The TensorFlow Authors. All Rights Reserved.
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
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import image_grad_test_base as test_base
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class ResizeBilinearOpDeterministicTest(test_base.ResizeBilinearOpTestBase):
  """Test that ResizeBilinearGrad operates reproducibly.

  Inheriting from test_base.ResizeBilinearOpTestBase ensures that regular op
  functionality is correct when the deterministic code-path is selected.
  """

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
  @test_util.run_gpu_only
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
        # gradient result, not passing the upstream gradients through the op's
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


class CropAndResizeOpDeterminismExceptionsTest(test.TestCase):
  """Test d9m-unimplemented exceptions from CropAndResizeBackprop{Image|Boxes}.

  Test that tf.errors.UnimplementedError is thrown or not thrown, as
  appropriate, by the GPU code-paths for CropAndResizeBackprop{Image|Boxes} when
  deterministic ops are enabled.

  This test assumes that test_base.CropAndResizeOpTestBase runs all the same
  test cases when deterministic ops are not enabled and will therefore detect
  erroneous exception throwing in those cases.
  """

  def _genParams(self, dtype=dtypes.float32):
    batch_size = 1
    image_height = 10
    image_width = 10
    channels = 1
    image_shape = (batch_size, image_height, image_width, channels)
    num_boxes = 3
    boxes_shape = (num_boxes, 4)
    random_seed.set_seed(123)
    image = random_ops.random_normal(shape=image_shape, dtype=dtype)
    boxes = random_ops.random_uniform(shape=boxes_shape, dtype=dtypes.float32)
    box_indices = random_ops.random_uniform(
        shape=(num_boxes,), minval=0, maxval=batch_size, dtype=dtypes.int32)
    crop_size = constant_op.constant([3, 3], dtype=dtypes.int32)
    return image, boxes, box_indices, crop_size

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_gpu_only
  def testExceptionThrowing(self):
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      image, boxes, box_indices, crop_size = self._genParams(dtype)
      with backprop.GradientTape(persistent=True) as tape:
        tape.watch(image)
        tape.watch(boxes)
        op_output = image_ops.crop_and_resize_v2(image, boxes, box_indices,
                                                 crop_size)
      image_error_message = ('Deterministic GPU implementation of' +
                             ' CropAndResizeBackpropImage not available')
      with self.assertRaisesRegex(errors_impl.UnimplementedError,
                                  image_error_message):
        result = tape.gradient(op_output, image)
        self.evaluate(result)
      expected_error_message = ('Deterministic GPU implementation of' +
                                ' CropAndResizeBackpropBoxes not available')
      if context.executing_eagerly():
        # With eager execution, the backprop-to-image code is apparently
        # executed (first), even when its output is never used.
        expected_error_message = image_error_message
      with self.assertRaisesRegex(errors_impl.UnimplementedError,
                                  expected_error_message):
        result = tape.gradient(op_output, boxes)
        self.evaluate(result)


class CropAndResizeOpDeterministicTest(test_base.CropAndResizeOpTestBase):
  """Test that CropAndResizeBackprop{Image|Boxes} operates reproducibly.

  Inheriting from test_base.CropAndResizeOpTestBase ensures that regular op
  functionality is correct when the deterministic code-path is selected.
  """

  def _randomFloats(self, shape, low=0.0, high=1.0, dtype=dtypes.float32):
    """Generate a tensor of random floating-point values.

    Values will be continuously distributed in the range [low, high).

    Note that we use numpy to generate random numbers and then feed the result
    through a constant op to avoid the re-rolling of TensorFlow random ops on
    each run in graph mode.

    Args:
      shape: The output shape.
      low: Lower bound of random numbers generated, inclusive.
      high: Upper bound of random numbers generated, exclusive.
      dtype: The output dtype.

    Returns:
      A random tensor
    """
    val = np.random.random_sample(
        shape)  # float64 continuous uniform [0.0, 1.0)
    diff = high - low
    val *= diff
    val += low
    return constant_op.constant(val, dtype=dtype)

  def _randomInts(self, shape, low, high):
    """Generate a tensor of random 32-bit integer values.

    Note that we use numpy to generate random numbers and then feed the result
    through a constant op to avoid the re-rolling of TensorFlow random ops on
    each run in graph mode.

    Args:
      shape: The output shape.
      low: Lower bound of random numbers generated, inclusive.
      high: Upper bound of random numbers generated, exclusive.

    Returns:
      A random tensor
    """
    val = np.random.randint(low=low, high=high, size=shape)
    return constant_op.constant(val, dtype=dtypes.int32)

  def _genParams(self, dtype=dtypes.float32):
    batch_size = 16
    input_height = 64
    input_width = 64
    depth = 1
    input_shape = (batch_size, input_height, input_width, depth)
    np.random.seed(456)
    image = self._randomFloats(input_shape, low=-1.0, high=1.0, dtype=dtype)
    box_count = 4 * batch_size
    boxes = self._randomFloats((box_count, 4),
                               low=0.0,
                               high=1.01,
                               dtype=dtypes.float32)
    box_indices = self._randomInts((box_count,), low=0, high=batch_size)
    crop_size = [input_height * 2, input_width * 2]
    output_shape = (box_count, *crop_size, depth)
    # The output of this op is always float32, regardless of image data type
    injected_gradients = self._randomFloats(
        output_shape, low=-0.001, high=0.001, dtype=dtypes.float32)
    return image, boxes, box_indices, crop_size, injected_gradients

  def _testReproducibleBackprop(self, test_image_not_boxes):
    with test_util.force_cpu():
      for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
        params = self._genParams(dtype)
        image, boxes, box_indices, crop_size, injected_gradients = params

        with backprop.GradientTape(persistent=True) as tape:
          tape.watch([image, boxes])
          output = image_ops.crop_and_resize_v2(
              image, boxes, box_indices, crop_size, method='bilinear')
          upstream = output * injected_gradients

        image_gradients_a, boxes_gradients_a = tape.gradient(
            upstream, [image, boxes])
        for _ in range(5):
          image_gradients_b, boxes_gradients_b = tape.gradient(
              upstream, [image, boxes])
          if test_image_not_boxes:
            self.assertAllEqual(image_gradients_a, image_gradients_b)
          else:
            self.assertAllEqual(boxes_gradients_a, boxes_gradients_b)

  @test_util.run_in_graph_and_eager_modes
  def testReproducibleBackpropToImage(self):
    """Test that backprop to image is reproducible.

    With non-reproducible ordering of reduction operations, upsampling of a
    crop, leading to three or more output pixels being derived from an input
    pixel, can contribute to nondeterminism in the gradient associated with that
    input pixel location.

    Note that the number of boxes can be less than, equal to, or greater than
    the batch size. Wth non-reproducible ordering of reduction operations, three
    or more crops overlapping on the same input image pixel can independently
    contribute to nondeterminism in the image gradient associated with that
    input pixel location. This is independent of contributions caused by the
    upsampling of any given crop.
    """
    self._testReproducibleBackprop(test_image_not_boxes=True)

  @test_util.run_in_graph_and_eager_modes
  def testReproducibleBackpropToBoxes(self):
    """Test that backprop to boxes is reproducible.

    If the input and output dimensions are the same, then the boxes gradients
    will be deterministically zero. Otherwise, in the presence of
    non-reproducible ordering of reduction operations, nondeterminism can be
    introduced, whether there is upsampling or downsampling and whether or not
    there are overlapping crops.
    """
    self._testReproducibleBackprop(test_image_not_boxes=False)


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
