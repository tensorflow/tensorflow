# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Tests for tensorflow.ops.image_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import test_util
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import googletest


class RGBToHSVTest(test_util.TensorFlowTestCase):

  def testBatch(self):
    # Build an arbitrary RGB image
    np.random.seed(7)
    batch_size = 5
    shape = (batch_size, 2, 7, 3)
    inp = np.random.rand(*shape).astype(np.float32)

    # Convert to HSV and back, as a batch and individually
    with self.test_session() as sess:
      batch0 = constant_op.constant(inp)
      batch1 = image_ops.rgb_to_hsv(batch0)
      batch2 = image_ops.hsv_to_rgb(batch1)
      split0 = array_ops.unpack(batch0)
      split1 = list(map(image_ops.rgb_to_hsv, split0))
      split2 = list(map(image_ops.hsv_to_rgb, split1))
      join1 = array_ops.pack(split1)
      join2 = array_ops.pack(split2)
      batch1, batch2, join1, join2 = sess.run([batch1, batch2, join1, join2])

    # Verify that processing batch elements together is the same as separate
    self.assertAllClose(batch1, join1)
    self.assertAllClose(batch2, join2)
    self.assertAllClose(batch2, inp)

  def testRGBToHSVRoundTrip(self):
    data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    rgb_np = np.array(data, dtype=np.float32).reshape([2, 2, 3]) / 255.
    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu):
        hsv = image_ops.rgb_to_hsv(rgb_np)
        rgb = image_ops.hsv_to_rgb(hsv)
        rgb_tf = rgb.eval()
    self.assertAllClose(rgb_tf, rgb_np)


class GrayscaleToRGBTest(test_util.TensorFlowTestCase):

  def _RGBToGrayscale(self, images):
    is_batch = True
    if len(images.shape) == 3:
      is_batch = False
      images = np.expand_dims(images, axis=0)
    out_shape = images.shape[0:3] + (1,)
    out = np.zeros(shape=out_shape, dtype=np.uint8)
    for batch in xrange(images.shape[0]):
      for y in xrange(images.shape[1]):
        for x in xrange(images.shape[2]):
          red = images[batch, y, x, 0]
          green = images[batch, y, x, 1]
          blue = images[batch, y, x, 2]
          gray = 0.2989 * red + 0.5870 * green + 0.1140 * blue
          out[batch, y, x, 0] = int(gray)
    if not is_batch:
      out = np.squeeze(out, axis=0)
    return out

  def _TestRGBToGrayscale(self, x_np):
    y_np = self._RGBToGrayscale(x_np)

    with self.test_session():
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.rgb_to_grayscale(x_tf)
      y_tf = y.eval()
      self.assertAllEqual(y_tf, y_np)

  def testBasicRGBToGrayscale(self):
    # 4-D input with batch dimension.
    x_np = np.array([[1, 2, 3], [4, 10, 1]],
                    dtype=np.uint8).reshape([1, 1, 2, 3])
    self._TestRGBToGrayscale(x_np)

    # 3-D input with no batch dimension.
    x_np = np.array([[1, 2, 3], [4, 10, 1]], dtype=np.uint8).reshape([1, 2, 3])
    self._TestRGBToGrayscale(x_np)

  def testBasicGrayscaleToRGB(self):
    # 4-D input with batch dimension.
    x_np = np.array([[1, 2]], dtype=np.uint8).reshape([1, 1, 2, 1])
    y_np = np.array([[1, 1, 1], [2, 2, 2]],
                    dtype=np.uint8).reshape([1, 1, 2, 3])

    with self.test_session():
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.grayscale_to_rgb(x_tf)
      y_tf = y.eval()
      self.assertAllEqual(y_tf, y_np)

    # 3-D input with no batch dimension.
    x_np = np.array([[1, 2]], dtype=np.uint8).reshape([1, 2, 1])
    y_np = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.uint8).reshape([1, 2, 3])

    with self.test_session():
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.grayscale_to_rgb(x_tf)
      y_tf = y.eval()
      self.assertAllEqual(y_tf, y_np)

  def testShapeInference(self):
    # Shape inference works and produces expected output where possible
    rgb_shape = [7, None, 19, 3]
    gray_shape = rgb_shape[:-1] + [1]
    with self.test_session():
      rgb_tf = array_ops.placeholder(dtypes.uint8, shape=rgb_shape)
      gray = image_ops.rgb_to_grayscale(rgb_tf)
      self.assertEqual(gray_shape, gray.get_shape().as_list())

    with self.test_session():
      gray_tf = array_ops.placeholder(dtypes.uint8, shape=gray_shape)
      rgb = image_ops.grayscale_to_rgb(gray_tf)
      self.assertEqual(rgb_shape, rgb.get_shape().as_list())

    # Shape inference does not break for unknown shapes
    with self.test_session():
      rgb_tf_unknown = array_ops.placeholder(dtypes.uint8)
      gray_unknown = image_ops.rgb_to_grayscale(rgb_tf_unknown)
      self.assertFalse(gray_unknown.get_shape())

    with self.test_session():
      gray_tf_unknown = array_ops.placeholder(dtypes.uint8)
      rgb_unknown = image_ops.grayscale_to_rgb(gray_tf_unknown)
      self.assertFalse(rgb_unknown.get_shape())


class AdjustHueTest(test_util.TensorFlowTestCase):

  def testAdjustNegativeHue(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    delta = -0.25
    y_data = [0, 13, 1, 54, 226, 59, 8, 234, 150, 255, 39, 1]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.test_session():
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.adjust_hue(x, delta)
      y_tf = y.eval()
      self.assertAllEqual(y_tf, y_np)

  def testAdjustPositiveHue(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    delta = 0.25
    y_data = [13, 0, 11, 226, 54, 221, 234, 8, 92, 1, 217, 255]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.test_session():
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.adjust_hue(x, delta)
      y_tf = y.eval()
      self.assertAllEqual(y_tf, y_np)


class AdjustSaturationTest(test_util.TensorFlowTestCase):

  def testHalfSaturation(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    saturation_factor = 0.5
    y_data = [6, 9, 13, 140, 180, 226, 135, 121, 234, 172, 255, 128]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.test_session():
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.adjust_saturation(x, saturation_factor)
      y_tf = y.eval()
      self.assertAllEqual(y_tf, y_np)

  def testTwiceSaturation(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    saturation_factor = 2.0
    y_data = [0, 5, 13, 0, 106, 226, 30, 0, 234, 89, 255, 0]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.test_session():
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.adjust_saturation(x, saturation_factor)
      y_tf = y.eval()
      self.assertAllEqual(y_tf, y_np)


class FlipTest(test_util.TensorFlowTestCase):

  def testIdempotentLeftRight(self):
    x_np = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.uint8).reshape([2, 3, 1])
    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        x_tf = constant_op.constant(x_np, shape=x_np.shape)
        y = image_ops.flip_left_right(image_ops.flip_left_right(x_tf))
        y_tf = y.eval()
        self.assertAllEqual(y_tf, x_np)

  def testLeftRight(self):
    x_np = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.uint8).reshape([2, 3, 1])
    y_np = np.array([[3, 2, 1], [3, 2, 1]], dtype=np.uint8).reshape([2, 3, 1])

    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        x_tf = constant_op.constant(x_np, shape=x_np.shape)
        y = image_ops.flip_left_right(x_tf)
        y_tf = y.eval()
        self.assertAllEqual(y_tf, y_np)

  def testIdempotentUpDown(self):
    x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8).reshape([2, 3, 1])

    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        x_tf = constant_op.constant(x_np, shape=x_np.shape)
        y = image_ops.flip_up_down(image_ops.flip_up_down(x_tf))
        y_tf = y.eval()
        self.assertAllEqual(y_tf, x_np)

  def testUpDown(self):
    x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8).reshape([2, 3, 1])
    y_np = np.array([[4, 5, 6], [1, 2, 3]], dtype=np.uint8).reshape([2, 3, 1])

    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        x_tf = constant_op.constant(x_np, shape=x_np.shape)
        y = image_ops.flip_up_down(x_tf)
        y_tf = y.eval()
        self.assertAllEqual(y_tf, y_np)

  def testIdempotentTranspose(self):
    x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8).reshape([2, 3, 1])

    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        x_tf = constant_op.constant(x_np, shape=x_np.shape)
        y = image_ops.transpose_image(image_ops.transpose_image(x_tf))
        y_tf = y.eval()
        self.assertAllEqual(y_tf, x_np)

  def testTranspose(self):
    x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8).reshape([2, 3, 1])
    y_np = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.uint8).reshape([3, 2, 1])

    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        x_tf = constant_op.constant(x_np, shape=x_np.shape)
        y = image_ops.transpose_image(x_tf)
        y_tf = y.eval()
        self.assertAllEqual(y_tf, y_np)

  def testPartialShapes(self):
    p_unknown_rank = array_ops.placeholder(dtypes.uint8)
    p_unknown_dims = array_ops.placeholder(dtypes.uint8,
                                           shape=[None, None, None])
    p_unknown_width = array_ops.placeholder(dtypes.uint8, shape=[64, None, 3])

    p_wrong_rank = array_ops.placeholder(dtypes.uint8, shape=[None, None])
    p_zero_dim = array_ops.placeholder(dtypes.uint8, shape=[64, 0, 3])

    for op in [image_ops.flip_left_right,
               image_ops.flip_up_down,
               image_ops.random_flip_left_right,
               image_ops.random_flip_up_down,
               image_ops.transpose_image]:
      transformed_unknown_rank = op(p_unknown_rank)
      self.assertEqual(3, transformed_unknown_rank.get_shape().ndims)
      transformed_unknown_dims = op(p_unknown_dims)
      self.assertEqual(3, transformed_unknown_dims.get_shape().ndims)
      transformed_unknown_width = op(p_unknown_width)
      self.assertEqual(3, transformed_unknown_width.get_shape().ndims)

      with self.assertRaisesRegexp(ValueError, 'must be three-dimensional'):
        op(p_wrong_rank)
      with self.assertRaisesRegexp(ValueError, 'must be > 0'):
        op(p_zero_dim)


class RandomFlipTest(test_util.TensorFlowTestCase):

  def testRandomLeftRight(self):
    x_np = np.array([0, 1], dtype=np.uint8).reshape([1, 2, 1])
    num_iterations = 500

    hist = [0, 0]
    with self.test_session():
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.random_flip_left_right(x_tf)
      for _ in xrange(num_iterations):
        y_np = y.eval().flatten()[0]
        hist[y_np] += 1

    # Ensure that each entry is observed within 4 standard deviations.
    four_stddev = 4.0 * np.sqrt(num_iterations / 2.0)
    self.assertAllClose(hist, [num_iterations / 2.0] * 2, atol=four_stddev)

  def testRandomUpDown(self):
    x_np = np.array([0, 1], dtype=np.uint8).reshape([2, 1, 1])
    num_iterations = 500

    hist = [0, 0]
    with self.test_session():
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.random_flip_up_down(x_tf)
      for _ in xrange(num_iterations):
        y_np = y.eval().flatten()[0]
        hist[y_np] += 1

    # Ensure that each entry is observed within 4 standard deviations.
    four_stddev = 4.0 * np.sqrt(num_iterations / 2.0)
    self.assertAllClose(hist, [num_iterations / 2.0] * 2, atol=four_stddev)


class AdjustContrastTest(test_util.TensorFlowTestCase):

  def _testContrast(self, x_np, y_np, contrast_factor):
    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu):
        x = constant_op.constant(x_np, shape=x_np.shape)
        y = image_ops.adjust_contrast(x, contrast_factor)
        y_tf = y.eval()
        self.assertAllClose(y_tf, y_np, 1e-6)

  def testDoubleContrastUint8(self):
    x_shape = [1, 2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    y_data = [0, 0, 0, 62, 169, 255, 28, 0, 255, 135, 255, 0]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    self._testContrast(x_np, y_np, contrast_factor=2.0)

  def testDoubleContrastFloat(self):
    x_shape = [1, 2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.float).reshape(x_shape) / 255.

    y_data = [-45.25, -90.75, -92.5, 62.75, 169.25, 333.5, 28.75, -84.75, 349.5,
              134.75, 409.25, -116.5]
    y_np = np.array(y_data, dtype=np.float).reshape(x_shape) / 255.

    self._testContrast(x_np, y_np, contrast_factor=2.0)

  def testHalfContrastUint8(self):
    x_shape = [1, 2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    y_data = [22, 52, 65, 49, 118, 172, 41, 54, 176, 67, 178, 59]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    self._testContrast(x_np, y_np, contrast_factor=0.5)

  def testBatchDoubleContrast(self):
    x_shape = [2, 1, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    y_data = [0, 0, 0, 81, 200, 255, 10, 0, 255, 116, 255, 0]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    self._testContrast(x_np, y_np, contrast_factor=2.0)


class AdjustBrightnessTest(test_util.TensorFlowTestCase):

  def _testBrightness(self, x_np, y_np, delta):
    with self.test_session():
      x = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.adjust_brightness(x, delta)
      y_tf = y.eval()
      self.assertAllClose(y_tf, y_np, 1e-6)

  def testPositiveDeltaUint8(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    y_data = [10, 15, 23, 64, 145, 236, 47, 18, 244, 100, 255, 11]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    self._testBrightness(x_np, y_np, delta=10. / 255.)

  def testPositiveDeltaFloat(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.float32).reshape(x_shape) / 255.

    y_data = [10, 15, 23, 64, 145, 236, 47, 18, 244, 100, 265, 11]
    y_np = np.array(y_data, dtype=np.float32).reshape(x_shape) / 255.

    self._testBrightness(x_np, y_np, delta=10. / 255.)

  def testNegativeDelta(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    y_data = [0, 0, 3, 44, 125, 216, 27, 0, 224, 80, 245, 0]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    self._testBrightness(x_np, y_np, delta=-10. / 255.)


class PerImageWhiteningTest(test_util.TensorFlowTestCase):

  def _NumpyPerImageWhitening(self, x):
    num_pixels = np.prod(x.shape)
    x2 = np.square(x).astype(np.float32)
    mn = np.mean(x)
    vr = np.mean(x2) - (mn * mn)
    stddev = max(math.sqrt(vr), 1.0 / math.sqrt(num_pixels))

    y = x.astype(np.float32)
    y -= mn
    y /= stddev
    return y

  def testBasic(self):
    x_shape = [13, 9, 3]
    x_np = np.arange(0, np.prod(x_shape), dtype=np.int32).reshape(x_shape)
    y_np = self._NumpyPerImageWhitening(x_np)

    with self.test_session():
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.per_image_whitening(x)
      y_tf = y.eval()
      self.assertAllClose(y_tf, y_np, atol=1e-4)

  def testUniformImage(self):
    im_np = np.ones([19, 19, 3]).astype(np.float32) * 249
    im = constant_op.constant(im_np)
    whiten = image_ops.per_image_whitening(im)
    with self.test_session():
      whiten_np = whiten.eval()
      self.assertFalse(np.any(np.isnan(whiten_np)))


class CropToBoundingBoxTest(test_util.TensorFlowTestCase):

  def testNoOp(self):
    x_shape = [13, 9, 3]
    x_np = np.ones(x_shape, dtype=np.float32)

    with self.test_session():
      x = constant_op.constant(x_np, shape=x_shape)
      target_height = x_shape[0]
      target_width = x_shape[1]
      y = image_ops.crop_to_bounding_box(x, 0, 0, target_height, target_width)
      y_tf = y.eval()
      self.assertAllEqual(y_tf, x_np)

  def testCropping(self):
    x_np = np.arange(0, 30, dtype=np.int32).reshape([6, 5, 1])

    offset_height = 1
    after_height = 2

    offset_width = 0
    after_width = 3

    target_height = x_np.shape[0] - offset_height - after_height
    target_width = x_np.shape[1] - offset_width - after_width

    y_np = x_np[offset_height:offset_height + target_height,
                offset_width:offset_width + target_width, :]

    with self.test_session():
      x = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.crop_to_bounding_box(x, offset_height, offset_width,
                                         target_height, target_width)
      y_tf = y.eval()
      self.assertAllEqual(y_tf.flatten(), y_np.flatten())


class CentralCropTest(test_util.TensorFlowTestCase):

  def testNoOp(self):
    x_shape = [13, 9, 3]
    x_np = np.ones(x_shape, dtype=np.float32)
    with self.test_session():
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.central_crop(x, 1.0)
      y_tf = y.eval()
      self.assertAllEqual(y_tf, x_np)
      self.assertEqual(y.op.name, x.op.name)

  def testCropping(self):
    x_shape = [4, 8, 1]
    x_np = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8],
                     [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]],
                    dtype=np.int32).reshape(x_shape)
    y_np = np.array([[3, 4, 5, 6], [3, 4, 5, 6]]).reshape([2, 4, 1])
    with self.test_session():
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.central_crop(x, 0.5)
      y_tf = y.eval()
      self.assertAllEqual(y_tf, y_np)

  def testError(self):
    x_shape = [13, 9, 3]
    x_np = np.ones(x_shape, dtype=np.float32)
    with self.test_session():
      x = constant_op.constant(x_np, shape=x_shape)
      with self.assertRaises(ValueError):
        _ = image_ops.central_crop(x, 0.0)
      with self.assertRaises(ValueError):
        _ = image_ops.central_crop(x, 1.01)


class PadToBoundingBoxTest(test_util.TensorFlowTestCase):

  def testNoOp(self):
    x_shape = [13, 9, 3]
    x_np = np.ones(x_shape, dtype=np.float32)

    target_height = x_shape[0]
    target_width = x_shape[1]

    with self.test_session():
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.pad_to_bounding_box(x, 0, 0, target_height, target_width)
      y_tf = y.eval()
      self.assertAllEqual(y_tf, x_np)

  def testPadding(self):
    x_shape = [3, 4, 1]
    x_np = np.ones(x_shape, dtype=np.float32)

    offset_height = 2
    after_height = 3

    offset_width = 1
    after_width = 4

    target_height = x_shape[0] + offset_height + after_height
    target_width = x_shape[1] + offset_width + after_width

    # Note the padding are along batch, height, width and depth.
    paddings = ((offset_height, after_height),
                (offset_width, after_width),
                (0, 0))

    y_np = np.pad(x_np, paddings, 'constant')

    with self.test_session():
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.pad_to_bounding_box(x, offset_height, offset_width,
                                        target_height, target_width)
      y_tf = y.eval()
      self.assertAllEqual(y_tf, y_np)


class SelectDistortedCropBoxTest(test_util.TensorFlowTestCase):

  def _testSampleDistortedBoundingBox(self, image, bounding_box,
                                      min_object_covered, aspect_ratio_range,
                                      area_range):
    original_area = float(np.prod(image.shape))
    bounding_box_area = float((bounding_box[3] - bounding_box[1]) *
                              (bounding_box[2] - bounding_box[0]))

    image_size_np = np.array(image.shape, dtype=np.int32)
    bounding_box_np = (np.array(bounding_box, dtype=np.float32)
                       .reshape([1, 1, 4]))

    aspect_ratios = []
    area_ratios = []

    fraction_object_covered = []

    num_iter = 1000
    with self.test_session():
      image_tf = constant_op.constant(image,
                                      shape=image.shape)
      image_size_tf = constant_op.constant(image_size_np,
                                           shape=image_size_np.shape)
      bounding_box_tf = constant_op.constant(bounding_box_np,
                                             dtype=dtypes.float32,
                                             shape=bounding_box_np.shape)
      begin, end, _ = image_ops.sample_distorted_bounding_box(
          image_size=image_size_tf,
          bounding_boxes=bounding_box_tf,
          min_object_covered=min_object_covered,
          aspect_ratio_range=aspect_ratio_range,
          area_range=area_range)
      y = array_ops.slice(image_tf, begin, end)

      for _ in xrange(num_iter):
        y_tf = y.eval()
        crop_height = y_tf.shape[0]
        crop_width = y_tf.shape[1]
        aspect_ratio = float(crop_width) / float(crop_height)
        area = float(crop_width * crop_height)

        aspect_ratios.append(aspect_ratio)
        area_ratios.append(area / original_area)
        fraction_object_covered.append(float(np.sum(y_tf)) / bounding_box_area)

    # Ensure that each entry is observed within 3 standard deviations.
    # num_bins = 10
    # aspect_ratio_hist, _ = np.histogram(aspect_ratios,
    #                                     bins=num_bins,
    #                                     range=aspect_ratio_range)
    # mean = np.mean(aspect_ratio_hist)
    # stddev = np.sqrt(mean)
    # TODO(wicke, shlens, dga): Restore this test so that it is no longer flaky.
    # TODO(irving): Since the rejection probability is not independent of the
    # aspect ratio, the aspect_ratio random value is not exactly uniformly
    # distributed in [min_aspect_ratio, max_aspect_ratio).  This test should be
    # fixed to reflect the true statistical property, then tightened to enforce
    # a stricter bound.  Or, ideally, the sample_distorted_bounding_box Op
    # be fixed to not use rejection sampling and generate correctly uniform
    # aspect ratios.
    # self.assertAllClose(aspect_ratio_hist,
    #                     [mean] * num_bins, atol=3.6 * stddev)

    # The resulting crop will not be uniformly distributed in area. In practice,
    # we find that the area skews towards the small sizes. Instead, we perform
    # a weaker test to ensure that the area ratios are merely within the
    # specified bounds.
    self.assertLessEqual(max(area_ratios), area_range[1])
    self.assertGreaterEqual(min(area_ratios), area_range[0])

    # For reference, here is what the distribution of area ratios look like.
    area_ratio_hist, _ = np.histogram(area_ratios, bins=10, range=area_range)
    print('area_ratio_hist ', area_ratio_hist)

    # Ensure that fraction_object_covered is satisfied.
    # TODO(wicke, shlens, dga): Restore this test so that it is no longer flaky.
    # self.assertGreaterEqual(min(fraction_object_covered), min_object_covered)

  def testWholeImageBoundingBox(self):
    height = 40
    width = 50
    image_size = [height, width, 1]
    bounding_box = [0.0, 0.0, 1.0, 1.0]
    image = np.arange(0, np.prod(image_size),
                      dtype=np.int32).reshape(image_size)
    self._testSampleDistortedBoundingBox(image,
                                         bounding_box,
                                         min_object_covered=0.1,
                                         aspect_ratio_range=(0.75, 1.33),
                                         area_range=(0.05, 1.0))

  def testWithBoundingBox(self):
    height = 40
    width = 50
    x_shape = [height, width, 1]
    image = np.zeros(x_shape, dtype=np.int32)

    # Create an object with 1's in a region with area A and require that
    # the total pixel values >= 0.1 * A.
    min_object_covered = 0.1

    xmin = 2
    ymin = 3
    xmax = 12
    ymax = 13
    for x in np.arange(xmin, xmax + 1, 1):
      for y in np.arange(ymin, ymax + 1, 1):
        image[x, y] = 1

    # Bounding box is specified as (ymin, xmin, ymax, xmax) in
    # relative coordinates.
    bounding_box = (float(ymin) / height, float(xmin) / width,
                    float(ymax) / height, float(xmax) / width)

    self._testSampleDistortedBoundingBox(image,
                                         bounding_box=bounding_box,
                                         min_object_covered=min_object_covered,
                                         aspect_ratio_range=(0.75, 1.33),
                                         area_range=(0.05, 1.0))

  def testSampleDistortedBoundingBoxShape(self):
    with self.test_session():
      image_size = constant_op.constant([40, 50, 1],
                                        shape=[3],
                                        dtype=dtypes.int32)
      bounding_box = constant_op.constant([0.0, 0.0, 1.0, 1.0],
                                          shape=[4],
                                          dtype=dtypes.float32,)
      begin, end, bbox_for_drawing = image_ops.sample_distorted_bounding_box(
          image_size=image_size,
          bounding_boxes=bounding_box,
          min_object_covered=0.1,
          aspect_ratio_range=(0.75, 1.33),
          area_range=(0.05, 1.0))

      # Test that the shapes are correct.
      self.assertAllEqual([3], begin.get_shape().as_list())
      self.assertAllEqual([3], end.get_shape().as_list())
      self.assertAllEqual([1, 1, 4], bbox_for_drawing.get_shape().as_list())


class ResizeImagesTest(test_util.TensorFlowTestCase):

  OPTIONS = [image_ops.ResizeMethod.BILINEAR,
             image_ops.ResizeMethod.NEAREST_NEIGHBOR,
             image_ops.ResizeMethod.BICUBIC,
             image_ops.ResizeMethod.AREA]

  TYPES = [np.uint8, np.int8, np.int16, np.int32, np.int64,
           np.float32, np.float64]

  def availableGPUModes(self, opt, nptype):
    if opt == image_ops.ResizeMethod.NEAREST_NEIGHBOR \
            and nptype in [np.float32, np.float64]:
      return [True, False]
    else:
      return [False]

  def testNoOp(self):
    img_shape = [1, 6, 4, 1]
    single_shape = [6, 4, 1]
    # This test is also conducted with int8, so 127 is the maximum
    # value that can be used.
    data = [127, 127, 64, 64,
            127, 127, 64, 64,
            64, 64, 127, 127,
            64, 64, 127, 127,
            50, 50, 100, 100,
            50, 50, 100, 100]
    target_height = 6
    target_width = 4

    for nptype in self.TYPES:
      img_np = np.array(data, dtype=nptype).reshape(img_shape)

      for opt in self.OPTIONS:
        for use_gpu in self.availableGPUModes(opt, nptype):
          with self.test_session(use_gpu=use_gpu) as sess:
            image = constant_op.constant(img_np, shape=img_shape)
            y = image_ops.resize_images(image, target_height, target_width, opt)
            yshape = array_ops.shape(y)
            resized, newshape = sess.run([y, yshape])
            self.assertAllEqual(img_shape, newshape)
            self.assertAllClose(resized, img_np, atol=1e-5)

      # Resizing with a single image must leave the shape unchanged also.
      with self.test_session():
        img_single = img_np.reshape(single_shape)
        image = constant_op.constant(img_single, shape=single_shape)
        y = image_ops.resize_images(image, target_height, target_width,
                                    self.OPTIONS[0])
        yshape = array_ops.shape(y)
        newshape = yshape.eval()
        self.assertAllEqual(single_shape, newshape)

  def testTensorArguments(self):
    img_shape = [1, 6, 4, 1]
    single_shape = [6, 4, 1]
    # This test is also conducted with int8, so 127 is the maximum
    # value that can be used.
    data = [127, 127, 64, 64,
            127, 127, 64, 64,
            64, 64, 127, 127,
            64, 64, 127, 127,
            50, 50, 100, 100,
            50, 50, 100, 100]
    target_height = array_ops.placeholder(dtypes.int32)
    target_width = array_ops.placeholder(dtypes.int32)

    img_np = np.array(data, dtype=np.uint8).reshape(img_shape)

    for opt in self.OPTIONS:
      with self.test_session() as sess:
        image = constant_op.constant(img_np, shape=img_shape)
        y = image_ops.resize_images(image, target_height, target_width, opt)
        yshape = array_ops.shape(y)
        resized, newshape = sess.run([y, yshape], {target_height: 6,
                                                   target_width: 4})
        self.assertAllEqual(img_shape, newshape)
        self.assertAllClose(resized, img_np, atol=1e-5)

    # Resizing with a single image must leave the shape unchanged also.
    with self.test_session():
      img_single = img_np.reshape(single_shape)
      image = constant_op.constant(img_single, shape=single_shape)
      y = image_ops.resize_images(image, target_height, target_width,
                                  self.OPTIONS[0])
      yshape = array_ops.shape(y)
      newshape = yshape.eval(feed_dict={target_height: 6, target_width: 4})
      self.assertAllEqual(single_shape, newshape)

    # Incorrect shape.
    with self.assertRaises(ValueError):
      _ = image_ops.resize_images(
          image, [12, 32], 4, image_ops.ResizeMethod.BILINEAR)
    with self.assertRaises(ValueError):
      _ = image_ops.resize_images(
          image, 6, [12, 32], image_ops.ResizeMethod.BILINEAR)

    # Incorrect dtypes.
    with self.assertRaises(ValueError):
      _ = image_ops.resize_images(
          image, 6.0, 4, image_ops.ResizeMethod.BILINEAR)
    with self.assertRaises(ValueError):
      _ = image_ops.resize_images(
          image, 6, 4.0, image_ops.ResizeMethod.BILINEAR)

  def testResizeDown(self):
    # This test is also conducted with int8, so 127 is the maximum
    # value that can be used.
    data = [127, 127, 64, 64,
            127, 127, 64, 64,
            64, 64, 127, 127,
            64, 64, 127, 127,
            50, 50, 100, 100,
            50, 50, 100, 100]
    expected_data = [127, 64,
                     64, 127,
                     50, 100]
    target_height = 3
    target_width = 2

    # Test out 3-D and 4-D image shapes.
    img_shapes = [[1, 6, 4, 1], [6, 4, 1]]
    target_shapes = [[1, target_height, target_width, 1],
                     [target_height, target_width, 1]]

    for target_shape, img_shape in zip(target_shapes, img_shapes):

      for nptype in self.TYPES:
        img_np = np.array(data, dtype=nptype).reshape(img_shape)

        for opt in self.OPTIONS:
          for use_gpu in self.availableGPUModes(opt, nptype):
            with self.test_session(use_gpu=use_gpu):
              image = constant_op.constant(img_np, shape=img_shape)
              y = image_ops.resize_images(image, target_height, target_width, opt)
              expected = np.array(expected_data).reshape(target_shape)
              resized = y.eval()
              self.assertAllClose(resized, expected, atol=1e-5)

  def testResizeUp(self):
    img_shape = [1, 3, 2, 1]
    data = [64, 32,
            32, 64,
            50, 100]
    target_height = 6
    target_width = 4
    expected_data = {}
    expected_data[image_ops.ResizeMethod.BILINEAR] = [
        64.0, 48.0, 32.0, 32.0,
        48.0, 48.0, 48.0, 48.0,
        32.0, 48.0, 64.0, 64.0,
        41.0, 61.5, 82.0, 82.0,
        50.0, 75.0, 100.0, 100.0,
        50.0, 75.0, 100.0, 100.0]
    expected_data[image_ops.ResizeMethod.NEAREST_NEIGHBOR] = [
        64.0, 64.0, 32.0, 32.0,
        64.0, 64.0, 32.0, 32.0,
        32.0, 32.0, 64.0, 64.0,
        32.0, 32.0, 64.0, 64.0,
        50.0, 50.0, 100.0, 100.0,
        50.0, 50.0, 100.0, 100.0]
    expected_data[image_ops.ResizeMethod.AREA] = [
        64.0, 64.0, 32.0, 32.0,
        64.0, 64.0, 32.0, 32.0,
        32.0, 32.0, 64.0, 64.0,
        32.0, 32.0, 64.0, 64.0,
        50.0, 50.0, 100.0, 100.0,
        50.0, 50.0, 100.0, 100.0]

    for nptype in self.TYPES:
      for opt in [
          image_ops.ResizeMethod.BILINEAR,
          image_ops.ResizeMethod.NEAREST_NEIGHBOR,
          image_ops.ResizeMethod.AREA]:
        for use_gpu in self.availableGPUModes(opt, nptype):
          with self.test_session(use_gpu=use_gpu):
            img_np = np.array(data, dtype=nptype).reshape(img_shape)
            image = constant_op.constant(img_np, shape=img_shape)
            y = image_ops.resize_images(image, target_height, target_width, opt)
            resized = y.eval()
            expected = np.array(expected_data[opt]).reshape(
                [1, target_height, target_width, 1])
            self.assertAllClose(resized, expected, atol=1e-05)

  def testResizeUpBicubic(self):
    img_shape = [1, 6, 6, 1]
    data = [128, 128, 64, 64, 128, 128, 64, 64,
            64, 64, 128, 128, 64, 64, 128, 128,
            50, 50, 100, 100, 50, 50, 100, 100,
            50, 50, 100, 100, 50, 50, 100, 100,
            50, 50, 100, 100]
    img_np = np.array(data, dtype=np.uint8).reshape(img_shape)

    target_height = 8
    target_width = 8
    expected_data = [128, 135, 96, 55, 64, 114, 134, 128,
                     78, 81, 68, 52, 57, 118, 144, 136,
                     55, 49, 79, 109, 103, 89, 83, 84,
                     74, 70, 95, 122, 115, 69, 49, 55,
                     100, 105, 75, 43, 50, 89, 105, 100,
                     57, 54, 74, 96, 91, 65, 55, 58,
                     70, 69, 75, 81, 80, 72, 69, 70,
                     105, 112, 75, 36, 45, 92, 111, 105]

    with self.test_session():
      image = constant_op.constant(img_np, shape=img_shape)
      y = image_ops.resize_images(image, target_height, target_width,
                                  image_ops.ResizeMethod.BICUBIC)
      resized = y.eval()
      expected = np.array(expected_data).reshape(
          [1, target_height, target_width, 1])
      self.assertAllClose(resized, expected, atol=1)

  def testResizeDownArea(self):
    img_shape = [1, 6, 6, 1]
    data = [128, 64, 32, 16, 8, 4,
            4, 8, 16, 32, 64, 128,
            128, 64, 32, 16, 8, 4,
            5, 10, 15, 20, 25, 30,
            30, 25, 20, 15, 10, 5,
            5, 10, 15, 20, 25, 30]
    img_np = np.array(data, dtype=np.uint8).reshape(img_shape)

    target_height = 4
    target_width = 4
    expected_data = [73, 33, 23, 39,
                     73, 33, 23, 39,
                     14, 16, 19, 21,
                     14, 16, 19, 21]

    with self.test_session():
      image = constant_op.constant(img_np, shape=img_shape)
      y = image_ops.resize_images(image, target_height, target_width,
                                  image_ops.ResizeMethod.AREA)
      expected = np.array(expected_data).reshape(
          [1, target_height, target_width, 1])
      resized = y.eval()
      self.assertAllClose(resized, expected, atol=1)


  def testCompareNearestNeighbor(self):
    input_shape = [1, 5, 6, 3]
    target_height = 8
    target_width = 12
    for nptype in [np.float32, np.float64]:
      for align_corners in [True, False]:
        img_np = np.arange(0, np.prod(input_shape), dtype=nptype).reshape(input_shape)
        with self.test_session(use_gpu=True):
          image = constant_op.constant(img_np, shape=input_shape)
          out_op = image_ops.resize_images(image, target_height, target_width,
                                           image_ops.ResizeMethod.NEAREST_NEIGHBOR,
                                           align_corners=align_corners)
          gpu_val = out_op.eval()
        with self.test_session(use_gpu=False):
          image = constant_op.constant(img_np, shape=input_shape)
          out_op = image_ops.resize_images(image, target_height, target_width,
                                           image_ops.ResizeMethod.NEAREST_NEIGHBOR,
                                           align_corners=align_corners)
          cpu_val = out_op.eval()
        self.assertAllClose(cpu_val, gpu_val, rtol=1e-5, atol=1e-5)


class ResizeImageWithCropOrPadTest(test_util.TensorFlowTestCase):

  def _ResizeImageWithCropOrPad(self, original, original_shape,
                                expected, expected_shape):
    x_np = np.array(original, dtype=np.uint8).reshape(original_shape)
    y_np = np.array(expected).reshape(expected_shape)

    target_height = expected_shape[0]
    target_width = expected_shape[1]

    with self.test_session():
      image = constant_op.constant(x_np, shape=original_shape)
      y = image_ops.resize_image_with_crop_or_pad(image,
                                                  target_height,
                                                  target_width)
      resized = y.eval()
      self.assertAllClose(resized, y_np, atol=1e-5)

  def testBasic(self):
    # Basic no-op.
    original = [1, 2, 3, 4,
                5, 6, 7, 8]
    self._ResizeImageWithCropOrPad(original, [2, 4, 1],
                                   original, [2, 4, 1])

  def testPad(self):
    # Pad even along col.
    original = [1, 2, 3, 4, 5, 6, 7, 8]
    expected = [0, 1, 2, 3, 4, 0,
                0, 5, 6, 7, 8, 0]
    self._ResizeImageWithCropOrPad(original, [2, 4, 1],
                                   expected, [2, 6, 1])
    # Pad odd along col.
    original = [1, 2, 3, 4,
                5, 6, 7, 8]
    expected = [0, 1, 2, 3, 4, 0, 0,
                0, 5, 6, 7, 8, 0, 0]
    self._ResizeImageWithCropOrPad(original, [2, 4, 1],
                                   expected, [2, 7, 1])

    # Pad even along row.
    original = [1, 2, 3, 4,
                5, 6, 7, 8]
    expected = [0, 0, 0, 0,
                1, 2, 3, 4,
                5, 6, 7, 8,
                0, 0, 0, 0]
    self._ResizeImageWithCropOrPad(original, [2, 4, 1],
                                   expected, [4, 4, 1])
    # Pad odd along row.
    original = [1, 2, 3, 4,
                5, 6, 7, 8]
    expected = [0, 0, 0, 0,
                1, 2, 3, 4,
                5, 6, 7, 8,
                0, 0, 0, 0,
                0, 0, 0, 0]
    self._ResizeImageWithCropOrPad(original, [2, 4, 1],
                                   expected, [5, 4, 1])

  def testCrop(self):
    # Crop even along col.
    original = [1, 2, 3, 4,
                5, 6, 7, 8]
    expected = [2, 3,
                6, 7]
    self._ResizeImageWithCropOrPad(original, [2, 4, 1],
                                   expected, [2, 2, 1])
    # Crop odd along col.

    original = [1, 2, 3, 4, 5, 6,
                7, 8, 9, 10, 11, 12]
    expected = [2, 3, 4,
                8, 9, 10]
    self._ResizeImageWithCropOrPad(original, [2, 6, 1],
                                   expected, [2, 3, 1])

    # Crop even along row.
    original = [1, 2,
                3, 4,
                5, 6,
                7, 8]
    expected = [3, 4,
                5, 6]
    self._ResizeImageWithCropOrPad(original, [4, 2, 1],
                                   expected, [2, 2, 1])

    # Crop odd along row.
    original = [1, 2,
                3, 4,
                5, 6,
                7, 8,
                9, 10,
                11, 12,
                13, 14,
                15, 16]
    expected = [3, 4,
                5, 6,
                7, 8,
                9, 10,
                11, 12]
    self._ResizeImageWithCropOrPad(original, [8, 2, 1],
                                   expected, [5, 2, 1])

  def testCropAndPad(self):
    # Pad along row but crop along col.
    original = [1, 2, 3, 4,
                5, 6, 7, 8]
    expected = [0, 0,
                2, 3,
                6, 7,
                0, 0]
    self._ResizeImageWithCropOrPad(original, [2, 4, 1],
                                   expected, [4, 2, 1])

    # Crop along row but pad along col.
    original = [1, 2,
                3, 4,
                5, 6,
                7, 8]
    expected = [0, 3, 4, 0,
                0, 5, 6, 0]
    self._ResizeImageWithCropOrPad(original, [4, 2, 1],
                                   expected, [2, 4, 1])


def _SimpleColorRamp():
  """Build a simple color ramp RGB image."""
  w, h = 256, 200
  i = np.arange(h)[:, None]
  j = np.arange(w)
  image = np.empty((h, w, 3), dtype=np.uint8)
  image[:, :, 0] = i
  image[:, :, 1] = j
  image[:, :, 2] = (i + j) >> 1
  return image


class JpegTest(test_util.TensorFlowTestCase):

  # TODO(irving): Add self.assertAverageLess or similar to test_util
  def averageError(self, image0, image1):
    self.assertEqual(image0.shape, image1.shape)
    image0 = image0.astype(int)  # Avoid overflow
    return np.abs(image0 - image1).sum() / np.prod(image0.shape)

  def testExisting(self):
    # Read a real jpeg and verify shape
    path = ('tensorflow/core/lib/jpeg/testdata/'
            'jpeg_merge_test1.jpg')
    with self.test_session() as sess:
      jpeg0 = io_ops.read_file(path)
      image0 = image_ops.decode_jpeg(jpeg0)
      image1 = image_ops.decode_jpeg(image_ops.encode_jpeg(image0))
      jpeg0, image0, image1 = sess.run([jpeg0, image0, image1])
      self.assertEqual(len(jpeg0), 3771)
      self.assertEqual(image0.shape, (256, 128, 3))
      self.assertLess(self.averageError(image0, image1), 0.8)

  def testCmyk(self):
    # Confirm that CMYK reads in as RGB
    base = 'tensorflow/core/lib/jpeg/testdata'
    rgb_path = os.path.join(base, 'jpeg_merge_test1.jpg')
    cmyk_path = os.path.join(base, 'jpeg_merge_test1_cmyk.jpg')
    shape = 256, 128, 3
    for channels in 3, 0:
      with self.test_session() as sess:
        rgb = image_ops.decode_jpeg(io_ops.read_file(rgb_path),
                                    channels=channels)
        cmyk = image_ops.decode_jpeg(io_ops.read_file(cmyk_path),
                                     channels=channels)
        rgb, cmyk = sess.run([rgb, cmyk])
        self.assertEqual(rgb.shape, shape)
        self.assertEqual(cmyk.shape, shape)
        error = self.averageError(rgb, cmyk)
        self.assertLess(error, 4)

  def testSynthetic(self):
    with self.test_session() as sess:
      # Encode it, then decode it, then encode it
      image0 = constant_op.constant(_SimpleColorRamp())
      jpeg0 = image_ops.encode_jpeg(image0)
      image1 = image_ops.decode_jpeg(jpeg0)
      image2 = image_ops.decode_jpeg(image_ops.encode_jpeg(image1))
      jpeg0, image0, image1, image2 = sess.run([jpeg0, image0, image1, image2])

      # The decoded-encoded image should be similar to the input
      self.assertLess(self.averageError(image0, image1), 0.6)

      # We should be very close to a fixpoint
      self.assertLess(self.averageError(image1, image2), 0.02)

      # Smooth ramps compress well (input size is 153600)
      self.assertGreaterEqual(len(jpeg0), 5000)
      self.assertLessEqual(len(jpeg0), 6000)

  def testShape(self):
    with self.test_session() as sess:
      jpeg = constant_op.constant('nonsense')
      for channels in 0, 1, 3:
        image = image_ops.decode_jpeg(jpeg, channels=channels)
        self.assertEqual(image.get_shape().as_list(),
                         [None, None, channels or None])


class PngTest(test_util.TensorFlowTestCase):

  def testExisting(self):
    # Read some real PNGs, converting to different channel numbers
    prefix = 'tensorflow/core/lib/png/testdata/'
    inputs = (1, 'lena_gray.png'), (4, 'lena_rgba.png')
    for channels_in, filename in inputs:
      for channels in 0, 1, 3, 4:
        with self.test_session() as sess:
          png0 = io_ops.read_file(prefix + filename)
          image0 = image_ops.decode_png(png0, channels=channels)
          png0, image0 = sess.run([png0, image0])
          self.assertEqual(image0.shape, (26, 51, channels or channels_in))
          if channels == channels_in:
            image1 = image_ops.decode_png(image_ops.encode_png(image0))
            self.assertAllEqual(image0, image1.eval())

  def testSynthetic(self):
    with self.test_session() as sess:
      # Encode it, then decode it
      image0 = constant_op.constant(_SimpleColorRamp())
      png0 = image_ops.encode_png(image0, compression=7)
      image1 = image_ops.decode_png(png0)
      png0, image0, image1 = sess.run([png0, image0, image1])

      # PNG is lossless
      self.assertAllEqual(image0, image1)

      # Smooth ramps compress well, but not too well
      self.assertGreaterEqual(len(png0), 400)
      self.assertLessEqual(len(png0), 750)

  def testSyntheticUint16(self):
    with self.test_session() as sess:
      # Encode it, then decode it
      image0 = constant_op.constant(_SimpleColorRamp(), dtype=dtypes.uint16)
      png0 = image_ops.encode_png(image0, compression=7)
      image1 = image_ops.decode_png(png0, dtype=dtypes.uint16)
      png0, image0, image1 = sess.run([png0, image0, image1])

      # PNG is lossless
      self.assertAllEqual(image0, image1)

      # Smooth ramps compress well, but not too well
      self.assertGreaterEqual(len(png0), 800)
      self.assertLessEqual(len(png0), 1500)

  def testSyntheticTwoChannel(self):
    with self.test_session() as sess:
      # Strip the b channel from an rgb image to get a two-channel image.
      gray_alpha = _SimpleColorRamp()[:, :, 0:2]
      image0 = constant_op.constant(gray_alpha)
      png0 = image_ops.encode_png(image0, compression=7)
      image1 = image_ops.decode_png(png0)
      png0, image0, image1 = sess.run([png0, image0, image1])
      self.assertEqual(2, image0.shape[-1])
      self.assertAllEqual(image0, image1)

  def testSyntheticTwoChannelUint16(self):
    with self.test_session() as sess:
      # Strip the b channel from an rgb image to get a two-channel image.
      gray_alpha = _SimpleColorRamp()[:, :, 0:2]
      image0 = constant_op.constant(gray_alpha, dtype=dtypes.uint16)
      png0 = image_ops.encode_png(image0, compression=7)
      image1 = image_ops.decode_png(png0, dtype=dtypes.uint16)
      png0, image0, image1 = sess.run([png0, image0, image1])
      self.assertEqual(2, image0.shape[-1])
      self.assertAllEqual(image0, image1)

  def testShape(self):
    with self.test_session():
      png = constant_op.constant('nonsense')
      for channels in 0, 1, 3:
        image = image_ops.decode_png(png, channels=channels)
        self.assertEqual(image.get_shape().as_list(),
                         [None, None, channels or None])


class ConvertImageTest(test_util.TensorFlowTestCase):

  def _convert(self, original, original_dtype, output_dtype, expected):
    x_np = np.array(original, dtype=original_dtype.as_numpy_dtype())
    y_np = np.array(expected, dtype=output_dtype.as_numpy_dtype())

    with self.test_session():
      image = constant_op.constant(x_np)
      y = image_ops.convert_image_dtype(image, output_dtype)
      self.assertTrue(y.dtype == output_dtype)
      self.assertAllClose(y.eval(), y_np, atol=1e-5)

  def testNoConvert(self):
    # Make sure converting to the same data type creates no ops
    with self.test_session():
      image = constant_op.constant([1], dtype=dtypes.uint8)
      y = image_ops.convert_image_dtype(image, dtypes.uint8)
      self.assertEquals(image, y)

  def testConvertBetweenInteger(self):
    # Make sure converting to between integer types scales appropriately
    with self.test_session():
      self._convert([0, 255], dtypes.uint8, dtypes.int16, [0, 255 * 128])
      self._convert([0, 32767], dtypes.int16, dtypes.uint8, [0, 255])

  def testConvertBetweenFloat(self):
    # Make sure converting to between float types does nothing interesting
    with self.test_session():
      self._convert([-1.0, 0, 1.0, 200000], dtypes.float32, dtypes.float64,
                    [-1.0, 0, 1.0, 200000])
      self._convert([-1.0, 0, 1.0, 200000], dtypes.float64, dtypes.float32,
                    [-1.0, 0, 1.0, 200000])

  def testConvertBetweenIntegerAndFloat(self):
    # Make sure converting from and to a float type scales appropriately
    with self.test_session():
      self._convert([0, 1, 255], dtypes.uint8, dtypes.float32,
                    [0, 1.0 / 255.0, 1])
      self._convert([0, 1.1 / 255.0, 1], dtypes.float32, dtypes.uint8,
                    [0, 1, 255])

if __name__ == '__main__':
  googletest.main()
