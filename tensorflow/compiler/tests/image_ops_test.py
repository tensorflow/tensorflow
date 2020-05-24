# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for image ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import colorsys
import math
import os

from absl.testing import parameterized
import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.platform import test


def _generate_numpy_random_rgb(shape):
  # Only generate floating points that are fractions like n / 256, since they
  # are RGB pixels. Some low-precision floating point types in this test can't
  # handle arbitrary precision floating points well.
  return np.random.randint(0, 256, shape) / 256.


class RGBToHSVTest(xla_test.XLATestCase):

  def testBatch(self):
    # Build an arbitrary RGB image
    np.random.seed(7)
    batch_size = 5
    shape = (batch_size, 2, 7, 3)

    for nptype in self.float_types:
      inp = _generate_numpy_random_rgb(shape).astype(nptype)

      # Convert to HSV and back, as a batch and individually
      with self.session() as sess:
        batch0 = array_ops.placeholder(nptype, shape=shape)
        with self.test_scope():
          batch1 = image_ops.rgb_to_hsv(batch0)
          batch2 = image_ops.hsv_to_rgb(batch1)
        split0 = array_ops.unstack(batch0)
        with self.test_scope():
          split1 = list(map(image_ops.rgb_to_hsv, split0))
          split2 = list(map(image_ops.hsv_to_rgb, split1))
        join1 = array_ops.stack(split1)
        join2 = array_ops.stack(split2)
        batch1, batch2, join1, join2 = sess.run([batch1, batch2, join1, join2],
                                                {batch0: inp})

      # Verify that processing batch elements together is the same as separate
      self.assertAllCloseAccordingToType(batch1, join1, half_rtol=0.000002)
      self.assertAllCloseAccordingToType(batch2, join2, half_rtol=0.000002)
      self.assertAllCloseAccordingToType(
          batch2, inp, bfloat16_atol=0.03, half_rtol=0.02)

  def testRGBToHSVRoundTrip(self):
    data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    for nptype in self.float_types:
      rgb_np = np.array(data, dtype=nptype).reshape([2, 2, 3]) / 255.
      with self.session():
        placeholder = array_ops.placeholder(nptype)
        with self.test_scope():
          hsv = image_ops.rgb_to_hsv(placeholder)
          rgb = image_ops.hsv_to_rgb(hsv)
        rgb_tf = rgb.eval(feed_dict={placeholder: rgb_np})
      self.assertAllCloseAccordingToType(rgb_tf, rgb_np, bfloat16_atol=0.03)

  def testRGBToHSVNumpy(self):
    """Tests the RGB to HSV conversion matches a reference implementation."""
    for nptype in self.float_types:
      rgb_flat = _generate_numpy_random_rgb((64, 3)).astype(nptype)
      rgb_np = rgb_flat.reshape(4, 4, 4, 3)
      hsv_np = np.array([
          colorsys.rgb_to_hsv(
              r.astype(np.float64), g.astype(np.float64), b.astype(np.float64))
          for r, g, b in rgb_flat
      ])
      hsv_np = hsv_np.reshape(4, 4, 4, 3)
      with self.session():
        placeholder = array_ops.placeholder(nptype)
        with self.test_scope():
          hsv_op = image_ops.rgb_to_hsv(placeholder)
        hsv_tf = hsv_op.eval(feed_dict={placeholder: rgb_np})
      self.assertAllCloseAccordingToType(hsv_tf, hsv_np)


class AdjustContrastTest(xla_test.XLATestCase):

  def _testContrast(self, x_np, y_np, contrast_factor):
    with self.session():
      x = array_ops.placeholder(x_np.dtype, shape=x_np.shape)
      flt_x = image_ops.convert_image_dtype(x, dtypes.float32)
      with self.test_scope():
        y = image_ops.adjust_contrast(flt_x, contrast_factor)
      y = image_ops.convert_image_dtype(y, x.dtype, saturate=True)
      y_tf = y.eval({x: x_np})
      self.assertAllClose(y_tf, y_np, 1e-6)

  def testFloatContrast(self):
    x_shape = [1, 2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.float32).reshape(x_shape) / 255.

    y_data = [
        -45.25, -90.75, -92.5, 62.75, 169.25, 333.5, 28.75, -84.75, 349.5,
        134.75, 409.25, -116.5
    ]
    y_np = np.array(y_data, dtype=np.float32).reshape(x_shape) / 255.

    self._testContrast(x_np, y_np, contrast_factor=2.0)

  def testBatchContrast(self):
    x_shape = [2, 1, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    y_data = [0, 0, 0, 81, 200, 255, 10, 0, 255, 116, 255, 0]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    self._testContrast(x_np, y_np, contrast_factor=2.0)

  def _adjustContrastNp(self, x_np, contrast_factor):
    mean = np.mean(x_np, (1, 2), keepdims=True)
    y_np = mean + contrast_factor * (x_np - mean)
    return y_np

  def _adjustContrastTf(self, x_np, contrast_factor):
    with self.session():
      x = array_ops.placeholder(np.float32)
      with self.test_scope():
        y = image_ops.adjust_contrast(x, contrast_factor)
      y_tf = y.eval({x: x_np})
    return y_tf

  def testRandomContrast(self):
    x_shapes = [
        [1, 2, 2, 3],
        [2, 1, 2, 3],
        [1, 2, 2, 3],
        [2, 5, 5, 3],
        [2, 1, 1, 3],
    ]
    for x_shape in x_shapes:
      x_np = np.random.rand(*x_shape) * 255.
      contrast_factor = np.random.rand() * 2.0 + 0.1
      y_np = self._adjustContrastNp(x_np, contrast_factor)
      y_tf = self._adjustContrastTf(x_np, contrast_factor)
      self.assertAllClose(y_tf, y_np, rtol=1e-5, atol=1e-5)


class AdjustHueTest(xla_test.XLATestCase):

  def testAdjustNegativeHue(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    delta = -0.25
    y_data = [0, 13, 1, 54, 226, 59, 8, 234, 150, 255, 39, 1]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.session():
      x = array_ops.placeholder(x_np.dtype, shape=x_shape)
      flt_x = image_ops.convert_image_dtype(x, dtypes.float32)
      with self.test_scope():
        y = gen_image_ops.adjust_hue(flt_x, delta)
      y = image_ops.convert_image_dtype(y, x.dtype, saturate=True)
      y_tf = y.eval({x: x_np})
      self.assertAllEqual(y_tf, y_np)

  def testAdjustPositiveHue(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    delta = 0.25
    y_data = [13, 0, 11, 226, 54, 221, 234, 8, 92, 1, 217, 255]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.session():
      x = array_ops.placeholder(x_np.dtype, shape=x_shape)
      flt_x = image_ops.convert_image_dtype(x, dtypes.float32)
      with self.test_scope():
        y = gen_image_ops.adjust_hue(flt_x, delta)
      y = image_ops.convert_image_dtype(y, x.dtype, saturate=True)
      y_tf = y.eval({x: x_np})
      self.assertAllEqual(y_tf, y_np)

  def testBatchAdjustHue(self):
    x_shape = [2, 1, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    delta = 0.25
    y_data = [13, 0, 11, 226, 54, 221, 234, 8, 92, 1, 217, 255]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.session():
      x = array_ops.placeholder(x_np.dtype, shape=x_shape)
      flt_x = image_ops.convert_image_dtype(x, dtypes.float32)
      with self.test_scope():
        y = gen_image_ops.adjust_hue(flt_x, delta)
      y = image_ops.convert_image_dtype(y, x.dtype, saturate=True)
      y_tf = y.eval({x: x_np})
      self.assertAllEqual(y_tf, y_np)

  def _adjustHueNp(self, x_np, delta_h):
    self.assertEqual(x_np.shape[-1], 3)
    x_v = x_np.reshape([-1, 3])
    y_v = np.ndarray(x_v.shape, dtype=x_v.dtype)
    channel_count = x_v.shape[0]
    for i in xrange(channel_count):
      r = x_v[i][0]
      g = x_v[i][1]
      b = x_v[i][2]
      h, s, v = colorsys.rgb_to_hsv(r, g, b)
      h += delta_h
      h = math.fmod(h + 10.0, 1.0)
      r, g, b = colorsys.hsv_to_rgb(h, s, v)
      y_v[i][0] = r
      y_v[i][1] = g
      y_v[i][2] = b
    return y_v.reshape(x_np.shape)

  def _adjustHueTf(self, x_np, delta_h):
    with self.session():
      x = array_ops.placeholder(dtypes.float32)
      with self.test_scope():
        y = gen_image_ops.adjust_hue(x, delta_h)
      y_tf = y.eval({x: x_np})
    return y_tf

  def testAdjustRandomHue(self):
    x_shapes = [
        [2, 2, 3],
        [4, 2, 3],
        [2, 4, 3],
        [2, 5, 3],
        [1000, 1, 3],
    ]
    test_styles = [
        "all_random",
        "rg_same",
        "rb_same",
        "gb_same",
        "rgb_same",
    ]
    for x_shape in x_shapes:
      for test_style in test_styles:
        x_np = np.random.rand(*x_shape) * 255.
        delta_h = np.random.rand() * 2.0 - 1.0
        if test_style == "all_random":
          pass
        elif test_style == "rg_same":
          x_np[..., 1] = x_np[..., 0]
        elif test_style == "rb_same":
          x_np[..., 2] = x_np[..., 0]
        elif test_style == "gb_same":
          x_np[..., 2] = x_np[..., 1]
        elif test_style == "rgb_same":
          x_np[..., 1] = x_np[..., 0]
          x_np[..., 2] = x_np[..., 0]
        else:
          raise AssertionError("Invalid test style: %s" % (test_style))
        y_np = self._adjustHueNp(x_np, delta_h)
        y_tf = self._adjustHueTf(x_np, delta_h)
        self.assertAllClose(y_tf, y_np, rtol=2e-5, atol=1e-4)

  def testInvalidShapes(self):
    fused = False
    if not fused:
      # The tests are known to pass with the fused adjust_hue. We will enable
      # them when the fused implementation is the default.
      return
    x_np = np.random.rand(2, 3) * 255.
    delta_h = np.random.rand() * 2.0 - 1.0
    fused = False
    with self.assertRaisesRegexp(ValueError, "Shape must be at least rank 3"):
      self._adjustHueTf(x_np, delta_h)
    x_np = np.random.rand(4, 2, 4) * 255.
    delta_h = np.random.rand() * 2.0 - 1.0
    with self.assertRaisesOpError("input must have 3 channels"):
      self._adjustHueTf(x_np, delta_h)


class AdjustSaturationTest(xla_test.XLATestCase):

  def _adjust_saturation(self, image, saturation_factor):
    image = ops.convert_to_tensor(image, name="image")
    orig_dtype = image.dtype
    flt_image = image_ops.convert_image_dtype(image, dtypes.float32)
    with self.test_scope():
      saturation_adjusted_image = gen_image_ops.adjust_saturation(
          flt_image, saturation_factor)
    return image_ops.convert_image_dtype(saturation_adjusted_image, orig_dtype)

  def testHalfSaturation(self):
    x_shape = [2, 2, 3]
    x_rgb_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_rgb_data, dtype=np.uint8).reshape(x_shape)

    saturation_factor = 0.5
    y_rgb_data = [6, 9, 13, 140, 180, 226, 135, 121, 234, 172, 255, 128]
    y_np = np.array(y_rgb_data, dtype=np.uint8).reshape(x_shape)

    with self.session():
      x = array_ops.placeholder(x_np.dtype, shape=x_shape)
      y = self._adjust_saturation(x, saturation_factor)
      y_tf = y.eval({x: x_np})
      self.assertAllEqual(y_tf, y_np)

  def testTwiceSaturation(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    saturation_factor = 2.0
    y_data = [0, 5, 13, 0, 106, 226, 30, 0, 234, 89, 255, 0]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.session():
      x = array_ops.placeholder(x_np.dtype, shape=x_shape)
      y = self._adjust_saturation(x, saturation_factor)
      y_tf = y.eval({x: x_np})
      self.assertAllEqual(y_tf, y_np)

  def _adjustSaturationNp(self, x_np, scale):
    self.assertEqual(x_np.shape[-1], 3)
    x_v = x_np.reshape([-1, 3])
    y_v = np.ndarray(x_v.shape, dtype=x_v.dtype)
    channel_count = x_v.shape[0]
    for i in xrange(channel_count):
      r = x_v[i][0]
      g = x_v[i][1]
      b = x_v[i][2]
      h, s, v = colorsys.rgb_to_hsv(r, g, b)
      s *= scale
      s = min(1.0, max(0.0, s))
      r, g, b = colorsys.hsv_to_rgb(h, s, v)
      y_v[i][0] = r
      y_v[i][1] = g
      y_v[i][2] = b
    return y_v.reshape(x_np.shape)

  def testAdjustRandomSaturation(self):
    x_shapes = [
        [2, 2, 3],
        [4, 2, 3],
        [2, 4, 3],
        [2, 5, 3],
        [1000, 1, 3],
    ]
    test_styles = [
        "all_random",
        "rg_same",
        "rb_same",
        "gb_same",
        "rgb_same",
    ]
    with self.session():
      for x_shape in x_shapes:
        for test_style in test_styles:
          x_np = np.random.rand(*x_shape) * 255.
          scale = np.random.rand()
          if test_style == "all_random":
            pass
          elif test_style == "rg_same":
            x_np[..., 1] = x_np[..., 0]
          elif test_style == "rb_same":
            x_np[..., 2] = x_np[..., 0]
          elif test_style == "gb_same":
            x_np[..., 2] = x_np[..., 1]
          elif test_style == "rgb_same":
            x_np[..., 1] = x_np[..., 0]
            x_np[..., 2] = x_np[..., 0]
          else:
            raise AssertionError("Invalid test style: %s" % (test_style))
          y_baseline = self._adjustSaturationNp(x_np, scale)
          x = array_ops.placeholder(dtypes.float32, shape=x_shape)
          with self.test_scope():
            y_fused = self._adjust_saturation(x,
                                              scale).eval(feed_dict={x: x_np})
          self.assertAllClose(y_fused, y_baseline, rtol=2e-5, atol=1e-5)


class ResizeNearestNeighborTest(xla_test.XLATestCase):
  # TODO(ilch): Wrap each test with `for dtype in self.float_types:`
  # Some work to understand how that should be done was presented here:
  # cl/227850213

  def _assertForwardOpMatchesExpected(self,
                                      image_np,
                                      target_shape,
                                      expected=None,
                                      large_tolerance=False,
                                      align_corners=True):
    if expected is None:
      self.fail("expected must be specified")
    with self.session() as sess, self.test_scope():
      image = array_ops.placeholder(image_np.dtype)
      resized = gen_image_ops.resize_nearest_neighbor(
          image, target_shape, align_corners=align_corners)
      out = sess.run(resized, {image: image_np[np.newaxis, :, :, np.newaxis]})
      if large_tolerance:
        self.assertAllClose(
            expected[np.newaxis, :, :, np.newaxis], out, rtol=2e-4, atol=2e-4)
      else:
        self.assertAllClose(expected[np.newaxis, :, :, np.newaxis], out)

  def testAlignCorners2x2To1x1(self):
    self._assertForwardOpMatchesExpected(
        np.array([[1, 2], [3, 4]], dtype=np.float32), [1, 1],
        expected=np.array([[1]], dtype=np.float32))

  def testAlignCorners1x1To2x2(self):
    self._assertForwardOpMatchesExpected(
        np.array([[1]], dtype=np.float32), [2, 2],
        expected=np.array([[1, 1], [1, 1]], dtype=np.float32))

  def testAlignCorners1x1To3x3(self):
    self._assertForwardOpMatchesExpected(
        np.array([[1]], dtype=np.float32), [3, 3],
        expected=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32))

  def testAlignCorners2x2To3x3(self):
    self._assertForwardOpMatchesExpected(
        np.array([[1, 2], [3, 4]], dtype=np.float32), [3, 3],
        expected=np.array([[1, 2, 2], [3, 4, 4], [3, 4, 4]], dtype=np.float32))

  def testAlignCorners2x2To4x4(self):
    self._assertForwardOpMatchesExpected(
        np.array([[1, 2], [3, 4]], dtype=np.float32), [4, 4],
        expected=np.array(
            [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]],
            dtype=np.float32), large_tolerance=True)

  def testAlignCorners3x3To2x2(self):
    self._assertForwardOpMatchesExpected(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32), [2, 2],
        expected=np.array([[1, 3], [7, 9]], dtype=np.float32))

  def testAlignCorners4x4To3x3(self):
    self._assertForwardOpMatchesExpected(
        np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=np.float32), [3, 3],
        expected=np.array([[1, 3, 4], [9, 11, 12], [13, 15, 16]],
                          dtype=np.float32))

  def testAlignCorners3x3To4x4(self):
    self._assertForwardOpMatchesExpected(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32), [4, 4],
        expected=np.array(
            [[1, 2, 2, 3], [4, 5, 5, 6], [4, 5, 5, 6], [7, 8, 8, 9]],
            dtype=np.float32))

  def testAlignCorners3x3To6x6(self):
    self._assertForwardOpMatchesExpected(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32), [6, 6],
        expected=np.array(
            [[1, 1, 2, 2, 3, 3], [1, 1, 2, 2, 3, 3], [4, 4, 5, 5, 6, 6],
             [4, 4, 5, 5, 6, 6], [7, 7, 8, 8, 9, 9], [7, 7, 8, 8, 9, 9]],
            dtype=np.float32))

  def testAlignCorners3x3To9x9(self):
    # The expected matrix might look uneven in terms of how many of each number
    # there is, but this is an artifact of doing the dilation and convolution
    # iteratively. The behavior is less esoteric in the 3x3To12x12 case below.
    self._assertForwardOpMatchesExpected(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32), [9, 9],
        expected=np.array(
            [[1, 1, 2, 2, 2, 2, 3, 3, 3], [1, 1, 2, 2, 2, 2, 3, 3, 3],
             [4, 4, 5, 5, 5, 5, 6, 6, 6], [4, 4, 5, 5, 5, 5, 6, 6, 6],
             [4, 4, 5, 5, 5, 5, 6, 6, 6], [4, 4, 5, 5, 5, 5, 6, 6, 6],
             [7, 7, 8, 8, 8, 8, 9, 9, 9], [7, 7, 8, 8, 8, 8, 9, 9, 9],
             [7, 7, 8, 8, 8, 8, 9, 9, 9]],
            dtype=np.float32))

  def testAlignCorners3x3To12x12(self):
    self._assertForwardOpMatchesExpected(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32), [12, 12],
        expected=np.array([[1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3],
                           [1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3],
                           [1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3],
                           [4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6],
                           [4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6],
                           [4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6],
                           [4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6],
                           [4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6],
                           [4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6],
                           [7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9],
                           [7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9],
                           [7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9]],
                          dtype=np.float32))

  def testBFloat16(self):
    img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                   dtype=dtypes.bfloat16.as_numpy_dtype)
    self._assertForwardOpMatchesExpected(img, [4, 4], expected=np.array(
        [[1, 2, 2, 3], [4, 5, 5, 6], [4, 5, 5, 6], [7, 8, 8, 9]],
        dtype=np.float32))

  def testAlignCorners3x3To12x12_uint8(self):
    # TODO(b/72099414): enable the test for TPU when the issue is fixed.
    if (self.device not in ["XLA_GPU", "XLA_CPU"]):
      return
    # Ensure that resize with convolution works on XLA/GPU for integer types
    self._assertForwardOpMatchesExpected(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8), [12, 12],
        expected=np.array([[1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3],
                           [1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3],
                           [1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3],
                           [4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6],
                           [4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6],
                           [4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6],
                           [4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6],
                           [4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6],
                           [4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6],
                           [7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9],
                           [7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9],
                           [7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9]],
                          dtype=np.uint8))


class ResizeBilinearTest(parameterized.TestCase, xla_test.XLATestCase):

  def _assertForwardOpMatchesExpected(self,
                                      image_np,
                                      target_shape,
                                      expected=None,
                                      large_tolerance=False,
                                      align_corners=True):
    if expected is None:
      self.fail("expected must be specified")
    with self.session() as sess, self.test_scope():
      image = array_ops.placeholder(image_np.dtype)
      resized = gen_image_ops.resize_bilinear(
          image, target_shape, align_corners=align_corners)
      out = sess.run(resized, {image: image_np[np.newaxis, :, :, np.newaxis]})
      if large_tolerance:
        self.assertAllClose(
            expected[np.newaxis, :, :, np.newaxis], out, rtol=0.1, atol=0.01)
      else:
        self.assertAllClose(expected[np.newaxis, :, :, np.newaxis], out)

  @parameterized.named_parameters(
      [("1x2To3x3", 1, 2, 3, 3), ("2x2To1x1", 2, 2, 1, 1),
       ("2x2To3x3", 2, 2, 3, 3), ("3x3To2x2", 3, 3, 2, 2),
       ("4x4To3x3", 4, 4, 3, 3), ("3x3To9x9", 3, 3, 9, 9),
       ("4x4To8x8", 4, 4, 8, 8), ("8x8To16x16", 8, 8, 16, 16),
       ("64x64To512x512", 64, 64, 512, 512),
       ("80x80To512x512", 80, 80, 512, 512),
       ("96x96To512x512", 96, 96, 512, 512),
       ("112x112To512x512", 112, 112, 512, 512),
       ("256x48To2048x384", 256, 48, 2048, 384),
       ("320x60To2048x384", 320, 60, 2048, 384),
       ("448x84To2048x384", 448, 84, 2048, 384),
       ("69x69To545x545", 69, 69, 545, 545),
       ("86x86To545x545", 86, 86, 545, 545),
       ("103x103To545x545", 103, 103, 545, 545),
       ("120x120To545x545", 120, 120, 545, 545),
       ("57x57To456x456", 57, 57, 456, 456),
       ("72x72To456x456", 72, 72, 456, 456),
       ("86x86To456x456", 86, 86, 456, 456),
       ("100x100To456x456", 100, 100, 456, 456),
       ("64x64To224x224", 64, 64, 224, 224),
       ("128x128To224x224", 128, 128, 224, 224),
       ("256x256To224x224", 256, 256, 224, 224),
       ("512x512To224x224", 512, 512, 224, 224),
       ("64x64To299x299", 64, 64, 299, 299),
       ("128x128To299x299", 128, 128, 299, 299),
       ("256x256To299x299", 256, 256, 299, 299),
       ("512x512To299x299", 512, 512, 299, 299),
       ("224x224To224x224", 224, 224, 224, 224)] +
      # On windows, initialization of the following or any larger np.arrays
      # where we set the dtype explicitly fails with:
      #   TypeError: expected number, got int
      ([] if os.name == "nt" else [("224x224To224x224-bfloat", 224, 224, 224,
                                    224, dtypes.bfloat16.as_numpy_dtype)]),
      # This test is disabled because it is very slow. It is slow because
      # 383 is prime, 383 and 2047 are coprime, and 2048 is large.
      # ("Disabled_384x72To2048x384", 384, 72, 2048, 384),
  )

  def test(self, src_y, src_x, dst_y, dst_x, dtype=np.float32):
    if test.is_built_with_rocm():
      self.skipTest("Disabled on ROCm, because it runs out of memory")

    max_y = max(src_y - 1, 1) * (dst_y - 1) + 1
    max_x = max(src_x - 1, 1) * (dst_x - 1) + 1

    input_data = [
        range(y * max_x, (y + 1) * max_x, max(dst_x - 1, 1))
        for y in range(0, max_y, max(dst_y - 1, 1))
    ]

    result = [
        range(y * max_x, (y + 1) * max_x, max(src_x - 1, 1))
        for y in range(0, max_y, max(src_y - 1, 1))
    ]

    self._assertForwardOpMatchesExpected(
        np.array(input_data, dtype=dtype), [dst_y, dst_x],
        expected=np.array(result, dtype=np.float32),
        large_tolerance=True)


class ResizeBilinearGradTest(parameterized.TestCase, xla_test.XLATestCase):

  def _assertBackwardOpMatchesExpected(self,
                                       grads_np,
                                       input_shape=None,
                                       dtype=None,
                                       expected=None,
                                       large_tolerance=False):
    if input_shape is None:
      self.fail("input_shape must be specified")
    if expected is None:
      self.fail("expected must be specified")
    with self.session() as sess, self.test_scope():
      dtype = dtype or np.float32
      grads = array_ops.placeholder(np.float32)
      resized = gen_image_ops.resize_bilinear_grad(
          grads,
          np.zeros([1, input_shape[0], input_shape[1], 1], dtype=dtype),
          align_corners=True)
      out = sess.run(resized, {grads: grads_np[np.newaxis, :, :, np.newaxis]})
      if large_tolerance:
        self.assertAllClose(
            expected[np.newaxis, :, :, np.newaxis], out, rtol=0.1, atol=0.01)
      else:
        self.assertAllCloseAccordingToType(
            expected[np.newaxis, :, :, np.newaxis], out)

  @parameterized.named_parameters(
      ("1x3To1x3", 1, 2, 1, 3),
      ("1x2To3x2", 1, 2, 3, 2),
      ("1x2To3x3", 1, 2, 3, 3),
      ("1x1To4x1", 1, 1, 4, 1),
      ("1x1To5x1", 1, 1, 5, 1),
      ("2x2To1x1", 2, 2, 1, 1),
      ("2x2To3x3", 2, 2, 3, 3),
      ("3x3To2x2", 3, 3, 2, 2),
      ("4x4To3x3", 4, 4, 3, 3),
      ("3x3To9x9", 3, 3, 9, 9),
      ("4x4To8x8", 4, 4, 8, 8),
      ("8x8To16x16", 8, 8, 16, 16),
      ("2x64To2x512", 2, 64, 2, 512),
      ("64x64To512x512", 64, 64, 512, 512),
      ("80x80To512x512", 80, 80, 512, 512),
      ("96x96To512x512", 96, 96, 512, 512),
      ("112x112To512x512", 112, 112, 512, 512),
      # ("Disabled_256x48To2048x384", 256, 48, 2048, 384),
      # ("Disabled_320x60To2048x384", 320, 60, 2048, 384),
      # ("Disabled_448x84To2048x384", 448, 84, 2048, 384),
      ("69x69To545x545", 69, 69, 545, 545),
      ("86x86To545x545", 86, 86, 545, 545),
      ("103x103To545x545", 103, 103, 545, 545),
      ("120x120To545x545", 120, 120, 545, 545),
      ("57x57To456x456", 57, 57, 456, 456),
      ("72x72To456x456", 72, 72, 456, 456),
      ("86x86To456x456", 86, 86, 456, 456),
      ("100x100To456x456", 100, 100, 456, 456),
      # This test is disabled because it is very slow. It is slow because
      # 383 is prime, 383 and 2047 are coprime, and 2048 is large.
      # ("Disabled_384x72To2048x384", 384, 72, 2048, 384),
  )

  def test(self, src_y, src_x, dst_y, dst_x):
    def GetRow(src, dst):
      if src == 1:
        return np.array([[max(dst**2 - dst, 1)]])
      row = [0] * src
      for i in range(0, (dst - 1) * max(src - 1, 1) + 1, src - 1):
        prev = int(math.floor(i / max(dst - 1, 1)))
        row[prev] += max(dst - 1, 1) - i % max(dst - 1, 1)
        if prev + 1 < src:
          row[prev + 1] += i % max(dst - 1, 1)
      return np.array([row])

    input_element = max(dst_x - 1, 1) * max(dst_y - 1, 1)
    input_data = [[input_element] * dst_x] * dst_y
    result = GetRow(src_x, dst_x) * np.transpose(GetRow(src_y, dst_y))
    self._assertBackwardOpMatchesExpected(
        np.array(input_data, dtype=np.float32), [src_y, src_x],
        expected=np.array(result, dtype=np.float32),
        large_tolerance=True)


class ResizeBilinearNonAlignCornersTest(xla_test.XLATestCase):

  def _assertForwardOpMatchesExpected(self,
                                      image_np,
                                      target_shape,
                                      expected=None,
                                      large_tolerance=False,
                                      align_corners=True):
    if expected is None:
      self.fail("expected must be specified")
    with self.session() as sess, self.test_scope():
      image = array_ops.placeholder(image_np.dtype)
      resized = gen_image_ops.resize_bilinear(
          image, target_shape, align_corners=align_corners)
      out = sess.run(resized, {image: image_np[np.newaxis, :, :, np.newaxis]})
      if large_tolerance:
        self.assertAllClose(
            expected[np.newaxis, :, :, np.newaxis], out, rtol=0.1, atol=0.01)
      else:
        self.assertAllClose(expected[np.newaxis, :, :, np.newaxis], out)

  def testNonAlignCorners3x2To6x4(self):
    input_data = [[64, 32], [32, 64], [50, 100]]
    expected_data = [[64.0, 48.0, 32.0, 32.0], [48.0, 48.0, 48.0, 48.0],
                     [32.0, 48.0, 64.0, 64.0], [41.0, 61.5, 82.0, 82.0],
                     [50.0, 75.0, 100.0, 100.0], [50.0, 75.0, 100.0, 100.0]]
    for dtype in self.float_types:
      self._assertForwardOpMatchesExpected(
          np.array(input_data, dtype=dtype), [6, 4],
          expected=np.array(expected_data, dtype=np.float32),
          align_corners=False)

  def testNonAlignCorners6x4To3x2(self):
    input_data = [[127, 127, 64, 64], [127, 127, 64, 64], [64, 64, 127, 127],
                  [64, 64, 127, 127], [50, 50, 100, 100], [50, 50, 100, 100]]
    expected_data = [[127, 64], [64, 127], [50, 100]]
    for dtype in self.float_types:
      self._assertForwardOpMatchesExpected(
          np.array(input_data, dtype=dtype), [3, 2],
          expected=np.array(expected_data, dtype=dtype),
          align_corners=False)

  def testNonAlignCorners3x2To6x4Batch2(self):
    input_data = [[[64, 32], [32, 64], [50, 100]], [[32, 16], [16, 32],
                                                    [25, 50]]]
    expected_data = [[[64.0, 48.0, 32.0, 32.0], [48.0, 48.0, 48.0, 48.0],
                      [32.0, 48.0, 64.0, 64.0], [41.0, 61.5, 82.0, 82.0],
                      [50.0, 75.0, 100.0, 100.0], [50.0, 75.0, 100.0, 100.0]],
                     [[32.0, 24.0, 16.0, 16.0], [24.0, 24.0, 24.0, 24.0],
                      [16.0, 24.0, 32.0, 32.0], [20.5, 30.75, 41.0, 41.0],
                      [25.0, 37.5, 50.0, 50.0], [25.0, 37.5, 50.0, 50.0]]]

    for dtype in self.float_types:
      input_image = np.array(input_data, dtype=dtype)
      expected = np.array(expected_data, dtype=dtype)
      with self.session() as sess, self.test_scope():
        image = array_ops.placeholder(input_image.dtype)
        resized = gen_image_ops.resize_bilinear(
            image, [6, 4], align_corners=False)
        out = sess.run(resized, {image: input_image[:, :, :, np.newaxis]})
        self.assertAllClose(expected[:, :, :, np.newaxis], out)


class NonMaxSuppressionTest(xla_test.XLATestCase):

  def testNMS128From1024(self):
    num_boxes = 1024
    boxes_np = np.random.normal(50, 10, (num_boxes, 4)).astype("f4")
    scores_np = np.random.normal(0.5, 0.1, (num_boxes,)).astype("f4")

    max_output_size = 128
    iou_threshold_np = np.array(0.5, dtype=np.float32)
    score_threshold_np = np.array(0.0, dtype=np.float32)

    with self.session() as sess:
      boxes = array_ops.placeholder(boxes_np.dtype, shape=boxes_np.shape)
      scores = array_ops.placeholder(scores_np.dtype, shape=scores_np.shape)
      iou_threshold = array_ops.placeholder(iou_threshold_np.dtype,
                                            iou_threshold_np.shape)
      score_threshold = array_ops.placeholder(score_threshold_np.dtype,
                                              score_threshold_np.shape)
      with self.test_scope():
        selected_indices = image_ops.non_max_suppression_padded(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            pad_to_max_output_size=True)
      inputs_feed = {
          boxes: boxes_np,
          scores: scores_np,
          score_threshold: score_threshold_np,
          iou_threshold: iou_threshold_np
      }
      (indices_tf, _) = sess.run(selected_indices, feed_dict=inputs_feed)

      self.assertEqual(indices_tf.size, max_output_size)

  def testNMS3From6Boxes(self):
    # Three boxes are selected based on IOU.
    boxes_data = [[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, -0.1, 1, 0.9],
                  [0, 10, 1, 11], [0, 10.1, 1, 11.1], [0, 100, 1, 101]]
    boxes_np = np.array(boxes_data, dtype=np.float32)

    scores_data = [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]
    scores_np = np.array(scores_data, dtype=np.float32)

    max_output_size = 3
    iou_threshold_np = np.array(0.5, dtype=np.float32)
    score_threshold_np = np.array(0.0, dtype=np.float32)

    with self.session() as sess:
      boxes = array_ops.placeholder(boxes_np.dtype, shape=boxes_np.shape)
      scores = array_ops.placeholder(scores_np.dtype, shape=scores_np.shape)
      iou_threshold = array_ops.placeholder(iou_threshold_np.dtype,
                                            iou_threshold_np.shape)
      score_threshold = array_ops.placeholder(score_threshold_np.dtype,
                                              score_threshold_np.shape)
      with self.test_scope():
        selected_indices = image_ops.non_max_suppression_padded(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            pad_to_max_output_size=True)
      inputs_feed = {
          boxes: boxes_np,
          scores: scores_np,
          score_threshold: score_threshold_np,
          iou_threshold: iou_threshold_np
      }
      (indices_tf, num_valid) = sess.run(
          selected_indices, feed_dict=inputs_feed)

      self.assertEqual(indices_tf.size, max_output_size)
      self.assertEqual(num_valid, 3)
      self.assertAllClose(indices_tf[:num_valid], [3, 0, 5])

  def testNMS3Then2WithScoreThresh(self):
    # Three boxes are selected based on IOU.
    # One is filtered out by score threshold.

    boxes_data = [[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, -0.1, 1, 0.9],
                  [0, 10, 1, 11], [0, 10.1, 1, 11.1], [0, 100, 1, 101]]
    boxes_np = np.array(boxes_data, dtype=np.float32)

    scores_data = [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]
    scores_np = np.array(scores_data, dtype=np.float32)
    max_output_size = 3
    iou_threshold_np = np.array(0.5, dtype=np.float32)
    score_threshold_np = np.array(0.4, dtype=np.float32)

    with self.session() as sess:
      boxes = array_ops.placeholder(boxes_np.dtype, shape=boxes_np.shape)
      scores = array_ops.placeholder(scores_np.dtype, shape=scores_np.shape)
      iou_threshold = array_ops.placeholder(iou_threshold_np.dtype,
                                            iou_threshold_np.shape)
      score_threshold = array_ops.placeholder(score_threshold_np.dtype,
                                              score_threshold_np.shape)
      with self.test_scope():
        selected_indices = image_ops.non_max_suppression_padded(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            pad_to_max_output_size=True)
      inputs_feed = {
          boxes: boxes_np,
          scores: scores_np,
          iou_threshold: iou_threshold_np,
          score_threshold: score_threshold_np
      }
      (indices_tf, num_valid) = sess.run(
          selected_indices, feed_dict=inputs_feed)

      self.assertEqual(indices_tf.size, max_output_size)
      self.assertEqual(num_valid, 2)
      self.assertAllClose(indices_tf[:num_valid], [3, 0])

  def testNMS3Then1WithScoreMaxThresh(self):
    # Three boxes are selected based on IOU.
    # One is filtered out by score threshold.
    # One is filtered out by max_output_size.

    boxes_data = [[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, -0.1, 1, 0.9],
                  [0, 10, 1, 11], [0, 10.1, 1, 11.1], [0, 100, 1, 101]]
    boxes_np = np.array(boxes_data, dtype=np.float32)

    scores_data = [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]
    scores_np = np.array(scores_data, dtype=np.float32)
    max_output_size = 1
    iou_threshold_np = np.array(0.5, dtype=np.float32)
    score_threshold_np = np.array(0.4, dtype=np.float32)

    with self.session() as sess:
      boxes = array_ops.placeholder(boxes_np.dtype, shape=boxes_np.shape)
      scores = array_ops.placeholder(scores_np.dtype, shape=scores_np.shape)
      iou_threshold = array_ops.placeholder(iou_threshold_np.dtype,
                                            iou_threshold_np.shape)
      score_threshold = array_ops.placeholder(score_threshold_np.dtype,
                                              score_threshold_np.shape)
      with self.test_scope():
        selected_indices = image_ops.non_max_suppression_padded(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            pad_to_max_output_size=True)
      inputs_feed = {
          boxes: boxes_np,
          scores: scores_np,
          iou_threshold: iou_threshold_np,
          score_threshold: score_threshold_np
      }
      (indices_tf, num_valid) = sess.run(
          selected_indices, feed_dict=inputs_feed)

      self.assertEqual(indices_tf.size, max_output_size)
      self.assertEqual(num_valid, 1)
      self.assertAllClose(indices_tf[:num_valid], [3])

  def testSelectFromContinuousOverLap(self):
    # Tests that a suppressed box does not itself suppress other boxes.

    boxes_data = [[0, 0, 1, 1], [0, 0.2, 1, 1.2], [0, 0.4, 1, 1.4],
                  [0, 0.6, 1, 1.6], [0, 0.8, 1, 1.8], [0, 2, 1, 3]]
    boxes_np = np.array(boxes_data, dtype=np.float32)

    scores_data = [0.9, 0.75, 0.6, 0.5, 0.4, 0.3]
    scores_np = np.array(scores_data, dtype=np.float32)
    max_output_size = 3
    iou_threshold_np = np.array(0.5, dtype=np.float32)
    score_threshold_np = np.array(0.1, dtype=np.float32)

    with self.session() as sess:
      boxes = array_ops.placeholder(boxes_np.dtype, shape=boxes_np.shape)
      scores = array_ops.placeholder(scores_np.dtype, shape=scores_np.shape)
      iou_threshold = array_ops.placeholder(iou_threshold_np.dtype,
                                            iou_threshold_np.shape)
      score_threshold = array_ops.placeholder(score_threshold_np.dtype,
                                              score_threshold_np.shape)
      with self.test_scope():
        selected_indices = image_ops.non_max_suppression_padded(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            pad_to_max_output_size=True)
      inputs_feed = {
          boxes: boxes_np,
          scores: scores_np,
          iou_threshold: iou_threshold_np,
          score_threshold: score_threshold_np
      }
      (indices_tf, num_valid) = sess.run(
          selected_indices, feed_dict=inputs_feed)

      self.assertEqual(indices_tf.size, max_output_size)
      self.assertEqual(num_valid, 3)
      self.assertAllClose(indices_tf[:num_valid], [0, 2, 4])


class BatchedNonMaxSuppressionCorrectnessTest(xla_test.XLATestCase):

  def testBatchedNMSFrom6(self):
    boxes_data = [[[0, 0, 1, 1], [3, 3, 4, 4], [0, 0.4, 1, 1.4],
                   [0, 0.6, 1, 1.6], [0, 0.8, 1, 1.8], [0, 2, 1, 2]],
                  [[0, 2, 1, 2], [0, 0.8, 1, 1.8], [0, 0.6, 1, 1.6],
                   [0, 0.4, 1, 1.4], [0, 0.2, 1, 1.2], [0, 0, 1, 1]]]
    scores_data = [[0.9, 0.7, 0.6, 0.5, 0.4, 0.3],
                   [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]]
    max_output_size = 6
    iou_threshold = 0.5
    boxes_np = np.array(boxes_data, dtype=np.float32)
    scores_np = np.array(scores_data, dtype=np.float32)

    with self.session() as sess:
      boxes = array_ops.placeholder(boxes_np.dtype, shape=boxes_np.shape)
      scores = array_ops.placeholder(scores_np.dtype, shape=scores_np.shape)

      with self.test_scope():
        (indices, num_valid) = image_ops.non_max_suppression_padded(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            pad_to_max_output_size=True,
            sorted_input=True,
            canonicalized_coordinates=True)

      inputs = {
          boxes: boxes_np,
          scores: scores_np
      }
      indices_output, num_valid_output = sess.run([indices, num_valid], inputs)
    invalid_index = 0
    self.assertAllEqual([[0, 1, 2, 4, 5, invalid_index],
                         [0, 1, 3, 5, invalid_index, invalid_index]],
                        indices_output)
    self.assertAllEqual([5, 4], num_valid_output)

  def testBatchedNMSFrom6Max3(self):
    boxes_data = [[[0, 0, 1, 1], [3, 3, 4, 4], [0, 0.4, 1, 1.4],
                   [0, 0.6, 1, 1.6], [0, 0.8, 1, 1.8], [0, 2, 1, 2]],
                  [[0, 2, 1, 2], [0, 0.8, 1, 1.8], [0, 0.6, 1, 1.6],
                   [0, 0.4, 1, 1.4], [0, 0.2, 1, 1.2], [0, 0, 1, 1]]]
    scores_data = [[0.9, 0.7, 0.6, 0.5, 0.4, 0.3],
                   [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]]
    max_output_size = 3
    iou_threshold = 0.5
    boxes_np = np.array(boxes_data, dtype=np.float32)
    scores_np = np.array(scores_data, dtype=np.float32)

    with self.session() as sess:
      boxes = array_ops.placeholder(boxes_np.dtype, shape=boxes_np.shape)
      scores = array_ops.placeholder(scores_np.dtype, shape=scores_np.shape)
      with self.test_scope():
        (indices, num_valid) = image_ops.non_max_suppression_padded(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            pad_to_max_output_size=True,
            sorted_input=True,
            canonicalized_coordinates=True)

      inputs = {
          boxes: boxes_np,
          scores: scores_np
      }
      indices_output, num_valid_output = sess.run([indices, num_valid], inputs)
    self.assertAllEqual([[0, 1, 2], [0, 1, 3]], indices_output)
    self.assertAllEqual([3, 3], num_valid_output)

  def testBatchedNMSSingleFrom6Max3(self):
    boxes_data = [[0, 0, 1, 1], [3, 3, 4, 4], [0, 0.4, 1, 1.4],
                  [0, 0.6, 1, 1.6], [0, 0.8, 1, 1.8], [0, 2, 1, 2]]
    scores_data = [0.9, 0.7, 0.6, 0.5, 0.4, 0.3]
    max_output_size = 3
    iou_threshold = 0.5
    boxes_np = np.array(boxes_data, dtype=np.float32)
    scores_np = np.array(scores_data, dtype=np.float32)

    with self.session() as sess:
      boxes = array_ops.placeholder(boxes_np.dtype, shape=boxes_np.shape)
      scores = array_ops.placeholder(scores_np.dtype, shape=scores_np.shape)
      with self.test_scope():
        (indices, num_valid) = image_ops.non_max_suppression_padded(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            pad_to_max_output_size=True,
            sorted_input=True,
            canonicalized_coordinates=True)

      inputs = {
          boxes: boxes_np,
          scores: scores_np
      }
      indices_output, num_valid_output = sess.run([indices, num_valid], inputs)
    self.assertAllEqual([0, 1, 2], indices_output)
    self.assertAllEqual(3, num_valid_output)

  def testBatchedNMSSingleFrom6NoPad(self):
    boxes_data = [[0, 0, 1, 1], [3, 3, 4, 4], [0, 0.4, 1, 1.4],
                  [0, 0.6, 1, 1.6], [0, 0.8, 1, 1.8], [0, 2, 1, 2]]
    scores_data = [0.9, 0.7, 0.6, 0.5, 0.4, 0.3]
    max_output_size = 6
    iou_threshold = 0.5
    boxes_np = np.array(boxes_data, dtype=np.float32)
    scores_np = np.array(scores_data, dtype=np.float32)

    with self.session() as sess:
      boxes = array_ops.placeholder(boxes_np.dtype, shape=boxes_np.shape)
      scores = array_ops.placeholder(scores_np.dtype, shape=scores_np.shape)
      with self.test_scope():
        (indices, num_valid) = image_ops.non_max_suppression_padded(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            sorted_input=True,
            canonicalized_coordinates=True)

      inputs = {
          boxes: boxes_np,
          scores: scores_np
      }
      indices_output, num_valid_output = sess.run([indices, num_valid], inputs)
    self.assertAllEqual([0, 1, 2, 4, 5], indices_output)
    self.assertAllEqual(5, num_valid_output)

  def testBatchedNMSBatchDimsFrom6Max3(self):
    boxes_data = [[[[0, 0, 1, 1], [3, 3, 4, 4], [0, 0.4, 1, 1.4],
                    [0, 0.6, 1, 1.6], [0, 0.8, 1, 1.8], [0, 2, 1, 2]],
                   [[0, 2, 1, 2], [0, 0.8, 1, 1.8], [0, 0.6, 1, 1.6],
                    [0, 0.4, 1, 1.4], [0, 0.2, 1, 1.2], [0, 0, 1, 1]]]]
    scores_data = [[[0.9, 0.7, 0.6, 0.5, 0.4, 0.3],
                    [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]]]
    max_output_size = 3
    iou_threshold = 0.5
    boxes_np = np.array(boxes_data, dtype=np.float32)
    scores_np = np.array(scores_data, dtype=np.float32)

    with self.session() as sess:
      boxes = array_ops.placeholder(boxes_np.dtype, shape=boxes_np.shape)
      scores = array_ops.placeholder(scores_np.dtype, shape=scores_np.shape)
      with self.test_scope():
        (indices, num_valid) = image_ops.non_max_suppression_padded(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            pad_to_max_output_size=True,
            sorted_input=True,
            canonicalized_coordinates=True)

      inputs = {
          boxes: boxes_np,
          scores: scores_np
      }
      indices_output, num_valid_output = sess.run([indices, num_valid], inputs)
    self.assertAllEqual([[[0, 1, 2], [0, 1, 3]]], indices_output)
    self.assertAllEqual([[3, 3]], num_valid_output)

  def testBatchedNMSScoreThresholdFrom6Max3(self):
    boxes_data = [[[0, 0, 1, 1], [3, 3, 4, 4], [0, 0.4, 1, 1.4],
                   [0, 0.6, 1, 1.6], [0, 0.8, 1, 1.8], [0, 2, 1, 2]],
                  [[0, 2, 1, 2], [0, 0.8, 1, 1.8], [0, 0.6, 1, 1.6],
                   [0, 0.4, 1, 1.4], [0, 0.2, 1, 1.2], [0, 0, 1, 1]]]
    scores_data = [[0.9, 0.7, 0.6, 0.4, 0.3, 0.2],
                   [0.8, 0.7, 0.6, 0.4, 0.3, 0.1]]
    max_output_size = 3
    iou_threshold = 0.5
    boxes_np = np.array(boxes_data, dtype=np.float32)
    scores_np = np.array(scores_data, dtype=np.float32)

    with self.session() as sess:
      boxes = array_ops.placeholder(boxes_np.dtype, shape=boxes_np.shape)
      scores = array_ops.placeholder(scores_np.dtype, shape=scores_np.shape)
      with self.test_scope():
        (indices, num_valid) = image_ops.non_max_suppression_padded(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            score_threshold=0.5,
            pad_to_max_output_size=True,
            sorted_input=True,
            canonicalized_coordinates=True)

      inputs = {
          boxes: boxes_np,
          scores: scores_np
      }
      indices_output, num_valid_output = sess.run([indices, num_valid], inputs)
    invalid_index = 0
    self.assertAllEqual([3, 2], num_valid_output)
    self.assertAllEqual([[0, 1, 2], [0, 1, invalid_index]], indices_output)

  def testBatchedNMSUnsortedInputFrom6(self):
    boxes_data = [[[0, 2, 1, 2], [3, 3, 4, 4], [0, 0, 1, 1],
                   [0, 0.4, 1, 1.4], [0, 0.6, 1, 1.6], [0, 0.8, 1, 1.8]],
                  [[0, 0.4, 1, 1.4], [0, 2, 1, 2], [0, 0.2, 1, 1.2],
                   [0, 0, 1, 1], [0, 0.6, 1, 1.6], [0, 0.8, 1, 1.8]]]
    scores_data = [[0.3, 0.7, 0.9, 0.6, 0.5, 0.4],
                   [0.5, 0.8, 0.4, 0.3, 0.6, 0.7]]
    max_output_size = 6
    iou_threshold = 0.5
    boxes_np = np.array(boxes_data, dtype=np.float32)
    scores_np = np.array(scores_data, dtype=np.float32)

    with self.session() as sess:
      boxes = array_ops.placeholder(boxes_np.dtype, shape=boxes_np.shape)
      scores = array_ops.placeholder(scores_np.dtype, shape=scores_np.shape)

      with self.test_scope():
        (indices, num_valid) = image_ops.non_max_suppression_padded(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            pad_to_max_output_size=True,
            canonicalized_coordinates=True)

      inputs = {
          boxes: boxes_np,
          scores: scores_np
      }
      indices_output, num_valid_output = sess.run([indices, num_valid], inputs)
    invalid_index = 0
    self.assertAllEqual([[2, 1, 3, 5, 0, invalid_index],
                         [1, 5, 0, 3, invalid_index, invalid_index]],
                        indices_output)
    self.assertAllEqual([5, 4], num_valid_output)

  def testBatchedNMSNoncanonicalizedInputFrom6(self):
    boxes_data = [[[1, 0, 0, 1], [4, 3, 3, 4], [1, 0.4, 0, 1.4],
                   [1, 0.6, 0, 1.6], [1, 0.8, 0, 1.8], [1, 2, 0, 2]],
                  [[1, 2, 0, 2], [1, 0.8, 0, 1.8], [1, 0.6, 0, 1.6],
                   [1, 0.4, 0, 1.4], [1, 0.2, 0, 1.2], [1, 0, 0, 1]]]

    scores_data = [[0.9, 0.7, 0.6, 0.5, 0.4, 0.3],
                   [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]]
    max_output_size = 6
    iou_threshold = 0.5
    boxes_np = np.array(boxes_data, dtype=np.float32)
    scores_np = np.array(scores_data, dtype=np.float32)

    with self.session() as sess:
      boxes = array_ops.placeholder(boxes_np.dtype, shape=boxes_np.shape)
      scores = array_ops.placeholder(scores_np.dtype, shape=scores_np.shape)

      with self.test_scope():
        (indices, num_valid) = image_ops.non_max_suppression_padded(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            pad_to_max_output_size=True,
            sorted_input=True)

      inputs = {
          boxes: boxes_np,
          scores: scores_np
      }
      indices_output, num_valid_output = sess.run([indices, num_valid], inputs)
    invalid_index = 0
    self.assertAllEqual([[0, 1, 2, 4, 5, invalid_index],
                         [0, 1, 3, 5, invalid_index, invalid_index]],
                        indices_output)
    self.assertAllEqual([5, 4], num_valid_output)

  def testBatchedNMSScoreThresholdCanInputsFrom6Max3(self):
    boxes_data = [[[0, 0, 1, 1], [3, 3, 4, 4], [0, 0.4, 1, 1.4],
                   [0, 0.6, 1, 1.6], [0, 0.8, 1, 1.8], [0, 2, 1, 2]],
                  [[0, 2, 1, 2], [0, 0.8, 1, 1.8], [0, 0.6, 1, 1.6],
                   [0, 0.4, 1, 1.4], [0, 0.2, 1, 1.2], [0, 0, 1, 1]]]
    scores_data = [[0.9, 0.7, 0.6, 0.4, 0.3, 0.2],
                   [0.8, 0.7, 0.6, 0.4, 0.3, 0.1]]
    max_output_size = 3
    iou_threshold = 0.5
    boxes_np = np.array(boxes_data, dtype=np.float32)
    scores_np = np.array(scores_data, dtype=np.float32)

    with self.session() as sess:
      boxes = array_ops.placeholder(boxes_np.dtype, shape=boxes_np.shape)
      scores = array_ops.placeholder(scores_np.dtype, shape=scores_np.shape)
      with self.test_scope():
        (indices, num_valid) = image_ops.non_max_suppression_padded(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            score_threshold=0.5,
            pad_to_max_output_size=True,
            sorted_input=True,
            canonicalized_coordinates=False)

      inputs = {
          boxes: boxes_np,
          scores: scores_np
      }
      indices_output, num_valid_output = sess.run([indices, num_valid], inputs)
    invalid_index = 0
    self.assertAllEqual([3, 2], num_valid_output)
    self.assertAllEqual([[0, 1, 2], [0, 1, invalid_index]], indices_output)

  def testBatchedNMSFrom6DynamicInput(self):
    boxes_data = [[[0, 0, 1, 1], [3, 3, 4, 4], [0, 0.4, 1, 1.4],
                   [0, 0.6, 1, 1.6], [0, 0.8, 1, 1.8], [0, 2, 1, 2]],
                  [[0, 2, 1, 2], [0, 0.8, 1, 1.8], [0, 0.6, 1, 1.6],
                   [0, 0.4, 1, 1.4], [0, 0.2, 1, 1.2], [0, 0, 1, 1]]]
    scores_data = [[0.9, 0.7, 0.6, 0.5, 0.4, 0.3],
                   [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]]
    max_output_size = 6
    iou_threshold = 0.5
    boxes_np = np.array(boxes_data, dtype=np.float32)
    scores_np = np.array(scores_data, dtype=np.float32)

    with self.session() as sess:
      boxes = array_ops.placeholder(boxes_np.dtype)
      scores = array_ops.placeholder(scores_np.dtype)

      with self.test_scope():
        (indices, num_valid) = image_ops.non_max_suppression_padded(
            boxes=boxes,
            scores=scores,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            pad_to_max_output_size=True,
            sorted_input=True,
            canonicalized_coordinates=True)

      inputs = {
          boxes: boxes_np,
          scores: scores_np
      }
      indices_output, num_valid_output = sess.run([indices, num_valid], inputs)
    invalid_index = 0
    self.assertAllEqual([[0, 1, 2, 4, 5, invalid_index],
                         [0, 1, 3, 5, invalid_index, invalid_index]],
                        indices_output)
    self.assertAllEqual([5, 4], num_valid_output)
if __name__ == "__main__":
  test.main()
