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

import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.compiler.tests.xla_test import XLATestCase
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.platform import test


def GenerateNumpyRandomRGB(shape):
  # Only generate floating points that are fractions like n / 256, since they
  # are RGB pixels. Some low-precision floating point types in this test can't
  # handle arbitrary precision floating points well.
  return np.random.randint(0, 256, shape) / 256.


class RGBToHSVTest(XLATestCase):

  def testBatch(self):
    # Build an arbitrary RGB image
    np.random.seed(7)
    batch_size = 5
    shape = (batch_size, 2, 7, 3)

    for nptype in self.float_types:
      inp = GenerateNumpyRandomRGB(shape).astype(nptype)

      # Convert to HSV and back, as a batch and individually
      with self.test_session() as sess:
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
                                                {
                                                    batch0: inp
                                                })

      # Verify that processing batch elements together is the same as separate
      self.assertAllClose(batch1, join1)
      self.assertAllClose(batch2, join2)
      self.assertAllCloseAccordingToType(
          batch2, inp, bfloat16_atol=0.03, half_rtol=0.02)

  def testRGBToHSVRoundTrip(self):
    data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    for nptype in self.float_types:
      rgb_np = np.array(data, dtype=nptype).reshape([2, 2, 3]) / 255.
      with self.test_session():
        placeholder = array_ops.placeholder(nptype)
        with self.test_scope():
          hsv = image_ops.rgb_to_hsv(placeholder)
          rgb = image_ops.hsv_to_rgb(hsv)
        rgb_tf = rgb.eval(feed_dict={placeholder: rgb_np})
      self.assertAllCloseAccordingToType(rgb_tf, rgb_np, bfloat16_atol=0.03)

  def testRGBToHSVNumpy(self):
    """Tests the RGB to HSV conversion matches a reference implementation."""
    for nptype in self.float_types:
      rgb_flat = GenerateNumpyRandomRGB((64, 3)).astype(nptype)
      rgb_np = rgb_flat.reshape(4, 4, 4, 3)
      hsv_np = np.array([
          colorsys.rgb_to_hsv(
              r.astype(np.float64), g.astype(np.float64), b.astype(np.float64))
          for r, g, b in rgb_flat
      ])
      hsv_np = hsv_np.reshape(4, 4, 4, 3)
      with self.test_session():
        placeholder = array_ops.placeholder(nptype)
        with self.test_scope():
          hsv_op = image_ops.rgb_to_hsv(placeholder)
        hsv_tf = hsv_op.eval(feed_dict={placeholder: rgb_np})
      self.assertAllCloseAccordingToType(hsv_tf, hsv_np)


class AdjustContrastTest(XLATestCase):

  def _testContrast(self, x_np, y_np, contrast_factor):
    with self.test_session():
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
    with self.test_session():
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


class AdjustHueTest(XLATestCase):

  def testAdjustNegativeHue(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    delta = -0.25
    y_data = [0, 13, 1, 54, 226, 59, 8, 234, 150, 255, 39, 1]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.test_session():
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

    with self.test_session():
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

    with self.test_session():
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
    with self.test_session():
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


class AdjustSaturationTest(XLATestCase):

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

    with self.test_session():
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

    with self.test_session():
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
    with self.test_session():
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
                                              scale).eval(feed_dict={
                                                  x: x_np
                                              })
          self.assertAllClose(y_fused, y_baseline, rtol=2e-5, atol=1e-5)


class ResizeBilinearTest(XLATestCase):

  def _assertForwardOpMatchesExpected(self,
                                      image_np,
                                      target_shape,
                                      expected=None):
    if expected is None:
      self.fail("expected must be specified")
    with self.test_session() as sess, self.test_scope():
      image = array_ops.placeholder(image_np.dtype)
      resized = gen_image_ops.resize_bilinear(
          image, target_shape, align_corners=True)
      out = sess.run(resized, {image: image_np[np.newaxis, :, :, np.newaxis]})
      self.assertAllClose(expected[np.newaxis, :, :, np.newaxis], out)

  def _assertBackwardOpMatchesExpected(self,
                                       grads_np,
                                       input_shape=None,
                                       dtype=None,
                                       expected=None):
    if input_shape is None:
      self.fail("input_shape must be specified")
    if expected is None:
      self.fail("expected must be specified")
    with self.test_session() as sess, self.test_scope():
      dtype = dtype or np.float32
      grads = array_ops.placeholder(np.float32)
      resized = gen_image_ops.resize_bilinear_grad(
          grads,
          np.zeros([1, input_shape[0], input_shape[1], 1], dtype=dtype),
          align_corners=True)
      out = sess.run(resized, {grads: grads_np[np.newaxis, :, :, np.newaxis]})
      self.assertAllCloseAccordingToType(expected[np.newaxis, :, :, np.newaxis],
                                         out)

  def testAlignCorners1x2To3x2(self):
    for dtype in self.float_types:
      self._assertForwardOpMatchesExpected(
          np.array([[1, 2]], dtype=dtype), [3, 3],
          expected=np.array(
              [[1, 1.5, 2], [1, 1.5, 2], [1, 1.5, 2]], dtype=np.float32))

  def testAlignCorners1x2To3x2Grad(self):
    for dtype in self.float_types:
      self._assertBackwardOpMatchesExpected(
          np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
          input_shape=[1, 2],
          dtype=dtype,
          expected=np.array([[9, 12]], dtype=np.float32))

  def testAlignCorners2x2To1x1(self):
    for dtype in self.float_types:
      self._assertForwardOpMatchesExpected(
          np.array([[1, 2], [3, 4]], dtype=dtype), [1, 1],
          expected=np.array([[1]], dtype=np.float32))

  def testAlignCorners2x2To1x1Grad(self):
    for dtype in self.float_types:
      self._assertBackwardOpMatchesExpected(
          np.array([[7]], dtype=np.float32),
          input_shape=[2, 2],
          dtype=dtype,
          expected=np.array([[7, 0], [0, 0]], dtype=np.float32))

  def testAlignCorners2x2To3x3(self):
    for dtype in self.float_types:
      self._assertForwardOpMatchesExpected(
          np.array([[1, 2], [3, 4]], dtype=dtype), [3, 3],
          expected=np.array(
              [[1, 1.5, 2], [2, 2.5, 3], [3, 3.5, 4]], dtype=np.float32))

  def testAlignCorners2x2To3x3Grad(self):
    self._assertBackwardOpMatchesExpected(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        input_shape=[2, 2],
        expected=np.array([[5.25, 8.25], [14.25, 17.25]], dtype=np.float32))

  def testAlignCorners3x3To2x2(self):
    for dtype in self.float_types:
      self._assertForwardOpMatchesExpected(
          np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype), [2, 2],
          expected=np.array([[1, 3], [7, 9]], dtype=np.float32))

  def testAlignCorners3x3To2x2Grad(self):
    for dtype in self.float_types:
      self._assertBackwardOpMatchesExpected(
          np.array([[7, 13], [22, 4]], dtype=np.float32),
          input_shape=[3, 3],
          dtype=dtype,
          expected=np.array(
              [[7, 0, 13], [0, 0, 0], [22, 0, 4]], dtype=np.float32))

  def testAlignCorners4x4To3x3(self):
    for dtype in self.float_types:
      self._assertForwardOpMatchesExpected(
          np.array(
              [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
              dtype=dtype), [3, 3],
          expected=np.array(
              [[1, 2.5, 4], [7, 8.5, 10], [13, 14.5, 16]], dtype=np.float32))

  def testAlignCorners4x4To3x3Grad(self):
    for dtype in self.float_types:
      self._assertBackwardOpMatchesExpected(
          np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
          input_shape=[4, 4],
          dtype=dtype,
          expected=np.array(
              [[1, 1, 1, 3], [2, 1.25, 1.25, 3], [2, 1.25, 1.25, 3],
               [7, 4, 4, 9]],
              dtype=np.float32))

  def testAlignCorners3x3To9x9(self):
    for dtype in self.float_types:
      self._assertForwardOpMatchesExpected(
          np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype), [9, 9],
          expected=np.array(
              [[1.0, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00], [
                  1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75
              ], [2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50], [
                  3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00, 5.25
              ], [4.00, 4.25, 4.50, 4.75, 5.00, 5.25, 5.50, 5.75, 6.00], [
                  4.75, 5.00, 5.25, 5.50, 5.75, 6.00, 6.25, 6.50, 6.75
              ], [5.50, 5.75, 6.00, 6.25, 6.50, 6.75, 7.00, 7.25, 7.50], [
                  6.25, 6.50, 6.75, 7.00, 7.25, 7.50, 7.75, 8.00, 8.25
              ], [7.00, 7.25, 7.50, 7.75, 8.00, 8.25, 8.50, 8.75, 9.00]],
              dtype=np.float32))

  def testAlignCorners3x3To9x9Grad(self):
    for dtype in self.float_types:
      self._assertBackwardOpMatchesExpected(
          np.array(
              [[1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00], [
                  1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75
              ], [2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50], [
                  3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00, 5.25
              ], [4.00, 4.25, 4.50, 4.75, 5.00, 5.25, 5.50, 5.75, 6.00], [
                  4.75, 5.00, 5.25, 5.50, 5.75, 6.00, 6.25, 6.50, 6.75
              ], [5.50, 5.75, 6.00, 6.25, 6.50, 6.75, 7.00, 7.25, 7.50], [
                  6.25, 6.50, 6.75, 7.00, 7.25, 7.50, 7.75, 8.00, 8.25
              ], [7.00, 7.25, 7.50, 7.75, 8.00, 8.25, 8.50, 8.75, 9.00]],
              dtype=np.float32),
          input_shape=[3, 3],
          dtype=dtype,
          expected=np.array(
              [[12.5, 27.5, 21.875], [42.5, 80.0, 57.5], [40.625, 72.5, 50]],
              dtype=np.float32))


if __name__ == "__main__":
  test.main()
