# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

import colorsys
import functools
import itertools
import math
import os
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import image_ops_impl
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test


class RGBToHSVTest(test_util.TensorFlowTestCase):

  def testBatch(self):
    # Build an arbitrary RGB image
    np.random.seed(7)
    batch_size = 5
    shape = (batch_size, 2, 7, 3)

    for nptype in [np.float32, np.float64]:
      inp = np.random.rand(*shape).astype(nptype)

      # Convert to HSV and back, as a batch and individually
      with self.cached_session(use_gpu=True) as sess:
        batch0 = constant_op.constant(inp)
        batch1 = image_ops.rgb_to_hsv(batch0)
        batch2 = image_ops.hsv_to_rgb(batch1)
        split0 = array_ops.unstack(batch0)
        split1 = list(map(image_ops.rgb_to_hsv, split0))
        split2 = list(map(image_ops.hsv_to_rgb, split1))
        join1 = array_ops.stack(split1)
        join2 = array_ops.stack(split2)
        batch1, batch2, join1, join2 = self.evaluate(
            [batch1, batch2, join1, join2])

      # Verify that processing batch elements together is the same as separate
      self.assertAllClose(batch1, join1)
      self.assertAllClose(batch2, join2)
      self.assertAllClose(batch2, inp)

  def testRGBToHSVRoundTrip(self):
    data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    for nptype in [np.float32, np.float64]:
      rgb_np = np.array(data, dtype=nptype).reshape([2, 2, 3]) / 255.
      with self.cached_session(use_gpu=True):
        hsv = image_ops.rgb_to_hsv(rgb_np)
        rgb = image_ops.hsv_to_rgb(hsv)
        rgb_tf = self.evaluate(rgb)
      self.assertAllClose(rgb_tf, rgb_np)


class RGBToYIQTest(test_util.TensorFlowTestCase):

  def testBatch(self):
    # Build an arbitrary RGB image
    np.random.seed(7)
    batch_size = 5
    shape = (batch_size, 2, 7, 3)

    for nptype in [np.float32, np.float64]:
      inp = np.random.rand(*shape).astype(nptype)

      # Convert to YIQ and back, as a batch and individually
      with self.cached_session(use_gpu=True) as sess:
        batch0 = constant_op.constant(inp)
        batch1 = image_ops.rgb_to_yiq(batch0)
        batch2 = image_ops.yiq_to_rgb(batch1)
        split0 = array_ops.unstack(batch0)
        split1 = list(map(image_ops.rgb_to_yiq, split0))
        split2 = list(map(image_ops.yiq_to_rgb, split1))
        join1 = array_ops.stack(split1)
        join2 = array_ops.stack(split2)
        batch1, batch2, join1, join2 = self.evaluate(
            [batch1, batch2, join1, join2])

      # Verify that processing batch elements together is the same as separate
      self.assertAllClose(batch1, join1, rtol=1e-4, atol=1e-4)
      self.assertAllClose(batch2, join2, rtol=1e-4, atol=1e-4)
      self.assertAllClose(batch2, inp, rtol=1e-4, atol=1e-4)


class RGBToYUVTest(test_util.TensorFlowTestCase):

  def testBatch(self):
    # Build an arbitrary RGB image
    np.random.seed(7)
    batch_size = 5
    shape = (batch_size, 2, 7, 3)

    for nptype in [np.float32, np.float64]:
      inp = np.random.rand(*shape).astype(nptype)

      # Convert to YUV and back, as a batch and individually
      with self.cached_session(use_gpu=True) as sess:
        batch0 = constant_op.constant(inp)
        batch1 = image_ops.rgb_to_yuv(batch0)
        batch2 = image_ops.yuv_to_rgb(batch1)
        split0 = array_ops.unstack(batch0)
        split1 = list(map(image_ops.rgb_to_yuv, split0))
        split2 = list(map(image_ops.yuv_to_rgb, split1))
        join1 = array_ops.stack(split1)
        join2 = array_ops.stack(split2)
        batch1, batch2, join1, join2 = self.evaluate(
            [batch1, batch2, join1, join2])

      # Verify that processing batch elements together is the same as separate
      self.assertAllClose(batch1, join1, rtol=1e-4, atol=1e-4)
      self.assertAllClose(batch2, join2, rtol=1e-4, atol=1e-4)
      self.assertAllClose(batch2, inp, rtol=1e-4, atol=1e-4)


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

    with self.cached_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.rgb_to_grayscale(x_tf)
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)

  def testBasicRGBToGrayscale(self):
    # 4-D input with batch dimension.
    x_np = np.array(
        [[1, 2, 3], [4, 10, 1]], dtype=np.uint8).reshape([1, 1, 2, 3])
    self._TestRGBToGrayscale(x_np)

    # 3-D input with no batch dimension.
    x_np = np.array([[1, 2, 3], [4, 10, 1]], dtype=np.uint8).reshape([1, 2, 3])
    self._TestRGBToGrayscale(x_np)

  def testBasicGrayscaleToRGB(self):
    # 4-D input with batch dimension.
    x_np = np.array([[1, 2]], dtype=np.uint8).reshape([1, 1, 2, 1])
    y_np = np.array(
        [[1, 1, 1], [2, 2, 2]], dtype=np.uint8).reshape([1, 1, 2, 3])

    with self.cached_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.grayscale_to_rgb(x_tf)
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)

    # 3-D input with no batch dimension.
    x_np = np.array([[1, 2]], dtype=np.uint8).reshape([1, 2, 1])
    y_np = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.uint8).reshape([1, 2, 3])

    with self.cached_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.grayscale_to_rgb(x_tf)
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)

  def testGrayscaleToRGBInputValidation(self):
    # tests whether the grayscale_to_rgb function raises
    # an exception if the input images' last dimension is
    # not of size 1, i.e. the images have shape
    # [batch size, height, width] or [height, width]

    # tests if an exception is raised if a three dimensional
    # input is used, i.e. the images have shape [batch size, height, width]
    with self.cached_session(use_gpu=True):
      # 3-D input with batch dimension.
      x_np = np.array([[1, 2]], dtype=np.uint8).reshape([1, 1, 2])

      x_tf = constant_op.constant(x_np, shape=x_np.shape)

      # this is the error message we expect the function to raise
      err_msg = "Last dimension of a grayscale image should be size 1"
      with self.assertRaisesRegexp(ValueError, err_msg):
        image_ops.grayscale_to_rgb(x_tf)

    # tests if an exception is raised if a two dimensional
    # input is used, i.e. the images have shape [height, width]
    with self.cached_session(use_gpu=True):
      # 1-D input without batch dimension.
      x_np = np.array([[1, 2]], dtype=np.uint8).reshape([2])

      x_tf = constant_op.constant(x_np, shape=x_np.shape)

      # this is the error message we expect the function to raise
      err_msg = "must be at least two-dimensional"
      with self.assertRaisesRegexp(ValueError, err_msg):
        image_ops.grayscale_to_rgb(x_tf)

  @test_util.run_deprecated_v1
  def testShapeInference(self):
    # Shape inference works and produces expected output where possible
    rgb_shape = [7, None, 19, 3]
    gray_shape = rgb_shape[:-1] + [1]
    with self.cached_session(use_gpu=True):
      rgb_tf = array_ops.placeholder(dtypes.uint8, shape=rgb_shape)
      gray = image_ops.rgb_to_grayscale(rgb_tf)
      self.assertEqual(gray_shape, gray.get_shape().as_list())

    with self.cached_session(use_gpu=True):
      gray_tf = array_ops.placeholder(dtypes.uint8, shape=gray_shape)
      rgb = image_ops.grayscale_to_rgb(gray_tf)
      self.assertEqual(rgb_shape, rgb.get_shape().as_list())

    # Shape inference does not break for unknown shapes
    with self.cached_session(use_gpu=True):
      rgb_tf_unknown = array_ops.placeholder(dtypes.uint8)
      gray_unknown = image_ops.rgb_to_grayscale(rgb_tf_unknown)
      self.assertFalse(gray_unknown.get_shape())

    with self.cached_session(use_gpu=True):
      gray_tf_unknown = array_ops.placeholder(dtypes.uint8)
      rgb_unknown = image_ops.grayscale_to_rgb(gray_tf_unknown)
      self.assertFalse(rgb_unknown.get_shape())


class AdjustGamma(test_util.TensorFlowTestCase):

  @test_util.run_deprecated_v1
  def test_adjust_gamma_less_zero_float32(self):
    """White image should be returned for gamma equal to zero"""
    with self.cached_session():
      x_data = np.random.uniform(0, 1.0, (8, 8))
      x_np = np.array(x_data, dtype=np.float32)

      x = constant_op.constant(x_np, shape=x_np.shape)

      err_msg = "Gamma should be a non-negative real number"
      with self.assertRaisesRegexp(ValueError, err_msg):
        image_ops.adjust_gamma(x, gamma=-1)

  @test_util.run_deprecated_v1
  def test_adjust_gamma_less_zero_uint8(self):
    """White image should be returned for gamma equal to zero"""
    with self.cached_session():
      x_data = np.random.uniform(0, 255, (8, 8))
      x_np = np.array(x_data, dtype=np.uint8)

      x = constant_op.constant(x_np, shape=x_np.shape)

      err_msg = "Gamma should be a non-negative real number"
      with self.assertRaisesRegexp(ValueError, err_msg):
        image_ops.adjust_gamma(x, gamma=-1)

  @test_util.run_deprecated_v1
  def test_adjust_gamma_less_zero_tensor(self):
    """White image should be returned for gamma equal to zero"""
    with self.cached_session():
      x_data = np.random.uniform(0, 1.0, (8, 8))
      x_np = np.array(x_data, dtype=np.float32)

      x = constant_op.constant(x_np, shape=x_np.shape)
      y = constant_op.constant(-1.0, dtype=dtypes.float32)

      image = image_ops.adjust_gamma(x, gamma=y)

      err_msg = "Gamma should be a non-negative real number"
      with self.assertRaisesRegexp(errors.InvalidArgumentError, err_msg):
        self.evaluate(image)

  def _test_adjust_gamma_uint8(self, gamma):
    """Verifying the output with expected results for gamma

    correction for uint8 images
    """
    with self.cached_session():
      x_np = np.random.uniform(0, 255, (8, 8)).astype(np.uint8)
      x = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.adjust_gamma(x, gamma=gamma)
      y_tf = np.trunc(y.eval())

      # calculate gamma correction using numpy
      # firstly, transform uint8 to float representation
      # then perform correction
      y_np = np.power(x_np / 255.0, gamma)
      # convert correct numpy image back to uint8 type
      y_np = np.trunc(np.clip(y_np * 255.5, 0, 255.0))

      self.assertAllClose(y_tf, y_np, 1e-6)

  def _test_adjust_gamma_float32(self, gamma):
    """Verifying the output with expected results for gamma

    correction for float32 images
    """
    with self.cached_session():
      x_np = np.random.uniform(0, 1.0, (8, 8))
      x = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.adjust_gamma(x, gamma=gamma)
      y_tf = y.eval()

      y_np = np.clip(np.power(x_np, gamma), 0, 1.0)

      self.assertAllClose(y_tf, y_np, 1e-6)

  @test_util.run_deprecated_v1
  def test_adjust_gamma_one_float32(self):
    """Same image should be returned for gamma equal to one"""
    self._test_adjust_gamma_float32(1.0)

  @test_util.run_deprecated_v1
  def test_adjust_gamma_one_uint8(self):
    self._test_adjust_gamma_uint8(1.0)

  @test_util.run_deprecated_v1
  def test_adjust_gamma_zero_uint8(self):
    """White image should be returned for gamma equal

    to zero for uint8 images
    """
    self._test_adjust_gamma_uint8(gamma=0.0)

  @test_util.run_deprecated_v1
  def test_adjust_gamma_less_one_uint8(self):
    """Verifying the output with expected results for gamma

    correction with gamma equal to half for uint8 images
    """
    self._test_adjust_gamma_uint8(gamma=0.5)

  @test_util.run_deprecated_v1
  def test_adjust_gamma_greater_one_uint8(self):
    """Verifying the output with expected results for gamma

    correction for uint8 images
    """
    self._test_adjust_gamma_uint8(gamma=1.0)

  @test_util.run_deprecated_v1
  def test_adjust_gamma_less_one_float32(self):
    """Verifying the output with expected results for gamma

    correction with gamma equal to half for float32 images
    """
    self._test_adjust_gamma_float32(0.5)

  @test_util.run_deprecated_v1
  def test_adjust_gamma_greater_one_float32(self):
    """Verifying the output with expected results for gamma

    correction with gamma equal to two for float32 images
    """
    self._test_adjust_gamma_float32(1.0)

  @test_util.run_deprecated_v1
  def test_adjust_gamma_zero_float32(self):
    """White image should be returned for gamma equal

    to zero for float32 images
    """
    self._test_adjust_gamma_float32(0.0)


class AdjustHueTest(test_util.TensorFlowTestCase):

  def testAdjustNegativeHue(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    delta = -0.25
    y_data = [0, 13, 1, 54, 226, 59, 8, 234, 150, 255, 39, 1]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.cached_session(use_gpu=True):
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.adjust_hue(x, delta)
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)

  def testAdjustPositiveHue(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    delta = 0.25
    y_data = [13, 0, 11, 226, 54, 221, 234, 8, 92, 1, 217, 255]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.cached_session(use_gpu=True):
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.adjust_hue(x, delta)
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)

  def testBatchAdjustHue(self):
    x_shape = [2, 1, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    delta = 0.25
    y_data = [13, 0, 11, 226, 54, 221, 234, 8, 92, 1, 217, 255]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.cached_session(use_gpu=True):
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.adjust_hue(x, delta)
      y_tf = self.evaluate(y)
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
    with self.cached_session(use_gpu=True):
      x = constant_op.constant(x_np)
      y = image_ops.adjust_hue(x, delta_h)
      y_tf = self.evaluate(y)
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
        self.assertAllClose(y_tf, y_np, rtol=2e-5, atol=1e-5)

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


class FlipImageBenchmark(test.Benchmark):

  def _benchmarkFlipLeftRight(self, device, cpu_count):
    image_shape = [299, 299, 3]
    warmup_rounds = 100
    benchmark_rounds = 1000
    config = config_pb2.ConfigProto()
    if cpu_count is not None:
      config.inter_op_parallelism_threads = 1
      config.intra_op_parallelism_threads = cpu_count
    with session.Session("", graph=ops.Graph(), config=config) as sess:
      with ops.device(device):
        inputs = variables.Variable(
            random_ops.random_uniform(image_shape, dtype=dtypes.float32) * 255,
            trainable=False,
            dtype=dtypes.float32)
        run_op = image_ops.flip_left_right(inputs)
        self.evaluate(variables.global_variables_initializer())
        for i in xrange(warmup_rounds + benchmark_rounds):
          if i == warmup_rounds:
            start = time.time()
          self.evaluate(run_op)
    end = time.time()
    step_time = (end - start) / benchmark_rounds
    tag = device + "_%s" % (cpu_count if cpu_count is not None else "_all")
    print("benchmarkFlipLeftRight_299_299_3_%s step_time: %.2f us" %
          (tag, step_time * 1e6))
    self.report_benchmark(
        name="benchmarkFlipLeftRight_299_299_3_%s" % (tag),
        iters=benchmark_rounds,
        wall_time=step_time)

  def _benchmarkRandomFlipLeftRight(self, device, cpu_count):
    image_shape = [299, 299, 3]
    warmup_rounds = 100
    benchmark_rounds = 1000
    config = config_pb2.ConfigProto()
    if cpu_count is not None:
      config.inter_op_parallelism_threads = 1
      config.intra_op_parallelism_threads = cpu_count
    with session.Session("", graph=ops.Graph(), config=config) as sess:
      with ops.device(device):
        inputs = variables.Variable(
            random_ops.random_uniform(image_shape, dtype=dtypes.float32) * 255,
            trainable=False,
            dtype=dtypes.float32)
        run_op = image_ops.random_flip_left_right(inputs)
        self.evaluate(variables.global_variables_initializer())
        for i in xrange(warmup_rounds + benchmark_rounds):
          if i == warmup_rounds:
            start = time.time()
          self.evaluate(run_op)
    end = time.time()
    step_time = (end - start) / benchmark_rounds
    tag = device + "_%s" % (cpu_count if cpu_count is not None else "_all")
    print("benchmarkRandomFlipLeftRight_299_299_3_%s step_time: %.2f us" %
          (tag, step_time * 1e6))
    self.report_benchmark(
        name="benchmarkRandomFlipLeftRight_299_299_3_%s" % (tag),
        iters=benchmark_rounds,
        wall_time=step_time)

  def _benchmarkBatchedRandomFlipLeftRight(self, device, cpu_count):
    image_shape = [16, 299, 299, 3]
    warmup_rounds = 100
    benchmark_rounds = 1000
    config = config_pb2.ConfigProto()
    if cpu_count is not None:
      config.inter_op_parallelism_threads = 1
      config.intra_op_parallelism_threads = cpu_count
    with session.Session("", graph=ops.Graph(), config=config) as sess:
      with ops.device(device):
        inputs = variables.Variable(
            random_ops.random_uniform(image_shape, dtype=dtypes.float32) * 255,
            trainable=False,
            dtype=dtypes.float32)
        run_op = image_ops.random_flip_left_right(inputs)
        self.evaluate(variables.global_variables_initializer())
        for i in xrange(warmup_rounds + benchmark_rounds):
          if i == warmup_rounds:
            start = time.time()
          self.evaluate(run_op)
    end = time.time()
    step_time = (end - start) / benchmark_rounds
    tag = device + "_%s" % (cpu_count if cpu_count is not None else "_all")
    print("benchmarkBatchedRandomFlipLeftRight_16_299_299_3_%s step_time: "
          "%.2f us" %
          (tag, step_time * 1e6))
    self.report_benchmark(
        name="benchmarkBatchedRandomFlipLeftRight_16_299_299_3_%s" % (tag),
        iters=benchmark_rounds,
        wall_time=step_time)

  def benchmarkFlipLeftRightCpu1(self):
    self._benchmarkFlipLeftRight("/cpu:0", 1)

  def benchmarkFlipLeftRightCpuAll(self):
    self._benchmarkFlipLeftRight("/cpu:0", None)

  def benchmarkFlipLeftRightGpu(self):
    self._benchmarkFlipLeftRight(test.gpu_device_name(), None)

  def benchmarkRandomFlipLeftRightCpu1(self):
    self._benchmarkRandomFlipLeftRight("/cpu:0", 1)

  def benchmarkRandomFlipLeftRightCpuAll(self):
    self._benchmarkRandomFlipLeftRight("/cpu:0", None)

  def benchmarkRandomFlipLeftRightGpu(self):
    self._benchmarkRandomFlipLeftRight(test.gpu_device_name(), None)

  def benchmarkBatchedRandomFlipLeftRightCpu1(self):
    self._benchmarkBatchedRandomFlipLeftRight("/cpu:0", 1)

  def benchmarkBatchedRandomFlipLeftRightCpuAll(self):
    self._benchmarkBatchedRandomFlipLeftRight("/cpu:0", None)

  def benchmarkBatchedRandomFlipLeftRightGpu(self):
    self._benchmarkBatchedRandomFlipLeftRight(test.gpu_device_name(), None)


class AdjustHueBenchmark(test.Benchmark):

  def _benchmarkAdjustHue(self, device, cpu_count):
    image_shape = [299, 299, 3]
    warmup_rounds = 100
    benchmark_rounds = 1000
    config = config_pb2.ConfigProto()
    if cpu_count is not None:
      config.inter_op_parallelism_threads = 1
      config.intra_op_parallelism_threads = cpu_count
    with self.benchmark_session(config=config, device=device) as sess:
      inputs = variables.Variable(
          random_ops.random_uniform(image_shape, dtype=dtypes.float32) * 255,
          trainable=False,
          dtype=dtypes.float32)
      delta = constant_op.constant(0.1, dtype=dtypes.float32)
      outputs = image_ops.adjust_hue(inputs, delta)
      run_op = control_flow_ops.group(outputs)
      self.evaluate(variables.global_variables_initializer())
      for i in xrange(warmup_rounds + benchmark_rounds):
        if i == warmup_rounds:
          start = time.time()
        self.evaluate(run_op)
    end = time.time()
    step_time = (end - start) / benchmark_rounds
    tag = device + "_%s" % (cpu_count if cpu_count is not None else "_all")
    print("benchmarkAdjustHue_299_299_3_%s step_time: %.2f us" %
          (tag, step_time * 1e6))
    self.report_benchmark(
        name="benchmarkAdjustHue_299_299_3_%s" % (tag),
        iters=benchmark_rounds,
        wall_time=step_time)

  def benchmarkAdjustHueCpu1(self):
    self._benchmarkAdjustHue("/cpu:0", 1)

  def benchmarkAdjustHueCpuAll(self):
    self._benchmarkAdjustHue("/cpu:0", None)

  def benchmarkAdjustHueGpu(self):
    self._benchmarkAdjustHue(test.gpu_device_name(), None)


class AdjustSaturationBenchmark(test.Benchmark):

  def _benchmarkAdjustSaturation(self, device, cpu_count):
    image_shape = [299, 299, 3]
    warmup_rounds = 100
    benchmark_rounds = 1000
    config = config_pb2.ConfigProto()
    if cpu_count is not None:
      config.inter_op_parallelism_threads = 1
      config.intra_op_parallelism_threads = cpu_count
    with self.benchmark_session(config=config, device=device) as sess:
      inputs = variables.Variable(
          random_ops.random_uniform(image_shape, dtype=dtypes.float32) * 255,
          trainable=False,
          dtype=dtypes.float32)
      delta = constant_op.constant(0.1, dtype=dtypes.float32)
      outputs = image_ops.adjust_saturation(inputs, delta)
      run_op = control_flow_ops.group(outputs)
      self.evaluate(variables.global_variables_initializer())
      for _ in xrange(warmup_rounds):
        self.evaluate(run_op)
      start = time.time()
      for _ in xrange(benchmark_rounds):
        self.evaluate(run_op)
    end = time.time()
    step_time = (end - start) / benchmark_rounds
    tag = device + "_%s" % (cpu_count if cpu_count is not None else "_all")
    print("benchmarkAdjustSaturation_299_299_3_%s step_time: %.2f us" %
          (tag, step_time * 1e6))
    self.report_benchmark(
        name="benchmarkAdjustSaturation_299_299_3_%s" % (tag),
        iters=benchmark_rounds,
        wall_time=step_time)

  def benchmarkAdjustSaturationCpu1(self):
    self._benchmarkAdjustSaturation("/cpu:0", 1)

  def benchmarkAdjustSaturationCpuAll(self):
    self._benchmarkAdjustSaturation("/cpu:0", None)

  def benchmarkAdjustSaturationGpu(self):
    self._benchmarkAdjustSaturation(test.gpu_device_name(), None)


class ResizeBilinearBenchmark(test.Benchmark):

  def _benchmarkResize(self, image_size, num_channels):
    batch_size = 1
    num_ops = 1000
    img = variables.Variable(
        random_ops.random_normal(
            [batch_size, image_size[0], image_size[1], num_channels]),
        name="img")

    deps = []
    for _ in xrange(num_ops):
      with ops.control_dependencies(deps):
        resize_op = image_ops.resize_bilinear(
            img, [299, 299], align_corners=False)
        deps = [resize_op]
      benchmark_op = control_flow_ops.group(*deps)

    with self.benchmark_session() as sess:
      self.evaluate(variables.global_variables_initializer())
      results = self.run_op_benchmark(
          sess,
          benchmark_op,
          name=("resize_bilinear_%s_%s_%s" % (image_size[0], image_size[1],
                                              num_channels)))
      print("%s   : %.2f ms/img" %
            (results["name"],
             1000 * results["wall_time"] / (batch_size * num_ops)))

  def benchmarkSimilar3Channel(self):
    self._benchmarkResize((183, 229), 3)

  def benchmarkScaleUp3Channel(self):
    self._benchmarkResize((141, 186), 3)

  def benchmarkScaleDown3Channel(self):
    self._benchmarkResize((749, 603), 3)

  def benchmarkSimilar1Channel(self):
    self._benchmarkResize((183, 229), 1)

  def benchmarkScaleUp1Channel(self):
    self._benchmarkResize((141, 186), 1)

  def benchmarkScaleDown1Channel(self):
    self._benchmarkResize((749, 603), 1)


class ResizeBicubicBenchmark(test.Benchmark):

  def _benchmarkResize(self, image_size, num_channels):
    batch_size = 1
    num_ops = 1000
    img = variables.Variable(
        random_ops.random_normal(
            [batch_size, image_size[0], image_size[1], num_channels]),
        name="img")

    deps = []
    for _ in xrange(num_ops):
      with ops.control_dependencies(deps):
        resize_op = image_ops.resize_bicubic(
            img, [299, 299], align_corners=False)
        deps = [resize_op]
      benchmark_op = control_flow_ops.group(*deps)

    with self.benchmark_session() as sess:
      self.evaluate(variables.global_variables_initializer())
      results = self.run_op_benchmark(
          sess,
          benchmark_op,
          min_iters=20,
          name=("resize_bicubic_%s_%s_%s" % (image_size[0], image_size[1],
                                             num_channels)))
      print("%s   : %.2f ms/img" %
            (results["name"],
             1000 * results["wall_time"] / (batch_size * num_ops)))

  def benchmarkSimilar3Channel(self):
    self._benchmarkResize((183, 229), 3)

  def benchmarkScaleUp3Channel(self):
    self._benchmarkResize((141, 186), 3)

  def benchmarkScaleDown3Channel(self):
    self._benchmarkResize((749, 603), 3)

  def benchmarkSimilar1Channel(self):
    self._benchmarkResize((183, 229), 1)

  def benchmarkScaleUp1Channel(self):
    self._benchmarkResize((141, 186), 1)

  def benchmarkScaleDown1Channel(self):
    self._benchmarkResize((749, 603), 1)

  def benchmarkSimilar4Channel(self):
    self._benchmarkResize((183, 229), 4)

  def benchmarkScaleUp4Channel(self):
    self._benchmarkResize((141, 186), 4)

  def benchmarkScaleDown4Channel(self):
    self._benchmarkResize((749, 603), 4)


class ResizeAreaBenchmark(test.Benchmark):

  def _benchmarkResize(self, image_size, num_channels):
    batch_size = 1
    num_ops = 1000
    img = variables.Variable(
        random_ops.random_normal(
            [batch_size, image_size[0], image_size[1], num_channels]),
        name="img")

    deps = []
    for _ in xrange(num_ops):
      with ops.control_dependencies(deps):
        resize_op = image_ops.resize_area(img, [299, 299], align_corners=False)
        deps = [resize_op]
      benchmark_op = control_flow_ops.group(*deps)

    with self.benchmark_session() as sess:
      self.evaluate(variables.global_variables_initializer())
      results = self.run_op_benchmark(
          sess,
          benchmark_op,
          name=("resize_area_%s_%s_%s" % (image_size[0], image_size[1],
                                          num_channels)))
      print("%s   : %.2f ms/img" %
            (results["name"],
             1000 * results["wall_time"] / (batch_size * num_ops)))

  def benchmarkSimilar3Channel(self):
    self._benchmarkResize((183, 229), 3)

  def benchmarkScaleUp3Channel(self):
    self._benchmarkResize((141, 186), 3)

  def benchmarkScaleDown3Channel(self):
    self._benchmarkResize((749, 603), 3)

  def benchmarkSimilar1Channel(self):
    self._benchmarkResize((183, 229), 1)

  def benchmarkScaleUp1Channel(self):
    self._benchmarkResize((141, 186), 1)

  def benchmarkScaleDown1Channel(self):
    self._benchmarkResize((749, 603), 1)


class AdjustSaturationTest(test_util.TensorFlowTestCase):

  def testHalfSaturation(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    saturation_factor = 0.5
    y_data = [6, 9, 13, 140, 180, 226, 135, 121, 234, 172, 255, 128]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.cached_session(use_gpu=True):
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.adjust_saturation(x, saturation_factor)
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)

  def testTwiceSaturation(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    saturation_factor = 2.0
    y_data = [0, 5, 13, 0, 106, 226, 30, 0, 234, 89, 255, 0]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.cached_session(use_gpu=True):
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.adjust_saturation(x, saturation_factor)
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)

  def testBatchSaturation(self):
    x_shape = [2, 1, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    saturation_factor = 0.5
    y_data = [6, 9, 13, 140, 180, 226, 135, 121, 234, 172, 255, 128]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.cached_session(use_gpu=True):
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.adjust_saturation(x, saturation_factor)
      y_tf = self.evaluate(y)
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

  @test_util.run_deprecated_v1
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
    with self.cached_session(use_gpu=True):
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
          y_fused = image_ops.adjust_saturation(x_np, scale).eval()
          self.assertAllClose(y_fused, y_baseline, rtol=2e-5, atol=1e-5)


class FlipTransposeRotateTest(test_util.TensorFlowTestCase):

  def testInvolutionLeftRight(self):
    x_np = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.uint8).reshape([2, 3, 1])
    with self.cached_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.flip_left_right(image_ops.flip_left_right(x_tf))
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, x_np)

  def testInvolutionLeftRightWithBatch(self):
    x_np = np.array(
        [[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]],
        dtype=np.uint8).reshape([2, 2, 3, 1])
    with self.cached_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.flip_left_right(image_ops.flip_left_right(x_tf))
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, x_np)

  @test_util.run_deprecated_v1
  def testLeftRight(self):
    x_np = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.uint8).reshape([2, 3, 1])
    y_np = np.array([[3, 2, 1], [3, 2, 1]], dtype=np.uint8).reshape([2, 3, 1])

    with self.cached_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.flip_left_right(x_tf)
      self.assertTrue(y.op.name.startswith("flip_left_right"))
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)

  def testLeftRightWithBatch(self):
    x_np = np.array(
        [[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]],
        dtype=np.uint8).reshape([2, 2, 3, 1])
    y_np = np.array(
        [[[3, 2, 1], [3, 2, 1]], [[3, 2, 1], [3, 2, 1]]],
        dtype=np.uint8).reshape([2, 2, 3, 1])

    with self.cached_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.flip_left_right(x_tf)
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)

  @test_util.run_deprecated_v1
  def testRandomFlipLeftRight(self):
    x_np = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.uint8).reshape([2, 3, 1])
    y_np = np.array([[3, 2, 1], [3, 2, 1]], dtype=np.uint8).reshape([2, 3, 1])
    seed = 42

    with self.cached_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.random_flip_left_right(x_tf, seed=seed)
      self.assertTrue(y.op.name.startswith("random_flip_left_right"))

      count_flipped = 0
      count_unflipped = 0
      for _ in range(100):
        y_tf = self.evaluate(y)
        if y_tf[0][0] == 1:
          self.assertAllEqual(y_tf, x_np)
          count_unflipped += 1
        else:
          self.assertAllEqual(y_tf, y_np)
          count_flipped += 1

      # 100 trials
      # Mean: 50
      # Std Dev: ~5
      # Six Sigma: 50 - (5 * 6) = 20
      self.assertGreaterEqual(count_flipped, 20)
      self.assertGreaterEqual(count_unflipped, 20)

  @test_util.run_deprecated_v1
  def testRandomFlipLeftRightWithBatch(self):
    batch_size = 16
    seed = 42

    # create single item of test data
    x_np_raw = np.array(
        [[1, 2, 3], [1, 2, 3]], dtype=np.uint8
    ).reshape([1, 2, 3, 1])
    y_np_raw = np.array(
        [[3, 2, 1], [3, 2, 1]], dtype=np.uint8
    ).reshape([1, 2, 3, 1])

    # create batched test data
    x_np = np.vstack([x_np_raw for _ in range(batch_size)])
    y_np = np.vstack([y_np_raw for _ in range(batch_size)])

    with self.cached_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.random_flip_left_right(x_tf, seed=seed)
      self.assertTrue(y.op.name.startswith("random_flip_left_right"))

      count_flipped = 0
      count_unflipped = 0
      for _ in range(100):
        y_tf = self.evaluate(y)

        # check every element of the batch
        for i in range(batch_size):
          if y_tf[i][0][0] == 1:
            self.assertAllEqual(y_tf[i], x_np[i])
            count_unflipped += 1
          else:
            self.assertAllEqual(y_tf[i], y_np[i])
            count_flipped += 1

      # 100 trials, each containing batch_size elements
      # Mean: 50 * batch_size
      # Std Dev: ~5 * sqrt(batch_size)
      # Six Sigma: 50 * batch_size - (5 * 6 * sqrt(batch_size))
      #          = 50 * batch_size - 30 * sqrt(batch_size) = 800 - 30 * 4 = 680
      six_sigma = 50 * batch_size - 30 * np.sqrt(batch_size)
      self.assertGreaterEqual(count_flipped, six_sigma)
      self.assertGreaterEqual(count_unflipped, six_sigma)

  def testInvolutionUpDown(self):
    x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8).reshape([2, 3, 1])

    with self.cached_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.flip_up_down(image_ops.flip_up_down(x_tf))
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, x_np)

  def testInvolutionUpDownWithBatch(self):
    x_np = np.array(
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
        dtype=np.uint8).reshape([2, 2, 3, 1])

    with self.cached_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.flip_up_down(image_ops.flip_up_down(x_tf))
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, x_np)

  @test_util.run_deprecated_v1
  def testUpDown(self):
    x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8).reshape([2, 3, 1])
    y_np = np.array([[4, 5, 6], [1, 2, 3]], dtype=np.uint8).reshape([2, 3, 1])

    with self.cached_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.flip_up_down(x_tf)
      self.assertTrue(y.op.name.startswith("flip_up_down"))
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)

  def testUpDownWithBatch(self):
    x_np = np.array(
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
        dtype=np.uint8).reshape([2, 2, 3, 1])
    y_np = np.array(
        [[[4, 5, 6], [1, 2, 3]], [[10, 11, 12], [7, 8, 9]]],
        dtype=np.uint8).reshape([2, 2, 3, 1])

    with self.cached_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.flip_up_down(x_tf)
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)

  @test_util.run_deprecated_v1
  def testRandomFlipUpDown(self):
    x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8).reshape([2, 3, 1])
    y_np = np.array([[4, 5, 6], [1, 2, 3]], dtype=np.uint8).reshape([2, 3, 1])

    seed = 42

    with self.cached_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.random_flip_up_down(x_tf, seed=seed)
      self.assertTrue(y.op.name.startswith("random_flip_up_down"))
      count_flipped = 0
      count_unflipped = 0
      for _ in range(100):
        y_tf = self.evaluate(y)
        if y_tf[0][0] == 1:
          self.assertAllEqual(y_tf, x_np)
          count_unflipped += 1
        else:
          self.assertAllEqual(y_tf, y_np)
          count_flipped += 1

      # 100 trials
      # Mean: 50
      # Std Dev: ~5
      # Six Sigma: 50 - (5 * 6) = 20
      self.assertGreaterEqual(count_flipped, 20)
      self.assertGreaterEqual(count_unflipped, 20)

  @test_util.run_deprecated_v1
  def testRandomFlipUpDownWithBatch(self):
    batch_size = 16
    seed = 42

    # create single item of test data
    x_np_raw = np.array(
        [[1, 2, 3], [4, 5, 6]], dtype=np.uint8
    ).reshape([1, 2, 3, 1])
    y_np_raw = np.array(
        [[4, 5, 6], [1, 2, 3]], dtype=np.uint8
    ).reshape([1, 2, 3, 1])

    # create batched test data
    x_np = np.vstack([x_np_raw for _ in range(batch_size)])
    y_np = np.vstack([y_np_raw for _ in range(batch_size)])

    with self.cached_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.random_flip_up_down(x_tf, seed=seed)
      self.assertTrue(y.op.name.startswith("random_flip_up_down"))

      count_flipped = 0
      count_unflipped = 0
      for _ in range(100):
        y_tf = self.evaluate(y)

        # check every element of the batch
        for i in range(batch_size):
          if y_tf[i][0][0] == 1:
            self.assertAllEqual(y_tf[i], x_np[i])
            count_unflipped += 1
          else:
            self.assertAllEqual(y_tf[i], y_np[i])
            count_flipped += 1

      # 100 trials, each containing batch_size elements
      # Mean: 50 * batch_size
      # Std Dev: ~5 * sqrt(batch_size)
      # Six Sigma: 50 * batch_size - (5 * 6 * sqrt(batch_size))
      #          = 50 * batch_size - 30 * sqrt(batch_size) = 800 - 30 * 4 = 680
      six_sigma = 50 * batch_size - 30 * np.sqrt(batch_size)
      self.assertGreaterEqual(count_flipped, six_sigma)
      self.assertGreaterEqual(count_unflipped, six_sigma)

  def testInvolutionTranspose(self):
    x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8).reshape([2, 3, 1])

    with self.cached_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.transpose(image_ops.transpose(x_tf))
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, x_np)

  def testInvolutionTransposeWithBatch(self):
    x_np = np.array(
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
        dtype=np.uint8).reshape([2, 2, 3, 1])

    with self.cached_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.transpose(image_ops.transpose(x_tf))
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, x_np)

  @test_util.run_deprecated_v1
  def testTranspose(self):
    x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8).reshape([2, 3, 1])
    y_np = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.uint8).reshape([3, 2, 1])

    with self.cached_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.transpose(x_tf)
      self.assertTrue(y.op.name.startswith("transpose"))
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)

  def testTransposeWithBatch(self):
    x_np = np.array(
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
        dtype=np.uint8).reshape([2, 2, 3, 1])

    y_np = np.array(
        [[[1, 4], [2, 5], [3, 6]], [[7, 10], [8, 11], [9, 12]]],
        dtype=np.uint8).reshape([2, 3, 2, 1])

    with self.cached_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.transpose(x_tf)
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)

  @test_util.run_deprecated_v1
  def testPartialShapes(self):
    p_unknown_rank = array_ops.placeholder(dtypes.uint8)
    p_unknown_dims_3 = array_ops.placeholder(
        dtypes.uint8, shape=[None, None, None])
    p_unknown_dims_4 = array_ops.placeholder(
        dtypes.uint8, shape=[None, None, None, None])
    p_unknown_width = array_ops.placeholder(dtypes.uint8, shape=[64, None, 3])
    p_unknown_batch = array_ops.placeholder(
        dtypes.uint8, shape=[None, 64, 64, 3])
    p_wrong_rank = array_ops.placeholder(dtypes.uint8, shape=[None, None])
    p_zero_dim = array_ops.placeholder(dtypes.uint8, shape=[64, 0, 3])

    #Ops that support 3D input
    for op in [
        image_ops.flip_left_right, image_ops.flip_up_down,
        image_ops.random_flip_left_right, image_ops.random_flip_up_down,
        image_ops.transpose, image_ops.rot90
    ]:
      transformed_unknown_rank = op(p_unknown_rank)
      self.assertEqual(3, transformed_unknown_rank.get_shape().ndims)
      transformed_unknown_dims_3 = op(p_unknown_dims_3)
      self.assertEqual(3, transformed_unknown_dims_3.get_shape().ndims)
      transformed_unknown_width = op(p_unknown_width)
      self.assertEqual(3, transformed_unknown_width.get_shape().ndims)

      with self.assertRaisesRegexp(ValueError, "must be > 0"):
        op(p_zero_dim)

    #Ops that support 4D input
    for op in [
        image_ops.flip_left_right, image_ops.flip_up_down,
        image_ops.random_flip_left_right, image_ops.random_flip_up_down,
        image_ops.transpose, image_ops.rot90
    ]:
      transformed_unknown_dims_4 = op(p_unknown_dims_4)
      self.assertEqual(4, transformed_unknown_dims_4.get_shape().ndims)
      transformed_unknown_batch = op(p_unknown_batch)
      self.assertEqual(4, transformed_unknown_batch.get_shape().ndims)
      with self.assertRaisesRegexp(ValueError,
                                   "must be at least three-dimensional"):
        op(p_wrong_rank)

  def testRot90GroupOrder(self):
    image = np.arange(24, dtype=np.uint8).reshape([2, 4, 3])
    with self.cached_session(use_gpu=True):
      rotated = image
      for _ in xrange(4):
        rotated = image_ops.rot90(rotated)
      self.assertAllEqual(image, self.evaluate(rotated))

  def testRot90GroupOrderWithBatch(self):
    image = np.arange(48, dtype=np.uint8).reshape([2, 2, 4, 3])
    with self.cached_session(use_gpu=True):
      rotated = image
      for _ in xrange(4):
        rotated = image_ops.rot90(rotated)
      self.assertAllEqual(image, self.evaluate(rotated))

  @test_util.run_deprecated_v1
  def testRot90NumpyEquivalence(self):
    image = np.arange(24, dtype=np.uint8).reshape([2, 4, 3])
    with self.cached_session(use_gpu=True):
      k_placeholder = array_ops.placeholder(dtypes.int32, shape=[])
      y_tf = image_ops.rot90(image, k_placeholder)
      for k in xrange(4):
        y_np = np.rot90(image, k=k)
        self.assertAllEqual(y_np, y_tf.eval({k_placeholder: k}))

  @test_util.run_deprecated_v1
  def testRot90NumpyEquivalenceWithBatch(self):
    image = np.arange(48, dtype=np.uint8).reshape([2, 2, 4, 3])
    with self.cached_session(use_gpu=True):
      k_placeholder = array_ops.placeholder(dtypes.int32, shape=[])
      y_tf = image_ops.rot90(image, k_placeholder)
      for k in xrange(4):
        y_np = np.rot90(image, k=k, axes=(1, 2))
        self.assertAllEqual(y_np, y_tf.eval({k_placeholder: k}))

class AdjustContrastTest(test_util.TensorFlowTestCase):

  def _testContrast(self, x_np, y_np, contrast_factor):
    with self.cached_session(use_gpu=True):
      x = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.adjust_contrast(x, contrast_factor)
      y_tf = self.evaluate(y)
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

    y_data = [
        -45.25, -90.75, -92.5, 62.75, 169.25, 333.5, 28.75, -84.75, 349.5,
        134.75, 409.25, -116.5
    ]
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

  def _adjustContrastNp(self, x_np, contrast_factor):
    mean = np.mean(x_np, (1, 2), keepdims=True)
    y_np = mean + contrast_factor * (x_np - mean)
    return y_np

  def _adjustContrastTf(self, x_np, contrast_factor):
    with self.cached_session(use_gpu=True):
      x = constant_op.constant(x_np)
      y = image_ops.adjust_contrast(x, contrast_factor)
      y_tf = self.evaluate(y)
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

  @test_util.run_deprecated_v1
  def testContrastFactorShape(self):
    x_shape = [1, 2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
    with self.assertRaisesRegexp(
        ValueError, 'Shape must be rank 0 but is rank 1'):
      image_ops.adjust_contrast(x_np, [2.0])


class AdjustBrightnessTest(test_util.TensorFlowTestCase):

  def _testBrightness(self, x_np, y_np, delta, tol=1e-6):
    with self.cached_session(use_gpu=True):
      x = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.adjust_brightness(x, delta)
      y_tf = self.evaluate(y)
      self.assertAllClose(y_tf, y_np, tol)

  def testPositiveDeltaUint8(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    y_data = [10, 15, 23, 64, 145, 236, 47, 18, 244, 100, 255, 11]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    self._testBrightness(x_np, y_np, delta=10. / 255.)

  def testPositiveDeltaFloat32(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.float32).reshape(x_shape) / 255.

    y_data = [10, 15, 23, 64, 145, 236, 47, 18, 244, 100, 265, 11]
    y_np = np.array(y_data, dtype=np.float32).reshape(x_shape) / 255.

    self._testBrightness(x_np, y_np, delta=10. / 255.)

  def testPositiveDeltaFloat16(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.float16).reshape(x_shape) / 255.

    y_data = [10, 15, 23, 64, 145, 236, 47, 18, 244, 100, 265, 11]
    y_np = np.array(y_data, dtype=np.float16).reshape(x_shape) / 255.

    self._testBrightness(x_np, y_np, delta=10. / 255., tol=1e-3)

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
    mn = np.mean(x)
    std = np.std(x)
    stddev = max(std, 1.0 / math.sqrt(num_pixels))

    y = x.astype(np.float32)
    y -= mn
    y /= stddev
    return y

  @test_util.run_deprecated_v1
  def testBasic(self):
    x_shape = [13, 9, 3]
    x_np = np.arange(0, np.prod(x_shape), dtype=np.float32).reshape(x_shape)
    y_np = self._NumpyPerImageWhitening(x_np)

    with self.cached_session(use_gpu=True):
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.per_image_standardization(x)
      self.assertTrue(y.op.name.startswith("per_image_standardization"))
      y_tf = self.evaluate(y)
      self.assertAllClose(y_tf, y_np, atol=1e-4)

  def testUniformImage(self):
    im_np = np.ones([19, 19, 3]).astype(np.float32) * 249
    im = constant_op.constant(im_np)
    whiten = image_ops.per_image_standardization(im)
    with self.cached_session(use_gpu=True):
      whiten_np = self.evaluate(whiten)
      self.assertFalse(np.any(np.isnan(whiten_np)))

  def testBatchWhitening(self):
    imgs_np = np.random.uniform(0., 255., [4, 24, 24, 3])
    whiten_np = [self._NumpyPerImageWhitening(img) for img in imgs_np]
    with self.cached_session(use_gpu=True):
      imgs = constant_op.constant(imgs_np)
      whiten = image_ops.per_image_standardization(imgs)
      whiten_tf = self.evaluate(whiten)
      for w_tf, w_np in zip(whiten_tf, whiten_np):
        self.assertAllClose(w_tf, w_np, atol=1e-4)

  def testPreservesDtype(self):
    imgs_npu8 = np.random.uniform(0., 255., [2, 5, 5, 3]).astype(np.uint8)
    imgs_tfu8 = constant_op.constant(imgs_npu8)
    whiten_tfu8 = image_ops.per_image_standardization(imgs_tfu8)
    self.assertEqual(whiten_tfu8.dtype, dtypes.uint8)

    imgs_npf16 = np.random.uniform(0., 255., [2, 5, 5, 3]).astype(np.float16)
    imgs_tff16 = constant_op.constant(imgs_npf16)
    whiten_tff16 = image_ops.per_image_standardization(imgs_tff16)
    self.assertEqual(whiten_tff16.dtype, dtypes.float16)


class CropToBoundingBoxTest(test_util.TensorFlowTestCase):

  def _CropToBoundingBox(self, x, offset_height, offset_width, target_height,
                         target_width, use_tensor_inputs):
    if use_tensor_inputs:
      offset_height = ops.convert_to_tensor(offset_height)
      offset_width = ops.convert_to_tensor(offset_width)
      target_height = ops.convert_to_tensor(target_height)
      target_width = ops.convert_to_tensor(target_width)
      x_tensor = array_ops.placeholder(x.dtype, shape=[None] * x.ndim)
      feed_dict = {x_tensor: x}
    else:
      x_tensor = x
      feed_dict = {}

    y = image_ops.crop_to_bounding_box(x_tensor, offset_height, offset_width,
                                       target_height, target_width)
    if not use_tensor_inputs:
      self.assertTrue(y.get_shape().is_fully_defined())

    with self.cached_session(use_gpu=True):
      return y.eval(feed_dict=feed_dict)

  def _assertReturns(self,
                     x,
                     x_shape,
                     offset_height,
                     offset_width,
                     y,
                     y_shape,
                     use_tensor_inputs_options=None):
    use_tensor_inputs_options = use_tensor_inputs_options or [False, True]
    target_height, target_width, _ = y_shape
    x = np.array(x).reshape(x_shape)
    y = np.array(y).reshape(y_shape)

    for use_tensor_inputs in use_tensor_inputs_options:
      y_tf = self._CropToBoundingBox(x, offset_height, offset_width,
                                     target_height, target_width,
                                     use_tensor_inputs)
      self.assertAllClose(y, y_tf)

  def _assertRaises(self,
                    x,
                    x_shape,
                    offset_height,
                    offset_width,
                    target_height,
                    target_width,
                    err_msg,
                    use_tensor_inputs_options=None):
    use_tensor_inputs_options = use_tensor_inputs_options or [False, True]
    x = np.array(x).reshape(x_shape)

    for use_tensor_inputs in use_tensor_inputs_options:
      try:
        self._CropToBoundingBox(x, offset_height, offset_width, target_height,
                                target_width, use_tensor_inputs)
      except Exception as e:
        if err_msg not in str(e):
          raise
      else:
        raise AssertionError("Exception not raised: %s" % err_msg)

  def _assertShapeInference(self, pre_shape, height, width, post_shape):
    image = array_ops.placeholder(dtypes.float32, shape=pre_shape)
    y = image_ops.crop_to_bounding_box(image, 0, 0, height, width)
    self.assertEqual(y.get_shape().as_list(), post_shape)

  @test_util.run_deprecated_v1
  def testNoOp(self):
    x_shape = [10, 10, 10]
    x = np.random.uniform(size=x_shape)
    self._assertReturns(x, x_shape, 0, 0, x, x_shape)

  @test_util.run_deprecated_v1
  def testCrop(self):
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    x_shape = [3, 3, 1]

    offset_height, offset_width = [1, 0]
    y_shape = [2, 3, 1]
    y = [4, 5, 6, 7, 8, 9]
    self._assertReturns(x, x_shape, offset_height, offset_width, y, y_shape)

    offset_height, offset_width = [0, 1]
    y_shape = [3, 2, 1]
    y = [2, 3, 5, 6, 8, 9]
    self._assertReturns(x, x_shape, offset_height, offset_width, y, y_shape)

    offset_height, offset_width = [0, 0]
    y_shape = [2, 3, 1]
    y = [1, 2, 3, 4, 5, 6]
    self._assertReturns(x, x_shape, offset_height, offset_width, y, y_shape)

    offset_height, offset_width = [0, 0]
    y_shape = [3, 2, 1]
    y = [1, 2, 4, 5, 7, 8]
    self._assertReturns(x, x_shape, offset_height, offset_width, y, y_shape)

  @test_util.run_deprecated_v1
  def testShapeInference(self):
    self._assertShapeInference([55, 66, 3], 55, 66, [55, 66, 3])
    self._assertShapeInference([59, 69, 3], 55, 66, [55, 66, 3])
    self._assertShapeInference([None, 66, 3], 55, 66, [55, 66, 3])
    self._assertShapeInference([None, 69, 3], 55, 66, [55, 66, 3])
    self._assertShapeInference([55, None, 3], 55, 66, [55, 66, 3])
    self._assertShapeInference([59, None, 3], 55, 66, [55, 66, 3])
    self._assertShapeInference([None, None, 3], 55, 66, [55, 66, 3])
    self._assertShapeInference([55, 66, None], 55, 66, [55, 66, None])
    self._assertShapeInference([59, 69, None], 55, 66, [55, 66, None])
    self._assertShapeInference([None, None, None], 55, 66, [55, 66, None])
    self._assertShapeInference(None, 55, 66, [55, 66, None])

  @test_util.run_deprecated_v1
  def testNon3DInput(self):
    # Input image is not 3D
    x = [0] * 15
    offset_height, offset_width = [0, 0]
    target_height, target_width = [2, 2]

    for x_shape in ([3, 5], [1, 3, 5, 1, 1]):
      self._assertRaises(x, x_shape, offset_height, offset_width, target_height,
                         target_width,
                         "must have either 3 or 4 dimensions.")

  @test_util.run_deprecated_v1
  def testZeroLengthInput(self):
    # Input image has 0-length dimension(s).
    # Each line is a test configuration:
    #   x_shape, target_height, target_width
    test_config = (([0, 2, 2], 1, 1), ([2, 0, 2], 1, 1), ([2, 2, 0], 1, 1),
                   ([0, 2, 2], 0, 1), ([2, 0, 2], 1, 0))
    offset_height, offset_width = [0, 0]
    x = []

    for x_shape, target_height, target_width in test_config:
      self._assertRaises(
          x,
          x_shape,
          offset_height,
          offset_width,
          target_height,
          target_width,
          "inner 3 dims of 'image.shape' must be > 0",
          use_tensor_inputs_options=[False])
      # Multiple assertion could fail, but the evaluation order is arbitrary.
      # Match gainst generic pattern.
      self._assertRaises(
          x,
          x_shape,
          offset_height,
          offset_width,
          target_height,
          target_width,
          "assertion failed:",
          use_tensor_inputs_options=[True])

  @test_util.run_deprecated_v1
  def testBadParams(self):
    x_shape = [4, 4, 1]
    x = np.zeros(x_shape)

    # Each line is a test configuration:
    #   (offset_height, offset_width, target_height, target_width), err_msg
    test_config = (([-1, 0, 3, 3], "offset_height must be >= 0"), ([
        0, -1, 3, 3
    ], "offset_width must be >= 0"), ([0, 0, 0, 3],
                                      "target_height must be > 0"),
                   ([0, 0, 3, 0], "target_width must be > 0"),
                   ([2, 0, 3, 3], "height must be >= target + offset"),
                   ([0, 2, 3, 3], "width must be >= target + offset"))

    for params, err_msg in test_config:
      self._assertRaises(x, x_shape, *params, err_msg=err_msg)

  @test_util.run_deprecated_v1
  def testNameScope(self):
    image = array_ops.placeholder(dtypes.float32, shape=[55, 66, 3])
    y = image_ops.crop_to_bounding_box(image, 0, 0, 55, 66)
    self.assertTrue(y.name.startswith("crop_to_bounding_box"))


class CentralCropTest(test_util.TensorFlowTestCase):

  def _assertShapeInference(self, pre_shape, fraction, post_shape):
    image = array_ops.placeholder(dtypes.float32, shape=pre_shape)
    y = image_ops.central_crop(image, fraction)
    if post_shape is None:
      self.assertEqual(y.get_shape().dims, None)
    else:
      self.assertEqual(y.get_shape().as_list(), post_shape)

  @test_util.run_deprecated_v1
  def testNoOp(self):
    x_shapes = [[13, 9, 3], [5, 13, 9, 3]]
    for x_shape in x_shapes:
      x_np = np.ones(x_shape, dtype=np.float32)
      for use_gpu in [True, False]:
        with self.cached_session(use_gpu=use_gpu):
          x = constant_op.constant(x_np, shape=x_shape)
          y = image_ops.central_crop(x, 1.0)
          y_tf = self.evaluate(y)
          self.assertAllEqual(y_tf, x_np)
          self.assertEqual(y.op.name, x.op.name)

  def testCropping(self):
    x_shape = [4, 8, 1]
    x_np = np.array(
        [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8],
         [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]],
        dtype=np.int32).reshape(x_shape)
    y_np = np.array([[3, 4, 5, 6], [3, 4, 5, 6]]).reshape([2, 4, 1])
    for use_gpu in [True, False]:
      with self.cached_session(use_gpu=use_gpu):
        x = constant_op.constant(x_np, shape=x_shape)
        y = image_ops.central_crop(x, 0.5)
        y_tf = self.evaluate(y)
        self.assertAllEqual(y_tf, y_np)
        self.assertAllEqual(y_tf.shape, y_np.shape)

    x_shape = [2, 4, 8, 1]
    x_np = np.array(
        [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8],
         [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8],
         [8, 7, 6, 5, 4, 3, 2, 1], [8, 7, 6, 5, 4, 3, 2, 1],
         [8, 7, 6, 5, 4, 3, 2, 1], [8, 7, 6, 5, 4, 3, 2, 1]],
        dtype=np.int32).reshape(x_shape)
    y_np = np.array([[[3, 4, 5, 6], [3, 4, 5, 6]],
                     [[6, 5, 4, 3], [6, 5, 4, 3]]]).reshape([2, 2, 4, 1])
    with self.cached_session(use_gpu=True):
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.central_crop(x, 0.5)
      y_tf = self.evaluate(y)
      self.assertAllEqual(y_tf, y_np)
      self.assertAllEqual(y_tf.shape, y_np.shape)

  @test_util.run_deprecated_v1
  def testCropping2(self):
    # Test case for 10315
    x_shapes = [[240, 320, 3], [5, 240, 320, 3]]
    expected_y_shapes = [[80, 106, 3], [5, 80, 106, 3]]

    for x_shape, y_shape in zip(x_shapes, expected_y_shapes):
      x_np = np.zeros(x_shape, dtype=np.int32)
      y_np = np.zeros(y_shape, dtype=np.int32)
      for use_gpu in [True, False]:
        with self.cached_session(use_gpu=use_gpu):
          x = array_ops.placeholder(shape=x_shape, dtype=dtypes.int32)
          y = image_ops.central_crop(x, 0.33)
          y_tf = y.eval(feed_dict={x: x_np})
          self.assertAllEqual(y_tf, y_np)
          self.assertAllEqual(y_tf.shape, y_np.shape)

  @test_util.run_deprecated_v1
  def testShapeInference(self):
    # Test no-op fraction=1.0, with 3-D tensors.
    self._assertShapeInference([50, 60, 3], 1.0, [50, 60, 3])
    self._assertShapeInference([None, 60, 3], 1.0, [None, 60, 3])
    self._assertShapeInference([50, None, 3], 1.0, [50, None, 3])
    self._assertShapeInference([None, None, 3], 1.0, [None, None, 3])
    self._assertShapeInference([50, 60, None], 1.0, [50, 60, None])
    self._assertShapeInference([None, None, None], 1.0, [None, None, None])

    # Test no-op fraction=0.5, with 3-D tensors.
    self._assertShapeInference([50, 60, 3], 0.5, [26, 30, 3])
    self._assertShapeInference([None, 60, 3], 0.5, [None, 30, 3])
    self._assertShapeInference([50, None, 3], 0.5, [26, None, 3])
    self._assertShapeInference([None, None, 3], 0.5, [None, None, 3])
    self._assertShapeInference([50, 60, None], 0.5, [26, 30, None])
    self._assertShapeInference([None, None, None], 0.5, [None, None, None])

    # Test no-op fraction=1.0, with 4-D tensors.
    self._assertShapeInference([5, 50, 60, 3], 1.0, [5, 50, 60, 3])
    self._assertShapeInference([5, None, 60, 3], 1.0, [5, None, 60, 3])
    self._assertShapeInference([5, 50, None, 3], 1.0, [5, 50, None, 3])
    self._assertShapeInference([5, None, None, 3], 1.0, [5, None, None, 3])
    self._assertShapeInference([5, 50, 60, None], 1.0, [5, 50, 60, None])
    self._assertShapeInference([5, None, None, None], 1.0,
                               [5, None, None, None])
    self._assertShapeInference([None, None, None, None], 1.0,
                               [None, None, None, None])

    # Test no-op fraction=0.5, with 4-D tensors.
    self._assertShapeInference([5, 50, 60, 3], 0.5, [5, 26, 30, 3])
    self._assertShapeInference([5, None, 60, 3], 0.5, [5, None, 30, 3])
    self._assertShapeInference([5, 50, None, 3], 0.5, [5, 26, None, 3])
    self._assertShapeInference([5, None, None, 3], 0.5, [5, None, None, 3])
    self._assertShapeInference([5, 50, 60, None], 0.5, [5, 26, 30, None])
    self._assertShapeInference([5, None, None, None], 0.5,
                               [5, None, None, None])
    self._assertShapeInference([None, None, None, None], 0.5,
                               [None, None, None, None])

  def testErrorOnInvalidCentralCropFractionValues(self):
    x_shape = [13, 9, 3]
    x_np = np.ones(x_shape, dtype=np.float32)
    for use_gpu in [True, False]:
      with self.cached_session(use_gpu=use_gpu):
        x = constant_op.constant(x_np, shape=x_shape)
        with self.assertRaises(ValueError):
          _ = image_ops.central_crop(x, 0.0)
        with self.assertRaises(ValueError):
          _ = image_ops.central_crop(x, 1.01)

  def testErrorOnInvalidShapes(self):
    x_shapes = [None, [], [3], [3, 9], [3, 9, 3, 9, 3]]
    for x_shape in x_shapes:
      x_np = np.ones(x_shape, dtype=np.float32)
      for use_gpu in [True, False]:
        with self.cached_session(use_gpu=use_gpu):
          x = constant_op.constant(x_np, shape=x_shape)
          with self.assertRaises(ValueError):
            _ = image_ops.central_crop(x, 0.5)

  @test_util.run_deprecated_v1
  def testNameScope(self):
    x_shape = [13, 9, 3]
    x_np = np.ones(x_shape, dtype=np.float32)
    for use_gpu in [True, False]:
      with self.cached_session(use_gpu=use_gpu):
        y = image_ops.central_crop(x_np, 1.0)
        self.assertTrue(y.op.name.startswith("central_crop"))


class PadToBoundingBoxTest(test_util.TensorFlowTestCase):

  def _PadToBoundingBox(self, x, offset_height, offset_width, target_height,
                        target_width, use_tensor_inputs):
    if use_tensor_inputs:
      offset_height = ops.convert_to_tensor(offset_height)
      offset_width = ops.convert_to_tensor(offset_width)
      target_height = ops.convert_to_tensor(target_height)
      target_width = ops.convert_to_tensor(target_width)
      x_tensor = array_ops.placeholder(x.dtype, shape=[None] * x.ndim)
      feed_dict = {x_tensor: x}
    else:
      x_tensor = x
      feed_dict = {}

    y = image_ops.pad_to_bounding_box(x_tensor, offset_height, offset_width,
                                      target_height, target_width)
    if not use_tensor_inputs:
      self.assertTrue(y.get_shape().is_fully_defined())

    with self.cached_session(use_gpu=True):
      return y.eval(feed_dict=feed_dict)

  def _assertReturns(self,
                     x,
                     x_shape,
                     offset_height,
                     offset_width,
                     y,
                     y_shape,
                     use_tensor_inputs_options=None):
    use_tensor_inputs_options = use_tensor_inputs_options or [False, True]
    target_height, target_width, _ = y_shape
    x = np.array(x).reshape(x_shape)
    y = np.array(y).reshape(y_shape)

    for use_tensor_inputs in use_tensor_inputs_options:
      y_tf = self._PadToBoundingBox(x, offset_height, offset_width,
                                    target_height, target_width,
                                    use_tensor_inputs)
      self.assertAllClose(y, y_tf)

  def _assertRaises(self,
                    x,
                    x_shape,
                    offset_height,
                    offset_width,
                    target_height,
                    target_width,
                    err_msg,
                    use_tensor_inputs_options=None):
    use_tensor_inputs_options = use_tensor_inputs_options or [False, True]
    x = np.array(x).reshape(x_shape)

    for use_tensor_inputs in use_tensor_inputs_options:
      try:
        self._PadToBoundingBox(x, offset_height, offset_width, target_height,
                               target_width, use_tensor_inputs)
      except Exception as e:
        if err_msg not in str(e):
          raise
      else:
        raise AssertionError("Exception not raised: %s" % err_msg)

  def _assertShapeInference(self, pre_shape, height, width, post_shape):
    image = array_ops.placeholder(dtypes.float32, shape=pre_shape)
    y = image_ops.pad_to_bounding_box(image, 0, 0, height, width)
    self.assertEqual(y.get_shape().as_list(), post_shape)

  def testInt64(self):
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    x_shape = [3, 3, 1]

    y = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    y_shape = [4, 3, 1]
    x = np.array(x).reshape(x_shape)
    y = np.array(y).reshape(y_shape)

    i = constant_op.constant([1, 0, 4, 3], dtype=dtypes.int64)
    y_tf = image_ops.pad_to_bounding_box(x, i[0], i[1], i[2], i[3])
    with self.cached_session(use_gpu=True):
      self.assertAllClose(y, self.evaluate(y_tf))

  @test_util.run_deprecated_v1
  def testNoOp(self):
    x_shape = [10, 10, 10]
    x = np.random.uniform(size=x_shape)
    offset_height, offset_width = [0, 0]
    self._assertReturns(x, x_shape, offset_height, offset_width, x, x_shape)

  @test_util.run_deprecated_v1
  def testPadding(self):
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    x_shape = [3, 3, 1]

    offset_height, offset_width = [1, 0]
    y = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    y_shape = [4, 3, 1]
    self._assertReturns(x, x_shape, offset_height, offset_width, y, y_shape)

    offset_height, offset_width = [0, 1]
    y = [0, 1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9]
    y_shape = [3, 4, 1]
    self._assertReturns(x, x_shape, offset_height, offset_width, y, y_shape)

    offset_height, offset_width = [0, 0]
    y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0]
    y_shape = [4, 3, 1]
    self._assertReturns(x, x_shape, offset_height, offset_width, y, y_shape)

    offset_height, offset_width = [0, 0]
    y = [1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0]
    y_shape = [3, 4, 1]
    self._assertReturns(x, x_shape, offset_height, offset_width, y, y_shape)

  @test_util.run_deprecated_v1
  def testShapeInference(self):
    self._assertShapeInference([55, 66, 3], 55, 66, [55, 66, 3])
    self._assertShapeInference([50, 60, 3], 55, 66, [55, 66, 3])
    self._assertShapeInference([None, 66, 3], 55, 66, [55, 66, 3])
    self._assertShapeInference([None, 60, 3], 55, 66, [55, 66, 3])
    self._assertShapeInference([55, None, 3], 55, 66, [55, 66, 3])
    self._assertShapeInference([50, None, 3], 55, 66, [55, 66, 3])
    self._assertShapeInference([None, None, 3], 55, 66, [55, 66, 3])
    self._assertShapeInference([55, 66, None], 55, 66, [55, 66, None])
    self._assertShapeInference([50, 60, None], 55, 66, [55, 66, None])
    self._assertShapeInference([None, None, None], 55, 66, [55, 66, None])
    self._assertShapeInference(None, 55, 66, [55, 66, None])

  @test_util.run_deprecated_v1
  def testNon3DInput(self):
    # Input image is not 3D
    x = [0] * 15
    offset_height, offset_width = [0, 0]
    target_height, target_width = [2, 2]

    for x_shape in ([3, 5], [1, 3, 5, 1, 1]):
      self._assertRaises(x, x_shape, offset_height, offset_width, target_height,
                         target_width,
                         "must have either 3 or 4 dimensions.")

  @test_util.run_deprecated_v1
  def testZeroLengthInput(self):
    # Input image has 0-length dimension(s).
    # Each line is a test configuration:
    #   x_shape, target_height, target_width
    test_config = (([0, 2, 2], 2, 2), ([2, 0, 2], 2, 2), ([2, 2, 0], 2, 2))
    offset_height, offset_width = [0, 0]
    x = []

    for x_shape, target_height, target_width in test_config:
      self._assertRaises(
          x,
          x_shape,
          offset_height,
          offset_width,
          target_height,
          target_width,
          "inner 3 dims of 'image.shape' must be > 0",
          use_tensor_inputs_options=[False])

      # The original error message does not contain back slashes. However, they
      # are added by either the assert op or the runtime. If this behavior
      # changes in the future, the match string will also needs to be changed.
      self._assertRaises(
          x,
          x_shape,
          offset_height,
          offset_width,
          target_height,
          target_width,
          "inner 3 dims of \\'image.shape\\' must be > 0",
          use_tensor_inputs_options=[True])

  @test_util.run_deprecated_v1
  def testBadParams(self):
    x_shape = [3, 3, 1]
    x = np.zeros(x_shape)

    # Each line is a test configuration:
    #   offset_height, offset_width, target_height, target_width, err_msg
    test_config = ((-1, 0, 4, 4, "offset_height must be >= 0"),
                   (0, -1, 4, 4, "offset_width must be >= 0"),
                   (2, 0, 4, 4, "height must be <= target - offset"),
                   (0, 2, 4, 4, "width must be <= target - offset"))

    for config_item in test_config:
      self._assertRaises(x, x_shape, *config_item)

  @test_util.run_deprecated_v1
  def testNameScope(self):
    image = array_ops.placeholder(dtypes.float32, shape=[55, 66, 3])
    y = image_ops.pad_to_bounding_box(image, 0, 0, 55, 66)
    self.assertTrue(y.op.name.startswith("pad_to_bounding_box"))


class SelectDistortedCropBoxTest(test_util.TensorFlowTestCase):

  def _testSampleDistortedBoundingBox(self, image, bounding_box,
                                      min_object_covered, aspect_ratio_range,
                                      area_range):
    original_area = float(np.prod(image.shape))
    bounding_box_area = float((bounding_box[3] - bounding_box[1]) *
                              (bounding_box[2] - bounding_box[0]))

    image_size_np = np.array(image.shape, dtype=np.int32)
    bounding_box_np = (
        np.array(bounding_box, dtype=np.float32).reshape([1, 1, 4]))

    aspect_ratios = []
    area_ratios = []

    fraction_object_covered = []

    num_iter = 1000
    with self.cached_session(use_gpu=True):
      image_tf = constant_op.constant(image, shape=image.shape)
      image_size_tf = constant_op.constant(
          image_size_np, shape=image_size_np.shape)
      bounding_box_tf = constant_op.constant(
          bounding_box_np, dtype=dtypes.float32, shape=bounding_box_np.shape)

      begin, size, _ = image_ops.sample_distorted_bounding_box(
          image_size=image_size_tf,
          bounding_boxes=bounding_box_tf,
          min_object_covered=min_object_covered,
          aspect_ratio_range=aspect_ratio_range,
          area_range=area_range)
      y = array_ops.strided_slice(image_tf, begin, begin + size)

      for _ in xrange(num_iter):
        y_tf = self.evaluate(y)
        crop_height = y_tf.shape[0]
        crop_width = y_tf.shape[1]
        aspect_ratio = float(crop_width) / float(crop_height)
        area = float(crop_width * crop_height)

        aspect_ratios.append(aspect_ratio)
        area_ratios.append(area / original_area)
        fraction_object_covered.append(float(np.sum(y_tf)) / bounding_box_area)

      # min_object_covered as tensor
      min_object_covered_placeholder = array_ops.placeholder(dtypes.float32)
      begin, size, _ = image_ops.sample_distorted_bounding_box(
          image_size=image_size_tf,
          bounding_boxes=bounding_box_tf,
          min_object_covered=min_object_covered_placeholder,
          aspect_ratio_range=aspect_ratio_range,
          area_range=area_range)
      y = array_ops.strided_slice(image_tf, begin, begin + size)

      for _ in xrange(num_iter):
        y_tf = y.eval(feed_dict={
            min_object_covered_placeholder: min_object_covered
        })
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
    print("area_ratio_hist ", area_ratio_hist)

    # Ensure that fraction_object_covered is satisfied.
    # TODO(wicke, shlens, dga): Restore this test so that it is no longer flaky.
    # self.assertGreaterEqual(min(fraction_object_covered), min_object_covered)

  @test_util.run_deprecated_v1
  def testWholeImageBoundingBox(self):
    height = 40
    width = 50
    image_size = [height, width, 1]
    bounding_box = [0.0, 0.0, 1.0, 1.0]
    image = np.arange(
        0, np.prod(image_size), dtype=np.int32).reshape(image_size)
    self._testSampleDistortedBoundingBox(
        image,
        bounding_box,
        min_object_covered=0.1,
        aspect_ratio_range=(0.75, 1.33),
        area_range=(0.05, 1.0))

  @test_util.run_deprecated_v1
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

    self._testSampleDistortedBoundingBox(
        image,
        bounding_box=bounding_box,
        min_object_covered=min_object_covered,
        aspect_ratio_range=(0.75, 1.33),
        area_range=(0.05, 1.0))

  @test_util.run_deprecated_v1
  def testSampleDistortedBoundingBoxShape(self):
    with self.cached_session(use_gpu=True):
      image_size = constant_op.constant(
          [40, 50, 1], shape=[3], dtype=dtypes.int32)
      bounding_box = constant_op.constant(
          [[[0.0, 0.0, 1.0, 1.0]]],
          shape=[1, 1, 4],
          dtype=dtypes.float32,
      )
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
      # Actual run to make sure shape is correct inside Compute().
      begin = self.evaluate(begin)
      end = self.evaluate(end)
      bbox_for_drawing = self.evaluate(bbox_for_drawing)

      begin, end, bbox_for_drawing = image_ops.sample_distorted_bounding_box(
          image_size=image_size,
          bounding_boxes=bounding_box,
          min_object_covered=array_ops.placeholder(dtypes.float32),
          aspect_ratio_range=(0.75, 1.33),
          area_range=(0.05, 1.0))

      # Test that the shapes are correct.
      self.assertAllEqual([3], begin.get_shape().as_list())
      self.assertAllEqual([3], end.get_shape().as_list())
      self.assertAllEqual([1, 1, 4], bbox_for_drawing.get_shape().as_list())

  def testDefaultMinObjectCovered(self):
    # By default min_object_covered=0.1 if not provided
    with self.cached_session(use_gpu=True):
      image_size = constant_op.constant(
          [40, 50, 1], shape=[3], dtype=dtypes.int32)
      bounding_box = constant_op.constant(
          [[[0.0, 0.0, 1.0, 1.0]]],
          shape=[1, 1, 4],
          dtype=dtypes.float32,
      )
      begin, end, bbox_for_drawing = image_ops.sample_distorted_bounding_box(
          image_size=image_size,
          bounding_boxes=bounding_box,
          aspect_ratio_range=(0.75, 1.33),
          area_range=(0.05, 1.0))

      self.assertAllEqual([3], begin.get_shape().as_list())
      self.assertAllEqual([3], end.get_shape().as_list())
      self.assertAllEqual([1, 1, 4], bbox_for_drawing.get_shape().as_list())
      # Actual run to make sure shape is correct inside Compute().
      begin = self.evaluate(begin)
      end = self.evaluate(end)
      bbox_for_drawing = self.evaluate(bbox_for_drawing)


class ResizeImagesV2Test(test_util.TensorFlowTestCase):

  METHODS = [
      image_ops.ResizeMethod.BILINEAR, image_ops.ResizeMethod.NEAREST_NEIGHBOR,
      image_ops.ResizeMethod.BICUBIC, image_ops.ResizeMethod.AREA,
      image_ops.ResizeMethod.LANCZOS3, image_ops.ResizeMethod.LANCZOS5,
      image_ops.ResizeMethod.GAUSSIAN, image_ops.ResizeMethod.MITCHELLCUBIC
  ]

  # Some resize methods, such as Gaussian, are non-interpolating in that they
  # change the image even if there is no scale change, for some test, we only
  # check the value on the value preserving methods.
  INTERPOLATING_METHODS = [
      image_ops.ResizeMethod.BILINEAR, image_ops.ResizeMethod.NEAREST_NEIGHBOR,
      image_ops.ResizeMethod.BICUBIC, image_ops.ResizeMethod.AREA,
      image_ops.ResizeMethod.LANCZOS3, image_ops.ResizeMethod.LANCZOS5
  ]

  TYPES = [
      np.uint8, np.int8, np.uint16, np.int16, np.int32, np.int64, np.float16,
      np.float32, np.float64
  ]

  def _assertShapeInference(self, pre_shape, size, post_shape):
    # Try single image resize
    single_image = array_ops.placeholder(dtypes.float32, shape=pre_shape)
    y = image_ops.resize_images_v2(single_image, size)
    self.assertEqual(y.get_shape().as_list(), post_shape)
    # Try batch images resize with known batch size
    images = array_ops.placeholder(dtypes.float32, shape=[99] + pre_shape)
    y = image_ops.resize_images_v2(images, size)
    self.assertEqual(y.get_shape().as_list(), [99] + post_shape)
    # Try batch images resize with unknown batch size
    images = array_ops.placeholder(dtypes.float32, shape=[None] + pre_shape)
    y = image_ops.resize_images_v2(images, size)
    self.assertEqual(y.get_shape().as_list(), [None] + post_shape)

  def shouldRunOnGPU(self, method, nptype):
    if (method == image_ops.ResizeMethod.NEAREST_NEIGHBOR and
        nptype in [np.float32, np.float64]):
      return True
    else:
      return False

  @test_util.disable_xla("align_corners=False not supported by XLA")
  @test_util.run_deprecated_v1
  def testNoOp(self):
    img_shape = [1, 6, 4, 1]
    single_shape = [6, 4, 1]
    # This test is also conducted with int8, so 127 is the maximum
    # value that can be used.
    data = [
        127, 127, 64, 64, 127, 127, 64, 64, 64, 64, 127, 127, 64, 64, 127, 127,
        50, 50, 100, 100, 50, 50, 100, 100
    ]
    target_height = 6
    target_width = 4

    for nptype in self.TYPES:
      img_np = np.array(data, dtype=nptype).reshape(img_shape)

      for method in self.METHODS:
        with self.cached_session(use_gpu=True):
          image = constant_op.constant(img_np, shape=img_shape)
          y = image_ops.resize_images_v2(image, [target_height, target_width],
                                         method)
          yshape = array_ops.shape(y)
          resized, newshape = self.evaluate([y, yshape])
          self.assertAllEqual(img_shape, newshape)
          if method in self.INTERPOLATING_METHODS:
            self.assertAllClose(resized, img_np, atol=1e-5)

      # Resizing with a single image must leave the shape unchanged also.
      with self.cached_session(use_gpu=True):
        img_single = img_np.reshape(single_shape)
        image = constant_op.constant(img_single, shape=single_shape)
        y = image_ops.resize_images_v2(image, [target_height, target_width],
                                       self.METHODS[0])
        yshape = array_ops.shape(y)
        newshape = self.evaluate(yshape)
        self.assertAllEqual(single_shape, newshape)

  # half_pixel_centers unsupported in ResizeBilinear
  @test_util.run_deprecated_v1
  @test_util.disable_xla("b/127616992")
  def testTensorArguments(self):
    img_shape = [1, 6, 4, 1]
    single_shape = [6, 4, 1]
    # This test is also conducted with int8, so 127 is the maximum
    # value that can be used.
    data = [
        127, 127, 64, 64, 127, 127, 64, 64, 64, 64, 127, 127, 64, 64, 127, 127,
        50, 50, 100, 100, 50, 50, 100, 100
    ]
    new_size = array_ops.placeholder(dtypes.int32, shape=(2))

    img_np = np.array(data, dtype=np.uint8).reshape(img_shape)

    for method in self.METHODS:
      with self.cached_session(use_gpu=True) as sess:
        image = constant_op.constant(img_np, shape=img_shape)
        y = image_ops.resize_images_v2(image, new_size, method)
        yshape = array_ops.shape(y)
        resized, newshape = sess.run([y, yshape], {new_size: [6, 4]})
        self.assertAllEqual(img_shape, newshape)
        if method in self.INTERPOLATING_METHODS:
          self.assertAllClose(resized, img_np, atol=1e-5)

      # Resizing with a single image must leave the shape unchanged also.
      with self.cached_session(use_gpu=True):
        img_single = img_np.reshape(single_shape)
        image = constant_op.constant(img_single, shape=single_shape)
        y = image_ops.resize_images_v2(image, new_size, self.METHODS[0])
        yshape = array_ops.shape(y)
        resized, newshape = sess.run([y, yshape], {new_size: [6, 4]})
        self.assertAllEqual(single_shape, newshape)
        if method in self.INTERPOLATING_METHODS:
          self.assertAllClose(resized, img_single, atol=1e-5)

    # Incorrect shape.
    with self.assertRaises(ValueError):
      new_size = constant_op.constant(4)
      _ = image_ops.resize_images_v2(image, new_size,
                                     image_ops.ResizeMethod.BILINEAR)
    with self.assertRaises(ValueError):
      new_size = constant_op.constant([4])
      _ = image_ops.resize_images_v2(image, new_size,
                                     image_ops.ResizeMethod.BILINEAR)
    with self.assertRaises(ValueError):
      new_size = constant_op.constant([1, 2, 3])
      _ = image_ops.resize_images_v2(image, new_size,
                                     image_ops.ResizeMethod.BILINEAR)

    # Incorrect dtypes.
    with self.assertRaises(ValueError):
      new_size = constant_op.constant([6.0, 4])
      _ = image_ops.resize_images_v2(image, new_size,
                                     image_ops.ResizeMethod.BILINEAR)
    with self.assertRaises(ValueError):
      _ = image_ops.resize_images_v2(image, [6, 4.0],
                                     image_ops.ResizeMethod.BILINEAR)
    with self.assertRaises(ValueError):
      _ = image_ops.resize_images_v2(image, [None, 4],
                                     image_ops.ResizeMethod.BILINEAR)
    with self.assertRaises(ValueError):
      _ = image_ops.resize_images_v2(image, [6, None],
                                     image_ops.ResizeMethod.BILINEAR)

  @test_util.run_deprecated_v1
  def testReturnDtype(self):
    target_shapes = [[6, 4], [3, 2],
                     [
                         array_ops.placeholder(dtypes.int32),
                         array_ops.placeholder(dtypes.int32)
                     ]]
    for nptype in self.TYPES:
      image = array_ops.placeholder(nptype, shape=[1, 6, 4, 1])
      for method in self.METHODS:
        for target_shape in target_shapes:
          y = image_ops.resize_images_v2(image, target_shape, method)
          if method == image_ops.ResizeMethod.NEAREST_NEIGHBOR:
            expected_dtype = image.dtype
          else:
            expected_dtype = dtypes.float32
          self.assertEqual(y.dtype, expected_dtype)

  # half_pixel_centers not supported by XLA
  @test_util.disable_xla("b/127616992")
  def testSumTensor(self):
    img_shape = [1, 6, 4, 1]
    # This test is also conducted with int8, so 127 is the maximum
    # value that can be used.
    data = [
        127, 127, 64, 64, 127, 127, 64, 64, 64, 64, 127, 127, 64, 64, 127, 127,
        50, 50, 100, 100, 50, 50, 100, 100
    ]
    # Test size where width is specified as a tensor which is a sum
    # of two tensors.
    width_1 = constant_op.constant(1)
    width_2 = constant_op.constant(3)
    width = math_ops.add(width_1, width_2)
    height = constant_op.constant(6)

    img_np = np.array(data, dtype=np.uint8).reshape(img_shape)

    for method in self.METHODS:
      with self.cached_session():
        image = constant_op.constant(img_np, shape=img_shape)
        y = image_ops.resize_images_v2(image, [height, width], method)
        yshape = array_ops.shape(y)
        resized, newshape = self.evaluate([y, yshape])
        self.assertAllEqual(img_shape, newshape)
        if method in self.INTERPOLATING_METHODS:
          self.assertAllClose(resized, img_np, atol=1e-5)

  @test_util.disable_xla("align_corners=False not supported by XLA")
  def testResizeDown(self):
    # This test is also conducted with int8, so 127 is the maximum
    # value that can be used.
    data = [
        127, 127, 64, 64, 127, 127, 64, 64, 64, 64, 127, 127, 64, 64, 127, 127,
        50, 50, 100, 100, 50, 50, 100, 100
    ]
    expected_data = [127, 64, 64, 127, 50, 100]
    target_height = 3
    target_width = 2

    # Test out 3-D and 4-D image shapes.
    img_shapes = [[1, 6, 4, 1], [6, 4, 1]]
    target_shapes = [[1, target_height, target_width, 1],
                     [target_height, target_width, 1]]

    for target_shape, img_shape in zip(target_shapes, img_shapes):

      for nptype in self.TYPES:
        img_np = np.array(data, dtype=nptype).reshape(img_shape)

        for method in self.METHODS:
          if test.is_gpu_available() and self.shouldRunOnGPU(method, nptype):
            with self.cached_session(use_gpu=True):
              image = constant_op.constant(img_np, shape=img_shape)
              y = image_ops.resize_images_v2(
                  image, [target_height, target_width], method)
              expected = np.array(expected_data).reshape(target_shape)
              resized = self.evaluate(y)
              self.assertAllClose(resized, expected, atol=1e-5)

  @test_util.disable_xla("align_corners=False not supported by XLA")
  def testResizeUp(self):
    img_shape = [1, 3, 2, 1]
    data = [64, 32, 32, 64, 50, 100]
    target_height = 6
    target_width = 4
    expected_data = {}
    expected_data[image_ops.ResizeMethod.BILINEAR] = [
        64.0, 56.0, 40.0, 32.0, 56.0, 52.0, 44.0, 40.0, 40.0, 44.0, 52.0, 56.0,
        36.5, 45.625, 63.875, 73.0, 45.5, 56.875, 79.625, 91.0, 50.0, 62.5,
        87.5, 100.0
    ]
    expected_data[image_ops.ResizeMethod.NEAREST_NEIGHBOR] = [
        64.0, 64.0, 32.0, 32.0, 64.0, 64.0, 32.0, 32.0, 32.0, 32.0, 64.0, 64.0,
        32.0, 32.0, 64.0, 64.0, 50.0, 50.0, 100.0, 100.0, 50.0, 50.0, 100.0,
        100.0
    ]
    expected_data[image_ops.ResizeMethod.AREA] = [
        64.0, 64.0, 32.0, 32.0, 64.0, 64.0, 32.0, 32.0, 32.0, 32.0, 64.0, 64.0,
        32.0, 32.0, 64.0, 64.0, 50.0, 50.0, 100.0, 100.0, 50.0, 50.0, 100.0,
        100.0
    ]
    expected_data[image_ops.ResizeMethod.LANCZOS3] = [
        75.8294, 59.6281, 38.4313, 22.23, 60.6851, 52.0037, 40.6454, 31.964,
        35.8344, 41.0779, 47.9383, 53.1818, 24.6968, 43.0769, 67.1244, 85.5045,
        35.7939, 56.4713, 83.5243, 104.2017, 44.8138, 65.1949, 91.8603, 112.2413
    ]
    expected_data[image_ops.ResizeMethod.LANCZOS5] = [
        77.5699, 60.0223, 40.6694, 23.1219, 61.8253, 51.2369, 39.5593, 28.9709,
        35.7438, 40.8875, 46.5604, 51.7041, 21.5942, 43.5299, 67.7223, 89.658,
        32.1213, 56.784, 83.984, 108.6467, 44.5802, 66.183, 90.0082, 111.6109
    ]
    expected_data[image_ops.ResizeMethod.GAUSSIAN] = [
        61.1087, 54.6926, 41.3074, 34.8913, 54.6926, 51.4168, 44.5832, 41.3074,
        41.696, 45.2456, 52.6508, 56.2004, 39.4273, 47.0526, 62.9602, 70.5855,
        47.3008, 57.3042, 78.173, 88.1764, 51.4771, 62.3638, 85.0752, 95.9619
    ]
    expected_data[image_ops.ResizeMethod.BICUBIC] = [
        70.1453, 59.0252, 36.9748, 25.8547, 59.3195, 53.3386, 41.4789, 35.4981,
        36.383, 41.285, 51.0051, 55.9071, 30.2232, 42.151, 65.8032, 77.731,
        41.6492, 55.823, 83.9288, 98.1026, 47.0363, 62.2744, 92.4903, 107.7284
    ]
    expected_data[image_ops.ResizeMethod.MITCHELLCUBIC] = [
        66.0382, 56.6079, 39.3921, 29.9618, 56.7255, 51.9603, 43.2611, 38.4959,
        39.1828, 43.4664, 51.2864, 55.57, 34.6287, 45.1812, 64.4458, 74.9983,
        43.8523, 56.8078, 80.4594, 93.4149, 48.9943, 63.026, 88.6422, 102.6739
    ]
    for nptype in self.TYPES:
      for method in expected_data:
        with self.cached_session(use_gpu=True):
          img_np = np.array(data, dtype=nptype).reshape(img_shape)
          image = constant_op.constant(img_np, shape=img_shape)
          y = image_ops.resize_images_v2(image, [target_height, target_width],
                                         method)
          resized = self.evaluate(y)
          expected = np.array(expected_data[method]).reshape(
              [1, target_height, target_width, 1])
          self.assertAllClose(resized, expected, atol=1e-04)

  # XLA doesn't implement half_pixel_centers
  @test_util.disable_xla("b/127616992")
  def testLegacyBicubicMethodsMatchNewMethods(self):
    img_shape = [1, 3, 2, 1]
    data = [64, 32, 32, 64, 50, 100]
    target_height = 6
    target_width = 4
    methods_to_test = ((gen_image_ops.resize_bilinear, "triangle"),
                       (gen_image_ops.resize_bicubic, "keyscubic"))
    for legacy_method, new_method in methods_to_test:
      with self.cached_session(use_gpu=True):
        img_np = np.array(data, dtype=np.float32).reshape(img_shape)
        image = constant_op.constant(img_np, shape=img_shape)
        legacy_result = legacy_method(
            image,
            constant_op.constant([target_height, target_width],
                                 dtype=dtypes.int32),
            half_pixel_centers=True)
        scale = (
            constant_op.constant([target_height, target_width],
                                 dtype=dtypes.float32) /
            math_ops.cast(array_ops.shape(image)[1:3], dtype=dtypes.float32))
        new_result = gen_image_ops.scale_and_translate(
            image,
            constant_op.constant([target_height, target_width],
                                 dtype=dtypes.int32),
            scale,
            array_ops.zeros([2]),
            kernel_type=new_method,
            antialias=False)
        self.assertAllClose(
            self.evaluate(legacy_result), self.evaluate(new_result), atol=1e-04)

  def testResizeDownArea(self):
    img_shape = [1, 6, 6, 1]
    data = [
        128, 64, 32, 16, 8, 4, 4, 8, 16, 32, 64, 128, 128, 64, 32, 16, 8, 4, 5,
        10, 15, 20, 25, 30, 30, 25, 20, 15, 10, 5, 5, 10, 15, 20, 25, 30
    ]
    img_np = np.array(data, dtype=np.uint8).reshape(img_shape)

    target_height = 4
    target_width = 4
    expected_data = [
        73, 33, 23, 39, 73, 33, 23, 39, 14, 16, 19, 21, 14, 16, 19, 21
    ]

    with self.cached_session(use_gpu=True):
      image = constant_op.constant(img_np, shape=img_shape)
      y = image_ops.resize_images_v2(image, [target_height, target_width],
                                     image_ops.ResizeMethod.AREA)
      expected = np.array(expected_data).reshape(
          [1, target_height, target_width, 1])
      resized = self.evaluate(y)
      self.assertAllClose(resized, expected, atol=1)

  @test_util.disable_xla("align_corners=False not supported by XLA")
  def testCompareNearestNeighbor(self):
    if test.is_gpu_available():
      input_shape = [1, 5, 6, 3]
      target_height = 8
      target_width = 12
      for nptype in [np.float32, np.float64]:
        img_np = np.arange(
            0, np.prod(input_shape), dtype=nptype).reshape(input_shape)
        with self.cached_session(use_gpu=True):
          image = constant_op.constant(img_np, shape=input_shape)
          new_size = constant_op.constant([target_height, target_width])
          out_op = image_ops.resize_images_v2(
              image, new_size, image_ops.ResizeMethod.NEAREST_NEIGHBOR)
          gpu_val = self.evaluate(out_op)
        with self.cached_session(use_gpu=False):
          image = constant_op.constant(img_np, shape=input_shape)
          new_size = constant_op.constant([target_height, target_width])
          out_op = image_ops.resize_images_v2(
              image, new_size, image_ops.ResizeMethod.NEAREST_NEIGHBOR)
          cpu_val = self.evaluate(out_op)
        self.assertAllClose(cpu_val, gpu_val, rtol=1e-5, atol=1e-5)

  @test_util.disable_xla("align_corners=False not supported by XLA")
  def testBfloat16MultipleOps(self):
    target_height = 8
    target_width = 12
    img = np.random.uniform(0, 100, size=(30, 10, 2)).astype(np.float32)
    img_bf16 = ops.convert_to_tensor(img, dtype="bfloat16")
    new_size = constant_op.constant([target_height, target_width])
    img_methods = [
        image_ops.ResizeMethod.BILINEAR,
        image_ops.ResizeMethod.NEAREST_NEIGHBOR, image_ops.ResizeMethod.BICUBIC,
        image_ops.ResizeMethod.AREA
    ]
    for method in img_methods:
      out_op_bf16 = image_ops.resize_images_v2(img_bf16, new_size, method)
      out_op_f32 = image_ops.resize_images_v2(img, new_size, method)
      bf16_val = self.evaluate(out_op_bf16)
      f32_val = self.evaluate(out_op_f32)
      self.assertAllClose(bf16_val, f32_val, rtol=1e-2, atol=1e-2)

  def testCompareBilinear(self):
    if test.is_gpu_available():
      input_shape = [1, 5, 6, 3]
      target_height = 8
      target_width = 12
      for nptype in [np.float32, np.float64]:
        img_np = np.arange(
            0, np.prod(input_shape), dtype=nptype).reshape(input_shape)
        value = {}
        for use_gpu in [True, False]:
          with self.cached_session(use_gpu=use_gpu):
            image = constant_op.constant(img_np, shape=input_shape)
            new_size = constant_op.constant([target_height, target_width])
            out_op = image_ops.resize_images(image, new_size,
                                             image_ops.ResizeMethod.BILINEAR)
            value[use_gpu] = self.evaluate(out_op)
        self.assertAllClose(value[True], value[False], rtol=1e-5, atol=1e-5)

  @test_util.run_deprecated_v1
  def testShapeInference(self):
    self._assertShapeInference([50, 60, 3], [55, 66], [55, 66, 3])
    self._assertShapeInference([55, 66, 3], [55, 66], [55, 66, 3])
    self._assertShapeInference([59, 69, 3], [55, 66], [55, 66, 3])
    self._assertShapeInference([50, 69, 3], [55, 66], [55, 66, 3])
    self._assertShapeInference([59, 60, 3], [55, 66], [55, 66, 3])
    self._assertShapeInference([None, 60, 3], [55, 66], [55, 66, 3])
    self._assertShapeInference([None, 66, 3], [55, 66], [55, 66, 3])
    self._assertShapeInference([None, 69, 3], [55, 66], [55, 66, 3])
    self._assertShapeInference([50, None, 3], [55, 66], [55, 66, 3])
    self._assertShapeInference([55, None, 3], [55, 66], [55, 66, 3])
    self._assertShapeInference([59, None, 3], [55, 66], [55, 66, 3])
    self._assertShapeInference([None, None, 3], [55, 66], [55, 66, 3])
    self._assertShapeInference([50, 60, None], [55, 66], [55, 66, None])
    self._assertShapeInference([55, 66, None], [55, 66], [55, 66, None])
    self._assertShapeInference([59, 69, None], [55, 66], [55, 66, None])
    self._assertShapeInference([50, 69, None], [55, 66], [55, 66, None])
    self._assertShapeInference([59, 60, None], [55, 66], [55, 66, None])
    self._assertShapeInference([None, None, None], [55, 66], [55, 66, None])

  @test_util.run_deprecated_v1
  def testNameScope(self):
    with self.cached_session(use_gpu=True):
      single_image = array_ops.placeholder(dtypes.float32, shape=[50, 60, 3])
      y = image_ops.resize_images(single_image, [55, 66])
      self.assertTrue(y.op.name.startswith("resize"))

  def _ResizeImageCall(self, x, max_h, max_w, preserve_aspect_ratio,
                       use_tensor_inputs):
    if use_tensor_inputs:
      target_max = ops.convert_to_tensor([max_h, max_w])
      x_tensor = array_ops.placeholder(x.dtype, shape=[None] * x.ndim)
      feed_dict = {x_tensor: x}
    else:
      target_max = [max_h, max_w]
      x_tensor = x
      feed_dict = {}

    y = image_ops.resize_images(
        x_tensor,
        ops.convert_to_tensor(target_max),
        preserve_aspect_ratio=preserve_aspect_ratio)

    with self.cached_session(use_gpu=True):
      return y.eval(feed_dict=feed_dict)

  def _assertResizeEqual(self,
                         x,
                         x_shape,
                         y,
                         y_shape,
                         preserve_aspect_ratio=True,
                         use_tensor_inputs_options=None):
    use_tensor_inputs_options = use_tensor_inputs_options or [False, True]
    target_height, target_width, _ = y_shape
    x = np.array(x).reshape(x_shape)
    y = np.array(y).reshape(y_shape)

    for use_tensor_inputs in use_tensor_inputs_options:
      y_tf = self._ResizeImageCall(x, target_height, target_width,
                                   preserve_aspect_ratio, use_tensor_inputs)
      self.assertAllClose(y, y_tf)

  def _assertResizeCheckShape(self,
                              x,
                              x_shape,
                              target_shape,
                              y_shape,
                              preserve_aspect_ratio=True,
                              use_tensor_inputs_options=None):
    use_tensor_inputs_options = use_tensor_inputs_options or [False, True]
    target_height, target_width = target_shape
    x = np.array(x).reshape(x_shape)
    y = np.zeros(y_shape)

    for use_tensor_inputs in use_tensor_inputs_options:
      y_tf = self._ResizeImageCall(x, target_height, target_width,
                                   preserve_aspect_ratio, use_tensor_inputs)
      self.assertShapeEqual(y, ops.convert_to_tensor(y_tf))

  @test_util.run_deprecated_v1
  def testPreserveAspectRatioMultipleImages(self):
    x_shape = [10, 100, 80, 10]
    x = np.random.uniform(size=x_shape)
    for preserve_aspect_ratio in [True, False]:
      with self.subTest(preserve_aspect_ratio=preserve_aspect_ratio):
        expect_shape = [10, 250, 200, 10] if preserve_aspect_ratio \
            else [10, 250, 250, 10]
        self._assertResizeCheckShape(
            x,
            x_shape, [250, 250],
            expect_shape,
            preserve_aspect_ratio=preserve_aspect_ratio)

  @test_util.run_deprecated_v1
  def testPreserveAspectRatioNoOp(self):
    x_shape = [10, 10, 10]
    x = np.random.uniform(size=x_shape)

    self._assertResizeEqual(x, x_shape, x, x_shape)

  @test_util.run_deprecated_v1
  def testPreserveAspectRatioSmaller(self):
    x_shape = [100, 100, 10]
    x = np.random.uniform(size=x_shape)

    self._assertResizeCheckShape(x, x_shape, [75, 50], [50, 50, 10])

  @test_util.run_deprecated_v1
  def testPreserveAspectRatioSmallerMultipleImages(self):
    x_shape = [10, 100, 100, 10]
    x = np.random.uniform(size=x_shape)

    self._assertResizeCheckShape(x, x_shape, [75, 50], [10, 50, 50, 10])

  @test_util.run_deprecated_v1
  def testPreserveAspectRatioLarger(self):
    x_shape = [100, 100, 10]
    x = np.random.uniform(size=x_shape)

    self._assertResizeCheckShape(x, x_shape, [150, 200], [150, 150, 10])

  @test_util.run_deprecated_v1
  def testPreserveAspectRatioSameRatio(self):
    x_shape = [1920, 1080, 3]
    x = np.random.uniform(size=x_shape)

    self._assertResizeCheckShape(x, x_shape, [3840, 2160], [3840, 2160, 3])

  @test_util.run_deprecated_v1
  def testPreserveAspectRatioSquare(self):
    x_shape = [299, 299, 3]
    x = np.random.uniform(size=x_shape)

    self._assertResizeCheckShape(x, x_shape, [320, 320], [320, 320, 3])


class ResizeImagesTest(test_util.TensorFlowTestCase):

  METHODS = [
      image_ops.ResizeMethodV1.BILINEAR,
      image_ops.ResizeMethodV1.NEAREST_NEIGHBOR,
      image_ops.ResizeMethodV1.BICUBIC, image_ops.ResizeMethodV1.AREA
  ]

  TYPES = [
      np.uint8, np.int8, np.uint16, np.int16, np.int32, np.int64, np.float16,
      np.float32, np.float64
  ]

  def _assertShapeInference(self, pre_shape, size, post_shape):
    # Try single image resize
    single_image = array_ops.placeholder(dtypes.float32, shape=pre_shape)
    y = image_ops.resize_images(single_image, size)
    self.assertEqual(y.get_shape().as_list(), post_shape)
    # Try batch images resize with known batch size
    images = array_ops.placeholder(dtypes.float32, shape=[99] + pre_shape)
    y = image_ops.resize_images(images, size)
    self.assertEqual(y.get_shape().as_list(), [99] + post_shape)
    # Try batch images resize with unknown batch size
    images = array_ops.placeholder(dtypes.float32, shape=[None] + pre_shape)
    y = image_ops.resize_images(images, size)
    self.assertEqual(y.get_shape().as_list(), [None] + post_shape)

  def shouldRunOnGPU(self, method, nptype):
    if (method == image_ops.ResizeMethodV1.NEAREST_NEIGHBOR and
        nptype in [np.float32, np.float64]):
      return True
    else:
      return False

  @test_util.disable_xla("align_corners=False not supported by XLA")
  @test_util.run_deprecated_v1
  def testNoOp(self):
    img_shape = [1, 6, 4, 1]
    single_shape = [6, 4, 1]
    # This test is also conducted with int8, so 127 is the maximum
    # value that can be used.
    data = [
        127, 127, 64, 64, 127, 127, 64, 64, 64, 64, 127, 127, 64, 64, 127, 127,
        50, 50, 100, 100, 50, 50, 100, 100
    ]
    target_height = 6
    target_width = 4

    for nptype in self.TYPES:
      img_np = np.array(data, dtype=nptype).reshape(img_shape)

      for method in self.METHODS:
        with self.cached_session(use_gpu=True) as sess:
          image = constant_op.constant(img_np, shape=img_shape)
          y = image_ops.resize_images(image, [target_height, target_width],
                                      method)
          yshape = array_ops.shape(y)
          resized, newshape = self.evaluate([y, yshape])
          self.assertAllEqual(img_shape, newshape)
          self.assertAllClose(resized, img_np, atol=1e-5)

      # Resizing with a single image must leave the shape unchanged also.
      with self.cached_session(use_gpu=True):
        img_single = img_np.reshape(single_shape)
        image = constant_op.constant(img_single, shape=single_shape)
        y = image_ops.resize_images(image, [target_height, target_width],
                                    self.METHODS[0])
        yshape = array_ops.shape(y)
        newshape = self.evaluate(yshape)
        self.assertAllEqual(single_shape, newshape)

  @test_util.run_deprecated_v1
  def testTensorArguments(self):
    img_shape = [1, 6, 4, 1]
    single_shape = [6, 4, 1]
    # This test is also conducted with int8, so 127 is the maximum
    # value that can be used.
    data = [
        127, 127, 64, 64, 127, 127, 64, 64, 64, 64, 127, 127, 64, 64, 127, 127,
        50, 50, 100, 100, 50, 50, 100, 100
    ]
    new_size = array_ops.placeholder(dtypes.int32, shape=(2))

    img_np = np.array(data, dtype=np.uint8).reshape(img_shape)

    for method in self.METHODS:
      with self.cached_session(use_gpu=True) as sess:
        image = constant_op.constant(img_np, shape=img_shape)
        y = image_ops.resize_images(image, new_size, method)
        yshape = array_ops.shape(y)
        resized, newshape = sess.run([y, yshape], {new_size: [6, 4]})
        self.assertAllEqual(img_shape, newshape)
        self.assertAllClose(resized, img_np, atol=1e-5)

    # Resizing with a single image must leave the shape unchanged also.
    with self.cached_session(use_gpu=True):
      img_single = img_np.reshape(single_shape)
      image = constant_op.constant(img_single, shape=single_shape)
      y = image_ops.resize_images(image, new_size, self.METHODS[0])
      yshape = array_ops.shape(y)
      resized, newshape = sess.run([y, yshape], {new_size: [6, 4]})
      self.assertAllEqual(single_shape, newshape)
      self.assertAllClose(resized, img_single, atol=1e-5)

    # Incorrect shape.
    with self.assertRaises(ValueError):
      new_size = constant_op.constant(4)
      _ = image_ops.resize_images(image, new_size,
                                  image_ops.ResizeMethodV1.BILINEAR)
    with self.assertRaises(ValueError):
      new_size = constant_op.constant([4])
      _ = image_ops.resize_images(image, new_size,
                                  image_ops.ResizeMethodV1.BILINEAR)
    with self.assertRaises(ValueError):
      new_size = constant_op.constant([1, 2, 3])
      _ = image_ops.resize_images(image, new_size,
                                  image_ops.ResizeMethodV1.BILINEAR)

    # Incorrect dtypes.
    with self.assertRaises(ValueError):
      new_size = constant_op.constant([6.0, 4])
      _ = image_ops.resize_images(image, new_size,
                                  image_ops.ResizeMethodV1.BILINEAR)
    with self.assertRaises(ValueError):
      _ = image_ops.resize_images(image, [6, 4.0],
                                  image_ops.ResizeMethodV1.BILINEAR)
    with self.assertRaises(ValueError):
      _ = image_ops.resize_images(image, [None, 4],
                                  image_ops.ResizeMethodV1.BILINEAR)
    with self.assertRaises(ValueError):
      _ = image_ops.resize_images(image, [6, None],
                                  image_ops.ResizeMethodV1.BILINEAR)

  @test_util.run_deprecated_v1
  def testReturnDtype(self):
    target_shapes = [[6, 4], [3, 2], [
        array_ops.placeholder(dtypes.int32),
        array_ops.placeholder(dtypes.int32)
    ]]
    for nptype in self.TYPES:
      image = array_ops.placeholder(nptype, shape=[1, 6, 4, 1])
      for method in self.METHODS:
        for target_shape in target_shapes:
          y = image_ops.resize_images(image, target_shape, method)
          if (method == image_ops.ResizeMethodV1.NEAREST_NEIGHBOR or
              target_shape == image.shape[1:3]):
            expected_dtype = image.dtype
          else:
            expected_dtype = dtypes.float32
          self.assertEqual(y.dtype, expected_dtype)

  @test_util.disable_xla("align_corners=False not supported by XLA")
  def testSumTensor(self):
    img_shape = [1, 6, 4, 1]
    # This test is also conducted with int8, so 127 is the maximum
    # value that can be used.
    data = [
        127, 127, 64, 64, 127, 127, 64, 64, 64, 64, 127, 127, 64, 64, 127, 127,
        50, 50, 100, 100, 50, 50, 100, 100
    ]
    # Test size where width is specified as a tensor which is a sum
    # of two tensors.
    width_1 = constant_op.constant(1)
    width_2 = constant_op.constant(3)
    width = math_ops.add(width_1, width_2)
    height = constant_op.constant(6)

    img_np = np.array(data, dtype=np.uint8).reshape(img_shape)

    for method in self.METHODS:
      with self.cached_session() as sess:
        image = constant_op.constant(img_np, shape=img_shape)
        y = image_ops.resize_images(image, [height, width], method)
        yshape = array_ops.shape(y)
        resized, newshape = self.evaluate([y, yshape])
        self.assertAllEqual(img_shape, newshape)
        self.assertAllClose(resized, img_np, atol=1e-5)

  @test_util.disable_xla("align_corners=False not supported by XLA")
  def testResizeDown(self):
    # This test is also conducted with int8, so 127 is the maximum
    # value that can be used.
    data = [
        127, 127, 64, 64, 127, 127, 64, 64, 64, 64, 127, 127, 64, 64, 127, 127,
        50, 50, 100, 100, 50, 50, 100, 100
    ]
    expected_data = [127, 64, 64, 127, 50, 100]
    target_height = 3
    target_width = 2

    # Test out 3-D and 4-D image shapes.
    img_shapes = [[1, 6, 4, 1], [6, 4, 1]]
    target_shapes = [[1, target_height, target_width, 1],
                     [target_height, target_width, 1]]

    for target_shape, img_shape in zip(target_shapes, img_shapes):

      for nptype in self.TYPES:
        img_np = np.array(data, dtype=nptype).reshape(img_shape)

        for method in self.METHODS:
          if test.is_gpu_available() and self.shouldRunOnGPU(method, nptype):
            with self.cached_session(use_gpu=True):
              image = constant_op.constant(img_np, shape=img_shape)
              y = image_ops.resize_images(image, [target_height, target_width],
                                          method)
              expected = np.array(expected_data).reshape(target_shape)
              resized = self.evaluate(y)
              self.assertAllClose(resized, expected, atol=1e-5)

  @test_util.disable_xla("align_corners=False not supported by XLA")
  def testResizeUpAlignCornersFalse(self):
    img_shape = [1, 3, 2, 1]
    data = [64, 32, 32, 64, 50, 100]
    target_height = 6
    target_width = 4
    expected_data = {}
    expected_data[image_ops.ResizeMethodV1.BILINEAR] = [
        64.0, 48.0, 32.0, 32.0, 48.0, 48.0, 48.0, 48.0, 32.0, 48.0, 64.0, 64.0,
        41.0, 61.5, 82.0, 82.0, 50.0, 75.0, 100.0, 100.0, 50.0, 75.0, 100.0,
        100.0
    ]
    expected_data[image_ops.ResizeMethodV1.NEAREST_NEIGHBOR] = [
        64.0, 64.0, 32.0, 32.0, 64.0, 64.0, 32.0, 32.0, 32.0, 32.0, 64.0, 64.0,
        32.0, 32.0, 64.0, 64.0, 50.0, 50.0, 100.0, 100.0, 50.0, 50.0, 100.0,
        100.0
    ]
    expected_data[image_ops.ResizeMethodV1.AREA] = [
        64.0, 64.0, 32.0, 32.0, 64.0, 64.0, 32.0, 32.0, 32.0, 32.0, 64.0, 64.0,
        32.0, 32.0, 64.0, 64.0, 50.0, 50.0, 100.0, 100.0, 50.0, 50.0, 100.0,
        100.0
    ]

    for nptype in self.TYPES:
      for method in [
          image_ops.ResizeMethodV1.BILINEAR,
          image_ops.ResizeMethodV1.NEAREST_NEIGHBOR,
          image_ops.ResizeMethodV1.AREA
      ]:
        with self.cached_session(use_gpu=True):
          img_np = np.array(data, dtype=nptype).reshape(img_shape)
          image = constant_op.constant(img_np, shape=img_shape)
          y = image_ops.resize_images(
              image, [target_height, target_width], method, align_corners=False)
          resized = self.evaluate(y)
          expected = np.array(expected_data[method]).reshape(
              [1, target_height, target_width, 1])
          self.assertAllClose(resized, expected, atol=1e-05)

  def testResizeUpAlignCornersTrue(self):
    img_shape = [1, 3, 2, 1]
    data = [6, 3, 3, 6, 6, 9]
    target_height = 5
    target_width = 4
    expected_data = {}
    expected_data[image_ops.ResizeMethodV1.BILINEAR] = [
        6.0, 5.0, 4.0, 3.0, 4.5, 4.5, 4.5, 4.5, 3.0, 4.0, 5.0, 6.0, 4.5, 5.5,
        6.5, 7.5, 6.0, 7.0, 8.0, 9.0
    ]
    expected_data[image_ops.ResizeMethodV1.NEAREST_NEIGHBOR] = [
        6.0, 6.0, 3.0, 3.0, 3.0, 3.0, 6.0, 6.0, 3.0, 3.0, 6.0, 6.0, 6.0, 6.0,
        9.0, 9.0, 6.0, 6.0, 9.0, 9.0
    ]
    # TODO(b/37749740): Improve alignment of ResizeMethodV1.AREA when
    # align_corners=True.
    expected_data[image_ops.ResizeMethodV1.AREA] = [
        6.0, 6.0, 6.0, 3.0, 6.0, 6.0, 6.0, 3.0, 3.0, 3.0, 3.0, 6.0, 3.0, 3.0,
        3.0, 6.0, 6.0, 6.0, 6.0, 9.0
    ]

    for nptype in self.TYPES:
      for method in [
          image_ops.ResizeMethodV1.BILINEAR,
          image_ops.ResizeMethodV1.NEAREST_NEIGHBOR,
          image_ops.ResizeMethodV1.AREA
      ]:
        with self.cached_session(use_gpu=True):
          img_np = np.array(data, dtype=nptype).reshape(img_shape)
          image = constant_op.constant(img_np, shape=img_shape)
          y = image_ops.resize_images(
              image, [target_height, target_width], method, align_corners=True)
          resized = self.evaluate(y)
          expected = np.array(expected_data[method]).reshape(
              [1, target_height, target_width, 1])
          self.assertAllClose(resized, expected, atol=1e-05)

  def testResizeUpBicubic(self):
    img_shape = [1, 6, 6, 1]
    data = [
        128, 128, 64, 64, 128, 128, 64, 64, 64, 64, 128, 128, 64, 64, 128, 128,
        50, 50, 100, 100, 50, 50, 100, 100, 50, 50, 100, 100, 50, 50, 100, 100,
        50, 50, 100, 100
    ]
    img_np = np.array(data, dtype=np.uint8).reshape(img_shape)

    target_height = 8
    target_width = 8
    expected_data = [
        128, 135, 96, 55, 64, 114, 134, 128, 78, 81, 68, 52, 57, 118, 144, 136,
        55, 49, 79, 109, 103, 89, 83, 84, 74, 70, 95, 122, 115, 69, 49, 55, 100,
        105, 75, 43, 50, 89, 105, 100, 57, 54, 74, 96, 91, 65, 55, 58, 70, 69,
        75, 81, 80, 72, 69, 70, 105, 112, 75, 36, 45, 92, 111, 105
    ]

    with self.cached_session(use_gpu=True):
      image = constant_op.constant(img_np, shape=img_shape)
      y = image_ops.resize_images(image, [target_height, target_width],
                                  image_ops.ResizeMethodV1.BICUBIC)
      resized = self.evaluate(y)
      expected = np.array(expected_data).reshape(
          [1, target_height, target_width, 1])
      self.assertAllClose(resized, expected, atol=1)

  def testResizeDownArea(self):
    img_shape = [1, 6, 6, 1]
    data = [
        128, 64, 32, 16, 8, 4, 4, 8, 16, 32, 64, 128, 128, 64, 32, 16, 8, 4, 5,
        10, 15, 20, 25, 30, 30, 25, 20, 15, 10, 5, 5, 10, 15, 20, 25, 30
    ]
    img_np = np.array(data, dtype=np.uint8).reshape(img_shape)

    target_height = 4
    target_width = 4
    expected_data = [
        73, 33, 23, 39, 73, 33, 23, 39, 14, 16, 19, 21, 14, 16, 19, 21
    ]

    with self.cached_session(use_gpu=True):
      image = constant_op.constant(img_np, shape=img_shape)
      y = image_ops.resize_images(image, [target_height, target_width],
                                  image_ops.ResizeMethodV1.AREA)
      expected = np.array(expected_data).reshape(
          [1, target_height, target_width, 1])
      resized = self.evaluate(y)
      self.assertAllClose(resized, expected, atol=1)

  @test_util.disable_xla("align_corners=False not supported by XLA")
  def testCompareNearestNeighbor(self):
    if test.is_gpu_available():
      input_shape = [1, 5, 6, 3]
      target_height = 8
      target_width = 12
      for nptype in [np.float32, np.float64]:
        for align_corners in [True, False]:
          img_np = np.arange(
              0, np.prod(input_shape), dtype=nptype).reshape(input_shape)
          with self.cached_session(use_gpu=True):
            image = constant_op.constant(img_np, shape=input_shape)
            new_size = constant_op.constant([target_height, target_width])
            out_op = image_ops.resize_images(
                image,
                new_size,
                image_ops.ResizeMethodV1.NEAREST_NEIGHBOR,
                align_corners=align_corners)
            gpu_val = self.evaluate(out_op)
          with self.cached_session(use_gpu=False):
            image = constant_op.constant(img_np, shape=input_shape)
            new_size = constant_op.constant([target_height, target_width])
            out_op = image_ops.resize_images(
                image,
                new_size,
                image_ops.ResizeMethodV1.NEAREST_NEIGHBOR,
                align_corners=align_corners)
            cpu_val = self.evaluate(out_op)
          self.assertAllClose(cpu_val, gpu_val, rtol=1e-5, atol=1e-5)

  def testCompareBilinear(self):
    if test.is_gpu_available():
      input_shape = [1, 5, 6, 3]
      target_height = 8
      target_width = 12
      for nptype in [np.float32, np.float64]:
        for align_corners in [True, False]:
          img_np = np.arange(
              0, np.prod(input_shape), dtype=nptype).reshape(input_shape)
          value = {}
          for use_gpu in [True, False]:
            with self.cached_session(use_gpu=use_gpu):
              image = constant_op.constant(img_np, shape=input_shape)
              new_size = constant_op.constant([target_height, target_width])
              out_op = image_ops.resize_images(
                  image,
                  new_size,
                  image_ops.ResizeMethodV1.BILINEAR,
                  align_corners=align_corners)
              value[use_gpu] = self.evaluate(out_op)
          self.assertAllClose(value[True], value[False], rtol=1e-5, atol=1e-5)

  @test_util.run_deprecated_v1
  def testShapeInference(self):
    self._assertShapeInference([50, 60, 3], [55, 66], [55, 66, 3])
    self._assertShapeInference([55, 66, 3], [55, 66], [55, 66, 3])
    self._assertShapeInference([59, 69, 3], [55, 66], [55, 66, 3])
    self._assertShapeInference([50, 69, 3], [55, 66], [55, 66, 3])
    self._assertShapeInference([59, 60, 3], [55, 66], [55, 66, 3])
    self._assertShapeInference([None, 60, 3], [55, 66], [55, 66, 3])
    self._assertShapeInference([None, 66, 3], [55, 66], [55, 66, 3])
    self._assertShapeInference([None, 69, 3], [55, 66], [55, 66, 3])
    self._assertShapeInference([50, None, 3], [55, 66], [55, 66, 3])
    self._assertShapeInference([55, None, 3], [55, 66], [55, 66, 3])
    self._assertShapeInference([59, None, 3], [55, 66], [55, 66, 3])
    self._assertShapeInference([None, None, 3], [55, 66], [55, 66, 3])
    self._assertShapeInference([50, 60, None], [55, 66], [55, 66, None])
    self._assertShapeInference([55, 66, None], [55, 66], [55, 66, None])
    self._assertShapeInference([59, 69, None], [55, 66], [55, 66, None])
    self._assertShapeInference([50, 69, None], [55, 66], [55, 66, None])
    self._assertShapeInference([59, 60, None], [55, 66], [55, 66, None])
    self._assertShapeInference([None, None, None], [55, 66], [55, 66, None])

  @test_util.run_deprecated_v1
  def testNameScope(self):
    img_shape = [1, 3, 2, 1]
    with self.cached_session(use_gpu=True):
      single_image = array_ops.placeholder(dtypes.float32, shape=[50, 60, 3])
      y = image_ops.resize_images(single_image, [55, 66])
      self.assertTrue(y.op.name.startswith("resize"))

  def _ResizeImageCall(self, x, max_h, max_w, preserve_aspect_ratio,
                       use_tensor_inputs):
    if use_tensor_inputs:
      target_max = ops.convert_to_tensor([max_h, max_w])
      x_tensor = array_ops.placeholder(x.dtype, shape=[None] * x.ndim)
      feed_dict = {x_tensor: x}
    else:
      target_max = [max_h, max_w]
      x_tensor = x
      feed_dict = {}

    y = image_ops.resize_images(x_tensor, target_max,
                                preserve_aspect_ratio=preserve_aspect_ratio)

    with self.cached_session(use_gpu=True):
      return y.eval(feed_dict=feed_dict)

  def _assertResizeEqual(self, x, x_shape, y, y_shape,
                         preserve_aspect_ratio=True,
                         use_tensor_inputs_options=None):
    use_tensor_inputs_options = use_tensor_inputs_options or [False, True]
    target_height, target_width, _ = y_shape
    x = np.array(x).reshape(x_shape)
    y = np.array(y).reshape(y_shape)

    for use_tensor_inputs in use_tensor_inputs_options:
      y_tf = self._ResizeImageCall(x, target_height, target_width,
                                   preserve_aspect_ratio, use_tensor_inputs)
      self.assertAllClose(y, y_tf)

  def _assertResizeCheckShape(self, x, x_shape, target_shape,
                              y_shape, preserve_aspect_ratio=True,
                              use_tensor_inputs_options=None):
    use_tensor_inputs_options = use_tensor_inputs_options or [False, True]
    target_height, target_width = target_shape
    x = np.array(x).reshape(x_shape)
    y = np.zeros(y_shape)

    for use_tensor_inputs in use_tensor_inputs_options:
      y_tf = self._ResizeImageCall(x, target_height, target_width,
                                   preserve_aspect_ratio, use_tensor_inputs)
      self.assertShapeEqual(y, ops.convert_to_tensor(y_tf))

  @test_util.run_deprecated_v1
  def testPreserveAspectRatioMultipleImages(self):
    x_shape = [10, 100, 100, 10]
    x = np.random.uniform(size=x_shape)

    self._assertResizeCheckShape(x, x_shape, [250, 250], [10, 250, 250, 10],
                                 preserve_aspect_ratio=False)

  @test_util.run_deprecated_v1
  def testPreserveAspectRatioNoOp(self):
    x_shape = [10, 10, 10]
    x = np.random.uniform(size=x_shape)

    self._assertResizeEqual(x, x_shape, x, x_shape)

  @test_util.run_deprecated_v1
  def testPreserveAspectRatioSmaller(self):
    x_shape = [100, 100, 10]
    x = np.random.uniform(size=x_shape)

    self._assertResizeCheckShape(x, x_shape, [75, 50], [50, 50, 10])

  @test_util.run_deprecated_v1
  def testPreserveAspectRatioSmallerMultipleImages(self):
    x_shape = [10, 100, 100, 10]
    x = np.random.uniform(size=x_shape)

    self._assertResizeCheckShape(x, x_shape, [75, 50], [10, 50, 50, 10])

  @test_util.run_deprecated_v1
  def testPreserveAspectRatioLarger(self):
    x_shape = [100, 100, 10]
    x = np.random.uniform(size=x_shape)

    self._assertResizeCheckShape(x, x_shape, [150, 200], [150, 150, 10])

  @test_util.run_deprecated_v1
  def testPreserveAspectRatioSameRatio(self):
    x_shape = [1920, 1080, 3]
    x = np.random.uniform(size=x_shape)

    self._assertResizeCheckShape(x, x_shape, [3840, 2160], [3840, 2160, 3])

  @test_util.run_deprecated_v1
  def testPreserveAspectRatioSquare(self):
    x_shape = [299, 299, 3]
    x = np.random.uniform(size=x_shape)

    self._assertResizeCheckShape(x, x_shape, [320, 320], [320, 320, 3])


class ResizeImageWithPadV1Test(test_util.TensorFlowTestCase):

  def _ResizeImageWithPad(self, x, target_height, target_width,
                          use_tensor_inputs):
    if use_tensor_inputs:
      target_height = ops.convert_to_tensor(target_height)
      target_width = ops.convert_to_tensor(target_width)
      x_tensor = array_ops.placeholder(x.dtype, shape=[None] * x.ndim)
      feed_dict = {x_tensor: x}
    else:
      x_tensor = x
      feed_dict = {}

    y = image_ops.resize_image_with_pad_v1(x_tensor, target_height,
                                           target_width)
    if not use_tensor_inputs:
      self.assertTrue(y.get_shape().is_fully_defined())

    with self.cached_session(use_gpu=True):
      return y.eval(feed_dict=feed_dict)

  def _assertReturns(self,
                     x,
                     x_shape,
                     y,
                     y_shape,
                     use_tensor_inputs_options=None):
    use_tensor_inputs_options = use_tensor_inputs_options or [False, True]
    target_height, target_width, _ = y_shape
    x = np.array(x).reshape(x_shape)
    y = np.array(y).reshape(y_shape)

    for use_tensor_inputs in use_tensor_inputs_options:
      y_tf = self._ResizeImageWithPad(x, target_height, target_width,
                                      use_tensor_inputs)
      self.assertAllClose(y, y_tf)

  def _assertRaises(self,
                    x,
                    x_shape,
                    target_height,
                    target_width,
                    err_msg,
                    use_tensor_inputs_options=None):
    use_tensor_inputs_options = use_tensor_inputs_options or [False, True]
    x = np.array(x).reshape(x_shape)

    for use_tensor_inputs in use_tensor_inputs_options:
      try:
        self._ResizeImageWithPad(x, target_height, target_width,
                                 use_tensor_inputs)
      except Exception as e:  # pylint: disable=broad-except
        if err_msg not in str(e):
          raise
      else:
        raise AssertionError("Exception not raised: %s" % err_msg)

  def _assertShapeInference(self, pre_shape, height, width, post_shape):
    image = array_ops.placeholder(dtypes.float32, shape=pre_shape)
    y = image_ops.resize_image_with_pad_v1(image, height, width)
    self.assertEqual(y.get_shape().as_list(), post_shape)

  @test_util.run_deprecated_v1
  def testNoOp(self):
    x_shape = [10, 10, 10]
    x = np.random.uniform(size=x_shape)

    self._assertReturns(x, x_shape, x, x_shape)

  @test_util.run_deprecated_v1
  def testPad(self):
    # Reduce vertical dimension
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    x_shape = [2, 4, 1]

    y = [0, 1, 3, 0]
    y_shape = [1, 4, 1]

    self._assertReturns(x, x_shape, y, y_shape)

    # Reduce horizontal dimension
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    x_shape = [2, 4, 1]

    y = [1, 3, 0, 0]
    y_shape = [2, 2, 1]

    self._assertReturns(x, x_shape, y, y_shape)

    x = [1, 2, 3, 4, 5, 6, 7, 8]
    x_shape = [2, 4, 1]

    y = [1, 3]
    y_shape = [1, 2, 1]

    self._assertReturns(x, x_shape, y, y_shape)


# half_pixel_centers not supported by XLA
@test_util.for_all_test_methods(test_util.disable_xla, "b/127616992")
class ResizeImageWithPadV2Test(test_util.TensorFlowTestCase):

  def _ResizeImageWithPad(self, x, target_height, target_width,
                          use_tensor_inputs):
    if use_tensor_inputs:
      target_height = ops.convert_to_tensor(target_height)
      target_width = ops.convert_to_tensor(target_width)
      x_tensor = array_ops.placeholder(x.dtype, shape=[None] * x.ndim)
      feed_dict = {x_tensor: x}
    else:
      x_tensor = x
      feed_dict = {}

    y = image_ops.resize_image_with_pad_v2(x_tensor, target_height,
                                           target_width)
    if not use_tensor_inputs:
      self.assertTrue(y.get_shape().is_fully_defined())

    with self.cached_session(use_gpu=True):
      return y.eval(feed_dict=feed_dict)

  def _assertReturns(self,
                     x,
                     x_shape,
                     y,
                     y_shape,
                     use_tensor_inputs_options=None):
    use_tensor_inputs_options = use_tensor_inputs_options or [False, True]
    target_height, target_width, _ = y_shape
    x = np.array(x).reshape(x_shape)
    y = np.array(y).reshape(y_shape)

    for use_tensor_inputs in use_tensor_inputs_options:
      y_tf = self._ResizeImageWithPad(x, target_height, target_width,
                                      use_tensor_inputs)
      self.assertAllClose(y, y_tf)

  def _assertRaises(self,
                    x,
                    x_shape,
                    target_height,
                    target_width,
                    err_msg,
                    use_tensor_inputs_options=None):
    use_tensor_inputs_options = use_tensor_inputs_options or [False, True]
    x = np.array(x).reshape(x_shape)

    for use_tensor_inputs in use_tensor_inputs_options:
      try:
        self._ResizeImageWithPad(x, target_height, target_width,
                                 use_tensor_inputs)
      except Exception as e:  # pylint: disable=broad-except
        if err_msg not in str(e):
          raise
      else:
        raise AssertionError("Exception not raised: %s" % err_msg)

  def _assertShapeInference(self, pre_shape, height, width, post_shape):
    image = array_ops.placeholder(dtypes.float32, shape=pre_shape)
    y = image_ops.resize_image_with_pad_v1(image, height, width)
    self.assertEqual(y.get_shape().as_list(), post_shape)


  @test_util.run_deprecated_v1
  def testNoOp(self):
    x_shape = [10, 10, 10]
    x = np.random.uniform(size=x_shape)

    self._assertReturns(x, x_shape, x, x_shape)

  @test_util.run_deprecated_v1
  def testPad(self):
    # Reduce vertical dimension
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    x_shape = [2, 4, 1]

    y = [0, 3.5, 5.5, 0]
    y_shape = [1, 4, 1]

    self._assertReturns(x, x_shape, y, y_shape)

    # Reduce horizontal dimension
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    x_shape = [2, 4, 1]

    y = [3.5, 5.5, 0, 0]
    y_shape = [2, 2, 1]

    self._assertReturns(x, x_shape, y, y_shape)

    x = [1, 2, 3, 4, 5, 6, 7, 8]
    x_shape = [2, 4, 1]

    y = [3.5, 5.5]
    y_shape = [1, 2, 1]

    self._assertReturns(x, x_shape, y, y_shape)


class ResizeImageWithCropOrPadTest(test_util.TensorFlowTestCase):

  def _ResizeImageWithCropOrPad(self, x, target_height, target_width,
                                use_tensor_inputs):
    if use_tensor_inputs:
      target_height = ops.convert_to_tensor(target_height)
      target_width = ops.convert_to_tensor(target_width)
      x_tensor = array_ops.placeholder(x.dtype, shape=[None] * x.ndim)
      feed_dict = {x_tensor: x}
    else:
      x_tensor = x
      feed_dict = {}

    y = image_ops.resize_image_with_crop_or_pad(x_tensor, target_height,
                                                target_width)
    if not use_tensor_inputs:
      self.assertTrue(y.get_shape().is_fully_defined())

    with self.cached_session(use_gpu=True):
      return y.eval(feed_dict=feed_dict)

  def _assertReturns(self,
                     x,
                     x_shape,
                     y,
                     y_shape,
                     use_tensor_inputs_options=None):
    use_tensor_inputs_options = use_tensor_inputs_options or [False, True]
    target_height, target_width, _ = y_shape
    x = np.array(x).reshape(x_shape)
    y = np.array(y).reshape(y_shape)

    for use_tensor_inputs in use_tensor_inputs_options:
      y_tf = self._ResizeImageWithCropOrPad(x, target_height, target_width,
                                            use_tensor_inputs)
      self.assertAllClose(y, y_tf)

  def _assertRaises(self,
                    x,
                    x_shape,
                    target_height,
                    target_width,
                    err_msg,
                    use_tensor_inputs_options=None):
    use_tensor_inputs_options = use_tensor_inputs_options or [False, True]
    x = np.array(x).reshape(x_shape)

    for use_tensor_inputs in use_tensor_inputs_options:
      try:
        self._ResizeImageWithCropOrPad(x, target_height, target_width,
                                       use_tensor_inputs)
      except Exception as e:
        if err_msg not in str(e):
          raise
      else:
        raise AssertionError("Exception not raised: %s" % err_msg)

  def _assertShapeInference(self, pre_shape, height, width, post_shape):
    image = array_ops.placeholder(dtypes.float32, shape=pre_shape)
    y = image_ops.resize_image_with_crop_or_pad(image, height, width)
    self.assertEqual(y.get_shape().as_list(), post_shape)

  @test_util.run_deprecated_v1
  def testNoOp(self):
    x_shape = [10, 10, 10]
    x = np.random.uniform(size=x_shape)

    self._assertReturns(x, x_shape, x, x_shape)

  @test_util.run_deprecated_v1
  def testPad(self):
    # Pad even along col.
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    x_shape = [2, 4, 1]

    y = [0, 1, 2, 3, 4, 0, 0, 5, 6, 7, 8, 0]
    y_shape = [2, 6, 1]

    self._assertReturns(x, x_shape, y, y_shape)

    # Pad odd along col.
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    x_shape = [2, 4, 1]

    y = [0, 1, 2, 3, 4, 0, 0, 0, 5, 6, 7, 8, 0, 0]
    y_shape = [2, 7, 1]

    self._assertReturns(x, x_shape, y, y_shape)

    # Pad even along row.
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    x_shape = [2, 4, 1]

    y = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0]
    y_shape = [4, 4, 1]

    self._assertReturns(x, x_shape, y, y_shape)

    # Pad odd along row.
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    x_shape = [2, 4, 1]

    y = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0]
    y_shape = [5, 4, 1]

    self._assertReturns(x, x_shape, y, y_shape)

  @test_util.run_deprecated_v1
  def testCrop(self):
    # Crop even along col.
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    x_shape = [2, 4, 1]

    y = [2, 3, 6, 7]
    y_shape = [2, 2, 1]

    self._assertReturns(x, x_shape, y, y_shape)

    # Crop odd along col.
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    x_shape = [2, 6, 1]

    y = [2, 3, 4, 8, 9, 10]
    y_shape = [2, 3, 1]

    self._assertReturns(x, x_shape, y, y_shape)

    # Crop even along row.
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    x_shape = [4, 2, 1]

    y = [3, 4, 5, 6]
    y_shape = [2, 2, 1]

    self._assertReturns(x, x_shape, y, y_shape)

    # Crop odd along row.
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    x_shape = [8, 2, 1]

    y = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    y_shape = [5, 2, 1]

    self._assertReturns(x, x_shape, y, y_shape)

  @test_util.run_deprecated_v1
  def testCropAndPad(self):
    # Pad along row but crop along col.
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    x_shape = [2, 4, 1]

    y = [0, 0, 2, 3, 6, 7, 0, 0]
    y_shape = [4, 2, 1]

    self._assertReturns(x, x_shape, y, y_shape)

    # Crop along row but pad along col.
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    x_shape = [4, 2, 1]

    y = [0, 3, 4, 0, 0, 5, 6, 0]
    y_shape = [2, 4, 1]

    self._assertReturns(x, x_shape, y, y_shape)

  @test_util.run_deprecated_v1
  def testShapeInference(self):
    self._assertShapeInference([50, 60, 3], 55, 66, [55, 66, 3])
    self._assertShapeInference([55, 66, 3], 55, 66, [55, 66, 3])
    self._assertShapeInference([59, 69, 3], 55, 66, [55, 66, 3])
    self._assertShapeInference([50, 69, 3], 55, 66, [55, 66, 3])
    self._assertShapeInference([59, 60, 3], 55, 66, [55, 66, 3])
    self._assertShapeInference([None, 60, 3], 55, 66, [55, 66, 3])
    self._assertShapeInference([None, 66, 3], 55, 66, [55, 66, 3])
    self._assertShapeInference([None, 69, 3], 55, 66, [55, 66, 3])
    self._assertShapeInference([50, None, 3], 55, 66, [55, 66, 3])
    self._assertShapeInference([55, None, 3], 55, 66, [55, 66, 3])
    self._assertShapeInference([59, None, 3], 55, 66, [55, 66, 3])
    self._assertShapeInference([None, None, 3], 55, 66, [55, 66, 3])
    self._assertShapeInference([50, 60, None], 55, 66, [55, 66, None])
    self._assertShapeInference([55, 66, None], 55, 66, [55, 66, None])
    self._assertShapeInference([59, 69, None], 55, 66, [55, 66, None])
    self._assertShapeInference([50, 69, None], 55, 66, [55, 66, None])
    self._assertShapeInference([59, 60, None], 55, 66, [55, 66, None])
    self._assertShapeInference([None, None, None], 55, 66, [55, 66, None])
    self._assertShapeInference(None, 55, 66, [55, 66, None])

  @test_util.run_deprecated_v1
  def testNon3DInput(self):
    # Input image is not 3D
    x = [0] * 15
    target_height, target_width = [4, 4]

    for x_shape in ([3, 5],):
      self._assertRaises(x, x_shape, target_height, target_width,
                         "must have either 3 or 4 dimensions.")

    for x_shape in ([1, 3, 5, 1, 1],):
      self._assertRaises(x, x_shape, target_height, target_width,
                         "must have either 3 or 4 dimensions.")

  @test_util.run_deprecated_v1
  def testZeroLengthInput(self):
    # Input image has 0-length dimension(s).
    target_height, target_width = [1, 1]
    x = []

    for x_shape in ([0, 2, 2], [2, 0, 2], [2, 2, 0]):
      self._assertRaises(
          x,
          x_shape,
          target_height,
          target_width,
          "inner 3 dims of 'image.shape' must be > 0",
          use_tensor_inputs_options=[False])

      # The original error message does not contain back slashes. However, they
      # are added by either the assert op or the runtime. If this behavior
      # changes in the future, the match string will also needs to be changed.
      self._assertRaises(
          x,
          x_shape,
          target_height,
          target_width,
          "inner 3 dims of \\'image.shape\\' must be > 0",
          use_tensor_inputs_options=[True])

  @test_util.run_deprecated_v1
  def testBadParams(self):
    x_shape = [4, 4, 1]
    x = np.zeros(x_shape)

    # target_height <= 0
    target_height, target_width = [0, 5]
    self._assertRaises(x, x_shape, target_height, target_width,
                       "target_height must be > 0")

    # target_width <= 0
    target_height, target_width = [5, 0]
    self._assertRaises(x, x_shape, target_height, target_width,
                       "target_width must be > 0")

  @test_util.run_deprecated_v1
  def testNameScope(self):
    image = array_ops.placeholder(dtypes.float32, shape=[50, 60, 3])
    y = image_ops.resize_image_with_crop_or_pad(image, 55, 66)
    self.assertTrue(y.op.name.startswith("resize_image_with_crop_or_pad"))


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
    path = ("tensorflow/core/lib/jpeg/testdata/"
            "jpeg_merge_test1.jpg")
    with self.cached_session(use_gpu=True) as sess:
      jpeg0 = io_ops.read_file(path)
      image0 = image_ops.decode_jpeg(jpeg0)
      image1 = image_ops.decode_jpeg(image_ops.encode_jpeg(image0))
      jpeg0, image0, image1 = self.evaluate([jpeg0, image0, image1])
      self.assertEqual(len(jpeg0), 3771)
      self.assertEqual(image0.shape, (256, 128, 3))
      self.assertLess(self.averageError(image0, image1), 1.4)

  def testCmyk(self):
    # Confirm that CMYK reads in as RGB
    base = "tensorflow/core/lib/jpeg/testdata"
    rgb_path = os.path.join(base, "jpeg_merge_test1.jpg")
    cmyk_path = os.path.join(base, "jpeg_merge_test1_cmyk.jpg")
    shape = 256, 128, 3
    for channels in 3, 0:
      with self.cached_session(use_gpu=True) as sess:
        rgb = image_ops.decode_jpeg(
            io_ops.read_file(rgb_path), channels=channels)
        cmyk = image_ops.decode_jpeg(
            io_ops.read_file(cmyk_path), channels=channels)
        rgb, cmyk = self.evaluate([rgb, cmyk])
        self.assertEqual(rgb.shape, shape)
        self.assertEqual(cmyk.shape, shape)
        error = self.averageError(rgb, cmyk)
        self.assertLess(error, 4)

  def testCropAndDecodeJpeg(self):
    with self.cached_session() as sess:
      # Encode it, then decode it, then encode it
      base = "tensorflow/core/lib/jpeg/testdata"
      jpeg0 = io_ops.read_file(os.path.join(base, "jpeg_merge_test1.jpg"))

      h, w, _ = 256, 128, 3
      crop_windows = [[0, 0, 5, 5], [0, 0, 5, w], [0, 0, h, 5],
                      [h - 6, w - 5, 6, 5], [6, 5, 15, 10], [0, 0, h, w]]
      for crop_window in crop_windows:
        # Explicit two stages: decode + crop.
        image1 = image_ops.decode_jpeg(jpeg0)
        y, x, h, w = crop_window
        image1_crop = image_ops.crop_to_bounding_box(image1, y, x, h, w)

        # Combined decode+crop.
        image2 = image_ops.decode_and_crop_jpeg(jpeg0, crop_window)

        # Combined decode+crop should have the same shape inference
        self.assertAllEqual(image1_crop.get_shape().as_list(),
                            image2.get_shape().as_list())

        # CropAndDecode should be equal to DecodeJpeg+Crop.
        image1_crop, image2 = self.evaluate([image1_crop, image2])
        self.assertAllEqual(image1_crop, image2)

  @test_util.run_deprecated_v1
  def testCropAndDecodeJpegWithInvalidCropWindow(self):
    with self.cached_session() as sess:
      # Encode it, then decode it, then encode it
      base = "tensorflow/core/lib/jpeg/testdata"
      jpeg0 = io_ops.read_file(os.path.join(base, "jpeg_merge_test1.jpg"))

      h, w, _ = 256, 128, 3
      # Invalid crop windows.
      crop_windows = [[-1, 11, 11, 11], [11, -1, 11, 11], [11, 11, -1, 11],
                      [11, 11, 11, -1], [11, 11, 0, 11], [11, 11, 11, 0],
                      [0, 0, h + 1, w], [0, 0, h, w + 1]]
      for crop_window in crop_windows:
        result = image_ops.decode_and_crop_jpeg(jpeg0, crop_window)
        with self.assertRaisesWithPredicateMatch(
            errors.InvalidArgumentError,
            lambda e: "Invalid JPEG data or crop window" in str(e)):
          self.evaluate(result)

  def testSynthetic(self):
    with self.cached_session(use_gpu=True) as sess:
      # Encode it, then decode it, then encode it
      image0 = constant_op.constant(_SimpleColorRamp())
      jpeg0 = image_ops.encode_jpeg(image0)
      image1 = image_ops.decode_jpeg(jpeg0, dct_method="INTEGER_ACCURATE")
      image2 = image_ops.decode_jpeg(
          image_ops.encode_jpeg(image1), dct_method="INTEGER_ACCURATE")
      jpeg0, image0, image1, image2 = self.evaluate(
          [jpeg0, image0, image1, image2])

      # The decoded-encoded image should be similar to the input
      self.assertLess(self.averageError(image0, image1), 0.6)

      # We should be very close to a fixpoint
      self.assertLess(self.averageError(image1, image2), 0.02)

      # Smooth ramps compress well (input size is 153600)
      self.assertGreaterEqual(len(jpeg0), 5000)
      self.assertLessEqual(len(jpeg0), 6000)

  def testSyntheticFasterAlgorithm(self):
    with self.cached_session(use_gpu=True) as sess:
      # Encode it, then decode it, then encode it
      image0 = constant_op.constant(_SimpleColorRamp())
      jpeg0 = image_ops.encode_jpeg(image0)
      image1 = image_ops.decode_jpeg(jpeg0, dct_method="INTEGER_FAST")
      image2 = image_ops.decode_jpeg(
          image_ops.encode_jpeg(image1), dct_method="INTEGER_FAST")
      jpeg0, image0, image1, image2 = self.evaluate(
          [jpeg0, image0, image1, image2])

      # The decoded-encoded image should be similar to the input, but
      # note this is worse than the slower algorithm because it is
      # less accurate.
      self.assertLess(self.averageError(image0, image1), 0.95)

      # Repeated compression / decompression will have a higher error
      # with a lossier algorithm.
      self.assertLess(self.averageError(image1, image2), 1.05)

      # Smooth ramps compress well (input size is 153600)
      self.assertGreaterEqual(len(jpeg0), 5000)
      self.assertLessEqual(len(jpeg0), 6000)

  def testDefaultDCTMethodIsIntegerFast(self):
    with self.cached_session(use_gpu=True) as sess:
      # Compare decoding with both dct_option=INTEGER_FAST and
      # default.  They should be the same.
      image0 = constant_op.constant(_SimpleColorRamp())
      jpeg0 = image_ops.encode_jpeg(image0)
      image1 = image_ops.decode_jpeg(jpeg0, dct_method="INTEGER_FAST")
      image2 = image_ops.decode_jpeg(jpeg0)
      image1, image2 = self.evaluate([image1, image2])

      # The images should be the same.
      self.assertAllClose(image1, image2)

  @test_util.run_deprecated_v1
  def testShape(self):
    with self.cached_session(use_gpu=True) as sess:
      jpeg = constant_op.constant("nonsense")
      for channels in 0, 1, 3:
        image = image_ops.decode_jpeg(jpeg, channels=channels)
        self.assertEqual(image.get_shape().as_list(),
                         [None, None, channels or None])

  @test_util.run_deprecated_v1
  def testExtractJpegShape(self):
    # Read a real jpeg and verify shape.
    path = ("tensorflow/core/lib/jpeg/testdata/"
            "jpeg_merge_test1.jpg")
    with self.cached_session(use_gpu=True) as sess:
      jpeg = io_ops.read_file(path)
      # Extract shape without decoding.
      [image_shape] = sess.run([image_ops.extract_jpeg_shape(jpeg)])
      self.assertEqual(image_shape.tolist(), [256, 128, 3])

  @test_util.run_deprecated_v1
  def testExtractJpegShapeforCmyk(self):
    # Read a cmyk jpeg image, and verify its shape.
    path = ("tensorflow/core/lib/jpeg/testdata/"
            "jpeg_merge_test1_cmyk.jpg")
    with self.cached_session(use_gpu=True) as sess:
      jpeg = io_ops.read_file(path)
      [image_shape] = sess.run([image_ops.extract_jpeg_shape(jpeg)])
      # Cmyk jpeg image has 4 channels.
      self.assertEqual(image_shape.tolist(), [256, 128, 4])

  def testRandomJpegQuality(self):
    # Previous implementation of random_jpeg_quality had a bug.
    # This unit test tests the fixed version, but due to forward compatibility
    # this test can only be done when fixed version is used.
    # Test jpeg quality dynamic randomization.
    with ops.Graph().as_default(), self.test_session():
      np.random.seed(7)
      path = ("tensorflow/core/lib/jpeg/testdata/medium.jpg")
      jpeg = io_ops.read_file(path)
      image = image_ops.decode_jpeg(jpeg)
      random_jpeg_image = image_ops.random_jpeg_quality(image, 40, 100)
      with self.cached_session(use_gpu=True) as sess:
        # Test randomization.
        random_jpeg_images = [sess.run(random_jpeg_image) for _ in range(5)]
        are_images_equal = []
        for i in range(1, len(random_jpeg_images)):
          # Most of them should be different if randomization is occurring
          # correctly.
          are_images_equal.append(
              np.array_equal(random_jpeg_images[0], random_jpeg_images[i]))
        self.assertFalse(all(are_images_equal))

  def testAdjustJpegQuality(self):
    # Test if image_ops.adjust_jpeg_quality works when jpeq quality
    # is an int (not tensor) for backward compatibility.
    with ops.Graph().as_default(), self.test_session():
      np.random.seed(7)
      jpeg_quality = np.random.randint(40, 100)
      path = ("tensorflow/core/lib/jpeg/testdata/medium.jpg")
      jpeg = io_ops.read_file(path)
      image = image_ops.decode_jpeg(jpeg)
      adjust_jpeg_quality_image = image_ops.adjust_jpeg_quality(
          image, jpeg_quality)
      with self.cached_session(use_gpu=True) as sess:
        sess.run(adjust_jpeg_quality_image)

  @test_util.run_deprecated_v1
  def testAdjustJpegQualityShape(self):
    with self.cached_session(use_gpu=True):
      image = constant_op.constant(
          np.arange(24, dtype=np.uint8).reshape([2, 4, 3]))
      adjusted_image = image_ops.adjust_jpeg_quality(image, 80)
      self.assertListEqual(adjusted_image.shape.as_list(),
                           [None, None, 3])


class PngTest(test_util.TensorFlowTestCase):

  def testExisting(self):
    # Read some real PNGs, converting to different channel numbers
    prefix = "tensorflow/core/lib/png/testdata/"
    inputs = ((1, "lena_gray.png"), (4, "lena_rgba.png"),
              (3, "lena_palette.png"), (4, "lena_palette_trns.png"))
    for channels_in, filename in inputs:
      for channels in 0, 1, 3, 4:
        with self.cached_session(use_gpu=True) as sess:
          png0 = io_ops.read_file(prefix + filename)
          image0 = image_ops.decode_png(png0, channels=channels)
          png0, image0 = self.evaluate([png0, image0])
          self.assertEqual(image0.shape, (26, 51, channels or channels_in))
          if channels == channels_in:
            image1 = image_ops.decode_png(image_ops.encode_png(image0))
            self.assertAllEqual(image0, self.evaluate(image1))

  def testSynthetic(self):
    with self.cached_session(use_gpu=True) as sess:
      # Encode it, then decode it
      image0 = constant_op.constant(_SimpleColorRamp())
      png0 = image_ops.encode_png(image0, compression=7)
      image1 = image_ops.decode_png(png0)
      png0, image0, image1 = self.evaluate([png0, image0, image1])

      # PNG is lossless
      self.assertAllEqual(image0, image1)

      # Smooth ramps compress well, but not too well
      self.assertGreaterEqual(len(png0), 400)
      self.assertLessEqual(len(png0), 750)

  def testSyntheticUint16(self):
    with self.cached_session(use_gpu=True) as sess:
      # Encode it, then decode it
      image0 = constant_op.constant(_SimpleColorRamp(), dtype=dtypes.uint16)
      png0 = image_ops.encode_png(image0, compression=7)
      image1 = image_ops.decode_png(png0, dtype=dtypes.uint16)
      png0, image0, image1 = self.evaluate([png0, image0, image1])

      # PNG is lossless
      self.assertAllEqual(image0, image1)

      # Smooth ramps compress well, but not too well
      self.assertGreaterEqual(len(png0), 800)
      self.assertLessEqual(len(png0), 1500)

  def testSyntheticTwoChannel(self):
    with self.cached_session(use_gpu=True) as sess:
      # Strip the b channel from an rgb image to get a two-channel image.
      gray_alpha = _SimpleColorRamp()[:, :, 0:2]
      image0 = constant_op.constant(gray_alpha)
      png0 = image_ops.encode_png(image0, compression=7)
      image1 = image_ops.decode_png(png0)
      png0, image0, image1 = self.evaluate([png0, image0, image1])
      self.assertEqual(2, image0.shape[-1])
      self.assertAllEqual(image0, image1)

  def testSyntheticTwoChannelUint16(self):
    with self.cached_session(use_gpu=True) as sess:
      # Strip the b channel from an rgb image to get a two-channel image.
      gray_alpha = _SimpleColorRamp()[:, :, 0:2]
      image0 = constant_op.constant(gray_alpha, dtype=dtypes.uint16)
      png0 = image_ops.encode_png(image0, compression=7)
      image1 = image_ops.decode_png(png0, dtype=dtypes.uint16)
      png0, image0, image1 = self.evaluate([png0, image0, image1])
      self.assertEqual(2, image0.shape[-1])
      self.assertAllEqual(image0, image1)

  @test_util.run_deprecated_v1
  def testShape(self):
    with self.cached_session(use_gpu=True):
      png = constant_op.constant("nonsense")
      for channels in 0, 1, 3:
        image = image_ops.decode_png(png, channels=channels)
        self.assertEqual(image.get_shape().as_list(),
                         [None, None, channels or None])


class GifTest(test_util.TensorFlowTestCase):

  def _testValid(self, filename):
    # Read some real GIFs
    prefix = "tensorflow/core/lib/gif/testdata/"
    WIDTH = 20
    HEIGHT = 40
    STRIDE = 5
    shape = (12, HEIGHT, WIDTH, 3)

    with self.cached_session(use_gpu=True) as sess:
      gif0 = io_ops.read_file(prefix + filename)
      image0 = image_ops.decode_gif(gif0)
      gif0, image0 = self.evaluate([gif0, image0])

      self.assertEqual(image0.shape, shape)

      for frame_idx, frame in enumerate(image0):
        gt = np.zeros(shape[1:], dtype=np.uint8)
        start = frame_idx * STRIDE
        end = (frame_idx + 1) * STRIDE
        print(frame_idx)
        if end <= WIDTH:
          gt[:, start:end, :] = 255
        else:
          start -= WIDTH
          end -= WIDTH
          gt[start:end, :, :] = 255

        self.assertAllClose(frame, gt)

  def testValid(self):
    self._testValid("scan.gif")
    self._testValid("optimized.gif")

  @test_util.run_deprecated_v1
  def testShape(self):
    with self.cached_session(use_gpu=True) as sess:
      gif = constant_op.constant("nonsense")
      image = image_ops.decode_gif(gif)
      self.assertEqual(image.get_shape().as_list(), [None, None, None, 3])


class ConvertImageTest(test_util.TensorFlowTestCase):

  def _convert(self, original, original_dtype, output_dtype, expected):
    x_np = np.array(original, dtype=original_dtype.as_numpy_dtype())
    y_np = np.array(expected, dtype=output_dtype.as_numpy_dtype())

    with self.cached_session(use_gpu=True):
      image = constant_op.constant(x_np)
      y = image_ops.convert_image_dtype(image, output_dtype)
      self.assertTrue(y.dtype == output_dtype)
      self.assertAllClose(y.eval(), y_np, atol=1e-5)
      if output_dtype in [
          dtypes.float32, dtypes.float64, dtypes.int32, dtypes.int64
      ]:
        y_saturate = image_ops.convert_image_dtype(
            image, output_dtype, saturate=True)
        self.assertTrue(y_saturate.dtype == output_dtype)
        self.assertAllClose(y_saturate.eval(), y_np, atol=1e-5)

  @test_util.run_deprecated_v1
  def testNoConvert(self):
    # Make sure converting to the same data type creates only an identity op
    with self.cached_session(use_gpu=True):
      image = constant_op.constant([1], dtype=dtypes.uint8)
      image_ops.convert_image_dtype(image, dtypes.uint8)
      y = image_ops.convert_image_dtype(image, dtypes.uint8)
      self.assertEquals(y.op.type, "Identity")
      self.assertEquals(y.op.inputs[0], image)

  @test_util.run_deprecated_v1
  def testConvertBetweenInteger(self):
    # Make sure converting to between integer types scales appropriately
    with self.cached_session(use_gpu=True):
      self._convert([0, 255], dtypes.uint8, dtypes.int16, [0, 255 * 128])
      self._convert([0, 32767], dtypes.int16, dtypes.uint8, [0, 255])
      self._convert([0, 2**32], dtypes.int64, dtypes.int32, [0, 1])
      self._convert([0, 1], dtypes.int32, dtypes.int64, [0, 2**32])

  @test_util.run_deprecated_v1
  def testConvertBetweenFloat(self):
    # Make sure converting to between float types does nothing interesting
    with self.cached_session(use_gpu=True):
      self._convert([-1.0, 0, 1.0, 200000], dtypes.float32, dtypes.float64,
                    [-1.0, 0, 1.0, 200000])
      self._convert([-1.0, 0, 1.0, 200000], dtypes.float64, dtypes.float32,
                    [-1.0, 0, 1.0, 200000])

  @test_util.run_deprecated_v1
  def testConvertBetweenIntegerAndFloat(self):
    # Make sure converting from and to a float type scales appropriately
    with self.cached_session(use_gpu=True):
      self._convert([0, 1, 255], dtypes.uint8, dtypes.float32,
                    [0, 1.0 / 255.0, 1])
      self._convert([0, 1.1 / 255.0, 1], dtypes.float32, dtypes.uint8,
                    [0, 1, 255])

  @test_util.run_deprecated_v1
  def testConvertBetweenInt16AndInt8(self):
    with self.cached_session(use_gpu=True):
      # uint8, uint16
      self._convert([0, 255 * 256], dtypes.uint16, dtypes.uint8, [0, 255])
      self._convert([0, 255], dtypes.uint8, dtypes.uint16, [0, 255 * 256])
      # int8, uint16
      self._convert([0, 127 * 2 * 256], dtypes.uint16, dtypes.int8, [0, 127])
      self._convert([0, 127], dtypes.int8, dtypes.uint16, [0, 127 * 2 * 256])
      # int16, uint16
      self._convert([0, 255 * 256], dtypes.uint16, dtypes.int16, [0, 255 * 128])
      self._convert([0, 255 * 128], dtypes.int16, dtypes.uint16, [0, 255 * 256])


class TotalVariationTest(test_util.TensorFlowTestCase):
  """Tests the function total_variation() in image_ops.

  We test a few small handmade examples, as well as
  some larger examples using an equivalent numpy
  implementation of the total_variation() function.

  We do NOT test for overflows and invalid / edge-case arguments.
  """

  def _test(self, x_np, y_np):
    """Test that the TensorFlow implementation of
    total_variation(x_np) calculates the values in y_np.

    Note that these may be float-numbers so we only test
    for approximate equality within some narrow error-bound.
    """

    # Create a TensorFlow session.
    with self.cached_session(use_gpu=True):
      # Add a constant to the TensorFlow graph that holds the input.
      x_tf = constant_op.constant(x_np, shape=x_np.shape)

      # Add ops for calculating the total variation using TensorFlow.
      y = image_ops.total_variation(images=x_tf)

      # Run the TensorFlow session to calculate the result.
      y_tf = self.evaluate(y)

      # Assert that the results are as expected within
      # some small error-bound in case they are float-values.
      self.assertAllClose(y_tf, y_np)

  def _total_variation_np(self, x_np):
    """Calculate the total variation of x_np using numpy.
    This implements the same function as TensorFlow but
    using numpy instead.

    Args:
        x_np: Numpy array with 3 or 4 dimensions.
    """

    dim = len(x_np.shape)

    if dim == 3:
      # Calculate differences for neighboring pixel-values using slices.
      dif1 = x_np[1:, :, :] - x_np[:-1, :, :]
      dif2 = x_np[:, 1:, :] - x_np[:, :-1, :]

      # Sum for all axis.
      sum_axis = None
    elif dim == 4:
      # Calculate differences for neighboring pixel-values using slices.
      dif1 = x_np[:, 1:, :, :] - x_np[:, :-1, :, :]
      dif2 = x_np[:, :, 1:, :] - x_np[:, :, :-1, :]

      # Only sum for the last 3 axis.
      sum_axis = (1, 2, 3)
    else:
      # This should not occur in this test-code.
      pass

    tot_var = np.sum(np.abs(dif1), axis=sum_axis) + \
              np.sum(np.abs(dif2), axis=sum_axis)

    return tot_var

  def _test_tensorflow_vs_numpy(self, x_np):
    """Test the TensorFlow implementation against a numpy implementation.

    Args:
        x_np: Numpy array with 3 or 4 dimensions.
    """

    # Calculate the y-values using the numpy implementation.
    y_np = self._total_variation_np(x_np)

    self._test(x_np, y_np)

  def _generateArray(self, shape):
    """Generate an array of the given shape for use in testing.
    The numbers are calculated as the cumulative sum, which
    causes the difference between neighboring numbers to vary."""

    # Flattened length of the array.
    flat_len = np.prod(shape)

    a = np.array(range(flat_len), dtype=int)
    a = np.cumsum(a)
    a = a.reshape(shape)

    return a

  # TODO(b/133851381): re-enable this test.
  def disabledtestTotalVariationNumpy(self):
    """Test the TensorFlow implementation against a numpy implementation.
    The two implementations are very similar so it is possible that both
    have the same bug, which would not be detected by this test. It is
    therefore necessary to test with manually crafted data as well."""

    # Generate a test-array.
    # This is an 'image' with 100x80 pixels and 3 color channels.
    a = self._generateArray(shape=(100, 80, 3))

    # Test the TensorFlow implementation vs. numpy implementation.
    # We use a numpy implementation to check the results that are
    # calculated using TensorFlow are correct.
    self._test_tensorflow_vs_numpy(a)
    self._test_tensorflow_vs_numpy(a + 1)
    self._test_tensorflow_vs_numpy(-a)
    self._test_tensorflow_vs_numpy(1.1 * a)

    # Expand to a 4-dim array.
    b = a[np.newaxis, :]

    # Combine several variations of the image into a single 4-dim array.
    multi = np.vstack((b, b + 1, -b, 1.1 * b))

    # Test that the TensorFlow function can also handle 4-dim arrays.
    self._test_tensorflow_vs_numpy(multi)

  def testTotalVariationHandmade(self):
    """Test the total variation for a few handmade examples."""

    # We create an image that is 2x2 pixels with 3 color channels.
    # The image is very small so we can check the result by hand.

    # Red color channel.
    # The following are the sum of absolute differences between the pixels.
    # sum row dif = (4-1) + (7-2) = 3 + 5 = 8
    # sum col dif = (2-1) + (7-4) = 1 + 3 = 4
    r = [[1, 2], [4, 7]]

    # Blue color channel.
    # sum row dif = 18 + 29 = 47
    # sum col dif = 7 + 18 = 25
    g = [[11, 18], [29, 47]]

    # Green color channel.
    # sum row dif = 120 + 193 = 313
    # sum col dif = 47 + 120 = 167
    b = [[73, 120], [193, 313]]

    # Combine the 3 color channels into a single 3-dim array.
    # The shape is (2, 2, 3) corresponding to (height, width and color).
    a = np.dstack((r, g, b))

    # Total variation for this image.
    # Sum of all pixel differences = 8 + 4 + 47 + 25 + 313 + 167 = 564
    tot_var = 564

    # Calculate the total variation using TensorFlow and assert it is correct.
    self._test(a, tot_var)

    # If we add 1 to all pixel-values then the total variation is unchanged.
    self._test(a + 1, tot_var)

    # If we negate all pixel-values then the total variation is unchanged.
    self._test(-a, tot_var)

    # Scale the pixel-values by a float. This scales the total variation as
    # well.
    b = 1.1 * a
    self._test(b, 1.1 * tot_var)

    # Scale by another float.
    c = 1.2 * a
    self._test(c, 1.2 * tot_var)

    # Combine these 3 images into a single array of shape (3, 2, 2, 3)
    # where the first dimension is for the image-number.
    multi = np.vstack((a[np.newaxis, :], b[np.newaxis, :], c[np.newaxis, :]))

    # Check that TensorFlow correctly calculates the total variation
    # for each image individually and returns the correct array.
    self._test(multi, tot_var * np.array([1.0, 1.1, 1.2]))


class FormatTest(test_util.TensorFlowTestCase):

  @test_util.run_deprecated_v1
  def testFormats(self):
    prefix = "tensorflow/core/lib"
    paths = ("png/testdata/lena_gray.png", "jpeg/testdata/jpeg_merge_test1.jpg",
             "gif/testdata/lena.gif")
    decoders = {
        "jpeg": functools.partial(image_ops.decode_jpeg, channels=3),
        "png": functools.partial(image_ops.decode_png, channels=3),
        "gif": lambda s: array_ops.squeeze(image_ops.decode_gif(s), axis=0),
    }
    with self.cached_session():
      for path in paths:
        contents = io_ops.read_file(os.path.join(prefix, path)).eval()
        images = {}
        for name, decode in decoders.items():
          image = decode(contents).eval()
          self.assertEqual(image.ndim, 3)
          for prev_name, prev in images.items():
            print("path %s, names %s %s, shapes %s %s" %
                  (path, name, prev_name, image.shape, prev.shape))
            self.assertAllEqual(image, prev)
          images[name] = image

  def testError(self):
    path = "tensorflow/core/lib/gif/testdata/scan.gif"
    with self.cached_session():
      for decode in image_ops.decode_jpeg, image_ops.decode_png:
        with self.assertRaisesOpError(r"Got 12 frames"):
          decode(io_ops.read_file(path)).eval()


class NonMaxSuppressionTest(test_util.TensorFlowTestCase):

  @test_util.run_deprecated_v1
  def NonMaxSuppressionTest(self):
    boxes_np = [[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, -0.1, 1, 0.9],
                [0, 10, 1, 11], [0, 10.1, 1, 11.1], [0, 100, 1, 101]]
    scores_np = [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]
    max_output_size_np = 3
    iou_threshold_np = 0.5
    with self.cached_session():
      boxes = constant_op.constant(boxes_np)
      scores = constant_op.constant(scores_np)
      max_output_size = constant_op.constant(max_output_size_np)
      iou_threshold = constant_op.constant(iou_threshold_np)
      selected_indices = image_ops.non_max_suppression(
          boxes, scores, max_output_size, iou_threshold)
      self.assertAllClose(selected_indices.eval(), [3, 0, 5])

  @test_util.run_deprecated_v1
  def testInvalidShape(self):
    # The boxes should be 2D of shape [num_boxes, 4].
    with self.assertRaisesRegexp(ValueError,
                                 "Shape must be rank 2 but is rank 1"):
      boxes = constant_op.constant([0.0, 0.0, 1.0, 1.0])
      scores = constant_op.constant([0.9])
      image_ops.non_max_suppression(boxes, scores, 3, 0.5)

    with self.assertRaisesRegexp(ValueError, "Dimension must be 4 but is 3"):
      boxes = constant_op.constant([[0.0, 0.0, 1.0]])
      scores = constant_op.constant([0.9])
      image_ops.non_max_suppression(boxes, scores, 3, 0.5)

    # The boxes is of shape [num_boxes, 4], and the scores is
    # of shape [num_boxes]. So an error will be thrown.
    with self.assertRaisesRegexp(ValueError,
                                 "Dimensions must be equal, but are 1 and 2"):
      boxes = constant_op.constant([[0.0, 0.0, 1.0, 1.0]])
      scores = constant_op.constant([0.9, 0.75])
      image_ops.non_max_suppression(boxes, scores, 3, 0.5)

    # The scores should be 1D of shape [num_boxes].
    with self.assertRaisesRegexp(ValueError,
                                 "Shape must be rank 1 but is rank 2"):
      boxes = constant_op.constant([[0.0, 0.0, 1.0, 1.0]])
      scores = constant_op.constant([[0.9]])
      image_ops.non_max_suppression(boxes, scores, 3, 0.5)

    # The max_output_size should be a scalar (0-D).
    with self.assertRaisesRegexp(ValueError,
                                 "Shape must be rank 0 but is rank 1"):
      boxes = constant_op.constant([[0.0, 0.0, 1.0, 1.0]])
      scores = constant_op.constant([0.9])
      image_ops.non_max_suppression(boxes, scores, [3], 0.5)

    # The iou_threshold should be a scalar (0-D).
    with self.assertRaisesRegexp(ValueError,
                                 "Shape must be rank 0 but is rank 2"):
      boxes = constant_op.constant([[0.0, 0.0, 1.0, 1.0]])
      scores = constant_op.constant([0.9])
      image_ops.non_max_suppression(boxes, scores, 3, [[0.5]])

  @test_util.run_deprecated_v1
  @test_util.xla_allow_fallback(
      "non_max_suppression with dynamic output shape unsupported.")
  def testDataTypes(self):
    # Test case for GitHub issue 20199.
    boxes_np = [[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, -0.1, 1, 0.9],
                [0, 10, 1, 11], [0, 10.1, 1, 11.1], [0, 100, 1, 101]]
    scores_np = [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]
    max_output_size_np = 3
    iou_threshold_np = 0.5
    score_threshold_np = float("-inf")
    # Note: There are multiple versions of non_max_suppression v2, v3, v4.
    # gen_image_ops.non_max_suppression_v2:
    for dtype in [np.float16, np.float32]:
      with self.cached_session():
        boxes = constant_op.constant(boxes_np, dtype=dtype)
        scores = constant_op.constant(scores_np, dtype=dtype)
        max_output_size = constant_op.constant(max_output_size_np)
        iou_threshold = constant_op.constant(iou_threshold_np, dtype=dtype)
        selected_indices = gen_image_ops.non_max_suppression_v2(
            boxes, scores, max_output_size, iou_threshold).eval()
        self.assertAllClose(selected_indices, [3, 0, 5])
    # gen_image_ops.non_max_suppression_v3
    for dtype in [np.float16, np.float32]:
      with self.cached_session():
        boxes = constant_op.constant(boxes_np, dtype=dtype)
        scores = constant_op.constant(scores_np, dtype=dtype)
        max_output_size = constant_op.constant(max_output_size_np)
        iou_threshold = constant_op.constant(iou_threshold_np, dtype=dtype)
        score_threshold = constant_op.constant(score_threshold_np, dtype=dtype)
        selected_indices = gen_image_ops.non_max_suppression_v3(
            boxes, scores, max_output_size, iou_threshold, score_threshold)
        selected_indices = self.evaluate(selected_indices)
        self.assertAllClose(selected_indices, [3, 0, 5])
    # gen_image_ops.non_max_suppression_v4.
    for dtype in [np.float16, np.float32]:
      with self.cached_session():
        boxes = constant_op.constant(boxes_np, dtype=dtype)
        scores = constant_op.constant(scores_np, dtype=dtype)
        max_output_size = constant_op.constant(max_output_size_np)
        iou_threshold = constant_op.constant(iou_threshold_np, dtype=dtype)
        score_threshold = constant_op.constant(score_threshold_np, dtype=dtype)
        selected_indices, _ = gen_image_ops.non_max_suppression_v4(
            boxes, scores, max_output_size, iou_threshold, score_threshold)
        selected_indices = self.evaluate(selected_indices)
        self.assertAllClose(selected_indices, [3, 0, 5])
    # gen_image_ops.non_max_suppression_v5.
    soft_nms_sigma_np = float(0.0)
    for dtype in [np.float16, np.float32]:
      with self.cached_session():
        boxes = constant_op.constant(boxes_np, dtype=dtype)
        scores = constant_op.constant(scores_np, dtype=dtype)
        max_output_size = constant_op.constant(max_output_size_np)
        iou_threshold = constant_op.constant(iou_threshold_np, dtype=dtype)
        score_threshold = constant_op.constant(score_threshold_np, dtype=dtype)
        soft_nms_sigma = constant_op.constant(soft_nms_sigma_np, dtype=dtype)
        selected_indices, _, _ = gen_image_ops.non_max_suppression_v5(
            boxes, scores, max_output_size, iou_threshold, score_threshold,
            soft_nms_sigma)
        selected_indices = self.evaluate(selected_indices)
        self.assertAllClose(selected_indices, [3, 0, 5])


class NonMaxSuppressionWithScoresTest(test_util.TensorFlowTestCase):

  @test_util.run_deprecated_v1
  @test_util.xla_allow_fallback(
      "non_max_suppression with dynamic output shape unsupported.")
  def testSelectFromThreeClustersWithSoftNMS(self):
    boxes_np = [[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, -0.1, 1, 0.9],
                [0, 10, 1, 11], [0, 10.1, 1, 11.1], [0, 100, 1, 101]]
    scores_np = [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]
    max_output_size_np = 6
    iou_threshold_np = 1.0
    score_threshold_np = 0.0
    soft_nms_sigma_np = 0.5
    boxes = constant_op.constant(boxes_np)
    scores = constant_op.constant(scores_np)
    max_output_size = constant_op.constant(max_output_size_np)
    iou_threshold = constant_op.constant(iou_threshold_np)
    score_threshold = constant_op.constant(score_threshold_np)
    soft_nms_sigma = constant_op.constant(soft_nms_sigma_np)
    selected_indices, selected_scores = \
        image_ops.non_max_suppression_with_scores(
            boxes,
            scores,
            max_output_size,
            iou_threshold,
            score_threshold,
            soft_nms_sigma)
    selected_indices, selected_scores = self.evaluate(
        [selected_indices, selected_scores])
    self.assertAllClose(selected_indices, [3, 0, 1, 5, 4, 2])
    self.assertAllClose(selected_scores,
                        [0.95, 0.9, 0.384, 0.3, 0.256, 0.197],
                        rtol=1e-2, atol=1e-2)


class NonMaxSuppressionPaddedTest(test_util.TensorFlowTestCase):

  @test_util.run_deprecated_v1
  @test_util.disable_xla(
      "b/141236442: "
      "non_max_suppression with dynamic output shape unsupported.")
  def testSelectFromThreeClusters(self):
    boxes_np = [[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, -0.1, 1, 0.9],
                [0, 10, 1, 11], [0, 10.1, 1, 11.1], [0, 100, 1, 101]]
    scores_np = [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]
    max_output_size_np = 5
    iou_threshold_np = 0.5
    boxes = constant_op.constant(boxes_np)
    scores = constant_op.constant(scores_np)
    max_output_size = constant_op.constant(max_output_size_np)
    iou_threshold = constant_op.constant(iou_threshold_np)
    selected_indices_padded, num_valid_padded = \
        image_ops.non_max_suppression_padded(
            boxes,
            scores,
            max_output_size,
            iou_threshold,
            pad_to_max_output_size=True)
    selected_indices, num_valid = image_ops.non_max_suppression_padded(
        boxes,
        scores,
        max_output_size,
        iou_threshold,
        pad_to_max_output_size=False)
    # The output shape of the padded operation must be fully defined.
    self.assertEqual(selected_indices_padded.shape.is_fully_defined(), True)
    self.assertEqual(selected_indices.shape.is_fully_defined(), False)
    with self.cached_session():
      self.assertAllClose(selected_indices_padded.eval(),
                          [3, 0, 5, 0, 0])
      self.assertEqual(num_valid_padded.eval(), 3)
      self.assertAllClose(selected_indices.eval(), [3, 0, 5])
      self.assertEqual(num_valid.eval(), 3)

  @test_util.run_deprecated_v1
  @test_util.xla_allow_fallback(
      "non_max_suppression with dynamic output shape unsupported.")
  def testSelectFromContinuousOverLap(self):
    boxes_np = [[0, 0, 1, 1], [0, 0.2, 1, 1.2], [0, 0.4, 1, 1.4],
                [0, 0.6, 1, 1.6], [0, 0.8, 1, 1.8], [0, 2, 1, 2]]
    scores_np = [0.9, 0.75, 0.6, 0.5, 0.4, 0.3]
    max_output_size_np = 3
    iou_threshold_np = 0.5
    score_threshold_np = 0.1
    boxes = constant_op.constant(boxes_np)
    scores = constant_op.constant(scores_np)
    max_output_size = constant_op.constant(max_output_size_np)
    iou_threshold = constant_op.constant(iou_threshold_np)
    score_threshold = constant_op.constant(score_threshold_np)
    selected_indices, num_valid = image_ops.non_max_suppression_padded(
        boxes,
        scores,
        max_output_size,
        iou_threshold,
        score_threshold)
    # The output shape of the padded operation must be fully defined.
    self.assertEqual(selected_indices.shape.is_fully_defined(), False)
    with self.cached_session():
      self.assertAllClose(selected_indices.eval(), [0, 2, 4])
      self.assertEqual(num_valid.eval(), 3)


class NonMaxSuppressionWithOverlapsTest(test_util.TensorFlowTestCase):

  @test_util.run_deprecated_v1
  def testSelectOneFromThree(self):
    overlaps_np = [
        [1.0, 0.7, 0.2],
        [0.7, 1.0, 0.0],
        [0.2, 0.0, 1.0],
    ]
    scores_np = [0.7, 0.9, 0.1]
    max_output_size_np = 3

    overlaps = constant_op.constant(overlaps_np)
    scores = constant_op.constant(scores_np)
    max_output_size = constant_op.constant(max_output_size_np)
    overlap_threshold = 0.6
    score_threshold = 0.4

    selected_indices = image_ops.non_max_suppression_with_overlaps(
        overlaps, scores, max_output_size, overlap_threshold, score_threshold)

    with self.cached_session():
      self.assertAllClose(selected_indices.eval(), [1])


class VerifyCompatibleImageShapesTest(test_util.TensorFlowTestCase):
  """Tests utility function used by ssim() and psnr()."""

  @test_util.run_deprecated_v1
  def testWrongDims(self):
    img = array_ops.placeholder(dtype=dtypes.float32)
    img_np = np.array((2, 2))

    with self.cached_session(use_gpu=True) as sess:
      _, _, checks = image_ops_impl._verify_compatible_image_shapes(img, img)
      with self.assertRaises(errors.InvalidArgumentError):
        sess.run(checks, {img: img_np})

  @test_util.run_deprecated_v1
  def testShapeMismatch(self):
    img1 = array_ops.placeholder(dtype=dtypes.float32)
    img2 = array_ops.placeholder(dtype=dtypes.float32)

    img1_np = np.array([1, 2, 2, 1])
    img2_np = np.array([1, 3, 3, 1])

    with self.cached_session(use_gpu=True) as sess:
      _, _, checks = image_ops_impl._verify_compatible_image_shapes(img1, img2)
      with self.assertRaises(errors.InvalidArgumentError):
        sess.run(checks, {img1: img1_np, img2: img2_np})


class PSNRTest(test_util.TensorFlowTestCase):
  """Tests for PSNR."""

  def _LoadTestImage(self, sess, filename):
    content = io_ops.read_file(os.path.join(
        "tensorflow/core/lib/psnr/testdata", filename))
    im = image_ops.decode_jpeg(content, dct_method="INTEGER_ACCURATE")
    im = image_ops.convert_image_dtype(im, dtypes.float32)
    im, = self.evaluate([im])
    return np.expand_dims(im, axis=0)

  def _LoadTestImages(self):
    with self.cached_session(use_gpu=True) as sess:
      q20 = self._LoadTestImage(sess, "cat_q20.jpg")
      q72 = self._LoadTestImage(sess, "cat_q72.jpg")
      q95 = self._LoadTestImage(sess, "cat_q95.jpg")
      return q20, q72, q95

  def _PSNR_NumPy(self, orig, target, max_value):
    """Numpy implementation of PSNR."""
    mse = ((orig - target) ** 2).mean(axis=(-3, -2, -1))
    return 20 * np.log10(max_value) - 10 * np.log10(mse)

  def _RandomImage(self, shape, max_val):
    """Returns an image or image batch with given shape."""
    return np.random.rand(*shape).astype(np.float32) * max_val

  @test_util.run_deprecated_v1
  def testPSNRSingleImage(self):
    image1 = self._RandomImage((8, 8, 1), 1)
    image2 = self._RandomImage((8, 8, 1), 1)
    psnr = self._PSNR_NumPy(image1, image2, 1)

    with self.cached_session(use_gpu=True):
      tf_image1 = constant_op.constant(image1, shape=image1.shape,
                                       dtype=dtypes.float32)
      tf_image2 = constant_op.constant(image2, shape=image2.shape,
                                       dtype=dtypes.float32)
      tf_psnr = image_ops.psnr(tf_image1, tf_image2, 1.0, "psnr").eval()
      self.assertAllClose(psnr, tf_psnr, atol=0.001)

  @test_util.run_deprecated_v1
  def testPSNRMultiImage(self):
    image1 = self._RandomImage((10, 8, 8, 1), 1)
    image2 = self._RandomImage((10, 8, 8, 1), 1)
    psnr = self._PSNR_NumPy(image1, image2, 1)

    with self.cached_session(use_gpu=True):
      tf_image1 = constant_op.constant(image1, shape=image1.shape,
                                       dtype=dtypes.float32)
      tf_image2 = constant_op.constant(image2, shape=image2.shape,
                                       dtype=dtypes.float32)
      tf_psnr = image_ops.psnr(tf_image1, tf_image2, 1, "psnr").eval()
      self.assertAllClose(psnr, tf_psnr, atol=0.001)

  @test_util.run_deprecated_v1
  def testGoldenPSNR(self):
    q20, q72, q95 = self._LoadTestImages()

    # Verify NumPy implementation first.
    # Golden values are generated using GNU Octave's psnr() function.
    psnr1 = self._PSNR_NumPy(q20, q72, 1)
    self.assertNear(30.321, psnr1, 0.001, msg="q20.dtype=" + str(q20.dtype))
    psnr2 = self._PSNR_NumPy(q20, q95, 1)
    self.assertNear(29.994, psnr2, 0.001)
    psnr3 = self._PSNR_NumPy(q72, q95, 1)
    self.assertNear(35.302, psnr3, 0.001)

    # Test TensorFlow implementation.
    with self.cached_session(use_gpu=True):
      tf_q20 = constant_op.constant(q20, shape=q20.shape, dtype=dtypes.float32)
      tf_q72 = constant_op.constant(q72, shape=q72.shape, dtype=dtypes.float32)
      tf_q95 = constant_op.constant(q95, shape=q95.shape, dtype=dtypes.float32)
      tf_psnr1 = image_ops.psnr(tf_q20, tf_q72, 1, "psnr1").eval()
      tf_psnr2 = image_ops.psnr(tf_q20, tf_q95, 1, "psnr2").eval()
      tf_psnr3 = image_ops.psnr(tf_q72, tf_q95, 1, "psnr3").eval()
      self.assertAllClose(psnr1, tf_psnr1, atol=0.001)
      self.assertAllClose(psnr2, tf_psnr2, atol=0.001)
      self.assertAllClose(psnr3, tf_psnr3, atol=0.001)

  @test_util.run_deprecated_v1
  def testInfinity(self):
    q20, _, _ = self._LoadTestImages()
    psnr = self._PSNR_NumPy(q20, q20, 1)
    with self.cached_session(use_gpu=True):
      tf_q20 = constant_op.constant(q20, shape=q20.shape, dtype=dtypes.float32)
      tf_psnr = image_ops.psnr(tf_q20, tf_q20, 1, "psnr").eval()
      self.assertAllClose(psnr, tf_psnr, atol=0.001)

  @test_util.run_deprecated_v1
  def testInt(self):
    img1 = self._RandomImage((10, 8, 8, 1), 255)
    img2 = self._RandomImage((10, 8, 8, 1), 255)
    img1 = constant_op.constant(img1, dtypes.uint8)
    img2 = constant_op.constant(img2, dtypes.uint8)
    psnr_uint8 = image_ops.psnr(img1, img2, 255)
    img1 = image_ops.convert_image_dtype(img1, dtypes.float32)
    img2 = image_ops.convert_image_dtype(img2, dtypes.float32)
    psnr_float32 = image_ops.psnr(img1, img2, 1.0)
    with self.cached_session(use_gpu=True):
      self.assertAllClose(
          psnr_uint8.eval(), self.evaluate(psnr_float32), atol=0.001)


class SSIMTest(test_util.TensorFlowTestCase):
  """Tests for SSIM."""

  _filenames = ["checkerboard1.png",
                "checkerboard2.png",
                "checkerboard3.png",]

  _ssim = np.asarray([[1.000000, 0.230880, 0.231153],
                      [0.230880, 1.000000, 0.996828],
                      [0.231153, 0.996828, 1.000000]])

  def _LoadTestImage(self, sess, filename):
    content = io_ops.read_file(os.path.join(
        "tensorflow/core/lib/ssim/testdata", filename))
    im = image_ops.decode_png(content)
    im = image_ops.convert_image_dtype(im, dtypes.float32)
    im, = self.evaluate([im])
    return np.expand_dims(im, axis=0)

  def _LoadTestImages(self):
    with self.cached_session(use_gpu=True) as sess:
      return [self._LoadTestImage(sess, f) for f in self._filenames]

  def _RandomImage(self, shape, max_val):
    """Returns an image or image batch with given shape."""
    return np.random.rand(*shape).astype(np.float32) * max_val

  @test_util.run_deprecated_v1
  def testAgainstMatlab(self):
    """Tests against values produced by Matlab."""
    img = self._LoadTestImages()
    expected = self._ssim[np.triu_indices(3)]

    ph = [array_ops.placeholder(dtype=dtypes.float32) for _ in range(2)]
    ssim = image_ops.ssim(
        *ph, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
    with self.cached_session(use_gpu=True):
      scores = [ssim.eval(dict(zip(ph, t)))
                for t in itertools.combinations_with_replacement(img, 2)]
    self.assertAllClose(expected, np.squeeze(scores), atol=1e-4)

  def testBatch(self):
    img = self._LoadTestImages()
    expected = self._ssim[np.triu_indices(3, k=1)]

    img1, img2 = zip(*itertools.combinations(img, 2))
    img1 = np.concatenate(img1)
    img2 = np.concatenate(img2)

    ssim = image_ops.ssim(
        constant_op.constant(img1),
        constant_op.constant(img2),
        1.0,
        filter_size=11,
        filter_sigma=1.5,
        k1=0.01,
        k2=0.03)
    with self.cached_session(use_gpu=True):
      self.assertAllClose(expected, self.evaluate(ssim), atol=1e-4)

  def testBroadcast(self):
    img = self._LoadTestImages()[:2]
    expected = self._ssim[:2, :2]

    img = constant_op.constant(np.concatenate(img))
    img1 = array_ops.expand_dims(img, axis=0)  # batch dims: 1, 2.
    img2 = array_ops.expand_dims(img, axis=1)  # batch dims: 2, 1.

    ssim = image_ops.ssim(
        img1, img2, 1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
    with self.cached_session(use_gpu=True):
      self.assertAllClose(expected, self.evaluate(ssim), atol=1e-4)

  @test_util.run_deprecated_v1
  def testNegative(self):
    """Tests against negative SSIM index."""
    step = np.expand_dims(np.arange(0, 256, 16, dtype=np.uint8), axis=0)
    img1 = np.tile(step, (16, 1))
    img2 = np.fliplr(img1)

    img1 = img1.reshape((1, 16, 16, 1))
    img2 = img2.reshape((1, 16, 16, 1))

    ssim = image_ops.ssim(
        constant_op.constant(img1),
        constant_op.constant(img2),
        255,
        filter_size=11,
        filter_sigma=1.5,
        k1=0.01,
        k2=0.03)
    with self.cached_session(use_gpu=True):
      self.assertLess(ssim.eval(), 0)

  @test_util.run_deprecated_v1
  def testInt(self):
    img1 = self._RandomImage((1, 16, 16, 3), 255)
    img2 = self._RandomImage((1, 16, 16, 3), 255)
    img1 = constant_op.constant(img1, dtypes.uint8)
    img2 = constant_op.constant(img2, dtypes.uint8)
    ssim_uint8 = image_ops.ssim(
        img1, img2, 255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
    img1 = image_ops.convert_image_dtype(img1, dtypes.float32)
    img2 = image_ops.convert_image_dtype(img2, dtypes.float32)
    ssim_float32 = image_ops.ssim(
        img1, img2, 1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
    with self.cached_session(use_gpu=True):
      self.assertAllClose(
          ssim_uint8.eval(), self.evaluate(ssim_float32), atol=0.001)


class MultiscaleSSIMTest(test_util.TensorFlowTestCase):
  """Tests for MS-SSIM."""

  _filenames = ["checkerboard1.png",
                "checkerboard2.png",
                "checkerboard3.png",]

  _msssim = np.asarray([[1.000000, 0.091016, 0.091025],
                        [0.091016, 1.000000, 0.999567],
                        [0.091025, 0.999567, 1.000000]])

  def _LoadTestImage(self, sess, filename):
    content = io_ops.read_file(os.path.join(
        "tensorflow/core/lib/ssim/testdata", filename))
    im = image_ops.decode_png(content)
    im = image_ops.convert_image_dtype(im, dtypes.float32)
    im, = self.evaluate([im])
    return np.expand_dims(im, axis=0)

  def _LoadTestImages(self):
    with self.cached_session(use_gpu=True) as sess:
      return [self._LoadTestImage(sess, f) for f in self._filenames]

  def _RandomImage(self, shape, max_val):
    """Returns an image or image batch with given shape."""
    return np.random.rand(*shape).astype(np.float32) * max_val

  @test_util.run_deprecated_v1
  def testAgainstMatlab(self):
    """Tests against MS-SSIM computed with Matlab implementation.

    For color images, MS-SSIM scores are averaged over color channels.
    """
    img = self._LoadTestImages()
    expected = self._msssim[np.triu_indices(3)]

    ph = [array_ops.placeholder(dtype=dtypes.float32) for _ in range(2)]
    msssim = image_ops.ssim_multiscale(
        *ph, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
    with self.cached_session(use_gpu=True):
      scores = [msssim.eval(dict(zip(ph, t)))
                for t in itertools.combinations_with_replacement(img, 2)]

    self.assertAllClose(expected, np.squeeze(scores), atol=1e-4)

  @test_util.run_deprecated_v1
  def testUnweightedIsDifferentiable(self):
    img = self._LoadTestImages()
    ph = [array_ops.placeholder(dtype=dtypes.float32) for _ in range(2)]
    scalar = constant_op.constant(1.0, dtype=dtypes.float32)
    scaled_ph = [x * scalar for x in ph]
    msssim = image_ops.ssim_multiscale(
        *scaled_ph,
        max_val=1.0,
        power_factors=(1, 1, 1, 1, 1),
        filter_size=11,
        filter_sigma=1.5,
        k1=0.01,
        k2=0.03)
    grads = gradients.gradients(msssim, scalar)
    with self.cached_session(use_gpu=True) as sess:
      np_grads = sess.run(grads, feed_dict={ph[0]: img[0], ph[1]: img[1]})
    self.assertTrue(np.isfinite(np_grads).all())

  def testBatch(self):
    """Tests MS-SSIM computed in batch."""
    img = self._LoadTestImages()
    expected = self._msssim[np.triu_indices(3, k=1)]

    img1, img2 = zip(*itertools.combinations(img, 2))
    img1 = np.concatenate(img1)
    img2 = np.concatenate(img2)

    msssim = image_ops.ssim_multiscale(
        constant_op.constant(img1),
        constant_op.constant(img2),
        1.0,
        filter_size=11,
        filter_sigma=1.5,
        k1=0.01,
        k2=0.03)
    with self.cached_session(use_gpu=True):
      self.assertAllClose(expected, self.evaluate(msssim), 1e-4)

  def testBroadcast(self):
    """Tests MS-SSIM broadcasting."""
    img = self._LoadTestImages()[:2]
    expected = self._msssim[:2, :2]

    img = constant_op.constant(np.concatenate(img))
    img1 = array_ops.expand_dims(img, axis=0)  # batch dims: 1, 2.
    img2 = array_ops.expand_dims(img, axis=1)  # batch dims: 2, 1.

    score_tensor = image_ops.ssim_multiscale(
        img1, img2, 1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
    with self.cached_session(use_gpu=True):
      self.assertAllClose(expected, self.evaluate(score_tensor), 1e-4)

  def testRange(self):
    """Tests against low MS-SSIM score.

    MS-SSIM is a geometric mean of SSIM and CS scores of various scales.
    If any of the value is negative so that the geometric mean is not
    well-defined, then treat the MS-SSIM score as zero.
    """
    with self.cached_session(use_gpu=True) as sess:
      img1 = self._LoadTestImage(sess, "checkerboard1.png")
      img2 = self._LoadTestImage(sess, "checkerboard3.png")
      images = [img1, img2, np.zeros_like(img1),
                np.full_like(img1, fill_value=255)]

      images = [ops.convert_to_tensor(x, dtype=dtypes.float32) for x in images]
      msssim_ops = [
          image_ops.ssim_multiscale(
              x, y, 1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
          for x, y in itertools.combinations(images, 2)
      ]
      msssim = self.evaluate(msssim_ops)
      msssim = np.squeeze(msssim)

    self.assertTrue(np.all(msssim >= 0.0))
    self.assertTrue(np.all(msssim <= 1.0))

  @test_util.run_deprecated_v1
  def testInt(self):
    img1 = self._RandomImage((1, 180, 240, 3), 255)
    img2 = self._RandomImage((1, 180, 240, 3), 255)
    img1 = constant_op.constant(img1, dtypes.uint8)
    img2 = constant_op.constant(img2, dtypes.uint8)
    ssim_uint8 = image_ops.ssim_multiscale(
        img1, img2, 255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
    img1 = image_ops.convert_image_dtype(img1, dtypes.float32)
    img2 = image_ops.convert_image_dtype(img2, dtypes.float32)
    ssim_float32 = image_ops.ssim_multiscale(
        img1, img2, 1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
    with self.cached_session(use_gpu=True):
      self.assertAllClose(
          ssim_uint8.eval(), self.evaluate(ssim_float32), atol=0.001)

  def testNumpyInput(self):
    """Test case for GitHub issue 28241."""
    image = np.random.random([512, 512, 1])
    score_tensor = image_ops.ssim_multiscale(image, image, max_val=1.0)
    with self.cached_session(use_gpu=True):
      _ = self.evaluate(score_tensor)


class ImageGradientsTest(test_util.TensorFlowTestCase):

  def testImageGradients(self):
    shape = [1, 2, 4, 1]
    img = constant_op.constant([[1, 3, 4, 2], [8, 7, 5, 6]])
    img = array_ops.reshape(img, shape)

    expected_dy = np.reshape([[7, 4, 1, 4], [0, 0, 0, 0]], shape)
    expected_dx = np.reshape([[2, 1, -2, 0], [-1, -2, 1, 0]], shape)

    dy, dx = image_ops.image_gradients(img)
    with self.cached_session():
      actual_dy = self.evaluate(dy)
      actual_dx = self.evaluate(dx)
      self.assertAllClose(expected_dy, actual_dy)
      self.assertAllClose(expected_dx, actual_dx)

  def testImageGradientsMultiChannelBatch(self):
    batch = [[[[1, 2], [2, 5], [3, 3]],
              [[8, 4], [5, 1], [9, 8]]],
             [[[5, 3], [7, 9], [1, 6]],
              [[1, 2], [6, 3], [6, 3]]]]

    expected_dy = [[[[7, 2], [3, -4], [6, 5]],
                    [[0, 0], [0, 0], [0, 0]]],
                   [[[-4, -1], [-1, -6], [5, -3]],
                    [[0, 0], [0, 0], [0, 0]]]]

    expected_dx = [[[[1, 3], [1, -2], [0, 0]],
                    [[-3, -3], [4, 7], [0, 0]]],
                   [[[2, 6], [-6, -3], [0, 0]],
                    [[5, 1], [0, 0], [0, 0]]]]

    batch = constant_op.constant(batch)
    assert batch.get_shape().as_list() == [2, 2, 3, 2]
    dy, dx = image_ops.image_gradients(batch)
    with self.cached_session(use_gpu=True):
      actual_dy = self.evaluate(dy)
      actual_dx = self.evaluate(dx)
      self.assertAllClose(expected_dy, actual_dy)
      self.assertAllClose(expected_dx, actual_dx)

  def testImageGradientsBadShape(self):
    # [2 x 4] image but missing batch and depth dimensions.
    img = constant_op.constant([[1, 3, 4, 2], [8, 7, 5, 6]])
    with self.assertRaises(ValueError):
      image_ops.image_gradients(img)


class SobelEdgesTest(test_util.TensorFlowTestCase):

  def disabled_testSobelEdges1x2x3x1(self):
    img = constant_op.constant([[1, 3, 6], [4, 1, 5]],
                               dtype=dtypes.float32, shape=[1, 2, 3, 1])
    expected = np.reshape([[[0, 0], [0, 12], [0, 0]],
                           [[0, 0], [0, 12], [0, 0]]], [1, 2, 3, 1, 2])
    sobel = image_ops.sobel_edges(img)
    with self.cached_session(use_gpu=True):
      actual_sobel = self.evaluate(sobel)
      self.assertAllClose(expected, actual_sobel)

  def testSobelEdges5x3x4x2(self):
    batch_size = 5
    plane = np.reshape([[1, 3, 6, 2], [4, 1, 5, 7], [2, 5, 1, 4]],
                       [1, 3, 4, 1])
    two_channel = np.concatenate([plane, plane], axis=3)
    batch = np.concatenate([two_channel] * batch_size, axis=0)
    img = constant_op.constant(batch, dtype=dtypes.float32,
                               shape=[batch_size, 3, 4, 2])

    expected_plane = np.reshape([[[0, 0], [0, 12], [0, 10], [0, 0]],
                                 [[6, 0], [0, 6], [-6, 10], [-6, 0]],
                                 [[0, 0], [0, 0], [0, 10], [0, 0]]],
                                [1, 3, 4, 1, 2])
    expected_two_channel = np.concatenate(
        [expected_plane, expected_plane], axis=3)
    expected_batch = np.concatenate([expected_two_channel] * batch_size, axis=0)

    sobel = image_ops.sobel_edges(img)
    with self.cached_session(use_gpu=True):
      actual_sobel = self.evaluate(sobel)
      self.assertAllClose(expected_batch, actual_sobel)


@test_util.run_all_in_graph_and_eager_modes
class DecodeImageTest(test_util.TensorFlowTestCase):

  def testJpegUint16(self):
    with self.cached_session(use_gpu=True) as sess:
      base = "tensorflow/core/lib/jpeg/testdata"
      jpeg0 = io_ops.read_file(os.path.join(base, "jpeg_merge_test1.jpg"))
      image0 = image_ops.decode_image(jpeg0, dtype=dtypes.uint16)
      image1 = image_ops.convert_image_dtype(image_ops.decode_jpeg(jpeg0),
                                             dtypes.uint16)
      image0, image1 = self.evaluate([image0, image1])
      self.assertAllEqual(image0, image1)

  def testPngUint16(self):
    with self.cached_session(use_gpu=True) as sess:
      base = "tensorflow/core/lib/png/testdata"
      png0 = io_ops.read_file(os.path.join(base, "lena_rgba.png"))
      image0 = image_ops.decode_image(png0, dtype=dtypes.uint16)
      image1 = image_ops.convert_image_dtype(
          image_ops.decode_png(png0, dtype=dtypes.uint16), dtypes.uint16)
      image0, image1 = self.evaluate([image0, image1])
      self.assertAllEqual(image0, image1)

      # NumPy conversions should happen before
      x = np.random.randint(256, size=(4, 4, 3), dtype=np.uint16)
      x_str = image_ops_impl.encode_png(x)
      x_dec = image_ops_impl.decode_image(
          x_str, channels=3, dtype=dtypes.uint16)
      self.assertAllEqual(x, x_dec)

  def testGifUint16(self):
    with self.cached_session(use_gpu=True) as sess:
      base = "tensorflow/core/lib/gif/testdata"
      gif0 = io_ops.read_file(os.path.join(base, "scan.gif"))
      image0 = image_ops.decode_image(gif0, dtype=dtypes.uint16)
      image1 = image_ops.convert_image_dtype(image_ops.decode_gif(gif0),
                                             dtypes.uint16)
      image0, image1 = self.evaluate([image0, image1])
      self.assertAllEqual(image0, image1)

  def testBmpUint16(self):
    with self.cached_session(use_gpu=True) as sess:
      base = "tensorflow/core/lib/bmp/testdata"
      bmp0 = io_ops.read_file(os.path.join(base, "lena.bmp"))
      image0 = image_ops.decode_image(bmp0, dtype=dtypes.uint16)
      image1 = image_ops.convert_image_dtype(image_ops.decode_bmp(bmp0),
                                             dtypes.uint16)
      image0, image1 = self.evaluate([image0, image1])
      self.assertAllEqual(image0, image1)

  def testJpegFloat32(self):
    with self.cached_session(use_gpu=True) as sess:
      base = "tensorflow/core/lib/jpeg/testdata"
      jpeg0 = io_ops.read_file(os.path.join(base, "jpeg_merge_test1.jpg"))
      image0 = image_ops.decode_image(jpeg0, dtype=dtypes.float32)
      image1 = image_ops.convert_image_dtype(image_ops.decode_jpeg(jpeg0),
                                             dtypes.float32)
      image0, image1 = self.evaluate([image0, image1])
      self.assertAllEqual(image0, image1)

  def testPngFloat32(self):
    with self.cached_session(use_gpu=True) as sess:
      base = "tensorflow/core/lib/png/testdata"
      png0 = io_ops.read_file(os.path.join(base, "lena_rgba.png"))
      image0 = image_ops.decode_image(png0, dtype=dtypes.float32)
      image1 = image_ops.convert_image_dtype(
          image_ops.decode_png(png0, dtype=dtypes.uint16), dtypes.float32)
      image0, image1 = self.evaluate([image0, image1])
      self.assertAllEqual(image0, image1)

  def testGifFloat32(self):
    with self.cached_session(use_gpu=True) as sess:
      base = "tensorflow/core/lib/gif/testdata"
      gif0 = io_ops.read_file(os.path.join(base, "scan.gif"))
      image0 = image_ops.decode_image(gif0, dtype=dtypes.float32)
      image1 = image_ops.convert_image_dtype(image_ops.decode_gif(gif0),
                                             dtypes.float32)
      image0, image1 = self.evaluate([image0, image1])
      self.assertAllEqual(image0, image1)

  def testBmpFloat32(self):
    with self.cached_session(use_gpu=True) as sess:
      base = "tensorflow/core/lib/bmp/testdata"
      bmp0 = io_ops.read_file(os.path.join(base, "lena.bmp"))
      image0 = image_ops.decode_image(bmp0, dtype=dtypes.float32)
      image1 = image_ops.convert_image_dtype(image_ops.decode_bmp(bmp0),
                                             dtypes.float32)
      image0, image1 = self.evaluate([image0, image1])
      self.assertAllEqual(image0, image1)

  def testExpandAnimations(self):
    with self.cached_session(use_gpu=True) as sess:
      base = "tensorflow/core/lib/gif/testdata"
      gif0 = io_ops.read_file(os.path.join(base, "scan.gif"))
      image0 = image_ops.decode_image(
          gif0, dtype=dtypes.float32, expand_animations=False)
      # image_ops.decode_png() handles GIFs and returns 3D tensors
      animation = image_ops.decode_gif(gif0)
      first_frame = array_ops.gather(animation, 0)
      image1 = image_ops.convert_image_dtype(first_frame, dtypes.float32)
      image0, image1 = self.evaluate([image0, image1])
      self.assertEqual(len(image0.shape), 3)
      self.assertAllEqual(list(image0.shape), [40, 20, 3])
      self.assertAllEqual(image0, image1)


if __name__ == "__main__":
  googletest.main()
