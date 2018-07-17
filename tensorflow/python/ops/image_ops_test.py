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
      with self.test_session(use_gpu=True) as sess:
        batch0 = constant_op.constant(inp)
        batch1 = image_ops.rgb_to_hsv(batch0)
        batch2 = image_ops.hsv_to_rgb(batch1)
        split0 = array_ops.unstack(batch0)
        split1 = list(map(image_ops.rgb_to_hsv, split0))
        split2 = list(map(image_ops.hsv_to_rgb, split1))
        join1 = array_ops.stack(split1)
        join2 = array_ops.stack(split2)
        batch1, batch2, join1, join2 = sess.run([batch1, batch2, join1, join2])

      # Verify that processing batch elements together is the same as separate
      self.assertAllClose(batch1, join1)
      self.assertAllClose(batch2, join2)
      self.assertAllClose(batch2, inp)

  def testRGBToHSVRoundTrip(self):
    data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    for nptype in [np.float32, np.float64]:
      rgb_np = np.array(data, dtype=nptype).reshape([2, 2, 3]) / 255.
      with self.test_session(use_gpu=True):
        hsv = image_ops.rgb_to_hsv(rgb_np)
        rgb = image_ops.hsv_to_rgb(hsv)
        rgb_tf = rgb.eval()
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
      with self.test_session(use_gpu=True) as sess:
        batch0 = constant_op.constant(inp)
        batch1 = image_ops.rgb_to_yiq(batch0)
        batch2 = image_ops.yiq_to_rgb(batch1)
        split0 = array_ops.unstack(batch0)
        split1 = list(map(image_ops.rgb_to_yiq, split0))
        split2 = list(map(image_ops.yiq_to_rgb, split1))
        join1 = array_ops.stack(split1)
        join2 = array_ops.stack(split2)
        batch1, batch2, join1, join2 = sess.run([batch1, batch2, join1, join2])

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
      with self.test_session(use_gpu=True) as sess:
        batch0 = constant_op.constant(inp)
        batch1 = image_ops.rgb_to_yuv(batch0)
        batch2 = image_ops.yuv_to_rgb(batch1)
        split0 = array_ops.unstack(batch0)
        split1 = list(map(image_ops.rgb_to_yuv, split0))
        split2 = list(map(image_ops.yuv_to_rgb, split1))
        join1 = array_ops.stack(split1)
        join2 = array_ops.stack(split2)
        batch1, batch2, join1, join2 = sess.run([batch1, batch2, join1, join2])

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

    with self.test_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.rgb_to_grayscale(x_tf)
      y_tf = y.eval()
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

    with self.test_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.grayscale_to_rgb(x_tf)
      y_tf = y.eval()
      self.assertAllEqual(y_tf, y_np)

    # 3-D input with no batch dimension.
    x_np = np.array([[1, 2]], dtype=np.uint8).reshape([1, 2, 1])
    y_np = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.uint8).reshape([1, 2, 3])

    with self.test_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.grayscale_to_rgb(x_tf)
      y_tf = y.eval()
      self.assertAllEqual(y_tf, y_np)

  def testShapeInference(self):
    # Shape inference works and produces expected output where possible
    rgb_shape = [7, None, 19, 3]
    gray_shape = rgb_shape[:-1] + [1]
    with self.test_session(use_gpu=True):
      rgb_tf = array_ops.placeholder(dtypes.uint8, shape=rgb_shape)
      gray = image_ops.rgb_to_grayscale(rgb_tf)
      self.assertEqual(gray_shape, gray.get_shape().as_list())

    with self.test_session(use_gpu=True):
      gray_tf = array_ops.placeholder(dtypes.uint8, shape=gray_shape)
      rgb = image_ops.grayscale_to_rgb(gray_tf)
      self.assertEqual(rgb_shape, rgb.get_shape().as_list())

    # Shape inference does not break for unknown shapes
    with self.test_session(use_gpu=True):
      rgb_tf_unknown = array_ops.placeholder(dtypes.uint8)
      gray_unknown = image_ops.rgb_to_grayscale(rgb_tf_unknown)
      self.assertFalse(gray_unknown.get_shape())

    with self.test_session(use_gpu=True):
      gray_tf_unknown = array_ops.placeholder(dtypes.uint8)
      rgb_unknown = image_ops.grayscale_to_rgb(gray_tf_unknown)
      self.assertFalse(rgb_unknown.get_shape())


class AdjustGamma(test_util.TensorFlowTestCase):

  def test_adjust_gamma_one(self):
    """Same image should be returned for gamma equal to one"""
    with self.test_session():
      x_data = np.random.uniform(0, 255, (8, 8))
      x_np = np.array(x_data, dtype=np.float32)

      x = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.adjust_gamma(x, gamma=1)

      y_tf = y.eval()
      y_np = x_np

      self.assertAllClose(y_tf, y_np, 1e-6)

  def test_adjust_gamma_less_zero(self):
    """White image should be returned for gamma equal to zero"""
    with self.test_session():
      x_data = np.random.uniform(0, 255, (8, 8))
      x_np = np.array(x_data, dtype=np.float32)

      x = constant_op.constant(x_np, shape=x_np.shape)

      err_msg = "Gamma should be a non-negative real number."

      try:
        image_ops.adjust_gamma(x, gamma=-1)
      except Exception as e:
        if err_msg not in str(e):
          raise
      else:
        raise AssertionError("Exception not raised: %s" % err_msg)

  def test_adjust_gamma_less_zero_tensor(self):
    """White image should be returned for gamma equal to zero"""
    with self.test_session():
      x_data = np.random.uniform(0, 255, (8, 8))
      x_np = np.array(x_data, dtype=np.float32)

      x = constant_op.constant(x_np, shape=x_np.shape)
      y = constant_op.constant(-1.0, dtype=dtypes.float32)

      image = image_ops.adjust_gamma(x, gamma=y)

      err_msg = "Gamma should be a non-negative real number."
      try:
        image.eval()
      except Exception as e:
        if err_msg not in str(e):
          raise
      else:
        raise AssertionError("Exception not raised: %s" % err_msg)

  def test_adjust_gamma_zero(self):
    """White image should be returned for gamma equal to zero"""
    with self.test_session():
      x_data = np.random.uniform(0, 255, (8, 8))
      x_np = np.array(x_data, dtype=np.float32)

      x = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.adjust_gamma(x, gamma=0)

      y_tf = y.eval()

      dtype = x.dtype.as_numpy_dtype
      y_np = np.array([dtypes.dtype_range[dtype][1]] * x_np.size)
      y_np = y_np.reshape((8, 8))

      self.assertAllClose(y_tf, y_np, 1e-6)

  def test_adjust_gamma_less_one(self):
    """Verifying the output with expected results for gamma
    correction with gamma equal to half"""
    with self.test_session():
      x_np = np.arange(0, 255, 4, np.uint8).reshape(8, 8)
      y = image_ops.adjust_gamma(x_np, gamma=0.5)
      y_tf = np.trunc(y.eval())

      y_np = np.array(
          [[0, 31, 45, 55, 63, 71, 78, 84], [
              90, 95, 100, 105, 110, 115, 119, 123
          ], [127, 131, 135, 139, 142, 146, 149, 153], [
              156, 159, 162, 165, 168, 171, 174, 177
          ], [180, 183, 186, 188, 191, 194, 196, 199], [
              201, 204, 206, 209, 211, 214, 216, 218
          ], [221, 223, 225, 228, 230, 232, 234, 236],
           [238, 241, 243, 245, 247, 249, 251, 253]],
          dtype=np.float32)

      self.assertAllClose(y_tf, y_np, 1e-6)

  def test_adjust_gamma_greater_one(self):
    """Verifying the output with expected results for gamma
    correction with gamma equal to two"""
    with self.test_session():
      x_np = np.arange(0, 255, 4, np.uint8).reshape(8, 8)
      y = image_ops.adjust_gamma(x_np, gamma=2)
      y_tf = np.trunc(y.eval())

      y_np = np.array(
          [[0, 0, 0, 0, 1, 1, 2, 3], [4, 5, 6, 7, 9, 10, 12, 14], [
              16, 18, 20, 22, 25, 27, 30, 33
          ], [36, 39, 42, 45, 49, 52, 56, 60], [64, 68, 72, 76, 81, 85, 90, 95],
           [100, 105, 110, 116, 121, 127, 132, 138], [
               144, 150, 156, 163, 169, 176, 182, 189
           ], [196, 203, 211, 218, 225, 233, 241, 249]],
          dtype=np.float32)

      self.assertAllClose(y_tf, y_np, 1e-6)


class AdjustHueTest(test_util.TensorFlowTestCase):

  def testAdjustNegativeHue(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    delta = -0.25
    y_data = [0, 13, 1, 54, 226, 59, 8, 234, 150, 255, 39, 1]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.test_session(use_gpu=True):
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

    with self.test_session(use_gpu=True):
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.adjust_hue(x, delta)
      y_tf = y.eval()
      self.assertAllEqual(y_tf, y_np)

  def testBatchAdjustHue(self):
    x_shape = [2, 1, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    delta = 0.25
    y_data = [13, 0, 11, 226, 54, 221, 234, 8, 92, 1, 217, 255]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.test_session(use_gpu=True):
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.adjust_hue(x, delta)
      y_tf = y.eval()
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
    with self.test_session(use_gpu=True):
      x = constant_op.constant(x_np)
      y = image_ops.adjust_hue(x, delta_h)
      y_tf = y.eval()
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
        sess.run(variables.global_variables_initializer())
        for i in xrange(warmup_rounds + benchmark_rounds):
          if i == warmup_rounds:
            start = time.time()
          sess.run(run_op)
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
        sess.run(variables.global_variables_initializer())
        for i in xrange(warmup_rounds + benchmark_rounds):
          if i == warmup_rounds:
            start = time.time()
          sess.run(run_op)
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
        sess.run(variables.global_variables_initializer())
        for i in xrange(warmup_rounds + benchmark_rounds):
          if i == warmup_rounds:
            start = time.time()
          sess.run(run_op)
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
    with session.Session("", graph=ops.Graph(), config=config) as sess:
      with ops.device(device):
        inputs = variables.Variable(
            random_ops.random_uniform(image_shape, dtype=dtypes.float32) * 255,
            trainable=False,
            dtype=dtypes.float32)
        delta = constant_op.constant(0.1, dtype=dtypes.float32)
        outputs = image_ops.adjust_hue(inputs, delta)
        run_op = control_flow_ops.group(outputs)
        sess.run(variables.global_variables_initializer())
        for i in xrange(warmup_rounds + benchmark_rounds):
          if i == warmup_rounds:
            start = time.time()
          sess.run(run_op)
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
    with session.Session("", graph=ops.Graph(), config=config) as sess:
      with ops.device(device):
        inputs = variables.Variable(
            random_ops.random_uniform(image_shape, dtype=dtypes.float32) * 255,
            trainable=False,
            dtype=dtypes.float32)
        delta = constant_op.constant(0.1, dtype=dtypes.float32)
        outputs = image_ops.adjust_saturation(inputs, delta)
        run_op = control_flow_ops.group(outputs)
        sess.run(variables.global_variables_initializer())
        for _ in xrange(warmup_rounds):
          sess.run(run_op)
        start = time.time()
        for _ in xrange(benchmark_rounds):
          sess.run(run_op)
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

    with session.Session() as sess:
      sess.run(variables.global_variables_initializer())
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

    with session.Session() as sess:
      sess.run(variables.global_variables_initializer())
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

    with session.Session() as sess:
      sess.run(variables.global_variables_initializer())
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

    with self.test_session(use_gpu=True):
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

    with self.test_session(use_gpu=True):
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.adjust_saturation(x, saturation_factor)
      y_tf = y.eval()
      self.assertAllEqual(y_tf, y_np)

  def testBatchSaturation(self):
    x_shape = [2, 1, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    saturation_factor = 0.5
    y_data = [6, 9, 13, 140, 180, 226, 135, 121, 234, 172, 255, 128]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.test_session(use_gpu=True):
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.adjust_saturation(x, saturation_factor)
      y_tf = y.eval()
      self.assertAllEqual(y_tf, y_np)

  def _adjust_saturation(self, image, saturation_factor):
    image = ops.convert_to_tensor(image, name="image")
    orig_dtype = image.dtype
    flt_image = image_ops.convert_image_dtype(image, dtypes.float32)
    saturation_adjusted_image = gen_image_ops.adjust_saturation(
        flt_image, saturation_factor)
    return image_ops.convert_image_dtype(saturation_adjusted_image, orig_dtype)

  def testHalfSaturationFused(self):
    x_shape = [2, 2, 3]
    x_rgb_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_rgb_data, dtype=np.uint8).reshape(x_shape)

    saturation_factor = 0.5
    y_rgb_data = [6, 9, 13, 140, 180, 226, 135, 121, 234, 172, 255, 128]
    y_np = np.array(y_rgb_data, dtype=np.uint8).reshape(x_shape)

    with self.test_session(use_gpu=True):
      x = constant_op.constant(x_np, shape=x_shape)
      y = self._adjust_saturation(x, saturation_factor)
      y_tf = y.eval()
      self.assertAllEqual(y_tf, y_np)

  def testTwiceSaturationFused(self):
    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)

    saturation_factor = 2.0
    y_data = [0, 5, 13, 0, 106, 226, 30, 0, 234, 89, 255, 0]
    y_np = np.array(y_data, dtype=np.uint8).reshape(x_shape)

    with self.test_session(use_gpu=True):
      x = constant_op.constant(x_np, shape=x_shape)
      y = self._adjust_saturation(x, saturation_factor)
      y_tf = y.eval()
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
    with self.test_session(use_gpu=True):
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
          y_fused = self._adjust_saturation(x_np, scale).eval()
          self.assertAllClose(y_fused, y_baseline, rtol=2e-5, atol=1e-5)


class FlipTransposeRotateTest(test_util.TensorFlowTestCase):

  def testInvolutionLeftRight(self):
    x_np = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.uint8).reshape([2, 3, 1])
    with self.test_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.flip_left_right(image_ops.flip_left_right(x_tf))
      y_tf = y.eval()
      self.assertAllEqual(y_tf, x_np)

  def testInvolutionLeftRightWithBatch(self):
    x_np = np.array(
        [[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]],
        dtype=np.uint8).reshape([2, 2, 3, 1])
    with self.test_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.flip_left_right(image_ops.flip_left_right(x_tf))
      y_tf = y.eval()
      self.assertAllEqual(y_tf, x_np)

  def testLeftRight(self):
    x_np = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.uint8).reshape([2, 3, 1])
    y_np = np.array([[3, 2, 1], [3, 2, 1]], dtype=np.uint8).reshape([2, 3, 1])

    with self.test_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.flip_left_right(x_tf)
      self.assertTrue(y.op.name.startswith("flip_left_right"))
      y_tf = y.eval()
      self.assertAllEqual(y_tf, y_np)

  def testLeftRightWithBatch(self):
    x_np = np.array(
        [[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]],
        dtype=np.uint8).reshape([2, 2, 3, 1])
    y_np = np.array(
        [[[3, 2, 1], [3, 2, 1]], [[3, 2, 1], [3, 2, 1]]],
        dtype=np.uint8).reshape([2, 2, 3, 1])

    with self.test_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.flip_left_right(x_tf)
      y_tf = y.eval()
      self.assertAllEqual(y_tf, y_np)

  def testRandomFlipLeftRight(self):
    x_np = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.uint8).reshape([2, 3, 1])
    y_np = np.array([[3, 2, 1], [3, 2, 1]], dtype=np.uint8).reshape([2, 3, 1])
    seed = 42

    with self.test_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.random_flip_left_right(x_tf, seed=seed)
      self.assertTrue(y.op.name.startswith("random_flip_left_right"))

      count_flipped = 0
      count_unflipped = 0
      for _ in range(100):
        y_tf = y.eval()
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

    with self.test_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.random_flip_left_right(x_tf, seed=seed)
      self.assertTrue(y.op.name.startswith("random_flip_left_right"))

      count_flipped = 0
      count_unflipped = 0
      for _ in range(100):
        y_tf = y.eval()

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

    with self.test_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.flip_up_down(image_ops.flip_up_down(x_tf))
      y_tf = y.eval()
      self.assertAllEqual(y_tf, x_np)

  def testInvolutionUpDownWithBatch(self):
    x_np = np.array(
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
        dtype=np.uint8).reshape([2, 2, 3, 1])

    with self.test_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.flip_up_down(image_ops.flip_up_down(x_tf))
      y_tf = y.eval()
      self.assertAllEqual(y_tf, x_np)

  def testUpDown(self):
    x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8).reshape([2, 3, 1])
    y_np = np.array([[4, 5, 6], [1, 2, 3]], dtype=np.uint8).reshape([2, 3, 1])

    with self.test_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.flip_up_down(x_tf)
      self.assertTrue(y.op.name.startswith("flip_up_down"))
      y_tf = y.eval()
      self.assertAllEqual(y_tf, y_np)

  def testUpDownWithBatch(self):
    x_np = np.array(
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
        dtype=np.uint8).reshape([2, 2, 3, 1])
    y_np = np.array(
        [[[4, 5, 6], [1, 2, 3]], [[10, 11, 12], [7, 8, 9]]],
        dtype=np.uint8).reshape([2, 2, 3, 1])

    with self.test_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.flip_up_down(x_tf)
      y_tf = y.eval()
      self.assertAllEqual(y_tf, y_np)

  def testRandomFlipUpDown(self):
    x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8).reshape([2, 3, 1])
    y_np = np.array([[4, 5, 6], [1, 2, 3]], dtype=np.uint8).reshape([2, 3, 1])

    seed = 42

    with self.test_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.random_flip_up_down(x_tf, seed=seed)
      self.assertTrue(y.op.name.startswith("random_flip_up_down"))
      count_flipped = 0
      count_unflipped = 0
      for _ in range(100):
        y_tf = y.eval()
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

    with self.test_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.random_flip_up_down(x_tf, seed=seed)
      self.assertTrue(y.op.name.startswith("random_flip_up_down"))

      count_flipped = 0
      count_unflipped = 0
      for _ in range(100):
        y_tf = y.eval()

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

    with self.test_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.transpose_image(image_ops.transpose_image(x_tf))
      y_tf = y.eval()
      self.assertAllEqual(y_tf, x_np)

  def testInvolutionTransposeWithBatch(self):
    x_np = np.array(
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
        dtype=np.uint8).reshape([2, 2, 3, 1])

    with self.test_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.transpose_image(image_ops.transpose_image(x_tf))
      y_tf = y.eval()
      self.assertAllEqual(y_tf, x_np)

  def testTranspose(self):
    x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8).reshape([2, 3, 1])
    y_np = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.uint8).reshape([3, 2, 1])

    with self.test_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.transpose_image(x_tf)
      self.assertTrue(y.op.name.startswith("transpose_image"))
      y_tf = y.eval()
      self.assertAllEqual(y_tf, y_np)

  def testTransposeWithBatch(self):
    x_np = np.array(
        [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
        dtype=np.uint8).reshape([2, 2, 3, 1])

    y_np = np.array(
        [[[1, 4], [2, 5], [3, 6]], [[7, 10], [8, 11], [9, 12]]],
        dtype=np.uint8).reshape([2, 3, 2, 1])

    with self.test_session(use_gpu=True):
      x_tf = constant_op.constant(x_np, shape=x_np.shape)
      y = image_ops.transpose_image(x_tf)
      y_tf = y.eval()
      self.assertAllEqual(y_tf, y_np)

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
        image_ops.transpose_image, image_ops.rot90
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
        image_ops.transpose_image, image_ops.rot90
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
    with self.test_session(use_gpu=True):
      rotated = image
      for _ in xrange(4):
        rotated = image_ops.rot90(rotated)
      self.assertAllEqual(image, rotated.eval())

  def testRot90GroupOrderWithBatch(self):
    image = np.arange(48, dtype=np.uint8).reshape([2, 2, 4, 3])
    with self.test_session(use_gpu=True):
      rotated = image
      for _ in xrange(4):
        rotated = image_ops.rot90(rotated)
      self.assertAllEqual(image, rotated.eval())

  def testRot90NumpyEquivalence(self):
    image = np.arange(24, dtype=np.uint8).reshape([2, 4, 3])
    with self.test_session(use_gpu=True):
      k_placeholder = array_ops.placeholder(dtypes.int32, shape=[])
      y_tf = image_ops.rot90(image, k_placeholder)
      for k in xrange(4):
        y_np = np.rot90(image, k=k)
        self.assertAllEqual(y_np, y_tf.eval({k_placeholder: k}))

  def testRot90NumpyEquivalenceWithBatch(self):
    image = np.arange(48, dtype=np.uint8).reshape([2, 2, 4, 3])
    with self.test_session(use_gpu=True):
      k_placeholder = array_ops.placeholder(dtypes.int32, shape=[])
      y_tf = image_ops.rot90(image, k_placeholder)
      for k in xrange(4):
        y_np = np.rot90(image, k=k, axes=(1, 2))
        self.assertAllEqual(y_np, y_tf.eval({k_placeholder: k}))

class AdjustContrastTest(test_util.TensorFlowTestCase):

  def _testContrast(self, x_np, y_np, contrast_factor):
    with self.test_session(use_gpu=True):
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
    with self.test_session(use_gpu=True):
      x = constant_op.constant(x_np)
      y = image_ops.adjust_contrast(x, contrast_factor)
      y_tf = y.eval()
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


class AdjustBrightnessTest(test_util.TensorFlowTestCase):

  def _testBrightness(self, x_np, y_np, delta):
    with self.test_session(use_gpu=True):
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

    with self.test_session(use_gpu=True):
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.per_image_standardization(x)
      self.assertTrue(y.op.name.startswith("per_image_standardization"))
      y_tf = y.eval()
      self.assertAllClose(y_tf, y_np, atol=1e-4)

  def testUniformImage(self):
    im_np = np.ones([19, 19, 3]).astype(np.float32) * 249
    im = constant_op.constant(im_np)
    whiten = image_ops.per_image_standardization(im)
    with self.test_session(use_gpu=True):
      whiten_np = whiten.eval()
      self.assertFalse(np.any(np.isnan(whiten_np)))


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

    with self.test_session(use_gpu=True):
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

  def testNoOp(self):
    x_shape = [10, 10, 10]
    x = np.random.uniform(size=x_shape)
    self._assertReturns(x, x_shape, 0, 0, x, x_shape)

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

  def testNon3DInput(self):
    # Input image is not 3D
    x = [0] * 15
    offset_height, offset_width = [0, 0]
    target_height, target_width = [2, 2]

    for x_shape in ([3, 5], [1, 3, 5, 1, 1]):
      self._assertRaises(x, x_shape, offset_height, offset_width, target_height,
                         target_width,
                         "'image' must have either 3 or 4 dimensions.")

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
          "all dims of 'image.shape' must be > 0",
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

  def testNoOp(self):
    x_shapes = [[13, 9, 3], [5, 13, 9, 3]]
    for x_shape in x_shapes:
      x_np = np.ones(x_shape, dtype=np.float32)
      for use_gpu in [True, False]:
        with self.test_session(use_gpu=use_gpu):
          x = constant_op.constant(x_np, shape=x_shape)
          y = image_ops.central_crop(x, 1.0)
          y_tf = y.eval()
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
      with self.test_session(use_gpu=use_gpu):
        x = constant_op.constant(x_np, shape=x_shape)
        y = image_ops.central_crop(x, 0.5)
        y_tf = y.eval()
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
    with self.test_session(use_gpu=True):
      x = constant_op.constant(x_np, shape=x_shape)
      y = image_ops.central_crop(x, 0.5)
      y_tf = y.eval()
      self.assertAllEqual(y_tf, y_np)
      self.assertAllEqual(y_tf.shape, y_np.shape)

  def testCropping2(self):
    # Test case for 10315
    x_shapes = [[240, 320, 3], [5, 240, 320, 3]]
    expected_y_shapes = [[80, 106, 3], [5, 80, 106, 3]]

    for x_shape, y_shape in zip(x_shapes, expected_y_shapes):
      x_np = np.zeros(x_shape, dtype=np.int32)
      y_np = np.zeros(y_shape, dtype=np.int32)
      for use_gpu in [True, False]:
        with self.test_session(use_gpu=use_gpu):
          x = array_ops.placeholder(shape=x_shape, dtype=dtypes.int32)
          y = image_ops.central_crop(x, 0.33)
          y_tf = y.eval(feed_dict={x: x_np})
          self.assertAllEqual(y_tf, y_np)
          self.assertAllEqual(y_tf.shape, y_np.shape)

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
      with self.test_session(use_gpu=use_gpu):
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
        with self.test_session(use_gpu=use_gpu):
          x = constant_op.constant(x_np, shape=x_shape)
          with self.assertRaises(ValueError):
            _ = image_ops.central_crop(x, 0.5)

  def testNameScope(self):
    x_shape = [13, 9, 3]
    x_np = np.ones(x_shape, dtype=np.float32)
    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu):
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

    with self.test_session(use_gpu=True):
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
    with self.test_session(use_gpu=True):
      self.assertAllClose(y, y_tf.eval())

  def testNoOp(self):
    x_shape = [10, 10, 10]
    x = np.random.uniform(size=x_shape)
    offset_height, offset_width = [0, 0]
    self._assertReturns(x, x_shape, offset_height, offset_width, x, x_shape)

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

  def testNon3DInput(self):
    # Input image is not 3D
    x = [0] * 15
    offset_height, offset_width = [0, 0]
    target_height, target_width = [2, 2]

    for x_shape in ([3, 5], [1, 3, 5, 1, 1]):
      self._assertRaises(x, x_shape, offset_height, offset_width, target_height,
                         target_width,
                         "'image' must have either 3 or 4 dimensions.")

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
          "all dims of 'image.shape' must be > 0",
          use_tensor_inputs_options=[False])

      # The orignal error message does not contain back slashes. However, they
      # are added by either the assert op or the runtime. If this behavior
      # changes in the future, the match string will also needs to be changed.
      self._assertRaises(
          x,
          x_shape,
          offset_height,
          offset_width,
          target_height,
          target_width,
          "all dims of \\'image.shape\\' must be > 0",
          use_tensor_inputs_options=[True])

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
    with self.test_session(use_gpu=True):
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
        y_tf = y.eval()
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

  def testSampleDistortedBoundingBoxShape(self):
    with self.test_session(use_gpu=True):
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
      begin = begin.eval()
      end = end.eval()
      bbox_for_drawing = bbox_for_drawing.eval()

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
    with self.test_session(use_gpu=True):
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
      begin = begin.eval()
      end = end.eval()
      bbox_for_drawing = bbox_for_drawing.eval()


class ResizeImagesTest(test_util.TensorFlowTestCase):

  OPTIONS = [
      image_ops.ResizeMethod.BILINEAR, image_ops.ResizeMethod.NEAREST_NEIGHBOR,
      image_ops.ResizeMethod.BICUBIC, image_ops.ResizeMethod.AREA
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

  def shouldRunOnGPU(self, opt, nptype):
    if (opt == image_ops.ResizeMethod.NEAREST_NEIGHBOR and
        nptype in [np.float32, np.float64]):
      return True
    else:
      return False

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

      for opt in self.OPTIONS:
        with self.test_session(use_gpu=True) as sess:
          image = constant_op.constant(img_np, shape=img_shape)
          y = image_ops.resize_images(image, [target_height, target_width], opt)
          yshape = array_ops.shape(y)
          resized, newshape = sess.run([y, yshape])
          self.assertAllEqual(img_shape, newshape)
          self.assertAllClose(resized, img_np, atol=1e-5)

      # Resizing with a single image must leave the shape unchanged also.
      with self.test_session(use_gpu=True):
        img_single = img_np.reshape(single_shape)
        image = constant_op.constant(img_single, shape=single_shape)
        y = image_ops.resize_images(image, [target_height, target_width],
                                    self.OPTIONS[0])
        yshape = array_ops.shape(y)
        newshape = yshape.eval()
        self.assertAllEqual(single_shape, newshape)

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

    for opt in self.OPTIONS:
      with self.test_session(use_gpu=True) as sess:
        image = constant_op.constant(img_np, shape=img_shape)
        y = image_ops.resize_images(image, new_size, opt)
        yshape = array_ops.shape(y)
        resized, newshape = sess.run([y, yshape], {new_size: [6, 4]})
        self.assertAllEqual(img_shape, newshape)
        self.assertAllClose(resized, img_np, atol=1e-5)

    # Resizing with a single image must leave the shape unchanged also.
    with self.test_session(use_gpu=True):
      img_single = img_np.reshape(single_shape)
      image = constant_op.constant(img_single, shape=single_shape)
      y = image_ops.resize_images(image, new_size, self.OPTIONS[0])
      yshape = array_ops.shape(y)
      resized, newshape = sess.run([y, yshape], {new_size: [6, 4]})
      self.assertAllEqual(single_shape, newshape)
      self.assertAllClose(resized, img_single, atol=1e-5)

    # Incorrect shape.
    with self.assertRaises(ValueError):
      new_size = constant_op.constant(4)
      _ = image_ops.resize_images(image, new_size,
                                  image_ops.ResizeMethod.BILINEAR)
    with self.assertRaises(ValueError):
      new_size = constant_op.constant([4])
      _ = image_ops.resize_images(image, new_size,
                                  image_ops.ResizeMethod.BILINEAR)
    with self.assertRaises(ValueError):
      new_size = constant_op.constant([1, 2, 3])
      _ = image_ops.resize_images(image, new_size,
                                  image_ops.ResizeMethod.BILINEAR)

    # Incorrect dtypes.
    with self.assertRaises(ValueError):
      new_size = constant_op.constant([6.0, 4])
      _ = image_ops.resize_images(image, new_size,
                                  image_ops.ResizeMethod.BILINEAR)
    with self.assertRaises(ValueError):
      _ = image_ops.resize_images(image, [6, 4.0],
                                  image_ops.ResizeMethod.BILINEAR)
    with self.assertRaises(ValueError):
      _ = image_ops.resize_images(image, [None, 4],
                                  image_ops.ResizeMethod.BILINEAR)
    with self.assertRaises(ValueError):
      _ = image_ops.resize_images(image, [6, None],
                                  image_ops.ResizeMethod.BILINEAR)

  def testReturnDtype(self):
    target_shapes = [[6, 4], [3, 2], [
        array_ops.placeholder(dtypes.int32),
        array_ops.placeholder(dtypes.int32)
    ]]
    for nptype in self.TYPES:
      image = array_ops.placeholder(nptype, shape=[1, 6, 4, 1])
      for opt in self.OPTIONS:
        for target_shape in target_shapes:
          y = image_ops.resize_images(image, target_shape, opt)
          if (opt == image_ops.ResizeMethod.NEAREST_NEIGHBOR or
              target_shape == image.shape[1:3]):
            expected_dtype = image.dtype
          else:
            expected_dtype = dtypes.float32
          self.assertEqual(y.dtype, expected_dtype)

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

    for opt in self.OPTIONS:
      with self.test_session() as sess:
        image = constant_op.constant(img_np, shape=img_shape)
        y = image_ops.resize_images(image, [height, width], opt)
        yshape = array_ops.shape(y)
        resized, newshape = sess.run([y, yshape])
        self.assertAllEqual(img_shape, newshape)
        self.assertAllClose(resized, img_np, atol=1e-5)

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

        for opt in self.OPTIONS:
          if test.is_gpu_available() and self.shouldRunOnGPU(opt, nptype):
            with self.test_session(use_gpu=True):
              image = constant_op.constant(img_np, shape=img_shape)
              y = image_ops.resize_images(image, [target_height, target_width],
                                          opt)
              expected = np.array(expected_data).reshape(target_shape)
              resized = y.eval()
              self.assertAllClose(resized, expected, atol=1e-5)

  def testResizeUpAlignCornersFalse(self):
    img_shape = [1, 3, 2, 1]
    data = [64, 32, 32, 64, 50, 100]
    target_height = 6
    target_width = 4
    expected_data = {}
    expected_data[image_ops.ResizeMethod.BILINEAR] = [
        64.0, 48.0, 32.0, 32.0, 48.0, 48.0, 48.0, 48.0, 32.0, 48.0, 64.0, 64.0,
        41.0, 61.5, 82.0, 82.0, 50.0, 75.0, 100.0, 100.0, 50.0, 75.0, 100.0,
        100.0
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

    for nptype in self.TYPES:
      for opt in [
          image_ops.ResizeMethod.BILINEAR,
          image_ops.ResizeMethod.NEAREST_NEIGHBOR, image_ops.ResizeMethod.AREA
      ]:
        with self.test_session(use_gpu=True):
          img_np = np.array(data, dtype=nptype).reshape(img_shape)
          image = constant_op.constant(img_np, shape=img_shape)
          y = image_ops.resize_images(
              image, [target_height, target_width], opt, align_corners=False)
          resized = y.eval()
          expected = np.array(expected_data[opt]).reshape(
              [1, target_height, target_width, 1])
          self.assertAllClose(resized, expected, atol=1e-05)

  def testResizeUpAlignCornersTrue(self):
    img_shape = [1, 3, 2, 1]
    data = [6, 3, 3, 6, 6, 9]
    target_height = 5
    target_width = 4
    expected_data = {}
    expected_data[image_ops.ResizeMethod.BILINEAR] = [
        6.0, 5.0, 4.0, 3.0, 4.5, 4.5, 4.5, 4.5, 3.0, 4.0, 5.0, 6.0, 4.5, 5.5,
        6.5, 7.5, 6.0, 7.0, 8.0, 9.0
    ]
    expected_data[image_ops.ResizeMethod.NEAREST_NEIGHBOR] = [
        6.0, 6.0, 3.0, 3.0, 3.0, 3.0, 6.0, 6.0, 3.0, 3.0, 6.0, 6.0, 6.0, 6.0,
        9.0, 9.0, 6.0, 6.0, 9.0, 9.0
    ]
    # TODO(b/37749740): Improve alignment of ResizeMethod.AREA when
    # align_corners=True.
    expected_data[image_ops.ResizeMethod.AREA] = [
        6.0, 6.0, 6.0, 3.0, 6.0, 6.0, 6.0, 3.0, 3.0, 3.0, 3.0, 6.0, 3.0, 3.0,
        3.0, 6.0, 6.0, 6.0, 6.0, 9.0
    ]

    for nptype in self.TYPES:
      for opt in [
          image_ops.ResizeMethod.BILINEAR,
          image_ops.ResizeMethod.NEAREST_NEIGHBOR, image_ops.ResizeMethod.AREA
      ]:
        with self.test_session(use_gpu=True):
          img_np = np.array(data, dtype=nptype).reshape(img_shape)
          image = constant_op.constant(img_np, shape=img_shape)
          y = image_ops.resize_images(
              image, [target_height, target_width], opt, align_corners=True)
          resized = y.eval()
          expected = np.array(expected_data[opt]).reshape(
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

    with self.test_session(use_gpu=True):
      image = constant_op.constant(img_np, shape=img_shape)
      y = image_ops.resize_images(image, [target_height, target_width],
                                  image_ops.ResizeMethod.BICUBIC)
      resized = y.eval()
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

    with self.test_session(use_gpu=True):
      image = constant_op.constant(img_np, shape=img_shape)
      y = image_ops.resize_images(image, [target_height, target_width],
                                  image_ops.ResizeMethod.AREA)
      expected = np.array(expected_data).reshape(
          [1, target_height, target_width, 1])
      resized = y.eval()
      self.assertAllClose(resized, expected, atol=1)

  def testCompareNearestNeighbor(self):
    if test.is_gpu_available():
      input_shape = [1, 5, 6, 3]
      target_height = 8
      target_width = 12
      for nptype in [np.float32, np.float64]:
        for align_corners in [True, False]:
          img_np = np.arange(
              0, np.prod(input_shape), dtype=nptype).reshape(input_shape)
          with self.test_session(use_gpu=True):
            image = constant_op.constant(img_np, shape=input_shape)
            new_size = constant_op.constant([target_height, target_width])
            out_op = image_ops.resize_images(
                image,
                new_size,
                image_ops.ResizeMethod.NEAREST_NEIGHBOR,
                align_corners=align_corners)
            gpu_val = out_op.eval()
          with self.test_session(use_gpu=False):
            image = constant_op.constant(img_np, shape=input_shape)
            new_size = constant_op.constant([target_height, target_width])
            out_op = image_ops.resize_images(
                image,
                new_size,
                image_ops.ResizeMethod.NEAREST_NEIGHBOR,
                align_corners=align_corners)
            cpu_val = out_op.eval()
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
            with self.test_session(use_gpu=use_gpu):
              image = constant_op.constant(img_np, shape=input_shape)
              new_size = constant_op.constant([target_height, target_width])
              out_op = image_ops.resize_images(
                  image,
                  new_size,
                  image_ops.ResizeMethod.BILINEAR,
                  align_corners=align_corners)
              value[use_gpu] = out_op.eval()
          self.assertAllClose(value[True], value[False], rtol=1e-5, atol=1e-5)

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

  def testNameScope(self):
    img_shape = [1, 3, 2, 1]
    with self.test_session(use_gpu=True):
      single_image = array_ops.placeholder(dtypes.float32, shape=[50, 60, 3])
      y = image_ops.resize_images(single_image, [55, 66])
      self.assertTrue(y.op.name.startswith("resize_images"))

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

    with self.test_session(use_gpu=True):
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

  def testPreserveAspectRatioMultipleImages(self):
    x_shape = [10, 100, 100, 10]
    x = np.random.uniform(size=x_shape)

    self._assertResizeCheckShape(x, x_shape, [250, 250], [10, 250, 250, 10],
                                 preserve_aspect_ratio=False)

  def testPreserveAspectRatioNoOp(self):
    x_shape = [10, 10, 10]
    x = np.random.uniform(size=x_shape)

    self._assertResizeEqual(x, x_shape, x, x_shape)

  def testPreserveAspectRatioSmaller(self):
    x_shape = [100, 100, 10]
    x = np.random.uniform(size=x_shape)

    self._assertResizeCheckShape(x, x_shape, [75, 50], [50, 50, 10])

  def testPreserveAspectRatioSmallerMultipleImages(self):
    x_shape = [10, 100, 100, 10]
    x = np.random.uniform(size=x_shape)

    self._assertResizeCheckShape(x, x_shape, [75, 50], [10, 50, 50, 10])

  def testPreserveAspectRatioLarger(self):
    x_shape = [100, 100, 10]
    x = np.random.uniform(size=x_shape)

    self._assertResizeCheckShape(x, x_shape, [150, 200], [150, 150, 10])

  def testPreserveAspectRatioSameRatio(self):
    x_shape = [1920, 1080, 3]
    x = np.random.uniform(size=x_shape)

    self._assertResizeCheckShape(x, x_shape, [3840, 2160], [3840, 2160, 3])


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

    with self.test_session(use_gpu=True):
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

  def testNoOp(self):
    x_shape = [10, 10, 10]
    x = np.random.uniform(size=x_shape)

    self._assertReturns(x, x_shape, x, x_shape)

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

  def testNon3DInput(self):
    # Input image is not 3D
    x = [0] * 15
    target_height, target_width = [4, 4]

    for x_shape in ([3, 5],):
      self._assertRaises(x, x_shape, target_height, target_width,
                         "'image' must have either 3 or 4 dimensions.")

    for x_shape in ([1, 3, 5, 1, 1],):
      self._assertRaises(x, x_shape, target_height, target_width,
                         "'image' must have either 3 or 4 dimensions.")

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
          "all dims of 'image.shape' must be > 0",
          use_tensor_inputs_options=[False])

      # The orignal error message does not contain back slashes. However, they
      # are added by either the assert op or the runtime. If this behavior
      # changes in the future, the match string will also needs to be changed.
      self._assertRaises(
          x,
          x_shape,
          target_height,
          target_width,
          "all dims of \\'image.shape\\' must be > 0",
          use_tensor_inputs_options=[True])

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
    with self.test_session(use_gpu=True) as sess:
      jpeg0 = io_ops.read_file(path)
      image0 = image_ops.decode_jpeg(jpeg0)
      image1 = image_ops.decode_jpeg(image_ops.encode_jpeg(image0))
      jpeg0, image0, image1 = sess.run([jpeg0, image0, image1])
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
      with self.test_session(use_gpu=True) as sess:
        rgb = image_ops.decode_jpeg(
            io_ops.read_file(rgb_path), channels=channels)
        cmyk = image_ops.decode_jpeg(
            io_ops.read_file(cmyk_path), channels=channels)
        rgb, cmyk = sess.run([rgb, cmyk])
        self.assertEqual(rgb.shape, shape)
        self.assertEqual(cmyk.shape, shape)
        error = self.averageError(rgb, cmyk)
        self.assertLess(error, 4)

  def testCropAndDecodeJpeg(self):
    with self.test_session() as sess:
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
        image1_crop, image2 = sess.run([image1_crop, image2])
        self.assertAllEqual(image1_crop, image2)

  def testCropAndDecodeJpegWithInvalidCropWindow(self):
    with self.test_session() as sess:
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
          sess.run(result)

  def testSynthetic(self):
    with self.test_session(use_gpu=True) as sess:
      # Encode it, then decode it, then encode it
      image0 = constant_op.constant(_SimpleColorRamp())
      jpeg0 = image_ops.encode_jpeg(image0)
      image1 = image_ops.decode_jpeg(jpeg0, dct_method="INTEGER_ACCURATE")
      image2 = image_ops.decode_jpeg(
          image_ops.encode_jpeg(image1), dct_method="INTEGER_ACCURATE")
      jpeg0, image0, image1, image2 = sess.run([jpeg0, image0, image1, image2])

      # The decoded-encoded image should be similar to the input
      self.assertLess(self.averageError(image0, image1), 0.6)

      # We should be very close to a fixpoint
      self.assertLess(self.averageError(image1, image2), 0.02)

      # Smooth ramps compress well (input size is 153600)
      self.assertGreaterEqual(len(jpeg0), 5000)
      self.assertLessEqual(len(jpeg0), 6000)

  def testSyntheticFasterAlgorithm(self):
    with self.test_session(use_gpu=True) as sess:
      # Encode it, then decode it, then encode it
      image0 = constant_op.constant(_SimpleColorRamp())
      jpeg0 = image_ops.encode_jpeg(image0)
      image1 = image_ops.decode_jpeg(jpeg0, dct_method="INTEGER_FAST")
      image2 = image_ops.decode_jpeg(
          image_ops.encode_jpeg(image1), dct_method="INTEGER_FAST")
      jpeg0, image0, image1, image2 = sess.run([jpeg0, image0, image1, image2])

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
    with self.test_session(use_gpu=True) as sess:
      # Compare decoding with both dct_option=INTEGER_FAST and
      # default.  They should be the same.
      image0 = constant_op.constant(_SimpleColorRamp())
      jpeg0 = image_ops.encode_jpeg(image0)
      image1 = image_ops.decode_jpeg(jpeg0, dct_method="INTEGER_FAST")
      image2 = image_ops.decode_jpeg(jpeg0)
      image1, image2 = sess.run([image1, image2])

      # The images should be the same.
      self.assertAllClose(image1, image2)

  def testShape(self):
    with self.test_session(use_gpu=True) as sess:
      jpeg = constant_op.constant("nonsense")
      for channels in 0, 1, 3:
        image = image_ops.decode_jpeg(jpeg, channels=channels)
        self.assertEqual(image.get_shape().as_list(),
                         [None, None, channels or None])

  def testExtractJpegShape(self):
    # Read a real jpeg and verify shape.
    path = ("tensorflow/core/lib/jpeg/testdata/"
            "jpeg_merge_test1.jpg")
    with self.test_session(use_gpu=True) as sess:
      jpeg = io_ops.read_file(path)
      # Extract shape without decoding.
      [image_shape] = sess.run([image_ops.extract_jpeg_shape(jpeg)])
      self.assertEqual(image_shape.tolist(), [256, 128, 3])

  def testExtractJpegShapeforCmyk(self):
    # Read a cmyk jpeg image, and verify its shape.
    path = ("tensorflow/core/lib/jpeg/testdata/"
            "jpeg_merge_test1_cmyk.jpg")
    with self.test_session(use_gpu=True) as sess:
      jpeg = io_ops.read_file(path)
      [image_shape] = sess.run([image_ops.extract_jpeg_shape(jpeg)])
      # Cmyk jpeg image has 4 channels.
      self.assertEqual(image_shape.tolist(), [256, 128, 4])


class PngTest(test_util.TensorFlowTestCase):

  def testExisting(self):
    # Read some real PNGs, converting to different channel numbers
    prefix = "tensorflow/core/lib/png/testdata/"
    inputs = ((1, "lena_gray.png"), (4, "lena_rgba.png"),
              (3, "lena_palette.png"), (4, "lena_palette_trns.png"))
    for channels_in, filename in inputs:
      for channels in 0, 1, 3, 4:
        with self.test_session(use_gpu=True) as sess:
          png0 = io_ops.read_file(prefix + filename)
          image0 = image_ops.decode_png(png0, channels=channels)
          png0, image0 = sess.run([png0, image0])
          self.assertEqual(image0.shape, (26, 51, channels or channels_in))
          if channels == channels_in:
            image1 = image_ops.decode_png(image_ops.encode_png(image0))
            self.assertAllEqual(image0, image1.eval())

  def testSynthetic(self):
    with self.test_session(use_gpu=True) as sess:
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
    with self.test_session(use_gpu=True) as sess:
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
    with self.test_session(use_gpu=True) as sess:
      # Strip the b channel from an rgb image to get a two-channel image.
      gray_alpha = _SimpleColorRamp()[:, :, 0:2]
      image0 = constant_op.constant(gray_alpha)
      png0 = image_ops.encode_png(image0, compression=7)
      image1 = image_ops.decode_png(png0)
      png0, image0, image1 = sess.run([png0, image0, image1])
      self.assertEqual(2, image0.shape[-1])
      self.assertAllEqual(image0, image1)

  def testSyntheticTwoChannelUint16(self):
    with self.test_session(use_gpu=True) as sess:
      # Strip the b channel from an rgb image to get a two-channel image.
      gray_alpha = _SimpleColorRamp()[:, :, 0:2]
      image0 = constant_op.constant(gray_alpha, dtype=dtypes.uint16)
      png0 = image_ops.encode_png(image0, compression=7)
      image1 = image_ops.decode_png(png0, dtype=dtypes.uint16)
      png0, image0, image1 = sess.run([png0, image0, image1])
      self.assertEqual(2, image0.shape[-1])
      self.assertAllEqual(image0, image1)

  def testShape(self):
    with self.test_session(use_gpu=True):
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

    with self.test_session(use_gpu=True) as sess:
      gif0 = io_ops.read_file(prefix + filename)
      image0 = image_ops.decode_gif(gif0)
      gif0, image0 = sess.run([gif0, image0])

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

  def testShape(self):
    with self.test_session(use_gpu=True) as sess:
      gif = constant_op.constant("nonsense")
      image = image_ops.decode_gif(gif)
      self.assertEqual(image.get_shape().as_list(), [None, None, None, 3])


class ConvertImageTest(test_util.TensorFlowTestCase):

  def _convert(self, original, original_dtype, output_dtype, expected):
    x_np = np.array(original, dtype=original_dtype.as_numpy_dtype())
    y_np = np.array(expected, dtype=output_dtype.as_numpy_dtype())

    with self.test_session(use_gpu=True):
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

  def testNoConvert(self):
    # Make sure converting to the same data type creates only an identity op
    with self.test_session(use_gpu=True):
      image = constant_op.constant([1], dtype=dtypes.uint8)
      image_ops.convert_image_dtype(image, dtypes.uint8)
      y = image_ops.convert_image_dtype(image, dtypes.uint8)
      self.assertEquals(y.op.type, "Identity")
      self.assertEquals(y.op.inputs[0], image)

  def testConvertBetweenInteger(self):
    # Make sure converting to between integer types scales appropriately
    with self.test_session(use_gpu=True):
      self._convert([0, 255], dtypes.uint8, dtypes.int16, [0, 255 * 128])
      self._convert([0, 32767], dtypes.int16, dtypes.uint8, [0, 255])
      self._convert([0, 2**32], dtypes.int64, dtypes.int32, [0, 1])
      self._convert([0, 1], dtypes.int32, dtypes.int64, [0, 2**32])

  def testConvertBetweenFloat(self):
    # Make sure converting to between float types does nothing interesting
    with self.test_session(use_gpu=True):
      self._convert([-1.0, 0, 1.0, 200000], dtypes.float32, dtypes.float64,
                    [-1.0, 0, 1.0, 200000])
      self._convert([-1.0, 0, 1.0, 200000], dtypes.float64, dtypes.float32,
                    [-1.0, 0, 1.0, 200000])

  def testConvertBetweenIntegerAndFloat(self):
    # Make sure converting from and to a float type scales appropriately
    with self.test_session(use_gpu=True):
      self._convert([0, 1, 255], dtypes.uint8, dtypes.float32,
                    [0, 1.0 / 255.0, 1])
      self._convert([0, 1.1 / 255.0, 1], dtypes.float32, dtypes.uint8,
                    [0, 1, 255])

  def testConvertBetweenInt16AndInt8(self):
    with self.test_session(use_gpu=True):
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
    with self.test_session(use_gpu=True):
      # Add a constant to the TensorFlow graph that holds the input.
      x_tf = constant_op.constant(x_np, shape=x_np.shape)

      # Add ops for calculating the total variation using TensorFlow.
      y = image_ops.total_variation(images=x_tf)

      # Run the TensorFlow session to calculate the result.
      y_tf = y.eval()

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

  def testTotalVariationNumpy(self):
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

    # Scale the pixel-values by a float. This scales the total variation as well.
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

  def testFormats(self):
    prefix = "tensorflow/core/lib"
    paths = ("png/testdata/lena_gray.png", "jpeg/testdata/jpeg_merge_test1.jpg",
             "gif/testdata/lena.gif")
    decoders = {
        "jpeg": functools.partial(image_ops.decode_jpeg, channels=3),
        "png": functools.partial(image_ops.decode_png, channels=3),
        "gif": lambda s: array_ops.squeeze(image_ops.decode_gif(s), axis=0),
    }
    with self.test_session():
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
    with self.test_session():
      for decode in image_ops.decode_jpeg, image_ops.decode_png:
        with self.assertRaisesOpError(r"Got 12 frames"):
          decode(io_ops.read_file(path)).eval()


class NonMaxSuppressionTest(test_util.TensorFlowTestCase):

  def testSelectFromThreeClusters(self):
    boxes_np = [[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, -0.1, 1, 0.9],
                [0, 10, 1, 11], [0, 10.1, 1, 11.1], [0, 100, 1, 101]]
    scores_np = [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]
    max_output_size_np = 3
    iou_threshold_np = 0.5
    with self.test_session():
      boxes = constant_op.constant(boxes_np)
      scores = constant_op.constant(scores_np)
      max_output_size = constant_op.constant(max_output_size_np)
      iou_threshold = constant_op.constant(iou_threshold_np)
      selected_indices = image_ops.non_max_suppression(
          boxes, scores, max_output_size, iou_threshold).eval()
      self.assertAllClose(selected_indices, [3, 0, 5])

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
    # of shape [num_boxes]. So an error will thrown.
    with self.assertRaisesRegexp(ValueError,
                                 "Dimensions must be equal, but are 1 and 2"):
      boxes = constant_op.constant([[0.0, 0.0, 1.0, 1.0]])
      scores = constant_op.constant([0.9, 0.75])
      selected_indices = image_ops.non_max_suppression(boxes, scores, 3, 0.5)

    # The scores should be 1D of shape [num_boxes].
    with self.assertRaisesRegexp(ValueError,
                                 "Shape must be rank 1 but is rank 2"):
      boxes = constant_op.constant([[0.0, 0.0, 1.0, 1.0]])
      scores = constant_op.constant([[0.9]])
      image_ops.non_max_suppression(boxes, scores, 3, 0.5)

    # The max_output_size should be a scaler (0-D).
    with self.assertRaisesRegexp(ValueError,
                                 "Shape must be rank 0 but is rank 1"):
      boxes = constant_op.constant([[0.0, 0.0, 1.0, 1.0]])
      scores = constant_op.constant([0.9])
      image_ops.non_max_suppression(boxes, scores, [3], 0.5)

    # The iou_threshold should be a scaler (0-D).
    with self.assertRaisesRegexp(ValueError,
                                 "Shape must be rank 0 but is rank 2"):
      boxes = constant_op.constant([[0.0, 0.0, 1.0, 1.0]])
      scores = constant_op.constant([0.9])
      image_ops.non_max_suppression(boxes, scores, 3, [[0.5]])


class VerifyCompatibleImageShapesTest(test_util.TensorFlowTestCase):
  """Tests utility function used by ssim() and psnr()."""

  def testWrongDims(self):
    img = array_ops.placeholder(dtype=dtypes.float32)
    img_np = np.array((2, 2))

    with self.test_session(use_gpu=True) as sess:
      _, _, checks = image_ops_impl._verify_compatible_image_shapes(img, img)
      with self.assertRaises(errors.InvalidArgumentError):
        sess.run(checks, {img: img_np})

  def testShapeMismatch(self):
    img1 = array_ops.placeholder(dtype=dtypes.float32)
    img2 = array_ops.placeholder(dtype=dtypes.float32)

    img1_np = np.array([1, 2, 2, 1])
    img2_np = np.array([1, 3, 3, 1])

    with self.test_session(use_gpu=True) as sess:
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
    im, = sess.run([im])
    return np.expand_dims(im, axis=0)

  def _LoadTestImages(self):
    with self.test_session(use_gpu=True) as sess:
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

  def testPSNRSingleImage(self):
    image1 = self._RandomImage((8, 8, 1), 1)
    image2 = self._RandomImage((8, 8, 1), 1)
    psnr = self._PSNR_NumPy(image1, image2, 1)

    with self.test_session(use_gpu=True):
      tf_image1 = constant_op.constant(image1, shape=image1.shape,
                                       dtype=dtypes.float32)
      tf_image2 = constant_op.constant(image2, shape=image2.shape,
                                       dtype=dtypes.float32)
      tf_psnr = image_ops.psnr(tf_image1, tf_image2, 1.0, "psnr").eval()
      self.assertAllClose(psnr, tf_psnr, atol=0.001)

  def testPSNRMultiImage(self):
    image1 = self._RandomImage((10, 8, 8, 1), 1)
    image2 = self._RandomImage((10, 8, 8, 1), 1)
    psnr = self._PSNR_NumPy(image1, image2, 1)

    with self.test_session(use_gpu=True):
      tf_image1 = constant_op.constant(image1, shape=image1.shape,
                                       dtype=dtypes.float32)
      tf_image2 = constant_op.constant(image2, shape=image2.shape,
                                       dtype=dtypes.float32)
      tf_psnr = image_ops.psnr(tf_image1, tf_image2, 1, "psnr").eval()
      self.assertAllClose(psnr, tf_psnr, atol=0.001)

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
    with self.test_session(use_gpu=True):
      tf_q20 = constant_op.constant(q20, shape=q20.shape, dtype=dtypes.float32)
      tf_q72 = constant_op.constant(q72, shape=q72.shape, dtype=dtypes.float32)
      tf_q95 = constant_op.constant(q95, shape=q95.shape, dtype=dtypes.float32)
      tf_psnr1 = image_ops.psnr(tf_q20, tf_q72, 1, "psnr1").eval()
      tf_psnr2 = image_ops.psnr(tf_q20, tf_q95, 1, "psnr2").eval()
      tf_psnr3 = image_ops.psnr(tf_q72, tf_q95, 1, "psnr3").eval()
      self.assertAllClose(psnr1, tf_psnr1, atol=0.001)
      self.assertAllClose(psnr2, tf_psnr2, atol=0.001)
      self.assertAllClose(psnr3, tf_psnr3, atol=0.001)

  def testInfinity(self):
    q20, _, _ = self._LoadTestImages()
    psnr = self._PSNR_NumPy(q20, q20, 1)
    with self.test_session(use_gpu=True):
      tf_q20 = constant_op.constant(q20, shape=q20.shape, dtype=dtypes.float32)
      tf_psnr = image_ops.psnr(tf_q20, tf_q20, 1, "psnr").eval()
      self.assertAllClose(psnr, tf_psnr, atol=0.001)

  def testInt(self):
    img1 = self._RandomImage((10, 8, 8, 1), 255)
    img2 = self._RandomImage((10, 8, 8, 1), 255)
    img1 = constant_op.constant(img1, dtypes.uint8)
    img2 = constant_op.constant(img2, dtypes.uint8)
    psnr_uint8 = image_ops.psnr(img1, img2, 255)
    img1 = image_ops.convert_image_dtype(img1, dtypes.float32)
    img2 = image_ops.convert_image_dtype(img2, dtypes.float32)
    psnr_float32 = image_ops.psnr(img1, img2, 1.0)
    with self.test_session(use_gpu=True):
      self.assertAllClose(psnr_uint8.eval(), psnr_float32.eval(), atol=0.001)


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
    im, = sess.run([im])
    return np.expand_dims(im, axis=0)

  def _LoadTestImages(self):
    with self.test_session(use_gpu=True) as sess:
      return [self._LoadTestImage(sess, f) for f in self._filenames]

  def _RandomImage(self, shape, max_val):
    """Returns an image or image batch with given shape."""
    return np.random.rand(*shape).astype(np.float32) * max_val

  def testAgainstMatlab(self):
    """Tests against values produced by Matlab."""
    img = self._LoadTestImages()
    expected = self._ssim[np.triu_indices(3)]

    ph = [array_ops.placeholder(dtype=dtypes.float32) for _ in range(2)]
    ssim = image_ops.ssim(*ph, max_val=1.0)
    with self.test_session(use_gpu=True):
      scores = [ssim.eval(dict(zip(ph, t)))
                for t in itertools.combinations_with_replacement(img, 2)]
    self.assertAllClose(expected, np.squeeze(scores), atol=1e-4)

  def testBatch(self):
    img = self._LoadTestImages()
    expected = self._ssim[np.triu_indices(3, k=1)]

    img1, img2 = zip(*itertools.combinations(img, 2))
    img1 = np.concatenate(img1)
    img2 = np.concatenate(img2)

    ssim = image_ops.ssim(constant_op.constant(img1),
                          constant_op.constant(img2), 1.0)
    with self.test_session(use_gpu=True):
      self.assertAllClose(expected, ssim.eval(), atol=1e-4)

  def testBroadcast(self):
    img = self._LoadTestImages()[:2]
    expected = self._ssim[:2, :2]

    img = constant_op.constant(np.concatenate(img))
    img1 = array_ops.expand_dims(img, axis=0)  # batch dims: 1, 2.
    img2 = array_ops.expand_dims(img, axis=1)  # batch dims: 2, 1.

    ssim = image_ops.ssim(img1, img2, 1.0)
    with self.test_session(use_gpu=True):
      self.assertAllClose(expected, ssim.eval(), atol=1e-4)

  def testNegative(self):
    """Tests against negative SSIM index."""
    step = np.expand_dims(np.arange(0, 256, 16, dtype=np.uint8), axis=0)
    img1 = np.tile(step, (16, 1))
    img2 = np.fliplr(img1)

    img1 = img1.reshape((1, 16, 16, 1))
    img2 = img2.reshape((1, 16, 16, 1))

    ssim = image_ops.ssim(constant_op.constant(img1),
                          constant_op.constant(img2), 255)
    with self.test_session(use_gpu=True):
      self.assertLess(ssim.eval(), 0)

  def testInt(self):
    img1 = self._RandomImage((1, 16, 16, 3), 255)
    img2 = self._RandomImage((1, 16, 16, 3), 255)
    img1 = constant_op.constant(img1, dtypes.uint8)
    img2 = constant_op.constant(img2, dtypes.uint8)
    ssim_uint8 = image_ops.ssim(img1, img2, 255)
    img1 = image_ops.convert_image_dtype(img1, dtypes.float32)
    img2 = image_ops.convert_image_dtype(img2, dtypes.float32)
    ssim_float32 = image_ops.ssim(img1, img2, 1.0)
    with self.test_session(use_gpu=True):
      self.assertAllClose(ssim_uint8.eval(), ssim_float32.eval(), atol=0.001)


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
    im, = sess.run([im])
    return np.expand_dims(im, axis=0)

  def _LoadTestImages(self):
    with self.test_session(use_gpu=True) as sess:
      return [self._LoadTestImage(sess, f) for f in self._filenames]

  def _RandomImage(self, shape, max_val):
    """Returns an image or image batch with given shape."""
    return np.random.rand(*shape).astype(np.float32) * max_val

  def testAgainstMatlab(self):
    """Tests against MS-SSIM computed with Matlab implementation.

    For color images, MS-SSIM scores are averaged over color channels.
    """
    img = self._LoadTestImages()
    expected = self._msssim[np.triu_indices(3)]

    ph = [array_ops.placeholder(dtype=dtypes.float32) for _ in range(2)]
    msssim = image_ops.ssim_multiscale(*ph, max_val=1.0)
    with self.test_session(use_gpu=True):
      scores = [msssim.eval(dict(zip(ph, t)))
                for t in itertools.combinations_with_replacement(img, 2)]

    self.assertAllClose(expected, np.squeeze(scores), atol=1e-4)

  def testUnweightedIsDifferentiable(self):
    img = self._LoadTestImages()
    ph = [array_ops.placeholder(dtype=dtypes.float32) for _ in range(2)]
    scalar = constant_op.constant(1.0, dtype=dtypes.float32)
    scaled_ph = [x * scalar for x in ph]
    msssim = image_ops.ssim_multiscale(*scaled_ph, max_val=1.0,
                                       power_factors=(1, 1, 1, 1, 1))
    grads = gradients.gradients(msssim, scalar)
    with self.test_session(use_gpu=True) as sess:
      np_grads = sess.run(grads, feed_dict={ph[0]: img[0], ph[1]: img[1]})
    self.assertTrue(np.isfinite(np_grads).all())

  def testBatch(self):
    """Tests MS-SSIM computed in batch."""
    img = self._LoadTestImages()
    expected = self._msssim[np.triu_indices(3, k=1)]

    img1, img2 = zip(*itertools.combinations(img, 2))
    img1 = np.concatenate(img1)
    img2 = np.concatenate(img2)

    msssim = image_ops.ssim_multiscale(constant_op.constant(img1),
                                       constant_op.constant(img2), 1.0)
    with self.test_session(use_gpu=True):
      self.assertAllClose(expected, msssim.eval(), 1e-4)

  def testBroadcast(self):
    """Tests MS-SSIM broadcasting."""
    img = self._LoadTestImages()[:2]
    expected = self._msssim[:2, :2]

    img = constant_op.constant(np.concatenate(img))
    img1 = array_ops.expand_dims(img, axis=0)  # batch dims: 1, 2.
    img2 = array_ops.expand_dims(img, axis=1)  # batch dims: 2, 1.

    score_tensor = image_ops.ssim_multiscale(img1, img2, 1.0)
    with self.test_session(use_gpu=True):
      self.assertAllClose(expected, score_tensor.eval(), 1e-4)

  def testRange(self):
    """Tests against low MS-SSIM score.

    MS-SSIM is a geometric mean of SSIM and CS scores of various scales.
    If any of the value is negative so that the geometric mean is not
    well-defined, then treat the MS-SSIM score as zero.
    """
    with self.test_session(use_gpu=True) as sess:
      img1 = self._LoadTestImage(sess, "checkerboard1.png")
      img2 = self._LoadTestImage(sess, "checkerboard3.png")
      images = [img1, img2, np.zeros_like(img1),
                np.full_like(img1, fill_value=255)]

      images = [ops.convert_to_tensor(x, dtype=dtypes.float32) for x in images]
      msssim_ops = [image_ops.ssim_multiscale(x, y, 1.0)
                    for x, y in itertools.combinations(images, 2)]
      msssim = sess.run(msssim_ops)
      msssim = np.squeeze(msssim)

    self.assertTrue(np.all(msssim >= 0.0))
    self.assertTrue(np.all(msssim <= 1.0))

  def testInt(self):
    img1 = self._RandomImage((1, 180, 240, 3), 255)
    img2 = self._RandomImage((1, 180, 240, 3), 255)
    img1 = constant_op.constant(img1, dtypes.uint8)
    img2 = constant_op.constant(img2, dtypes.uint8)
    ssim_uint8 = image_ops.ssim_multiscale(img1, img2, 255)
    img1 = image_ops.convert_image_dtype(img1, dtypes.float32)
    img2 = image_ops.convert_image_dtype(img2, dtypes.float32)
    ssim_float32 = image_ops.ssim_multiscale(img1, img2, 1.0)
    with self.test_session(use_gpu=True):
      self.assertAllClose(ssim_uint8.eval(), ssim_float32.eval(), atol=0.001)


class ImageGradientsTest(test_util.TensorFlowTestCase):

  def testImageGradients(self):
    shape = [1, 2, 4, 1]
    img = constant_op.constant([[1, 3, 4, 2], [8, 7, 5, 6]])
    img = array_ops.reshape(img, shape)

    expected_dy = np.reshape([[7, 4, 1, 4], [0, 0, 0, 0]], shape)
    expected_dx = np.reshape([[2, 1, -2, 0], [-1, -2, 1, 0]], shape)

    dy, dx = image_ops.image_gradients(img)
    with self.test_session():
      actual_dy = dy.eval()
      actual_dx = dx.eval()
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
    with self.test_session(use_gpu=True):
      actual_dy = dy.eval()
      actual_dx = dx.eval()
      self.assertAllClose(expected_dy, actual_dy)
      self.assertAllClose(expected_dx, actual_dx)

  def testImageGradientsBadShape(self):
    # [2 x 4] image but missing batch and depth dimensions.
    img = constant_op.constant([[1, 3, 4, 2], [8, 7, 5, 6]])
    with self.assertRaises(ValueError):
      image_ops.image_gradients(img)


class SobelEdgesTest(test_util.TensorFlowTestCase):

  def testSobelEdges1x2x3x1(self):
    img = constant_op.constant([[1, 3, 6], [4, 1, 5]],
                               dtype=dtypes.float32, shape=[1, 2, 3, 1])
    expected = np.reshape([[[0, 0], [0, 12], [0, 0]],
                           [[0, 0], [0, 12], [0, 0]]], [1, 2, 3, 1, 2])
    sobel = image_ops.sobel_edges(img)
    with self.test_session(use_gpu=True):
      actual_sobel = sobel.eval()
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
    with self.test_session(use_gpu=True):
      actual_sobel = sobel.eval()
      self.assertAllClose(expected_batch, actual_sobel)


class DecodeImageTest(test_util.TensorFlowTestCase):

  def testJpegUint16(self):
    with self.test_session(use_gpu=True) as sess:
      base = "tensorflow/core/lib/jpeg/testdata"
      jpeg0 = io_ops.read_file(os.path.join(base, "jpeg_merge_test1.jpg"))
      image0 = image_ops.decode_image(jpeg0, dtype=dtypes.uint16)
      image1 = image_ops.convert_image_dtype(image_ops.decode_jpeg(jpeg0),
                                             dtypes.uint16)
      image0, image1 = sess.run([image0, image1])
      self.assertAllEqual(image0, image1)

  def testPngUint16(self):
    with self.test_session(use_gpu=True) as sess:
      base = "tensorflow/core/lib/png/testdata"
      png0 = io_ops.read_file(os.path.join(base, "lena_rgba.png"))
      image0 = image_ops.decode_image(png0, dtype=dtypes.uint16)
      image1 = image_ops.convert_image_dtype(
          image_ops.decode_png(png0, dtype=dtypes.uint16), dtypes.uint16)
      image0, image1 = sess.run([image0, image1])
      self.assertAllEqual(image0, image1)

  def testGifUint16(self):
    with self.test_session(use_gpu=True) as sess:
      base = "tensorflow/core/lib/gif/testdata"
      gif0 = io_ops.read_file(os.path.join(base, "scan.gif"))
      image0 = image_ops.decode_image(gif0, dtype=dtypes.uint16)
      image1 = image_ops.convert_image_dtype(image_ops.decode_gif(gif0),
                                             dtypes.uint16)
      image0, image1 = sess.run([image0, image1])
      self.assertAllEqual(image0, image1)

  def testBmpUint16(self):
    with self.test_session(use_gpu=True) as sess:
      base = "tensorflow/core/lib/bmp/testdata"
      bmp0 = io_ops.read_file(os.path.join(base, "lena.bmp"))
      image0 = image_ops.decode_image(bmp0, dtype=dtypes.uint16)
      image1 = image_ops.convert_image_dtype(image_ops.decode_bmp(bmp0),
                                             dtypes.uint16)
      image0, image1 = sess.run([image0, image1])
      self.assertAllEqual(image0, image1)

  def testJpegFloat32(self):
    with self.test_session(use_gpu=True) as sess:
      base = "tensorflow/core/lib/jpeg/testdata"
      jpeg0 = io_ops.read_file(os.path.join(base, "jpeg_merge_test1.jpg"))
      image0 = image_ops.decode_image(jpeg0, dtype=dtypes.float32)
      image1 = image_ops.convert_image_dtype(image_ops.decode_jpeg(jpeg0),
                                             dtypes.float32)
      image0, image1 = sess.run([image0, image1])
      self.assertAllEqual(image0, image1)

  def testPngFloat32(self):
    with self.test_session(use_gpu=True) as sess:
      base = "tensorflow/core/lib/png/testdata"
      png0 = io_ops.read_file(os.path.join(base, "lena_rgba.png"))
      image0 = image_ops.decode_image(png0, dtype=dtypes.float32)
      image1 = image_ops.convert_image_dtype(
          image_ops.decode_png(png0, dtype=dtypes.uint16), dtypes.float32)
      image0, image1 = sess.run([image0, image1])
      self.assertAllEqual(image0, image1)

  def testGifFloat32(self):
    with self.test_session(use_gpu=True) as sess:
      base = "tensorflow/core/lib/gif/testdata"
      gif0 = io_ops.read_file(os.path.join(base, "scan.gif"))
      image0 = image_ops.decode_image(gif0, dtype=dtypes.float32)
      image1 = image_ops.convert_image_dtype(image_ops.decode_gif(gif0),
                                             dtypes.float32)
      image0, image1 = sess.run([image0, image1])
      self.assertAllEqual(image0, image1)

  def testBmpFloat32(self):
    with self.test_session(use_gpu=True) as sess:
      base = "tensorflow/core/lib/bmp/testdata"
      bmp0 = io_ops.read_file(os.path.join(base, "lena.bmp"))
      image0 = image_ops.decode_image(bmp0, dtype=dtypes.float32)
      image1 = image_ops.convert_image_dtype(image_ops.decode_bmp(bmp0),
                                             dtypes.float32)
      image0, image1 = sess.run([image0, image1])
      self.assertAllEqual(image0, image1)


if __name__ == "__main__":
  googletest.main()
