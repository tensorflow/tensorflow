# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for python distort_image_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.image.python.ops import distort_image_ops
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test


# TODO(huangyp): also measure the differences between AdjustHsvInYiq and
# AdjustHsv in core.
class AdjustHueInYiqTest(test_util.TensorFlowTestCase):

  def _adjust_hue_in_yiq_np(self, x_np, delta_h):
    """Rotate hue in YIQ space.

    Mathematically we first convert rgb color to yiq space, rotate the hue
    degrees, and then convert back to rgb.

    Args:
      x_np: input x with last dimension = 3.
      delta_h: degree of hue rotation, in radians.

    Returns:
      Adjusted y with the same shape as x_np.
    """
    self.assertEqual(x_np.shape[-1], 3)
    x_v = x_np.reshape([-1, 3])
    y_v = np.ndarray(x_v.shape, dtype=x_v.dtype)
    u = np.cos(delta_h)
    w = np.sin(delta_h)
    # Projection matrix from RGB to YIQ. Numbers from wikipedia
    # https://en.wikipedia.org/wiki/YIQ
    tyiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322],
                     [0.211, -0.523, 0.312]])
    y_v = np.dot(x_v, tyiq.T)
    # Hue rotation matrix in YIQ space.
    hue_rotation = np.array([[1.0, 0.0, 0.0], [0.0, u, -w], [0.0, w, u]])
    y_v = np.dot(y_v, hue_rotation.T)
    # Projecting back to RGB space.
    y_v = np.dot(y_v, np.linalg.inv(tyiq).T)
    return y_v.reshape(x_np.shape)

  def _adjust_hue_in_yiq_tf(self, x_np, delta_h):
    with self.test_session(use_gpu=True):
      x = constant_op.constant(x_np)
      y = distort_image_ops.adjust_hsv_in_yiq(x, delta_h, 1, 1)
      y_tf = y.eval()
    return y_tf

  def test_adjust_random_hue_in_yiq(self):
    x_shapes = [
        [2, 2, 3],
        [4, 2, 3],
        [2, 4, 3],
        [2, 5, 3],
        [1000, 1, 3],
    ]
    test_styles = [
        'all_random',
        'rg_same',
        'rb_same',
        'gb_same',
        'rgb_same',
    ]
    for x_shape in x_shapes:
      for test_style in test_styles:
        x_np = np.random.rand(*x_shape) * 255.
        delta_h = (np.random.rand() * 2.0 - 1.0) * np.pi
        if test_style == 'all_random':
          pass
        elif test_style == 'rg_same':
          x_np[..., 1] = x_np[..., 0]
        elif test_style == 'rb_same':
          x_np[..., 2] = x_np[..., 0]
        elif test_style == 'gb_same':
          x_np[..., 2] = x_np[..., 1]
        elif test_style == 'rgb_same':
          x_np[..., 1] = x_np[..., 0]
          x_np[..., 2] = x_np[..., 0]
        else:
          raise AssertionError('Invalid test style: %s' % (test_style))
        y_np = self._adjust_hue_in_yiq_np(x_np, delta_h)
        y_tf = self._adjust_hue_in_yiq_tf(x_np, delta_h)
        self.assertAllClose(y_tf, y_np, rtol=2e-4, atol=1e-4)

  def test_invalid_shapes(self):
    x_np = np.random.rand(2, 3) * 255.
    delta_h = np.random.rand() * 2.0 - 1.0
    with self.assertRaisesRegexp(ValueError, 'Shape must be at least rank 3'):
      self._adjust_hue_in_yiq_tf(x_np, delta_h)
    x_np = np.random.rand(4, 2, 4) * 255.
    delta_h = np.random.rand() * 2.0 - 1.0
    with self.assertRaisesOpError('input must have 3 channels but instead has '
                                  '4 channels'):
      self._adjust_hue_in_yiq_tf(x_np, delta_h)


class AdjustValueInYiqTest(test_util.TensorFlowTestCase):

  def _adjust_value_in_yiq_np(self, x_np, scale):
    return x_np * scale

  def _adjust_value_in_yiq_tf(self, x_np, scale):
    with self.test_session(use_gpu=True):
      x = constant_op.constant(x_np)
      y = distort_image_ops.adjust_hsv_in_yiq(x, 0, 1, scale)
      y_tf = y.eval()
    return y_tf

  def test_adjust_random_value_in_yiq(self):
    x_shapes = [
        [2, 2, 3],
        [4, 2, 3],
        [2, 4, 3],
        [2, 5, 3],
        [1000, 1, 3],
    ]
    test_styles = [
        'all_random',
        'rg_same',
        'rb_same',
        'gb_same',
        'rgb_same',
    ]
    for x_shape in x_shapes:
      for test_style in test_styles:
        x_np = np.random.rand(*x_shape) * 255.
        scale = np.random.rand() * 2.0 - 1.0
        if test_style == 'all_random':
          pass
        elif test_style == 'rg_same':
          x_np[..., 1] = x_np[..., 0]
        elif test_style == 'rb_same':
          x_np[..., 2] = x_np[..., 0]
        elif test_style == 'gb_same':
          x_np[..., 2] = x_np[..., 1]
        elif test_style == 'rgb_same':
          x_np[..., 1] = x_np[..., 0]
          x_np[..., 2] = x_np[..., 0]
        else:
          raise AssertionError('Invalid test style: %s' % (test_style))
        y_np = self._adjust_value_in_yiq_np(x_np, scale)
        y_tf = self._adjust_value_in_yiq_tf(x_np, scale)
        self.assertAllClose(y_tf, y_np, rtol=2e-4, atol=1e-4)

  def test_invalid_shapes(self):
    x_np = np.random.rand(2, 3) * 255.
    scale = np.random.rand() * 2.0 - 1.0
    with self.assertRaisesRegexp(ValueError, 'Shape must be at least rank 3'):
      self._adjust_value_in_yiq_tf(x_np, scale)
    x_np = np.random.rand(4, 2, 4) * 255.
    scale = np.random.rand() * 2.0 - 1.0
    with self.assertRaisesOpError('input must have 3 channels but instead has '
                                  '4 channels'):
      self._adjust_value_in_yiq_tf(x_np, scale)


class AdjustSaturationInYiqTest(test_util.TensorFlowTestCase):

  def _adjust_saturation_in_yiq_tf(self, x_np, scale):
    with self.test_session(use_gpu=True):
      x = constant_op.constant(x_np)
      y = distort_image_ops.adjust_hsv_in_yiq(x, 0, scale, 1)
      y_tf = y.eval()
    return y_tf

  def _adjust_saturation_in_yiq_np(self, x_np, scale):
    """Adjust saturation using linear interpolation."""
    rgb_weights = np.array([0.299, 0.587, 0.114])
    gray = np.sum(x_np * rgb_weights, axis=-1, keepdims=True)
    y_v = x_np * scale + gray * (1 - scale)
    return y_v

  def test_adjust_random_saturation_in_yiq(self):
    x_shapes = [
        [2, 2, 3],
        [4, 2, 3],
        [2, 4, 3],
        [2, 5, 3],
        [1000, 1, 3],
    ]
    test_styles = [
        'all_random',
        'rg_same',
        'rb_same',
        'gb_same',
        'rgb_same',
    ]
    with self.test_session():
      for x_shape in x_shapes:
        for test_style in test_styles:
          x_np = np.random.rand(*x_shape) * 255.
          scale = np.random.rand() * 2.0 - 1.0
          if test_style == 'all_random':
            pass
          elif test_style == 'rg_same':
            x_np[..., 1] = x_np[..., 0]
          elif test_style == 'rb_same':
            x_np[..., 2] = x_np[..., 0]
          elif test_style == 'gb_same':
            x_np[..., 2] = x_np[..., 1]
          elif test_style == 'rgb_same':
            x_np[..., 1] = x_np[..., 0]
            x_np[..., 2] = x_np[..., 0]
          else:
            raise AssertionError('Invalid test style: %s' % (test_style))
          y_baseline = self._adjust_saturation_in_yiq_np(x_np, scale)
          y_tf = self._adjust_saturation_in_yiq_tf(x_np, scale)
          self.assertAllClose(y_tf, y_baseline, rtol=2e-4, atol=1e-4)

  def test_invalid_shapes(self):
    x_np = np.random.rand(2, 3) * 255.
    scale = np.random.rand() * 2.0 - 1.0
    with self.assertRaisesRegexp(ValueError, 'Shape must be at least rank 3'):
      self._adjust_saturation_in_yiq_tf(x_np, scale)
    x_np = np.random.rand(4, 2, 4) * 255.
    scale = np.random.rand() * 2.0 - 1.0
    with self.assertRaisesOpError('input must have 3 channels but instead has '
                                  '4 channels'):
      self._adjust_saturation_in_yiq_tf(x_np, scale)


class AdjustHueInYiqBenchmark(test.Benchmark):

  def _benchmark_adjust_hue_in_yiq(self, device, cpu_count):
    image_shape = [299, 299, 3]
    warmup_rounds = 100
    benchmark_rounds = 1000
    config = config_pb2.ConfigProto()
    if cpu_count is not None:
      config.inter_op_parallelism_threads = 1
      config.intra_op_parallelism_threads = cpu_count
    with session.Session('', graph=ops.Graph(), config=config) as sess:
      with ops.device(device):
        inputs = variables.Variable(
            random_ops.random_uniform(image_shape, dtype=dtypes.float32) * 255,
            trainable=False,
            dtype=dtypes.float32)
        delta = constant_op.constant(0.1, dtype=dtypes.float32)
        outputs = distort_image_ops.adjust_hsv_in_yiq(inputs, delta, 1, 1)
        run_op = control_flow_ops.group(outputs)
        sess.run(variables.global_variables_initializer())
        for i in xrange(warmup_rounds + benchmark_rounds):
          if i == warmup_rounds:
            start = time.time()
          sess.run(run_op)
    end = time.time()
    step_time = (end - start) / benchmark_rounds
    tag = device + '_%s' % (cpu_count if cpu_count is not None else 'all')
    print('benchmarkadjust_hue_in_yiq_299_299_3_%s step_time: %.2f us' %
          (tag, step_time * 1e6))
    self.report_benchmark(
        name='benchmarkadjust_hue_in_yiq_299_299_3_%s' % (tag),
        iters=benchmark_rounds,
        wall_time=step_time)

  def benchmark_adjust_hue_in_yiqCpu1(self):
    self._benchmark_adjust_hue_in_yiq('/cpu:0', 1)

  def benchmark_adjust_hue_in_yiqCpuAll(self):
    self._benchmark_adjust_hue_in_yiq('/cpu:0', None)

  def benchmark_adjust_hue_in_yiq_gpu_all(self):
    self._benchmark_adjust_hue_in_yiq(test.gpu_device_name(), None)


class AdjustSaturationInYiqBenchmark(test.Benchmark):

  def _benchmark_adjust_saturation_in_yiq(self, device, cpu_count):
    image_shape = [299, 299, 3]
    warmup_rounds = 100
    benchmark_rounds = 1000
    config = config_pb2.ConfigProto()
    if cpu_count is not None:
      config.inter_op_parallelism_threads = 1
      config.intra_op_parallelism_threads = cpu_count
    with session.Session('', graph=ops.Graph(), config=config) as sess:
      with ops.device(device):
        inputs = variables.Variable(
            random_ops.random_uniform(image_shape, dtype=dtypes.float32) * 255,
            trainable=False,
            dtype=dtypes.float32)
        scale = constant_op.constant(0.1, dtype=dtypes.float32)
        outputs = distort_image_ops.adjust_hsv_in_yiq(inputs, 0, scale, 1)
        run_op = control_flow_ops.group(outputs)
        sess.run(variables.global_variables_initializer())
        for _ in xrange(warmup_rounds):
          sess.run(run_op)
        start = time.time()
        for _ in xrange(benchmark_rounds):
          sess.run(run_op)
    end = time.time()
    step_time = (end - start) / benchmark_rounds
    tag = '%s' % (cpu_count) if cpu_count is not None else '_all'
    print('benchmarkAdjustSaturationInYiq_299_299_3_cpu%s step_time: %.2f us' %
          (tag, step_time * 1e6))
    self.report_benchmark(
        name='benchmarkAdjustSaturationInYiq_299_299_3_cpu%s' % (tag),
        iters=benchmark_rounds,
        wall_time=step_time)

  def benchmark_adjust_saturation_in_yiq_cpu1(self):
    self._benchmark_adjust_saturation_in_yiq('/cpu:0', 1)

  def benchmark_adjust_saturation_in_yiq_cpu_all(self):
    self._benchmark_adjust_saturation_in_yiq('/cpu:0', None)

  def benchmark_adjust_saturation_in_yiq_gpu_all(self):
    self._benchmark_adjust_saturation_in_yiq(test.gpu_device_name(), None)


if __name__ == '__main__':
  googletest.main()
