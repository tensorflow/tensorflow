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
"""Tests for local response normalization."""

import copy

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class LRNOpTest(test.TestCase):

  def _LRN(self, input_image, lrn_depth_radius=5, bias=1.0, alpha=1.0,
           beta=0.5):
    """Compute expected result."""
    output = copy.deepcopy(input_image)
    batch_size = input_image.shape[0]
    rows = input_image.shape[1]
    cols = input_image.shape[2]
    depth = input_image.shape[3]
    for b in range(batch_size):
      for r in range(rows):
        for c in range(cols):
          for d in range(depth):
            begin = max(0, d - lrn_depth_radius)
            end = min(depth, d + lrn_depth_radius + 1)
            patch = input_image[b, r, c, begin:end]
            output[b, r, c, d] /= (
                np.power(bias + alpha * np.sum(patch * patch), beta))
    return output

  def _RunAndVerify(self, dtype):
    with self.cached_session():
      # random shape
      shape = np.random.randint(1, 16, size=4)
      # Make depth at least 2 to make it meaningful
      shape[3] += 1
      p = array_ops.placeholder(dtype, shape=shape)
      # random depth_radius, bias, alpha, beta. cuDNN requires depth_radius to
      # be in [1, 7].
      lrn_depth_radius = np.random.randint(1, min(8, shape[3]))

      bias = 1.0 + np.random.rand()
      alpha = 2.0 * np.random.rand()
      # cuDNN requires beta >= 0.01.
      beta = 0.01 + 2.0 * np.random.rand()
      lrn_t = nn.local_response_normalization(
          p,
          name="lrn",
          depth_radius=lrn_depth_radius,
          bias=bias,
          alpha=alpha,
          beta=beta)
      params = {p: np.random.rand(*shape).astype("f")}
      result = lrn_t.eval(feed_dict=params)
    expected = self._LRN(
        params[p],
        lrn_depth_radius=lrn_depth_radius,
        bias=bias,
        alpha=alpha,
        beta=beta)
    err = np.amax(np.abs(result - expected))
    print("LRN error for bias ", bias, "alpha ", alpha, " beta ", beta, " is ",
          err)
    if dtype == dtypes.float32:
      self.assertTrue(err < 1e-4)
    else:
      self.assertTrue(err < 1e-2)
    self.assertShapeEqual(expected, lrn_t)

  @test_util.run_deprecated_v1
  def testCompute(self):
    for _ in range(2):
      self._RunAndVerify(dtypes.float32)
      # Enable when LRN supports tf.float16 on GPU.
      if not test.is_gpu_available():
        self._RunAndVerify(dtypes.float16)

  @test_util.run_deprecated_v1
  def testGradientsZeroInput(self):
    with self.session():
      shape = [4, 4, 4, 4]
      p = array_ops.placeholder(dtypes.float32, shape=shape)
      inp_array = np.zeros(shape).astype("f")
      lrn_op = nn.local_response_normalization(p, 2, 1.0, 0.0, 1.0, name="lrn")
      grad = gradients_impl.gradients([lrn_op], [p])[0]
      params = {p: inp_array}
      r = grad.eval(feed_dict=params)
    expected = np.ones(shape).astype("f")
    self.assertAllClose(r, expected)
    self.assertShapeEqual(expected, grad)

  def _RunAndVerifyGradients(self, dtype):
    with self.cached_session():
      # random shape
      shape = np.random.randint(1, 5, size=4)
      # Make depth at least 2 to make it meaningful
      shape[3] += 1
      # random depth_radius, bias, alpha, beta. cuDNN requires depth_radius to
      # be in [1, 7].
      lrn_depth_radius = np.random.randint(1, min(8, shape[3]))
      bias = 1.0 + np.random.rand()
      alpha = 1.0 * np.random.rand()
      # cuDNN requires beta >= 0.01.
      beta = 0.01 + 1.0 * np.random.rand()
      if dtype == dtypes.float32:
        inp_array = np.random.rand(*shape).astype(np.float32)
      else:
        inp_array = np.random.rand(*shape).astype(np.float16)

      inp = constant_op.constant(
          list(inp_array.ravel(order="C")), shape=shape, dtype=dtype)
      lrn_op = nn.local_response_normalization(
          inp,
          name="lrn",
          depth_radius=lrn_depth_radius,
          bias=bias,
          alpha=alpha,
          beta=beta)
      err = gradient_checker.compute_gradient_error(inp, shape, lrn_op, shape)
    print("LRN Gradient error for bias ", bias, "alpha ", alpha, " beta ", beta,
          " is ", err)
    if dtype == dtypes.float32:
      self.assertLess(err, 1e-4)
    else:
      self.assertLess(err, 1.0)

  @test_util.run_deprecated_v1
  def testGradients(self):
    for _ in range(2):
      self._RunAndVerifyGradients(dtypes.float32)
      # Enable when LRN supports tf.float16 on GPU.
      if not test.is_gpu_available():
        self._RunAndVerifyGradients(dtypes.float16)


if __name__ == "__main__":
  test.main()
