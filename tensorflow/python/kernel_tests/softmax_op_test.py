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
"""Tests for SoftmaxOp and LogSoftmaxOp."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np


from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging


class SoftmaxTest(test.TestCase):

  def _npSoftmax(self, features, dim=-1, log=False):
    if dim is -1:
      dim = len(features.shape) - 1
    one_only_on_dim = list(features.shape)
    one_only_on_dim[dim] = 1
    is_fp16 = features.dtype == np.float16
    if is_fp16:
      # Do the compute in fp32 and cast the input back to fp32.
      features = features.astype(np.float32)
    e = np.exp(features - np.reshape(
        np.amax(
            features, axis=dim), one_only_on_dim))
    softmax = e / np.reshape(np.sum(e, axis=dim), one_only_on_dim)
    if log:
      res = np.log(softmax)
    else:
      res = softmax
    if is_fp16:
      res = res.astype(np.float16)
    return res

  def _testSoftmax(self, np_features, dim=-1, log=False, use_gpu=False):
    # A previous version of the code checked the op name rather than the op type
    # to distinguish between log and non-log.  Use an arbitrary name to catch
    # this bug in future.
    name = "arbitrary"
    np_softmax = self._npSoftmax(np_features, dim=dim, log=log)
    with self.cached_session(use_gpu=use_gpu):
      if log:
        tf_softmax = nn_ops.log_softmax(np_features, axis=dim, name=name)
      else:
        tf_softmax = nn_ops.softmax(np_features, axis=dim, name=name)
      out = self.evaluate(tf_softmax)
    self.assertAllCloseAccordingToType(np_softmax, out)
    self.assertShapeEqual(np_softmax, tf_softmax)
    if not log:
      # Bonus check: the softmaxes should add to one in dimension dim.
      sum_along_dim = np.sum(out, axis=dim)
      self.assertAllCloseAccordingToType(
          np.ones(sum_along_dim.shape), sum_along_dim)

  def _testAll(self, features):
    self._testSoftmax(features, use_gpu=True)
    self._testSoftmax(features, log=True, use_gpu=True)
    self._testOverflow(use_gpu=True)

  def testNpSoftmax(self):
    features = [[1., 1., 1., 1.], [1., 2., 3., 4.]]
    # Batch 0: All exps are 1.  The expected result is
    # Softmaxes = [0.25, 0.25, 0.25, 0.25]
    # LogSoftmaxes = [-1.386294, -1.386294, -1.386294, -1.386294]
    #
    # Batch 1:
    # exps = [1., 2.718, 7.389, 20.085]
    # sum = 31.192
    # Softmaxes = exps / sum = [0.0320586, 0.08714432, 0.23688282, 0.64391426]
    # LogSoftmaxes = [-3.44019 , -2.44019 , -1.44019 , -0.44019]
    np_sm = self._npSoftmax(np.array(features))
    self.assertAllClose(
        np.array([[0.25, 0.25, 0.25, 0.25],
                  [0.0320586, 0.08714432, 0.23688282, 0.64391426]]),
        np_sm,
        rtol=1.e-5,
        atol=1.e-5)
    np_lsm = self._npSoftmax(np.array(features), log=True)
    self.assertAllClose(
        np.array([[-1.386294, -1.386294, -1.386294, -1.386294],
                  [-3.4401897, -2.4401897, -1.4401897, -0.4401897]]),
        np_lsm,
        rtol=1.e-5,
        atol=1.e-5)

  def _testOverflow(self, use_gpu=False):
    if use_gpu:
      type = np.float32  # pylint: disable=redefined-builtin
    else:
      type = np.float64  # pylint: disable=redefined-builtin
    max = np.finfo(type).max  # pylint: disable=redefined-builtin
    features = np.array([[1., 1., 1., 1.], [max, 1., 2., 3.]]).astype(type)
    with self.cached_session(use_gpu=use_gpu):
      tf_log_softmax = nn_ops.log_softmax(features)
      out = self.evaluate(tf_log_softmax)
    self.assertAllClose(
        np.array([[-1.386294, -1.386294, -1.386294, -1.386294],
                  [0, -max, -max, -max]]),
        out,
        rtol=1.e-5,
        atol=1.e-5)

  def testFloat(self):
    self._testAll(
        np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float32))

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testFloatGPU(self):
    if test.is_gpu_available(cuda_only=True):
      rows = [2**x + np.random.randint(0, 16) for x in range(1, 4)]
      cols = [2**x + np.random.randint(0, 16) for x in range(1, 4)]
      for row, col in zip(rows, cols):
        logging.info("Testing softmax float dtype in shape [%d, %d]", row, col)
        data = np.random.rand(row, col)
        self._testAll(data.astype(np.float32))

  def testHalf(self):
    self._testAll(
        np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float16))

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testHalfGPU(self):
    if test.is_gpu_available(cuda_only=True):
      rows = [2**x + np.random.randint(0, 16) for x in range(1, 4)]
      cols = [2**x + np.random.randint(0, 16) for x in range(1, 4)]
      for row, col in zip(rows, cols):
        logging.info("Testing softmax half dtype in shape [%d, %d]", row, col)
        data = np.random.rand(row, col)
        self._testAll(data.astype(np.float16))

  def testDouble(self):
    self._testSoftmax(
        np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float64))
    self._testOverflow()

  def test1DTensorAsInput(self):
    self._testSoftmax(
        np.array([3., 2., 3., 9.]).astype(np.float64), use_gpu=False)
    self._testOverflow(use_gpu=False)

  def test1DTensorAsInputNoReshape(self):
    self._testSoftmax(
        np.array([3., 2., 3., 9.]).astype(np.float64), use_gpu=False)
    self._testOverflow(use_gpu=False)

  def test3DTensorAsInput(self):
    self._testSoftmax(
        np.array([[[1., 1., 1., 1.], [1., 2., 3., 4.]],
                  [[2., 3., 4., 5.], [6., 7., 8., 9.]],
                  [[5., 4., 3., 2.], [1., 2., 3., 4.]]]).astype(np.float32),
        use_gpu=False)
    self._testOverflow(use_gpu=False)

  def test3DTensorAsInputNoReshape(self):
    self._testSoftmax(
        np.array([[[1., 1., 1., 1.], [1., 2., 3., 4.]],
                  [[2., 3., 4., 5.], [6., 7., 8., 9.]],
                  [[5., 4., 3., 2.], [1., 2., 3., 4.]]]).astype(np.float32),
        use_gpu=False)
    self._testOverflow(use_gpu=False)

  def testAlongFirstDimension(self):
    self._testSoftmax(
        np.array([[[1., 1., 1., 1.], [1., 2., 3., 4.]],
                  [[2., 3., 4., 5.], [6., 7., 8., 9.]],
                  [[5., 4., 3., 2.], [1., 2., 3., 4.]]]).astype(np.float32),
        dim=0,
        use_gpu=False)
    self._testOverflow(use_gpu=False)

  def testAlongSecondDimension(self):
    self._testSoftmax(
        np.array([[[1., 1., 1., 1.], [1., 2., 3., 4.]],
                  [[2., 3., 4., 5.], [6., 7., 8., 9.]],
                  [[5., 4., 3., 2.], [1., 2., 3., 4.]]]).astype(np.float32),
        dim=1,
        use_gpu=False)
    self._testOverflow(use_gpu=False)

  def testShapeInference(self):
    op = nn_ops.softmax([[[1., 1., 1., 1.], [1., 2., 3., 4.]],
                         [[2., 3., 4., 5.], [6., 7., 8., 9.]],
                         [[5., 4., 3., 2.], [1., 2., 3., 4.]]])
    self.assertEqual([3, 2, 4], op.get_shape())

  @test_util.run_deprecated_v1
  def testEmptyInput(self):
    with self.cached_session():
      x = array_ops.placeholder(dtypes.float32, shape=[0, 3])
      self.assertEqual(0, array_ops.size(x).eval())
      # reshape would raise if logits is empty
      with self.assertRaises(errors_impl.InvalidArgumentError):
        nn_ops.softmax(x, axis=0).eval()

  def testDimTooLarge(self):
    with self.cached_session():
      # Use placeholder to make sure we get runtime error instead of shape
      # inference error.
      dim = array_ops.placeholder_with_default(100, shape=[])
      with self.assertRaises(errors_impl.InvalidArgumentError):
        nn_ops.softmax([1., 2., 3., 4.], axis=dim).eval()

  def testInvalidAxis(self):
    # Test case for GitHub issue 22793.
    with self.cached_session():
      ones = array_ops.ones(shape=[2, 3])
      with self.assertRaises(errors_impl.InvalidArgumentError):
        nn_ops.softmax(ones, axis=2).eval()

  @test_util.run_deprecated_v1
  def testLargeDims(self):
    # Make sure that we properly handle large inputs. See
    # https://github.com/tensorflow/tensorflow/issues/4425 for details
    for dims in [129, 256]:
      ones = np.random.rand(dims, dims).astype(np.float32)
      np_softmax = self._npSoftmax(ones)

      for use_gpu in [True, False]:
        with self.cached_session(use_gpu=use_gpu) as sess:
          x = array_ops.placeholder(dtypes.float32)
          y = nn_ops.softmax(x)
          tf_softmax = sess.run(y, feed_dict={x: ones})
        self.assertAllClose(tf_softmax, np_softmax)


if __name__ == "__main__":
  test.main()
