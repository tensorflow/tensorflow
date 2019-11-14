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
"""Tests for tensorflow.ops.reshape_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.platform import test


class ReshapeTest(test.TestCase):

  def _testReshape(self, x, y, use_gpu=False):
    with self.cached_session(use_gpu=use_gpu):
      np_ans = x.reshape(y)
      tf_ans = array_ops.reshape(x, y)
      out = self.evaluate(tf_ans)
      self.assertEqual(tf_ans.get_shape(), out.shape)
      self.assertShapeEqual(np_ans, tf_ans)

      # Repeat with an int64 shape tensor.
      y64 = constant_op.constant(y, dtype=dtypes.int64)
      tf_ans = array_ops.reshape(x, y64)
      out = self.evaluate(tf_ans)
      self.assertEqual(tf_ans.get_shape(), out.shape)
      self.assertShapeEqual(np_ans, tf_ans)

  def _testZeroDimReshape(self, x, shape, expected, use_gpu=False):
    with self.cached_session(use_gpu=use_gpu):
      y = array_ops.reshape(x, shape)
      out = self.evaluate(y)
      self.assertEqual(expected, out.shape)

      # Repeat with an int64 shape tensor.
      shape64 = constant_op.constant(shape, dtype=dtypes.int64)
      y = array_ops.reshape(x, shape64)
      out = self.evaluate(y)
      self.assertEqual(expected, out.shape)

  def _testBothReshape(self, x, y):
    self._testReshape(x, y, False)
    self._testReshape(x, y, True)

  def testBoolBasic(self):
    x = np.arange(1., 7.).reshape([1, 6]) > 3
    self._testBothReshape(x, [2, 3])

  def testFloatBasic(self):
    x = np.arange(1., 7.).reshape([1, 6]).astype(np.float32)
    self._testBothReshape(x, [2, 3])

  def testDoubleBasic(self):
    x = np.arange(1., 7.).reshape([1, 6]).astype(np.float64)
    self._testBothReshape(x, [2, 3])

  def testInt32Basic(self):
    x = np.arange(1., 7.).reshape([1, 6]).astype(np.int32)
    self._testBothReshape(x, [2, 3])

  def testComplex64Basic(self):
    x = np.arange(1., 7.).reshape([1, 6]).astype(np.complex64)
    self._testBothReshape(x, [2, 3])

  def testComplex128Basic(self):
    x = np.arange(1., 7.).reshape([1, 6]).astype(np.complex128)
    self._testBothReshape(x, [2, 3])

  def testFloatReshapeThreeDimensions(self):
    x = np.arange(1., 28.).reshape([1, 27]).astype(np.float32)
    self._testBothReshape(x, [3, 3, 3])

  def testFloatUnspecifiedDimOnly(self):
    x = np.arange(1., 7.).reshape([6]).astype(np.float32)
    self._testBothReshape(x, [-1])

  def testFloatUnspecifiedDimBegin(self):
    x = np.arange(1., 7.).reshape([6]).astype(np.float32)
    self._testBothReshape(x, [-1, 2])

  def testFloatUnspecifiedDimEnd(self):
    x = np.arange(1., 7.).reshape([6]).astype(np.float32)
    self._testBothReshape(x, [3, -1])

  def testZeroDimBasic(self):
    x = np.zeros([0, 6]).astype(np.float32)
    self._testBothReshape(x, [0, 2, 3])

  def testZeroDimReshapeR1(self):
    x = np.zeros([0, 6]).astype(np.float32)
    self._testBothReshape(x, [-1])

  def testZeroDimReshapeR3(self):
    x = np.zeros([0, 6]).astype(np.float32)
    self._testBothReshape(x, [-1, 2, 3])

  # TODO(vrv): Add tests for failure conditions once python test_util
  # reports errors.

  @test_util.run_deprecated_v1
  def testFloatReshapeGradThreeDimensions(self):
    x = np.arange(1., 25.).reshape([2, 3, 4]).astype(np.float32)
    s = list(np.shape(x))
    with self.cached_session():
      input_tensor = constant_op.constant(x)
      reshape_out = array_ops.reshape(input_tensor, [1, 8, 3])
      err = gradient_checker.compute_gradient_error(
          input_tensor, s, reshape_out, s, x_init_value=x)
    print("Reshape gradient error = " % err)
    self.assertLess(err, 1e-3)

  def testFloatEmpty(self):
    x = np.empty((0, 0, 0, 0), dtype=np.float32)
    self._testBothReshape(x, [1, 2, 3, 0])
    self._testBothReshape(x, [1, 0, 0, 4])
    self._testBothReshape(x, [0, 0, 0, 0])
    self._testBothReshape(x, [1, 2, 0])
    self._testBothReshape(x, [0, 0, 0])
    self._testBothReshape(x, [1, -1, 5])

  def testZeroDimWithUnspecifiedDim(self):
    for use_gpu in (True, False):
      self._testZeroDimReshape(x=np.zeros([0, 6]).astype(np.float32),
                               shape=[0, -1, 3],
                               expected=(0, 2, 3),
                               use_gpu=use_gpu)

  @test_util.run_deprecated_v1
  def testErrors(self):
    y = constant_op.constant(0.0, shape=[23, 29, 31])
    with self.assertRaisesRegexp(ValueError, "must be evenly divisible by 17"):
      array_ops.reshape(y, [17, -1])

    z = constant_op.constant(0.0, shape=[32, 128])
    with self.assertRaisesRegexp(ValueError,
                                 "Cannot reshape a tensor with 4096 elements"):
      array_ops.reshape(z, [4095])

  @test_util.run_deprecated_v1
  def testPartialShapes(self):
    x = array_ops.placeholder(dtypes.float32)

    # Unknown input shape, partial new shape.
    y = array_ops.reshape(x, [1, 1, -1, 1])
    self.assertEqual([1, 1, None, 1], y.get_shape().as_list())

    # Unknown input shape, unknown new shape.
    y = array_ops.reshape(x, array_ops.placeholder(dtypes.int32))
    self.assertEqual(None, y.get_shape().ndims)

    # Unknown input shape, known rank for new shape.
    y = array_ops.reshape(x, array_ops.placeholder(dtypes.int32, shape=(3,)))
    self.assertEqual([None, None, None], y.get_shape().as_list())

    # Unknown input shape, partial new shape using `tf.stack()`.
    y = array_ops.reshape(x, [array_ops.placeholder(dtypes.int32), 37])
    self.assertEqual([None, 37], y.get_shape().as_list())

    # Unknown input shape, partial new shape using `tf.concat()`.
    y = array_ops.reshape(
        x,
        array_ops.concat(
            [array_ops.placeholder(
                dtypes.int32, shape=(2,)), [37, 42]], 0))
    self.assertEqual([None, None, 37, 42], y.get_shape().as_list())

    # Unknown input shape, partial new shape using `tf.shape()`.
    y = array_ops.reshape(
        x,
        array_ops.shape(
            array_ops.placeholder(
                dtypes.float32, shape=[None, 37, None])))
    self.assertEqual([None, 37, None], y.get_shape().as_list())


if __name__ == "__main__":
  test.main()
