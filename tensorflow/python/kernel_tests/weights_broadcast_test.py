# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for broadcast rules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.platform import test


def _test_values(shape):
  return np.reshape(np.cumsum(np.ones(shape), dtype=np.int32), newshape=shape)


class AssertBroadcastableTest(test.TestCase):

  def setUp(self):
    ops.reset_default_graph()

  def _test_valid(self, weights, values):
    static_op = weights_broadcast_ops.assert_broadcastable(
        weights=weights, values=values)
    weights_placeholder = array_ops.placeholder(dtypes_lib.float32)
    values_placeholder = array_ops.placeholder(dtypes_lib.float32)
    dynamic_op = weights_broadcast_ops.assert_broadcastable(
        weights=weights_placeholder, values=values_placeholder)
    with self.cached_session():
      static_op.run()
      dynamic_op.run(feed_dict={
          weights_placeholder: weights,
          values_placeholder: values,
      })

  @test_util.run_deprecated_v1
  def testScalar(self):
    self._test_valid(weights=5, values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def test1x1x1(self):
    self._test_valid(
        weights=np.asarray((5,)).reshape((1, 1, 1)),
        values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def test1x1xN(self):
    self._test_valid(
        weights=np.asarray((5, 7, 11, 3)).reshape((1, 1, 4)),
        values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def test1xNx1(self):
    self._test_valid(
        weights=np.asarray((5, 11)).reshape((1, 2, 1)),
        values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def test1xNxN(self):
    self._test_valid(
        weights=np.asarray((5, 7, 11, 3, 2, 13, 7, 5)).reshape((1, 2, 4)),
        values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def testNx1x1(self):
    self._test_valid(
        weights=np.asarray((5, 7, 11)).reshape((3, 1, 1)),
        values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def testNx1xN(self):
    self._test_valid(
        weights=np.asarray((
            5, 7, 11, 3, 2, 12, 7, 5, 2, 17, 11, 3)).reshape((3, 1, 4)),
        values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def testNxNxN(self):
    self._test_valid(
        weights=np.asarray((
            5, 7, 11, 3, 2, 12, 7, 5, 2, 17, 11, 3,
            2, 17, 11, 3, 5, 7, 11, 3, 2, 12, 7, 5)).reshape((3, 2, 4)),
        values=_test_values((3, 2, 4)))

  def _test_invalid(self, weights, values):
    error_msg = 'weights can not be broadcast to values'
    with self.assertRaisesRegex(ValueError, error_msg):
      weights_broadcast_ops.assert_broadcastable(weights=weights, values=values)
    weights_placeholder = array_ops.placeholder(dtypes_lib.float32)
    values_placeholder = array_ops.placeholder(dtypes_lib.float32)
    dynamic_op = weights_broadcast_ops.assert_broadcastable(
        weights=weights_placeholder, values=values_placeholder)
    with self.cached_session():
      with self.assertRaisesRegex(errors_impl.OpError, error_msg):
        dynamic_op.run(feed_dict={
            weights_placeholder: weights,
            values_placeholder: values,
        })

  @test_util.run_deprecated_v1
  def testInvalid1(self):
    self._test_invalid(weights=np.asarray((5,)), values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def testInvalid1x1(self):
    self._test_invalid(
        weights=np.asarray((5,)).reshape((1, 1)),
        values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def testInvalidPrefixMatch(self):
    self._test_invalid(
        weights=np.asarray((5, 7, 11, 3, 2, 12)).reshape((3, 2)),
        values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def testInvalidSuffixMatch(self):
    self._test_invalid(
        weights=np.asarray((5, 7, 11, 3, 2, 12, 7, 5)).reshape((2, 4)),
        values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def testInvalidOnesExtraDim(self):
    self._test_invalid(
        weights=np.asarray((5,)).reshape((1, 1, 1, 1)),
        values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def testInvalidPrefixMatchExtraDim(self):
    self._test_invalid(
        weights=np.asarray((
            5, 7, 11, 3, 2, 12, 7, 5, 2, 17, 11, 3,
            2, 17, 11, 3, 5, 7, 11, 3, 2, 12, 7, 5)).reshape((3, 2, 4, 1)),
        values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def testInvalidSuffixMatchExtraDim(self):
    self._test_invalid(
        weights=np.asarray((
            5, 7, 11, 3, 2, 12, 7, 5, 2, 17, 11, 3,
            2, 17, 11, 3, 5, 7, 11, 3, 2, 12, 7, 5)).reshape((1, 3, 2, 4)),
        values=_test_values((3, 2, 4)))


class BroadcastWeightsTest(test.TestCase):

  def setUp(self):
    ops.reset_default_graph()

  def _test_valid(self, weights, values, expected):
    static_op = weights_broadcast_ops.broadcast_weights(
        weights=weights, values=values)
    weights_placeholder = array_ops.placeholder(dtypes_lib.float32)
    values_placeholder = array_ops.placeholder(dtypes_lib.float32)
    dynamic_op = weights_broadcast_ops.broadcast_weights(
        weights=weights_placeholder, values=values_placeholder)
    with self.cached_session():
      self.assertAllEqual(expected, self.evaluate(static_op))
      self.assertAllEqual(expected, dynamic_op.eval(feed_dict={
          weights_placeholder: weights,
          values_placeholder: values,
      }))

  @test_util.run_deprecated_v1
  def testScalar(self):
    self._test_valid(
        weights=5,
        values=_test_values((3, 2, 4)),
        expected=5 * np.ones((3, 2, 4)))

  @test_util.run_deprecated_v1
  def test1x1x1(self):
    self._test_valid(
        weights=np.asarray((5,)).reshape((1, 1, 1)),
        values=_test_values((3, 2, 4)),
        expected=5 * np.ones((3, 2, 4)))

  @test_util.run_deprecated_v1
  def test1x1xN(self):
    weights = np.asarray((5, 7, 11, 3)).reshape((1, 1, 4))
    self._test_valid(
        weights=weights,
        values=_test_values((3, 2, 4)),
        expected=np.tile(weights, reps=(3, 2, 1)))

  @test_util.run_deprecated_v1
  def test1xNx1(self):
    weights = np.asarray((5, 11)).reshape((1, 2, 1))
    self._test_valid(
        weights=weights,
        values=_test_values((3, 2, 4)),
        expected=np.tile(weights, reps=(3, 1, 4)))

  @test_util.run_deprecated_v1
  def test1xNxN(self):
    weights = np.asarray((5, 7, 11, 3, 2, 13, 7, 5)).reshape((1, 2, 4))
    self._test_valid(
        weights=weights,
        values=_test_values((3, 2, 4)),
        expected=np.tile(weights, reps=(3, 1, 1)))

  @test_util.run_deprecated_v1
  def testNx1x1(self):
    weights = np.asarray((5, 7, 11)).reshape((3, 1, 1))
    self._test_valid(
        weights=weights,
        values=_test_values((3, 2, 4)),
        expected=np.tile(weights, reps=(1, 2, 4)))

  @test_util.run_deprecated_v1
  def testNx1xN(self):
    weights = np.asarray((
        5, 7, 11, 3, 2, 12, 7, 5, 2, 17, 11, 3)).reshape((3, 1, 4))
    self._test_valid(
        weights=weights,
        values=_test_values((3, 2, 4)),
        expected=np.tile(weights, reps=(1, 2, 1)))

  @test_util.run_deprecated_v1
  def testNxNxN(self):
    weights = np.asarray((
        5, 7, 11, 3, 2, 12, 7, 5, 2, 17, 11, 3,
        2, 17, 11, 3, 5, 7, 11, 3, 2, 12, 7, 5)).reshape((3, 2, 4))
    self._test_valid(
        weights=weights, values=_test_values((3, 2, 4)), expected=weights)

  def _test_invalid(self, weights, values):
    error_msg = 'weights can not be broadcast to values'
    with self.assertRaisesRegex(ValueError, error_msg):
      weights_broadcast_ops.broadcast_weights(weights=weights, values=values)
    weights_placeholder = array_ops.placeholder(dtypes_lib.float32)
    values_placeholder = array_ops.placeholder(dtypes_lib.float32)
    dynamic_op = weights_broadcast_ops.broadcast_weights(
        weights=weights_placeholder, values=values_placeholder)
    with self.cached_session():
      with self.assertRaisesRegex(errors_impl.OpError, error_msg):
        dynamic_op.eval(feed_dict={
            weights_placeholder: weights,
            values_placeholder: values,
        })

  @test_util.run_deprecated_v1
  def testInvalid1(self):
    self._test_invalid(weights=np.asarray((5,)), values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def testInvalid1x1(self):
    self._test_invalid(
        weights=np.asarray((5,)).reshape((1, 1)),
        values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def testInvalidPrefixMatch(self):
    self._test_invalid(
        weights=np.asarray((5, 7, 11, 3, 2, 12)).reshape((3, 2)),
        values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def testInvalidSuffixMatch(self):
    self._test_invalid(
        weights=np.asarray((5, 7, 11, 3, 2, 12, 7, 5)).reshape((2, 4)),
        values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def testInvalidOnesExtraDim(self):
    self._test_invalid(
        weights=np.asarray((5,)).reshape((1, 1, 1, 1)),
        values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def testInvalidPrefixMatchExtraDim(self):
    self._test_invalid(
        weights=np.asarray((
            5, 7, 11, 3, 2, 12, 7, 5, 2, 17, 11, 3,
            2, 17, 11, 3, 5, 7, 11, 3, 2, 12, 7, 5)).reshape((3, 2, 4, 1)),
        values=_test_values((3, 2, 4)))

  @test_util.run_deprecated_v1
  def testInvalidSuffixMatchExtraDim(self):
    self._test_invalid(
        weights=np.asarray((
            5, 7, 11, 3, 2, 12, 7, 5, 2, 17, 11, 3,
            2, 17, 11, 3, 5, 7, 11, 3, 2, 12, 7, 5)).reshape((1, 3, 2, 4)),
        values=_test_values((3, 2, 4)))


if __name__ == '__main__':
  test.main()
