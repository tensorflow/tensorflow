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
"""Tests for utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.contrib.linalg.python.ops import linear_operator_diag
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class ShapesFromLocAndScaleTest(test.TestCase):

  def test_static_loc_static_scale_non_matching_event_size_raises(self):
    loc = constant_op.constant(np.zeros((2, 4)))
    scale = linear_operator_diag.LinearOperatorDiag(np.ones((5, 1, 3)))
    with self.assertRaisesRegexp(ValueError, "could not be broadcast"):
      distribution_util.shapes_from_loc_and_scale(loc, scale)

  def test_static_loc_static_scale(self):
    loc = constant_op.constant(np.zeros((2, 3)))
    scale = linear_operator_diag.LinearOperatorDiag(np.ones((5, 1, 3)))
    batch_shape, event_shape = distribution_util.shapes_from_loc_and_scale(
        loc, scale)

    self.assertEqual(tensor_shape.TensorShape([5, 2]), batch_shape)
    self.assertEqual(tensor_shape.TensorShape([3]), event_shape)

  def test_static_loc_dynamic_scale(self):
    loc = constant_op.constant(np.zeros((2, 3)))
    diag = array_ops.placeholder(dtypes.float64)
    scale = linear_operator_diag.LinearOperatorDiag(diag)
    with self.test_session() as sess:
      batch_shape, event_shape = sess.run(
          distribution_util.shapes_from_loc_and_scale(loc, scale),
          feed_dict={diag: np.ones((5, 1, 3))})
      self.assertAllEqual([5, 2], batch_shape)
      self.assertAllEqual([3], event_shape)

  def test_dynamic_loc_static_scale(self):
    loc = array_ops.placeholder(dtypes.float64)
    diag = constant_op.constant(np.ones((5, 2, 3)))
    scale = linear_operator_diag.LinearOperatorDiag(diag)
    with self.test_session():
      batch_shape, event_shape = distribution_util.shapes_from_loc_and_scale(
          loc, scale)
      # batch_shape depends on both args, and so is dynamic.  Since loc did not
      # have static shape, we inferred event shape entirely from scale, and this
      # is available statically.
      self.assertAllEqual(
          [5, 2], batch_shape.eval(feed_dict={loc: np.zeros((2, 3))}))
      self.assertAllEqual([3], event_shape)

  def test_dynamic_loc_dynamic_scale(self):
    loc = array_ops.placeholder(dtypes.float64)
    diag = array_ops.placeholder(dtypes.float64)
    scale = linear_operator_diag.LinearOperatorDiag(diag)
    with self.test_session() as sess:
      batch_shape, event_shape = sess.run(
          distribution_util.shapes_from_loc_and_scale(loc, scale),
          feed_dict={diag: np.ones((5, 2, 3)), loc: np.zeros((2, 3))})
      self.assertAllEqual([5, 2], batch_shape)
      self.assertAllEqual([3], event_shape)

  def test_none_loc_static_scale(self):
    loc = None
    scale = linear_operator_diag.LinearOperatorDiag(np.ones((5, 1, 3)))
    batch_shape, event_shape = distribution_util.shapes_from_loc_and_scale(
        loc, scale)

    self.assertEqual(tensor_shape.TensorShape([5, 1]), batch_shape)
    self.assertEqual(tensor_shape.TensorShape([3]), event_shape)

  def test_none_loc_dynamic_scale(self):
    loc = None
    diag = array_ops.placeholder(dtypes.float64)
    scale = linear_operator_diag.LinearOperatorDiag(diag)
    with self.test_session() as sess:
      batch_shape, event_shape = sess.run(
          distribution_util.shapes_from_loc_and_scale(loc, scale),
          feed_dict={diag: np.ones((5, 1, 3))})
      self.assertAllEqual([5, 1], batch_shape)
      self.assertAllEqual([3], event_shape)


class TridiagTest(test.TestCase):

  def testWorksCorrectlyNoBatches(self):
    with self.test_session():
      self.assertAllEqual(
          [[4., 8., 0., 0.],
           [1., 5., 9., 0.],
           [0., 2., 6., 10.],
           [0., 0., 3, 7.]],
          distribution_util.tridiag(
              [1., 2., 3.],
              [4., 5., 6., 7.],
              [8., 9., 10.]).eval())

  def testWorksCorrectlyBatches(self):
    with self.test_session():
      self.assertAllClose(
          [[[4., 8., 0., 0.],
            [1., 5., 9., 0.],
            [0., 2., 6., 10.],
            [0., 0., 3, 7.]],
           [[0.7, 0.1, 0.0, 0.0],
            [0.8, 0.6, 0.2, 0.0],
            [0.0, 0.9, 0.5, 0.3],
            [0.0, 0.0, 1.0, 0.4]]],
          distribution_util.tridiag(
              [[1., 2., 3.],
               [0.8, 0.9, 1.]],
              [[4., 5., 6., 7.],
               [0.7, 0.6, 0.5, 0.4]],
              [[8., 9., 10.],
               [0.1, 0.2, 0.3]]).eval(),
          rtol=1e-5, atol=0.)

  def testHandlesNone(self):
    with self.test_session():
      self.assertAllClose(
          [[[4., 0., 0., 0.],
            [0., 5., 0., 0.],
            [0., 0., 6., 0.],
            [0., 0., 0, 7.]],
           [[0.7, 0.0, 0.0, 0.0],
            [0.0, 0.6, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.4]]],
          distribution_util.tridiag(
              diag=[[4., 5., 6., 7.],
                    [0.7, 0.6, 0.5, 0.4]]).eval(),
          rtol=1e-5, atol=0.)


if __name__ == "__main__":
  test.main()
