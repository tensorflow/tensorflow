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
"""Tests for tensorflow.ops.data_flow_ops.dynamic_stitch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gradient_checker
import tensorflow.python.ops.data_flow_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class DynamicStitchTest(test.TestCase):

  def testScalar(self):
    with self.test_session():
      indices = [constant_op.constant(0), constant_op.constant(1)]
      data = [constant_op.constant(40.), constant_op.constant(60.)]
      for step in -1, 1:
        stitched_t = data_flow_ops.dynamic_stitch(indices[::step], data)
        stitched_val = stitched_t.eval()
        self.assertAllEqual([40, 60][::step], stitched_val)
        # Dimension 0 is determined by the max index in indices, so we
        # can only infer that the output is a vector of some unknown
        # length.
        self.assertEqual([None], stitched_t.get_shape().as_list())

        # Test gradients
        self.assertLess(
            gradient_checker.compute_gradient_error(
                data, [x.get_shape().as_list() for x in data],
                stitched_t, array_ops.shape(stitched_t).eval()),
            1e-4)

  def testSimpleOneDimensional(self):
    with self.test_session():
      indices = [
          constant_op.constant([0, 4, 7]), constant_op.constant([1, 6, 2, 3, 5])
      ]
      data = [
          constant_op.constant([0, 40, 70], dtypes.float32),
          constant_op.constant([10, 60, 20, 30, 50], dtypes.float32)
      ]
      stitched_t = data_flow_ops.dynamic_stitch(indices, data)
      stitched_val = stitched_t.eval()
      self.assertAllEqual([0, 10, 20, 30, 40, 50, 60, 70], stitched_val)
      # Dimension 0 is determined by the max index in indices, so we
      # can only infer that the output is a vector of some unknown
      # length.
      self.assertEqual([None], stitched_t.get_shape().as_list())

      # Test gradients
      self.assertLess(
          gradient_checker.compute_gradient_error(
              data, [x.get_shape().as_list() for x in data],
              stitched_t, array_ops.shape(stitched_t).eval()),
          1e-4)

  def testOneListOneDimensional(self):
    with self.test_session():
      indices = [constant_op.constant([1, 6, 2, 3, 5, 0, 4, 7])]
      data = [constant_op.constant([10, 60, 20, 30, 50, 0, 40, 70],
                                   dtypes.float32)]
      stitched_t = data_flow_ops.dynamic_stitch(indices, data)
      stitched_val = stitched_t.eval()
      self.assertAllEqual([0, 10, 20, 30, 40, 50, 60, 70], stitched_val)
      # Dimension 0 is determined by the max index in indices, so we
      # can only infer that the output is a vector of some unknown
      # length.
      self.assertEqual([None], stitched_t.get_shape().as_list())

      # Test gradients
      self.assertLess(
          gradient_checker.compute_gradient_error(
              data, [x.get_shape().as_list() for x in data],
              stitched_t, array_ops.shape(stitched_t).eval()),
          1e-4)

  def testSimpleTwoDimensional(self):
    with self.test_session():
      indices = [
          constant_op.constant([0, 4, 7]), constant_op.constant([1, 6]),
          constant_op.constant([2, 3, 5])
      ]
      data = [
          constant_op.constant([[0, 1], [40, 41], [70, 71]], dtypes.float32),
          constant_op.constant([[10, 11], [60, 61]], dtypes.float32),
          constant_op.constant([[20, 21], [30, 31], [50, 51]], dtypes.float32)
      ]
      stitched_t = data_flow_ops.dynamic_stitch(indices, data)
      stitched_val = stitched_t.eval()
      self.assertAllEqual([[0, 1], [10, 11], [20, 21], [30, 31], [40, 41],
                           [50, 51], [60, 61], [70, 71]], stitched_val)
      # Dimension 0 is determined by the max index in indices, so we
      # can only infer that the output is a matrix with 2 columns and
      # some unknown number of rows.
      self.assertEqual([None, 2], stitched_t.get_shape().as_list())

      # Test gradients
      self.assertLess(
          gradient_checker.compute_gradient_error(
              data, [x.get_shape().as_list() for x in data],
              stitched_t, array_ops.shape(stitched_t).eval()),
          1e-4)

  def testHigherRank(self):
    with self.test_session():
      indices = [
          constant_op.constant(6), constant_op.constant([4, 1]),
          constant_op.constant([[5, 2], [0, 3]])
      ]
      data = [
          constant_op.constant([61, 62], dtypes.float32),
          constant_op.constant([[41, 42], [11, 12]], dtypes.float32),
          constant_op.constant([[[51, 52], [21, 22]], [[1, 2], [31, 32]]],
                               dtypes.float32)
      ]
      stitched_t = data_flow_ops.dynamic_stitch(indices, data)
      stitched_val = stitched_t.eval()
      correct = 10 * np.arange(7)[:, None] + [1, 2]
      self.assertAllEqual(correct, stitched_val)
      self.assertEqual([None, 2], stitched_t.get_shape().as_list())
      # Test gradients
      self.assertLess(
          gradient_checker.compute_gradient_error(
              data, [x.get_shape().as_list() for x in data],
              stitched_t, array_ops.shape(stitched_t).eval()),
          1e-4)

  def testDuplicates(self):
    with self.test_session():
      indices = [
        constant_op.constant([0, 1]), constant_op.constant([1, 2, 3]),
        constant_op.constant([3, 4])
      ]
      data = [
        constant_op.constant([1, 2], dtypes.float32),
          constant_op.constant([12, 13, 14], dtypes.float32),
        constant_op.constant([24, 25], dtypes.float32)
      ]
      stitched_t = data_flow_ops.dynamic_stitch(indices, data)
      stitched_val = stitched_t.eval()
      correct = [1, 12, 13, 24, 25]
      self.assertAllEqual(correct, stitched_val)
      self.assertEqual([None], stitched_t.get_shape().as_list())

      # Test gradients
      self.assertLess(
        gradient_checker.compute_gradient_error(
          data, [x.get_shape().as_list() for x in data],
          stitched_t, array_ops.shape(stitched_t).eval()),
        1e-4)

  def testErrorIndicesMultiDimensional(self):
    indices = [
        constant_op.constant([0, 4, 7]), constant_op.constant([[1, 6, 2, 3, 5]])
    ]
    data = [
        constant_op.constant([[0, 40, 70]]),
        constant_op.constant([10, 60, 20, 30, 50])
    ]
    with self.assertRaises(ValueError):
      data_flow_ops.dynamic_stitch(indices, data)

  def testErrorDataNumDimsMismatch(self):
    indices = [
        constant_op.constant([0, 4, 7]), constant_op.constant([1, 6, 2, 3, 5])
    ]
    data = [
        constant_op.constant([0, 40, 70]),
        constant_op.constant([[10, 60, 20, 30, 50]])
    ]
    with self.assertRaises(ValueError):
      data_flow_ops.dynamic_stitch(indices, data)

  def testErrorDataDimSizeMismatch(self):
    indices = [
        constant_op.constant([0, 4, 5]), constant_op.constant([1, 6, 2, 3])
    ]
    data = [
        constant_op.constant([[0], [40], [70]]),
        constant_op.constant([[10, 11], [60, 61], [20, 21], [30, 31]])
    ]
    with self.assertRaises(ValueError):
      data_flow_ops.dynamic_stitch(indices, data)

  def testErrorDataAndIndicesSizeMismatch(self):
    indices = [
        constant_op.constant([0, 4, 7]), constant_op.constant([1, 6, 2, 3, 5])
    ]
    data = [
        constant_op.constant([0, 40, 70]),
        constant_op.constant([10, 60, 20, 30])
    ]
    with self.assertRaises(ValueError):
      data_flow_ops.dynamic_stitch(indices, data)


if __name__ == "__main__":
  test.main()
