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
"""Tests for tensorflow.ops.data_flow_ops.{,parallel_}dynamic_stitch."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gradients_impl
import tensorflow.python.ops.data_flow_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class DynamicStitchTestBase(object):

  def __init__(self, stitch_op):
    self.stitch_op = stitch_op

  def testScalar(self):
    with test_util.use_gpu():
      indices = [constant_op.constant(0), constant_op.constant(1)]
      data = [constant_op.constant(40), constant_op.constant(60)]
      for step in -1, 1:
        stitched_t = self.stitch_op(indices[::step], data)
        stitched_val = self.evaluate(stitched_t)
        self.assertAllEqual([40, 60][::step], stitched_val)
        # Dimension 0 is max(flatten(indices))+1.
        self.assertEqual([2], stitched_t.get_shape().as_list())

  @test_util.run_deprecated_v1
  def testShapeInferenceForScalarWithNonConstantIndices(self):
    with test_util.use_gpu():
      indices = [
          array_ops.placeholder(dtype=dtypes.int32),
          constant_op.constant(1)
      ]
      data = [constant_op.constant(40), constant_op.constant(60)]
      for step in -1, 1:
        stitched_t = self.stitch_op(indices[::step], data)
        # Dimension 0 is max(flatten(indices))+1, but the first indices input is
        # not a constant tensor, so we can only infer it as a vector of unknown
        # length.
        self.assertEqual([None], stitched_t.get_shape().as_list())

  @test_util.disable_tfrt("b/169901260")
  def testSimpleOneDimensional(self):
    # Test various datatypes in the simple case to ensure that the op was
    # registered under those types.
    dtypes_to_test = [
        dtypes.float32, dtypes.qint8, dtypes.quint8, dtypes.qint32
    ]
    for dtype in dtypes_to_test:
      indices = [
          constant_op.constant([0, 4, 7]),
          constant_op.constant([1, 6, 2, 3, 5])
      ]
      data = [
          math_ops.cast(constant_op.constant([0, 40, 70]), dtype=dtype),
          math_ops.cast(
              constant_op.constant([10, 60, 20, 30, 50]), dtype=dtype)
      ]
      stitched_t = self.stitch_op(indices, data)
      stitched_val = self.evaluate(stitched_t)
      self.assertAllEqual([0, 10, 20, 30, 40, 50, 60, 70], stitched_val)
      # Dimension 0 is max(flatten(indices))+1.
      self.assertEqual([8], stitched_t.get_shape().as_list())

  def testOneListOneDimensional(self):
    indices = [constant_op.constant([1, 6, 2, 3, 5, 0, 4, 7])]
    data = [constant_op.constant([10, 60, 20, 30, 50, 0, 40, 70])]
    stitched_t = self.stitch_op(indices, data)
    stitched_val = self.evaluate(stitched_t)
    self.assertAllEqual([0, 10, 20, 30, 40, 50, 60, 70], stitched_val)
    # Dimension 0 is max(flatten(indices))+1.
    self.assertEqual([8], stitched_t.get_shape().as_list())

  def testSimpleTwoDimensional(self):
    indices = [
        constant_op.constant([0, 4, 7]),
        constant_op.constant([1, 6]),
        constant_op.constant([2, 3, 5])
    ]
    data = [
        constant_op.constant([[0, 1], [40, 41], [70, 71]]),
        constant_op.constant([[10, 11], [60, 61]]),
        constant_op.constant([[20, 21], [30, 31], [50, 51]])
    ]
    stitched_t = self.stitch_op(indices, data)
    stitched_val = self.evaluate(stitched_t)
    self.assertAllEqual([[0, 1], [10, 11], [20, 21], [30, 31], [40, 41],
                         [50, 51], [60, 61], [70, 71]], stitched_val)
    # Dimension 0 is max(flatten(indices))+1.
    self.assertEqual([8, 2], stitched_t.get_shape().as_list())

  def testZeroSizeTensor(self):
    indices = [
        constant_op.constant([0, 4, 7]),
        constant_op.constant([1, 6]),
        constant_op.constant([2, 3, 5]),
        array_ops.zeros([0], dtype=dtypes.int32)
    ]
    data = [
        constant_op.constant([[0, 1], [40, 41], [70, 71]]),
        constant_op.constant([[10, 11], [60, 61]]),
        constant_op.constant([[20, 21], [30, 31], [50, 51]]),
        array_ops.zeros([0, 2], dtype=dtypes.int32)
    ]
    stitched_t = self.stitch_op(indices, data)
    stitched_val = self.evaluate(stitched_t)
    self.assertAllEqual([[0, 1], [10, 11], [20, 21], [30, 31], [40, 41],
                         [50, 51], [60, 61], [70, 71]], stitched_val)
    # Dimension 0 is max(flatten(indices))+1.
    self.assertEqual([8, 2], stitched_t.get_shape().as_list())

  def testAllZeroSizeTensor(self):
    indices = [
        array_ops.zeros([0], dtype=dtypes.int32),
        array_ops.zeros([0], dtype=dtypes.int32)
    ]
    data = [
        array_ops.zeros([0, 2], dtype=dtypes.int32),
        array_ops.zeros([0, 2], dtype=dtypes.int32)
    ]
    stitched_t = self.stitch_op(indices, data)
    stitched_val = self.evaluate(stitched_t)
    self.assertAllEqual(np.zeros((0, 2)), stitched_val)
    self.assertEqual([0, 2], stitched_t.get_shape().as_list())

  @test_util.run_deprecated_v1
  def testHigherRank(self):
    indices = [
        constant_op.constant(6),
        constant_op.constant([4, 1]),
        constant_op.constant([[5, 2], [0, 3]])
    ]
    data = [
        constant_op.constant([61., 62.]),
        constant_op.constant([[41., 42.], [11., 12.]]),
        constant_op.constant([[[51., 52.], [21., 22.]],
                              [[1., 2.], [31., 32.]]])
    ]
    stitched_t = self.stitch_op(indices, data)
    stitched_val = self.evaluate(stitched_t)
    correct = 10. * np.arange(7)[:, None] + [1., 2.]
    self.assertAllEqual(correct, stitched_val)
    self.assertEqual([7, 2], stitched_t.get_shape().as_list())
    # Test gradients
    stitched_grad = 7. * stitched_val
    grads = gradients_impl.gradients(stitched_t, indices + data,
                                     stitched_grad)
    self.assertEqual(grads[:3], [None] * 3)  # Indices have no gradients
    for datum, grad in zip(data, self.evaluate(grads[3:])):
      self.assertAllEqual(7. * self.evaluate(datum), grad)

  @test_util.run_deprecated_v1
  def testErrorIndicesMultiDimensional(self):
    indices = [
        constant_op.constant([0, 4, 7]),
        constant_op.constant([[1, 6, 2, 3, 5]])
    ]
    data = [
        constant_op.constant([[0, 40, 70]]),
        constant_op.constant([10, 60, 20, 30, 50])
    ]
    with self.assertRaises(ValueError):
      self.stitch_op(indices, data)

  @test_util.run_deprecated_v1
  def testErrorDataNumDimsMismatch(self):
    indices = [
        constant_op.constant([0, 4, 7]),
        constant_op.constant([1, 6, 2, 3, 5])
    ]
    data = [
        constant_op.constant([0, 40, 70]),
        constant_op.constant([[10, 60, 20, 30, 50]])
    ]
    with self.assertRaises(ValueError):
      self.stitch_op(indices, data)

  @test_util.run_deprecated_v1
  def testErrorDataDimSizeMismatch(self):
    indices = [
        constant_op.constant([0, 4, 5]),
        constant_op.constant([1, 6, 2, 3])
    ]
    data = [
        constant_op.constant([[0], [40], [70]]),
        constant_op.constant([[10, 11], [60, 61], [20, 21], [30, 31]])
    ]
    with self.assertRaises(ValueError):
      self.stitch_op(indices, data)

  @test_util.run_deprecated_v1
  def testErrorDataAndIndicesSizeMismatch(self):
    indices = [
        constant_op.constant([0, 4, 7]),
        constant_op.constant([1, 6, 2, 3, 5])
    ]
    data = [
        constant_op.constant([0, 40, 70]),
        constant_op.constant([10, 60, 20, 30])
    ]
    with self.assertRaises(ValueError):
      self.stitch_op(indices, data)


class DynamicStitchTest(DynamicStitchTestBase, test.TestCase):

  def __init__(self, *test_case_args):
    test.TestCase.__init__(self, *test_case_args)
    DynamicStitchTestBase.__init__(self, data_flow_ops.dynamic_stitch)


class ParallelDynamicStitchTest(DynamicStitchTestBase, test.TestCase):

  def __init__(self, *test_case_args):
    test.TestCase.__init__(self, *test_case_args)
    DynamicStitchTestBase.__init__(self, data_flow_ops.parallel_dynamic_stitch)

  def testScalar(self):
    with test_util.use_gpu():
      indices = [constant_op.constant(0), constant_op.constant(1)]
      data = [constant_op.constant(40.0), constant_op.constant(60.0)]
      for step in -1, 1:
        stitched_t = data_flow_ops.dynamic_stitch(indices[::step], data)
        stitched_val = self.evaluate(stitched_t)
        self.assertAllEqual([40.0, 60.0][::step], stitched_val)
        # Dimension 0 is max(flatten(indices))+1.
        self.assertEqual([2], stitched_t.get_shape().as_list())

  @test_util.run_deprecated_v1
  def testHigherRank(self):
    indices = [
        constant_op.constant(6),
        constant_op.constant([4, 1]),
        constant_op.constant([[5, 2], [0, 3]])
    ]
    data = [
        constant_op.constant([61, 62], dtype=dtypes.float32),
        constant_op.constant([[41, 42], [11, 12]], dtype=dtypes.float32),
        constant_op.constant(
            [[[51, 52], [21, 22]], [[1, 2], [31, 32]]], dtype=dtypes.float32)
    ]
    stitched_t = data_flow_ops.dynamic_stitch(indices, data)
    stitched_val = self.evaluate(stitched_t)
    correct = 10 * np.arange(7)[:, None] + [1.0, 2.0]
    self.assertAllEqual(correct, stitched_val)
    self.assertEqual([7, 2], stitched_t.get_shape().as_list())
    # Test gradients
    stitched_grad = 7 * stitched_val
    grads = gradients_impl.gradients(stitched_t, indices + data,
                                     stitched_grad)
    self.assertEqual(grads[:3], [None] * 3)  # Indices have no gradients
    for datum, grad in zip(data, self.evaluate(grads[3:])):
      self.assertAllEqual(7.0 * self.evaluate(datum), grad)

  # GPU version unit tests
  def testScalarGPU(self):
    indices = [constant_op.constant(0), constant_op.constant(1)]
    data = [constant_op.constant(40.0), constant_op.constant(60.0)]
    for step in -1, 1:
      stitched_t = data_flow_ops.dynamic_stitch(indices[::step], data)
      stitched_val = self.evaluate(stitched_t)
      self.assertAllEqual([40.0, 60.0][::step], stitched_val)
      # Dimension 0 is max(flatten(indices))+1.
      self.assertEqual([2], stitched_t.get_shape().as_list())

  @test_util.run_deprecated_v1
  def testHigherRankGPU(self):
    indices = [
        constant_op.constant(6),
        constant_op.constant([4, 1]),
        constant_op.constant([[5, 2], [0, 3]])
    ]
    data = [
        constant_op.constant([61, 62], dtype=dtypes.float32),
        constant_op.constant([[41, 42], [11, 12]], dtype=dtypes.float32),
        constant_op.constant(
            [[[51, 52], [21, 22]], [[1, 2], [31, 32]]], dtype=dtypes.float32)
    ]
    stitched_t = data_flow_ops.dynamic_stitch(indices, data)
    stitched_val = self.evaluate(stitched_t)
    correct = 10 * np.arange(7)[:, None] + [1.0, 2.0]
    self.assertAllEqual(correct, stitched_val)
    self.assertEqual([7, 2], stitched_t.get_shape().as_list())
    # Test gradients
    stitched_grad = 7 * stitched_val
    grads = gradients_impl.gradients(stitched_t, indices + data,
                                     stitched_grad)
    self.assertEqual(grads[:3], [None] * 3)  # Indices have no gradients
    for datum, grad in zip(data, self.evaluate(grads[3:])):
      self.assertAllEqual(7.0 * self.evaluate(datum), grad)

  @test_util.run_in_graph_and_eager_modes
  def testMismatchedDataAndIndexListSizes(self):
    indices = [
        constant_op.constant([2]),
        constant_op.constant([1]),
        constant_op.constant([0]),
        constant_op.constant([3]),
    ]
    data = [
        constant_op.constant([1.0]),
        constant_op.constant([2.0]),
        constant_op.constant([3.0]),
        constant_op.constant([4.0])
    ]
    with self.assertRaisesRegex(
        (ValueError, errors.InvalidArgumentError),
        "expected inputs .* do not match|List argument .* must match"):
      self.evaluate(data_flow_ops.dynamic_stitch(indices[0:2], data))

    with self.assertRaisesRegex(
        (ValueError, errors.InvalidArgumentError),
        "expected inputs .* do not match|List argument .* must match"):
      self.evaluate(data_flow_ops.dynamic_stitch(indices, data[0:2]))

if __name__ == "__main__":
  test.main()
