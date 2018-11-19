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
"""Functional tests for BatchToSpace op.

Additional tests are included in spacetobatch_op_test.py, where the BatchToSpace
op is tested in tandem with its reverse SpaceToBatch op.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.platform import test


class PythonOpImpl(object):

  @staticmethod
  def batch_to_space(*args, **kwargs):
    return array_ops.batch_to_space(*args, **kwargs)


class CppOpImpl(object):

  @staticmethod
  def batch_to_space(*args, **kwargs):
    return gen_array_ops.batch_to_space(*args, **kwargs)


class BatchToSpaceDepthToSpace(test.TestCase, PythonOpImpl):

  # Verifies that: batch_to_space(x) = transpose(depth_to_space(transpose(x)))
  def testDepthToSpaceTranspose(self):
    x = np.arange(20 * 5 * 8 * 7, dtype=np.float32).reshape([20, 5, 8, 7])
    block_size = 2
    for crops_dtype in [dtypes.int64, dtypes.int32]:
      crops = array_ops.zeros((2, 2), dtype=crops_dtype)
      y1 = self.batch_to_space(x, crops, block_size=block_size)
      y2 = array_ops.transpose(
          array_ops.depth_to_space(
              array_ops.transpose(x, [3, 1, 2, 0]), block_size=block_size),
          [3, 1, 2, 0])
      with self.cached_session():
        self.assertAllEqual(y1.eval(), y2.eval())


class BatchToSpaceDepthToSpaceCpp(BatchToSpaceDepthToSpace, CppOpImpl):
  pass


class BatchToSpaceErrorHandlingTest(test.TestCase, PythonOpImpl):

  def testInputWrongDimMissingBatch(self):
    # The input is missing the first dimension ("batch")
    x_np = [[[1], [2]], [[3], [4]]]
    crops = np.zeros((2, 2), dtype=np.int32)
    block_size = 2
    with self.assertRaises(ValueError):
      _ = self.batch_to_space(x_np, crops, block_size)

  def testBlockSize0(self):
    # The block size is 0.
    x_np = [[[[1], [2]], [[3], [4]]]]
    crops = np.zeros((2, 2), dtype=np.int32)
    block_size = 0
    with self.assertRaises(ValueError):
      out_tf = self.batch_to_space(x_np, crops, block_size)
      out_tf.eval()

  def testBlockSizeOne(self):
    # The block size is 1. The block size needs to be > 1.
    x_np = [[[[1], [2]], [[3], [4]]]]
    crops = np.zeros((2, 2), dtype=np.int32)
    block_size = 1
    with self.assertRaises(ValueError):
      out_tf = self.batch_to_space(x_np, crops, block_size)
      out_tf.eval()

  def testBlockSizeLarger(self):
    # The block size is too large for this input.
    x_np = [[[[1], [2]], [[3], [4]]]]
    crops = np.zeros((2, 2), dtype=np.int32)
    block_size = 10
    with self.assertRaises(ValueError):
      out_tf = self.batch_to_space(x_np, crops, block_size)
      out_tf.eval()

  def testBlockSizeSquaredNotDivisibleBatch(self):
    # The block size squared does not divide the batch.
    x_np = [[[[1], [2], [3]], [[3], [4], [7]]]]
    crops = np.zeros((2, 2), dtype=np.int32)
    block_size = 3
    with self.assertRaises(ValueError):
      _ = self.batch_to_space(x_np, crops, block_size)

  def testUnknownShape(self):
    t = self.batch_to_space(
        array_ops.placeholder(dtypes.float32),
        array_ops.placeholder(dtypes.int32),
        block_size=4)
    self.assertEqual(4, t.get_shape().ndims)


class BatchToSpaceErrorHandlingCppTest(BatchToSpaceErrorHandlingTest,
                                       CppOpImpl):
  pass


class BatchToSpaceNDErrorHandlingTest(test.TestCase):

  def _testStaticShape(self, input_shape, block_shape, paddings, error):
    block_shape = np.array(block_shape)
    paddings = np.array(paddings)

    # Try with sizes known at graph construction time.
    with self.assertRaises(error):
      _ = array_ops.batch_to_space_nd(
          np.zeros(input_shape, np.float32), block_shape, paddings)

  def _testDynamicShape(self, input_shape, block_shape, paddings):
    block_shape = np.array(block_shape)
    paddings = np.array(paddings)

    # Try with sizes unknown at graph construction time.
    input_placeholder = array_ops.placeholder(dtypes.float32)
    block_shape_placeholder = array_ops.placeholder(
        dtypes.int32, shape=block_shape.shape)
    paddings_placeholder = array_ops.placeholder(dtypes.int32)
    t = array_ops.batch_to_space_nd(input_placeholder, block_shape_placeholder,
                                    paddings_placeholder)

    with self.assertRaises(ValueError):
      _ = t.eval({
          input_placeholder: np.zeros(input_shape, np.float32),
          block_shape_placeholder: block_shape,
          paddings_placeholder: paddings
      })

  def _testShape(self, input_shape, block_shape, paddings, error):
    self._testStaticShape(input_shape, block_shape, paddings, error)
    self._testDynamicShape(input_shape, block_shape, paddings)

  def testInputWrongDimMissingBatch(self):
    self._testShape([2, 2], [2, 2], [[0, 0], [0, 0]], ValueError)
    self._testShape([2, 2, 3], [2, 2, 3], [[0, 0], [0, 0]], ValueError)

  def testBlockSize0(self):
    # The block size is 0.
    self._testShape([1, 2, 2, 1], [0, 1], [[0, 0], [0, 0]], ValueError)

  def testBlockSizeNegative(self):
    self._testShape([1, 2, 2, 1], [-1, 1], [[0, 0], [0, 0]], ValueError)

  def testNegativePadding(self):
    self._testShape([1, 2, 2], [1, 1], [[0, -1], [0, 0]], ValueError)

  def testCropTooLarge(self):
    # The amount to crop exceeds the padded size.
    self._testShape([1 * 2 * 2, 2, 3, 1], [2, 2], [[3, 2], [0, 0]], ValueError)

  def testBlockSizeSquaredNotDivisibleBatch(self):
    # The batch dimension is not divisible by the product of the block_shape.
    self._testShape([3, 1, 1, 1], [2, 3], [[0, 0], [0, 0]], ValueError)

  def testUnknownShape(self):
    # Verify that input shape and paddings shape can be unknown.
    _ = array_ops.batch_to_space_nd(
        array_ops.placeholder(dtypes.float32),
        array_ops.placeholder(
            dtypes.int32, shape=(2,)),
        array_ops.placeholder(dtypes.int32))

    # Only number of input dimensions is known.
    t = array_ops.batch_to_space_nd(
        array_ops.placeholder(
            dtypes.float32, shape=(None, None, None, None)),
        array_ops.placeholder(
            dtypes.int32, shape=(2,)),
        array_ops.placeholder(dtypes.int32))
    self.assertEqual(4, t.get_shape().ndims)

    # Dimensions are partially known.
    t = array_ops.batch_to_space_nd(
        array_ops.placeholder(
            dtypes.float32, shape=(None, None, None, 2)),
        array_ops.placeholder(
            dtypes.int32, shape=(2,)),
        array_ops.placeholder(dtypes.int32))
    self.assertEqual([None, None, None, 2], t.get_shape().as_list())

    # Dimensions are partially known.
    t = array_ops.batch_to_space_nd(
        array_ops.placeholder(
            dtypes.float32, shape=(3 * 2 * 3, None, None, 2)), [2, 3],
        array_ops.placeholder(dtypes.int32))
    self.assertEqual([3, None, None, 2], t.get_shape().as_list())

    # Dimensions are partially known.
    t = array_ops.batch_to_space_nd(
        array_ops.placeholder(
            dtypes.float32, shape=(3 * 2 * 3, None, 2, 2)), [2, 3],
        [[1, 1], [0, 1]])
    self.assertEqual([3, None, 5, 2], t.get_shape().as_list())

    # Dimensions are fully known.
    t = array_ops.batch_to_space_nd(
        array_ops.placeholder(
            dtypes.float32, shape=(3 * 2 * 3, 2, 1, 2)), [2, 3],
        [[1, 1], [0, 0]])
    self.assertEqual([3, 2, 3, 2], t.get_shape().as_list())


class BatchToSpaceGradientTest(test.TestCase, PythonOpImpl):

  # Check the gradients.
  def _checkGrad(self, x, crops, block_size):
    assert 4 == x.ndim
    with self.cached_session():
      tf_x = ops.convert_to_tensor(x)
      tf_y = self.batch_to_space(tf_x, crops, block_size)
      epsilon = 1e-5
      ((x_jacob_t, x_jacob_n)) = gradient_checker.compute_gradient(
          tf_x,
          x.shape,
          tf_y,
          tf_y.get_shape().as_list(),
          x_init_value=x,
          delta=epsilon)

    self.assertAllClose(x_jacob_t, x_jacob_n, rtol=1e-2, atol=epsilon)

  # Tests a gradient for batch_to_space of x which is a four dimensional
  # tensor of shape [b * block_size * block_size, h, w, d].
  def _compare(self, b, h, w, d, block_size, crop_beg, crop_end):
    block_size_sq = block_size * block_size
    x = np.random.normal(0, 1, b * h * w * d *
                         block_size_sq).astype(np.float32).reshape(
                             [b * block_size * block_size, h, w, d])
    crops = np.array(
        [[crop_beg, crop_end], [crop_beg, crop_end]], dtype=np.int32)

    self._checkGrad(x, crops, block_size)

  # Don't use very large numbers as dimensions here as the result is tensor
  # with cartesian product of the dimensions.
  def testSmall(self):
    block_size = 2
    crop_beg = 0
    crop_end = 0
    self._compare(1, 2, 3, 5, block_size, crop_beg, crop_end)

  def testSmall2(self):
    block_size = 2
    crop_beg = 0
    crop_end = 0
    self._compare(2, 4, 3, 2, block_size, crop_beg, crop_end)

  def testSmallCrop1x1(self):
    block_size = 2
    crop_beg = 1
    crop_end = 1
    self._compare(1, 2, 3, 5, block_size, crop_beg, crop_end)


class BatchToSpaceGradientCppTest(BatchToSpaceGradientTest, CppOpImpl):
  pass


class BatchToSpaceNDGradientTest(test.TestCase):

  # Check the gradients.
  def _checkGrad(self, x, block_shape, crops, crops_dtype):
    block_shape = np.array(block_shape)
    crops = constant_op.constant(
        np.array(crops).reshape((len(block_shape), 2)), crops_dtype)
    with self.cached_session():
      tf_x = ops.convert_to_tensor(x)
      tf_y = array_ops.batch_to_space_nd(tf_x, block_shape, crops)
      epsilon = 1e-5
      ((x_jacob_t, x_jacob_n)) = gradient_checker.compute_gradient(
          tf_x,
          x.shape,
          tf_y,
          tf_y.get_shape().as_list(),
          x_init_value=x,
          delta=epsilon)

    self.assertAllClose(x_jacob_t, x_jacob_n, rtol=1e-2, atol=epsilon)

  def _compare(self, input_shape, block_shape, crops, crops_dtype):
    input_shape = list(input_shape)
    input_shape[0] *= np.prod(block_shape)
    x = np.random.normal(
        0, 1, np.prod(input_shape)).astype(np.float32).reshape(input_shape)
    self._checkGrad(x, block_shape, crops, crops_dtype)

  # Don't use very large numbers as dimensions here as the result is tensor
  # with cartesian product of the dimensions.
  def testSmall(self):
    for dtype in [dtypes.int64, dtypes.int32]:
      self._compare([1, 2, 3, 5], [2, 2], [[0, 0], [0, 0]], dtype)

  def testSmall2(self):
    for dtype in [dtypes.int64, dtypes.int32]:
      self._compare([2, 4, 3, 2], [2, 2], [[0, 0], [0, 0]], dtype)

  def testSmallCrop1x1(self):
    for dtype in [dtypes.int64, dtypes.int32]:
      self._compare([1, 2, 3, 5], [2, 2], [[1, 1], [1, 1]], dtype)


if __name__ == "__main__":
  test.main()
