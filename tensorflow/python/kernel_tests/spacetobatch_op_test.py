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
"""Functional tests for SpaceToBatch and BatchToSpace ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


def space_to_batch_direct(input_array, block_shape, paddings):
  """Direct Python implementation of space-to-batch conversion.

  This is used for tests only.

  Args:
    input_array: N-D array
    block_shape: 1-D array of shape [num_block_dims].
    paddings: 2-D array of shape [num_block_dims, 2].

  Returns:
    Converted tensor.
  """
  input_array = np.array(input_array)
  block_shape = np.array(block_shape)
  num_block_dims = len(block_shape)
  paddings = np.array(paddings).reshape((len(block_shape), 2))

  padded = np.pad(input_array,
                  pad_width=([[0, 0]] + list(paddings) + [[0, 0]] *
                             (input_array.ndim - 1 - num_block_dims)),
                  mode="constant")
  reshaped_padded_shape = [input_array.shape[0]]
  output_shape = [input_array.shape[0] * np.prod(block_shape)]
  for block_dim, block_shape_value in enumerate(block_shape):
    reduced_size = padded.shape[block_dim + 1] // block_shape_value
    reshaped_padded_shape.append(reduced_size)
    output_shape.append(reduced_size)
    reshaped_padded_shape.append(block_shape_value)
  reshaped_padded_shape.extend(input_array.shape[num_block_dims + 1:])
  output_shape.extend(input_array.shape[num_block_dims + 1:])

  reshaped_padded = padded.reshape(reshaped_padded_shape)
  permuted_reshaped_padded = np.transpose(reshaped_padded, (
      list(np.arange(num_block_dims) * 2 + 2) + [0] +
      list(np.arange(num_block_dims) * 2 + 1) + list(
          np.arange(input_array.ndim - num_block_dims - 1) + 1 + num_block_dims
          * 2)))
  return permuted_reshaped_padded.reshape(output_shape)


class PythonOpImpl(object):

  @staticmethod
  def space_to_batch(*args, **kwargs):
    return array_ops.space_to_batch(*args, **kwargs)

  @staticmethod
  def batch_to_space(*args, **kwargs):
    return array_ops.batch_to_space(*args, **kwargs)


class CppOpImpl(object):

  @staticmethod
  def space_to_batch(*args, **kwargs):
    return gen_array_ops.space_to_batch(*args, **kwargs)

  @staticmethod
  def batch_to_space(*args, **kwargs):
    return gen_array_ops.batch_to_space(*args, **kwargs)


class SpaceToBatchTest(test.TestCase, PythonOpImpl):
  """Tests input-output pairs for the SpaceToBatch and BatchToSpace ops.

  This uses the Python compatibility wrapper that forwards to space_to_batch_nd.
  """

  def _testPad(self, inputs, paddings, block_size, outputs):
    with self.cached_session(use_gpu=True):
      # outputs = space_to_batch(inputs)
      x_tf = self.space_to_batch(
          math_ops.to_float(inputs), paddings, block_size=block_size)
      self.assertAllEqual(x_tf.eval(), outputs)
      # inputs = batch_to_space(outputs)
      x_tf = self.batch_to_space(
          math_ops.to_float(outputs), paddings, block_size=block_size)
      self.assertAllEqual(x_tf.eval(), inputs)

  def _testOne(self, inputs, block_size, outputs):
    paddings = np.zeros((2, 2), dtype=np.int32)
    self._testPad(inputs, paddings, block_size, outputs)

  # [1, 2, 2, 1] <-> [4, 1, 1, 1]
  def testSmallInput2x2(self):
    x_np = [[[[1], [2]], [[3], [4]]]]
    block_size = 2
    x_out = [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
    self._testOne(x_np, block_size, x_out)

  # [1, 2, 2, 1] <-> [1, 3, 3, 1] (padding) <-> [9, 1, 1, 1]
  def testSmallInput2x2Pad1x0(self):
    x_np = [[[[1], [2]], [[3], [4]]]]
    paddings = np.array([[1, 0], [1, 0]], dtype=np.int32)
    block_size = 3
    x_out = [[[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[1]]], [[[2]]], [[[0]]],
             [[[3]]], [[[4]]]]
    self._testPad(x_np, paddings, block_size, x_out)

  # Test with depth larger than 1.
  # [1, 2, 2, 3] <-> [4, 1, 1, 3]
  def testDepthInput2x2(self):
    x_np = [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]
    block_size = 2
    x_out = [[[[1, 2, 3]]], [[[4, 5, 6]]], [[[7, 8, 9]]], [[[10, 11, 12]]]]
    self._testOne(x_np, block_size, x_out)

  # Test for larger input dimensions.
  # [1, 4, 4, 1] <-> [4, 2, 2, 1]
  def testLargerInput2x2(self):
    x_np = [[[[1], [2], [3], [4]], [[5], [6], [7], [8]],
             [[9], [10], [11], [12]], [[13], [14], [15], [16]]]]
    block_size = 2
    x_out = [[[[1], [3]], [[9], [11]]], [[[2], [4]], [[10], [12]]],
             [[[5], [7]], [[13], [15]]], [[[6], [8]], [[14], [16]]]]
    self._testOne(x_np, block_size, x_out)

  # Test with batch larger than 1.
  # [2, 2, 4, 1] <-> [8, 1, 2, 1]
  def testBatchInput2x2(self):
    x_np = [[[[1], [2], [3], [4]], [[5], [6], [7], [8]]],
            [[[9], [10], [11], [12]], [[13], [14], [15], [16]]]]
    block_size = 2
    x_out = [[[[1], [3]]], [[[9], [11]]], [[[2], [4]]], [[[10], [12]]],
             [[[5], [7]]], [[[13], [15]]], [[[6], [8]]], [[[14], [16]]]]
    self._testOne(x_np, block_size, x_out)

  # Tests for larger input spatial dimensions AND batch larger than 1, to ensure
  # that elements are correctly laid out spatially and properly interleaved
  # along the batch dimension.
  # [2, 4, 4, 1] <-> [8, 2, 2, 1]
  def testLargerInputBatch2x2(self):
    x_np = [[[[1], [2], [3], [4]], [[5], [6], [7], [8]],
             [[9], [10], [11], [12]], [[13], [14], [15], [16]]],
            [[[17], [18], [19], [20]], [[21], [22], [23], [24]],
             [[25], [26], [27], [28]], [[29], [30], [31], [32]]]]
    x_out = [[[[1], [3]], [[9], [11]]], [[[17], [19]], [[25], [27]]],
             [[[2], [4]], [[10], [12]]], [[[18], [20]], [[26], [28]]],
             [[[5], [7]], [[13], [15]]], [[[21], [23]], [[29], [31]]],
             [[[6], [8]], [[14], [16]]], [[[22], [24]], [[30], [32]]]]
    block_size = 2
    self._testOne(x_np, block_size, x_out)


class SpaceToBatchCppTest(SpaceToBatchTest, CppOpImpl):
  """Tests input-output pairs for the SpaceToBatch and BatchToSpace ops.

  This uses the C++ ops.
  """
  pass


class SpaceToBatchNDTest(test.TestCase):
  """Tests input-output pairs for the SpaceToBatchND and BatchToSpaceND ops."""

  def _testPad(self, inputs, block_shape, paddings, outputs):
    block_shape = np.array(block_shape)
    paddings = np.array(paddings).reshape((len(block_shape), 2))
    for use_gpu in [False, True]:
      with self.cached_session(use_gpu=use_gpu):
        # outputs = space_to_batch(inputs)
        x_tf = array_ops.space_to_batch_nd(
            math_ops.to_float(inputs), block_shape, paddings)
        self.assertAllEqual(x_tf.eval(), outputs)
        # inputs = batch_to_space(outputs)
        x_tf = array_ops.batch_to_space_nd(
            math_ops.to_float(outputs), block_shape, paddings)
        self.assertAllEqual(x_tf.eval(), inputs)

  def _testDirect(self, input_shape, block_shape, paddings):
    inputs = np.arange(np.prod(input_shape), dtype=np.float32)
    inputs = inputs.reshape(input_shape)
    self._testPad(inputs, block_shape, paddings,
                  space_to_batch_direct(inputs, block_shape, paddings))

  def testZeroBlockDimsZeroRemainingDims(self):
    self._testPad(
        inputs=[1, 2],
        block_shape=[],
        paddings=[],
        outputs=[1, 2],)

  def testZeroBlockDimsOneRemainingDim(self):
    self._testPad(
        inputs=[[1, 2], [3, 4]],
        block_shape=[],
        paddings=[],
        outputs=[[1, 2], [3, 4]])

    # Same thing, but with a no-op block dim.
    self._testPad(
        inputs=[[1, 2], [3, 4]],
        block_shape=[1],
        paddings=[[0, 0]],
        outputs=[[1, 2], [3, 4]])

  def testZeroBlockDimsTwoRemainingDims(self):
    self._testPad(
        inputs=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        block_shape=[],
        paddings=[],
        outputs=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    # Same thing, but with a no-op block dim.
    self._testPad(
        inputs=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        block_shape=[1],
        paddings=[[0, 0]],
        outputs=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    # Same thing, but with two no-op block dims.
    self._testPad(
        inputs=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        block_shape=[1, 1],
        paddings=[[0, 0], [0, 0]],
        outputs=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

  def testOneBlockDimZeroRemainingDims(self):
    self._testPad(
        inputs=[[1, 2, 3], [4, 5, 6]],
        block_shape=[2],
        paddings=[1, 0],
        outputs=[[0, 2], [0, 5], [1, 3], [4, 6]])

  def testOneBlockDimOneRemainingDim(self):
    self._testPad(
        inputs=[[[1, 11], [2, 21], [3, 31]], [[4, 41], [5, 51], [6, 61]]],
        block_shape=[2],
        paddings=[1, 0],
        outputs=[[[0, 0], [2, 21]], [[0, 0], [5, 51]], [[1, 11], [3, 31]],
                 [[4, 41], [6, 61]]])

  def testDirect(self):
    # Test with zero-size remaining dimension.
    self._testDirect(
        input_shape=[3, 1, 2, 0], block_shape=[3], paddings=[[0, 2]])

    # Test with zero-size blocked dimension.
    self._testDirect(
        input_shape=[3, 0, 2, 5], block_shape=[3], paddings=[[0, 0]])

    # Test with padding up from zero size.
    self._testDirect(
        input_shape=[3, 0, 2, 5], block_shape=[3], paddings=[[1, 2]])

    self._testDirect(
        input_shape=[3, 3, 4, 5, 2],
        block_shape=[3, 4, 2],
        paddings=[[1, 2], [0, 0], [3, 0]])

    self._testDirect(
        input_shape=[3, 3, 4, 5, 2],
        block_shape=[3, 4, 2, 2],
        paddings=[[1, 2], [0, 0], [3, 0], [0, 0]])

    self._testDirect(
        input_shape=[3, 2, 2, 3, 4, 5, 2, 5],
        block_shape=[1, 1, 3, 4, 2, 2],
        paddings=[[0, 0], [0, 0], [1, 2], [0, 0], [3, 0], [0, 0]])

    self._testDirect(
        input_shape=[3, 2, 2, 3, 4, 5, 2, 5],
        block_shape=[1, 1, 3, 4, 2, 2, 1],
        paddings=[[0, 0], [0, 0], [1, 2], [0, 0], [3, 0], [0, 0], [0, 0]])


class SpaceToBatchSpaceToDepth(test.TestCase, PythonOpImpl):

  # Verifies that: space_to_batch(x) = transpose(space_to_depth(transpose(x)))
  def testSpaceToDepthTranspose(self):
    x = np.arange(5 * 10 * 16 * 7, dtype=np.float32).reshape([5, 10, 16, 7])
    block_size = 2
    paddings = np.zeros((2, 2), dtype=np.int32)
    y1 = self.space_to_batch(x, paddings, block_size=block_size)
    y2 = array_ops.transpose(
        array_ops.space_to_depth(
            array_ops.transpose(x, [3, 1, 2, 0]), block_size=block_size),
        [3, 1, 2, 0])
    with self.session(use_gpu=True):
      self.assertAllEqual(y1.eval(), y2.eval())


class SpaceToBatchSpaceToDepthCpp(SpaceToBatchSpaceToDepth, CppOpImpl):
  pass


class SpaceToBatchErrorHandlingTest(test.TestCase, PythonOpImpl):

  def testInputWrongDimMissingBatch(self):
    # The input is missing the first dimension ("batch")
    x_np = [[[1], [2]], [[3], [4]]]
    paddings = np.zeros((2, 2), dtype=np.int32)
    block_size = 2
    with self.assertRaises(ValueError):
      _ = self.space_to_batch(x_np, paddings, block_size)

  def testBlockSize0(self):
    # The block size is 0.
    x_np = [[[[1], [2]], [[3], [4]]]]
    paddings = np.zeros((2, 2), dtype=np.int32)
    block_size = 0
    with self.assertRaises(ValueError):
      out_tf = self.space_to_batch(x_np, paddings, block_size)
      out_tf.eval()

  def testBlockSizeOne(self):
    # The block size is 1. The block size needs to be > 1.
    x_np = [[[[1], [2]], [[3], [4]]]]
    paddings = np.zeros((2, 2), dtype=np.int32)
    block_size = 1
    with self.assertRaises(ValueError):
      out_tf = self.space_to_batch(x_np, paddings, block_size)
      out_tf.eval()

  def testBlockSizeLarger(self):
    # The block size is too large for this input.
    x_np = [[[[1], [2]], [[3], [4]]]]
    paddings = np.zeros((2, 2), dtype=np.int32)
    block_size = 10
    with self.assertRaises(ValueError):
      out_tf = self.space_to_batch(x_np, paddings, block_size)
      out_tf.eval()

  def testBlockSizeNotDivisibleWidth(self):
    # The block size divides width but not height.
    x_np = [[[[1], [2], [3]], [[3], [4], [7]]]]
    paddings = np.zeros((2, 2), dtype=np.int32)
    block_size = 3
    with self.assertRaises(ValueError):
      _ = self.space_to_batch(x_np, paddings, block_size)

  def testBlockSizeNotDivisibleHeight(self):
    # The block size divides height but not width.
    x_np = [[[[1], [2]], [[3], [4]], [[5], [6]]]]
    paddings = np.zeros((2, 2), dtype=np.int32)
    block_size = 3
    with self.assertRaises(ValueError):
      _ = self.space_to_batch(x_np, paddings, block_size)

  def testBlockSizeNotDivisibleBoth(self):
    # The block size does not divide neither width or height.
    x_np = [[[[1], [2]], [[3], [4]]]]
    paddings = np.zeros((2, 2), dtype=np.int32)
    block_size = 3
    with self.assertRaises(ValueError):
      _ = self.space_to_batch(x_np, paddings, block_size)

  def testUnknownShape(self):
    t = self.space_to_batch(
        array_ops.placeholder(dtypes.float32),
        array_ops.placeholder(dtypes.int32),
        block_size=4)
    self.assertEqual(4, t.get_shape().ndims)


class SpaceToBatchErrorHandlingCppTest(SpaceToBatchErrorHandlingTest,
                                       CppOpImpl):
  pass


class SpaceToBatchNDErrorHandlingTest(test.TestCase):

  def _testStaticShape(self, input_shape, block_shape, paddings, error):
    block_shape = np.array(block_shape)
    paddings = np.array(paddings)

    # Try with sizes known at graph construction time.
    with self.assertRaises(error):
      _ = array_ops.space_to_batch_nd(
          np.zeros(input_shape, np.float32), block_shape, paddings)

  def _testDynamicShape(self, input_shape, block_shape, paddings):
    block_shape = np.array(block_shape)
    paddings = np.array(paddings)
    # Try with sizes unknown at graph construction time.
    input_placeholder = array_ops.placeholder(dtypes.float32)
    block_shape_placeholder = array_ops.placeholder(
        dtypes.int32, shape=block_shape.shape)
    paddings_placeholder = array_ops.placeholder(dtypes.int32)
    t = array_ops.space_to_batch_nd(input_placeholder, block_shape_placeholder,
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

  def testBlockSize0(self):
    # The block size is 0.
    self._testShape([1, 2, 2], [0, 2], [[0, 0], [0, 0]], ValueError)

  def testBlockSizeNegative(self):
    self._testShape([1, 2, 2], [-1, 2], [[0, 0], [0, 0]], ValueError)

  def testNegativePadding(self):
    # The padding is negative.
    self._testShape([1, 2, 2], [1, 1], [[0, -1], [0, 0]], ValueError)

  def testBlockSizeNotDivisible(self):
    # The padded size is not divisible by the block size.
    self._testShape([1, 2, 3, 1], [3, 3], [[0, 0], [0, 0]], ValueError)

  def testBlockDimsMismatch(self):
    # Shape of block_shape does not match shape of paddings.
    self._testStaticShape([1, 3, 3, 1], [3, 3], [[0, 0]], ValueError)

  def testUnknown(self):
    # Verify that input shape and paddings shape can be unknown.
    _ = array_ops.space_to_batch_nd(
        array_ops.placeholder(dtypes.float32),
        array_ops.placeholder(
            dtypes.int32, shape=(2,)),
        array_ops.placeholder(dtypes.int32))

    # Only number of input dimensions is known.
    t = array_ops.space_to_batch_nd(
        array_ops.placeholder(
            dtypes.float32, shape=(None, None, None, None)),
        array_ops.placeholder(
            dtypes.int32, shape=(2,)),
        array_ops.placeholder(dtypes.int32))
    self.assertEqual(4, t.get_shape().ndims)

    # Dimensions are partially known.
    t = array_ops.space_to_batch_nd(
        array_ops.placeholder(
            dtypes.float32, shape=(None, None, None, 2)),
        array_ops.placeholder(
            dtypes.int32, shape=(2,)),
        array_ops.placeholder(dtypes.int32))
    self.assertEqual([None, None, None, 2], t.get_shape().as_list())

    # Dimensions are partially known.
    t = array_ops.space_to_batch_nd(
        array_ops.placeholder(
            dtypes.float32, shape=(3, None, None, 2)), [2, 3],
        array_ops.placeholder(dtypes.int32))
    self.assertEqual([3 * 2 * 3, None, None, 2], t.get_shape().as_list())

    # Dimensions are partially known.
    t = array_ops.space_to_batch_nd(
        array_ops.placeholder(
            dtypes.float32, shape=(3, None, 2, 2)), [2, 3], [[1, 1], [0, 1]])
    self.assertEqual([3 * 2 * 3, None, 1, 2], t.get_shape().as_list())

    # Dimensions are fully known.
    t = array_ops.space_to_batch_nd(
        array_ops.placeholder(
            dtypes.float32, shape=(3, 2, 3, 2)), [2, 3], [[1, 1], [0, 0]])
    self.assertEqual([3 * 2 * 3, 2, 1, 2], t.get_shape().as_list())


class SpaceToBatchGradientTest(test.TestCase, PythonOpImpl):

  # Check the gradients.
  def _checkGrad(self, x, paddings, block_size):
    assert 4 == x.ndim
    with self.cached_session(use_gpu=True):
      tf_x = ops.convert_to_tensor(x)
      tf_y = self.space_to_batch(tf_x, paddings, block_size)
      epsilon = 1e-5
      ((x_jacob_t, x_jacob_n)) = gradient_checker.compute_gradient(
          tf_x,
          x.shape,
          tf_y,
          tf_y.get_shape().as_list(),
          x_init_value=x,
          delta=epsilon)

    self.assertAllClose(x_jacob_t, x_jacob_n, rtol=1e-2, atol=epsilon)

  # Tests a gradient for space_to_batch of x which is a four dimensional
  # tensor of shape [b, h * block_size, w * block_size, d].
  def _compare(self, b, h, w, d, block_size, pad_beg, pad_end):
    block_size_sq = block_size * block_size
    x = np.random.normal(0, 1, b * h * w * d *
                         block_size_sq).astype(np.float32).reshape(
                             [b, h * block_size, w * block_size, d])
    paddings = np.array(
        [[pad_beg, pad_end], [pad_beg, pad_end]], dtype=np.int32)

    self._checkGrad(x, paddings, block_size)

  # Don't use very large numbers as dimensions here as the result is tensor
  # with cartesian product of the dimensions.
  def testSmall(self):
    block_size = 2
    pad_beg = 0
    pad_end = 0
    self._compare(1, 2, 3, 5, block_size, pad_beg, pad_end)

  def testSmall2(self):
    block_size = 2
    pad_beg = 0
    pad_end = 0
    self._compare(2, 4, 3, 2, block_size, pad_beg, pad_end)

  def testSmallPad1x1(self):
    block_size = 2
    pad_beg = 1
    pad_end = 1
    self._compare(1, 2, 3, 5, block_size, pad_beg, pad_end)


class SpaceToBatchGradientCppTest(SpaceToBatchGradientTest, CppOpImpl):
  pass


class SpaceToBatchNDGradientTest(test.TestCase):

  # Check the gradients.
  def _checkGrad(self, x, block_shape, paddings):
    block_shape = np.array(block_shape)
    paddings = np.array(paddings).reshape((len(block_shape), 2))
    with self.cached_session():
      tf_x = ops.convert_to_tensor(x)
      tf_y = array_ops.space_to_batch_nd(tf_x, block_shape, paddings)
      epsilon = 1e-5
      ((x_jacob_t, x_jacob_n)) = gradient_checker.compute_gradient(
          tf_x,
          x.shape,
          tf_y,
          tf_y.get_shape().as_list(),
          x_init_value=x,
          delta=epsilon)

    self.assertAllClose(x_jacob_t, x_jacob_n, rtol=1e-2, atol=epsilon)

  def _compare(self, input_shape, block_shape, paddings):
    x = np.random.normal(
        0, 1, np.prod(input_shape)).astype(np.float32).reshape(input_shape)
    self._checkGrad(x, block_shape, paddings)

  # Don't use very large numbers as dimensions here as the result is tensor
  # with cartesian product of the dimensions.
  def testSmall(self):
    self._compare([1, 4, 6, 5], [2, 2], [[0, 0], [0, 0]])

  def testSmall2(self):
    self._compare([2, 8, 6, 2], [2, 2], [[0, 0], [0, 0]])

  def testSmallPad1(self):
    self._compare([2, 4, 6, 2], [2, 2], [[1, 1], [1, 1]])

  def testSmallPadThreeBlockDims(self):
    self._compare([2, 2, 4, 3, 2], [2, 2, 2], [[1, 1], [1, 1], [1, 0]])


class RequiredSpaceToBatchPaddingsTest(test.TestCase):

  def _checkProperties(self, input_shape, block_shape, base_paddings, paddings,
                       crops):
    """Checks that `paddings` and `crops` satisfy invariants."""
    num_block_dims = len(block_shape)
    self.assertEqual(len(input_shape), num_block_dims)
    if base_paddings is None:
      base_paddings = np.zeros((num_block_dims, 2), np.int32)
    self.assertEqual(base_paddings.shape, (num_block_dims, 2))
    self.assertEqual(paddings.shape, (num_block_dims, 2))
    self.assertEqual(crops.shape, (num_block_dims, 2))
    for i in range(num_block_dims):
      self.assertEqual(paddings[i, 0], base_paddings[i, 0])
      self.assertLessEqual(0, paddings[i, 1] - base_paddings[i, 1])
      self.assertLess(paddings[i, 1] - base_paddings[i, 1], block_shape[i])
      self.assertEqual(
          (input_shape[i] + paddings[i, 0] + paddings[i, 1]) % block_shape[i],
          0)
      self.assertEqual(crops[i, 0], 0)
      self.assertEqual(crops[i, 1], paddings[i, 1] - base_paddings[i, 1])

  def _test(self, input_shape, block_shape, base_paddings):
    input_shape = np.array(input_shape)
    block_shape = np.array(block_shape)
    if base_paddings is not None:
      base_paddings = np.array(base_paddings)
    # Check with constants.
    paddings, crops = array_ops.required_space_to_batch_paddings(input_shape,
                                                                 block_shape,
                                                                 base_paddings)
    paddings_const = tensor_util.constant_value(paddings)
    crops_const = tensor_util.constant_value(crops)
    self.assertIsNotNone(paddings_const)
    self.assertIsNotNone(crops_const)
    self._checkProperties(input_shape, block_shape, base_paddings,
                          paddings_const, crops_const)
    # Check with non-constants.
    assignments = {}
    input_shape_placeholder = array_ops.placeholder(dtypes.int32)
    assignments[input_shape_placeholder] = input_shape
    block_shape_placeholder = array_ops.placeholder(dtypes.int32,
                                                    [len(block_shape)])
    assignments[block_shape_placeholder] = block_shape
    if base_paddings is not None:
      base_paddings_placeholder = array_ops.placeholder(dtypes.int32,
                                                        [len(block_shape), 2])
      assignments[base_paddings_placeholder] = base_paddings
    else:
      base_paddings_placeholder = None
    t_paddings, t_crops = array_ops.required_space_to_batch_paddings(
        input_shape_placeholder, block_shape_placeholder,
        base_paddings_placeholder)
    with self.cached_session():
      paddings_result = t_paddings.eval(assignments)
      crops_result = t_crops.eval(assignments)
    self.assertAllEqual(paddings_result, paddings_const)
    self.assertAllEqual(crops_result, crops_const)

  def testSimple(self):
    self._test(
        input_shape=np.zeros((0,), np.int32),
        block_shape=np.zeros((0,), np.int32),
        base_paddings=None)
    self._test(
        input_shape=np.zeros((0,), np.int32),
        block_shape=np.zeros((0,), np.int32),
        base_paddings=np.zeros((0, 2), np.int32))
    self._test(input_shape=[1], block_shape=[2], base_paddings=None)
    self._test(input_shape=[1], block_shape=[2], base_paddings=[[1, 0]])
    self._test(input_shape=[3], block_shape=[1], base_paddings=[[1, 2]])
    self._test(input_shape=[1], block_shape=[2], base_paddings=[[2, 3]])
    self._test(input_shape=[4, 5], block_shape=[3, 2], base_paddings=None)
    self._test(
        input_shape=[4, 5], block_shape=[3, 2], base_paddings=[[0, 0], [0, 1]])


if __name__ == "__main__":
  test.main()
