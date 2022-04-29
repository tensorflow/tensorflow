# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
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


class SpaceToBatchTest(xla_test.XLATestCase):
  """Tests input-output pairs for the SpaceToBatch and BatchToSpace ops."""

  def _testPad(self, inputs, paddings, block_size, outputs):
    with self.session() as sess, self.test_scope():
      for dtype in self.float_types:
        # outputs = space_to_batch(inputs)
        placeholder = array_ops.placeholder(dtype)
        x_tf = gen_array_ops.space_to_batch(
            placeholder, paddings, block_size=block_size)
        self.assertAllEqual(sess.run(x_tf, {placeholder: inputs}), outputs)
        # inputs = batch_to_space(outputs)
        x_tf = gen_array_ops.batch_to_space(
            placeholder, paddings, block_size=block_size)
        self.assertAllEqual(sess.run(x_tf, {placeholder: outputs}), inputs)

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


class SpaceToBatchNDErrorHandlingTest(xla_test.XLATestCase):

  def testInvalidBlockShape(self):
    with self.assertRaisesRegex(ValueError, "block_shape must be positive"):
      with self.session() as sess, self.test_scope():
        tf_in = constant_op.constant(
            -3.5e+35, shape=[10, 20, 20], dtype=dtypes.float32)
        block_shape = constant_op.constant(-10, shape=[2], dtype=dtypes.int64)
        paddings = constant_op.constant(0, shape=[2, 2], dtype=dtypes.int32)
        sess.run(array_ops.space_to_batch_nd(tf_in, block_shape, paddings))

  def testOutputSizeOutOfBounds(self):
    with self.assertRaisesRegex(ValueError,
                                "Negative.* dimension size caused by overflow"):
      with self.session() as sess, self.test_scope():
        tf_in = constant_op.constant(
            -3.5e+35, shape=[10, 19, 22], dtype=dtypes.float32)
        block_shape = constant_op.constant(
            1879048192, shape=[2], dtype=dtypes.int64)
        paddings = constant_op.constant(0, shape=[2, 2], dtype=dtypes.int32)
        sess.run(array_ops.space_to_batch_nd(tf_in, block_shape, paddings))


class SpaceToBatchNDTest(xla_test.XLATestCase):
  """Tests input-output pairs for the SpaceToBatchND and BatchToSpaceND ops."""

  def _testPad(self, inputs, block_shape, paddings, outputs):
    block_shape = np.array(block_shape)
    paddings = np.array(paddings).reshape((len(block_shape), 2))
    with self.session() as sess, self.test_scope():
      for dtype in self.float_types:
        # TODO(b/68813416): Skip bfloat16's as the input type for direct is
        # float32 and results in a mismatch, while making testDirect provide the
        # correctly typed input results in 'no fill-function for data-type'
        # error.
        if dtype == dtypes.bfloat16.as_numpy_dtype:
          continue
        if dtype == np.float16:
          actual_inputs = np.array(inputs).astype(dtype)
          actual_paddings = np.array(paddings).astype(dtype)
          expected_outputs = np.array(outputs).astype(dtype)
        else:
          actual_inputs = inputs
          actual_paddings = paddings
          expected_outputs = outputs
        placeholder = array_ops.placeholder(dtype)
        # outputs = space_to_batch(inputs)
        x_tf = array_ops.space_to_batch_nd(placeholder, block_shape,
                                           actual_paddings)
        self.assertAllEqual(
            sess.run(x_tf, {placeholder: actual_inputs}), expected_outputs)
        # inputs = batch_to_space(outputs)
        placeholder = array_ops.placeholder(dtype)
        x_tf = array_ops.batch_to_space_nd(placeholder, block_shape,
                                           actual_paddings)
        self.assertAllEqual(
            sess.run(x_tf, {placeholder: expected_outputs}), actual_inputs)

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

  def testDirect0(self):
    # Test with zero-size remaining dimension.
    self._testDirect(
        input_shape=[3, 1, 2, 0], block_shape=[3], paddings=[[0, 2]])

  def testDirect1(self):
    # Test with zero-size blocked dimension.
    self._testDirect(
        input_shape=[3, 0, 2, 5], block_shape=[3], paddings=[[0, 0]])

  def testDirect2(self):
    # Test with padding up from zero size.
    self._testDirect(
        input_shape=[3, 0, 2, 5], block_shape=[3], paddings=[[1, 2]])

  def testDirect3(self):
    self._testDirect(
        input_shape=[3, 3, 4, 5, 2],
        block_shape=[3, 4, 2],
        paddings=[[1, 2], [0, 0], [3, 0]])

  def testDirect4(self):
    self._testDirect(
        input_shape=[3, 3, 4, 5, 2],
        block_shape=[3, 4, 2, 2],
        paddings=[[1, 2], [0, 0], [3, 0], [0, 0]])

  def testDirect5(self):
    self._testDirect(
        input_shape=[3, 2, 2, 3, 4, 5, 2, 5],
        block_shape=[1, 1, 3, 4, 2, 2],
        paddings=[[0, 0], [0, 0], [1, 2], [0, 0], [3, 0], [0, 0]])

  def testDirect6(self):
    self._testDirect(
        input_shape=[3, 2, 2, 3, 4, 5, 2, 5],
        block_shape=[1, 1, 3, 4, 2, 2, 1],
        paddings=[[0, 0], [0, 0], [1, 2], [0, 0], [3, 0], [0, 0], [0, 0]])


if __name__ == "__main__":
  test.main()
