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
"""Tests for SparseReshape."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test


class SparseReshapeTest(test.TestCase):

  def _SparseTensorPlaceholder(self):
    return sparse_tensor.SparseTensor(
        array_ops.placeholder(dtypes.int64),
        array_ops.placeholder(dtypes.float64),
        array_ops.placeholder(dtypes.int64))

  def _SparseTensorValue_5x6(self):
    ind = np.array([[0, 0], [1, 0], [1, 3], [1, 4], [3, 2],
                    [3, 3]]).astype(np.int64)
    val = np.array([0, 10, 13, 14, 32, 33]).astype(np.float64)
    shape = np.array([5, 6]).astype(np.int64)
    return sparse_tensor.SparseTensorValue(ind, val, shape)

  def _SparseTensorValue_2x3x4(self):
    ind = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 2], [1, 0, 3], [1, 1, 1],
                    [1, 1, 3], [1, 2, 2]])
    val = np.array([1, 10, 12, 103, 111, 113, 122])
    shape = np.array([2, 3, 4])
    return sparse_tensor.SparseTensorValue(ind, val, shape)

  def testStaticShapeInfoPreserved(self):
    sp_input = sparse_tensor.SparseTensor.from_value(
        self._SparseTensorValue_5x6())
    self.assertAllEqual((5, 6), sp_input.get_shape())
    sp_output = sparse_ops.sparse_reshape(sp_input, shape=(1, 5, 2, 3))
    self.assertAllEqual((1, 5, 2, 3), sp_output.get_shape())

  def testStaticShapeInfoPreservedWithInferredDims(self):
    sp_input = sparse_tensor.SparseTensor.from_value(
        self._SparseTensorValue_2x3x4())
    self.assertAllEqual((2, 3, 4), sp_input.get_shape())
    sp_output = sparse_ops.sparse_reshape(sp_input, shape=(2, -1))
    self.assertAllEqual((2, 3 * 4), sp_output.get_shape())

  @test_util.run_deprecated_v1
  def testRaisesIfMoreThanOneInferredDim(self):
    sp_input = sparse_tensor.SparseTensor.from_value(
        self._SparseTensorValue_2x3x4())
    with self.assertRaisesRegex(ValueError, "At most one dimension can"):
      sparse_ops.sparse_reshape(sp_input, shape=(-1, 2, -1))

  @test_util.run_deprecated_v1
  def testRaisesIfInferredShapeNotPossible(self):
    sp_input = sparse_tensor.SparseTensor.from_value(
        self._SparseTensorValue_2x3x4())
    with self.assertRaisesRegex(ValueError, "Cannot reshape"):
      sparse_ops.sparse_reshape(sp_input, shape=(-1, 7))

  @test_util.run_deprecated_v1
  def testPropagatesFullyKnownDenseShapeWhenShapePartiallyKnown(self):
    sp_input = sparse_tensor.SparseTensor.from_value(
        self._SparseTensorValue_2x3x4())
    self.assertAllEqual((2, 3, 4), sp_input.shape)
    sp_output = sparse_ops.sparse_reshape(
        sp_input, shape=array_ops.concat(
            (constant_op.constant([2], dtype=dtypes.int64),
             array_ops.placeholder(dtype=dtypes.int64, shape=[1])),
            axis=0))
    self.assertAllEqual((2, 3 * 4), sp_output.shape)

  def testSameShape(self):
    with self.session() as sess:
      input_val = self._SparseTensorValue_5x6()
      sp_output = sparse_ops.sparse_reshape(input_val, [5, 6])

      output_val = self.evaluate(sp_output)
      self.assertAllEqual(output_val.indices, input_val.indices)
      self.assertAllEqual(output_val.values, input_val.values)
      self.assertAllEqual(output_val.dense_shape, input_val.dense_shape)

  @test_util.run_deprecated_v1
  def testFeedSameShape(self):
    with self.session() as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_5x6()
      sp_output = sparse_ops.sparse_reshape(sp_input, [5, 6])

      output_val = sess.run(sp_output, {sp_input: input_val})
      self.assertAllEqual(output_val.indices, input_val.indices)
      self.assertAllEqual(output_val.values, input_val.values)
      self.assertAllEqual(output_val.dense_shape, input_val.dense_shape)

  @test_util.run_deprecated_v1
  def testWorksWellWithTfShape(self):
    with self.session() as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_5x6()
      shape = array_ops.shape(sp_input)  # tf.shape generates int32 output
      sp_output = sparse_ops.sparse_reshape(sp_input, shape)

      output_val = sess.run(sp_output, {sp_input: input_val})
      self.assertAllEqual(output_val.indices, input_val.indices)
      self.assertAllEqual(output_val.values, input_val.values)
      self.assertAllEqual(output_val.dense_shape, input_val.dense_shape)

  @test_util.run_deprecated_v1
  def testFeedSameShapeWithInferredDim(self):
    with self.session() as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_5x6()
      sp_output = sparse_ops.sparse_reshape(sp_input, [-1, 6])

      output_val = sess.run(sp_output, {sp_input: input_val})
      self.assertAllEqual(output_val.indices, input_val.indices)
      self.assertAllEqual(output_val.values, input_val.values)
      self.assertAllEqual(output_val.dense_shape, input_val.dense_shape)

  @test_util.run_deprecated_v1
  def testFeedNewShapeSameRank(self):
    with self.session() as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_5x6()
      sp_output = sparse_ops.sparse_reshape(sp_input, [3, 10])

      output_val = sess.run(sp_output, {sp_input: input_val})
      self.assertAllEqual(output_val.indices,
                          np.array([[0, 0], [0, 6], [0, 9], [1, 0], [2, 0],
                                    [2, 1]]))
      self.assertAllEqual(output_val.values, input_val.values)
      self.assertAllEqual(output_val.dense_shape, [3, 10])

  @test_util.run_deprecated_v1
  def testFeedNewShapeSameRankWithInferredDim(self):
    with self.session() as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_5x6()
      sp_output = sparse_ops.sparse_reshape(sp_input, [3, -1])

      output_val = sess.run(sp_output, {sp_input: input_val})
      self.assertAllEqual(output_val.indices,
                          np.array([[0, 0], [0, 6], [0, 9], [1, 0], [2, 0],
                                    [2, 1]]))
      self.assertAllEqual(output_val.values, input_val.values)
      self.assertAllEqual(output_val.dense_shape, [3, 10])

  def testUpRank(self):
    with self.session() as sess:
      input_val = self._SparseTensorValue_5x6()
      sp_output = sparse_ops.sparse_reshape(input_val, [2, 3, 5])

      output_val = self.evaluate(sp_output)
      self.assertAllEqual(output_val.indices,
                          np.array([[0, 0, 0], [0, 1, 1], [0, 1, 4], [0, 2, 0],
                                    [1, 1, 0], [1, 1, 1]]))
      self.assertAllEqual(output_val.values, input_val.values)
      self.assertAllEqual(output_val.dense_shape, [2, 3, 5])

  @test_util.run_deprecated_v1
  def testFeedUpRank(self):
    with self.session() as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_5x6()
      sp_output = sparse_ops.sparse_reshape(sp_input, [2, 3, 5])

      output_val = sess.run(sp_output, {sp_input: input_val})
      self.assertAllEqual(output_val.indices,
                          np.array([[0, 0, 0], [0, 1, 1], [0, 1, 4], [0, 2, 0],
                                    [1, 1, 0], [1, 1, 1]]))
      self.assertAllEqual(output_val.values, input_val.values)
      self.assertAllEqual(output_val.dense_shape, [2, 3, 5])

  @test_util.run_deprecated_v1
  def testFeedUpRankWithInferredDim(self):
    with self.session() as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_5x6()
      sp_output = sparse_ops.sparse_reshape(sp_input, [2, -1, 5])

      output_val = sess.run(sp_output, {sp_input: input_val})
      self.assertAllEqual(output_val.indices,
                          np.array([[0, 0, 0], [0, 1, 1], [0, 1, 4], [0, 2, 0],
                                    [1, 1, 0], [1, 1, 1]]))
      self.assertAllEqual(output_val.values, input_val.values)
      self.assertAllEqual(output_val.dense_shape, [2, 3, 5])

  @test_util.run_deprecated_v1
  def testFeedDownRank(self):
    with self.session() as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_2x3x4()
      sp_output = sparse_ops.sparse_reshape(sp_input, [6, 4])

      output_val = sess.run(sp_output, {sp_input: input_val})
      self.assertAllEqual(output_val.indices,
                          np.array([[0, 1], [1, 0], [1, 2], [3, 3], [4, 1],
                                    [4, 3], [5, 2]]))
      self.assertAllEqual(output_val.values, input_val.values)
      self.assertAllEqual(output_val.dense_shape, [6, 4])

  @test_util.run_deprecated_v1
  def testFeedDownRankWithInferredDim(self):
    with self.session() as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_2x3x4()
      sp_output = sparse_ops.sparse_reshape(sp_input, [6, -1])

      output_val = sess.run(sp_output, {sp_input: input_val})
      self.assertAllEqual(output_val.indices,
                          np.array([[0, 1], [1, 0], [1, 2], [3, 3], [4, 1],
                                    [4, 3], [5, 2]]))
      self.assertAllEqual(output_val.values, input_val.values)
      self.assertAllEqual(output_val.dense_shape, [6, 4])

  @test_util.run_deprecated_v1
  def testFeedMultipleInferredDims(self):
    with self.session() as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_5x6()
      sp_output = sparse_ops.sparse_reshape(sp_input, [4, -1, -1])
      with self.assertRaisesOpError("only one output dimension may be -1"):
        sess.run(sp_output, {sp_input: input_val})

  @test_util.run_deprecated_v1
  def testProvideStaticallyMismatchedSizes(self):
    input_val = self._SparseTensorValue_5x6()
    sp_input = sparse_tensor.SparseTensor.from_value(input_val)
    with self.assertRaisesRegex(ValueError, "Cannot reshape"):
      sparse_ops.sparse_reshape(sp_input, [4, 7])

  @test_util.run_deprecated_v1
  def testFeedMismatchedSizes(self):
    with self.session() as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_5x6()
      sp_output = sparse_ops.sparse_reshape(sp_input, [4, 7])
      with self.assertRaisesOpError(
          "Input to reshape is a tensor with 30 dense values"):
        sess.run(sp_output, {sp_input: input_val})

  @test_util.run_deprecated_v1
  def testFeedMismatchedSizesWithInferredDim(self):
    with self.session() as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_5x6()
      sp_output = sparse_ops.sparse_reshape(sp_input, [4, -1])
      with self.assertRaisesOpError("requested shape requires a multiple"):
        sess.run(sp_output, {sp_input: input_val})

  @test_util.run_deprecated_v1
  def testFeedPartialShapes(self):
    with self.session():
      # Incorporate new rank into shape information if known
      sp_input = self._SparseTensorPlaceholder()
      sp_output = sparse_ops.sparse_reshape(sp_input, [2, 3, 5])
      self.assertListEqual(sp_output.indices.get_shape().as_list(), [None, 3])
      self.assertListEqual(sp_output.dense_shape.get_shape().as_list(), [3])

      # Incorporate known shape information about input indices in output
      # indices
      sp_input = self._SparseTensorPlaceholder()
      sp_input.indices.set_shape([5, None])
      sp_output = sparse_ops.sparse_reshape(sp_input, [2, 3, 5])
      self.assertListEqual(sp_output.indices.get_shape().as_list(), [5, 3])
      self.assertListEqual(sp_output.dense_shape.get_shape().as_list(), [3])

      # Even if new_shape has no shape information, we know the ranks of
      # output indices and shape
      sp_input = self._SparseTensorPlaceholder()
      sp_input.indices.set_shape([5, None])
      new_shape = array_ops.placeholder(dtypes.int64)
      sp_output = sparse_ops.sparse_reshape(sp_input, new_shape)
      self.assertListEqual(sp_output.indices.get_shape().as_list(), [5, None])
      self.assertListEqual(sp_output.dense_shape.get_shape().as_list(), [None])

  @test_util.run_deprecated_v1
  def testFeedDenseReshapeSemantics(self):
    with self.session() as sess:
      # Compute a random rank-5 initial shape and new shape, randomly sparsify
      # it, and check that the output of SparseReshape has the same semantics
      # as a dense reshape.
      factors = np.array([2] * 4 + [3] * 4 + [5] * 4)  # 810k total elements
      orig_rank = np.random.randint(2, 7)
      orig_map = np.random.randint(orig_rank, size=factors.shape)
      orig_shape = [np.prod(factors[orig_map == d]) for d in range(orig_rank)]
      new_rank = np.random.randint(2, 7)
      new_map = np.random.randint(new_rank, size=factors.shape)
      new_shape = [np.prod(factors[new_map == d]) for d in range(new_rank)]

      orig_dense = np.random.uniform(size=orig_shape)
      orig_indices = np.transpose(np.nonzero(orig_dense < 0.5))
      orig_values = orig_dense[orig_dense < 0.5]

      new_dense = np.reshape(orig_dense, new_shape)
      new_indices = np.transpose(np.nonzero(new_dense < 0.5))
      new_values = new_dense[new_dense < 0.5]

      sp_input = self._SparseTensorPlaceholder()
      input_val = sparse_tensor.SparseTensorValue(orig_indices, orig_values,
                                                  orig_shape)
      sp_output = sparse_ops.sparse_reshape(sp_input, new_shape)

      output_val = sess.run(sp_output, {sp_input: input_val})
      self.assertAllEqual(output_val.indices, new_indices)
      self.assertAllEqual(output_val.values, new_values)
      self.assertAllEqual(output_val.dense_shape, new_shape)


class EmptySparseTensorReshapeTest(test.TestCase, parameterized.TestCase):
  """Tests for reshaping 0-sized SparseTensors, compared w/ dense tensors."""

  def _MakeAndReshapeTensor(self, tensor_class, original_shape, target_shape):
    if tensor_class == "sparse":
      ind = np.zeros([0, len(original_shape)]).astype(np.int64)
      val = np.array([]).astype(np.float64)
      shape = np.array(original_shape).astype(np.int64)
      sp_input = sparse_tensor.SparseTensorValue(ind, val, shape)
      sp_output = self.evaluate(
          sparse_ops.sparse_reshape(sp_input, target_shape))
      return sp_output.dense_shape
    else:
      dense_input = array_ops.zeros(original_shape)
      dense_output = self.evaluate(array_ops.reshape(dense_input, target_shape))
      return dense_output.shape

  @parameterized.named_parameters([
      ("Dense", "dense"),
      ("Sparse", "sparse"),
  ])
  def testImpliedReshapeEmpty1DTensor(self, tensor_class):
    self.assertAllEqual(
        self._MakeAndReshapeTensor(tensor_class, [0], [-1, 1]), [0, 1])
    self.assertAllEqual(
        self._MakeAndReshapeTensor(tensor_class, [0], [-1, 1, 2]), [0, 1, 2])

  @parameterized.named_parameters([
      ("Dense", "dense"),
      ("Sparse", "sparse"),
  ])
  def testImpliedReshapeEmpty2DTensor(self, tensor_class):
    self.assertAllEqual(
        self._MakeAndReshapeTensor(tensor_class, [1, 0], [-1, 1]), [0, 1])
    self.assertAllEqual(
        self._MakeAndReshapeTensor(tensor_class, [1, 0], [-1, 2, 3]), [0, 2, 3])

  @parameterized.named_parameters([
      ("Dense", "dense"),
      ("Sparse", "sparse"),
  ])
  def testImpliedReshapeEmpty3DTensor(self, tensor_class):
    self.assertAllEqual(
        self._MakeAndReshapeTensor(tensor_class, [1, 0, 0], [-1, 2, 3]),
        [0, 2, 3])

  @parameterized.named_parameters([
      ("Dense", "dense"),
      ("Sparse", "sparse"),
  ])
  def testImpliedReshapeEmpty4DTensor(self, tensor_class):
    self.assertAllEqual(
        self._MakeAndReshapeTensor(tensor_class, [2, 4, 0, 6], [-1, 4, 6, 2]),
        [0, 4, 6, 2])

  def testImpliedDimTogetherWithZeroDimCausesError(self):
    # NOTE: When implied dimensions and zero dimensions coexist in the target
    # shape, the behavior currently differs between sparse and regular tensors.
    with self.assertRaises(errors.InvalidArgumentError):
      self._MakeAndReshapeTensor("sparse", [0], [-1, 0])
    with self.assertRaises(errors.InvalidArgumentError):
      self._MakeAndReshapeTensor("sparse", [1, 0], [-1, 0])
    with self.assertRaises(errors.InvalidArgumentError):
      self._MakeAndReshapeTensor("sparse", [1, 2, 0], [2, -1, 0])
    with self.assertRaises(errors.InvalidArgumentError):
      self._MakeAndReshapeTensor("sparse", [1, 2, 3, 0], [2, 0, -1, 3])


if __name__ == "__main__":
  test.main()
