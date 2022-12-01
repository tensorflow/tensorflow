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
"""Tests for Python ops defined in sparse_ops."""

import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
import tensorflow.python.ops.sparse_grad  # pylint: disable=unused-import
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test


# TODO(zongheng): it'd be great to factor out this function and various random
# SparseTensor gen funcs.
def _sparsify(x, thresh=0.5, index_dtype=np.int64):
  x[x < thresh] = 0

  non_zero = np.where(x)
  x_indices = np.vstack(non_zero).astype(index_dtype).T
  x_values = x[non_zero]
  x_shape = x.shape

  return sparse_tensor.SparseTensor(
      indices=x_indices, values=x_values, dense_shape=x_shape), len(x_values)


class SparseToIndicatorTest(test_util.TensorFlowTestCase):

  def _SparseTensor_5x6(self, dtype):
    ind = np.array([[0, 0], [1, 0], [1, 3], [1, 4], [3, 2], [3, 3]])
    val = np.array([0, 10, 13, 14, 32, 33])
    shape = np.array([5, 6])
    return sparse_tensor.SparseTensor(
        constant_op.constant(ind, dtypes.int64),
        constant_op.constant(val, dtype),
        constant_op.constant(shape, dtypes.int64))

  def _SparseTensor_2x3x4(self, dtype):
    # Includes two entries with the form [1, 1, x] : 150.
    ind = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 2], [1, 0, 3], [1, 1, 0],
                    [1, 1, 1], [1, 1, 2], [1, 2, 2]])
    val = np.array([1, 10, 12, 103, 150, 149, 150, 122])
    shape = np.array([2, 3, 4])
    return sparse_tensor.SparseTensor(
        constant_op.constant(ind, dtypes.int64),
        constant_op.constant(val, dtype),
        constant_op.constant(shape, dtypes.int64))

  def testInt32(self):
    with test_util.force_cpu():
      sp_input = self._SparseTensor_5x6(dtypes.int32)
      output = sparse_ops.sparse_to_indicator(sp_input, 50)

      expected_output = np.zeros((5, 50), dtype=np.bool_)
      expected_trues = ((0, 0), (1, 10), (1, 13), (1, 14), (3, 32), (3, 33))
      for expected_true in expected_trues:
        expected_output[expected_true] = True

      self.assertAllEqual(output, expected_output)

  def testInt64(self):
    with test_util.force_cpu():
      sp_input = self._SparseTensor_5x6(dtypes.int64)
      output = sparse_ops.sparse_to_indicator(sp_input, 50)

      expected_output = np.zeros((5, 50), dtype=np.bool_)
      expected_trues = [(0, 0), (1, 10), (1, 13), (1, 14), (3, 32), (3, 33)]
      for expected_true in expected_trues:
        expected_output[expected_true] = True

      self.assertAllEqual(output, expected_output)

  def testHigherRank(self):
    with test_util.force_cpu():
      sp_input = self._SparseTensor_2x3x4(dtypes.int64)
      output = sparse_ops.sparse_to_indicator(sp_input, 200)

      expected_output = np.zeros((2, 3, 200), dtype=np.bool_)
      expected_trues = [(0, 0, 1), (0, 1, 10), (0, 1, 12), (1, 0, 103),
                        (1, 1, 149), (1, 1, 150), (1, 2, 122)]
      for expected_true in expected_trues:
        expected_output[expected_true] = True

      self.assertAllEqual(output, expected_output)


class SparseMergeTest(test_util.TensorFlowTestCase):

  def _SparseTensorValue_3x50(self, indices_dtype, values_dtype):
    # NOTE: This input is intentionally not sorted to validate the
    # already_sorted flag below.
    ind = np.array([[0, 0], [1, 0], [1, 2], [2, 0], [2, 1], [1, 1]])
    # NB: these are not sorted
    indices = np.array([0, 13, 10, 33, 32, 14])
    values = np.array([-3, 4, 1, 9, 5, 1])
    shape = np.array([3, 3])
    indices = sparse_tensor.SparseTensorValue(
        np.array(ind, np.int64),
        np.array(indices, indices_dtype), np.array(shape, np.int64))
    values = sparse_tensor.SparseTensorValue(
        np.array(ind, np.int64),
        np.array(values, values_dtype), np.array(shape, np.int64))
    return indices, values

  def _SparseTensor_3x50(self, indices_dtype, values_dtype):
    indices, values = self._SparseTensorValue_3x50(indices_dtype, values_dtype)
    return (sparse_tensor.SparseTensor.from_value(indices),
            sparse_tensor.SparseTensor.from_value(values))

  def _AssertResultsSorted(self, output, vocab_size):
    self.assertAllEqual(output.indices,
                        [[0, 0], [1, 10], [1, 13], [1, 14], [2, 32], [2, 33]])
    self.assertAllEqual(output.values, [-3, 1, 4, 1, 5, 9])
    self.assertAllEqual(output.dense_shape, [3, vocab_size])

  def _AssertResultsNotSorted(self, output, vocab_size):
    self.assertAllEqual(output.indices,
                        [[0, 0], [1, 13], [1, 10], [2, 33], [2, 32], [1, 14]])
    self.assertAllEqual(output.values, [-3, 4, 1, 9, 5, 1])
    self.assertAllEqual(output.dense_shape, [3, vocab_size])

  def testInt32AndFloat32(self):
    vocab_size = 50
    indices_v, values_v = self._SparseTensorValue_3x50(np.int32, np.float32)
    with test_util.force_cpu():
      for indices in (indices_v,
                      sparse_tensor.SparseTensor.from_value(indices_v)):
        for values in (values_v,
                       sparse_tensor.SparseTensor.from_value(values_v)):
          sp_output = sparse_ops.sparse_merge(indices, values, vocab_size)

          output = self.evaluate(sp_output)
          self._AssertResultsSorted(output, vocab_size)

  def testInt64AndFloat32(self):
    vocab_size = 50
    with test_util.force_cpu():
      indices, values = self._SparseTensor_3x50(np.int64, np.float32)
      sp_output = sparse_ops.sparse_merge(indices, values, vocab_size)

      output = self.evaluate(sp_output)
      self._AssertResultsSorted(output, vocab_size)

  def testInt64AndFloat64(self):
    vocab_size = 50
    with test_util.force_cpu():
      indices, values = self._SparseTensor_3x50(np.int64, np.float64)
      sp_output = sparse_ops.sparse_merge(indices, values, vocab_size)

      output = self.evaluate(sp_output)
      self._AssertResultsSorted(output, vocab_size)

  def testInt32AndFloat32NonCanonicalOrder(self):
    vocab_size = 50
    with test_util.force_cpu():
      indices, values = self._SparseTensor_3x50(np.int32, np.float32)
      sp_output = sparse_ops.sparse_merge(
          indices, values, vocab_size, already_sorted=True)

      output = self.evaluate(sp_output)
      self._AssertResultsNotSorted(output, vocab_size)

  def testInt64AndFloat32NonCanonicalOrder(self):
    vocab_size = 50
    with test_util.force_cpu():
      indices, values = self._SparseTensor_3x50(np.int64, np.float32)
      sp_output = sparse_ops.sparse_merge(
          indices, values, vocab_size, already_sorted=True)

      output = self.evaluate(sp_output)
      self._AssertResultsNotSorted(output, vocab_size)

  def testInt64AndFloat64NonCanonicalOrder(self):
    vocab_size = 50
    vocab_size_tensor = constant_op.constant(vocab_size, dtypes.int64)
    with test_util.force_cpu():
      indices, values = self._SparseTensor_3x50(np.int64, np.float64)
      sp_output = sparse_ops.sparse_merge(
          indices, values, vocab_size_tensor, already_sorted=True)

      output = self.evaluate(sp_output)
      self._AssertResultsNotSorted(output, vocab_size)

  def testShouldSetLastDimensionInDynamicShape(self):
    with ops.Graph().as_default():
      shape = constant_op.constant([2, 2], dtype=dtypes.int64)
      dynamic_shape = array_ops.placeholder_with_default(shape, shape=[2])
      ids = sparse_tensor.SparseTensor(
          indices=[[0, 0], [0, 1]],
          values=[1, 3],
          dense_shape=dynamic_shape)
      values = sparse_tensor.SparseTensor(
          indices=[[0, 0], [0, 1]],
          values=[0.4, 0.7],
          dense_shape=dynamic_shape)
      merged = sparse_ops.sparse_merge(
          sp_ids=ids, sp_values=values, vocab_size=5)
      self.assertEqual(5, merged.get_shape()[1])


class SparseMergeHighDimTest(test_util.TensorFlowTestCase):

  def _SparseTensor_3x50(self, indices_dtype, values_dtype):
    # NOTE: This input is intentionally not sorted to validate the
    # already_sorted flag below.
    ind = np.array([[0, 0], [1, 0], [1, 2], [2, 0], [2, 1], [1, 1]])
    # NB: these are not sorted
    indices0 = np.array([0, 13, 10, 33, 32, 14])
    indices1 = np.array([12, 4, 0, 0, 1, 30])
    values = np.array([-3, 4, 1, 9, 5, 1])
    shape = np.array([3, 3])
    indices0 = sparse_tensor.SparseTensorValue(
        np.array(ind, np.int64),
        np.array(indices0, indices_dtype), np.array(shape, np.int64))
    indices1 = sparse_tensor.SparseTensorValue(
        np.array(ind, np.int64),
        np.array(indices1, indices_dtype), np.array(shape, np.int64))
    values = sparse_tensor.SparseTensorValue(
        np.array(ind, np.int64),
        np.array(values, values_dtype), np.array(shape, np.int64))
    return ([sparse_tensor.SparseTensor.from_value(indices0),
             sparse_tensor.SparseTensor.from_value(indices1)],
            sparse_tensor.SparseTensor.from_value(values))

  def _AssertResultsSorted(self, output, vocab_size):
    self.assertAllEqual(
        output.indices,
        [[0, 0, 12], [1, 10, 0], [1, 13, 4], [1, 14, 30], [2, 32, 1],
         [2, 33, 0]])
    self.assertAllEqual(output.values, [-3, 1, 4, 1, 5, 9])
    self.assertAllEqual(output.dense_shape, [3] + vocab_size)

  def testInt64AndFloat32(self):
    vocab_size = [50, 31]
    with test_util.force_cpu():
      indices, values = self._SparseTensor_3x50(np.int64, np.float32)
      sp_output = sparse_ops.sparse_merge(indices, values, vocab_size)

      output = self.evaluate(sp_output)
      self._AssertResultsSorted(output, vocab_size)

  def testInt64AndFloat64(self):
    vocab_size = [50, 31]
    with test_util.force_cpu():
      indices, values = self._SparseTensor_3x50(np.int64, np.float64)
      sp_output = sparse_ops.sparse_merge(indices, values, vocab_size)

      output = self.evaluate(sp_output)
      self._AssertResultsSorted(output, vocab_size)

  def testInt64AndFloat64Shape(self):
    vocab_size = [50, 30]
    with test_util.force_cpu():
      indices, values = self._SparseTensor_3x50(np.int64, np.float64)
      sp_output = sparse_ops.sparse_merge(indices, values, vocab_size)

      output = self.evaluate(sp_output)
      self._AssertResultsSorted(output, vocab_size)


class SparseRetainTest(test_util.TensorFlowTestCase):

  def _SparseTensorValue_5x6(self):
    ind = np.array([[0, 0], [1, 0], [1, 3], [1, 4], [3, 2], [3, 3]])
    val = np.array([0, 10, 13, 14, 32, 33])
    shape = np.array([5, 6])
    return sparse_tensor.SparseTensorValue(
        np.array(ind, np.int64),
        np.array(val, np.int32), np.array(shape, np.int64))

  def _SparseTensor_5x6(self):
    return sparse_tensor.SparseTensor.from_value(self._SparseTensorValue_5x6())

  def testBasic(self):
    with test_util.force_cpu():
      for sp_input in (self._SparseTensorValue_5x6(), self._SparseTensor_5x6()):
        to_retain = np.array([1, 0, 0, 1, 1, 0], dtype=np.bool_)
        sp_output = sparse_ops.sparse_retain(sp_input, to_retain)

        output = self.evaluate(sp_output)

        self.assertAllEqual(output.indices, [[0, 0], [1, 4], [3, 2]])
        self.assertAllEqual(output.values, [0, 14, 32])
        self.assertAllEqual(output.dense_shape, [5, 6])

  def testRetainNone(self):
    with test_util.force_cpu():
      sp_input = self._SparseTensor_5x6()
      to_retain = np.zeros((6,), dtype=np.bool_)
      sp_output = sparse_ops.sparse_retain(sp_input, to_retain)

      output = self.evaluate(sp_output)

      self.assertAllEqual(output.indices, np.array([]).reshape((0, 2)))
      self.assertAllEqual(output.values, [])
      self.assertAllEqual(output.dense_shape, [5, 6])

  def testMismatchedRetainShape(self):
    with test_util.force_cpu():
      sp_input = self._SparseTensor_5x6()
      to_retain = np.array([1, 0, 0, 1, 0], dtype=np.bool_)
      with self.assertRaises(ValueError):
        sparse_ops.sparse_retain(sp_input, to_retain)


class SparseResetShapeTest(test_util.TensorFlowTestCase):

  _IND_2_5_6 = np.array(
      [[0, 0, 0], [0, 1, 0], [0, 1, 3], [1, 1, 4], [1, 3, 2], [1, 3, 3]],
      dtype=np.int64)
  _VAL_2_5_6 = np.array([0, 10, 13, 14, 32, 33], dtype=np.int32)
  _SHP_2_5_6 = np.array([2, 5, 6], dtype=np.int64)

  def _SparseTensor_2x5x6(self):
    return sparse_tensor.SparseTensor(
        constant_op.constant(self._IND_2_5_6, dtypes.int64),
        constant_op.constant(self._VAL_2_5_6, dtypes.int32),
        constant_op.constant(self._SHP_2_5_6, dtypes.int64))

  def _SparseTensor_2x5x6_Empty(self):
    return sparse_tensor.SparseTensor(
        constant_op.constant(
            np.empty(shape=[0, 3], dtype=np.int64), dtypes.int64),
        constant_op.constant(np.empty(shape=[0], dtype=np.int32), dtypes.int32),
        constant_op.constant(self._SHP_2_5_6, dtypes.int64))

  def _SparseTensorValue_2x5x6(self):
    return sparse_tensor.SparseTensorValue(self._IND_2_5_6, self._VAL_2_5_6,
                                           self._SHP_2_5_6)

  def testStaticShapeInfoPreservedWhenNewShapeIsProvidedAndStatic(self):
    sp_input = self._SparseTensor_2x5x6()
    new_shape = np.array([3, 6, 7], dtype=np.int64)
    sp_output = sparse_ops.sparse_reset_shape(sp_input, new_shape)
    self.assertAllEqual([3, 6, 7], sp_output.get_shape())

  def testBasic(self):
    with test_util.force_cpu():
      sp_input = self._SparseTensor_2x5x6()
      new_shape = np.array([3, 6, 7], dtype=np.int64)
      sp_output = sparse_ops.sparse_reset_shape(sp_input, new_shape)

      output = self.evaluate(sp_output)

      self.assertAllEqual(output.indices, [[0, 0, 0], [0, 1, 0], [0, 1, 3],
                                           [1, 1, 4], [1, 3, 2], [1, 3, 3]])
      self.assertAllEqual(output.values, [0, 10, 13, 14, 32, 33])
      self.assertAllEqual(output.dense_shape, [3, 6, 7])

  def testInputUnavailableInGraphConstructionOk(self):
    with test_util.force_cpu():
      sp_input = self._SparseTensorValue_2x5x6()
      new_shape = np.array([3, 6, 7], dtype=np.int64)
      sp_output = sparse_ops.sparse_reset_shape(sp_input, new_shape)

      output = self.evaluate(sp_output)

      self.assertAllEqual(output.indices, [[0, 0, 0], [0, 1, 0], [0, 1, 3],
                                           [1, 1, 4], [1, 3, 2], [1, 3, 3]])
      self.assertAllEqual(output.values, [0, 10, 13, 14, 32, 33])
      self.assertAllEqual(output.dense_shape, [3, 6, 7])

  @test_util.run_deprecated_v1
  def testFeedInputUnavailableInGraphConstructionOk(self):
    with self.session(use_gpu=False) as sess:
      sp_input = array_ops.sparse_placeholder(dtype=dtypes.int32)
      new_shape = np.array([3, 6, 7], dtype=np.int64)
      sp_output = sparse_ops.sparse_reset_shape(sp_input, new_shape)

      output = sess.run(sp_output,
                        feed_dict={sp_input: self._SparseTensorValue_2x5x6()})

      self.assertAllEqual(output.indices, [[0, 0, 0], [0, 1, 0], [0, 1, 3],
                                           [1, 1, 4], [1, 3, 2], [1, 3, 3]])
      self.assertAllEqual(output.values, [0, 10, 13, 14, 32, 33])
      self.assertAllEqual(output.dense_shape, [3, 6, 7])

  def testTightBoundingBox(self):
    with test_util.force_cpu():
      sp_input = self._SparseTensor_2x5x6()
      sp_output = sparse_ops.sparse_reset_shape(sp_input)

      output = self.evaluate(sp_output)

      self.assertAllEqual(output.indices, [[0, 0, 0], [0, 1, 0], [0, 1, 3],
                                           [1, 1, 4], [1, 3, 2], [1, 3, 3]])
      self.assertAllEqual(output.values, [0, 10, 13, 14, 32, 33])
      self.assertAllEqual(output.dense_shape, [2, 4, 5])

  def testTightBoundingBoxEmpty(self):
    with test_util.force_cpu():
      sp_input = self._SparseTensor_2x5x6_Empty()
      sp_output = sparse_ops.sparse_reset_shape(sp_input)

      output = self.evaluate(sp_output)

      self.assertAllEqual(output.indices.shape, [0, 3])
      self.assertAllEqual(output.values.shape, [0])
      self.assertAllEqual(output.dense_shape, [0, 0, 0])

  def testInvalidRank(self):
    with test_util.force_cpu():
      sp_input = self._SparseTensor_2x5x6()
      new_shape = np.array([3, 7], dtype=np.int64)

      with self.assertRaises(ValueError):
        sparse_ops.sparse_reset_shape(sp_input, new_shape)

  @test_util.run_deprecated_v1
  def testInvalidRankNewShapeUnavailableInGraphConstruction(self):
    with self.session(use_gpu=False) as sess:
      new_shape = array_ops.placeholder(dtype=dtypes.int64)
      sp_input = self._SparseTensor_2x5x6()
      out = sparse_ops.sparse_reset_shape(sp_input, new_shape)

      with self.assertRaisesOpError("x == y did not hold element-wise"):
        sess.run(out, feed_dict={new_shape: np.array([3, 7], dtype=np.int64)})

  def testInvalidDimensionSizeStatic(self):
    sp_input = self._SparseTensor_2x5x6()
    new_shape = np.array([3, 7, 5], dtype=np.int64)

    with self.assertRaisesRegex(ValueError, "should have dimension sizes"):
      sparse_ops.sparse_reset_shape(sp_input, new_shape)

  @test_util.run_deprecated_v1
  def testInvalidDimensionSizeDynamic(self):
    with self.session(use_gpu=False) as sess:
      sp_input = self._SparseTensor_2x5x6()
      new_shape = array_ops.placeholder(dtype=dtypes.int32)
      out = sparse_ops.sparse_reset_shape(sp_input, new_shape)

      with self.assertRaisesOpError("x <= y did not hold element-wise"):
        sess.run(out, feed_dict={new_shape: [3, 7, 5]})

  @test_util.run_deprecated_v1
  def testInvalidDimensionSizeInputUnavailableInGraphConstruction(self):
    sp_input = array_ops.sparse_placeholder(dtype=dtypes.int32)
    with self.session(use_gpu=False) as sess:
      new_shape = np.array([3, 7, 5], dtype=np.int64)
      out = sparse_ops.sparse_reset_shape(sp_input, new_shape)

      with self.assertRaisesOpError("x <= y did not hold element-wise"):
        sess.run(out, feed_dict={sp_input: self._SparseTensorValue_2x5x6()})


class SparseSetShapeTest(test_util.TensorFlowTestCase):

  def testSetShapeEagerValidates(self):
    ind = np.array([[0, 0], [1, 0], [1, 3], [1, 4], [3, 2], [3, 3]])
    val = np.array([0, 10, 13, 14, 32, 33])
    shape = np.array([5, 6])
    sp = sparse_tensor.SparseTensor(
        constant_op.constant(ind, dtypes.int64),
        constant_op.constant(val, dtypes.int64),
        constant_op.constant(shape, dtypes.int64))

    self.assertEqual(sp.shape, tensor_shape.TensorShape([5, 6]))

    sp.set_shape(tensor_shape.TensorShape(None))
    sp.set_shape(tensor_shape.TensorShape([None, None]))
    sp.set_shape(tensor_shape.TensorShape([5, None]))
    sp.set_shape(tensor_shape.TensorShape([None, 6]))
    sp.set_shape(tensor_shape.TensorShape([5, 6]))

    with self.assertRaises(ValueError):
      sp.set_shape([None, None, None])

    with self.assertRaises(ValueError):
      sp.set_shape([3, None])

    with self.assertRaises(ValueError):
      sp.set_shape([None, 7])

    with self.assertRaises(ValueError):
      sp.set_shape([3, 6])

  def testSetShapeFunctionMerges(self):

    @def_function.function
    def dynamic_shape_sparse(dense_shape):
      ind = np.array([[0, 0], [1, 0], [1, 3], [1, 4], [3, 2], [3, 3]])
      val = np.array([0, 10, 13, 14, 32, 33])
      sp = sparse_tensor.SparseTensor(
          constant_op.constant(ind, dtypes.int64),
          constant_op.constant(val, dtypes.int64),
          dense_shape)

      sp.set_shape(tensor_shape.TensorShape(None))
      self.assertEqual(sp.shape, tensor_shape.TensorShape(None))

      sp.set_shape(tensor_shape.TensorShape([None, None]))
      self.assertEqual(sp.shape, tensor_shape.TensorShape([None, None]))

      sp.set_shape(tensor_shape.TensorShape([5, None]))
      self.assertEqual(sp.shape, tensor_shape.TensorShape([5, None]))

      sp.set_shape(tensor_shape.TensorShape([None, 6]))
      self.assertEqual(sp.shape, tensor_shape.TensorShape([5, 6]))

      sp.set_shape(tensor_shape.TensorShape([None, None]))
      self.assertEqual(sp.shape, tensor_shape.TensorShape([5, 6]))

      sp.set_shape(tensor_shape.TensorShape([5, 6]))
      self.assertEqual(sp.shape, tensor_shape.TensorShape([5, 6]))

      with self.assertRaises(ValueError):
        sp.set_shape([None, None, None])

      with self.assertRaises(ValueError):
        sp.set_shape([3, None])

      with self.assertRaises(ValueError):
        sp.set_shape([None, 7])

      with self.assertRaises(ValueError):
        sp.set_shape([3, 6])

    dense_shape_spec = tensor_spec.TensorSpec(None, dtypes.int64)
    _ = dynamic_shape_sparse.get_concrete_function(dense_shape_spec)


class SparseFillEmptyRowsTest(test_util.TensorFlowTestCase):

  def _SparseTensorValue_5x6(self, dtype=np.int32):
    ind = np.array([[0, 0], [1, 0], [1, 3], [1, 4], [3, 2], [3, 3]])
    val = np.array([0, 10, 13, 14, 32, 33])
    shape = np.array([5, 6])
    return sparse_tensor.SparseTensorValue(
        np.array(ind, np.int64), np.array(val, dtype), np.array(
            shape, np.int64))

  def _SparseTensor_5x6(self):
    return sparse_tensor.SparseTensor.from_value(self._SparseTensorValue_5x6())

  def _SparseTensor_String5x6(self):
    ind = np.array([[0, 0], [1, 0], [1, 3], [1, 4], [3, 2], [3, 3]])
    val = np.array(["a", "b", "c", "d", "e", "f"])
    shape = np.array([5, 6])
    return sparse_tensor.SparseTensor(
        constant_op.constant(ind, dtypes.int64),
        constant_op.constant(val, dtypes.string),
        constant_op.constant(shape, dtypes.int64))

  def _SparseTensor_2x6(self):
    ind = np.array([[0, 0], [1, 0], [1, 3], [1, 4]])
    val = np.array([0, 10, 13, 14])
    shape = np.array([2, 6])
    return sparse_tensor.SparseTensor(
        constant_op.constant(ind, dtypes.int64),
        constant_op.constant(val, dtypes.int32),
        constant_op.constant(shape, dtypes.int64))

  def testFillNumber(self):
    with test_util.use_gpu():
      for sp_input in (self._SparseTensorValue_5x6(), self._SparseTensor_5x6()):
        sp_output, empty_row_indicator = (
            sparse_ops.sparse_fill_empty_rows(sp_input, -1))

        output, empty_row_indicator_out = self.evaluate(
            [sp_output, empty_row_indicator])

        self.assertAllEqual(
            output.indices,
            [[0, 0], [1, 0], [1, 3], [1, 4], [2, 0], [3, 2], [3, 3], [4, 0]])
        self.assertAllEqual(output.values, [0, 10, 13, 14, -1, 32, 33, -1])
        self.assertAllEqual(output.dense_shape, [5, 6])
        self.assertAllEqual(empty_row_indicator_out,
                            np.array([0, 0, 1, 0, 1]).astype(np.bool_))

  def testSparseFillEmptyRowsGradEmpty(self):
    with test_util.use_gpu():
      grad, _ = self.evaluate(
          sparse_ops.sparse_fill_empty_rows_grad(
              reverse_index_map=[], grad_values=[]))
      self.assertAllEqual(grad, [])

  @test_util.run_deprecated_v1
  def testFillFloat(self):
    with self.session():
      values = constant_op.constant(
          [0.0, 10.0, 13.0, 14.0, 32.0, 33.0], dtype=dtypes.float64)
      default_value = constant_op.constant(-1.0, dtype=dtypes.float64)
      sp_input = sparse_tensor.SparseTensorValue(
          indices=np.array([[0, 0], [1, 0], [1, 3], [1, 4], [3, 2], [3, 3]]),
          values=values,
          dense_shape=np.array([5, 6]))
      sp_output, empty_row_indicator = (sparse_ops.sparse_fill_empty_rows(
          sp_input, default_value))
      output, empty_row_indicator_out = self.evaluate(
          [sp_output, empty_row_indicator])

      self.assertAllEqual(output.indices, [[0, 0], [1, 0], [1, 3], [1, 4],
                                           [2, 0], [3, 2], [3, 3], [4, 0]])
      self.assertAllClose(output.values, [0, 10, 13, 14, -1, 32, 33, -1])
      self.assertAllEqual(output.dense_shape, [5, 6])
      self.assertAllEqual(empty_row_indicator_out,
                          np.array([0, 0, 1, 0, 1]).astype(np.bool_))

      values_grad_err = gradient_checker.compute_gradient_error(
          values, values.shape.as_list(), sp_output.values, [8], delta=1e-8)
      self.assertGreater(values_grad_err, 0)
      self.assertLess(values_grad_err, 1e-8)

      default_value_grad_err = gradient_checker.compute_gradient_error(
          default_value,
          default_value.shape.as_list(),
          sp_output.values, [8],
          delta=1e-8)
      self.assertGreater(default_value_grad_err, 0)
      self.assertLess(default_value_grad_err, 1e-8)

  def testFillString(self):
    with test_util.force_cpu():
      sp_input = self._SparseTensor_String5x6()
      sp_output, empty_row_indicator = (
          sparse_ops.sparse_fill_empty_rows(sp_input, ""))

      output, empty_row_indicator_out = self.evaluate(
          [sp_output, empty_row_indicator])

      self.assertAllEqual(
          output.indices,
          [[0, 0], [1, 0], [1, 3], [1, 4], [2, 0], [3, 2], [3, 3], [4, 0]])
      self.assertAllEqual(output.values,
                          [b"a", b"b", b"c", b"d", b"", b"e", b"f", b""])
      self.assertAllEqual(output.dense_shape, [5, 6])
      self.assertAllEqual(empty_row_indicator_out,
                          np.array([0, 0, 1, 0, 1]).astype(np.bool_))

  def testNoEmptyRows(self):
    with test_util.use_gpu():
      sp_input = self._SparseTensor_2x6()
      sp_output, empty_row_indicator = (
          sparse_ops.sparse_fill_empty_rows(sp_input, -1))

      output, empty_row_indicator_out = self.evaluate(
          [sp_output, empty_row_indicator])

      self.assertAllEqual(output.indices, [[0, 0], [1, 0], [1, 3], [1, 4]])
      self.assertAllEqual(output.values, [0, 10, 13, 14])
      self.assertAllEqual(output.dense_shape, [2, 6])
      self.assertAllEqual(empty_row_indicator_out, np.zeros(2).astype(np.bool_))

  def testNoEmptyRowsAndUnordered(self):
    with test_util.use_gpu():
      sp_input = sparse_tensor.SparseTensor(
          indices=np.array([[1, 2], [1, 3], [0, 1], [0, 3]]),
          values=np.array([1, 3, 2, 4]),
          dense_shape=np.array([2, 5]))
      sp_output, empty_row_indicator = (
          sparse_ops.sparse_fill_empty_rows(sp_input, -1))

      output, empty_row_indicator_out = self.evaluate(
          [sp_output, empty_row_indicator])

      self.assertAllEqual(output.indices, [[0, 1], [0, 3], [1, 2], [1, 3]])
      self.assertAllEqual(output.values, [2, 4, 1, 3])
      self.assertAllEqual(output.dense_shape, [2, 5])
      self.assertAllEqual(empty_row_indicator_out, np.zeros(2).astype(np.bool_))

  def testUnordered(self):
    with test_util.use_gpu():
      sp_input = sparse_tensor.SparseTensor(
          indices=np.array([[2, 3], [2, 2], [0, 1], [0, 3]]),
          values=np.array([1, 3, 2, 4]),
          dense_shape=np.array([3, 5]))
      sp_output, empty_row_indicator = (
          sparse_ops.sparse_fill_empty_rows(sp_input, -1))

      output, empty_row_indicator_out = self.evaluate(
          [sp_output, empty_row_indicator])

      self.assertAllEqual(output.indices,
                          [[0, 1], [0, 3], [1, 0], [2, 3], [2, 2]])
      self.assertAllEqual(output.values, [2, 4, -1, 1, 3])
      self.assertAllEqual(output.dense_shape, [3, 5])
      self.assertAllEqual(empty_row_indicator_out, [False, True, False])

  def testEmptyIndicesTensor(self):
    with test_util.use_gpu():
      sp_input = sparse_tensor.SparseTensor(
          indices=np.ones([0, 2]),
          values=np.ones([0]),
          dense_shape=np.array([2, 5]))
      sp_output, empty_row_indicator = (
          sparse_ops.sparse_fill_empty_rows(sp_input, -1))

      output, empty_row_indicator_out = self.evaluate(
          [sp_output, empty_row_indicator])

      self.assertAllEqual(output.indices, [[0, 0], [1, 0]])
      self.assertAllEqual(output.values, [-1, -1])
      self.assertAllEqual(output.dense_shape, [2, 5])
      self.assertAllEqual(empty_row_indicator_out, np.ones(2).astype(np.bool_))

  def testEmptyOutput(self):
    with test_util.use_gpu():
      sp_input = sparse_tensor.SparseTensor(
          indices=np.ones([0, 2]),
          values=np.ones([0]),
          dense_shape=np.array([0, 3]))
      sp_output, empty_row_indicator = (
          sparse_ops.sparse_fill_empty_rows(sp_input, -1))

      output, empty_row_indicator_out = self.evaluate(
          [sp_output, empty_row_indicator])

      self.assertAllEqual(output.indices, np.ones([0, 2]))
      self.assertAllEqual(output.values, np.ones([0]))
      self.assertAllEqual(output.dense_shape, [0, 3])
      self.assertAllEqual(empty_row_indicator_out, [])

  def testInvalidIndices(self):
    with test_util.use_gpu():
      sp_input = sparse_tensor.SparseTensor(
          indices=np.array([[1, 2], [1, 3], [99, 1], [99, 3]]),
          values=np.array([1, 3, 2, 4]),
          dense_shape=np.array([2, 5]))

      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  r"indices\(2, 0\) is invalid"):
        self.evaluate(sparse_ops.sparse_fill_empty_rows(sp_input, -1))


class SparseAddTest(test_util.TensorFlowTestCase):

  def testValuesInVariable(self):
    indices = constant_op.constant([[0]], dtype=dtypes.int64)
    values = variables.Variable([1], trainable=False, dtype=dtypes.float32)
    shape = constant_op.constant([1], dtype=dtypes.int64)

    sp_input = sparse_tensor.SparseTensor(indices, values, shape)
    sp_output = sparse_ops.sparse_add(sp_input, sp_input)

    with test_util.force_cpu():
      self.evaluate(variables.global_variables_initializer())
      output = self.evaluate(sp_output)
      self.assertAllEqual(output.values, [2])


class SparseReduceTest(test_util.TensorFlowTestCase):

  # [[1, ?, 2]
  #  [?, 3, ?]]
  # where ? is implicitly-zero.
  ind = np.array([[0, 0], [0, 2], [1, 1]]).astype(np.int64)
  vals = np.array([1, 1, 1]).astype(np.int32)
  dense_shape = np.array([2, 3]).astype(np.int64)

  def _compare(self, sp_t, reduction_axes, ndims, keep_dims, do_sum):
    densified = self.evaluate(sparse_ops.sparse_tensor_to_dense(sp_t))

    np_ans = densified
    if reduction_axes is None:
      if do_sum:
        np_ans = np.sum(np_ans, keepdims=keep_dims)
      else:
        np_ans = np.max(np_ans, keepdims=keep_dims)
    else:
      if not isinstance(reduction_axes, list):  # Single scalar.
        reduction_axes = [reduction_axes]
      reduction_axes = np.array(reduction_axes).astype(np.int32)
      # Handles negative axes.
      reduction_axes = (reduction_axes + ndims) % ndims
      # Loop below depends on sorted.
      reduction_axes.sort()
      for ra in reduction_axes.ravel()[::-1]:
        if do_sum:
          np_ans = np.sum(np_ans, axis=ra, keepdims=keep_dims)
        else:
          np_ans = np.max(np_ans, axis=ra, keepdims=keep_dims)

    with self.cached_session():
      if do_sum:
        tf_dense_ans = sparse_ops.sparse_reduce_sum(sp_t, reduction_axes,
                                                    keep_dims)
      else:
        tf_dense_ans = sparse_ops.sparse_reduce_max(sp_t, reduction_axes,
                                                    keep_dims)
      out_dense = self.evaluate(tf_dense_ans)

      if do_sum:
        tf_sparse_ans = sparse_ops.sparse_reduce_sum_sparse(sp_t,
                                                            reduction_axes,
                                                            keep_dims)
      else:
        tf_sparse_ans = sparse_ops.sparse_reduce_max_sparse(sp_t,
                                                            reduction_axes,
                                                            keep_dims)
      # Convert to dense for comparison purposes.
      out_sparse = sparse_ops.sparse_tensor_to_dense(tf_sparse_ans)

    self.assertAllClose(np_ans, out_dense)
    self.assertAllClose(np_ans, out_sparse)

  def _compare_all(self, sp_t, reduction_axes, ndims):
    self._compare(sp_t, reduction_axes, ndims, False, False)
    self._compare(sp_t, reduction_axes, ndims, False, True)
    self._compare(sp_t, reduction_axes, ndims, True, False)
    self._compare(sp_t, reduction_axes, ndims, True, True)

  # (TODO:b/133851381): Re-enable this test.
  def disabledtestSimpleAndRandomInputs(self):
    if np.__version__ == "1.13.0":
      self.skipTest("numpy 1.13.0 bug")

    sp_t = sparse_tensor.SparseTensor(self.ind, self.vals, self.dense_shape)

    with test_util.force_cpu():
      self._compare_all(sp_t, None, ndims=2)
      self._compare_all(sp_t, 0, ndims=2)
      self._compare_all(sp_t, [1], ndims=2)
      self._compare_all(sp_t, [0, 1], ndims=2)
      self._compare_all(sp_t, [1, 0], ndims=2)
      self._compare_all(sp_t, [-1], ndims=2)
      self._compare_all(sp_t, [1, -2], ndims=2)

    np.random.seed(1618)
    test_dims = [(1618, 1, 11, 7, 1), (1,), (1, 1, 1)]
    with test_util.force_cpu():
      for dims in test_dims:
        sp_t, unused_nnz = _sparsify(np.random.randn(*dims))
        # reduce all using None
        self._compare_all(sp_t, None, ndims=len(dims))
        # reduce random axes from 1D to N-D
        for d in range(1, len(dims) + 1):
          axes = np.random.choice(len(dims), size=d, replace=False).tolist()
          self._compare_all(sp_t, axes, ndims=len(dims))

  def testInvalidAxes(self):
    sp_t = sparse_tensor.SparseTensor(self.ind, self.vals, self.dense_shape)
    with test_util.force_cpu():
      with self.assertRaisesOpError("Invalid reduction dimension -3"):
        self.evaluate(sparse_ops.sparse_reduce_sum(sp_t, -3))
      with self.assertRaisesOpError("Invalid reduction dimension 2"):
        self.evaluate(sparse_ops.sparse_reduce_sum(sp_t, 2))
      with self.assertRaisesOpError("Invalid reduction dimension -3"):
        self.evaluate(sparse_ops.sparse_reduce_max(sp_t, -3))
      with self.assertRaisesOpError("Invalid reduction dimension 2"):
        self.evaluate(sparse_ops.sparse_reduce_max(sp_t, 2))

  @test_util.run_deprecated_v1
  def testGradient(self):
    np.random.seed(8161)
    test_dims = [(11, 1, 5, 7, 1), (2, 2)]
    with self.session(use_gpu=False):
      for dims in test_dims:
        sp_t, nnz = _sparsify(np.random.randn(*dims))
        # reduce random axes from 1D to N-D
        for d in range(1, len(dims) + 1):
          axes = np.random.choice(len(dims), size=d, replace=False).tolist()
          reduced = sparse_ops.sparse_reduce_sum(sp_t, axes)

          err = gradient_checker.compute_gradient_error(
              sp_t.values, (nnz,), reduced,
              self.evaluate(reduced).shape)
          self.assertLess(err, 1e-3)

        # Tests for negative axes.
        reduced = sparse_ops.sparse_reduce_sum(sp_t, -1)
        err = gradient_checker.compute_gradient_error(
            sp_t.values, (nnz,), reduced,
            self.evaluate(reduced).shape)
        self.assertLess(err, 1e-3)

  def _testSparseReduceShape(self, sp_t, reduction_axes, ndims, keep_dims,
                             do_sum):
    densified = self.evaluate(sparse_ops.sparse_tensor_to_dense(sp_t))

    np_op = np.sum
    tf_op = sparse_ops.sparse_reduce_sum
    if not do_sum:
      np_op = np.max
      tf_op = sparse_ops.sparse_reduce_max

    np_ans = densified
    if reduction_axes is None:
      np_ans = np_op(np_ans, keepdims=keep_dims)
    else:
      if not isinstance(reduction_axes, list):  # Single scalar.
        reduction_axes = [reduction_axes]
      reduction_axes = np.array(reduction_axes).astype(np.int32)
      # Handles negative axes.
      reduction_axes = (reduction_axes + ndims) % ndims
      # Loop below depends on sorted.
      reduction_axes.sort()
      for ra in reduction_axes.ravel()[::-1]:
        np_ans = np_op(np_ans, axis=ra, keepdims=keep_dims)

    tf_ans = tf_op(sp_t, reduction_axes, keep_dims)
    self.assertAllEqual(np_ans.shape, tf_ans.get_shape().as_list())

  # (TODO:b/133851381): Re-enable this test
  def disabledtestSparseReduceSumOrMaxShape(self):
    sp_t = sparse_tensor.SparseTensor(self.ind, self.vals, self.dense_shape)

    with test_util.force_cpu():
      for do_sum in [True, False]:
        for keep_dims in [True, False]:
          self._testSparseReduceShape(sp_t, None, 2, keep_dims, do_sum)
          self._testSparseReduceShape(sp_t, 0, 2, keep_dims, do_sum)
          self._testSparseReduceShape(sp_t, [1], 2, keep_dims, do_sum)
          self._testSparseReduceShape(sp_t, [0, 1], 2, keep_dims, do_sum)
          self._testSparseReduceShape(sp_t, [1, 0], 2, keep_dims, do_sum)
          self._testSparseReduceShape(sp_t, [-1], 2, keep_dims, do_sum)
          self._testSparseReduceShape(sp_t, [1, -2], 2, keep_dims, do_sum)

  def testIntegerOverflow(self):
    with self.cached_session(use_gpu=False):
      with self.assertRaises(errors.InvalidArgumentError):
        res = sparse_ops.gen_sparse_ops.sparse_reduce_max(
            input_indices=[[1, 2], [3, 4]],
            input_shape=[2**32, 2**31],
            input_values=[1, 3],
            reduction_axes=[0],
            keep_dims=False,
            name=None)

        self.evaluate(res)
      with self.assertRaises(errors.InvalidArgumentError):
        res = sparse_ops.gen_sparse_ops.sparse_reduce_max_sparse(
            input_indices=[[1, 2], [3, 4]],
            input_shape=[2**32, 2**31],
            input_values=[1, 3],
            reduction_axes=[0],
            keep_dims=False,
            name=None)

        self.evaluate(res)
      with self.assertRaises(errors.InvalidArgumentError):
        res = sparse_ops.gen_sparse_ops.sparse_reduce_sum(
            input_indices=[[1, 2], [3, 4]],
            input_shape=[2**32, 2**31],
            input_values=[1, 3],
            reduction_axes=[0],
            keep_dims=False,
            name=None)

        self.evaluate(res)


class SparseMathOpsTest(test_util.TensorFlowTestCase):

  def _check(self, result_tensor, result_np, input_sp_t):
    self.assertTrue(isinstance(result_tensor, sparse_tensor.SparseTensor))
    self.assertTrue(isinstance(input_sp_t, sparse_tensor.SparseTensor))
    self.assertAllCloseAccordingToType(input_sp_t.indices,
                                       result_tensor.indices)
    self.assertAllCloseAccordingToType(input_sp_t.dense_shape,
                                       result_tensor.dense_shape)

    res_densified = sparse_ops.sparse_to_dense(
        result_tensor.indices, result_tensor.dense_shape, result_tensor.values)
    self.assertAllCloseAccordingToType(result_np, res_densified)

  @test_util.run_deprecated_v1
  def testCwiseShapeValidation(self):
    # Test case for GitHub 24072.
    with test_util.force_cpu():
      a = array_ops.ones([3, 4, 1], dtype=dtypes.int32)
      b = sparse_tensor.SparseTensor([[0, 0, 1, 0], [0, 0, 3, 0]], [10, 20],
                                     [1, 1, 4, 2])
      c = a * b
      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          "broadcasts dense to sparse only; got incompatible shapes"):
        self.evaluate(c)

  def testCwiseDivAndMul(self):
    np.random.seed(1618)
    sp_shapes = [(10, 10, 10), (5, 5), (1618,), (3, 3, 7)]
    dense_shapes = [(10, 10, 1), (5, 5), (1,), (1, 7)]

    with test_util.force_cpu():
      for dtype in [np.float32, np.float64, np.int32, np.int64]:
        for sp_shape, dense_shape in zip(sp_shapes, dense_shapes):
          sp_vals_np = np.random.rand(*sp_shape).astype(dtype) + 1
          dense_vals_np = np.random.rand(*dense_shape).astype(dtype) + 1
          sp_t, unused_nnz = _sparsify(sp_vals_np, thresh=1.5)
          sp_t_densified = sparse_ops.sparse_tensor_to_dense(sp_t)
          dense_t = constant_op.constant(dense_vals_np)

          self._check(sp_t / dense_t, sp_t_densified / dense_vals_np, sp_t)
          # Check commutative.
          self._check(sp_t * dense_t, sp_t_densified * dense_vals_np, sp_t)
          self._check(dense_t * sp_t, sp_t_densified * dense_vals_np, sp_t)

          if dtype in [np.int32, np.int64]:
            res = sp_t / dense_t  # should invoke "__truediv__"
            self.assertEqual(res.values.dtype, np.float64)

  def testCwiseAdd(self):
    with test_util.force_cpu():
      # Identity(2) + AllOnes(2,2).  Should be equal to 2 * Identity(2).
      indices = [[0, 0], [1, 1]]
      vals = [1, 1]
      shape = (2, 2)

      sp_t = sparse_tensor.SparseTensor(indices, vals, shape)
      dense_t = array_ops.ones(shape, dtype=dtypes.int32)
      self._check(
          sparse_ops.sparse_dense_cwise_add(sp_t, dense_t),
          np.identity(2) * 2, sp_t)

      # Variant of above, but broadcasts the dense side.
      dense_t = array_ops.ones([1], dtype=dtypes.int32)
      self._check(
          sparse_ops.sparse_dense_cwise_add(sp_t, dense_t),
          np.identity(2) * 2, sp_t)

  @test_util.run_deprecated_v1
  def testGradients(self):
    np.random.seed(1618)
    sp_shapes = [(10, 10, 10), (5, 5), (1618,), (3, 3, 7)]
    dense_shapes = [(10, 10, 1), (5, 5), (1,), (1, 7)]

    with self.session(use_gpu=False):
      for dtype in [np.float32, np.float64]:
        for sp_shape, dense_shape in zip(sp_shapes, dense_shapes):
          sp_vals_np = np.random.rand(*sp_shape).astype(dtype) + 1
          dense_vals_np = np.random.rand(*dense_shape).astype(dtype) + 1
          sp_t, nnz = _sparsify(sp_vals_np, thresh=1.5)
          dense_t = constant_op.constant(dense_vals_np)

          cmul = sp_t * dense_t
          err = gradient_checker.compute_gradient_error([sp_t.values, dense_t],
                                                        [(nnz,), dense_shape],
                                                        cmul.values, (nnz,))
          self.assertLess(err, 1e-4)

          cdiv = sp_t / dense_t
          err = gradient_checker.compute_gradient_error(sp_t.values, (nnz,),
                                                        cdiv.values, (nnz,))
          self.assertLess(err, 1e-4)
          err = gradient_checker.compute_gradient_error(
              dense_t,
              dense_shape,
              cdiv.values, (nnz,),
              x_init_value=dense_vals_np)
          self.assertLess(err, 2e-4)


class SparseSoftmaxTest(test_util.TensorFlowTestCase):

  @test_util.run_deprecated_v1
  def testEquivalentToDensified(self):
    np.random.seed(1618)
    n, m = np.random.choice(20, size=2)

    for dtype in [np.float16, np.float32, np.float64]:
      sp_vals_np = np.random.rand(n, m).astype(dtype)

      batched_sp_t, unused_nnz1 = _sparsify(
          sp_vals_np.reshape((1, n, m)), thresh=0.)  # No masking.

      with test_util.force_cpu():
        densified = constant_op.constant(sp_vals_np)

        sp_result = self.evaluate(
            sparse_ops.sparse_softmax(batched_sp_t)).values.reshape((n, m))
        dense_result = nn_ops.softmax(densified)

        self.assertAllCloseAccordingToType(dense_result, sp_result)

  def testHigherRanks(self):
    # For the first shape:
    # First batch:
    # [?   e.]
    # [1.  ? ]
    # Second batch:
    # [e   ? ]
    # [e   e ]
    #
    # The softmax results should be:
    # [?   1.]     [1    ?]
    # [1.  ? ] and [.5  .5]
    # where ? means implicitly zero.
    #
    # The second shape: same input data, but with a higher-rank shape.
    shapes = [[2, 2, 2], [2, 1, 2, 2]]
    for shape in shapes:
      values = np.asarray(
          [0., np.e, 1., 0., np.e, 0., np.e, np.e]).reshape(shape)
      sp_t, unused_nnz = _sparsify(values, thresh=1e-2)
      expected_values = [1., 1., 1., .5, .5]

      with test_util.force_cpu():
        result = sparse_ops.sparse_softmax(sp_t)

        self.assertAllEqual(expected_values, result.values)
        self.assertAllEqual(sp_t.indices, result.indices)
        self.assertAllEqual(shape, result.dense_shape)

  @test_util.run_deprecated_v1
  def testGradient(self):
    x_shape = [2, 5, 10]
    with self.cached_session(use_gpu=False):
      for dtype in [np.float32, np.float64]:
        x_np = np.random.randn(*x_shape).astype(dtype)
        x_tf, nnz = _sparsify(x_np)
        y_tf = sparse_ops.sparse_softmax(x_tf)
        err = gradient_checker.compute_gradient_error(x_tf.values, (nnz,),
                                                      y_tf.values, (nnz,))
        self.assertLess(err, 1e-4)

  def testIntegerOverflow(self):
    with self.cached_session(use_gpu=False):
      with self.assertRaises(errors.InvalidArgumentError):
        res = sparse_ops.gen_sparse_ops.sparse_softmax(
            sp_indices=[[1, 1]],
            sp_values=[2.0],
            sp_shape=[2**32, 2**31],
            name=None)

        self.evaluate(res)

  def testReshapeNegativeShape(self):
    with self.cached_session(use_gpu=False):
      with self.assertRaises(errors.InvalidArgumentError):
        res = sparse_ops.gen_sparse_ops.sparse_softmax(
            sp_indices=[[1, 1]], sp_values=[2.0], sp_shape=[-1, 1], name=None)

        self.evaluate(res)


class SparseMinimumMaximumTest(test_util.TensorFlowTestCase):

  def _assertSparseTensorValueEqual(self, a, b):
    self.assertAllEqual(a.indices, b.indices)
    self.assertAllEqual(a.values, b.values)
    self.assertAllEqual(a.dense_shape, b.dense_shape)

  def testBasic(self):
    with test_util.force_cpu():
      # 1-D, values at index 0.
      sp_zero = sparse_tensor.SparseTensor([[0]], [0], [7])
      sp_one = sparse_tensor.SparseTensor([[0]], [1], [7])
      max_tf = sparse_ops.sparse_maximum(sp_zero, sp_one)
      min_tf = sparse_ops.sparse_minimum(sp_zero, sp_one)
      self._assertSparseTensorValueEqual(sp_one, max_tf)
      self._assertSparseTensorValueEqual(sp_zero, min_tf)

      # Values at different indices.
      sp_zero = sparse_tensor.SparseTensor([[0]], [0], [7])
      sp_zero_2 = sparse_tensor.SparseTensor([[1]], [0], [7])
      expected = sparse_tensor.SparseTensor([[0], [1]], [0, 0], [7])
      max_tf = sparse_ops.sparse_maximum(sp_zero, sp_zero_2)
      min_tf = sparse_ops.sparse_minimum(sp_zero, sp_zero_2)
      self._assertSparseTensorValueEqual(expected, max_tf)
      self._assertSparseTensorValueEqual(expected, min_tf)

  @test_util.run_deprecated_v1
  def testRandom(self):
    np.random.seed(1618)
    shapes = [(13,), (6, 8), (1, 7, 1)]
    for shape in shapes:
      for dtype in [np.int32, np.int64, np.float16, np.float32, np.float64]:
        a_np = np.random.randn(*shape).astype(dtype)
        b_np = np.random.randn(*shape).astype(dtype)
        sp_a, unused_a_nnz = _sparsify(a_np, thresh=-.5)
        sp_b, unused_b_nnz = _sparsify(b_np, thresh=-.5)

        with self.cached_session(use_gpu=False):
          maximum_tf = sparse_ops.sparse_maximum(sp_a, sp_b)
          maximum_tf_densified = sparse_ops.sparse_tensor_to_dense(
              maximum_tf).eval()
          minimum_tf = sparse_ops.sparse_minimum(sp_a, sp_b)
          minimum_tf_densified = sparse_ops.sparse_tensor_to_dense(
              minimum_tf).eval()

          a_densified = sparse_ops.sparse_tensor_to_dense(sp_a).eval()
          b_densified = sparse_ops.sparse_tensor_to_dense(sp_b).eval()

        self.assertAllEqual(
            np.maximum(a_densified, b_densified), maximum_tf_densified)
        self.assertAllEqual(
            np.minimum(a_densified, b_densified), minimum_tf_densified)

  def testMismatchedShapes(self):
    with test_util.force_cpu():
      sp_zero = sparse_tensor.SparseTensor([[0, 0]], [0], [1, 1])
      sp_one = sparse_tensor.SparseTensor([[0]], [1], [2])
      with self.assertRaisesOpError("Operands do not have the same ranks"):
        self.evaluate(sparse_ops.sparse_maximum(sp_zero, sp_one))

      sp_zero = sparse_tensor.SparseTensor([[0]], [0], [1])
      sp_one = sparse_tensor.SparseTensor([[0]], [1], [2])
      with self.assertRaisesOpError("Operands' shapes do not match"):
        self.evaluate(sparse_ops.sparse_maximum(sp_zero, sp_one))


class SparseTransposeTest(test.TestCase):

  def testTranspose(self):
    if np.__version__ == "1.13.0":
      self.skipTest("numpy 1.13.0 bug")

    with test_util.force_cpu():
      np.random.seed(1618)
      shapes = [np.random.randint(1, 10, size=rank) for rank in range(1, 6)]
      for shape in shapes:
        for dtype in [np.int32, np.int64, np.float32, np.float64]:
          dn_input = np.random.randn(*shape).astype(dtype)
          rank = self.evaluate(array_ops.rank(dn_input))
          perm = np.random.choice(rank, rank, False)
          sp_input, unused_a_nnz = _sparsify(dn_input)
          sp_trans = sparse_ops.sparse_transpose(sp_input, perm=perm)
          dn_trans = sparse_ops.sparse_tensor_to_dense(sp_trans)
          expected_trans = array_ops.transpose(dn_input, perm=perm)
          self.assertAllEqual(expected_trans.shape, sp_trans.get_shape())
          self.assertAllEqual(dn_trans, expected_trans)


class SparsePlaceholderTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testPlaceholder(self):
    foo = array_ops.sparse_placeholder(dtypes.float32, shape=(10, 47))
    self.assertAllEqual([10, 47], foo.get_shape())
    self.assertAllEqual([None, 2], foo.indices.get_shape().as_list())

  @test_util.run_deprecated_v1
  def testPartialShapePlaceholder(self):
    foo = array_ops.sparse_placeholder(dtypes.float32, shape=(None, 47))
    self.assertAllEqual([None, 47], foo.get_shape().as_list())
    self.assertAllEqual([None, 2], foo.indices.get_shape().as_list())

  @test_util.run_deprecated_v1
  def testNoShapePlaceholder(self):
    foo = array_ops.sparse_placeholder(dtypes.float32, shape=None)
    self.assertAllEqual(None, foo.get_shape())
    self.assertAllEqual([None, None], foo.indices.get_shape().as_list())


if __name__ == "__main__":
  googletest.main()
