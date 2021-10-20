# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for MapDefunOp."""
import time

from absl.testing import parameterized

from tensorflow.python.client import session
from tensorflow.python.data.experimental.ops import map_defun
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.eager import function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test


# TODO(b/123903858): Add eager and V2 test coverage
def _test_combinations():
  return combinations.combine(tf_api_version=[1], mode=["graph"])


class MapDefunTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(_test_combinations())
  def testNoIntraOpLimit(self):

    @function.defun(input_signature=[tensor_spec.TensorSpec([2], dtypes.int32)])
    def simple_fn(x):
      return x * 2 + 3

    nums = [[1, 2], [3, 4], [5, 6]]
    elems = constant_op.constant(nums, dtype=dtypes.int32, name="data")
    r = map_defun.map_defun(
        simple_fn, [elems], [dtypes.int32], [(2,)],
        max_intra_op_parallelism=0)[0]
    expected = elems * 2 + 3
    self.assertAllEqual(self.evaluate(r), self.evaluate(expected))

  @combinations.generate(_test_combinations())
  def testMapDefunSimple(self):

    @function.defun(input_signature=[tensor_spec.TensorSpec([2], dtypes.int32)])
    def simple_fn(x):
      return x * 2 + 3

    nums = [[1, 2], [3, 4], [5, 6]]
    elems = constant_op.constant(nums, dtype=dtypes.int32, name="data")
    r = map_defun.map_defun(simple_fn, [elems], [dtypes.int32], [(2,)])[0]
    expected = elems * 2 + 3
    self.assertAllEqual(self.evaluate(r), self.evaluate(expected))

  @combinations.generate(_test_combinations())
  def testMapDefunMismatchedTypes(self):

    @function.defun(input_signature=[tensor_spec.TensorSpec([], dtypes.int32)])
    def fn(x):
      return math_ops.cast(x, dtypes.float64)

    nums = [1, 2, 3, 4, 5, 6]
    elems = constant_op.constant(nums, dtype=dtypes.int32, name="data")
    r = map_defun.map_defun(fn, [elems], [dtypes.int32], [()])[0]
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(r)

  @combinations.generate(_test_combinations())
  def testMapDefunReduceDim(self):
    # Tests where the output has a different rank from the input

    @function.defun(input_signature=[tensor_spec.TensorSpec([2], dtypes.int32)])
    def fn(x):
      return array_ops.gather(x, 0)

    nums = [[1, 2], [3, 4], [5, 6]]
    elems = constant_op.constant(nums, dtype=dtypes.int32, name="data")
    r = map_defun.map_defun(fn, [elems], [dtypes.int32], [()])[0]
    expected = constant_op.constant([1, 3, 5])
    self.assertAllEqual(self.evaluate(r), self.evaluate(expected))

  @combinations.generate(_test_combinations())
  def testMapDefunMultipleOutputs(self):

    @function.defun(input_signature=[tensor_spec.TensorSpec([2], dtypes.int32)])
    def fn(x):
      return (x, math_ops.cast(x * 2 + 3, dtypes.float64))

    nums = [[1, 2], [3, 4], [5, 6]]
    elems = constant_op.constant(nums, dtype=dtypes.int32, name="data")
    r = map_defun.map_defun(fn, [elems], [dtypes.int32, dtypes.float64], [(2,),
                                                                          (2,)])
    expected = [elems, elems * 2 + 3]
    self.assertAllEqual(self.evaluate(r), self.evaluate(expected))

  @combinations.generate(_test_combinations())
  def testMapDefunShapeInference(self):

    @function.defun(input_signature=[tensor_spec.TensorSpec([2], dtypes.int32)])
    def fn(x):
      return x

    nums = [[1, 2], [3, 4], [5, 6]]
    elems = constant_op.constant(nums, dtype=dtypes.int32, name="data")
    result = map_defun.map_defun(fn, [elems], [dtypes.int32], [(2,)])[0]
    self.assertEqual(result.get_shape(), (3, 2))

  @combinations.generate(_test_combinations())
  def testMapDefunPartialShapeInference(self):

    @function.defun(input_signature=[tensor_spec.TensorSpec([2], dtypes.int32)])
    def fn(x):
      return x

    elems = array_ops.placeholder(dtypes.int64, (None, 2))
    result = map_defun.map_defun(fn, [elems], [dtypes.int32], [(2,)])
    self.assertEqual(result[0].get_shape().as_list(), [None, 2])

  @combinations.generate(_test_combinations())
  def testMapDefunRaisesErrorOnRuntimeShapeMismatch(self):

    @function.defun(input_signature=[
        tensor_spec.TensorSpec(None, dtypes.int32),
        tensor_spec.TensorSpec(None, dtypes.int32)
    ])
    def fn(x, y):
      return x, y

    elems1 = array_ops.placeholder(dtypes.int32)
    elems2 = array_ops.placeholder(dtypes.int32)
    result = map_defun.map_defun(fn, [elems1, elems2],
                                 [dtypes.int32, dtypes.int32], [(), ()])
    with self.cached_session() as sess:
      with self.assertRaisesWithPredicateMatch(
          errors.InvalidArgumentError,
          "All inputs must have the same dimension 0."):
        sess.run(result, feed_dict={elems1: [1, 2, 3, 4, 5], elems2: [1, 2, 3]})

  @combinations.generate(_test_combinations())
  def testMapDefunRaisesDefunError(self):

    @function.defun(input_signature=[tensor_spec.TensorSpec([], dtypes.int32)])
    def fn(x):
      with ops.control_dependencies([check_ops.assert_equal(x, 0)]):
        return array_ops.identity(x)

    elems = constant_op.constant([0, 0, 0, 37, 0])
    result = map_defun.map_defun(fn, [elems], [dtypes.int32], [()])
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(result)

  @combinations.generate(_test_combinations())
  def testMapDefunCancelledCorrectly(self):

    @function.defun(input_signature=[tensor_spec.TensorSpec([5], dtypes.int64)])
    def defun(x):
      # x has leading dimension 5, this will raise an error
      return array_ops.gather(x, 10)

    c = array_ops.tile(
        array_ops.expand_dims(
            constant_op.constant([1, 2, 3, 4, 5], dtype=dtypes.int64), 0),
        [100, 1])
    map_defun_op = map_defun.map_defun(defun, [c], [dtypes.int64], [()])[0]
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                r"indices = 10 is not in \[0, 5\)"):
      self.evaluate(map_defun_op)

  @combinations.generate(_test_combinations())
  def testMapDefunWithUnspecifiedOutputShape(self):

    @function.defun(input_signature=[tensor_spec.TensorSpec([2], dtypes.int32)])
    def simple_fn(x):
      res = x * 2 + 3
      return (res, res + 1, res + 2)

    nums = [[1, 2], [3, 4], [5, 6]]
    elems = constant_op.constant(nums, dtype=dtypes.int32, name="data")
    r = map_defun.map_defun(simple_fn, [elems],
                            [dtypes.int32, dtypes.int32, dtypes.int32],
                            [None, (None,), (2,)])
    expected = elems * 2 + 3
    self.assertAllEqual(self.evaluate(r[0]), self.evaluate(expected))
    self.assertAllEqual(self.evaluate(r[1]), self.evaluate(expected + 1))
    self.assertAllEqual(self.evaluate(r[2]), self.evaluate(expected + 2))

  @combinations.generate(_test_combinations())
  def testMapDefunWithDifferentOutputShapeEachRun(self):

    @function.defun(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
    def simple_fn(x):
      return x * 2 + 3

    elems = array_ops.placeholder(dtypes.int32, name="data")
    r = map_defun.map_defun(simple_fn, [elems], [dtypes.int32], [None])[0]
    with session.Session() as sess:
      self.assertAllEqual(sess.run(r, feed_dict={elems: [0]}), [3])
      self.assertAllEqual(
          sess.run(r, feed_dict={elems: [[0], [1]]}), [[3], [5]])

  @combinations.generate(_test_combinations())
  def testMapDefunWithWrongOutputShape(self):

    @function.defun(input_signature=[tensor_spec.TensorSpec([2], dtypes.int32)])
    def simple_fn(x):
      return x * 2 + 3

    nums = [[1, 2], [3, 4], [5, 6]]
    elems = constant_op.constant(nums, dtype=dtypes.int32, name="data")
    r = map_defun.map_defun(simple_fn, [elems], [dtypes.int32], [(1,)])[0]
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(r)

  @combinations.generate(_test_combinations())
  def testMapDefunWithInvalidInput(self):

    @function.defun(
        input_signature=[tensor_spec.TensorSpec(None, dtypes.int32)])
    def simple_fn(x):
      return x * 2

    c = constant_op.constant(2)
    with self.assertRaises(ValueError):
      # Fails at graph construction time for inputs with known shapes.
      r = map_defun.map_defun(simple_fn, [c], [dtypes.int32], [None])[0]
    p = array_ops.placeholder(dtypes.int32)
    r = map_defun.map_defun(simple_fn, [p], [dtypes.int32], [None])[0]
    with session.Session() as sess:
      with self.assertRaises(errors.InvalidArgumentError):
        sess.run(r, feed_dict={p: 0})

  @combinations.generate(_test_combinations())
  def testMapDefunWithParentCancellation(self):
    # Checks that a cancellation of the parent graph is threaded through to
    # MapDefunOp correctly.
    @function.defun(input_signature=[tensor_spec.TensorSpec([], dtypes.int32)])
    def simple_fn(x):
      del x
      queue = data_flow_ops.FIFOQueue(10, dtypes.int32, ())
      # Blocking
      return queue.dequeue_many(5)

    c = constant_op.constant([1, 2, 3, 4, 5])
    map_defun_op = map_defun.map_defun(simple_fn, [c], [dtypes.int32], [()])[0]

    with self.cached_session() as sess:
      thread = self.checkedThread(
          self.assert_op_cancelled, args=(map_defun_op,))
      thread.start()
      time.sleep(0.2)
      sess.close()
      thread.join()

  @combinations.generate(_test_combinations())
  def testMapDefunWithCapturedInputs(self):
    c = constant_op.constant(2)

    @function.defun(input_signature=[tensor_spec.TensorSpec([], dtypes.int32)])
    def fn(x):
      return x + c

    x = constant_op.constant([1, 2, 3, 4])
    map_defun_op = map_defun.map_defun(fn, [x], [dtypes.int32], [()])[0]
    expected = x + c
    self.assertAllEqual(self.evaluate(expected), self.evaluate(map_defun_op))

  @combinations.generate(_test_combinations())
  def testMapDefunWithVariantTensor(self):

    @function.defun(
        input_signature=[tensor_spec.TensorSpec([], dtypes.variant)])
    def fn(x):
      return x

    st = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])

    serialized = sparse_ops.serialize_sparse_v2(st, out_type=dtypes.variant)
    serialized = array_ops.stack([serialized, serialized])
    map_defun_op = map_defun.map_defun(fn, [serialized], [dtypes.variant],
                                       [None])[0]
    deserialized = sparse_ops.deserialize_sparse(map_defun_op, dtypes.int32)
    expected = sparse_tensor.SparseTensorValue(
        indices=[[0, 0, 0], [0, 1, 2], [1, 0, 0], [1, 1, 2]],
        values=[1, 2, 1, 2],
        dense_shape=[2, 3, 4])
    actual = self.evaluate(deserialized)
    self.assertValuesEqual(expected, actual)

  @combinations.generate(_test_combinations())
  def testMapDefunWithVariantTensorAsCaptured(self):

    st = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
    serialized = sparse_ops.serialize_sparse_v2(st, out_type=dtypes.variant)

    @function.defun(input_signature=[tensor_spec.TensorSpec([], dtypes.int32)])
    def fn(x):
      del x
      return serialized

    x = constant_op.constant([0, 0])
    map_defun_op = map_defun.map_defun(fn, [x], [dtypes.variant], [None])[0]
    deserialized = sparse_ops.deserialize_sparse(map_defun_op, dtypes.int32)
    expected = sparse_tensor.SparseTensorValue(
        indices=[[0, 0, 0], [0, 1, 2], [1, 0, 0], [1, 1, 2]],
        values=[1, 2, 1, 2],
        dense_shape=[2, 3, 4])
    actual = self.evaluate(deserialized)
    self.assertValuesEqual(expected, actual)

  @combinations.generate(_test_combinations())
  def testMapDefunWithStrTensor(self):

    @function.defun(input_signature=[tensor_spec.TensorSpec([], dtypes.string)])
    def fn(x):
      return x

    st = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])

    serialized = sparse_ops.serialize_sparse_v2(st, out_type=dtypes.string)
    serialized = array_ops.stack([serialized, serialized])
    map_defun_op = map_defun.map_defun(fn, [serialized], [dtypes.string],
                                       [None])[0]
    deserialized = sparse_ops.deserialize_sparse(map_defun_op, dtypes.int32)
    expected = sparse_tensor.SparseTensorValue(
        indices=[[0, 0, 0], [0, 1, 2], [1, 0, 0], [1, 1, 2]],
        values=[1, 2, 1, 2],
        dense_shape=[2, 3, 4])
    actual = self.evaluate(deserialized)
    self.assertValuesEqual(expected, actual)


if __name__ == "__main__":
  test.main()
