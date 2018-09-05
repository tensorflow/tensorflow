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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.data.python.ops import map_defun
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class MapDefunTest(test.TestCase):

  def testMapDefunSimple(self):

    @function.Defun(dtypes.int32)
    def simple_fn(x):
      return x * 2 + 3

    nums = [[1, 2], [3, 4], [5, 6]]
    elems = constant_op.constant(nums, dtype=dtypes.int32, name="data")
    r = map_defun.map_defun(simple_fn, [elems], [dtypes.int32], [(2,)])[0]
    expected = elems * 2 + 3
    self.assertAllEqual(self.evaluate(r), self.evaluate(expected))

  def testMapDefunMismatchedTypes(self):

    @function.Defun(dtypes.int32)
    def fn(x):
      return math_ops.cast(x, dtypes.float64)

    nums = [1, 2, 3, 4, 5, 6]
    elems = constant_op.constant(nums, dtype=dtypes.int32, name="data")
    r = map_defun.map_defun(fn, [elems], [dtypes.int32], [()])[0]
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(r)

  def testMapDefunReduceDim(self):
    # Tests where the output has a different rank from the input

    @function.Defun(dtypes.int32)
    def fn(x):
      return array_ops.gather(x, 0)

    nums = [[1, 2], [3, 4], [5, 6]]
    elems = constant_op.constant(nums, dtype=dtypes.int32, name="data")
    r = map_defun.map_defun(fn, [elems], [dtypes.int32], [()])[0]
    expected = constant_op.constant([1, 3, 5])
    self.assertAllEqual(self.evaluate(r), self.evaluate(expected))

  def testMapDefunMultipleOutputs(self):

    @function.Defun(dtypes.int32)
    def fn(x):
      return (x, math_ops.cast(x * 2 + 3, dtypes.float64))

    nums = [[1, 2], [3, 4], [5, 6]]
    elems = constant_op.constant(nums, dtype=dtypes.int32, name="data")
    r = map_defun.map_defun(fn, [elems], [dtypes.int32, dtypes.float64], [(2,),
                                                                          (2,)])
    expected = [elems, elems * 2 + 3]
    self.assertAllEqual(self.evaluate(r), self.evaluate(expected))

  def testMapDefunShapeInference(self):

    @function.Defun(dtypes.int32)
    def fn(x):
      return x

    nums = [[1, 2], [3, 4], [5, 6]]
    elems = constant_op.constant(nums, dtype=dtypes.int32, name="data")
    result = map_defun.map_defun(fn, [elems], [dtypes.int32], [(2,)])[0]
    self.assertEqual(result.get_shape(), (3, 2))

  def testMapDefunPartialShapeInference(self):

    @function.Defun(dtypes.int32)
    def fn(x):
      return x

    elems = array_ops.placeholder(dtypes.int64, (None, 2))
    result = map_defun.map_defun(fn, [elems], [dtypes.int32], [(2,)])
    self.assertEqual(result[0].get_shape().as_list(), [None, 2])

  def testMapDefunRaisesErrorOnRuntimeShapeMismatch(self):

    @function.Defun(dtypes.int32, dtypes.int32)
    def fn(x, y):
      return x, y

    elems1 = array_ops.placeholder(dtypes.int32)
    elems2 = array_ops.placeholder(dtypes.int32)
    result = map_defun.map_defun(fn, [elems1, elems2],
                                 [dtypes.int32, dtypes.int32], [(), ()])
    with self.test_session() as sess:
      with self.assertRaisesWithPredicateMatch(
          errors.InvalidArgumentError,
          "All inputs must have the same dimension 0."):
        sess.run(result, feed_dict={elems1: [1, 2, 3, 4, 5], elems2: [1, 2, 3]})

  def testMapDefunRaisesDefunError(self):

    @function.Defun(dtypes.int32)
    def fn(x):
      with ops.control_dependencies([check_ops.assert_equal(x, 0)]):
        return array_ops.identity(x)

    elems = constant_op.constant([0, 0, 0, 37, 0])
    result = map_defun.map_defun(fn, [elems], [dtypes.int32], [()])
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(result)

  def testMapDefunCancelledCorrectly(self):

    @function.Defun(dtypes.int64)
    def defun(x):
      # x has leading dimension 5, this will raise an error
      return array_ops.gather(x, 10)

    c = array_ops.tile(
        array_ops.expand_dims(
            constant_op.constant([1, 2, 3, 4, 5], dtype=dtypes.int64), 0),
        [100, 1])
    map_defun_op = map_defun.map_defun(defun, [c], [dtypes.int64], [()])[0]
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 r"indices = 10 is not in \[0, 5\)"):
      self.evaluate(map_defun_op)


if __name__ == "__main__":
  test.main()
