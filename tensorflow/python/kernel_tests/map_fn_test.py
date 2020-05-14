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
"""Tests for tensorflow.kernels.functional_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
import tensorflow.python.ops.tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


# pylint: disable=invalid-name
def simple_scoped_fn(a, x):
  """Simple function: (a, x) -> 2(x+a), but with "2" as a variable in scope."""
  with variable_scope.variable_scope("body"):
    # Dummy variable, just to check that scoping works as intended.
    two = variable_scope.get_variable(
        "two", [],
        dtype=dtypes.int32,
        initializer=init_ops.constant_initializer(2))
    return math_ops.multiply(math_ops.add(a, x), two)


@test_util.with_control_flow_v2
class MapFnTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testMap_Simple(self):
    nums = [1, 2, 3, 4, 5, 6]
    elems = constant_op.constant(nums, name="data")
    r = map_fn.map_fn(
        lambda x: math_ops.multiply(math_ops.add(x, 3), 2), elems)
    self.assertAllEqual(
        np.array([(x + 3) * 2 for x in nums]), self.evaluate(r))

  def testMapDtypeEager(self):
    with context.eager_mode():
      dtype = map_fn.map_fn(lambda x: constant_op.constant(""),
                            constant_op.constant([]),
                            dtype=dtypes.string).dtype
      self.assertEqual(dtype, dtypes.string)

  def testMapSparseTensor(self):
    with self.cached_session():
      st = sparse_tensor.SparseTensor(
          indices=[[0, 0], [0, 1], [1, 0]],
          values=constant_op.constant([0, 1, 2]),
          dense_shape=[2, 2])
      result = map_fn.map_fn(lambda x: x, st)
      self.assertAllEqual(result.indices, st.indices)
      self.assertAllEqual(result.values, st.values)
      self.assertAllEqual(result.dense_shape, st.dense_shape)

  @test_util.run_in_graph_and_eager_modes
  def testMapOverScalarErrors(self):
    with self.assertRaisesRegexp(ValueError, "not scalars"):
      map_fn.map_fn(lambda x: x, [1, 2])
    with self.assertRaisesRegexp(ValueError, "not a scalar"):
      map_fn.map_fn(lambda x: x, 1)

  @test_util.run_deprecated_v1
  def testMap_Scoped(self):
    with self.cached_session() as sess:

      def double_scoped(x):
        """2x with a dummy 2 that is scoped."""
        with variable_scope.variable_scope("body"):
          # Dummy variable, just to check that scoping works as intended.
          two = variable_scope.get_variable(
              "two", [],
              dtype=dtypes.int32,
              initializer=init_ops.constant_initializer(2))
          return math_ops.multiply(x, two)

      with variable_scope.variable_scope("root") as varscope:
        elems = constant_op.constant([1, 2, 3, 4, 5, 6], name="data")
        doubles = np.array([2 * x for x in [1, 2, 3, 4, 5, 6]])

        r = map_fn.map_fn(double_scoped, elems)
        # Check that we have the one variable we asked for here.
        self.assertEqual(len(variables.trainable_variables()), 1)
        self.assertEqual(variables.trainable_variables()[0].name,
                         "root/body/two:0")
        sess.run([variables.global_variables_initializer()])
        self.assertAllEqual(doubles, self.evaluate(r))

        # Now let's reuse our single variable.
        varscope.reuse_variables()
        r = map_fn.map_fn(double_scoped, elems)
        self.assertEqual(len(variables.trainable_variables()), 1)
        self.assertAllEqual(doubles, self.evaluate(r))

  @test_util.run_deprecated_v1
  def testMap_Grad(self):
    with self.cached_session():
      param = constant_op.constant(2.0)
      elems = constant_op.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name="elems")
      y = map_fn.map_fn(
          lambda x: math_ops.multiply(math_ops.square(x), param), elems)
      r = gradients_impl.gradients(y, param)[0]
      self.assertAllEqual(91.0, self.evaluate(r))
      r = gradients_impl.gradients(y, elems)[0]
      self.assertAllEqual([4.0, 8.0, 12.0, 16.0, 20.0, 24.0], self.evaluate(r))

  @test_util.run_in_graph_and_eager_modes
  def testMap_SimpleNotTensor(self):
    nums = np.array([1, 2, 3, 4, 5, 6])
    r = map_fn.map_fn(
        lambda x: math_ops.multiply(math_ops.add(x, 3), 2), nums)
    self.assertAllEqual(
        np.array([(x + 3) * 2 for x in nums]), self.evaluate(r))

  @test_util.run_in_graph_and_eager_modes
  def testMap_SingleInputMultiOutput(self):
    nums = np.array([1, 2, 3, 4, 5, 6])
    r = map_fn.map_fn(
        lambda x: ((x + 3) * 2, -(x + 3) * 2),
        nums,
        dtype=(dtypes.int64, dtypes.int64))
    self.assertEqual(2, len(r))
    self.assertEqual((6,), r[0].get_shape())
    self.assertEqual((6,), r[1].get_shape())
    received = self.evaluate(r)
    self.assertAllEqual((nums + 3) * 2, received[0])
    self.assertAllEqual(-(nums + 3) * 2, received[1])

  @test_util.run_in_graph_and_eager_modes
  def testMap_MultiOutputMismatchedDtype(self):
    nums = np.array([1, 2, 3, 4, 5, 6])
    with self.assertRaisesRegexp(
        TypeError, r"two structures don't have the same nested structure"):
      # lambda emits tuple, but dtype is a list
      map_fn.map_fn(
          lambda x: ((x + 3) * 2, -(x + 3) * 2),
          nums,
          dtype=[dtypes.int64, dtypes.int64])

  @test_util.run_in_graph_and_eager_modes
  def testMap_MultiInputSingleOutput(self):
    nums = np.array([1, 2, 3, 4, 5, 6])
    r = map_fn.map_fn(
        lambda x: x[0] * x[1][0] + x[1][1], (nums, (nums, -nums)),
        dtype=dtypes.int64)
    self.assertEqual((6,), r.get_shape())
    received = self.evaluate(r)
    self.assertAllEqual(nums * nums + (-nums), received)

  @test_util.run_in_graph_and_eager_modes
  def testMap_MultiInputSameStructureOutput(self):
    nums = np.array([1, 2, 3, 4, 5, 6])
    r = map_fn.map_fn(lambda x: (x[1][0], (x[1][1], x[0])),
                      (nums, (2 * nums, -nums)))
    r = [r[0], r[1][0], r[1][1]]
    self.assertEqual((6,), r[0].get_shape())
    self.assertEqual((6,), r[1].get_shape())
    self.assertEqual((6,), r[2].get_shape())
    received = self.evaluate(r)
    self.assertAllEqual(2 * nums, received[0])
    self.assertAllEqual(-nums, received[1])
    self.assertAllEqual(nums, received[2])

  @test_util.run_in_graph_and_eager_modes
  def testMap_autograph_indirect():
    def test_function(x):
      cond = tf.constant(-1)
      if cond == 0:
        result = x
      else:
        result = x
      return result

    @tf.function
    def map_call(x):
      return tf.map_fn(test_function, x)

    x = constant_op.constant([1])
    y = map_call(x)
    self.assertAllEqual([1], self.evaluate(y))

  @test_util.run_in_graph_and_eager_modes
  def testMapShape(self):
    x = constant_op.constant([[1, 2, 3], [4, 5, 6]])
    y = map_fn.map_fn(lambda e: e, x)
    self.assertAllEqual(y.get_shape(), self.evaluate(y).shape)

  @test_util.run_deprecated_v1
  def testMapUnknownShape(self):
    x = array_ops.placeholder(dtypes.float32)
    y = map_fn.map_fn(lambda e: e, x)
    self.assertIs(None, y.get_shape().dims)

  # TODO(b/124383826): this test fails in eager: the iterable is of length 0 so
  # so the body of the while loop never executes
  @test_util.run_v1_only("b/120545219")
  def testMapEmptyScalar(self):
    map_return = map_fn.map_fn(lambda x: 1,
                               constant_op.constant([], dtype=dtypes.int32))
    self.assertAllEqual([0], map_return.get_shape().dims)
    self.assertAllEqual([0], self.evaluate(map_return).shape)

  # TODO(b/124383826): this test fails in eager: the iterable is of length 0 so
  # so the body of the while loop never executes
  @test_util.run_v1_only("b/120545219")
  def testMapEmptyTensor(self):
    with self.cached_session():
      map_return = map_fn.map_fn(lambda x: array_ops.zeros([3, 2]),
                                 constant_op.constant([]))
      self.assertAllEqual([0, 3, 2], map_return.get_shape().dims)
      self.assertAllEqual([0, 3, 2], self.evaluate(map_return).shape)


if __name__ == "__main__":
  test.main()

# pylint: enable=invalid-name
