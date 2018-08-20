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
"""Tests for reading and writing variables."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer


class VariableOpsTest(xla_test.XLATestCase):
  """Test cases for resource variable operators."""

  def testOneWriteOneOutput(self):
    # Regression test for a bug where computations with one non-constant
    # output and one variable update were mishandled.
    for dtype in self.numeric_types:
      init = np.array([[1, 2j], [3, 4]]).astype(dtype)
      with self.test_session() as sess, self.test_scope():
        v = resource_variable_ops.ResourceVariable(init)
        sess.run(variables.variables_initializer([v]))
        p = array_ops.placeholder(dtype)
        x = v.assign_add(p)
        with ops.control_dependencies([x]):
          y = v.read_value()
        self.assertAllClose(
            np.array([[2, 1 + 2j], [4, 5]]).astype(dtype), sess.run(y, {p: 1}))

  def testSparseRead0DIndices(self):
    for dtype in self.numeric_types:
      init = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8j, 9, 10,
                                                    11]]).astype(dtype)
      with self.test_session() as sess, self.test_scope():
        v = resource_variable_ops.ResourceVariable(init)
        sess.run(variables.variables_initializer([v]))
        x = v.sparse_read(2)
        self.assertAllClose(
            np.array([8j, 9, 10, 11]).astype(dtype), sess.run(x))

  def testSparseRead1DIndices(self):
    for dtype in self.numeric_types:
      init = np.array([[0, 1, 2, 3], [4, 5, 6j, 7], [8, 9, 10,
                                                     11]]).astype(dtype)
      with self.test_session() as sess, self.test_scope():
        v = resource_variable_ops.ResourceVariable(init)
        sess.run(variables.variables_initializer([v]))
        x = v.sparse_read([2, 1])
        self.assertAllClose(
            np.array([[8, 9, 10, 11], [4, 5, 6j, 7]]).astype(dtype),
            sess.run(x))

  def testSparseRead2DIndices(self):
    for dtype in self.numeric_types:
      init = np.array([[0, 1, 2j, 3], [4, 5, 6, 7], [8, 9, 10,
                                                     11]]).astype(dtype)
      with self.test_session() as sess, self.test_scope():
        v = resource_variable_ops.ResourceVariable(init)
        sess.run(variables.variables_initializer([v]))
        x = v.sparse_read([[2, 1], [0, 2]])
        self.assertAllClose(
            np.array([[[8, 9, 10, 11], [4, 5, 6, 7]],
                      [[0, 1, 2j, 3], [8, 9, 10, 11]]]).astype(dtype),
            sess.run(x))

  def testSparseRead2DIndices3DTensor(self):
    for dtype in self.numeric_types:
      init = np.array([[[0, 1, 2], [3, 4, 5]], [[10, 11, 12], [13, 14, 15]],
                       [[20, 21, 22], [23, 24j, 25]],
                       [[30, 31, 32], [33, 34, 35]]]).astype(dtype)
      with self.test_session() as sess, self.test_scope():
        v = resource_variable_ops.ResourceVariable(init)
        sess.run(variables.variables_initializer([v]))
        x = v.sparse_read([[2, 1], [3, 0]])
        self.assertAllClose(
            np.array(
                [[[[20, 21, 22], [23, 24j, 25]], [[10, 11, 12], [13, 14, 15]]
                 ], [[[30, 31, 32], [33, 34, 35]], [[0, 1, 2], [3, 4, 5]]]
                ],).astype(dtype), sess.run(x))

  def testShape(self):
    for dtype in self.numeric_types:
      init = np.ones([2, 3]).astype(dtype)
      with self.test_session() as session, self.test_scope():
        v = resource_variable_ops.ResourceVariable(init)
        session.run(variables.variables_initializer([v]))
        h = v.handle
        s32, s64 = session.run([
            resource_variable_ops.variable_shape(h),
            resource_variable_ops.variable_shape(h, out_type=dtypes.int64)
        ])
        self.assertEqual(s32.dtype, np.int32)
        self.assertEqual(s64.dtype, np.int64)
        self.assertAllEqual(s32, [2, 3])
        self.assertAllEqual(s64, [2, 3])

  def testReadWrite(self):
    """Tests initialization, reading, and writing a resource variable."""
    for dtype in self.numeric_types:
      with self.test_session() as session:
        with self.test_scope():
          with variable_scope.variable_scope("ascope", use_resource=True):
            x = variable_scope.get_variable(
                "x",
                shape=[],
                dtype=dtype,
                initializer=init_ops.constant_initializer(2))
            a = x.read_value()
            with ops.control_dependencies([a]):
              b = state_ops.assign(x, dtype(47))
            with ops.control_dependencies([b]):
              c = x.read_value()
            with ops.control_dependencies([c]):
              d = state_ops.assign_add(x, np.array(6 + 2j).astype(dtype))
            with ops.control_dependencies([d]):
              e = state_ops.assign_sub(x, dtype(3))
            with ops.control_dependencies([e]):
              f = x.read_value()

        session.run(variables.global_variables_initializer())
        v1, v2, v3 = session.run([a, c, f])
        self.assertAllClose(dtype(2), v1)
        self.assertAllClose(dtype(47), v2)
        self.assertAllClose(np.array(50 + 2j).astype(dtype), v3)

  def testTraining(self):
    """Tests a gradient descent step for a simple model."""
    with self.test_session() as session:
      with self.test_scope():
        with variable_scope.variable_scope("ascope", use_resource=True):
          w = variable_scope.get_variable(
              "w",
              shape=[4, 2],
              dtype=dtypes.float32,
              initializer=init_ops.constant_initializer(
                  np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)))
          b = variable_scope.get_variable(
              "b",
              shape=[2],
              dtype=dtypes.float32,
              initializer=init_ops.constant_initializer(
                  np.array([2, 3], dtype=np.float32)))

          x = array_ops.placeholder(dtypes.float32, shape=[1, 4])
          y = math_ops.matmul(x, w) + b
          loss = math_ops.reduce_sum(y)
          optimizer = GradientDescentOptimizer(0.1)
          train = optimizer.minimize(loss)

      session.run(variables.global_variables_initializer())
      session.run(train, {x: np.array([[7, 3, 5, 9]], dtype=np.float32)})
      vw, vb = session.run([w, b])
      self.assertAllClose(
          np.array(
              [[0.3, 1.3], [2.7, 3.7], [4.5, 5.5], [6.1, 7.1]],
              dtype=np.float32),
          vw,
          rtol=1e-4)
      self.assertAllClose(np.array([1.9, 2.9], dtype=np.float32), vb, rtol=1e-4)

  def testWriteOfAliasedTensor(self):
    for dtype in self.numeric_types:
      init = np.array([[1, 2j], [3, 4]]).astype(dtype)
      update = np.array([[7, 1j], [2, 11]]).astype(dtype)
      with self.test_session() as sess, self.test_scope():
        v = resource_variable_ops.ResourceVariable(init)
        sess.run(variables.variables_initializer([v]))
        p = array_ops.placeholder(dtype)
        q = array_ops.identity(p)
        x = v.read_value()
        # Writes the value of 'p' to 'v', but keeps a reference to the original
        # value of 'v' so the variable update cannot reuse its buffer.
        with ops.control_dependencies([x]):
          y = v.assign(q)
        result = sess.run([x, y, q], {p: update})
        self.assertAllClose(init, result[0])
        self.assertAllClose(update, result[1])
        self.assertAllClose(update, result[2])

  def testScatterAdd(self):
    with self.test_session() as sess, self.test_scope():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[2, 1])
      sess.run(
          resource_variable_ops.assign_variable_op(
              handle, constant_op.constant([[1], [7]], dtype=dtypes.int32)))
      sess.run(
          resource_variable_ops.resource_scatter_add(
              handle, [0], constant_op.constant([[2]], dtype=dtypes.int32)))
      read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
      self.assertAllEqual(sess.run(read), [[3], [7]])

  def testScatterSub(self):
    with self.test_session() as sess, self.test_scope():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[2, 1])
      sess.run(
          resource_variable_ops.assign_variable_op(
              handle, constant_op.constant([[4], [1]], dtype=dtypes.int32)))
      sess.run(
          resource_variable_ops.resource_scatter_sub(
              handle, [1], constant_op.constant([[2]], dtype=dtypes.int32)))
      read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
      self.assertAllEqual(sess.run(read), [[4], [-1]])

  def testScatterMul(self):
    with self.test_session() as sess, self.test_scope():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1, 1])
      sess.run(
          resource_variable_ops.assign_variable_op(
              handle, constant_op.constant([[1]], dtype=dtypes.int32)))
      sess.run(
          resource_variable_ops.resource_scatter_mul(
              handle, [0], constant_op.constant([[5]], dtype=dtypes.int32)))
      read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
      self.assertEqual(sess.run(read), [[5]])

  def testScatterDiv(self):
    with self.test_session() as sess, self.test_scope():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1, 1])
      sess.run(
          resource_variable_ops.assign_variable_op(
              handle, constant_op.constant([[6]], dtype=dtypes.int32)))
      sess.run(
          resource_variable_ops.resource_scatter_div(
              handle, [0], constant_op.constant([[3]], dtype=dtypes.int32)))
      read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
      self.assertAllEqual(sess.run(read), [[2]])

  def testScatterMin(self):
    with self.test_session() as sess, self.test_scope():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1, 1])
      sess.run(
          resource_variable_ops.assign_variable_op(
              handle, constant_op.constant([[6]], dtype=dtypes.int32)))
      sess.run(
          resource_variable_ops.resource_scatter_min(
              handle, [0], constant_op.constant([[3]], dtype=dtypes.int32)))
      read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
      self.assertEqual(sess.run(read), [[3]])

  def testScatterMax(self):
    with self.test_session() as sess, self.test_scope():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1, 1])
      sess.run(
          resource_variable_ops.assign_variable_op(
              handle, constant_op.constant([[6]], dtype=dtypes.int32)))
      sess.run(
          resource_variable_ops.resource_scatter_max(
              handle, [0], constant_op.constant([[3]], dtype=dtypes.int32)))
      read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
      self.assertEqual(sess.run(read), [[6]])

  def testScatterUpdate(self):
    with self.test_session() as sess, self.test_scope():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1, 1])
      sess.run(
          resource_variable_ops.assign_variable_op(
              handle, constant_op.constant([[6]], dtype=dtypes.int32)))
      sess.run(
          resource_variable_ops.resource_scatter_update(
              handle, [0], constant_op.constant([[3]], dtype=dtypes.int32)))
      read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
      self.assertEqual(sess.run(read), [[3]])

  def testScatterAddScalar(self):
    with self.test_session() as sess, self.test_scope():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1, 1])
      sess.run(
          resource_variable_ops.assign_variable_op(
              handle, constant_op.constant([[1]], dtype=dtypes.int32)))
      sess.run(
          resource_variable_ops.resource_scatter_add(
              handle, [0], constant_op.constant(2, dtype=dtypes.int32)))
      read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
      self.assertEqual(sess.run(read), [[3]])

  def testScatterSubScalar(self):
    with self.test_session() as sess, self.test_scope():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1, 1])
      sess.run(
          resource_variable_ops.assign_variable_op(
              handle, constant_op.constant([[1]], dtype=dtypes.int32)))
      sess.run(
          resource_variable_ops.resource_scatter_sub(
              handle, [0], constant_op.constant(2, dtype=dtypes.int32)))
      read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
      self.assertEqual(sess.run(read), [[-1]])

  def testScatterMulScalar(self):
    with self.test_session() as sess, self.test_scope():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1, 1])
      sess.run(
          resource_variable_ops.assign_variable_op(
              handle, constant_op.constant([[1]], dtype=dtypes.int32)))
      sess.run(
          resource_variable_ops.resource_scatter_mul(
              handle, [0], constant_op.constant(5, dtype=dtypes.int32)))
      read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
      self.assertEqual(sess.run(read), [[5]])

  def testScatterDivScalar(self):
    with self.test_session() as sess, self.test_scope():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1, 1])
      sess.run(
          resource_variable_ops.assign_variable_op(
              handle, constant_op.constant([[6]], dtype=dtypes.int32)))
      sess.run(
          resource_variable_ops.resource_scatter_div(
              handle, [0], constant_op.constant(3, dtype=dtypes.int32)))
      read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
      self.assertEqual(sess.run(read), [[2]])

  def testScatterMinScalar(self):
    with self.test_session() as sess, self.test_scope():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1, 1])
      sess.run(
          resource_variable_ops.assign_variable_op(
              handle, constant_op.constant([[6]], dtype=dtypes.int32)))
      sess.run(
          resource_variable_ops.resource_scatter_min(
              handle, [0], constant_op.constant(3, dtype=dtypes.int32)))
      read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
      self.assertEqual(sess.run(read), [[3]])

  def testScatterMaxScalar(self):
    with self.test_session() as sess, self.test_scope():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1, 1])
      sess.run(
          resource_variable_ops.assign_variable_op(
              handle, constant_op.constant([[6]], dtype=dtypes.int32)))
      sess.run(
          resource_variable_ops.resource_scatter_max(
              handle, [0], constant_op.constant(3, dtype=dtypes.int32)))
      read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
      self.assertEqual(sess.run(read), [[6]])

  def testScatterNdAddOps(self):
    with self.test_session() as sess, self.test_scope():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.float32, shape=[8])
      sess.run(
          resource_variable_ops.assign_variable_op(
              handle, constant_op.constant([1] * 8, dtype=dtypes.float32)))
      indices = constant_op.constant([[4], [3], [1], [7]], dtype=dtypes.int32)
      updates = constant_op.constant([9, 10, 11, 12], dtype=dtypes.float32)
      expected = np.array([1, 12, 1, 11, 10, 1, 1, 13])
      sess.run(gen_state_ops.resource_scatter_nd_add(handle, indices, updates))
      read = resource_variable_ops.read_variable_op(
          handle, dtype=dtypes.float32)
      self.assertAllClose(expected, sess.run(read))

  def testScatterNdUpdateAddOps(self):
    with self.test_session() as sess, self.test_scope():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.float32, shape=[8])
      sess.run(
          resource_variable_ops.assign_variable_op(
              handle, constant_op.constant([1] * 8, dtype=dtypes.float32)))
      indices = constant_op.constant([[4], [3], [1], [7]], dtype=dtypes.int32)
      updates = constant_op.constant([9, 10, 11, 12], dtype=dtypes.float32)
      expected = np.array([1, 11, 1, 10, 9, 1, 1, 12])
      sess.run(
          gen_state_ops.resource_scatter_nd_update(handle, indices, updates))
      read = resource_variable_ops.read_variable_op(
          handle, dtype=dtypes.float32)
      self.assertAllClose(expected, sess.run(read))


class StridedSliceAssignChecker(object):
  """Compares the results of a slice assignment using Tensorflow and numpy."""

  def __init__(self, test, x, dtype):
    self.dtype = dtype
    self.test = test
    self.x_np = np.array(x).astype(dtype)
    # Randomly start on mode 0 or 1.
    self.which_mode = np.random.randint(2, size=1)[0]

  def __setitem__(self, index, value):
    self.which_mode = 1 - self.which_mode
    value = np.array(value).astype(self.dtype)

    with self.test.test_session() as sess, self.test.test_scope():
      x = constant_op.constant(self.x_np, dtype=self.dtype)
      var = resource_variable_ops.ResourceVariable(x)
      sess.run(variables.variables_initializer([var]))

      if self.which_mode == 0:
        val = sess.run(var[index].assign(value))
      else:
        assert self.which_mode == 1
        val = sess.run(state_ops.assign(var[index], value))
      valnp = np.copy(self.x_np)
      valnp[index] = np.array(value)
      self.test.assertAllEqual(val, valnp)


class SliceAssignTest(xla_test.XLATestCase):

  def testSliceAssign(self):
    for dtype in self.numeric_types:
      checker = StridedSliceAssignChecker(
          self, [[1, 2, 3], [4, 5, 6]], dtype=dtype)
      # No-op assignment
      checker[:] = [[10, 20, 30], [40, 50, 60]]
      # Checks trivial (1,1) shape tensor
      checker[1:2, 1:2] = [[66]]
      # shrink shape changes
      checker[1:2, 1] = [66]
      checker[1, 1:2] = [66]
      if dtype != dtypes.bfloat16.as_numpy_dtype:
        # TODO(b/68813416): valnp call above results in an ndarray and not a
        # number for bfloat16s.
        checker[1, 1] = 66
      # newaxis shape changes
      checker[:, None, :] = [[[10, 20, 30]], [[40, 50, 50]]]
      # shrink and newaxis
      checker[None, None, 0, 0:1] = [[[99]]]
      # Non unit strides
      checker[::1, 1::-1] = [[3, 33], [4, 44]]
      # degenerate interval
      checker[8:10, 0] = []
      checker[8:10, 8:10] = [[]]

      # Assign vector to scalar (rank-0) using newaxis
      checker2 = StridedSliceAssignChecker(self, 222, dtype=dtype)
      if dtype != dtypes.bfloat16.as_numpy_dtype:
        # TODO(b/68813416): valnp call above results in an ndarray and not a
        # number for bfloat16s.
        checker2[()] = 6  # no indices
        checker2[...] = 6  # ellipsis
      checker2[None] = [6]  # new axis

  def testUninitialized(self):
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 "uninitialized variable"):
      with self.test_session() as sess, self.test_scope():
        v = resource_variable_ops.ResourceVariable([1, 2])
        sess.run(v[:].assign([1, 2]))


if __name__ == "__main__":
  googletest.main()
