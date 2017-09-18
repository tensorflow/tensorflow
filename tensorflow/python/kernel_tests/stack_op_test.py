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
"""Functional tests for Pack Op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def np_split_squeeze(array, axis):
  axis_len = array.shape[axis]
  return [
      np.squeeze(
          arr, axis=(axis,)) for arr in np.split(
              array, axis_len, axis=axis)
  ]


class StackOpTest(test.TestCase):

  def testSimple(self):
    np.random.seed(7)
    with self.test_session(use_gpu=True):
      for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2):
        for dtype in [np.bool, np.float32, np.int32, np.int64]:
          data = np.random.randn(*shape).astype(dtype)
          # Convert [data[0], data[1], ...] separately to tensorflow
          # TODO(irving): Remove list() once we handle maps correctly
          xs = list(map(constant_op.constant, data))
          # Pack back into a single tensorflow tensor
          c = array_ops.stack(xs)
          self.assertAllEqual(c.eval(), data)

  def testSimpleParallel(self):
    np.random.seed(7)
    with self.test_session(use_gpu=True):
      for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2):
        data = np.random.randn(*shape).astype(np.float32)
        xs = list(map(constant_op.constant, data))
        c = array_ops.parallel_stack(xs)
        self.assertAllEqual(c.eval(), data)

  def testConst(self):
    np.random.seed(7)
    with self.test_session(use_gpu=True):
      for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2):
        for dtype in [np.bool, np.float32, np.int32, np.int64]:
          data = np.random.randn(*shape).astype(dtype)
          # Pack back into a single tensorflow tensor directly using np array
          c = array_ops.stack(data)
          # This is implemented via a Const:
          self.assertEqual(c.op.type, "Const")
          self.assertAllEqual(c.eval(), data)

          # Python lists also work for 1-D case:
          if len(shape) == 1:
            data_list = list(data)
            cl = array_ops.stack(data_list)
            self.assertEqual(cl.op.type, "Const")
            self.assertAllEqual(cl.eval(), data)

        # Verify that shape induction works with shapes produced via const stack
        a = constant_op.constant([1, 2, 3, 4, 5, 6])
        b = array_ops.reshape(a, array_ops.stack([2, 3]))
        self.assertAllEqual(b.get_shape(), [2, 3])

  def testConstParallel(self):
    np.random.seed(7)
    with self.test_session(use_gpu=True):
      for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2):
        data = np.random.randn(*shape).astype(np.float32)
        if len(shape) == 1:
          data_list = list(data)
          cl = array_ops.parallel_stack(data_list)
          self.assertAllEqual(cl.eval(), data)

        data = np.random.randn(*shape).astype(np.float32)
        c = array_ops.parallel_stack(data)
        self.assertAllEqual(c.eval(), data)

  def testGradientsAxis0(self):
    np.random.seed(7)
    for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2):
      data = np.random.randn(*shape)
      shapes = [shape[1:]] * shape[0]
      with self.test_session(use_gpu=True):
        # TODO(irving): Remove list() once we handle maps correctly
        xs = list(map(constant_op.constant, data))
        c = array_ops.stack(xs)
        err = gradient_checker.compute_gradient_error(xs, shapes, c, shape)
        self.assertLess(err, 1e-6)

  def testGradientsAxis1(self):
    np.random.seed(7)
    for shape in (2, 3), (3, 2), (4, 3, 2):
      data = np.random.randn(*shape)
      shapes = [shape[1:]] * shape[0]
      out_shape = list(shape[1:])
      out_shape.insert(1, shape[0])
      with self.test_session(use_gpu=True):
        # TODO(irving): Remove list() once we handle maps correctly
        xs = list(map(constant_op.constant, data))
        c = array_ops.stack(xs, axis=1)
        err = gradient_checker.compute_gradient_error(xs, shapes, c, out_shape)
        self.assertLess(err, 1e-6)

  def testZeroSize(self):
    # Verify that stack doesn't crash for zero size inputs
    with self.test_session(use_gpu=True):
      for shape in (0,), (3, 0), (0, 3):
        x = np.zeros((2,) + shape).astype(np.int32)
        p = array_ops.stack(list(x)).eval()
        self.assertAllEqual(p, x)

        p = array_ops.parallel_stack(list(x)).eval()
        self.assertAllEqual(p, x)

  def testAxis0Default(self):
    with self.test_session(use_gpu=True):
      t = [constant_op.constant([1, 2, 3]), constant_op.constant([4, 5, 6])]
      stacked = array_ops.stack(t).eval()
      parallel_stacked = array_ops.parallel_stack(t).eval()

    self.assertAllEqual(stacked, np.array([[1, 2, 3], [4, 5, 6]]))
    self.assertAllEqual(parallel_stacked, np.array([[1, 2, 3], [4, 5, 6]]))

  def testAgainstNumpy(self):
    # For 1 to 5 dimensions.
    for i in range(1, 6):
      expected = np.random.random(np.random.permutation(i) + 1)

      # For all the possible axis to split it, including negative indices.
      for j in range(-i, i):
        test_arrays = np_split_squeeze(expected, j)

        with self.test_session(use_gpu=True):
          actual_pack = array_ops.stack(test_arrays, axis=j)
          self.assertEqual(expected.shape, actual_pack.get_shape())
          actual_pack = actual_pack.eval()

          actual_stack = array_ops.stack(test_arrays, axis=j)
          self.assertEqual(expected.shape, actual_stack.get_shape())
          actual_stack = actual_stack.eval()

        self.assertNDArrayNear(expected, actual_stack, 1e-6)

  def testDimOutOfRange(self):
    t = [constant_op.constant([1, 2, 3]), constant_op.constant([4, 5, 6])]
    with self.assertRaisesRegexp(ValueError, r"axis = 2 not in \[-2, 2\)"):
      array_ops.stack(t, axis=2)

  def testDimOutOfNegativeRange(self):
    t = [constant_op.constant([1, 2, 3]), constant_op.constant([4, 5, 6])]
    with self.assertRaisesRegexp(ValueError, r"axis = -3 not in \[-2, 2\)"):
      array_ops.stack(t, axis=-3)


class AutomaticPackingTest(test.TestCase):

  def testSimple(self):
    with self.test_session(use_gpu=True):
      self.assertAllEqual(
          [1, 0, 2],
          ops.convert_to_tensor([1, constant_op.constant(0), 2]).eval())
      self.assertAllEqual([[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                          ops.convert_to_tensor(
                              [[0, 0, 0], [0, constant_op.constant(1), 0],
                               [0, 0, 0]]).eval())
      self.assertAllEqual([[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                          ops.convert_to_tensor(
                              [[0, 0, 0], constant_op.constant([0, 1, 0]),
                               [0, 0, 0]]).eval())
      self.assertAllEqual([[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                          ops.convert_to_tensor([
                              constant_op.constant([0, 0, 0]),
                              constant_op.constant([0, 1, 0]),
                              constant_op.constant([0, 0, 0])
                          ]).eval())

  def testWithNDArray(self):
    with self.test_session(use_gpu=True):
      result = ops.convert_to_tensor([[[0., 0.],
                                       constant_op.constant([1., 1.])],
                                      np.array(
                                          [[2., 2.], [3., 3.]],
                                          dtype=np.float32)])
      self.assertAllEqual([[[0., 0.], [1., 1.]], [[2., 2.], [3., 3.]]],
                          result.eval())

  def testVariable(self):
    with self.test_session(use_gpu=True):
      v = variables.Variable(17)
      result = ops.convert_to_tensor([[0, 0, 0], [0, v, 0], [0, 0, 0]])
      v.initializer.run()
      self.assertAllEqual([[0, 0, 0], [0, 17, 0], [0, 0, 0]], result.eval())

      v.assign(38).op.run()
      self.assertAllEqual([[0, 0, 0], [0, 38, 0], [0, 0, 0]], result.eval())

  def testDtype(self):
    t_0 = ops.convert_to_tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
    self.assertEqual(dtypes.float32, t_0.dtype)

    t_1 = ops.convert_to_tensor([[0., 0., 0.], constant_op.constant(
        [0., 0., 0.], dtype=dtypes.float64), [0., 0., 0.]])
    self.assertEqual(dtypes.float64, t_1.dtype)

    t_2 = ops.convert_to_tensor(
        [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]], dtype=dtypes.float64)
    self.assertEqual(dtypes.float64, t_2.dtype)

    with self.assertRaises(TypeError):
      ops.convert_to_tensor([
          constant_op.constant(
              [0., 0., 0.], dtype=dtypes.float32), constant_op.constant(
                  [0., 0., 0.], dtype=dtypes.float64), [0., 0., 0.]
      ])

    with self.assertRaises(TypeError):
      ops.convert_to_tensor(
          [[0., 0., 0.], constant_op.constant(
              [0., 0., 0.], dtype=dtypes.float64), [0., 0., 0.]],
          dtype=dtypes.float32)

    with self.assertRaises(TypeError):
      ops.convert_to_tensor(
          [constant_op.constant(
              [0., 0., 0.], dtype=dtypes.float64)],
          dtype=dtypes.float32)

  def testPlaceholder(self):
    with self.test_session(use_gpu=True):
      # Test using placeholder with a defined shape.
      ph_0 = array_ops.placeholder(dtypes.int32, shape=[])
      result_0 = ops.convert_to_tensor([[0, 0, 0], [0, ph_0, 0], [0, 0, 0]])
      self.assertAllEqual(
          [[0, 0, 0], [0, 1, 0], [0, 0, 0]], result_0.eval(feed_dict={ph_0: 1}))
      self.assertAllEqual(
          [[0, 0, 0], [0, 2, 0], [0, 0, 0]], result_0.eval(feed_dict={ph_0: 2}))

      # Test using placeholder with an undefined shape.
      ph_1 = array_ops.placeholder(dtypes.int32)
      result_1 = ops.convert_to_tensor([[0, 0, 0], [0, ph_1, 0], [0, 0, 0]])
      self.assertAllEqual(
          [[0, 0, 0], [0, 1, 0], [0, 0, 0]], result_1.eval(feed_dict={ph_1: 1}))
      self.assertAllEqual(
          [[0, 0, 0], [0, 2, 0], [0, 0, 0]], result_1.eval(feed_dict={ph_1: 2}))

  def testShapeErrors(self):
    # Static shape error.
    ph_0 = array_ops.placeholder(dtypes.int32, shape=[1])
    with self.assertRaises(ValueError):
      ops.convert_to_tensor([[0, 0, 0], [0, ph_0, 0], [0, 0, 0]])

    # Dynamic shape error.
    ph_1 = array_ops.placeholder(dtypes.int32)
    result_1 = ops.convert_to_tensor([[0, 0, 0], [0, ph_1, 0], [0, 0, 0]])
    with self.test_session(use_gpu=True):
      with self.assertRaises(errors_impl.InvalidArgumentError):
        result_1.eval(feed_dict={ph_1: [1]})


if __name__ == "__main__":
  test.main()
