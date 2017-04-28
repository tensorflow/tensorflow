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
"""Functional tests for XLA Concat Op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests.xla_test import XLATestCase
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


class ConcatTest(XLATestCase):

  def testHStack(self):
    with self.test_session():
      p1 = array_ops.placeholder(dtypes.float32, shape=[4, 4])
      p2 = array_ops.placeholder(dtypes.float32, shape=[4, 4])
      with self.test_scope():
        c = array_ops.concat([p1, p2], 0)
      params = {
          p1: np.random.rand(4, 4).astype("f"),
          p2: np.random.rand(4, 4).astype("f")
      }
      result = c.eval(feed_dict=params)

    self.assertEqual(result.shape, c.get_shape())
    self.assertAllEqual(result[:4, :], params[p1])
    self.assertAllEqual(result[4:, :], params[p2])

  def testVStack(self):
    with self.test_session():
      p1 = array_ops.placeholder(dtypes.float32, shape=[4, 4])
      p2 = array_ops.placeholder(dtypes.float32, shape=[4, 4])
      with self.test_scope():
        c = array_ops.concat([p1, p2], 1)
      params = {
          p1: np.random.rand(4, 4).astype("f"),
          p2: np.random.rand(4, 4).astype("f")
      }
      result = c.eval(feed_dict=params)

    self.assertEqual(result.shape, c.get_shape())
    self.assertAllEqual(result[:, :4], params[p1])
    self.assertAllEqual(result[:, 4:], params[p2])

  def testInt32(self):
    with self.test_session():
      p1 = np.random.rand(2, 3).astype("i")
      p2 = np.random.rand(2, 3).astype("i")
      x1 = constant_op.constant(p1)
      x2 = constant_op.constant(p2)
      with self.test_scope():
        c = array_ops.concat([x1, x2], 0)
      result = c.eval()
    self.assertAllEqual(result[:2, :], p1)
    self.assertAllEqual(result[2:, :], p2)

  def _testRandom(self, dtype):
    # Random dims of rank 5
    shape = np.random.randint(1, 5, size=5)
    # Random number of tensors, but always > 1.
    num_tensors = np.random.randint(2, 10)
    # Random dim to concat on
    concat_dim = np.random.randint(5)
    params = {}
    if dtype == dtypes.bfloat16:
      dtype_feed = dtypes.float32
    else:
      dtype_feed = dtype
    with self.test_session():
      p = []
      for i in np.arange(num_tensors):
        input_shape = shape
        input_shape[concat_dim] = np.random.randint(1, 5)
        placeholder = array_ops.placeholder(dtype_feed, shape=input_shape)
        p.append(placeholder)

        t = dtype_feed.as_numpy_dtype
        params[placeholder] = np.random.rand(*input_shape).astype(t)

      if dtype != dtype_feed:
        concat_inputs = [math_ops.cast(p_i, dtype) for p_i in p]
      else:
        concat_inputs = p
      with self.test_scope():
        c = array_ops.concat(concat_inputs, concat_dim)
        if dtype != dtype_feed:
          c = math_ops.cast(c, dtype_feed)
      result = c.eval(feed_dict=params)

    self.assertEqual(result.shape, c.get_shape())
    cur_offset = 0

    for i in np.arange(num_tensors):
      # The index into the result is the ':' along all dimensions
      # except the concat_dim. slice(0, size) is used for ':', and
      # a list of slices is used to index into result.
      ind = [slice(0, params[p[i]].shape[j]) for j in np.arange(5)]
      ind[concat_dim] = slice(cur_offset,
                              cur_offset + params[p[i]].shape[concat_dim])
      cur_offset += params[p[i]].shape[concat_dim]
      if dtype == dtype_feed:
        self.assertAllEqual(result[ind], params[p[i]])
      else:
        self.assertAllClose(result[ind], params[p[i]], 0.01)

  def testRandom(self):
    self._testRandom(dtypes.float32)
    self._testRandom(dtypes.int32)

  def _testGradientsSimple(self):
    with self.test_session():
      inp = []
      inp_tensors = []
      with self.test_scope():
        for x in [1, 2, 6]:
          shape = [10, x, 2]
          t = np.random.rand(*shape).astype("f")
          inp.append(t)
          inp_tensors.append(
              constant_op.constant(
                  [float(y) for y in t.flatten()],
                  shape=shape,
                  dtype=dtypes.float32))
        c = array_ops.concat(inp_tensors, 1)
        output_shape = [10, 9, 2]
        grad_inp = np.random.rand(*output_shape).astype("f")
        grad_tensor = constant_op.constant(
            [float(x) for x in grad_inp.flatten()], shape=output_shape)
        grad = gradients_impl.gradients([c], inp_tensors, [grad_tensor])
        concated_grad = array_ops.concat(grad, 1)
      result = concated_grad.eval()
    self.assertAllEqual(result, grad_inp)

  def testGradientsSimpleAll(self):
    self._testGradientsSimple()

  def _testGradientsFirstDim(self):
    with self.test_session():
      inp = []
      inp_tensors = []
      with self.test_scope():
        for x in [1, 2, 6]:
          shape = [x, 10, 2]
          t = np.random.rand(*shape).astype("f")
          inp.append(t)
          inp_tensors.append(
              constant_op.constant(
                  [float(y) for y in t.flatten()],
                  shape=shape,
                  dtype=dtypes.float32))
        c = array_ops.concat(inp_tensors, 0)
        output_shape = [9, 10, 2]
        grad_inp = np.random.rand(*output_shape).astype("f")
        grad_tensor = constant_op.constant(
            [float(x) for x in grad_inp.flatten()], shape=output_shape)
        grad = gradients_impl.gradients([c], inp_tensors, [grad_tensor])
        concated_grad = array_ops.concat(grad, 0)
        result = concated_grad.eval()

    self.assertAllEqual(result, grad_inp)

  def testGradientsFirstDimAll(self):
    self._testGradientsFirstDim()

  def _testGradientsLastDim(self):
    with self.test_session():
      inp = []
      inp_tensors = []
      with self.test_scope():
        for x in [1, 2, 6]:
          shape = [10, 2, x]
          t = np.random.rand(*shape).astype("f")
          inp.append(t)
          inp_tensors.append(
              constant_op.constant(
                  [float(y) for y in t.flatten()],
                  shape=shape,
                  dtype=dtypes.float32))
        c = array_ops.concat(inp_tensors, 2)
        output_shape = [10, 2, 9]
        grad_inp = np.random.rand(*output_shape).astype("f")
        grad_tensor = constant_op.constant(
            [float(x) for x in grad_inp.flatten()], shape=output_shape)
        grad = gradients_impl.gradients([c], inp_tensors, [grad_tensor])
        concated_grad = array_ops.concat(grad, 2)
        result = concated_grad.eval()

    self.assertAllEqual(result, grad_inp)

  def testGradientsLastDimAll(self):
    self._testGradientsLastDim()

  def _RunAndVerifyGradientsRandom(self):
    # Random dims of rank 5
    input_shape = np.random.randint(1, 5, size=5)
    # Random number of tensors
    num_tensors = np.random.randint(1, 10)
    # Random dim to concat on
    concat_dim = np.random.randint(5)
    concat_dim_sizes = np.random.randint(1, 5, size=num_tensors)
    with self.test_session():
      inp = []
      inp_tensors = []
      with self.test_scope():
        for x in concat_dim_sizes:
          shape = input_shape
          shape[concat_dim] = x
          t = np.random.rand(*shape).astype("f")
          inp.append(t)
          inp_tensors.append(
              constant_op.constant(
                  [float(y) for y in t.flatten()],
                  shape=shape,
                  dtype=dtypes.float32))
        c = array_ops.concat(inp_tensors, concat_dim)
        output_shape = input_shape
        output_shape[concat_dim] = concat_dim_sizes.sum()
        grad_inp = np.random.rand(*output_shape).astype("f")
        grad_tensor = constant_op.constant(
            [float(x) for x in grad_inp.flatten()], shape=output_shape)
        grad = gradients_impl.gradients([c], inp_tensors, [grad_tensor])
        concated_grad = array_ops.concat(grad, concat_dim)
        result = concated_grad.eval()

    self.assertAllEqual(result, grad_inp)

  def testGradientsRandom(self):
    for _ in range(5):
      self._RunAndVerifyGradientsRandom()

  # Re-enable once zero-element Retvals are handled correctly.
  def DISABLED_testZeroSize(self):
    # Verify that concat doesn't crash and burn for zero size inputs
    np.random.seed(7)
    with self.test_session() as sess:
      with self.test_scope():
        for shape0 in (), (2,):
          axis = len(shape0)
          for shape1 in (), (3,):
            for n0 in 0, 1, 2:
              for n1 in 0, 1, 2:
                x0 = np.random.randn(*(shape0 + (n0,) + shape1))
                x1 = np.random.randn(*(shape0 + (n1,) + shape1))
                correct = np.concatenate([x0, x1], axis=axis)
                # TODO(irving): Make tf.concat handle map, then drop list().
                xs = list(map(constant_op.constant, [x0, x1]))
                c = array_ops.concat(xs, axis)
                self.assertAllEqual(c.eval(), correct)
                # Check gradients
                dc = np.random.randn(*c.get_shape().as_list())
                dxs = sess.run(gradients_impl.gradients(c, xs, dc))
                self.assertAllEqual(dc, np.concatenate(dxs, axis=axis))

  def testConcatTuple(self):
    c1 = np.random.rand(4, 4).astype(np.float32)
    c2 = np.random.rand(4, 4).astype(np.float32)
    with self.test_session():
      with self.test_scope():
        concat_list_t = array_ops.concat([c1, c2], 0)
        concat_tuple_t = array_ops.concat((c1, c2), 0)
      self.assertAllEqual(concat_list_t.eval(), concat_tuple_t.eval())

  def testConcatNoScalars(self):
    with self.test_session():
      with self.test_scope():
        scalar = constant_op.constant(7)
        dim = array_ops.placeholder(dtypes.int32)
        with self.assertRaisesRegexp(
            ValueError, r"Can't concatenate scalars \(use tf\.stack instead\)"):
          array_ops.concat([scalar, scalar, scalar], dim)


class ConcatOffsetTest(XLATestCase):

  def testBasic(self):
    with self.test_session() as sess:
      with self.test_scope():
        cdim = constant_op.constant(1, dtypes.int32)
        s0 = constant_op.constant([2, 3, 5], dtypes.int32)
        s1 = constant_op.constant([2, 7, 5], dtypes.int32)
        s2 = constant_op.constant([2, 20, 5], dtypes.int32)
        off = gen_array_ops._concat_offset(cdim, [s0, s1, s2])
        ans = sess.run(off)
        self.assertAllEqual(ans, [[0, 0, 0], [0, 3, 0], [0, 10, 0]])


class PackTest(XLATestCase):

  def testBasic(self):
    with self.test_session() as sess:
      with self.test_scope():
        s0 = constant_op.constant([2, 3, 5], dtypes.int32)
        s1 = constant_op.constant([2, 7, 5], dtypes.int32)
        s2 = constant_op.constant([2, 20, 5], dtypes.int32)
        packed = array_ops.stack([s0, s1, s2])
        ans = sess.run(packed)
        self.assertAllEqual(ans, [[2, 3, 5], [2, 7, 5], [2, 20, 5]])

  def testScalars(self):
    with self.test_session() as sess:
      with self.test_scope():
        s0 = constant_op.constant(2, dtypes.int32)
        s1 = constant_op.constant(3, dtypes.int32)
        s2 = constant_op.constant(5, dtypes.int32)
        packed = array_ops.stack([s0, s1, s2])
        ans = sess.run(packed)
        self.assertAllEqual(ans, [2, 3, 5])

  def testEmpty(self):
    with self.test_session() as sess:
      with self.test_scope():
        s0 = constant_op.constant([[]], dtypes.int32)
        s1 = constant_op.constant([[]], dtypes.int32)
        s2 = constant_op.constant([[]], dtypes.int32)
        packed = array_ops.stack([s0, s1, s2])
        ans = sess.run(packed)
        self.assertAllEqual(ans, [[[]], [[]], [[]]])


if __name__ == "__main__":
  googletest.main()
