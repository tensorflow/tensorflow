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

"""Functional tests for Concat Op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import gen_array_ops


class ConcatOpTest(tf.test.TestCase):

  def testHStack(self):
    with self.test_session():
      p1 = tf.placeholder(tf.float32, shape=[4, 4])
      p2 = tf.placeholder(tf.float32, shape=[4, 4])
      c = tf.concat(0, [p1, p2])
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
      p1 = tf.placeholder(tf.float32, shape=[4, 4])
      p2 = tf.placeholder(tf.float32, shape=[4, 4])
      c = tf.concat(1, [p1, p2])
      params = {
          p1: np.random.rand(4, 4).astype("f"),
          p2: np.random.rand(4, 4).astype("f")
          }
      result = c.eval(feed_dict=params)

    self.assertEqual(result.shape, c.get_shape())
    self.assertAllEqual(result[:, :4], params[p1])
    self.assertAllEqual(result[:, 4:], params[p2])

  def testInt32GPU(self):
    with self.test_session(use_gpu=True):
      p1 = np.random.rand(2, 3).astype("i")
      p2 = np.random.rand(2, 3).astype("i")
      x1 = tf.constant(p1)
      x2 = tf.constant(p2)
      c = tf.concat(0, [x1, x2])
      result = c.eval()
    self.assertAllEqual(result[:2, :], p1)
    self.assertAllEqual(result[2:, :], p2)

  def testRefType(self):
    with self.test_session():
      p1 = np.random.rand(4, 4).astype("f")
      p2 = np.random.rand(4, 4).astype("f")
      v1 = tf.Variable(p1)
      v2 = tf.Variable(p2)
      c = tf.concat(0, [v1, v2])
      tf.initialize_all_variables().run()
      result = c.eval()

    self.assertEqual(result.shape, c.get_shape())
    self.assertAllEqual(result[:4, :], p1)
    self.assertAllEqual(result[4:, :], p2)

  def _testRandom(self, dtype, use_gpu=False):
    # Random dims of rank 5
    shape = np.random.randint(1, 5, size=5)
    # Random number of tensors, but always > 1.
    num_tensors = np.random.randint(2, 10)
    # Random dim to concat on
    concat_dim = np.random.randint(5)
    params = {}
    if dtype == tf.bfloat16:
      dtype_feed = tf.float32
    else:
      dtype_feed = dtype
    with self.test_session(use_gpu=use_gpu):
      p = []
      for i in np.arange(num_tensors):
        input_shape = shape
        input_shape[concat_dim] = np.random.randint(1, 5)
        placeholder = tf.placeholder(dtype_feed, shape=input_shape)
        p.append(placeholder)

        t = dtype_feed.as_numpy_dtype
        params[placeholder] = np.random.rand(*input_shape).astype(t)

      if dtype != dtype_feed:
        concat_inputs = [tf.cast(p_i, dtype) for p_i in p]
      else:
        concat_inputs = p
      c = tf.concat(concat_dim, concat_inputs)
      if dtype != dtype_feed:
        c = tf.cast(c, dtype_feed)
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
    self._testRandom(tf.float32)
    self._testRandom(tf.int16)
    self._testRandom(tf.int32, use_gpu=True)
    self._testRandom(tf.bfloat16)
    self._testRandom(tf.bfloat16, use_gpu=True)

  def testInvalidConcatDimTypeAndShape(self):
    a = tf.Variable(tf.constant(1.0, shape=[1]))
    b = tf.Variable(tf.constant(2.0, shape=[1]))
    with self.assertRaises(ValueError):
      tf.concat(a, b)
    with self.assertRaises(TypeError):
      tf.concat(4.2, 1)
    with self.assertRaises(ValueError):
      tf.concat(a, 1)
    with self.assertRaises(TypeError):
      tf.concat(a, [a, b])
    with self.assertRaises(ValueError):
      tf.concat([3], [a, b])
    with self.assertRaises(ValueError):
      tf.concat(0, [])
    # An integer tensor for shape dim should throw no error.
    tf.concat(tf.constant(0, shape=[]), 1)
    # A non-scalar tensor for shape should throw ValueError.
    with self.assertRaises(ValueError):
      tf.concat(tf.constant(0, shape=[1]), 1)

  def _testGradientsSimple(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      inp = []
      inp_tensors = []
      for x in [1, 2, 6]:
        shape = [10, x, 2]
        t = np.random.rand(*shape).astype("f")
        inp.append(t)
        inp_tensors.append(
            tf.constant([float(y) for y in t.flatten()],
                                 shape=shape, dtype=tf.float32))
      c = tf.concat(1, inp_tensors)
      output_shape = [10, 9, 2]
      grad_inp = np.random.rand(*output_shape).astype("f")
      grad_tensor = tf.constant([float(x) for x in grad_inp.flatten()],
                                         shape=output_shape)
      grad = tf.gradients([c], inp_tensors, [grad_tensor])
      concated_grad = tf.concat(1, grad)
      result = concated_grad.eval()
    self.assertAllEqual(result, grad_inp)

  def testGradientsSimpleAll(self):
    self._testGradientsSimple(use_gpu=True)
    self._testGradientsSimple(use_gpu=False)

  def _testGradientsFirstDim(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      inp = []
      inp_tensors = []
      for x in [1, 2, 6]:
        shape = [x, 10, 2]
        t = np.random.rand(*shape).astype("f")
        inp.append(t)
        inp_tensors.append(
            tf.constant([float(y) for y in t.flatten()],
                                 shape=shape, dtype=tf.float32))
      c = tf.concat(0, inp_tensors)
      output_shape = [9, 10, 2]
      grad_inp = np.random.rand(*output_shape).astype("f")
      grad_tensor = tf.constant([float(x) for x in grad_inp.flatten()],
                                         shape=output_shape)
      grad = tf.gradients([c], inp_tensors, [grad_tensor])
      concated_grad = tf.concat(0, grad)
      result = concated_grad.eval()

    self.assertAllEqual(result, grad_inp)

  def testGradientsFirstDimAll(self):
    self._testGradientsFirstDim(use_gpu=False)
    self._testGradientsFirstDim(use_gpu=True)

  def _testGradientsLastDim(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      inp = []
      inp_tensors = []
      for x in [1, 2, 6]:
        shape = [10, 2, x]
        t = np.random.rand(*shape).astype("f")
        inp.append(t)
        inp_tensors.append(
            tf.constant([float(y) for y in t.flatten()],
                                 shape=shape, dtype=tf.float32))
      c = tf.concat(2, inp_tensors)
      output_shape = [10, 2, 9]
      grad_inp = np.random.rand(*output_shape).astype("f")
      grad_tensor = tf.constant([float(x) for x in grad_inp.flatten()],
                                         shape=output_shape)
      grad = tf.gradients([c], inp_tensors, [grad_tensor])
      concated_grad = tf.concat(2, grad)
      result = concated_grad.eval()

    self.assertAllEqual(result, grad_inp)

  def testGradientsLastDimAll(self):
    self._testGradientsLastDim(use_gpu=False)
    self._testGradientsLastDim(use_gpu=True)

  def _RunAndVerifyGradientsRandom(self, use_gpu):
    # Random dims of rank 5
    input_shape = np.random.randint(1, 5, size=5)
    # Random number of tensors
    num_tensors = np.random.randint(1, 10)
    # Random dim to concat on
    concat_dim = np.random.randint(5)
    concat_dim_sizes = np.random.randint(1, 5, size=num_tensors)
    with self.test_session(use_gpu=use_gpu):
      inp = []
      inp_tensors = []
      for x in concat_dim_sizes:
        shape = input_shape
        shape[concat_dim] = x
        t = np.random.rand(*shape).astype("f")
        inp.append(t)
        inp_tensors.append(
            tf.constant([float(y) for y in t.flatten()],
                                 shape=shape, dtype=tf.float32))
      c = tf.concat(concat_dim, inp_tensors)
      output_shape = input_shape
      output_shape[concat_dim] = concat_dim_sizes.sum()
      grad_inp = np.random.rand(*output_shape).astype("f")
      grad_tensor = tf.constant([float(x) for x in grad_inp.flatten()],
                                         shape=output_shape)
      grad = tf.gradients([c], inp_tensors, [grad_tensor])
      concated_grad = tf.concat(concat_dim, grad)
      result = concated_grad.eval()

    self.assertAllEqual(result, grad_inp)

  def testGradientsRandom(self):
    for _ in range(5):
      self._RunAndVerifyGradientsRandom(use_gpu=False)
      self._RunAndVerifyGradientsRandom(use_gpu=True)

  def testShapeError(self):
    # Rank doesn't match.
    with self.assertRaises(ValueError):
      tf.concat(1, [tf.constant(10.0, shape=[4, 4, 4, 4]),
                    tf.constant(20.0, shape=[4, 4, 4])])

    # Dimensions don't match in a non-concat dim.
    with self.assertRaises(ValueError):
      tf.concat(1, [tf.constant(10.0, shape=[1, 2, 1]),
                    tf.constant(20.0, shape=[3, 2, 1])])

    # concat_dim out of range.
    with self.assertRaises(ValueError):
      tf.concat(3, [tf.constant(10.0, shape=[4, 4, 4]),
                    tf.constant(20.0, shape=[4, 4, 4])])

    # concat_dim < 0
    with self.assertRaises(ValueError):
      tf.concat(-1, [tf.constant(10.0, shape=[4, 4, 4]),
                     tf.constant(20.0, shape=[4, 4, 4])])

  def testShapeWithUnknownConcatDim(self):
    p1 = tf.placeholder(tf.float32)
    c1 = tf.constant(10.0, shape=[4, 4, 4, 4])
    p2 = tf.placeholder(tf.float32)
    c2 = tf.constant(20.0, shape=[4, 4, 4, 4])
    dim = tf.placeholder(tf.int32)
    concat = tf.concat(dim, [p1, c1, p2, c2])
    self.assertEqual(4, concat.get_shape().ndims)

    # All dimensions unknown.
    concat2 = tf.concat(dim, [p1, p2])
    self.assertEqual(None, concat2.get_shape())

    # Rank doesn't match.
    c3 = tf.constant(30.0, shape=[4, 4, 4])
    with self.assertRaises(ValueError):
      tf.concat(dim, [p1, c1, p2, c3])

  def testZeroSize(self):
    # Verify that concat doesn't crash and burn for zero size inputs
    np.random.seed(7)
    for use_gpu in False, True:
      with self.test_session(use_gpu=use_gpu) as sess:
        for shape0 in (), (2,):
          axis = len(shape0)
          for shape1 in (), (3,):
            for n0 in 0, 1, 2:
              for n1 in 0, 1, 2:
                x0 = np.random.randn(*(shape0 + (n0,) + shape1))
                x1 = np.random.randn(*(shape0 + (n1,) + shape1))
                correct = np.concatenate([x0, x1], axis=axis)
                # TODO(irving): Make tf.concat handle map, then drop list().
                xs = list(map(tf.constant, [x0, x1]))
                c = tf.concat(axis, xs)
                self.assertAllEqual(c.eval(), correct)
                # Check gradients
                dc = np.random.randn(*c.get_shape().as_list())
                dxs = sess.run(tf.gradients(c, xs, dc))
                self.assertAllEqual(dc, np.concatenate(dxs, axis=axis))

  def testTensorConcatDim0Grad(self):
    x_shapes = [[20, 7, 3], [10, 7, 3], [14, 7, 3]]
    output_shape = [44, 7, 3]
    x_vals = [np.random.random_sample(x_shape).astype(
        np.float64) for x_shape in x_shapes]
    with self.test_session():
      xs = [tf.constant(x_val) for x_val in x_vals]
      output = tf.concat(0, xs)
      err = tf.test.compute_gradient_error(xs, x_shapes, output, output_shape)
    self.assertLess(err, 1e-11)

  def testTensorConcatDim1Grad(self):
    x_shapes = [[20, 7, 3], [20, 3, 3], [20, 1, 3]]
    output_shape = [20, 11, 3]
    x_vals = [np.random.random_sample(x_shape).astype(
        np.float64) for x_shape in x_shapes]
    with self.test_session():
      xs = [tf.constant(x_val) for x_val in x_vals]
      output = tf.concat(1, xs)
      err = tf.test.compute_gradient_error(xs, x_shapes, output, output_shape)
    self.assertLess(err, 1e-11)

  def testIndexedSlicesConcatDim0Grad(self):
    x_shapes = [[20, 7, 3], [10, 7, 3], [14, 7, 3]]
    output_shape = [4, 7, 3]
    x_vals = [np.random.random_sample(x_shape).astype(
        np.float64) for x_shape in x_shapes]
    with self.test_session():
      xs = [tf.constant(x_val) for x_val in x_vals]
      x_concat = tf.concat(0, xs)
      output = tf.gather(x_concat, [1, 2, 0, 5])
      err = tf.test.compute_gradient_error(xs, x_shapes, output, output_shape)
    self.assertLess(err, 1e-11)

  def testIndexedSlicesConcatDim1Grad(self):
    x_shapes = [[20, 7, 3], [20, 3, 3], [20, 1, 3]]
    output_shape = [4, 11, 3]
    x_vals = [np.random.random_sample(x_shape).astype(
        np.float64) for x_shape in x_shapes]
    with self.test_session():
      xs = [tf.constant(x_val) for x_val in x_vals]
      x_concat = tf.concat(1, xs)
      output = tf.gather(x_concat, [1, 2, 0, 5])
      err = tf.test.compute_gradient_error(xs, x_shapes, output, output_shape)
    self.assertLess(err, 1e-11)

  def testIndexedSlicesConcatDim2Grad(self):
    x_shapes = [[20, 7, 3], [20, 7, 1], [20, 7, 2]]
    output_shape = [4, 7, 6]
    x_vals = [np.random.random_sample(x_shape).astype(
        np.float64) for x_shape in x_shapes]
    with self.test_session():
      xs = [tf.constant(x_val) for x_val in x_vals]
      x_concat = tf.concat(2, xs)
      output = tf.gather(x_concat, [1, 2, 0, 5])
      err = tf.test.compute_gradient_error(xs, x_shapes, output, output_shape)
    self.assertLess(err, 1e-11)

  def testConcatTuple(self):
    c1 = np.random.rand(4, 4)
    c2 = np.random.rand(4, 4)
    with self.test_session():
      concat_list_t = tf.concat(0, [c1, c2])
      concat_tuple_t = tf.concat(0, (c1, c2))
      self.assertAllEqual(concat_list_t.eval(), concat_tuple_t.eval())

  def testConcatNoScalars(self):
    with self.test_session():
      scalar = tf.constant(7)
      dim = tf.placeholder(tf.int32)
      with self.assertRaisesRegexp(
          ValueError, r"Can't concatenate scalars \(use tf\.pack instead\)"):
        tf.concat(dim, [scalar, scalar, scalar])

  def testConcatGradNumNodes(self):
    g = tf.Graph()
    n = 10
    with g.as_default():
      x = tf.constant([1, 1])
      y = tf.concat(0, [x] * n)
      before = len(g.get_operations())
      _ = tf.gradients([y], [x], [y])
      after = len(g.get_operations())
      self.assertEqual(2 * n + 2, after - before)
      print("graph = ", [x.name for x in g.get_operations()])

  def testConcatLargeTensors(self):
    # CPU-only test, because it fails on GPUs with <= 4GB memory.
    with tf.device("/cpu:0"):
      a = tf.ones([2**31 + 6], dtype=tf.int8)
      b = tf.zeros([1024], dtype=tf.int8)
      onezeros = tf.concat(0, [a, b])
    with self.test_session(use_gpu=False):
      # TODO(dga):  Add more depth to this test to validate correctness,
      # not just non-crashingness, once other large tensor fixes have gone in.
      _ = onezeros.eval()


class ConcatOffsetTest(tf.test.TestCase):

  def testBasic(self):
    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu) as sess:
        cdim = tf.constant(1, tf.int32)
        s0 = tf.constant([2, 3, 5], tf.int32)
        s1 = tf.constant([2, 7, 5], tf.int32)
        s2 = tf.constant([2, 20, 5], tf.int32)
        off = gen_array_ops._concat_offset(cdim, [s0, s1, s2])
        ans = sess.run(off)
        self.assertAllEqual(ans, [[0, 0, 0], [0, 3, 0], [0, 10, 0]])

  def testNotVector(self):
    with self.test_session() as sess:
      cdim = tf.constant(1, tf.int32)
      s0 = tf.constant([[2, 3, 5]], tf.int32)
      s1 = tf.constant([[2, 7, 5]], tf.int32)
      off = gen_array_ops._concat_offset(cdim, [s0, s1])
      with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                   r"should be a vector"):
        sess.run(off)

  def testConcatDimOutOfRange(self):
    with self.test_session() as sess:
      cdim = tf.constant(4, tf.int32)
      s0 = tf.constant([2, 3, 5], tf.int32)
      s1 = tf.constant([2, 7, 5], tf.int32)
      off = gen_array_ops._concat_offset(cdim, [s0, s1])
      with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                   r"Concat dim is out of range: 4 vs. 3"):
        sess.run(off)

  def testDimMismatch(self):
    with self.test_session() as sess:
      cdim = tf.constant(1, tf.int32)
      s0 = tf.constant([2, 3, 5], tf.int32)
      s1 = tf.constant([2, 7, 5, 10], tf.int32)
      off = gen_array_ops._concat_offset(cdim, [s0, s1])
      with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                   r"should contain 3 elem"):
        sess.run(off)

  def testSizeMismatch(self):
    with self.test_session() as sess:
      cdim = tf.constant(1, tf.int32)
      s0 = tf.constant([2, 3, 5], tf.int32)
      s1 = tf.constant([2, 7, 10], tf.int32)
      off = gen_array_ops._concat_offset(cdim, [s0, s1])
      with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                   r"mismatch: 5 vs. 10"):
        sess.run(off)

if __name__ == "__main__":
  tf.test.main()
