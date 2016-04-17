# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Functional tests for slice op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


class SliceTest(tf.test.TestCase):

  def _testEmpty(self, use_gpu):
    inp = np.random.rand(4, 4).astype("f")
    for k in xrange(4):
      with self.test_session(use_gpu=use_gpu):
        a = tf.constant(inp, shape=[4, 4], dtype=tf.float32)
        slice_t = a[2, k:k]
        slice_val = slice_t.eval()
      self.assertAllEqual(slice_val, inp[2, k:k])

  def testEmptyAll(self):
    self._testEmpty(use_gpu=False)
    self._testEmpty(use_gpu=True)

  def _testInt32(self, use_gpu):
    inp = np.random.rand(4, 4).astype("i")
    for k in xrange(4):
      with self.test_session(use_gpu=use_gpu):
        a = tf.constant(inp, shape=[4, 4], dtype=tf.int32)
        slice_t = a[2, k:k]
        slice_val = slice_t.eval()
      self.assertAllEqual(slice_val, inp[2, k:k])

  def testInt32(self):
    self._testEmpty(use_gpu=False)
    self._testEmpty(use_gpu=True)

  def _testSelectAll(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      inp = np.random.rand(4, 4, 4, 4).astype("f")
      a = tf.constant(inp, shape=[4, 4, 4, 4],
                               dtype=tf.float32)

      slice_explicit_t = tf.slice(a, [0, 0, 0, 0], [-1, -1, -1, -1])
      slice_implicit_t = a[:, :, :, :]

      self.assertAllEqual(inp, slice_explicit_t.eval())
      self.assertAllEqual(inp, slice_implicit_t.eval())
      self.assertEqual(inp.shape, slice_explicit_t.get_shape())
      self.assertEqual(inp.shape, slice_implicit_t.get_shape())

  def testSelectAll(self):
    for _ in range(10):
      self._testSelectAll(use_gpu=False)
      self._testSelectAll(use_gpu=True)

  def _testSingleDimension(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      inp = np.random.rand(10).astype("f")
      a = tf.constant(inp, shape=[10], dtype=tf.float32)

      hi = np.random.randint(0, 9)
      scalar_t = a[hi]
      scalar_val = scalar_t.eval()
      self.assertAllEqual(scalar_val, inp[hi])

      if hi > 0:
        lo = np.random.randint(0, hi)
      else:
        lo = 0
      slice_t = a[lo:hi]
      slice_val = slice_t.eval()
      self.assertAllEqual(slice_val, inp[lo:hi])

  def testSingleDimension(self):
    for _ in range(10):
      self._testSingleDimension(use_gpu=False)
      self._testSingleDimension(use_gpu=True)

  def _testSliceMatrixDim0(self, x, begin, size, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      tf_ans = tf.slice(x, [begin, 0], [size, x.shape[1]]).eval()
    np_ans = x[begin:begin+size, :]
    self.assertAllEqual(tf_ans, np_ans)

  def testSliceMatrixDim0(self):
    for use_gpu in [False, True]:
      x = np.random.rand(8, 4).astype("f")
      self._testSliceMatrixDim0(x, 1, 2, use_gpu)
      self._testSliceMatrixDim0(x, 3, 3, use_gpu)
      y = np.random.rand(8, 7).astype("f")    # 7 * sizeof(float) is not aligned
      self._testSliceMatrixDim0(y, 1, 2, use_gpu)
      self._testSliceMatrixDim0(y, 3, 3, use_gpu)

  def _testIndexAndSlice(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      inp = np.random.rand(4, 4).astype("f")
      a = tf.constant(inp, shape=[4, 4], dtype=tf.float32)

      x, y = np.random.randint(0, 3, size=2).tolist()
      slice_t = a[x, 0:y]
      slice_val = slice_t.eval()
    self.assertAllEqual(slice_val, inp[x, 0:y])

  def testSingleElementAll(self):
    for _ in range(10):
      self._testIndexAndSlice(use_gpu=False)
      self._testIndexAndSlice(use_gpu=True)

  def _testSimple(self, use_gpu):
    with self.test_session(use_gpu=use_gpu) as sess:
      inp = np.random.rand(4, 4).astype("f")
      a = tf.constant([float(x) for x in inp.ravel(order="C")],
                               shape=[4, 4], dtype=tf.float32)
      slice_t = tf.slice(a, [0, 0], [2, 2])
      slice2_t = a[:2, :2]
      slice_val, slice2_val = sess.run([slice_t, slice2_t])
    self.assertAllEqual(slice_val, inp[:2, :2])
    self.assertAllEqual(slice2_val, inp[:2, :2])
    self.assertEqual(slice_val.shape, slice_t.get_shape())
    self.assertEqual(slice2_val.shape, slice2_t.get_shape())

  def testSimpleAll(self):
    self._testSimple(use_gpu=False)
    self._testSimple(use_gpu=True)

  def _testComplex(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      inp = np.random.rand(4, 10, 10, 4).astype("f")
      a = tf.constant(inp, dtype=tf.float32)

      x = np.random.randint(0, 9)
      z = np.random.randint(0, 9)
      if z > 0:
        y = np.random.randint(0, z)
      else:
        y = 0
      slice_t = a[:, x, y:z, :]
      self.assertAllEqual(slice_t.eval(), inp[:, x, y:z, :])

  def testComplex(self):
    for _ in range(10):
      self._testComplex(use_gpu=False)
      self._testComplex(use_gpu=True)

  def _RunAndVerifyResult(self, use_gpu):
    # Random dims of rank 6
    input_shape = np.random.randint(0, 20, size=6)
    inp = np.random.rand(*input_shape).astype("f")
    with self.test_session(use_gpu=use_gpu) as sess:
      a = tf.constant([float(x) for x in inp.ravel(order="C")],
                               shape=input_shape, dtype=tf.float32)
      indices = [0 if x == 0 else np.random.randint(x) for x in input_shape]
      sizes = [np.random.randint(0, input_shape[i] - indices[i] + 1)
               for i in range(6)]
      slice_t = tf.slice(a, indices, sizes)
      slice2_t = a[indices[0]:indices[0]+sizes[0],
                   indices[1]:indices[1]+sizes[1],
                   indices[2]:indices[2]+sizes[2],
                   indices[3]:indices[3]+sizes[3],
                   indices[4]:indices[4]+sizes[4],
                   indices[5]:indices[5]+sizes[5]]

      slice_val, slice2_val = sess.run([slice_t, slice2_t])

    expected_val = inp[indices[0]:indices[0]+sizes[0],
                       indices[1]:indices[1]+sizes[1],
                       indices[2]:indices[2]+sizes[2],
                       indices[3]:indices[3]+sizes[3],
                       indices[4]:indices[4]+sizes[4],
                       indices[5]:indices[5]+sizes[5]]
    self.assertAllEqual(slice_val, expected_val)
    self.assertAllEqual(slice2_val, expected_val)
    self.assertEqual(expected_val.shape, slice_t.get_shape())
    self.assertEqual(expected_val.shape, slice2_t.get_shape())

  def testRandom(self):
    for _ in range(10):
      self._RunAndVerifyResult(use_gpu=False)
      self._RunAndVerifyResult(use_gpu=True)

  def _testGradientSlice(self, input_shape, slice_begin, slice_size, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      num_inputs = np.prod(input_shape)
      num_grads = np.prod(slice_size)
      inp = np.random.rand(num_inputs).astype("f").reshape(input_shape)
      a = tf.constant([float(x) for x in inp.ravel(order="C")],
                               shape=input_shape, dtype=tf.float32)
      slice_t = tf.slice(a, slice_begin, slice_size)
      grads = np.random.rand(num_grads).astype("f").reshape(slice_size)
      grad_tensor = tf.constant(grads)
      grad = tf.gradients(slice_t, [a], grad_tensor)[0]
      result = grad.eval()

    # Create a zero tensor of the input shape ane place
    # the grads into the right location to compare against TensorFlow.
    np_ans = np.zeros(input_shape)
    slices = []
    for i in xrange(len(input_shape)):
      slices.append(slice(slice_begin[i], slice_begin[i] + slice_size[i]))
    np_ans[slices] = grads

    self.assertAllClose(np_ans, result)

  def _testGradientVariableSize(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      inp = tf.constant([1.0, 2.0, 3.0], name="in")
      out = tf.slice(inp, [1], [-1])
      grad_actual = tf.gradients(out, inp)[0].eval()
    self.assertAllClose([0., 1., 1.], grad_actual)

  def _testGradientsSimple(self, use_gpu):
    # Slice the middle square out of a 4x4 input
    self._testGradientSlice([4, 4], [1, 1], [2, 2], use_gpu)

    # Slice the upper left square out of a 4x4 input
    self._testGradientSlice([4, 4], [0, 0], [2, 2], use_gpu)

    # Slice a non-square input starting from (2,1)
    self._testGradientSlice([4, 4], [2, 1], [1, 2], use_gpu)

    # Slice a 3D tensor
    self._testGradientSlice([3, 3, 3], [0, 1, 0], [2, 1, 1], use_gpu)

    # Use -1 as a slice dimension.
    self._testGradientVariableSize(use_gpu)

  def testGradientsAll(self):
    self._testGradientsSimple(use_gpu=False)
    self._testGradientsSimple(use_gpu=True)

  def testNotIterable(self):
    # NOTE(mrry): If we register __getitem__ as an overloaded
    # operator, Python will valiantly attempt to iterate over the
    # Tensor from 0 to infinity.  This test ensures that this
    # unintended behavior is prevented.
    c = tf.constant(5.0)
    with self.assertRaisesWithPredicateMatch(
        TypeError,
        lambda e: "'Tensor' object is not iterable" in str(e)):
      for _ in c:
        pass

if __name__ == "__main__":
  tf.test.main()
