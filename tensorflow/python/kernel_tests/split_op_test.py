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

"""Functional tests for Split Op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class SplitOpTest(tf.test.TestCase):

  def testExplicitNum(self):
    size_splits = tf.placeholder(dtype=tf.int32, shape=[None])

    value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    with self.test_session(use_gpu=False) as sess:
      with self.assertRaises(ValueError) as context:
        sess.run(tf.split(value, size_splits), {size_splits: [2, 2, 6]})

      self.assertTrue("Cannot infer num from shape" in str(context.exception))

      result = sess.run(tf.split(value, size_splits, num=3),
                        {size_splits: [2, 2, 6]})

    self.assertAllEqual(result[0], value[0:2])
    self.assertAllEqual(result[1], value[2:4])
    self.assertAllEqual(result[2], value[4:])

  def testListOfScalarTensors(self):
    a = tf.to_int32(5)
    b = tf.to_int32(6)

    value = np.random.rand(11, 11)

    with self.test_session(use_gpu=False) as sess:
      result = sess.run(tf.split(value, [a, b]))

    self.assertAllEqual(result[0], value[0:5, :])
    self.assertAllEqual(result[1], value[5:, :])

  def _RunAndVerifyVariable(self, use_gpu, large_num_splits=False):
    # Random dims of rank 5
    shape = np.random.randint(1, 5, size=5)
    split_dim = np.random.randint(0, 5)
    if large_num_splits:
      num_split = np.random.randint(16, 25)
    else:
      num_split = np.random.randint(2, 8)
    size_splits = np.random.randint(2, 8, num_split)
    shape[split_dim] = np.sum(size_splits)
    inp = np.random.rand(*shape).astype("f")
    with self.test_session(use_gpu=use_gpu) as sess:
      result = sess.run(tf.split(inp, size_splits, split_dim))
    slices = [slice(0, x) for x in shape]
    offset = 0
    for i in range(num_split):
      slices[split_dim] = slice(offset, offset + size_splits[i])
      offset += size_splits[i]
      self.assertAllEqual(result[i], inp[slices])

  def _testSpecialCasesVariable(self, use_gpu):
    inp = np.random.rand(4, 4).astype("f")

    with self.test_session(use_gpu=use_gpu) as sess:
      result = sess.run(tf.split(inp, [4], 0))
      self.assertAllEqual(result[0], inp)

      result = sess.run(tf.split(inp, [-1, 3], 0))
      self.assertAllEqual(result[0], inp[0:1, :])
      self.assertAllEqual(result[1], inp[1:4, :])

  def _testHugeNumberOfTensorsVariable(self, use_gpu):
    num_split = 10000
    size_splits = np.random.randint(1, 3, num_split)
    shape = [3, np.sum(size_splits)]
    split_dim = 1
    inp = np.random.rand(*shape).astype("f")
    with self.test_session(use_gpu=use_gpu) as sess:
      result = sess.run(tf.split(inp, size_splits, split_dim))
    slices = [slice(0, x) for x in shape]
    offset = 0
    for i in range(num_split):
      slices[split_dim] = slice(offset, offset + size_splits[i])
      offset += size_splits[i]
      self.assertAllEqual(result[i], inp[slices])

  def testSpecialCasesVariable(self):
    self._testSpecialCasesVariable(False)
    self._testSpecialCasesVariable(True)
    self._testHugeNumberOfTensorsVariable(False)
    self._testHugeNumberOfTensorsVariable(True)

  def _testGradientsSimpleVariable(self, use_gpu):
    inp = np.random.rand(4, 4).astype("f")
    with self.test_session(use_gpu=use_gpu):
      inp_tensor = tf.convert_to_tensor(inp)
      s = tf.split(inp_tensor, [1, 4], 1)
      inp_grads = [
          np.random.rand(4, 1).astype("f"), np.random.rand(4, 3).astype("f")
      ]
      grad_tensors = [tf.constant(x) for x in inp_grads]
      grad = tf.gradients(s, [inp_tensor], grad_tensors)[-1]
      result = grad.eval()

    self.assertAllEqual(result[:, 0:1], inp_grads[0])
    self.assertAllEqual(result[:, 1:4], inp_grads[1])

  def _compare(self, x, dim, num, use_gpu):
    np_ans = np.split(x, num, dim)
    with self.test_session(use_gpu=use_gpu) as sess:
      tf_ans = tf.split(value=x, num_or_size_splits=num, axis=dim)
      out = sess.run(tf_ans)
    self.assertEqual(num, len(np_ans))
    self.assertEqual(num, len(np_ans))
    self.assertEqual(num, len(out))
    for i in range(num):
      self.assertAllEqual(np_ans[i], out[i])
      self.assertShapeEqual(np_ans[i], tf_ans[i])

  def _testSplitRows(self, use_gpu):
    inp = np.random.rand(4, 4).astype("f")
    self._compare(inp, 0, 4, use_gpu)

  def testSplitRowsAll(self):
    self._testSplitRows(use_gpu=False)
    self._testSplitRows(use_gpu=True)

  def _testSplitCols(self, use_gpu):
    inp = np.random.rand(4, 4).astype("f")
    self._compare(inp, 1, 4, use_gpu)

  def testSplitColsAll(self):
    self._testSplitRows(use_gpu=False)
    self._testSplitCols(use_gpu=True)

  def _testEmpty(self, x, dim, num, expected_shape):
    with self.test_session() as sess:
      tf_ans = tf.split(value=x, num_or_size_splits=num, axis=dim)
      out = sess.run(tf_ans)
    self.assertEqual(x.size, 0)
    self.assertEqual(len(out), num)
    for i in range(num):
      self.assertEqual(out[i].shape, expected_shape)
      self.assertEqual(expected_shape, tf_ans[i].get_shape())

  def testEmpty(self):
    # Note: np.split returns a rank-0 empty ndarray
    # if the input ndarray is empty.
    inp = np.random.rand(8, 0, 21).astype("f")
    self._testEmpty(inp, 0, 2, (4, 0, 21))
    self._testEmpty(inp, 0, 4, (2, 0, 21))
    self._testEmpty(inp, 1, 4, (8, 0, 21))
    self._testEmpty(inp, 2, 3, (8, 0, 7))
    self._testEmpty(inp, 2, 7, (8, 0, 3))

  def testIdentity(self):
    inp = np.random.rand(2, 2, 2).astype("f")
    for use_gpu in [False, True]:
      self._compare(inp, 0, 1, use_gpu)
      self._compare(inp, 1, 1, use_gpu)
      self._compare(inp, 2, 1, use_gpu)

  def testSplitDim0(self):
    for use_gpu in [False, True]:
      self._compare(np.random.rand(6, 10, 18).astype("f"), 0, 3, use_gpu)
      self._compare(np.random.rand(6, 7, 18).astype("f"), 0, 3, use_gpu)
      self._compare(np.random.rand(6, 7, 9).astype("f"), 0, 3, use_gpu)

  def _RunAndVerify(self, use_gpu, large_num_splits=False):
    # Random dims of rank 5
    shape = np.random.randint(0, 5, size=5)
    split_dim = np.random.randint(0, 5)
    if large_num_splits:
      num_split = np.random.randint(9, 15)
    else:
      num_split = np.random.randint(2, 8)
    shape[split_dim] = np.random.randint(2, 5) * num_split
    inp = np.random.rand(*shape).astype("f")
    with self.test_session(use_gpu=use_gpu) as sess:
      result = sess.run(
          tf.split(
              value=inp, num_or_size_splits=num_split, axis=split_dim))
    slices = [slice(0, x) for x in shape]
    offset = 0
    length = shape[split_dim] // num_split
    for i in range(num_split):
      slices[split_dim] = slice(offset, offset + length)
      offset += length
      self.assertAllEqual(result[i], inp[slices])

  def testRandom(self):
    for _ in range(5):
      self._RunAndVerify(use_gpu=False)
      self._RunAndVerify(use_gpu=True)
      self._RunAndVerify(use_gpu=True, large_num_splits=True)
      self._RunAndVerifyVariable(use_gpu=False)
      self._RunAndVerifyVariable(use_gpu=True)
      self._RunAndVerifyVariable(use_gpu=True, large_num_splits=True)

  def _testGradientsSimple(self, use_gpu):
    inp = np.random.rand(4, 4).astype("f")
    with self.test_session(use_gpu=use_gpu):
      inp_tensor = tf.convert_to_tensor(inp)
      s = tf.split(value=inp_tensor, num_or_size_splits=4, axis=1)
      inp_grads = [np.random.rand(4, 1).astype("f") for _ in range(4)]
      grad_tensors = [tf.constant(x) for x in inp_grads]
      grad = tf.gradients(s, [inp_tensor], grad_tensors)[0]
      result = grad.eval()
    for i in range(4):
      self.assertAllEqual(result[:, i:i+1], inp_grads[i])

  def testGradientsAll(self):
    self._testGradientsSimple(use_gpu=False)
    self._testGradientsSimple(use_gpu=True)
    self._testGradientsSimpleVariable(use_gpu=False)
    self._testGradientsSimpleVariable(use_gpu=True)

  def testShapeFunctionEdgeCases(self):
    # split_dim greater than rank of input.
    with self.assertRaises(ValueError):
      tf.split(value=[[0, 1], [2, 3]], num_or_size_splits=4, axis=2)

    # num_split does not evenly divide the size in split_dim.
    with self.assertRaisesRegexp(ValueError, "should evenly divide"):
      tf.split(value=[0, 1, 2, 3], num_or_size_splits=3, axis=0)

    # Unknown split_dim.
    splits = tf.split(
        value=[[0, 1, 2, 3]],
        num_or_size_splits=4,
        axis=tf.placeholder(tf.int32))
    for s in splits:
      self.assertEqual([None, None], s.get_shape().as_list())

    # Unknown split_dim and input shape.
    splits = tf.split(
        value=tf.placeholder(tf.float32),
        num_or_size_splits=4,
        axis=tf.placeholder(tf.int32))
    for s in splits:
      self.assertEqual(None, s.get_shape().ndims)


if __name__ == "__main__":
  tf.test.main()
