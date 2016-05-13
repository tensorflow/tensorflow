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

"""Functional tests for Split Op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class SplitOpTest(tf.test.TestCase):

  def _compare(self, x, dim, num, use_gpu):
    np_ans = np.split(x, num, dim)
    with self.test_session(use_gpu=use_gpu) as sess:
      tf_ans = tf.split(dim, num, x)
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
      tf_ans = tf.split(dim, num, x)
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
      result = sess.run(tf.split(split_dim, num_split, inp))
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

  def _testGradientsSimple(self, use_gpu):
    inp = np.random.rand(4, 4).astype("f")
    with self.test_session(use_gpu=use_gpu):
      inp_tensor = tf.convert_to_tensor(inp)
      s = tf.split(1, 4, inp_tensor)
      inp_grads = [np.random.rand(4, 1).astype("f") for _ in range(4)]
      grad_tensors = [tf.constant(x) for x in inp_grads]
      grad = tf.gradients(s, [inp_tensor], grad_tensors)[0]
      result = grad.eval()
    for i in range(4):
      self.assertAllEqual(result[:, i:i+1], inp_grads[i])

  def testGradientsAll(self):
    self._testGradientsSimple(use_gpu=False)
    self._testGradientsSimple(use_gpu=True)

  def testShapeFunctionEdgeCases(self):
    # split_dim greater than rank of input.
    with self.assertRaises(ValueError):
      tf.split(2, 4, [[0, 1], [2, 3]])

    # num_split does not evenly divide the size in split_dim.
    with self.assertRaisesRegexp(ValueError, "should evenly divide"):
      tf.split(0, 3, [0, 1, 2, 3])

    # Unknown split_dim.
    splits = tf.split(tf.placeholder(tf.int32),
                             4, [[0, 1, 2, 3]])
    for s in splits:
      self.assertEqual([None, None], s.get_shape().as_list())

    # Unknown split_dim and input shape.
    splits = tf.split(tf.placeholder(tf.int32),
                             4, tf.placeholder(tf.float32))
    for s in splits:
      self.assertEqual(None, s.get_shape().ndims)


if __name__ == "__main__":
  tf.test.main()
