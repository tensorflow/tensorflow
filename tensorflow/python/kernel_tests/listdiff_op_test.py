"""Tests for tensorflow.kernels.listdiff_op."""

import tensorflow.python.platform

import numpy as np
import tensorflow as tf


class ListDiffTest(tf.test.TestCase):

  def _testListDiff(self, x, y, out, idx, dtype=np.int32):
    x = np.array(x, dtype=dtype)
    y = np.array(y, dtype=dtype)
    out = np.array(out, dtype=dtype)
    idx = np.array(idx, dtype=dtype)

    with self.test_session() as sess:
      x_tensor = tf.convert_to_tensor(x)
      y_tensor = tf.convert_to_tensor(y)
      out_tensor, idx_tensor = tf.listdiff(x_tensor, y_tensor)
      tf_out, tf_idx = sess.run([out_tensor, idx_tensor])

    self.assertAllEqual(tf_out, out)
    self.assertAllEqual(tf_idx, idx)
    self.assertEqual(1, out_tensor.get_shape().ndims)
    self.assertEqual(1, idx_tensor.get_shape().ndims)

  def testBasic1(self):
    x = [1, 2, 3, 4]
    y = [1, 2]
    out = [3, 4]
    idx = [2, 3]
    for t in [np.int32, np.int64, np.float, np.double]:
      self._testListDiff(x, y, out, idx, dtype=t)

  def testBasic2(self):
    x = [1, 2, 3, 4]
    y = [2]
    out = [1, 3, 4]
    idx = [0, 2, 3]
    for t in [np.int32, np.int64, np.float, np.double]:
      self._testListDiff(x, y, out, idx, dtype=t)

  def testBasic3(self):
    x = [1, 4, 3, 2]
    y = [4, 2]
    out = [1, 3]
    idx = [0, 2]
    for t in [np.int32, np.int64, np.float, np.double]:
      self._testListDiff(x, y, out, idx, dtype=t)

  def testDuplicates(self):
    x = [1, 2, 4, 3, 2, 3, 3, 1]
    y = [4, 2]
    out = [1, 3, 3, 3, 1]
    idx = [0, 3, 5, 6, 7]
    for t in [np.int32, np.int64, np.float, np.double]:
      self._testListDiff(x, y, out, idx, dtype=t)

  def testRandom(self):
    num_random_tests = 10
    int_low = -7
    int_high = 8
    max_size = 50
    for _ in xrange(num_random_tests):
      x_size = np.random.randint(max_size + 1)
      x = np.random.randint(int_low, int_high, size=x_size)
      y_size = np.random.randint(max_size + 1)
      y = np.random.randint(int_low, int_high, size=y_size)
      out_idx = [(entry, pos) for pos, entry in enumerate(x) if entry not in y]
      if out_idx:
        out_idx = map(list, zip(*out_idx))
        out = out_idx[0]
        idx = out_idx[1]
      else:
        out = []
        idx = []
      for t in [np.int32, np.int64, np.float, np.double]:
        self._testListDiff(x, y, out, idx, dtype=t)

  def testInt32FullyOverlapping(self):
    x = [1, 2, 3, 4]
    y = [1, 2, 3, 4]
    out = []
    idx = []
    self._testListDiff(x, y, out, idx)

  def testInt32NonOverlapping(self):
    x = [1, 2, 3, 4]
    y = [5, 6]
    out = x
    idx = range(len(x))
    self._testListDiff(x, y, out, idx)

  def testInt32EmptyX(self):
    x = []
    y = [1, 2]
    out = []
    idx = []
    self._testListDiff(x, y, out, idx)

  def testInt32EmptyY(self):
    x = [1, 2, 3, 4]
    y = []
    out = x
    idx = range(len(x))
    self._testListDiff(x, y, out, idx)

  def testInt32EmptyXY(self):
    x = []
    y = []
    out = []
    idx = []
    self._testListDiff(x, y, out, idx)

if __name__ == "__main__":
  tf.test.main()
