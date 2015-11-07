"""Tests for tensorflow.ops.tf.matmul."""

import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from tensorflow.python.kernel_tests import gradient_checker as gc


def RandMatrix(rows, cols, tr):
  if tr:
    rows, cols = cols, rows
  return (np.clip(np.random.uniform(low=-100.0, high=100.0, size=rows * cols),
                  0, 100) / 100).reshape([rows, cols]).astype(np.float32)


class SparseMatMulTest(tf.test.TestCase):

  def _testCpuMatmul(self, x, y, tr_a=False, tr_b=False,
                     sp_a=True, sp_b=False):
    x_mat = np.matrix(x)
    if tr_a:
      x_mat = np.transpose(x_mat)
    y_mat = np.matrix(y)
    if tr_b:
      y_mat = np.transpose(y_mat)
    np_ans = x_mat * y_mat
    with self.test_session(use_gpu=False):
      tf_ans = tf.matmul(x, y,
                                transpose_a=tr_a, transpose_b=tr_b,
                                a_is_sparse=sp_a,
                                b_is_sparse=sp_b)
      out = tf_ans.eval()
    self.assertAllClose(np_ans, out)
    self.assertShapeEqual(np_ans, tf_ans)

  def testFloatBasic(self):
    x = np.arange(0., 4.).reshape([4, 1]).astype(np.float32)
    y = np.arange(-1., 1.).reshape([1, 2]).astype(np.float32)
    self._testCpuMatmul(x, y)

  # Tests testing random sized matrices.
  def testFloatRandom(self):
    for _ in range(10):
      for tr_a in [True, False]:
        for tr_b in [True, False]:
          for sp_a in [True, False]:
            for sp_b in [True, False]:
              n, k, m = np.random.randint(1, 100, size=3)
              x = RandMatrix(n, k, tr_a)
              y = RandMatrix(k, m, tr_b)
              self._testCpuMatmul(x, y, tr_a, tr_b, sp_a, sp_b)


class MatMulGradientTest(tf.test.TestCase):

  def _testGradients(self, tr_a, tr_b, sp_a, sp_b, name):
    with self.test_session():
      a = tf.constant(RandMatrix(3, 2, tr_a), dtype=tf.float32)
      b = tf.constant(RandMatrix(2, 4, tr_b), dtype=tf.float32)
      m = tf.matmul(a, b,
                           name=name,
                           transpose_a=tr_a,
                           transpose_b=tr_b,
                           a_is_sparse=sp_a,
                           b_is_sparse=sp_b)
      err = (gc.ComputeGradientError(a, [2, 3] if tr_a else [3, 2], m, [3, 4]) +
             gc.ComputeGradientError(b, [4, 2] if tr_b else [2, 4], m, [3, 4]))
    print "sparse_matmul gradient err = ", err
    self.assertLess(err, 1e-3)

  def testGradientInput(self):
    for tr_a in [True, False]:
      for tr_b in [True, False]:
        for sp_a in [True, False]:
          for sp_b in [True, False]:
            name = "sparse_matmul_%s_%s_%s_%s" % (tr_a, tr_b, sp_a, sp_b)
            self._testGradients(tr_a, tr_b, sp_a, sp_b, name)

if __name__ == "__main__":
  tf.test.main()
