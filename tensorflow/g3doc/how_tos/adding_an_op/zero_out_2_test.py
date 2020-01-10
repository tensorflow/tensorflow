"""Test for version 2 of the zero_out op."""

import tensorflow.python.platform

import tensorflow as tf
from tensorflow.g3doc.how_tos.adding_an_op import gen_zero_out_op_2
from tensorflow.g3doc.how_tos.adding_an_op import zero_out_grad_2
from tensorflow.python.kernel_tests import gradient_checker


class ZeroOut2Test(tf.test.TestCase):

  def test(self):
    with self.test_session():
      result = gen_zero_out_op_2.zero_out([5, 4, 3, 2, 1])
      self.assertAllEqual(result.eval(), [5, 0, 0, 0, 0])

  def test_grad(self):
    with self.test_session():
      shape = (5,)
      x = tf.constant([5, 4, 3, 2, 1], dtype=tf.float32)
      y = gen_zero_out_op_2.zero_out(x)
      err = gradient_checker.ComputeGradientError(x, shape, y, shape)
      self.assertLess(err, 1e-4)


if __name__ == '__main__':
  tf.test.main()
