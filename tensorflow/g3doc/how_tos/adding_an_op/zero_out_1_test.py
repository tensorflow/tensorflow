"""Test for version 1 of the zero_out op."""

import tensorflow.python.platform

import tensorflow as tf
from tensorflow.g3doc.how_tos.adding_an_op import gen_zero_out_op_1


class ZeroOut1Test(tf.test.TestCase):

  def test(self):
    with self.test_session():
      result = gen_zero_out_op_1.zero_out([5, 4, 3, 2, 1])
      self.assertAllEqual(result.eval(), [5, 0, 0, 0, 0])


if __name__ == '__main__':
  tf.test.main()
