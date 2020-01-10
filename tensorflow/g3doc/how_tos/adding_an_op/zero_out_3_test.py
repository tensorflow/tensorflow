"""Test for version 3 of the zero_out op."""

import tensorflow.python.platform

import tensorflow as tf
from tensorflow.g3doc.how_tos.adding_an_op import gen_zero_out_op_3


class ZeroOut3Test(tf.test.TestCase):

  def test(self):
    with self.test_session():
      result = gen_zero_out_op_3.zero_out([5, 4, 3, 2, 1])
      self.assertAllEqual(result.eval(), [5, 0, 0, 0, 0])

  def testAttr(self):
    with self.test_session():
      result = gen_zero_out_op_3.zero_out([5, 4, 3, 2, 1], preserve_index=3)
      self.assertAllEqual(result.eval(), [0, 0, 0, 2, 0])

  def testNegative(self):
    with self.test_session():
      result = gen_zero_out_op_3.zero_out([5, 4, 3, 2, 1], preserve_index=-1)
      with self.assertRaisesOpError("Need preserve_index >= 0, got -1"):
        result.eval()

  def testLarge(self):
    with self.test_session():
      result = gen_zero_out_op_3.zero_out([5, 4, 3, 2, 1], preserve_index=17)
      with self.assertRaisesOpError("preserve_index out of range"):
        result.eval()


if __name__ == '__main__':
  tf.test.main()
