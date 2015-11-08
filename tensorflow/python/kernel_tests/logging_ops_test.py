"""Tests for tensorflow.kernels.logging_ops."""

import tensorflow.python.platform

import tensorflow as tf


class LoggingOpsTest(tf.test.TestCase):

  def testAssertDivideByZero(self):
    with self.test_session() as sess:
      epsilon = tf.convert_to_tensor(1e-20)
      x = tf.convert_to_tensor(0.0)
      y = tf.convert_to_tensor(1.0)
      z = tf.convert_to_tensor(2.0)
      # assert(epsilon < y)
      # z / y
      with sess.graph.control_dependencies(
          [tf.Assert(tf.less(epsilon, y), ["Divide-by-zero"])]):
        out = tf.div(z, y)
      self.assertAllEqual(2.0, out.eval())
      # assert(epsilon < x)
      # z / x
      #
      # This tests printing out multiple tensors
      with sess.graph.control_dependencies(
          [tf.Assert(tf.less(epsilon, x),
                     ["Divide-by-zero", "less than x"])]):
        out = tf.div(z, x)
      with self.assertRaisesOpError("less than x"):
        out.eval()


class PrintGradientTest(tf.test.TestCase):

  def testPrintShape(self):
    inp = tf.constant(2.0, shape=[100, 32])
    inp_printed = tf.Print(inp, [inp])
    self.assertEqual(inp.get_shape(), inp_printed.get_shape())

  def testPrintGradient(self):
    with self.test_session():
      inp = tf.constant(2.0, shape=[100, 32], name="in")
      w = tf.constant(4.0, shape=[10, 100], name="w")
      wx = tf.matmul(w, inp, name="wx")
      wx_print = tf.Print(wx, [w, w, w])
      wx_grad = tf.gradients(wx, w)[0]
      wx_print_grad = tf.gradients(wx_print, w)[0]
      wxg = wx_grad.eval()
      wxpg = wx_print_grad.eval()
    self.assertAllEqual(wxg, wxpg)


if __name__ == "__main__":
  tf.test.main()
