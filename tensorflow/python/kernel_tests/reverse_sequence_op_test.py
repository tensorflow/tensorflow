"""Tests for tensorflow.ops.reverse_sequence_op."""
import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from tensorflow.python.kernel_tests import gradient_checker as gc


class ReverseSequenceTest(tf.test.TestCase):

  def _testReverseSequence(self, x, seq_dim, seq_lengths,
                           truth, use_gpu=False, expected_err_re=None):
    with self.test_session(use_gpu=use_gpu):
      ans = tf.reverse_sequence(x,
                                seq_dim=seq_dim,
                                seq_lengths=seq_lengths)
      if expected_err_re is None:
        tf_ans = ans.eval()
        self.assertAllClose(tf_ans, truth, atol=1e-10)
        self.assertShapeEqual(truth, ans)
      else:
        with self.assertRaisesOpError(expected_err_re):
          ans.eval()

  def _testBothReverseSequence(self, x, seq_dim, seq_lengths,
                               truth, expected_err_re=None):
    self._testReverseSequence(x, seq_dim, seq_lengths,
                              truth, True, expected_err_re)
    self._testReverseSequence(x, seq_dim, seq_lengths,
                              truth, False, expected_err_re)

  def _testBasic(self, dtype):
    x = np.asarray([
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        [[9, 10, 11, 12], [13, 14, 15, 16]],
        [[17, 18, 19, 20], [21, 22, 23, 24]]], dtype=dtype)
    x = x.reshape(3, 2, 4, 1, 1)

    # reverse dim 2 up to (0:3, none, 0:4) along dim=0
    seq_dim = 2
    seq_lengths = np.asarray([3, 0, 4], dtype=np.int64)

    truth = np.asarray(
        [[[3, 2, 1, 4], [7, 6, 5, 8]],  # reverse 0:3
         [[9, 10, 11, 12], [13, 14, 15, 16]],  # reverse none
         [[20, 19, 18, 17], [24, 23, 22, 21]]],  # reverse 0:4 (all)
        dtype=dtype)
    truth = truth.reshape(3, 2, 4, 1, 1)
    self._testBothReverseSequence(x, seq_dim, seq_lengths, truth)

  def testFloatBasic(self):
    self._testBasic(np.float32)

  def testDoubleBasic(self):
    self._testBasic(np.float64)

  def testInt32Basic(self):
    self._testBasic(np.int32)

  def testInt64Basic(self):
    self._testBasic(np.int64)

  def testSComplexBasic(self):
    self._testBasic(np.complex64)

  def testFloatReverseSequenceGrad(self):
    x = np.asarray([
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        [[9, 10, 11, 12], [13, 14, 15, 16]],
        [[17, 18, 19, 20], [21, 22, 23, 24]]], dtype=np.float)
    x = x.reshape(3, 2, 4, 1, 1)

    # reverse dim 2 up to (0:3, none, 0:4) along dim=0
    seq_dim = 2
    seq_lengths = np.asarray([3, 0, 4], dtype=np.int64)

    with self.test_session():
      input_t = tf.constant(x, shape=x.shape)
      seq_lengths_t = tf.constant(seq_lengths, shape=seq_lengths.shape)
      reverse_sequence_out = tf.reverse_sequence(input_t,
                                                 seq_dim=seq_dim,
                                                 seq_lengths=seq_lengths_t)
      err = gc.ComputeGradientError(input_t,
                                    x.shape,
                                    reverse_sequence_out,
                                    x.shape,
                                    x_init_value=x)
    print "ReverseSequence gradient error = %g" % err
    self.assertLess(err, 1e-8)

  def testShapeFunctionEdgeCases(self):
    # Batch size mismatched between input and seq_lengths.
    with self.assertRaises(ValueError):
      tf.reverse_sequence(
          tf.placeholder(tf.float32, shape=(32, 2, 3)),
          seq_lengths=tf.placeholder(tf.int64, shape=(33,)),
          seq_dim=3)

    # seq_dim out of bounds.
    with self.assertRaisesRegexp(ValueError, "seq_dim must be < input.dims()"):
      tf.reverse_sequence(
          tf.placeholder(tf.float32, shape=(32, 2, 3)),
          seq_lengths=tf.placeholder(tf.int64, shape=(32,)),
          seq_dim=3)


if __name__ == "__main__":
  tf.test.main()
