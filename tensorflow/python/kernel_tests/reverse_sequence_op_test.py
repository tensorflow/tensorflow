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

"""Tests for tensorflow.ops.reverse_sequence_op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


class ReverseSequenceTest(tf.test.TestCase):

  def _testReverseSequence(self, x, batch_dim, seq_dim, seq_lengths,
                           truth, use_gpu=False, expected_err_re=None):
    with self.test_session(use_gpu=use_gpu):
      ans = tf.reverse_sequence(x,
                                batch_dim=batch_dim,
                                seq_dim=seq_dim,
                                seq_lengths=seq_lengths)
      if expected_err_re is None:
        tf_ans = ans.eval()
        self.assertAllClose(tf_ans, truth, atol=1e-10)
        self.assertShapeEqual(truth, ans)
      else:
        with self.assertRaisesOpError(expected_err_re):
          ans.eval()

  def _testBothReverseSequence(self, x, batch_dim, seq_dim, seq_lengths,
                               truth, expected_err_re=None):
    self._testReverseSequence(x, batch_dim, seq_dim, seq_lengths,
                              truth, True, expected_err_re)
    self._testReverseSequence(x, batch_dim, seq_dim, seq_lengths,
                              truth, False, expected_err_re)

  def _testBasic(self, dtype):
    x = np.asarray([
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        [[9, 10, 11, 12], [13, 14, 15, 16]],
        [[17, 18, 19, 20], [21, 22, 23, 24]]], dtype=dtype)
    x = x.reshape(3, 2, 4, 1, 1)
    x = x.transpose([2, 1, 0, 3, 4])  # permute axes 0 <=> 2

    # reverse dim 2 up to (0:3, none, 0:4) along dim=0
    seq_lengths = np.asarray([3, 0, 4], dtype=np.int64)

    truth_orig = np.asarray(
        [[[3, 2, 1, 4], [7, 6, 5, 8]],  # reverse 0:3
         [[9, 10, 11, 12], [13, 14, 15, 16]],  # reverse none
         [[20, 19, 18, 17], [24, 23, 22, 21]]],  # reverse 0:4 (all)
        dtype=dtype)
    truth_orig = truth_orig.reshape(3, 2, 4, 1, 1)
    truth = truth_orig.transpose([2, 1, 0, 3, 4])  # permute axes 0 <=> 2

    seq_dim = 0    # permute seq_dim and batch_dim (originally 2 and 0, resp.)
    batch_dim = 2
    self._testBothReverseSequence(x, batch_dim, seq_dim, seq_lengths, truth)

  def testFloatBasic(self):
    self._testBasic(np.float32)

  def testDoubleBasic(self):
    self._testBasic(np.float64)

  def testInt32Basic(self):
    self._testBasic(np.int32)

  def testInt64Basic(self):
    self._testBasic(np.int64)

  def testComplex64Basic(self):
    self._testBasic(np.complex64)

  def testComplex128Basic(self):
    self._testBasic(np.complex128)

  def testFloatReverseSequenceGrad(self):
    x = np.asarray([
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        [[9, 10, 11, 12], [13, 14, 15, 16]],
        [[17, 18, 19, 20], [21, 22, 23, 24]]], dtype=np.float)
    x = x.reshape(3, 2, 4, 1, 1)
    x = x.transpose([2, 1, 0, 3, 4])  # transpose axes 0 <=> 2

    # reverse dim 0 up to (0:3, none, 0:4) along dim=2
    seq_dim = 0
    batch_dim = 2
    seq_lengths = np.asarray([3, 0, 4], dtype=np.int64)

    with self.test_session():
      input_t = tf.constant(x, shape=x.shape)
      seq_lengths_t = tf.constant(seq_lengths, shape=seq_lengths.shape)
      reverse_sequence_out = tf.reverse_sequence(input_t,
                                                 batch_dim=batch_dim,
                                                 seq_dim=seq_dim,
                                                 seq_lengths=seq_lengths_t)
      err = tf.test.compute_gradient_error(input_t,
                                           x.shape,
                                           reverse_sequence_out,
                                           x.shape,
                                           x_init_value=x)
    print("ReverseSequence gradient error = %g" % err)
    self.assertLess(err, 1e-8)

  def testShapeFunctionEdgeCases(self):
    t = tf.reverse_sequence(
        tf.placeholder(tf.float32, shape=None),
        seq_lengths=tf.placeholder(tf.int64, shape=(32,)),
        batch_dim=0, seq_dim=1)
    self.assertIs(t.get_shape().ndims, None)

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

    # batch_dim out of bounds.
    with self.assertRaisesRegexp(
        ValueError, "batch_dim must be < input.dims()"):
      tf.reverse_sequence(
          tf.placeholder(tf.float32, shape=(32, 2, 3)),
          seq_lengths=tf.placeholder(tf.int64, shape=(32,)),
          seq_dim=0,
          batch_dim=3)

    with self.test_session():
      inputs = tf.placeholder(tf.float32, shape=(32, 2, 3))
      seq_lengths = tf.placeholder(tf.int64, shape=(32,))
      output = tf.reverse_sequence(
          inputs,
          seq_lengths=seq_lengths,
          seq_dim=0)  # batch_dim default is 0
      with self.assertRaisesOpError("batch_dim == seq_dim"):
        output.eval(feed_dict={inputs: np.random.rand(32, 2, 3),
                               seq_lengths: xrange(32)})


if __name__ == "__main__":
  tf.test.main()
