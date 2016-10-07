# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import special
import tensorflow as tf

from tensorflow.contrib.distributions.python.ops import distribution_util


class DistributionUtilTest(tf.test.TestCase):

  def testAssertCloseIntegerDtype(self):
    x = [1, 5, 10, 15, 20]
    y = x
    z = [2, 5, 10, 15, 20]
    with self.test_session():
      with tf.control_dependencies([distribution_util.assert_close(x, y)]):
        tf.identity(x).eval()

      with tf.control_dependencies([distribution_util.assert_close(y, x)]):
        tf.identity(x).eval()

      with self.assertRaisesOpError("Condition x ~= y"):
        with tf.control_dependencies([distribution_util.assert_close(x, z)]):
          tf.identity(x).eval()

      with self.assertRaisesOpError("Condition x ~= y"):
        with tf.control_dependencies([distribution_util.assert_close(y, z)]):
          tf.identity(y).eval()

  def testAssertCloseNonIntegerDtype(self):
    x = np.array([1., 5, 10, 15, 20], dtype=np.float32)
    y = x + 1e-8
    z = [2., 5, 10, 15, 20]
    with self.test_session():
      with tf.control_dependencies([distribution_util.assert_close(x, y)]):
        tf.identity(x).eval()

      with tf.control_dependencies([distribution_util.assert_close(y, x)]):
        tf.identity(x).eval()

      with self.assertRaisesOpError("Condition x ~= y"):
        with tf.control_dependencies([distribution_util.assert_close(x, z)]):
          tf.identity(x).eval()

      with self.assertRaisesOpError("Condition x ~= y"):
        with tf.control_dependencies([distribution_util.assert_close(y, z)]):
          tf.identity(y).eval()

  def testAssertCloseEpsilon(self):
    x = [0., 5, 10, 15, 20]
    # x != y
    y = [0.1, 5, 10, 15, 20]
    # x = z
    z = [1e-8, 5, 10, 15, 20]
    with self.test_session():
      with tf.control_dependencies([distribution_util.assert_close(x, z)]):
        tf.identity(x).eval()

      with self.assertRaisesOpError("Condition x ~= y"):
        with tf.control_dependencies([distribution_util.assert_close(x, y)]):
          tf.identity(x).eval()

      with self.assertRaisesOpError("Condition x ~= y"):
        with tf.control_dependencies([distribution_util.assert_close(y, z)]):
          tf.identity(y).eval()

  def testAssertIntegerForm(self):
    # This should only be detected as an integer.
    x = [1., 5, 10, 15, 20]
    y = [1.1, 5, 10, 15, 20]
    # First component isn't less than float32.eps = 1e-7
    z = [1.0001, 5, 10, 15, 20]
    # This shouldn"t be detected as an integer.
    w = [1e-8, 5, 10, 15, 20]
    with self.test_session():
      with tf.control_dependencies([distribution_util.assert_integer_form(x)]):
        tf.identity(x).eval()

      with self.assertRaisesOpError("x has non-integer components"):
        with tf.control_dependencies([
            distribution_util.assert_integer_form(y)]):
          tf.identity(y).eval()

      with self.assertRaisesOpError("x has non-integer components"):
        with tf.control_dependencies([
            distribution_util.assert_integer_form(z)]):
          tf.identity(z).eval()

      with self.assertRaisesOpError("x has non-integer components"):
        with tf.control_dependencies([
            distribution_util.assert_integer_form(w)]):
          tf.identity(w).eval()

  def testGetLogitsAndProbImproperArguments(self):
    with self.test_session():
      with self.assertRaises(ValueError):
        distribution_util.get_logits_and_prob(logits=None, p=None)

      with self.assertRaises(ValueError):
        distribution_util.get_logits_and_prob(logits=[0.1], p=[0.1])

  def testGetLogitsAndProbLogits(self):
    p = np.array([0.01, 0.2, 0.5, 0.7, .99], dtype=np.float32)
    logits = special.logit(p)

    with self.test_session():
      new_logits, new_p = distribution_util.get_logits_and_prob(
          logits=logits, validate_args=True)

      self.assertAllClose(logits, new_logits.eval())
      self.assertAllClose(p, new_p.eval())

  def testGetLogitsAndProbProbability(self):
    p = np.array([0.01, 0.2, 0.5, 0.7, .99], dtype=np.float32)

    with self.test_session():
      new_logits, new_p = distribution_util.get_logits_and_prob(
          p=p, validate_args=True)

      self.assertAllClose(special.logit(p), new_logits.eval())
      self.assertAllClose(p, new_p.eval())

  def testGetLogitsAndProbProbabilityMultidimensional(self):
    p = np.array([[0.3, 0.4, 0.3], [0.1, 0.5, 0.4]], dtype=np.float32)

    with self.test_session():
      new_logits, new_p = distribution_util.get_logits_and_prob(
          p=p, multidimensional=True, validate_args=True)

      self.assertAllClose(special.logit(p), new_logits.eval())
      self.assertAllClose(p, new_p.eval())

  def testGetLogitsAndProbProbabilityValidateArgs(self):
    p = [0.01, 0.2, 0.5, 0.7, .99]
    # Component less than 0.
    p2 = [-1, 0.2, 0.5, 0.3, .2]
    # Component greater than 1.
    p3 = [2, 0.2, 0.5, 0.3, .2]

    with self.test_session():
      _, prob = distribution_util.get_logits_and_prob(p=p, validate_args=True)
      prob.eval()

      with self.assertRaisesOpError("Condition x >= 0"):
        _, prob = distribution_util.get_logits_and_prob(
            p=p2, validate_args=True)
        prob.eval()

      _, prob = distribution_util.get_logits_and_prob(p=p2, validate_args=False)
      prob.eval()

      with self.assertRaisesOpError("p has components greater than 1"):
        _, prob = distribution_util.get_logits_and_prob(
            p=p3, validate_args=True)
        prob.eval()

      _, prob = distribution_util.get_logits_and_prob(p=p3, validate_args=False)
      prob.eval()

  def testGetLogitsAndProbProbabilityValidateArgsMultidimensional(self):
    p = np.array([[0.3, 0.4, 0.3], [0.1, 0.5, 0.4]], dtype=np.float32)
    # Component less than 0. Still sums to 1.
    p2 = np.array([[-.3, 0.4, 0.9], [0.1, 0.5, 0.4]], dtype=np.float32)
    # Component greater than 1. Does not sum to 1.
    p3 = np.array([[1.3, 0.0, 0.0], [0.1, 0.5, 0.4]], dtype=np.float32)
    # Does not sum to 1.
    p4 = np.array([[1.1, 0.3, 0.4], [0.1, 0.5, 0.4]], dtype=np.float32)

    with self.test_session():
      _, prob = distribution_util.get_logits_and_prob(
          p=p, multidimensional=True)
      prob.eval()

      with self.assertRaisesOpError("Condition x >= 0"):
        _, prob = distribution_util.get_logits_and_prob(
            p=p2, multidimensional=True, validate_args=True)
        prob.eval()

      _, prob = distribution_util.get_logits_and_prob(
          p=p2, multidimensional=True, validate_args=False)
      prob.eval()

      with self.assertRaisesOpError(
          "(p has components greater than 1|p does not sum to 1)"):
        _, prob = distribution_util.get_logits_and_prob(
            p=p3, multidimensional=True, validate_args=True)
        prob.eval()

      _, prob = distribution_util.get_logits_and_prob(
          p=p3, multidimensional=True, validate_args=False)
      prob.eval()

      with self.assertRaisesOpError("p does not sum to 1"):
        _, prob = distribution_util.get_logits_and_prob(
            p=p4, multidimensional=True, validate_args=True)
        prob.eval()

      _, prob = distribution_util.get_logits_and_prob(
          p=p4, multidimensional=True, validate_args=False)
      prob.eval()

  def testLogCombinationsBinomial(self):
    n = [2, 5, 12, 15]
    k = [1, 2, 4, 11]
    log_combs = np.log(special.binom(n, k))

    with self.test_session():
      n = np.array(n, dtype=np.float32)
      counts = [[1., 1], [2., 3], [4., 8], [11, 4]]
      log_binom = distribution_util.log_combinations(n, counts)
      self.assertEqual([4], log_binom.get_shape())
      self.assertAllClose(log_combs, log_binom.eval())

  def testLogCombinationsShape(self):
    # Shape [2, 2]
    n = [[2, 5], [12, 15]]

    with self.test_session():
      n = np.array(n, dtype=np.float32)
      # Shape [2, 2, 4]
      counts = [[[1., 1, 0, 0], [2., 2, 1, 0]], [[4., 4, 1, 3], [10, 1, 1, 4]]]
      log_binom = distribution_util.log_combinations(n, counts)
      self.assertEqual([2, 2], log_binom.get_shape())

  def _np_rotate_transpose(self, x, shift):
    if not isinstance(x, np.ndarray):
      x = np.array(x)
    return np.transpose(x, np.roll(np.arange(len(x.shape)), shift))

  def testRollStatic(self):
    with self.test_session():
      with self.assertRaisesRegexp(
          ValueError, "None values not supported."):
        distribution_util.rotate_transpose(None, 1)
      for x in (np.ones(1), np.ones((2, 1)), np.ones((3, 2, 1))):
        for shift in np.arange(-5, 5):
          y = distribution_util.rotate_transpose(x, shift)
          self.assertAllEqual(self._np_rotate_transpose(x, shift),
                              y.eval())
          self.assertAllEqual(np.roll(x.shape, shift),
                              y.get_shape().as_list())

  def testRollDynamic(self):
    with self.test_session() as sess:
      x = tf.placeholder(tf.float32)
      shift = tf.placeholder(tf.int32)
      for x_value in (np.ones(1, dtype=x.dtype.as_numpy_dtype()),
                      np.ones((2, 1), dtype=x.dtype.as_numpy_dtype()),
                      np.ones((3, 2, 1), dtype=x.dtype.as_numpy_dtype())):
        for shift_value in np.arange(-5, 5):
          self.assertAllEqual(
              self._np_rotate_transpose(x_value, shift_value),
              sess.run(distribution_util.rotate_transpose(x, shift),
                       feed_dict={x: x_value, shift: shift_value}))

  def testChooseVector(self):
    with self.test_session():
      x = np.arange(10, 12)
      y = np.arange(15, 18)
      self.assertAllEqual(
          x, distribution_util.pick_vector(
              tf.less(0, 5), x, y).eval())
      self.assertAllEqual(
          y, distribution_util.pick_vector(
              tf.less(5, 0), x, y).eval())
      self.assertAllEqual(
          x, distribution_util.pick_vector(
              tf.constant(True), x, y))  # No eval.
      self.assertAllEqual(
          y, distribution_util.pick_vector(
              tf.constant(False), x, y))  # No eval.


if __name__ == "__main__":
  tf.test.main()
