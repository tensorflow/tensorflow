# Copyright 2016 Google Inc. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class DirichletMultinomialTest(tf.test.TestCase):

  def test_num_classes(self):
    with self.test_session():
      for num_classes in range(3):
        alpha = np.random.rand(3, num_classes)
        dist = tf.contrib.distributions.DirichletMultinomial(alpha)
        self.assertEqual([], dist.num_classes.get_shape())
        self.assertEqual(num_classes, dist.num_classes.eval())

  def test_alpha_property(self):
    alpha = np.array([[1., 2, 3]])
    with self.test_session():
      dist = tf.contrib.distributions.DirichletMultinomial(alpha)
      self.assertEqual([1, 3], dist.alpha.get_shape())
      self.assertAllClose(alpha, dist.alpha.eval())

  def test_empty_alpha_and_empty_counts_returns_empty(self):
    with self.test_session():
      alpha = [[]]
      counts = [[]]
      dist = tf.contrib.distributions.DirichletMultinomial(alpha)
      self.assertAllEqual([], dist.pmf(counts).eval())
      self.assertAllEqual([0], dist.pmf(counts).get_shape())
      self.assertAllEqual([], dist.log_pmf(counts).eval())
      self.assertAllEqual([0], dist.log_pmf(counts).get_shape())
      self.assertAllEqual([[]], dist.mean.eval())
      self.assertAllEqual([1, 0], dist.mean.get_shape())
      self.assertAllEqual(0, dist.num_classes.eval())
      self.assertAllEqual([], dist.num_classes.get_shape())

  def test_pmf_both_zero_batches(self):
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    with self.test_session():
      # Both zero-batches.  No broadcast
      alpha = [1., 2]
      counts = [1, 0.]
      dist = tf.contrib.distributions.DirichletMultinomial(alpha)
      pmf = dist.pmf(counts)
      self.assertAllClose(1 / 3., pmf.eval())
      self.assertEqual((), pmf.get_shape())

  def test_pmf_alpha_stretched_in_broadcast_when_same_rank(self):
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    with self.test_session():
      alpha = [[1., 2]]
      counts = [[1, 0.], [0, 1.]]
      dist = tf.contrib.distributions.DirichletMultinomial(alpha)
      pmf = dist.pmf(counts)
      self.assertAllClose([1 / 3., 2 / 3.], pmf.eval())
      self.assertEqual((2), pmf.get_shape())

  def test_pmf_alpha_stretched_in_broadcast_when_lower_rank(self):
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    with self.test_session():
      alpha = [1., 2]
      counts = [[1, 0.], [0, 1.]]
      pmf = tf.contrib.distributions.DirichletMultinomial(alpha).pmf(counts)
      self.assertAllClose([1 / 3., 2 / 3.], pmf.eval())
      self.assertEqual((2), pmf.get_shape())

  def test_pmf_counts_stretched_in_broadcast_when_same_rank(self):
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    with self.test_session():
      alpha = [[1., 2], [2., 3]]
      counts = [[1, 0.]]
      pmf = tf.contrib.distributions.DirichletMultinomial(alpha).pmf(counts)
      self.assertAllClose([1 / 3., 2 / 5.], pmf.eval())
      self.assertEqual((2), pmf.get_shape())

  def test_pmf_counts_stretched_in_broadcast_when_lower_rank(self):
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    with self.test_session():
      alpha = [[1., 2], [2., 3]]
      counts = [1, 0.]
      pmf = tf.contrib.distributions.DirichletMultinomial(alpha).pmf(counts)
      self.assertAllClose([1 / 3., 2 / 5.], pmf.eval())
      self.assertEqual((2), pmf.get_shape())

  def test_pmf_for_one_vote_is_the_mean_with_one_record_input(self):
    # The probabilities of one vote falling into class k is the mean for class
    # k.
    alpha = [1., 2, 3]
    with self.test_session():
      for class_num in range(3):
        counts = np.zeros((3), dtype=np.float32)
        counts[class_num] = 1.0
        dist = tf.contrib.distributions.DirichletMultinomial(alpha)
        mean = dist.mean.eval()
        pmf = dist.pmf(counts).eval()

        self.assertAllClose(mean[class_num], pmf)
        self.assertTupleEqual((3,), mean.shape)
        self.assertTupleEqual((), pmf.shape)

  def test_zero_counts_results_in_pmf_equal_to_one(self):
    # There is only one way for zero items to be selected, and this happens with
    # probability 1.
    alpha = [5, 0.5]
    counts = [0., 0.]
    with self.test_session():
      dist = tf.contrib.distributions.DirichletMultinomial(alpha)
      pmf = dist.pmf(counts)
      self.assertAllClose(1.0, pmf.eval())
      self.assertEqual((), pmf.get_shape())

  def test_large_tau_gives_precise_probabilities(self):
    # If tau is large, we are doing coin flips with probability mu.
    mu = np.array([0.1, 0.1, 0.8], dtype=np.float32)
    tau = np.array([100.], dtype=np.float32)
    alpha = tau * mu

    # One (three sided) coin flip.  Prob[coin 3] = 0.8.
    # Note that since it was one flip, value of tau didn't matter.
    counts = [0., 0, 1]
    with self.test_session():
      dist = tf.contrib.distributions.DirichletMultinomial(alpha)
      pmf = dist.pmf(counts)
      self.assertAllClose(0.8, pmf.eval(), atol=1e-4)
      self.assertEqual((), pmf.get_shape())

    # Two (three sided) coin flips.  Prob[coin 3] = 0.8.
    counts = [0., 0, 2]
    with self.test_session():
      dist = tf.contrib.distributions.DirichletMultinomial(alpha)
      pmf = dist.pmf(counts)
      self.assertAllClose(0.8**2, pmf.eval(), atol=1e-2)
      self.assertEqual((), pmf.get_shape())

    # Three (three sided) coin flips.
    counts = [1., 0, 2]
    with self.test_session():
      dist = tf.contrib.distributions.DirichletMultinomial(alpha)
      pmf = dist.pmf(counts)
      self.assertAllClose(3 * 0.1 * 0.8 * 0.8, pmf.eval(), atol=1e-2)
      self.assertEqual((), pmf.get_shape())

  def test_small_tau_prefers_correlated_results(self):
    # If tau is small, then correlation between draws is large, so draws that
    # are both of the same class are more likely.
    mu = np.array([0.5, 0.5], dtype=np.float32)
    tau = np.array([0.1], dtype=np.float32)
    alpha = tau * mu

    # If there is only one draw, it is still a coin flip, even with small tau.
    counts = [1, 0.]
    with self.test_session():
      dist = tf.contrib.distributions.DirichletMultinomial(alpha)
      pmf = dist.pmf(counts)
      self.assertAllClose(0.5, pmf.eval())
      self.assertEqual((), pmf.get_shape())

    # If there are two draws, it is much more likely that they are the same.
    counts_same = [2, 0.]
    counts_different = [1, 1.]
    with self.test_session():
      dist = tf.contrib.distributions.DirichletMultinomial(alpha)
      pmf_same = dist.pmf(counts_same)
      pmf_different = dist.pmf(counts_different)
      self.assertLess(5 * pmf_different.eval(), pmf_same.eval())
      self.assertEqual((), pmf_same.get_shape())


if __name__ == '__main__':
  tf.test.main()
