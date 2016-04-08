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
"""Tests for histogram_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib.metrics.python.ops import histogram_ops


class Strict1dCumsumTest(tf.test.TestCase):
  """Test this private function."""

  def test_empty_tensor_returns_empty(self):
    with self.test_session():
      tensor = tf.constant([])
      result = histogram_ops._strict_1d_cumsum(tensor, 0)
      expected = tf.constant([])
      np.testing.assert_array_equal(expected.eval(), result.eval())

  def test_length_1_tensor_works(self):
    with self.test_session():
      tensor = tf.constant([3], dtype=tf.float32)
      result = histogram_ops._strict_1d_cumsum(tensor, 1)
      expected = tf.constant([3], dtype=tf.float32)
      np.testing.assert_array_equal(expected.eval(), result.eval())

  def test_length_3_tensor_works(self):
    with self.test_session():
      tensor = tf.constant([1, 2, 3], dtype=tf.float32)
      result = histogram_ops._strict_1d_cumsum(tensor, 3)
      expected = tf.constant([1, 3, 6], dtype=tf.float32)
      np.testing.assert_array_equal(expected.eval(), result.eval())


class AUCUsingHistogramTest(tf.test.TestCase):

  def setUp(self):
    self.rng = np.random.RandomState(0)

  def test_empty_labels_and_scores_gives_nan_auc(self):
    with self.test_session():
      labels = tf.constant([], shape=[0], dtype=tf.bool)
      scores = tf.constant([], shape=[0], dtype=tf.float32)
      score_range = [0, 1.]
      auc, update_op = tf.contrib.metrics.auc_using_histogram(labels, scores,
                                                              score_range)
      tf.initialize_local_variables().run()
      update_op.run()
      self.assertTrue(np.isnan(auc.eval()))

  def test_perfect_scores_gives_auc_1(self):
    self._check_auc(nbins=100,
                    desired_auc=1.0,
                    score_range=[0, 1.],
                    num_records=50,
                    frac_true=0.5,
                    atol=0.05,
                    num_updates=1)

  def test_terrible_scores_gives_auc_0(self):
    self._check_auc(nbins=100,
                    desired_auc=0.0,
                    score_range=[0, 1.],
                    num_records=50,
                    frac_true=0.5,
                    atol=0.05,
                    num_updates=1)

  def test_many_common_conditions(self):
    for nbins in [50]:
      for desired_auc in [0.3, 0.5, 0.8]:
        for score_range in [[-1, 1], [-10, 0]]:
          for frac_true in [0.3, 0.8]:
            # Tests pass with atol = 0.03.  Moved up to 0.05 to avoid flakes.
            self._check_auc(nbins=nbins,
                            desired_auc=desired_auc,
                            score_range=score_range,
                            num_records=100,
                            frac_true=frac_true,
                            atol=0.05,
                            num_updates=50)

  def test_large_class_imbalance_still_ok(self):
    # With probability frac_true ** num_records, each batch contains only True
    # records.  In this case, ~ 95%.
    # Tests pass with atol = 0.02.  Increased to 0.05 to avoid flakes.
    self._check_auc(nbins=100,
                    desired_auc=0.8,
                    score_range=[-1, 1.],
                    num_records=10,
                    frac_true=0.995,
                    atol=0.05,
                    num_updates=1000)

  def test_super_accuracy_with_many_bins_and_records(self):
    # Test passes with atol = 0.0005.  Increased atol to avoid flakes.
    self._check_auc(nbins=1000,
                    desired_auc=0.75,
                    score_range=[0, 1.],
                    num_records=1000,
                    frac_true=0.5,
                    atol=0.005,
                    num_updates=100)

  def _check_auc(self,
                 nbins=100,
                 desired_auc=0.75,
                 score_range=None,
                 num_records=50,
                 frac_true=0.5,
                 atol=0.05,
                 num_updates=10):
    """Check auc accuracy against synthetic data.

    Args:
      nbins:  nbins arg from contrib.metrics.auc_using_histogram.
      desired_auc:  Number in [0, 1].  The desired auc for synthetic data.
      score_range:  2-tuple, (low, high), giving the range of the resultant
        scores.  Defaults to [0, 1.].
      num_records:  Positive integer.  The number of records to return.
      frac_true:  Number in (0, 1).  Expected fraction of resultant labels that
        will be True.  This is just in expectation...more or less may actually
        be True.
      atol:  Absolute tolerance for final AUC estimate.
      num_updates:  Update internal histograms this many times, each with a new
        batch of synthetic data, before computing final AUC.

    Raises:
      AssertionError: If resultant AUC is not within atol of theoretical AUC
        from synthetic data.
    """
    score_range = [0, 1.] or score_range
    with self.test_session():
      labels = tf.placeholder(tf.bool, shape=[num_records])
      scores = tf.placeholder(tf.float32, shape=[num_records])
      auc, update_op = tf.contrib.metrics.auc_using_histogram(labels,
                                                              scores,
                                                              score_range,
                                                              nbins=nbins)
      tf.initialize_local_variables().run()
      # Updates, then extract auc.
      for _ in range(num_updates):
        labels_a, scores_a = synthetic_data(desired_auc, score_range,
                                            num_records, self.rng, frac_true)
        update_op.run(feed_dict={labels: labels_a, scores: scores_a})
      labels_a, scores_a = synthetic_data(desired_auc, score_range, num_records,
                                          self.rng, frac_true)
      # Fetch current auc, and verify that fetching again doesn't change it.
      auc_eval = auc.eval()
      self.assertEqual(auc_eval, auc.eval())

    msg = ('nbins: %s, desired_auc: %s, score_range: %s, '
           'num_records: %s, frac_true: %s, num_updates: %s') % (nbins,
                                                                 desired_auc,
                                                                 score_range,
                                                                 num_records,
                                                                 frac_true,
                                                                 num_updates)
    np.testing.assert_allclose(desired_auc, auc_eval, atol=atol, err_msg=msg)


def synthetic_data(desired_auc, score_range, num_records, rng, frac_true):
  """Create synthetic boolean_labels and scores with adjustable auc.

  Args:
    desired_auc:  Number in [0, 1], the theoretical AUC of resultant data.
    score_range:  2-tuple, (low, high), giving the range of the resultant scores
    num_records:  Positive integer.  The number of records to return.
    rng:  Initialized np.random.RandomState random number generator
    frac_true:  Number in (0, 1).  Expected fraction of resultant labels that
      will be True.  This is just in expectation...more or less may actually be
      True.

  Returns:
    boolean_labels:  np.array, dtype=bool.
    scores:  np.array, dtype=np.float32
  """
  # We prove here why the method (below) for computing AUC works.  Of course we
  # also checked this against sklearn.metrics.roc_auc_curve.
  #
  # First do this for score_range = [0, 1], then rescale.
  # WLOG assume AUC >= 0.5, otherwise we will solve for AUC >= 0.5 then swap
  # the labels.
  # So for AUC in [0, 1] we create False and True labels
  # and corresponding scores drawn from:
  # F ~ U[0, 1],  T ~ U[x, 1]
  # We have,
  # AUC
  #  = P[T > F]
  #  = P[T > F | F < x] P[F < x] + P[T > F | F > x] P[F > x]
  #  = (1 * x) + (0.5 * (1 - x)).
  # Inverting, we have:
  # x = 2 * AUC - 1, when AUC >= 0.5.
  assert 0 <= desired_auc <= 1
  assert 0 < frac_true < 1

  if desired_auc < 0.5:
    flip_labels = True
    desired_auc = 1 - desired_auc
    frac_true = 1 - frac_true
  else:
    flip_labels = False
  x = 2 * desired_auc - 1

  labels = rng.binomial(1, frac_true, size=num_records).astype(bool)
  num_true = labels.sum()
  num_false = num_records - labels.sum()

  # Draw F ~ U[0, 1], and T ~ U[x, 1]
  false_scores = rng.rand(num_false)
  true_scores = x + rng.rand(num_true) * (1 - x)

  # Reshape [0, 1] to score_range.
  def reshape(scores):
    return score_range[0] + scores * (score_range[1] - score_range[0])

  false_scores = reshape(false_scores)
  true_scores = reshape(true_scores)

  # Place into one array corresponding with the labels.
  scores = np.nan * np.ones(num_records, dtype=np.float32)
  scores[labels] = true_scores
  scores[~labels] = false_scores

  if flip_labels:
    labels = ~labels

  return labels, scores


if __name__ == '__main__':
  tf.test.main()
