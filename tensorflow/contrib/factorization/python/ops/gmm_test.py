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

"""Tests for ops.gmm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


class GMMTest(tf.test.TestCase):

  def setUp(self):
    np.random.seed(3)
    tf.set_random_seed(2)
    self.num_centers = 2
    self.num_dims = 2
    self.num_points = 4000
    self.batch_size = 100
    self.true_centers = self.make_random_centers(self.num_centers,
                                                 self.num_dims)
    self.points, self.assignments, self.scores = self.make_random_points(
        self.true_centers,
        self.num_points)
    self.true_score = np.add.reduce(self.scores)

    # Use initial means from kmeans (just like scikit-learn does).
    clusterer = tf.contrib.factorization.KMeansClustering(
        num_clusters=self.num_centers)
    clusterer.fit(self.points, steps=30)
    self.initial_means = clusterer.clusters()

  @staticmethod
  def make_random_centers(num_centers, num_dims):
    return np.round(np.random.rand(num_centers,
                                   num_dims).astype(np.float32) * 500)

  @staticmethod
  def make_random_points(centers, num_points):
    num_centers, num_dims = centers.shape
    assignments = np.random.choice(num_centers, num_points)
    offsets = np.round(np.random.randn(num_points,
                                       num_dims).astype(np.float32) * 20)
    points = centers[assignments] + offsets
    means = [np.mean(points[assignments == center], axis=0)
             for center in xrange(num_centers)]
    covs = [np.cov(points[assignments == center].T)
            for center in xrange(num_centers)]
    scores = []
    for r in xrange(num_points):
      scores.append(np.sqrt(np.dot(
          np.dot(points[r, :] - means[assignments[r]],
                 np.linalg.inv(covs[assignments[r]])),
          points[r, :] - means[assignments[r]])))
    return (points, assignments, scores)

  def test_clusters(self):
    """Tests the shape of the clusters."""
    gmm = tf.contrib.factorization.GMM(
        self.num_centers,
        initial_clusters=self.initial_means,
        batch_size=self.batch_size,
        steps=40,
        continue_training=True,
        random_seed=4,
        config=tf.contrib.learn.RunConfig(tf_random_seed=2))
    gmm.fit(x=self.points, steps=0)
    clusters = gmm.clusters()
    self.assertAllEqual(list(clusters.shape),
                        [self.num_centers, self.num_dims])

  def test_fit(self):
    gmm = tf.contrib.factorization.GMM(
        self.num_centers,
        initial_clusters='random',
        batch_size=self.batch_size,
        random_seed=4,
        config=tf.contrib.learn.RunConfig(tf_random_seed=2))
    gmm.fit(x=self.points, steps=1)
    score1 = gmm.score(x=self.points)
    gmm = tf.contrib.factorization.GMM(
        self.num_centers,
        initial_clusters='random',
        batch_size=self.batch_size,
        random_seed=4,
        config=tf.contrib.learn.RunConfig(tf_random_seed=2))
    gmm.fit(x=self.points, steps=10)
    score2 = gmm.score(x=self.points)
    self.assertGreater(score1, score2)
    self.assertNear(self.true_score, score2, self.true_score * 0.15)

  def test_infer(self):
    gmm = tf.contrib.factorization.GMM(
        self.num_centers,
        initial_clusters=self.initial_means,
        batch_size=self.batch_size,
        steps=40,
        continue_training=True,
        random_seed=4,
        config=tf.contrib.learn.RunConfig(tf_random_seed=2))
    gmm.fit(x=self.points, steps=60)
    clusters = gmm.clusters()

    # Make a small test set
    points, true_assignments, true_offsets = (
        self.make_random_points(clusters, 40))

    assignments = np.ravel(gmm.predict(points))
    self.assertAllEqual(true_assignments, assignments)

    # Test score
    score = gmm.score(points)
    self.assertNear(score, np.sum(true_offsets), 4.05)

  def _compare_with_sklearn(self, cov_type):
    # sklearn version.
    iterations = 40
    np.random.seed(5)
    sklearn_assignments = np.asarray([0, 0, 1, 0, 0, 0, 1, 0, 0, 1])
    sklearn_means = np.asarray([[144.83417719, 254.20130341],
                                [274.38754816, 353.16074346]])
    sklearn_covs = np.asarray([[[395.0081194, -4.50389512],
                                [-4.50389512, 408.27543989]],
                               [[385.17484203, -31.27834935],
                                [-31.27834935, 391.74249925]]])

    # skflow version.
    gmm = tf.contrib.factorization.GMM(
        self.num_centers,
        initial_clusters=self.initial_means,
        covariance_type=cov_type,
        batch_size=self.num_points,
        steps=iterations,
        continue_training=True,
        config=tf.contrib.learn.RunConfig(tf_random_seed=2))
    gmm.fit(self.points)
    skflow_assignments = gmm.predict(self.points[:10, :]).astype(int)
    self.assertAllClose(sklearn_assignments,
                        np.ravel(skflow_assignments))
    self.assertAllClose(sklearn_means, gmm.clusters())
    if cov_type == 'full':
      self.assertAllClose(sklearn_covs, gmm.covariances(), rtol=0.01)
    else:
      for d in [0, 1]:
        self.assertAllClose(np.diag(sklearn_covs[d]),
                            gmm.covariances()[d, :], rtol=0.01)

  def test_compare_full(self):
    self._compare_with_sklearn('full')

  def test_compare_diag(self):
    self._compare_with_sklearn('diag')


if __name__ == '__main__':
  tf.test.main()
