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

from tensorflow.contrib.factorization.python.ops import gmm as gmm_lib
from tensorflow.contrib.learn.python.learn.estimators import kmeans
from tensorflow.contrib.learn.python.learn.estimators import run_config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed as random_seed_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import flags
from tensorflow.python.platform import test

FLAGS = flags.FLAGS


class GMMTest(test.TestCase):

  def input_fn(self, batch_size=None, points=None):
    batch_size = batch_size or self.batch_size
    points = points if points is not None else self.points
    num_points = points.shape[0]

    def _fn():
      x = constant_op.constant(points)
      if batch_size == num_points:
        return x, None
      indices = random_ops.random_uniform(constant_op.constant([batch_size]),
                                          minval=0, maxval=num_points-1,
                                          dtype=dtypes.int32,
                                          seed=10)
      return array_ops.gather(x, indices), None
    return _fn

  def setUp(self):
    np.random.seed(3)
    random_seed_lib.set_random_seed(2)
    self.num_centers = 2
    self.num_dims = 2
    self.num_points = 4000
    self.batch_size = self.num_points
    self.true_centers = self.make_random_centers(self.num_centers,
                                                 self.num_dims)
    self.points, self.assignments, self.scores = self.make_random_points(
        self.true_centers, self.num_points)
    self.true_score = np.add.reduce(self.scores)

    # Use initial means from kmeans (just like scikit-learn does).
    clusterer = kmeans.KMeansClustering(num_clusters=self.num_centers)
    clusterer.fit(input_fn=lambda: (constant_op.constant(self.points), None),
                  steps=30)
    self.initial_means = clusterer.clusters()

  @staticmethod
  def make_random_centers(num_centers, num_dims):
    return np.round(
        np.random.rand(num_centers, num_dims).astype(np.float32) * 500)

  @staticmethod
  def make_random_points(centers, num_points):
    num_centers, num_dims = centers.shape
    assignments = np.random.choice(num_centers, num_points)
    offsets = np.round(
        np.random.randn(num_points, num_dims).astype(np.float32) * 20)
    points = centers[assignments] + offsets
    means = [
        np.mean(
            points[assignments == center], axis=0)
        for center in xrange(num_centers)
    ]
    covs = [
        np.cov(points[assignments == center].T)
        for center in xrange(num_centers)
    ]
    scores = []
    for r in xrange(num_points):
      scores.append(
          np.sqrt(
              np.dot(
                  np.dot(points[r, :] - means[assignments[r]],
                         np.linalg.inv(covs[assignments[r]])), points[r, :] -
                  means[assignments[r]])))
    return (points, assignments, scores)
  
  def test_weights(self):
    """Tests the shape of the weights."""
    gmm = gmm_lib.GMM(self.num_centers,
                      initial_clusters=self.initial_means,
                      random_seed=4,
                      config=run_config.RunConfig(tf_random_seed=2))
    gmm.fit(input_fn=self.input_fn(), steps=0)
    weights = gmm.weights()
    self.assertAllEqual(list(weights.shape), [self.num_centers])

  def test_clusters(self):
    """Tests the shape of the clusters."""
    gmm = gmm_lib.GMM(self.num_centers,
                      initial_clusters=self.initial_means,
                      random_seed=4,
                      config=run_config.RunConfig(tf_random_seed=2))
    gmm.fit(input_fn=self.input_fn(), steps=0)
    clusters = gmm.clusters()
    self.assertAllEqual(list(clusters.shape), [self.num_centers, self.num_dims])

  def test_fit(self):
    gmm = gmm_lib.GMM(self.num_centers,
                      initial_clusters='random',
                      random_seed=4,
                      config=run_config.RunConfig(tf_random_seed=2))
    gmm.fit(input_fn=self.input_fn(), steps=1)
    score1 = gmm.score(input_fn=self.input_fn(batch_size=self.num_points),
                       steps=1)
    gmm.fit(input_fn=self.input_fn(), steps=10)
    score2 = gmm.score(input_fn=self.input_fn(batch_size=self.num_points),
                       steps=1)
    self.assertGreater(score1, score2)
    self.assertNear(self.true_score, score2, self.true_score * 0.15)

  def test_infer(self):
    gmm = gmm_lib.GMM(self.num_centers,
                      initial_clusters=self.initial_means,
                      random_seed=4,
                      config=run_config.RunConfig(tf_random_seed=2))
    gmm.fit(input_fn=self.input_fn(), steps=60)
    clusters = gmm.clusters()

    # Make a small test set
    num_points = 40
    points, true_assignments, true_offsets = (
        self.make_random_points(clusters, num_points))

    assignments = []
    for item in gmm.predict_assignments(
        input_fn=self.input_fn(points=points, batch_size=num_points)):
      assignments.append(item)
    assignments = np.ravel(assignments)
    self.assertAllEqual(true_assignments, assignments)

    # Test score
    score = gmm.score(input_fn=self.input_fn(points=points,
                                             batch_size=num_points), steps=1)
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
    gmm = gmm_lib.GMM(self.num_centers,
                      initial_clusters=self.initial_means,
                      covariance_type=cov_type,
                      config=run_config.RunConfig(tf_random_seed=2))
    gmm.fit(input_fn=self.input_fn(), steps=iterations)
    points = self.points[:10, :]
    skflow_assignments = []
    for item in gmm.predict_assignments(
        input_fn=self.input_fn(points=points, batch_size=10)):
      skflow_assignments.append(item)
    self.assertAllClose(sklearn_assignments,
                        np.ravel(skflow_assignments).astype(int))
    self.assertAllClose(sklearn_means, gmm.clusters())
    if cov_type == 'full':
      self.assertAllClose(sklearn_covs, gmm.covariances(), rtol=0.01)
    else:
      for d in [0, 1]:
        self.assertAllClose(
            np.diag(sklearn_covs[d]), gmm.covariances()[d, :], rtol=0.01)

  def test_compare_full(self):
    self._compare_with_sklearn('full')

  def test_compare_diag(self):
    self._compare_with_sklearn('diag')

  def test_random_input_large(self):
    # sklearn version.
    iterations = 5  # that should be enough to know whether this diverges
    np.random.seed(5)
    num_classes = 20
    x = np.array([[np.random.random() for _ in range(100)]
                  for _ in range(num_classes)], dtype=np.float32)

    # skflow version.
    gmm = gmm_lib.GMM(num_classes,
                      covariance_type='full',
                      config=run_config.RunConfig(tf_random_seed=2))

    def get_input_fn(x):
      def input_fn():
        return constant_op.constant(x.astype(np.float32)), None
      return input_fn

    gmm.fit(input_fn=get_input_fn(x), steps=iterations)
    self.assertFalse(np.isnan(gmm.clusters()).any())


if __name__ == '__main__':
  test.main()
