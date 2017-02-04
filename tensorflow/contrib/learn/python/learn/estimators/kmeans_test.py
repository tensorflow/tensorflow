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
"""Tests for KMeans."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import time

# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, 'getdlopenflags') and hasattr(sys, 'setdlopenflags'):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

import numpy as np
from sklearn.cluster import KMeans as SklearnKMeans

from tensorflow.contrib import factorization
from tensorflow.contrib.learn.python.learn.estimators import kmeans as kmeans_lib
from tensorflow.contrib.learn.python.learn.estimators import run_config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import flags
from tensorflow.python.platform import test
from tensorflow.python.training import input as input_lib

FLAGS = flags.FLAGS


def normalize(x):
  return x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True))


def cosine_similarity(x, y):
  return np.dot(normalize(x), np.transpose(normalize(y)))


def make_random_centers(num_centers, num_dims, center_norm=500):
  return np.round(
      np.random.rand(num_centers, num_dims).astype(np.float32) * center_norm)


def make_random_points(centers, num_points, max_offset=20):
  num_centers, num_dims = centers.shape
  assignments = np.random.choice(num_centers, num_points)
  offsets = np.round(
      np.random.randn(num_points, num_dims).astype(np.float32) * max_offset)
  return (centers[assignments] + offsets, assignments,
          np.add.reduce(offsets * offsets, 1))


class KMeansTestBase(test.TestCase):

  def input_fn(self, batch_size=None, points=None, num_epochs=None):
    """Returns an input_fn that randomly selects batches from given points."""
    batch_size = batch_size or self.batch_size
    points = points if points is not None else self.points
    num_points = points.shape[0]

    def _fn():
      x = constant_op.constant(points)
      if batch_size == num_points:
        return input_lib.limit_epochs(x, num_epochs=num_epochs), None
      indices = random_ops.random_uniform(
          constant_op.constant([batch_size]),
          minval=0,
          maxval=num_points - 1,
          dtype=dtypes.int32,
          seed=10)
      batch = array_ops.gather(x, indices)
      return (input_lib.limit_epochs(batch, num_epochs=num_epochs), None)

    return _fn

  @staticmethod
  def config(tf_random_seed):
    return run_config.RunConfig(tf_random_seed=tf_random_seed)

  @property
  def batch_size(self):
    return self.num_points

  @property
  def use_mini_batch(self):
    return False


class KMeansTest(KMeansTestBase):

  def setUp(self):
    np.random.seed(3)
    self.num_centers = 5
    self.num_dims = 2
    self.num_points = 10000
    self.true_centers = make_random_centers(self.num_centers, self.num_dims)
    self.points, _, self.scores = make_random_points(self.true_centers,
                                                     self.num_points)
    self.true_score = np.add.reduce(self.scores)

  def _kmeans(self, relative_tolerance=None):
    return kmeans_lib.KMeansClustering(
        self.num_centers,
        initial_clusters=factorization.RANDOM_INIT,
        use_mini_batch=self.use_mini_batch,
        config=self.config(14),
        random_seed=10,
        relative_tolerance=relative_tolerance)

  def test_clusters(self):
    kmeans = self._kmeans()
    kmeans.fit(input_fn=self.input_fn(), steps=1)
    clusters = kmeans.clusters()
    self.assertAllEqual(list(clusters.shape), [self.num_centers, self.num_dims])

  def test_fit(self):
    kmeans = self._kmeans()
    kmeans.fit(input_fn=self.input_fn(), steps=1)
    score1 = kmeans.score(
        input_fn=self.input_fn(batch_size=self.num_points), steps=1)
    steps = 10 * self.num_points // self.batch_size
    kmeans.fit(input_fn=self.input_fn(), steps=steps)
    score2 = kmeans.score(
        input_fn=self.input_fn(batch_size=self.num_points), steps=1)
    self.assertTrue(score1 > score2)
    self.assertNear(self.true_score, score2, self.true_score * 0.05)

  def test_monitor(self):
    if self.use_mini_batch:
      return
    kmeans = kmeans_lib.KMeansClustering(
        self.num_centers,
        initial_clusters=factorization.RANDOM_INIT,
        use_mini_batch=self.use_mini_batch,
        config=run_config.RunConfig(tf_random_seed=14),
        random_seed=12,
        relative_tolerance=1e-4)

    kmeans.fit(
        input_fn=self.input_fn(),
        # Force it to train until the relative tolerance monitor stops it.
        steps=None)
    score = kmeans.score(
        input_fn=self.input_fn(batch_size=self.num_points), steps=1)
    self.assertNear(self.true_score, score, self.true_score * 0.005)

  def test_infer(self):
    kmeans = self._kmeans(relative_tolerance=1e-4)
    kmeans.fit(input_fn=self.input_fn())
    clusters = kmeans.clusters()

    # Make a small test set
    num_points = 10
    points, true_assignments, true_offsets = make_random_points(clusters,
                                                                num_points)
    # Test predict
    assignments = list(kmeans.predict_cluster_idx(input_fn=self.input_fn(
        batch_size=num_points, points=points, num_epochs=1)))
    self.assertAllEqual(assignments, true_assignments)

    # Test score
    score = kmeans.score(
        input_fn=lambda: (constant_op.constant(points), None), steps=1)
    self.assertNear(score, np.sum(true_offsets), 0.01 * score)

    # Test transform
    transform = kmeans.transform(
        input_fn=lambda: (constant_op.constant(points), None))
    true_transform = np.maximum(
        0,
        np.sum(np.square(points), axis=1, keepdims=True) - 2 * np.dot(
            points, np.transpose(clusters)) +
        np.transpose(np.sum(np.square(clusters), axis=1, keepdims=True)))
    self.assertAllClose(transform, true_transform, rtol=0.05, atol=10)

  def test_fit_raise_if_num_clusters_larger_than_num_points_random_init(self):
    points = np.array([[2.0, 3.0], [1.6, 8.2]], dtype=np.float32)

    with self.assertRaisesOpError('less'):
      kmeans = kmeans_lib.KMeansClustering(
          num_clusters=3, initial_clusters=factorization.RANDOM_INIT)
      kmeans.fit(input_fn=lambda: (constant_op.constant(points), None),
                 steps=10)

  def test_fit_raise_if_num_clusters_larger_than_num_points_kmeans_plus_plus(
      self):
    points = np.array([[2.0, 3.0], [1.6, 8.2]], dtype=np.float32)

    with self.assertRaisesOpError(AssertionError):
      kmeans = kmeans_lib.KMeansClustering(
          num_clusters=3, initial_clusters=factorization.KMEANS_PLUS_PLUS_INIT)
      kmeans.fit(input_fn=lambda: (constant_op.constant(points), None),
                 steps=10)


class KMeansTestCosineDistance(KMeansTestBase):

  def setUp(self):
    self.points = np.array(
        [[2.5, 0.1], [2, 0.2], [3, 0.1], [4, 0.2], [0.1, 2.5], [0.2, 2],
         [0.1, 3], [0.2, 4]],
        dtype=np.float32)
    self.num_points = self.points.shape[0]
    self.true_centers = np.array(
        [
            normalize(
                np.mean(
                    normalize(self.points)[0:4, :], axis=0, keepdims=True))[0],
            normalize(
                np.mean(
                    normalize(self.points)[4:, :], axis=0, keepdims=True))[0]
        ],
        dtype=np.float32)
    self.true_assignments = [0] * 4 + [1] * 4
    self.true_score = len(self.points) - np.tensordot(
        normalize(self.points), self.true_centers[self.true_assignments])

    self.num_centers = 2
    self.kmeans = kmeans_lib.KMeansClustering(
        self.num_centers,
        initial_clusters=factorization.RANDOM_INIT,
        distance_metric=factorization.COSINE_DISTANCE,
        use_mini_batch=self.use_mini_batch,
        config=self.config(3))

  def test_fit(self):
    self.kmeans.fit(input_fn=self.input_fn(), steps=10)
    centers = normalize(self.kmeans.clusters())
    self.assertAllClose(
        np.sort(
            centers, axis=0), np.sort(
                self.true_centers, axis=0))

  def test_transform(self):
    self.kmeans.fit(input_fn=self.input_fn(), steps=10)
    centers = normalize(self.kmeans.clusters())
    true_transform = 1 - cosine_similarity(self.points, centers)
    transform = self.kmeans.transform(input_fn=self.input_fn())
    self.assertAllClose(transform, true_transform, atol=1e-3)

  def test_predict(self):
    self.kmeans.fit(input_fn=self.input_fn(), steps=30)

    centers = normalize(self.kmeans.clusters())
    self.assertAllClose(
        np.sort(
            centers, axis=0), np.sort(
                self.true_centers, axis=0), atol=1e-2)

    assignments = list(self.kmeans.predict_cluster_idx(
        input_fn=self.input_fn(num_epochs=1)))
    self.assertAllClose(
        centers[assignments],
        self.true_centers[self.true_assignments],
        atol=1e-2)

    score = self.kmeans.score(input_fn=self.input_fn(), steps=1)
    self.assertAllClose(score, self.true_score, atol=1e-2)

  def test_predict_kmeans_plus_plus(self):
    # Most points are concetrated near one center. KMeans++ is likely to find
    # the less populated centers.
    points = np.array(
        [[2.5, 3.5], [2.5, 3.5], [-2, 3], [-2, 3], [-3, -3], [-3.1, -3.2],
         [-2.8, -3.], [-2.9, -3.1], [-3., -3.1], [-3., -3.1], [-3.2, -3.],
         [-3., -3.]],
        dtype=np.float32)
    true_centers = np.array(
        [
            normalize(
                np.mean(
                    normalize(points)[0:2, :], axis=0, keepdims=True))[0],
            normalize(
                np.mean(
                    normalize(points)[2:4, :], axis=0, keepdims=True))[0],
            normalize(np.mean(
                normalize(points)[4:, :], axis=0, keepdims=True))[0]
        ],
        dtype=np.float32)
    true_assignments = [0] * 2 + [1] * 2 + [2] * 8
    true_score = len(points) - np.tensordot(
        normalize(points), true_centers[true_assignments])

    kmeans = kmeans_lib.KMeansClustering(
        3,
        initial_clusters=factorization.KMEANS_PLUS_PLUS_INIT,
        distance_metric=factorization.COSINE_DISTANCE,
        use_mini_batch=self.use_mini_batch,
        config=self.config(3))
    kmeans.fit(input_fn=lambda: (constant_op.constant(points), None), steps=30)

    centers = normalize(kmeans.clusters())
    self.assertAllClose(
        sorted(centers.tolist()), sorted(true_centers.tolist()), atol=1e-2)

    def _input_fn():
      return (
          input_lib.limit_epochs(constant_op.constant(points), num_epochs=1),
          None)
    assignments = list(kmeans.predict_cluster_idx(input_fn=_input_fn))
    self.assertAllClose(
        centers[assignments], true_centers[true_assignments], atol=1e-2)

    score = kmeans.score(
        input_fn=lambda: (constant_op.constant(points), None), steps=1)
    self.assertAllClose(score, true_score, atol=1e-2)


class MiniBatchKMeansTest(KMeansTest):

  @property
  def batch_size(self):
    return 450

  @property
  def use_mini_batch(self):
    return True


class KMeansBenchmark(benchmark.Benchmark):
  """Base class for benchmarks."""

  def SetUp(self,
            dimension=50,
            num_clusters=50,
            points_per_cluster=10000,
            center_norm=500,
            cluster_width=20):
    np.random.seed(123456)
    self.num_clusters = num_clusters
    self.num_points = num_clusters * points_per_cluster
    self.centers = make_random_centers(
        self.num_clusters, dimension, center_norm=center_norm)
    self.points, _, scores = make_random_points(
        self.centers, self.num_points, max_offset=cluster_width)
    self.score = float(np.sum(scores))

  def _report(self, num_iters, start, end, scores):
    print(scores)
    self.report_benchmark(
        iters=num_iters,
        wall_time=(end - start) / num_iters,
        extras={'true_sum_squared_distances': self.score,
                'fit_scores': scores})

  def _fit(self, num_iters=10):
    pass

  def benchmark_01_2dim_5center_500point(self):
    self.SetUp(dimension=2, num_clusters=5, points_per_cluster=100)
    self._fit()

  def benchmark_02_20dim_20center_10kpoint(self):
    self.SetUp(dimension=20, num_clusters=20, points_per_cluster=500)
    self._fit()

  def benchmark_03_100dim_50center_50kpoint(self):
    self.SetUp(dimension=100, num_clusters=50, points_per_cluster=1000)
    self._fit()

  def benchmark_03_100dim_50center_50kpoint_unseparated(self):
    self.SetUp(
        dimension=100,
        num_clusters=50,
        points_per_cluster=1000,
        cluster_width=250)
    self._fit()

  def benchmark_04_100dim_500center_500kpoint(self):
    self.SetUp(dimension=100, num_clusters=500, points_per_cluster=1000)
    self._fit(num_iters=4)

  def benchmark_05_100dim_500center_500kpoint_unseparated(self):
    self.SetUp(
        dimension=100,
        num_clusters=500,
        points_per_cluster=1000,
        cluster_width=250)
    self._fit(num_iters=4)


class TensorflowKMeansBenchmark(KMeansBenchmark):

  def _fit(self, num_iters=10):
    scores = []
    start = time.time()
    for i in range(num_iters):
      print('Starting tensorflow KMeans: %d' % i)
      tf_kmeans = kmeans_lib.KMeansClustering(
          self.num_clusters,
          initial_clusters=factorization.KMEANS_PLUS_PLUS_INIT,
          kmeans_plus_plus_num_retries=int(math.log(self.num_clusters) + 2),
          random_seed=i * 42,
          config=run_config.RunConfig(tf_random_seed=3))
      tf_kmeans.fit(input_fn=lambda: (constant_op.constant(self.points), None),
                    steps=50,
                    relative_tolerance=1e-6)
      _ = tf_kmeans.clusters()
      scores.append(
          tf_kmeans.score(
              input_fn=lambda: (constant_op.constant(self.points), None),
              steps=1))
    self._report(num_iters, start, time.time(), scores)


class SklearnKMeansBenchmark(KMeansBenchmark):

  def _fit(self, num_iters=10):
    scores = []
    start = time.time()
    for i in range(num_iters):
      print('Starting sklearn KMeans: %d' % i)
      sklearn_kmeans = SklearnKMeans(
          n_clusters=self.num_clusters,
          init='k-means++',
          max_iter=50,
          n_init=1,
          tol=1e-4,
          random_state=i * 42)
      sklearn_kmeans.fit(self.points)
      scores.append(sklearn_kmeans.inertia_)
    self._report(num_iters, start, time.time(), scores)


if __name__ == '__main__':
  test.main()
