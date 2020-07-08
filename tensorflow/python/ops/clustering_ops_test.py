# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.
# ==============================================================================
"""Tests for clustering_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.ops import clustering_ops
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class KmeansPlusPlusInitializationTest(test.TestCase):

  # All but one input point are close to (101, 1). With uniform random sampling,
  # it is highly improbable for (-1, -1) to be selected.
  def setUp(self):
    self._points = np.array([[100., 0.],
                             [101., 2.],
                             [102., 0.],
                             [100., 1.],
                             [100., 2.],
                             [101., 0.],
                             [101., 0.],
                             [101., 1.],
                             [102., 0.],
                             [-1., -1.]]).astype(np.float32)

  def runTestWithSeed(self, seed):
    with self.cached_session():
      sampled_points = clustering_ops.kmeans_plus_plus_initialization(
          self._points, 3, seed, (seed % 5) - 1)
      self.assertAllClose(
          sorted(self.evaluate(sampled_points).tolist()),
          [[-1., -1.], [101., 1.], [101., 1.]],
          atol=1.0)

  def testBasic(self):
    for seed in range(100):
      self.runTestWithSeed(seed)


@test_util.run_all_in_graph_and_eager_modes
class KMC2InitializationTest(test.TestCase):

  def runTestWithSeed(self, seed):
    with self.cached_session():
      distances = np.zeros(1000).astype(np.float32)
      distances[6] = 10e7
      distances[4] = 10e3

      sampled_point = clustering_ops.kmc2_chain_initialization(distances, seed)
      self.assertAllEqual(sampled_point, 6)
      distances[6] = 0.0
      sampled_point = clustering_ops.kmc2_chain_initialization(distances, seed)
      self.assertAllEqual(sampled_point, 4)

  def testBasic(self):
    for seed in range(100):
      self.runTestWithSeed(seed)


@test_util.run_all_in_graph_and_eager_modes
class KMC2InitializationLargeTest(test.TestCase):

  def setUp(self):
    self._distances = np.zeros(1001)
    self._distances[500] = 100.0
    self._distances[1000] = 50.0

  def testBasic(self):
    with self.cached_session():
      counts = {}
      seed = 0
      for i in range(50):
        sample = self.evaluate(
            clustering_ops.kmc2_chain_initialization(self._distances, seed + i))
        counts[sample] = counts.get(sample, 0) + 1
      self.assertEqual(len(counts), 2)
      self.assertTrue(500 in counts)
      self.assertTrue(1000 in counts)
      self.assertGreaterEqual(counts[500], 5)
      self.assertGreaterEqual(counts[1000], 5)


@test_util.run_all_in_graph_and_eager_modes
class KMC2InitializationCornercaseTest(test.TestCase):

  def setUp(self):
    self._distances = np.zeros(10)

  def runTestWithSeed(self, seed):
    with self.cached_session():
      sampled_point = clustering_ops.kmc2_chain_initialization(
          self._distances, seed)
      self.assertAllEqual(sampled_point, 0)

  def testBasic(self):
    for seed in range(100):
      self.runTestWithSeed(seed)


@test_util.run_all_in_graph_and_eager_modes
# A simple test that can be verified by hand.
class NearestCentersTest(test.TestCase):

  def setUp(self):
    self._points = np.array([[100., 0.],
                             [101., 2.],
                             [99., 2.],
                             [1., 1.]]).astype(np.float32)

    self._centers = np.array([[100., 0.],
                              [99., 1.],
                              [50., 50.],
                              [0., 0.],
                              [1., 1.]]).astype(np.float32)

  def testNearest1(self):
    with self.cached_session():
      [indices, distances] = clustering_ops.nearest_neighbors(self._points,
                                                              self._centers, 1)
      self.assertAllClose(indices, [[0], [0], [1], [4]])
      self.assertAllClose(distances, [[0.], [5.], [1.], [0.]])

  def testNearest2(self):
    with self.cached_session():
      [indices, distances] = clustering_ops.nearest_neighbors(self._points,
                                                              self._centers, 2)
      self.assertAllClose(indices, [[0, 1], [0, 1], [1, 0], [4, 3]])
      self.assertAllClose(distances, [[0., 2.], [5., 5.], [1., 5.], [0., 2.]])


@test_util.run_all_in_graph_and_eager_modes
# A test with large inputs.
class NearestCentersLargeTest(test.TestCase):

  def setUp(self):
    num_points = 1000
    num_centers = 2000
    num_dim = 100
    max_k = 5
    # Construct a small number of random points and later tile them.
    points_per_tile = 10
    assert num_points % points_per_tile == 0
    points = np.random.standard_normal(
        [points_per_tile, num_dim]).astype(np.float32)
    # Construct random centers.
    self._centers = np.random.standard_normal(
        [num_centers, num_dim]).astype(np.float32)

    # Exhaustively compute expected nearest neighbors.
    def squared_distance(x, y):
      return np.linalg.norm(x - y, ord=2)**2

    nearest_neighbors = [
        sorted([(squared_distance(point, self._centers[j]), j)
                for j in range(num_centers)])[:max_k] for point in points
    ]
    expected_nearest_neighbor_indices = np.array(
        [[i for _, i in nn] for nn in nearest_neighbors])
    expected_nearest_neighbor_squared_distances = np.array(
        [[dist for dist, _ in nn] for nn in nearest_neighbors])
    # Tile points and expected results to reach requested size (num_points)
    (self._points, self._expected_nearest_neighbor_indices,
     self._expected_nearest_neighbor_squared_distances) = (
         np.tile(x, (int(num_points / points_per_tile), 1))
         for x in (points, expected_nearest_neighbor_indices,
                   expected_nearest_neighbor_squared_distances))

  def testNearest1(self):
    with self.cached_session():
      [indices, distances] = clustering_ops.nearest_neighbors(self._points,
                                                              self._centers, 1)
      self.assertAllClose(
          indices,
          self._expected_nearest_neighbor_indices[:, [0]])
      self.assertAllClose(
          distances,
          self._expected_nearest_neighbor_squared_distances[:, [0]])

  def testNearest5(self):
    with self.cached_session():
      [indices, distances] = clustering_ops.nearest_neighbors(self._points,
                                                              self._centers, 5)
      self.assertAllClose(
          indices,
          self._expected_nearest_neighbor_indices[:, 0:5])
      self.assertAllClose(
          distances,
          self._expected_nearest_neighbor_squared_distances[:, 0:5])


if __name__ == "__main__":
  np.random.seed(0)
  test.main()
