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

"""Tests for gmm_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.factorization.python.ops import gmm_ops
from tensorflow.python.platform import tf_logging as logging


class GmmOpsTest(tf.test.TestCase):

  def setUp(self):
    self.num_examples = 1000
    self.iterations = 40
    self.seed = 4
    tf.set_random_seed(self.seed)
    np.random.seed(self.seed * 2)
    self.data, self.true_assignments = self.make_data(self.num_examples)
    # Generate more complicated data.
    self.centers = [[1, 1], [-1, 0.5], [2, 1]]
    self.more_data, self.more_true_assignments = self.make_data_from_centers(
        self.num_examples, self.centers)

  @staticmethod
  def make_data(num_vectors):
    """Generates 2-dimensional data centered on (2,2), (-1,-1).

    Args:
      num_vectors: number of training examples.

    Returns:
      A tuple containing the data as a numpy array and the cluster ids.
    """
    vectors = []
    classes = []
    for _ in xrange(num_vectors):
      if np.random.random() > 0.5:
        vectors.append([np.random.normal(2.0, 0.6),
                        np.random.normal(2.0, 0.9)])
        classes.append(0)
      else:
        vectors.append([np.random.normal(-1.0, 0.4),
                        np.random.normal(-1.0, 0.5)])
        classes.append(1)
    return np.asarray(vectors), classes

  @staticmethod
  def make_data_from_centers(num_vectors, centers):
    """Generates 2-dimensional data with random centers.

    Args:
      num_vectors: number of training examples.
      centers: a list of random 2-dimensional centers.

    Returns:
      A tuple containing the data as a numpy array and the cluster ids.
    """
    vectors = []
    classes = []
    for _ in xrange(num_vectors):
      current_class = np.random.random_integers(0, len(centers) - 1)
      vectors.append([np.random.normal(centers[current_class][0],
                                       np.random.random_sample()),
                      np.random.normal(centers[current_class][1],
                                       np.random.random_sample())])
      classes.append(current_class)
    return np.asarray(vectors), len(centers)

  def test_covariance(self):
    start_time = time.time()
    data = self.data.T
    np_cov = np.cov(data)
    logging.info('Numpy took %f', time.time() - start_time)

    start_time = time.time()
    with self.test_session() as sess:
      op = gmm_ops._covariance(
          tf.constant(data.T, dtype=tf.float32),
          False)
      op_diag = gmm_ops._covariance(
          tf.constant(data.T, dtype=tf.float32),
          True)
      tf.global_variables_initializer().run()
      tf_cov = sess.run(op)
      np.testing.assert_array_almost_equal(np_cov, tf_cov)
      logging.info('Tensorflow took %f', time.time() - start_time)
      tf_cov = sess.run(op_diag)
      np.testing.assert_array_almost_equal(
          np.diag(np_cov), np.ravel(tf_cov), decimal=5)

  def test_simple_cluster(self):
    """Tests that the clusters are correct."""
    num_classes = 2
    graph = tf.Graph()
    with graph.as_default() as g:
      g.seed = 5
      with self.test_session() as sess:
        data = tf.constant(self.data, dtype=tf.float32)
        _, assignments, _, training_op = tf.contrib.factorization.gmm(
            data, 'random', num_classes, random_seed=self.seed)

        tf.global_variables_initializer().run()
        for _ in xrange(self.iterations):
          sess.run(training_op)
        assignments = sess.run(assignments)
        accuracy = np.mean(
            np.asarray(self.true_assignments) == np.squeeze(assignments))
        logging.info('Accuracy: %f', accuracy)
        self.assertGreater(accuracy, 0.98)

  def testParams(self):
    """Tests that the params work as intended."""
    num_classes = 2
    with self.test_session() as sess:
      # Experiment 1. Update weights only.
      data = tf.constant(self.data, dtype=tf.float32)
      gmm_tool = tf.contrib.factorization.GmmAlgorithm([data], num_classes,
                                                       [[3.0, 3.0], [0.0, 0.0]],
                                                       'w')
      training_ops = gmm_tool.training_ops()
      tf.global_variables_initializer().run()
      for _ in xrange(self.iterations):
        sess.run(training_ops)

      # Only the probability to each class is updated.
      alphas = sess.run(gmm_tool.alphas())
      self.assertGreater(alphas[1], 0.6)
      means = sess.run(gmm_tool.clusters())
      np.testing.assert_almost_equal(
          np.expand_dims([[3.0, 3.0], [0.0, 0.0]], 1), means)
      covs = sess.run(gmm_tool.covariances())
      np.testing.assert_almost_equal(covs[0], covs[1])

      # Experiment 2. Update means and covariances.
      gmm_tool = tf.contrib.factorization.GmmAlgorithm([data], num_classes,
                                                       [[3.0, 3.0], [0.0, 0.0]],
                                                       'mc')
      training_ops = gmm_tool.training_ops()
      tf.global_variables_initializer().run()
      for _ in xrange(self.iterations):
        sess.run(training_ops)
      alphas = sess.run(gmm_tool.alphas())
      self.assertAlmostEqual(alphas[0], alphas[1])
      means = sess.run(gmm_tool.clusters())
      np.testing.assert_almost_equal(
          np.expand_dims([[2.0, 2.0], [-1.0, -1.0]], 1), means, decimal=1)
      covs = sess.run(gmm_tool.covariances())
      np.testing.assert_almost_equal(
          [[0.371111, -0.0050774], [-0.0050774, 0.8651744]],
          covs[0], decimal=4)
      np.testing.assert_almost_equal(
          [[0.146976, 0.0259463], [0.0259463, 0.2543971]],
          covs[1], decimal=4)

      # Experiment 3. Update covariances only.
      gmm_tool = tf.contrib.factorization.GmmAlgorithm(
          [data], num_classes, [[-1.0, -1.0], [1.0, 1.0]], 'c')
      training_ops = gmm_tool.training_ops()
      tf.global_variables_initializer().run()
      for _ in xrange(self.iterations):
        sess.run(training_ops)
      alphas = sess.run(gmm_tool.alphas())
      self.assertAlmostEqual(alphas[0], alphas[1])
      means = sess.run(gmm_tool.clusters())
      np.testing.assert_almost_equal(
          np.expand_dims([[-1.0, -1.0], [1.0, 1.0]], 1), means)
      covs = sess.run(gmm_tool.covariances())
      np.testing.assert_almost_equal(
          [[0.1299582, 0.0435872], [0.0435872, 0.2558578]],
          covs[0], decimal=5)
      np.testing.assert_almost_equal(
          [[3.195385, 2.6989155], [2.6989155, 3.3881593]],
          covs[1], decimal=5)


if __name__ == '__main__':
  tf.test.main()
