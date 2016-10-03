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

"""Gaussian mixture models Operations."""
# TODO(xavigonzalvo): Factor out covariance matrix operations to make
# code reusable for different types (e.g. diag).

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.ops.embedding_ops import embedding_lookup

# Machine epsilon.
MEPS = np.finfo(float).eps
FULL_COVARIANCE = 'full'
DIAG_COVARIANCE = 'diag'


def _covariance(x, diag):
  """Defines the covariance operation of a matrix.

  Args:
    x: a matrix Tensor. Dimension 0 should contain the number of examples.
    diag: if True, it computes the diagonal covariance.

  Returns:
    A Tensor representing the covariance of x. In the case of
  diagonal matrix just the diagonal is returned.
  """
  num_points = tf.to_float(tf.shape(x)[0])
  x -= tf.reduce_mean(x, 0, keep_dims=True)
  if diag:
    cov = tf.reduce_sum(
        tf.square(x), 0, keep_dims=True) / (num_points - 1)
  else:
    cov = tf.matmul(x, x, transpose_a=True)  / (num_points - 1)
  return cov


def _init_clusters_random(data, num_clusters, random_seed):
  """Does random initialization of clusters.

  Args:
    data: a list of Tensors with a matrix of data, each row is an example.
    num_clusters: an integer with the number of clusters.
    random_seed: Seed for PRNG used to initialize seeds.

  Returns:
    A Tensor with num_clusters random rows of data.
  """
  assert isinstance(data, list)
  num_data = tf.add_n([tf.shape(inp)[0] for inp in data])
  with tf.control_dependencies([tf.assert_less_equal(num_clusters, num_data)]):
    indices = tf.random_uniform([num_clusters],
                                minval=0,
                                maxval=tf.cast(num_data, tf.int64),
                                seed=random_seed,
                                dtype=tf.int64)
  indices = tf.cast(indices, tf.int32) % num_data
  clusters_init = embedding_lookup(data, indices, partition_strategy='div')
  return clusters_init


class GmmAlgorithm(object):
  """Tensorflow Gaussian mixture model clustering class."""
  CLUSTERS_VARIABLE = 'clusters'
  CLUSTERS_COVS_VARIABLE = 'clusters_covs'

  def __init__(self, data, num_classes, initial_means=None, params='wmc',
               covariance_type=FULL_COVARIANCE, random_seed=0):
    """Constructor.

    Args:
      data: a list of Tensors with data, each row is a new example.
      num_classes: number of clusters.
      initial_means: a Tensor with a matrix of means. If None, means are
        computed by sampling randomly.
      params: Controls which parameters are updated in the training
        process. Can contain any combination of "w" for weights, "m" for
        means, and "c" for covariances.
      covariance_type: one of "full", "diag".
      random_seed: Seed for PRNG used to initialize seeds.

    Raises:
      Exception if covariance type is unknown.
    """
    self._params = params
    self._random_seed = random_seed
    self._covariance_type = covariance_type
    if self._covariance_type not in [DIAG_COVARIANCE, FULL_COVARIANCE]:
      raise Exception(  # pylint: disable=g-doc-exception
          'programmer error: Invalid covariance type: %s' %
          self._covariance_type)
    # Create sharded variables for multiple shards. The following
    # lists are indexed by shard.
    # Probability per example in a class.
    num_shards = len(data)
    self._probs = [None] * num_shards
    # Prior probability.
    self._prior_probs = [None] * num_shards
    # Membership weights w_{ik} where "i" is the i-th example and "k"
    # is the k-th mixture.
    self._w = [None] * num_shards
    # Number of examples in a class.
    self._points_in_k = [None] * num_shards
    first_shard = data[0]
    self._dimensions = tf.shape(first_shard)[1]
    self._num_classes = num_classes
    # Small value to guarantee that covariances are invertible.
    self._min_var = tf.diag(tf.ones(tf.pack([self._dimensions]))) * 1e-3
    self._create_variables(data, initial_means)
    # Operations of partial statistics for the computation of the means.
    self._w_mul_x = []
    # Operations of partial statistics for the computation of the covariances.
    self._w_mul_x2 = []
    self._define_graph(data)

  def _create_variables(self, data, initial_means=None):
    """Initializes GMM algorithm.

    Args:
      data: a list of Tensors with data, each row is a new example.
      initial_means: a Tensor with a matrix of means.
    """
    first_shard = data[0]
    # Initialize means: num_classes X 1 X dimensions.
    if initial_means is not None:
      self._means = tf.Variable(tf.expand_dims(initial_means, 1),
                                name=self.CLUSTERS_VARIABLE,
                                validate_shape=False, dtype=tf.float32)
    else:
      # Sample data randomly
      self._means = tf.Variable(tf.expand_dims(
          _init_clusters_random(data, self._num_classes, self._random_seed), 1),
                                name=self.CLUSTERS_VARIABLE,
                                validate_shape=False)

    # Initialize covariances.
    if self._covariance_type == FULL_COVARIANCE:
      cov = _covariance(first_shard, False) + self._min_var
      # A matrix per class, num_classes X dimensions X dimensions
      covs = tf.tile(
          tf.expand_dims(cov, 0), [self._num_classes, 1, 1])
    elif self._covariance_type == DIAG_COVARIANCE:
      cov = _covariance(first_shard, True) + self._min_var
      # A diagonal per row, num_classes X dimensions.
      covs = tf.tile(tf.expand_dims(tf.diag_part(cov), 0),
                     [self._num_classes, 1])
    self._covs = tf.Variable(covs, name='clusters_covs', validate_shape=False)
    # Mixture weights, representing the probability that a randomly
    # selected unobservable data (in EM terms) was generated by component k.
    self._alpha = tf.Variable(tf.tile([1.0 / self._num_classes],
                                      [self._num_classes]))

  def training_ops(self):
    """Returns the training operation."""
    return self._train_ops

  def alphas(self):
    return self._alpha

  def clusters(self):
    """Returns the clusters with dimensions num_classes X 1 X num_dimensions."""
    return self._means

  def covariances(self):
    """Returns the covariances matrices."""
    return self._covs

  def assignments(self):
    """Returns a list of Tensors with the matrix of assignments per shard."""
    ret = []
    for w in self._w:
      ret.append(tf.argmax(w, 1))
    return ret

  def scores(self):
    """Returns the distances to each class.

    Returns:
      A tuple with two Tensors. The first contains the distance to
    each class. The second contains the distance to the assigned
    class.
    """
    return (self._all_scores, self._scores)

  def _define_graph(self, data):
    """Define graph for a single iteration.

    Args:
      data: a list of Tensors defining the training data.
    """
    for shard_id, shard in enumerate(data):
      self._num_examples = tf.shape(shard)[0]
      shard = tf.expand_dims(shard, 0)
      self._define_log_prob_operation(shard_id, shard)
      self._define_prior_log_prob_operation(shard_id)
      self._define_expectation_operation(shard_id)
      self._define_partial_maximization_operation(shard_id, shard)
    self._define_maximization_operation(len(data))
    self._define_distance_to_clusters(data)

  def _define_full_covariance_probs(self, shard_id, shard):
    """Defines the full covariance probabilties per example in a class.

    Updates a matrix with dimension num_examples X num_classes.

    Args:
      shard_id: id of the current shard.
      shard: current data shard, 1 X num_examples X dimensions.
    """
    diff = shard - self._means
    cholesky = tf.cholesky(self._covs + self._min_var)
    log_det_covs = 2.0 * tf.reduce_sum(tf.log(tf.matrix_diag_part(cholesky)), 1)
    x_mu_cov = tf.square(
        tf.matrix_triangular_solve(
            cholesky, tf.transpose(
                diff, perm=[0, 2, 1]), lower=True))
    diag_m = tf.transpose(tf.reduce_sum(x_mu_cov, 1))
    self._probs[shard_id] = -0.5 * (
        diag_m + tf.to_float(self._dimensions) * tf.log(2 * np.pi) +
        log_det_covs)

  def _define_diag_covariance_probs(self, shard_id, shard):
    """Defines the diagonal covariance probabilities per example in a class.

    Args:
      shard_id: id of the current shard.
      shard: current data shard, 1 X num_examples X dimensions.

    Returns a matrix num_examples * num_classes.
    """
    # num_classes X 1
    # TODO(xavigonzalvo): look into alternatives to log for
    # reparametrization of variance parameters.
    det_expanded = tf.reduce_sum(tf.log(self._covs + 1e-3),
                                 1, keep_dims=True)
    diff = shard - self._means
    x2 = tf.square(diff)
    cov_expanded = tf.expand_dims(1.0 / (self._covs + 1e-3), 2)
    # num_classes X num_examples
    x2_cov = tf.batch_matmul(x2, cov_expanded)
    x2_cov = tf.transpose(tf.squeeze(x2_cov, [2]))
    self._probs[shard_id] = -0.5 * (
        tf.to_float(self._dimensions) * tf.log(2.0 * np.pi) +
        tf.transpose(det_expanded) + x2_cov)

  def _define_log_prob_operation(self, shard_id, shard):
    """Probability per example in a class.

    Updates a matrix with dimension num_examples X num_classes.

    Args:
      shard_id: id of the current shard.
      shard: current data shard, 1 X num_examples X dimensions.
    """
    # TODO(xavigonzalvo): Use the pdf defined in
    # third_party/tensorflow/contrib/distributions/python/ops/gaussian.py
    if self._covariance_type == FULL_COVARIANCE:
      self._define_full_covariance_probs(shard_id, shard)
    elif self._covariance_type == DIAG_COVARIANCE:
      self._define_diag_covariance_probs(shard_id, shard)
    self._probs[shard_id] += tf.log(self._alpha)

  def _define_prior_log_prob_operation(self, shard_id):
    """Computes the prior probability of all samples.

    Updates a vector where each item is the prior probabibility of an
    input example.

    Args:
      shard_id: id of current shard_id.
    """
    self._prior_probs[shard_id] = tf.log(
        tf.reduce_sum(tf.exp(self._probs[shard_id]), 1, keep_dims=True))

  def _define_expectation_operation(self, shard_id):
    # Shape broadcasting.
    probs = tf.expand_dims(self._probs[shard_id], 0)
    # Membership weights are computed as:
    # w_{ik} = \frac{\alpha_k f(\mathbf{y_i}|\mathbf{\theta}_k)}
    #               {\sum_{m=1}^{K}\alpha_mf(\mathbf{y_i}|\mathbf{\theta}_m)}
    # where "i" is the i-th example, "k" is the k-th mixture, theta are
    # the model parameters and y_i the observations.
    # These are defined for each shard.
    self._w[shard_id] = tf.reshape(
        tf.exp(probs - self._prior_probs[shard_id]),
        tf.pack([self._num_examples, self._num_classes]))

  def _define_partial_maximization_operation(self, shard_id, shard):
    """Computes the partial statistics of the means and covariances.

    Args:
      shard_id: current shard id.
      shard: current data shard, 1 X num_examples X dimensions.
    """
    # Soft assignment of each data point to each of the two clusters.
    self._points_in_k[shard_id] = tf.reduce_sum(self._w[shard_id], 0,
                                                keep_dims=True)
    # Partial means.
    w_mul_x = tf.expand_dims(
        tf.matmul(self._w[shard_id],
                  tf.squeeze(shard, [0]), transpose_a=True), 1)
    self._w_mul_x.append(w_mul_x)
    # Partial covariances.
    x = tf.concat(0, [shard for _ in range(self._num_classes)])
    x_trans = tf.transpose(x, perm=[0, 2, 1])
    x_mul_w = tf.concat(0, [
        tf.expand_dims(x_trans[k, :, :] * self._w[shard_id][:, k], 0)
        for k in range(self._num_classes)])
    self._w_mul_x2.append(tf.batch_matmul(x_mul_w, x))

  def _define_maximization_operation(self, num_batches):
    """Maximization operations."""
    # TODO(xavigonzalvo): some of these operations could be moved to C++.
    # Compute the effective number of data points assigned to component k.
    with tf.control_dependencies(self._w):
      points_in_k = tf.squeeze(tf.add_n(self._points_in_k), squeeze_dims=[0])
      # Update alpha.
      if 'w' in self._params:
        final_points_in_k = points_in_k / num_batches
        num_examples = tf.to_float(tf.reduce_sum(final_points_in_k))
        self._alpha_op = self._alpha.assign(
            final_points_in_k / (num_examples + MEPS))
      else:
        self._alpha_op = tf.no_op()
      self._train_ops = [self._alpha_op]

      # Update means.
      points_in_k_expanded = tf.reshape(points_in_k,
                                        [self._num_classes, 1, 1])
      if 'm' in self._params:
        self._means_op = self._means.assign(
            tf.div(tf.add_n(self._w_mul_x), points_in_k_expanded + MEPS))
      else:
        self._means_op = tf.no_op()
      # means are (num_classes x 1 x dims)

      # Update covariances.
      with tf.control_dependencies([self._means_op]):
        b = tf.add_n(self._w_mul_x2) / (points_in_k_expanded + MEPS)
        new_covs = []
        for k in range(self._num_classes):
          mean = self._means.ref()[k, :, :]
          square_mean = tf.matmul(mean, mean, transpose_a=True)
          new_cov = b[k, :, :] - square_mean + self._min_var
          if self._covariance_type == FULL_COVARIANCE:
            new_covs.append(tf.expand_dims(new_cov, 0))
          elif self._covariance_type == DIAG_COVARIANCE:
            new_covs.append(tf.expand_dims(tf.diag_part(new_cov), 0))
        new_covs = tf.concat(0, new_covs)
        if 'c' in self._params:
          # Train operations don't need to take care of the means
          # because covariances already depend on it.
          with tf.control_dependencies([self._means_op, new_covs]):
            self._train_ops.append(
                tf.assign(self._covs, new_covs, validate_shape=False))

  def _define_distance_to_clusters(self, data):
    """Defines the Mahalanobis distance to the assigned Gaussian."""
    # TODO(xavigonzalvo): reuse (input - mean) * cov^-1 * (input -
    # mean) from log probability function.
    self._all_scores = []
    for shard in data:
      all_scores = []
      shard = tf.expand_dims(shard, 0)
      for c in xrange(self._num_classes):
        if self._covariance_type == FULL_COVARIANCE:
          cov = self._covs[c, :, :]
        elif self._covariance_type == DIAG_COVARIANCE:
          cov = tf.diag(self._covs[c, :])
        inverse = tf.matrix_inverse(cov + self._min_var)
        inv_cov = tf.tile(
            tf.expand_dims(inverse, 0),
            tf.pack([self._num_examples, 1, 1]))
        diff = tf.transpose(shard - self._means[c, :, :], perm=[1, 0, 2])
        m_left = tf.batch_matmul(diff, inv_cov)
        all_scores.append(tf.sqrt(tf.batch_matmul(
            m_left, tf.transpose(diff, perm=[0, 2, 1])
        )))
      self._all_scores.append(tf.reshape(
          tf.concat(1, all_scores),
          tf.pack([self._num_examples, self._num_classes])))

    # Distance to the associated class.
    self._all_scores = tf.concat(0, self._all_scores)
    assignments = tf.concat(0, self.assignments())
    rows = tf.to_int64(tf.range(0, self._num_examples))
    indices = tf.concat(1, [tf.expand_dims(rows, 1),
                            tf.expand_dims(assignments, 1)])
    self._scores = tf.gather_nd(self._all_scores, indices)

  def _define_loglikelihood_operation(self):
    """Defines the total log-likelihood of current iteration."""
    self._ll_op = []
    for prior_probs in self._prior_probs:
      self._ll_op.append(tf.reduce_sum(tf.log(prior_probs)))
    tf.scalar_summary('ll', tf.reduce_sum(self._ll_op))


def gmm(inp, initial_clusters, num_clusters, random_seed,
        covariance_type=FULL_COVARIANCE, params='wmc'):
  """Creates the graph for Gaussian mixture model (GMM) clustering.

  Args:
    inp: An input tensor or list of input tensors
    initial_clusters: Specifies the clusters used during
      initialization. Can be a tensor or numpy array, or a function
      that generates the clusters. Can also be "random" to specify
      that clusters should be chosen randomly from input data. Note: type
      is diverse to be consistent with skflow.
    num_clusters: number of clusters.
    random_seed: Python integer. Seed for PRNG used to initialize centers.
    covariance_type: one of "diag", "full".
    params: Controls which parameters are updated in the training
      process. Can contain any combination of "w" for weights, "m" for
      means, and "c" for covars.

  Returns:
    Note: tuple of lists returned to be consistent with skflow
    A tuple consisting of:
    all_scores: A matrix (or list of matrices) of dimensions (num_input,
      num_clusters) where the value is the distance of an input vector and a
      cluster center.
    assignments: A vector (or list of vectors). Each element in the vector
      corresponds to an input row in 'inp' and specifies the cluster id
      corresponding to the input.
    scores: Similar to assignments but specifies the distance to the
      assigned cluster instead.
    training_op: an op that runs an iteration of training.
  """
  initial_means = None
  if initial_clusters != 'random' and not isinstance(
      initial_clusters, tf.Tensor):
    initial_means = tf.constant(initial_clusters, dtype=tf.float32)

  # Implementation of GMM.
  inp = inp if isinstance(inp, list) else [inp]
  gmm_tool = GmmAlgorithm(inp, num_clusters, initial_means, params,
                          covariance_type, random_seed)
  training_ops = gmm_tool.training_ops()
  assignments = gmm_tool.assignments()
  all_scores, scores = gmm_tool.scores()
  return [all_scores], [assignments], [scores], tf.group(*training_ops)
