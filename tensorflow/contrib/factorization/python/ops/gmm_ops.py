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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
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
  num_points = math_ops.cast(array_ops.shape(x)[0], dtypes.float32)
  x -= math_ops.reduce_mean(x, 0, keepdims=True)
  if diag:
    cov = math_ops.reduce_sum(
        math_ops.square(x), 0, keepdims=True) / (num_points - 1)
  else:
    cov = math_ops.matmul(x, x, transpose_a=True) / (num_points - 1)
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
  num_data = math_ops.add_n([array_ops.shape(inp)[0] for inp in data])
  with ops.control_dependencies(
      [check_ops.assert_less_equal(num_clusters, num_data)]):
    indices = random_ops.random_uniform(
        [num_clusters],
        minval=0,
        maxval=math_ops.cast(num_data, dtypes.int64),
        seed=random_seed,
        dtype=dtypes.int64)
  indices %= math_ops.cast(num_data, dtypes.int64)
  clusters_init = embedding_lookup(data, indices, partition_strategy='div')
  return clusters_init


class GmmAlgorithm(object):
  """Tensorflow Gaussian mixture model clustering class."""
  CLUSTERS_WEIGHT = 'alphas'
  CLUSTERS_VARIABLE = 'clusters'
  CLUSTERS_COVS_VARIABLE = 'clusters_covs'

  def __init__(self,
               data,
               num_classes,
               initial_means=None,
               params='wmc',
               covariance_type=FULL_COVARIANCE,
               random_seed=0):
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
    self._dimensions = array_ops.shape(first_shard)[1]
    self._num_classes = num_classes
    # Small value to guarantee that covariances are invertible.
    self._min_var = array_ops.diag(
        array_ops.ones(array_ops.stack([self._dimensions]))) * 1e-3
    self._create_variables()
    self._initialize_variables(data, initial_means)
    # Operations of partial statistics for the computation of the means.
    self._w_mul_x = []
    # Operations of partial statistics for the computation of the covariances.
    self._w_mul_x2 = []
    self._define_graph(data)

  def _create_variables(self):
    """Initializes GMM algorithm."""
    init_value = array_ops.constant([], dtype=dtypes.float32)
    self._means = variables.VariableV1(init_value,
                                       name=self.CLUSTERS_VARIABLE,
                                       validate_shape=False)
    self._covs = variables.VariableV1(
        init_value, name=self.CLUSTERS_COVS_VARIABLE, validate_shape=False)
    # Mixture weights, representing the probability that a randomly
    # selected unobservable data (in EM terms) was generated by component k.
    self._alpha = variable_scope.variable(
        array_ops.tile([1.0 / self._num_classes], [self._num_classes]),
        name=self.CLUSTERS_WEIGHT,
        validate_shape=False)
    self._cluster_centers_initialized = variables.VariableV1(False,
                                                             dtype=dtypes.bool,
                                                             name='initialized')

  def _initialize_variables(self, data, initial_means=None):
    """Initializes variables.

    Args:
      data: a list of Tensors with data, each row is a new example.
      initial_means: a Tensor with a matrix of means.
    """
    first_shard = data[0]
    # Initialize means: num_classes X 1 X dimensions.
    if initial_means is not None:
      means = array_ops.expand_dims(initial_means, 1)
    else:
      # Sample data randomly
      means = array_ops.expand_dims(
          _init_clusters_random(data, self._num_classes, self._random_seed), 1)

    # Initialize covariances.
    if self._covariance_type == FULL_COVARIANCE:
      cov = _covariance(first_shard, False) + self._min_var
      # A matrix per class, num_classes X dimensions X dimensions
      covs = array_ops.tile(
          array_ops.expand_dims(cov, 0), [self._num_classes, 1, 1])
    elif self._covariance_type == DIAG_COVARIANCE:
      cov = _covariance(first_shard, True) + self._min_var
      # A diagonal per row, num_classes X dimensions.
      covs = array_ops.tile(
          array_ops.expand_dims(array_ops.diag_part(cov), 0),
          [self._num_classes, 1])

    with ops.colocate_with(self._cluster_centers_initialized):
      initialized = control_flow_ops.with_dependencies(
          [means, covs],
          array_ops.identity(self._cluster_centers_initialized))
    self._init_ops = []
    with ops.colocate_with(self._means):
      init_means = state_ops.assign(self._means, means, validate_shape=False)
      init_means = control_flow_ops.with_dependencies(
          [init_means],
          state_ops.assign(self._cluster_centers_initialized, True))
      self._init_ops.append(control_flow_ops.cond(initialized,
                                                  control_flow_ops.no_op,
                                                  lambda: init_means).op)
    with ops.colocate_with(self._covs):
      init_covs = state_ops.assign(self._covs, covs, validate_shape=False)
      init_covs = control_flow_ops.with_dependencies(
          [init_covs],
          state_ops.assign(self._cluster_centers_initialized, True))
      self._init_ops.append(control_flow_ops.cond(initialized,
                                                  control_flow_ops.no_op,
                                                  lambda: init_covs).op)

  def init_ops(self):
    """Returns the initialization operation."""
    return control_flow_ops.group(*self._init_ops)

  def training_ops(self):
    """Returns the training operation."""
    return control_flow_ops.group(*self._train_ops)

  def is_initialized(self):
    """Returns a boolean operation for initialized variables."""
    return self._cluster_centers_initialized

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
      ret.append(math_ops.argmax(w, 1))
    return ret

  def scores(self):
    """Returns the per-sample likelihood fo the data.

    Returns:
      Log probabilities of each data point.
    """
    return self._scores

  def log_likelihood_op(self):
    """Returns the log-likelihood operation."""
    return self._log_likelihood_op

  def _define_graph(self, data):
    """Define graph for a single iteration.

    Args:
      data: a list of Tensors defining the training data.
    """
    for shard_id, shard in enumerate(data):
      self._num_examples = array_ops.shape(shard)[0]
      shard = array_ops.expand_dims(shard, 0)
      self._define_log_prob_operation(shard_id, shard)
      self._define_prior_log_prob_operation(shard_id)
      self._define_expectation_operation(shard_id)
      self._define_partial_maximization_operation(shard_id, shard)
    self._define_maximization_operation(len(data))
    self._define_loglikelihood_operation()
    self._define_score_samples()

  def _define_full_covariance_probs(self, shard_id, shard):
    """Defines the full covariance probabilities per example in a class.

    Updates a matrix with dimension num_examples X num_classes.

    Args:
      shard_id: id of the current shard.
      shard: current data shard, 1 X num_examples X dimensions.
    """
    diff = shard - self._means
    cholesky = linalg_ops.cholesky(self._covs + self._min_var)
    log_det_covs = 2.0 * math_ops.reduce_sum(
        math_ops.log(array_ops.matrix_diag_part(cholesky)), 1)
    x_mu_cov = math_ops.square(
        linalg_ops.matrix_triangular_solve(
            cholesky, array_ops.transpose(
                diff, perm=[0, 2, 1]), lower=True))
    diag_m = array_ops.transpose(math_ops.reduce_sum(x_mu_cov, 1))
    self._probs[shard_id] = (
        -0.5 * (diag_m + math_ops.cast(self._dimensions, dtypes.float32) *
                math_ops.log(2 * np.pi) + log_det_covs))

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
    det_expanded = math_ops.reduce_sum(
        math_ops.log(self._covs + 1e-3), 1, keepdims=True)
    x2 = math_ops.squared_difference(shard, self._means)
    cov_expanded = array_ops.expand_dims(1.0 / (self._covs + 1e-3), 2)
    # num_classes X num_examples
    x2_cov = math_ops.matmul(x2, cov_expanded)
    x2_cov = array_ops.transpose(array_ops.squeeze(x2_cov, [2]))
    self._probs[shard_id] = -0.5 * (
        math_ops.cast(self._dimensions, dtypes.float32) *
        math_ops.log(2.0 * np.pi) +
        array_ops.transpose(det_expanded) + x2_cov)

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
    self._probs[shard_id] += math_ops.log(self._alpha)

  def _define_prior_log_prob_operation(self, shard_id):
    """Computes the prior probability of all samples.

    Updates a vector where each item is the prior probability of an
    input example.

    Args:
      shard_id: id of current shard_id.
    """
    self._prior_probs[shard_id] = math_ops.reduce_logsumexp(
        self._probs[shard_id], axis=1, keepdims=True)

  def _define_expectation_operation(self, shard_id):
    # Shape broadcasting.
    probs = array_ops.expand_dims(self._probs[shard_id], 0)
    # Membership weights are computed as:
    # $$w_{ik} = \frac{\alpha_k f(\mathbf{y_i}|\mathbf{\theta}_k)}$$
    # $$            {\sum_{m=1}^{K}\alpha_mf(\mathbf{y_i}|\mathbf{\theta}_m)}$$
    # where "i" is the i-th example, "k" is the k-th mixture, theta are
    # the model parameters and y_i the observations.
    # These are defined for each shard.
    self._w[shard_id] = array_ops.reshape(
        math_ops.exp(probs - self._prior_probs[shard_id]),
        array_ops.stack([self._num_examples, self._num_classes]))

  def _define_partial_maximization_operation(self, shard_id, shard):
    """Computes the partial statistics of the means and covariances.

    Args:
      shard_id: current shard id.
      shard: current data shard, 1 X num_examples X dimensions.
    """
    # Soft assignment of each data point to each of the two clusters.
    self._points_in_k[shard_id] = math_ops.reduce_sum(
        self._w[shard_id], 0, keepdims=True)
    # Partial means.
    w_mul_x = array_ops.expand_dims(
        math_ops.matmul(
            self._w[shard_id], array_ops.squeeze(shard, [0]), transpose_a=True),
        1)
    self._w_mul_x.append(w_mul_x)
    # Partial covariances.
    x = array_ops.concat([shard for _ in range(self._num_classes)], 0)
    x_trans = array_ops.transpose(x, perm=[0, 2, 1])
    x_mul_w = array_ops.concat([
        array_ops.expand_dims(x_trans[k, :, :] * self._w[shard_id][:, k], 0)
        for k in range(self._num_classes)
    ], 0)
    self._w_mul_x2.append(math_ops.matmul(x_mul_w, x))

  def _define_maximization_operation(self, num_batches):
    """Maximization operations."""
    # TODO(xavigonzalvo): some of these operations could be moved to C++.
    # Compute the effective number of data points assigned to component k.
    with ops.control_dependencies(self._w):
      points_in_k = array_ops.squeeze(
          math_ops.add_n(self._points_in_k), axis=[0])
      # Update alpha.
      if 'w' in self._params:
        final_points_in_k = points_in_k / num_batches
        num_examples = math_ops.cast(math_ops.reduce_sum(final_points_in_k),
                                     dtypes.float32)
        self._alpha_op = self._alpha.assign(final_points_in_k /
                                            (num_examples + MEPS))
      else:
        self._alpha_op = control_flow_ops.no_op()
      self._train_ops = [self._alpha_op]

      # Update means.
      points_in_k_expanded = array_ops.reshape(points_in_k,
                                               [self._num_classes, 1, 1])
      if 'm' in self._params:
        self._means_op = self._means.assign(
            math_ops.div(
                math_ops.add_n(self._w_mul_x), points_in_k_expanded + MEPS))
      else:
        self._means_op = control_flow_ops.no_op()
      # means are (num_classes x 1 x dims)

      # Update covariances.
      with ops.control_dependencies([self._means_op]):
        b = math_ops.add_n(self._w_mul_x2) / (points_in_k_expanded + MEPS)
        new_covs = []
        for k in range(self._num_classes):
          mean = self._means.value()[k, :, :]
          square_mean = math_ops.matmul(mean, mean, transpose_a=True)
          new_cov = b[k, :, :] - square_mean + self._min_var
          if self._covariance_type == FULL_COVARIANCE:
            new_covs.append(array_ops.expand_dims(new_cov, 0))
          elif self._covariance_type == DIAG_COVARIANCE:
            new_covs.append(
                array_ops.expand_dims(array_ops.diag_part(new_cov), 0))
        new_covs = array_ops.concat(new_covs, 0)
        if 'c' in self._params:
          # Train operations don't need to take care of the means
          # because covariances already depend on it.
          with ops.control_dependencies([self._means_op, new_covs]):
            self._train_ops.append(
                state_ops.assign(
                    self._covs, new_covs, validate_shape=False))

  def _define_loglikelihood_operation(self):
    """Defines the total log-likelihood of current iteration."""
    op = []
    for prior_probs in self._prior_probs:
      op.append(math_ops.reduce_logsumexp(prior_probs))
    self._log_likelihood_op = math_ops.reduce_logsumexp(op)

  def _define_score_samples(self):
    """Defines the likelihood of each data sample."""
    op = []
    for shard_id, prior_probs in enumerate(self._prior_probs):
      op.append(prior_probs + math_ops.log(self._w[shard_id]))
    self._scores = array_ops.squeeze(
        math_ops.reduce_logsumexp(op, axis=2, keepdims=True), axis=0)


def gmm(inp,
        initial_clusters,
        num_clusters,
        random_seed,
        covariance_type=FULL_COVARIANCE,
        params='wmc'):
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
    assignments: A vector (or list of vectors). Each element in the vector
      corresponds to an input row in 'inp' and specifies the cluster id
      corresponding to the input.
    training_op: an op that runs an iteration of training.
    init_op: an op that runs the initialization.
  """
  initial_means = None
  if initial_clusters != 'random' and not isinstance(initial_clusters,
                                                     ops.Tensor):
    initial_means = constant_op.constant(initial_clusters, dtype=dtypes.float32)

  # Implementation of GMM.
  inp = inp if isinstance(inp, list) else [inp]
  gmm_tool = GmmAlgorithm(inp, num_clusters, initial_means, params,
                          covariance_type, random_seed)
  assignments = gmm_tool.assignments()
  scores = gmm_tool.scores()
  loss = gmm_tool.log_likelihood_op()
  return (loss, scores, [assignments], gmm_tool.training_ops(),
          gmm_tool.init_ops(), gmm_tool.is_initialized())
