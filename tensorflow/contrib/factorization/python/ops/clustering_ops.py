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
"""Clustering Operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_clustering_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.embedding_ops import embedding_lookup
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_clustering_ops import *
# pylint: enable=wildcard-import

# Euclidean distance between vectors U and V is defined as \\(||U - V||_F\\)
# which is the square root of the sum of the absolute squares of the elements
# difference.
SQUARED_EUCLIDEAN_DISTANCE = 'squared_euclidean'
# Cosine distance between vectors U and V is defined as
# \\(1 - (U \dot V) / (||U||_F ||V||_F)\\)
COSINE_DISTANCE = 'cosine'

RANDOM_INIT = 'random'
KMEANS_PLUS_PLUS_INIT = 'kmeans_plus_plus'
KMC2_INIT = 'kmc2'

# The name of the variable holding the cluster centers. Used by the Estimator.
CLUSTERS_VAR_NAME = 'clusters'


class KMeans(object):
  """Creates the graph for k-means clustering."""

  def __init__(self,
               inputs,
               num_clusters,
               initial_clusters=RANDOM_INIT,
               distance_metric=SQUARED_EUCLIDEAN_DISTANCE,
               use_mini_batch=False,
               mini_batch_steps_per_iteration=1,
               random_seed=0,
               kmeans_plus_plus_num_retries=2,
               kmc2_chain_length=200):
    """Creates an object for generating KMeans clustering graph.

    This class implements the following variants of K-means algorithm:

    If use_mini_batch is False, it runs standard full batch K-means. Each step
    runs a single iteration of K-Means. This step can be run sharded across
    multiple workers by passing a list of sharded inputs to this class. Note
    however that a single step needs to process the full input at once.

    If use_mini_batch is True, it runs a generalization of the mini-batch
    K-means algorithm. It runs multiple iterations, where each iteration is
    composed of mini_batch_steps_per_iteration steps. Two copies of cluster
    centers are maintained: one that is updated at the end of each iteration,
    and one that is updated every step. The first copy is used to compute
    cluster allocations for each step, and for inference, while the second copy
    is the one updated each step using the mini-batch update rule. After each
    iteration is complete, this second copy is copied back the first copy.

    Note that for use_mini_batch=True, when mini_batch_steps_per_iteration=1,
    the algorithm reduces to the standard mini-batch algorithm. Also by setting
    mini_batch_steps_per_iteration = num_inputs / batch_size, the algorithm
    becomes an asynchronous version of the full-batch algorithm. Note however
    that there is no guarantee by this implementation that each input is seen
    exactly once per iteration. Also, different updates are applied
    asynchronously without locking. So this asynchronous version may not behave
    exactly like a full-batch version.

    Args:
      inputs: An input tensor or list of input tensors. It is assumed that the
        data points have been previously randomly permuted.
      num_clusters: An integer tensor specifying the number of clusters. This
        argument is ignored if initial_clusters is a tensor or numpy array.
      initial_clusters: Specifies the clusters used during initialization. One
        of the following:
        - a tensor or numpy array with the initial cluster centers.
        - a function f(inputs, k) that returns up to k centers from `inputs`.
        - "random": Choose centers randomly from `inputs`.
        - "kmeans_plus_plus": Use kmeans++ to choose centers from `inputs`.
        - "kmc2": Use the fast k-MC2 algorithm to choose centers from `inputs`.
        In the last three cases, one batch of `inputs` may not yield
        `num_clusters` centers, in which case initialization will require
        multiple batches until enough centers are chosen. In the case of
        "random" or "kmeans_plus_plus", if the input size is <= `num_clusters`
        then the entire batch is chosen to be cluster centers.
      distance_metric: Distance metric used for clustering. Supported options:
        "squared_euclidean", "cosine".
      use_mini_batch: If true, use the mini-batch k-means algorithm. Else assume
        full batch.
      mini_batch_steps_per_iteration: Number of steps after which the updated
        cluster centers are synced back to a master copy.
      random_seed: Seed for PRNG used to initialize seeds.
      kmeans_plus_plus_num_retries: For each point that is sampled during
        kmeans++ initialization, this parameter specifies the number of
        additional points to draw from the current distribution before selecting
        the best. If a negative value is specified, a heuristic is used to
        sample O(log(num_to_sample)) additional points.
      kmc2_chain_length: Determines how many candidate points are used by the
        k-MC2 algorithm to produce one new cluster centers. If a (mini-)batch
        contains less points, one new cluster center is generated from the
        (mini-)batch.

    Raises:
      ValueError: An invalid argument was passed to initial_clusters or
        distance_metric.
    """
    if isinstance(initial_clusters, str) and initial_clusters not in [
        RANDOM_INIT, KMEANS_PLUS_PLUS_INIT, KMC2_INIT
    ]:
      raise ValueError(
          "Unsupported initialization algorithm '%s'" % initial_clusters)
    if distance_metric not in [SQUARED_EUCLIDEAN_DISTANCE, COSINE_DISTANCE]:
      raise ValueError("Unsupported distance metric '%s'" % distance_metric)
    self._inputs = inputs if isinstance(inputs, list) else [inputs]
    self._num_clusters = num_clusters
    self._initial_clusters = initial_clusters
    self._distance_metric = distance_metric
    self._use_mini_batch = use_mini_batch
    self._mini_batch_steps_per_iteration = int(mini_batch_steps_per_iteration)
    self._random_seed = random_seed
    self._kmeans_plus_plus_num_retries = kmeans_plus_plus_num_retries
    self._kmc2_chain_length = kmc2_chain_length

  @classmethod
  def _distance_graph(cls, inputs, clusters, distance_metric):
    """Computes distance between each input and each cluster center.

    Args:
      inputs: list of input Tensors.
      clusters: cluster Tensor.
      distance_metric: distance metric used for clustering

    Returns:
      list of Tensors, where each element corresponds to each element in inputs.
      The value is the distance of each row to all the cluster centers.
      Currently only Euclidean distance and cosine distance are supported.
    """
    assert isinstance(inputs, list)
    if distance_metric == SQUARED_EUCLIDEAN_DISTANCE:
      return cls._compute_euclidean_distance(inputs, clusters)
    elif distance_metric == COSINE_DISTANCE:
      return cls._compute_cosine_distance(
          inputs, clusters, inputs_normalized=True)
    else:
      assert False, str(distance_metric)

  @classmethod
  def _compute_euclidean_distance(cls, inputs, clusters):
    """Computes Euclidean distance between each input and each cluster center.

    Args:
      inputs: list of input Tensors.
      clusters: cluster Tensor.

    Returns:
      list of Tensors, where each element corresponds to each element in inputs.
      The value is the distance of each row to all the cluster centers.
    """
    output = []
    for inp in inputs:
      with ops.colocate_with(inp, ignore_existing=True):
        # Computes Euclidean distance. Note the first and third terms are
        # broadcast additions.
        squared_distance = (
            math_ops.reduce_sum(math_ops.square(inp), 1, keepdims=True) -
            2 * math_ops.matmul(inp, clusters, transpose_b=True) +
            array_ops.transpose(
                math_ops.reduce_sum(
                    math_ops.square(clusters), 1, keepdims=True)))
        output.append(squared_distance)

    return output

  @classmethod
  def _compute_cosine_distance(cls, inputs, clusters, inputs_normalized=True):
    """Computes cosine distance between each input and each cluster center.

    Args:
      inputs: list of input Tensor.
      clusters: cluster Tensor
      inputs_normalized: if True, it assumes that inp and clusters are
      normalized and computes the dot product which is equivalent to the cosine
      distance. Else it L2 normalizes the inputs first.

    Returns:
      list of Tensors, where each element corresponds to each element in inp.
      The value is the distance of each row to all the cluster centers.
    """
    output = []
    if not inputs_normalized:
      with ops.colocate_with(clusters, ignore_existing=True):
        clusters = nn_impl.l2_normalize(clusters, dim=1)
    for inp in inputs:
      with ops.colocate_with(inp, ignore_existing=True):
        if not inputs_normalized:
          inp = nn_impl.l2_normalize(inp, dim=1)
        output.append(1 - math_ops.matmul(inp, clusters, transpose_b=True))
    return output

  def _infer_graph(self, inputs, clusters):
    """Maps input to closest cluster and the score.

    Args:
      inputs: list of input Tensors.
      clusters: Tensor of cluster centers.

    Returns:
      List of tuple, where each value in tuple corresponds to a value in inp.
      The tuple has following three elements:
      all_scores: distance of each input to each cluster center.
      score: distance of each input to closest cluster center.
      cluster_idx: index of cluster center closest to the corresponding input.
    """
    assert isinstance(inputs, list)
    # Pairwise distances are used only by transform(). In all other cases, this
    # sub-graph is not evaluated.
    scores = self._distance_graph(inputs, clusters, self._distance_metric)
    output = []
    if (self._distance_metric == COSINE_DISTANCE and
        not self._clusters_l2_normalized()):
      # The cosine distance between normalized vectors x and y is the same as
      # 2 * squared_euclidean_distance. We are using this fact and reusing the
      # nearest_neighbors op.
      # TODO(ands): Support COSINE distance in nearest_neighbors and remove
      # this.
      with ops.colocate_with(clusters, ignore_existing=True):
        clusters = nn_impl.l2_normalize(clusters, dim=1)
    for inp, score in zip(inputs, scores):
      with ops.colocate_with(inp, ignore_existing=True):
        (indices, distances) = gen_clustering_ops.nearest_neighbors(
            inp, clusters, 1)
        if self._distance_metric == COSINE_DISTANCE:
          distances *= 0.5
        output.append((score, array_ops.squeeze(distances, [-1]),
                       array_ops.squeeze(indices, [-1])))
    return zip(*output)

  def _clusters_l2_normalized(self):
    """Returns True if clusters centers are kept normalized."""
    return (self._distance_metric == COSINE_DISTANCE and
            (not self._use_mini_batch or
             self._mini_batch_steps_per_iteration > 1))

  def _create_variables(self, num_clusters):
    """Creates variables.

    Args:
      num_clusters: an integer Tensor providing the number of clusters.

    Returns:
      Tuple with following elements:
      - cluster_centers: a Tensor for storing cluster centers
      - cluster_centers_initialized: bool Variable indicating whether clusters
            are initialized.
      - cluster_counts: a Tensor for storing counts of points assigned to this
            cluster. This is used by mini-batch training.
      - cluster_centers_updated: Tensor representing copy of cluster centers
            that are updated every step.
      - update_in_steps: numbers of steps left before we sync
            cluster_centers_updated back to cluster_centers.
    """
    init_value = array_ops.constant([], dtype=dtypes.float32)
    cluster_centers = variable_scope.variable(
        init_value, name=CLUSTERS_VAR_NAME, validate_shape=False)
    cluster_centers_initialized = variable_scope.variable(
        False, dtype=dtypes.bool, name='initialized')

    if self._use_mini_batch and self._mini_batch_steps_per_iteration > 1:
      # Copy of cluster centers actively updated each step according to
      # mini-batch update rule.
      cluster_centers_updated = variable_scope.variable(
          init_value, name='clusters_updated', validate_shape=False)
      # How many steps till we copy the updated clusters to cluster_centers.
      update_in_steps = variable_scope.variable(
          self._mini_batch_steps_per_iteration,
          dtype=dtypes.int64,
          name='update_in_steps')
      # Count of points assigned to cluster_centers_updated.
      cluster_counts = variable_scope.variable(
          array_ops.zeros([num_clusters], dtype=dtypes.int64))
    else:
      cluster_centers_updated = cluster_centers
      update_in_steps = None
      cluster_counts = (
          variable_scope.variable(
              array_ops.ones([num_clusters], dtype=dtypes.int64))
          if self._use_mini_batch else None)
    return (cluster_centers, cluster_centers_initialized, cluster_counts,
            cluster_centers_updated, update_in_steps)

  @classmethod
  def _l2_normalize_data(cls, inputs):
    """Normalized the input data."""
    output = []
    for inp in inputs:
      with ops.colocate_with(inp, ignore_existing=True):
        output.append(nn_impl.l2_normalize(inp, dim=1))
    return output

  def training_graph(self):
    """Generate a training graph for kmeans algorithm.

    This returns, among other things, an op that chooses initial centers
    (init_op), a boolean variable that is set to True when the initial centers
    are chosen (cluster_centers_initialized), and an op to perform either an
    entire Lloyd iteration or a mini-batch of a Lloyd iteration (training_op).
    The caller should use these components as follows. A single worker should
    execute init_op multiple times until cluster_centers_initialized becomes
    True. Then multiple workers may execute training_op any number of times.

    Returns:
      A tuple consisting of:
      all_scores: A matrix (or list of matrices) of dimensions (num_input,
        num_clusters) where the value is the distance of an input vector and a
        cluster center.
      cluster_idx: A vector (or list of vectors). Each element in the vector
        corresponds to an input row in 'inp' and specifies the cluster id
        corresponding to the input.
      scores: Similar to cluster_idx but specifies the distance to the
        assigned cluster instead.
      cluster_centers_initialized: scalar indicating whether clusters have been
        initialized.
      init_op: an op to initialize the clusters.
      training_op: an op that runs an iteration of training.
    """
    # Implementation of kmeans.
    if (isinstance(self._initial_clusters, str) or
        callable(self._initial_clusters)):
      initial_clusters = self._initial_clusters
      num_clusters = ops.convert_to_tensor(self._num_clusters)
    else:
      initial_clusters = ops.convert_to_tensor(self._initial_clusters)
      num_clusters = array_ops.shape(initial_clusters)[0]

    inputs = self._inputs
    (cluster_centers_var, cluster_centers_initialized, total_counts,
     cluster_centers_updated,
     update_in_steps) = self._create_variables(num_clusters)
    init_op = _InitializeClustersOpFactory(
        self._inputs, num_clusters, initial_clusters, self._distance_metric,
        self._random_seed, self._kmeans_plus_plus_num_retries,
        self._kmc2_chain_length, cluster_centers_var, cluster_centers_updated,
        cluster_centers_initialized).op()
    cluster_centers = cluster_centers_var

    if self._distance_metric == COSINE_DISTANCE:
      inputs = self._l2_normalize_data(inputs)
      if not self._clusters_l2_normalized():
        cluster_centers = nn_impl.l2_normalize(cluster_centers, dim=1)

    all_scores, scores, cluster_idx = self._infer_graph(inputs, cluster_centers)
    if self._use_mini_batch:
      sync_updates_op = self._mini_batch_sync_updates_op(
          update_in_steps, cluster_centers_var, cluster_centers_updated,
          total_counts)
      assert sync_updates_op is not None
      with ops.control_dependencies([sync_updates_op]):
        training_op = self._mini_batch_training_op(
            inputs, cluster_idx, cluster_centers_updated, total_counts)
    else:
      assert cluster_centers == cluster_centers_var
      training_op = self._full_batch_training_op(
          inputs, num_clusters, cluster_idx, cluster_centers_var)

    return (all_scores, cluster_idx, scores, cluster_centers_initialized,
            init_op, training_op)

  def _mini_batch_sync_updates_op(self, update_in_steps, cluster_centers_var,
                                  cluster_centers_updated, total_counts):
    if self._use_mini_batch and self._mini_batch_steps_per_iteration > 1:
      assert update_in_steps is not None
      with ops.colocate_with(update_in_steps, ignore_existing=True):

        def _f():
          # Note that there is a race condition here, so we do a best effort
          # updates here. We reset update_in_steps first so that other workers
          # don't duplicate the updates. Also we update cluster_center_vars
          # before resetting total_counts to avoid large updates to
          # cluster_centers_updated based on partially updated
          # cluster_center_vars.
          with ops.control_dependencies([
              state_ops.assign(update_in_steps,
                               self._mini_batch_steps_per_iteration - 1)
          ]):
            with ops.colocate_with(
                cluster_centers_updated, ignore_existing=True):
              if self._distance_metric == COSINE_DISTANCE:
                cluster_centers = nn_impl.l2_normalize(
                    cluster_centers_updated, dim=1)
              else:
                cluster_centers = cluster_centers_updated
            with ops.colocate_with(cluster_centers_var, ignore_existing=True):
              with ops.control_dependencies(
                  [state_ops.assign(cluster_centers_var, cluster_centers)]):
                with ops.colocate_with(None, ignore_existing=True):
                  with ops.control_dependencies([
                      state_ops.assign(total_counts,
                                       array_ops.zeros_like(total_counts))
                  ]):
                    return array_ops.identity(update_in_steps)

        return control_flow_ops.cond(
            update_in_steps <= 0, _f,
            lambda: state_ops.assign_sub(update_in_steps, 1))
    else:
      return control_flow_ops.no_op()

  def _mini_batch_training_op(self, inputs, cluster_idx_list, cluster_centers,
                              total_counts):
    """Creates an op for training for mini batch case.

    Args:
      inputs: list of input Tensors.
      cluster_idx_list: A vector (or list of vectors). Each element in the
        vector corresponds to an input row in 'inp' and specifies the cluster id
        corresponding to the input.
      cluster_centers: Tensor Ref of cluster centers.
      total_counts: Tensor Ref of cluster counts.

    Returns:
      An op for doing an update of mini-batch k-means.
    """
    update_ops = []
    for inp, cluster_idx in zip(inputs, cluster_idx_list):
      with ops.colocate_with(inp, ignore_existing=True):
        assert total_counts is not None
        cluster_idx = array_ops.reshape(cluster_idx, [-1])
        # Dedupe the unique ids of cluster_centers being updated so that updates
        # can be locally aggregated.
        unique_ids, unique_idx = array_ops.unique(cluster_idx)
        num_unique_cluster_idx = array_ops.size(unique_ids)
        # Fetch the old values of counts and cluster_centers.
        with ops.colocate_with(total_counts, ignore_existing=True):
          old_counts = array_ops.gather(total_counts, unique_ids)
        # TODO(agarwal): This colocation seems to run into problems. Fix it.
        with ops.colocate_with(cluster_centers, ignore_existing=True):
          old_cluster_centers = array_ops.gather(cluster_centers, unique_ids)
        # Locally aggregate the increment to counts.
        count_updates = math_ops.unsorted_segment_sum(
            array_ops.ones_like(unique_idx, dtype=total_counts.dtype),
            unique_idx, num_unique_cluster_idx)
        # Locally compute the sum of inputs mapped to each id.
        # For a cluster with old cluster value x, old count n, and with data
        # d_1,...d_k newly assigned to it, we recompute the new value as
        # \\(x += (sum_i(d_i) - k * x) / (n + k)\\).
        # Compute \\(sum_i(d_i)\\), see comment above.
        cluster_center_updates = math_ops.unsorted_segment_sum(
            inp, unique_idx, num_unique_cluster_idx)
        # Shape to enable broadcasting count_updates and learning_rate to inp.
        # It extends the shape with 1's to match the rank of inp.
        broadcast_shape = array_ops.concat([
            array_ops.reshape(num_unique_cluster_idx, [1]),
            array_ops.ones(
                array_ops.reshape(array_ops.rank(inp) - 1, [1]),
                dtype=dtypes.int32)
        ], 0)
        # Subtract k * x, see comment above.
        cluster_center_updates -= math_ops.cast(
            array_ops.reshape(count_updates, broadcast_shape),
            inp.dtype) * old_cluster_centers
        learning_rate = math_ops.reciprocal(
            math_ops.cast(old_counts + count_updates, inp.dtype))
        learning_rate = array_ops.reshape(learning_rate, broadcast_shape)
        # scale by 1 / (n + k), see comment above.
        cluster_center_updates *= learning_rate
        # Apply the updates.
      update_counts = state_ops.scatter_add(total_counts, unique_ids,
                                            count_updates)
      update_cluster_centers = state_ops.scatter_add(
          cluster_centers, unique_ids, cluster_center_updates)
      update_ops.extend([update_counts, update_cluster_centers])
    return control_flow_ops.group(*update_ops)

  def _full_batch_training_op(self, inputs, num_clusters, cluster_idx_list,
                              cluster_centers):
    """Creates an op for training for full batch case.

    Args:
      inputs: list of input Tensors.
      num_clusters: an integer Tensor providing the number of clusters.
      cluster_idx_list: A vector (or list of vectors). Each element in the
        vector corresponds to an input row in 'inp' and specifies the cluster id
        corresponding to the input.
      cluster_centers: Tensor Ref of cluster centers.

    Returns:
      An op for doing an update of mini-batch k-means.
    """
    cluster_sums = []
    cluster_counts = []
    epsilon = constant_op.constant(1e-6, dtype=inputs[0].dtype)
    for inp, cluster_idx in zip(inputs, cluster_idx_list):
      with ops.colocate_with(inp, ignore_existing=True):
        cluster_sums.append(
            math_ops.unsorted_segment_sum(inp, cluster_idx, num_clusters))
        cluster_counts.append(
            math_ops.unsorted_segment_sum(
                array_ops.reshape(
                    array_ops.ones(
                        array_ops.reshape(array_ops.shape(inp)[0], [-1])),
                    [-1, 1]), cluster_idx, num_clusters))
    with ops.colocate_with(cluster_centers, ignore_existing=True):
      new_clusters_centers = math_ops.add_n(cluster_sums) / (
          math_ops.cast(math_ops.add_n(cluster_counts), cluster_sums[0].dtype) +
          epsilon)
      if self._clusters_l2_normalized():
        new_clusters_centers = nn_impl.l2_normalize(new_clusters_centers, dim=1)
    return state_ops.assign(cluster_centers, new_clusters_centers)


class _InitializeClustersOpFactory(object):
  """Internal class to create the op to initialize the clusters.

    The op performs this algorithm (see constructor args):

    num_remaining = num_clusters - length(cluster_centers)
    if num_remaining == 0:
      assert that cluster_centers_initialized is true
    else:
      assert that num_remaining > 0
      new_centers = choose up to num_remaining initial centers
      l2-normalize new_centers if using cosine distance
      all_centers = concat(cluster_centers, new_centers)
      cluster_centers := all_centers
      if there is a cluster_centers_updated variable:
        cluster_centers_updated := cluster_centers
      num_now_remaining = num_clusters - length(cluster_centers)
      if num_now_remaining == 0:
        cluster_centers_initialized := true
  """

  # TODO(ccolby): Refactor this class so that kmc2 isn't so much a special case.

  def __init__(self, inputs, num_clusters, initial_clusters, distance_metric,
               random_seed, kmeans_plus_plus_num_retries, kmc2_chain_length,
               cluster_centers, cluster_centers_updated,
               cluster_centers_initialized):
    """Creates an op factory.

    Args:
      inputs: See KMeans constructor.
      num_clusters: An integer Tensor providing the number of clusters.
      initial_clusters: See KMeans constructor.
      distance_metric: See KMeans constructor.
      random_seed: See KMeans constructor.
      kmeans_plus_plus_num_retries: See KMeans constructor.
      kmc2_chain_length: See KMeans constructor.
      cluster_centers: The TF variable holding the initial centers. It may
          already contain some centers when the op is executed.
      cluster_centers_updated: A second TF variable to hold a copy of the
          initial centers, used for full-batch mode. In mini-batch mode,
          cluster_centers_updated is the same variable as cluster_centers.
      cluster_centers_initialized: A boolean TF variable that will be set
          to true when all the initial centers have been chosen.
    """
    # All of these instance variables are constants.
    self._inputs = inputs
    self._num_clusters = num_clusters
    self._initial_clusters = initial_clusters
    self._distance_metric = distance_metric
    self._random_seed = random_seed
    self._kmeans_plus_plus_num_retries = kmeans_plus_plus_num_retries
    self._kmc2_chain_length = kmc2_chain_length
    self._cluster_centers = cluster_centers
    self._cluster_centers_updated = cluster_centers_updated
    self._cluster_centers_initialized = cluster_centers_initialized

    self._num_selected = array_ops.shape(self._cluster_centers)[0]
    self._num_remaining = self._num_clusters - self._num_selected
    self._num_data = math_ops.add_n(
        [array_ops.shape(i)[0] for i in self._inputs])

  def _random(self):
    indices = random_ops.random_uniform(
        array_ops.reshape(self._num_remaining, [-1]),
        minval=0,
        maxval=math_ops.cast(self._num_data, dtypes.int64),
        seed=self._random_seed,
        dtype=dtypes.int64)
    return embedding_lookup(self._inputs, indices, partition_strategy='div')

  def _kmeans_plus_plus(self):
    # Points from only the first shard are used for initializing centers.
    # TODO(ands): Use all points.
    inp = self._inputs[0]
    if self._distance_metric == COSINE_DISTANCE:
      inp = nn_impl.l2_normalize(inp, dim=1)
    return gen_clustering_ops.kmeans_plus_plus_initialization(
        inp,
        math_ops.cast(self._num_remaining, dtypes.int64),
        self._random_seed,
        self._kmeans_plus_plus_num_retries)

  def _kmc2_multiple_centers(self):
    """Adds new initial cluster centers using the k-MC2 algorithm.

    In each call to the op, the provided batch is split into subsets based on
    the specified `kmc2_chain_length`. On each subset, a single Markov chain of
    the k-MC2 algorithm is used to add *one* new center cluster center. If there
    are less than `kmc2_chain_length` points in the subset, a single center is
    added using one Markov chain on the full input. It is assumed that the
    provided batch has previously been randomly permuted. Otherwise, k-MC2 may
    return suboptimal centers.

    Returns:
      An op that adds new cluster centers.
    """
    # The op only operates on the first shard of data.
    first_shard = self._inputs[0]
    # Number of points in the input that can be used.
    batch_size = array_ops.shape(first_shard)[0]
    # Maximum number of subsets such that the size of each subset is at least
    # `kmc2_chain_length`. Final subsets may be larger.
    max_to_sample = math_ops.cast(
        batch_size / self._kmc2_chain_length, dtype=dtypes.int32)
    # We sample at least one new center and at most all remaining centers.
    num_to_sample = math_ops.maximum(
        math_ops.minimum(self._num_remaining, max_to_sample), 1)

    def _cond(i, _):
      """Stopping condition for the while loop."""
      return math_ops.less(i, num_to_sample)

    def _body(i, _):
      """Body that adds a single new center based on a subset."""

      def _sample_random():
        """Returns a random point as a cluster center."""
        # By assumption the batch is reshuffled and _sample_random is always
        # called for i=0. Hence, we simply return the first point.
        new_center = array_ops.reshape(first_shard[0], [1, -1])
        if self._distance_metric == COSINE_DISTANCE:
          new_center = nn_impl.l2_normalize(new_center, dim=1)
        return new_center

      def _sample_kmc2_chain():
        """Returns previous centers as well as a new center sampled using k-MC2.
        """
        # Extract the subset from the underlying batch.
        start = i * self._kmc2_chain_length
        end = start + self._kmc2_chain_length
        subset = first_shard[start:end]
        # Compute the distances from points in the subset to previous centers.
        _, distances = gen_clustering_ops.nearest_neighbors(
            subset, self._cluster_centers, 1)
        # Sample index of new center using k-MC2 Markov chain.
        new_center_index = gen_clustering_ops.kmc2_chain_initialization(
            array_ops.squeeze(distances), self._random_seed)
        # Extract actual new center.
        newly_sampled_center = array_ops.reshape(subset[new_center_index],
                                                 [1, -1])
        # Return concatenation with previously sampled centers.
        if self._distance_metric == COSINE_DISTANCE:
          newly_sampled_center = nn_impl.l2_normalize(
              newly_sampled_center, dim=1)
        return array_ops.concat([self._cluster_centers, newly_sampled_center],
                                0)

      # Obtain a random point if there are no previously sampled centers.
      # Otherwise, construct a k-MC2 Markov chain.
      new_centers = control_flow_ops.cond(
          math_ops.equal(self._num_selected, 0), _sample_random,
          _sample_kmc2_chain)
      # Assign new cluster centers to underlying variable.
      assigned_centers = state_ops.assign(
          self._cluster_centers, new_centers, validate_shape=False)
      if self._cluster_centers_updated is not self._cluster_centers:
        assigned_centers = state_ops.assign(
            self._cluster_centers_updated,
            assigned_centers,
            validate_shape=False)
      return i + 1, self._num_clusters - array_ops.shape(assigned_centers)[0]

    # Add num_to_sample new data points.
    _, num_remaining = control_flow_ops.while_loop(_cond, _body, [0, 0])
    return num_remaining

  def _greedy_batch_sampler(self, sampler):
    # If the input dataset size is smaller than the number of centers
    # remaining, choose the entire input dataset as centers. This can happen
    # with mini-batch. Otherwise, sample the batch according to the provided
    # sampler.
    return control_flow_ops.cond(self._num_data <= self._num_remaining,
                                 lambda: array_ops.concat(self._inputs, 0),
                                 sampler)

  def _single_batch_sampler(self, sampler):
    # Enforce that there are at least as many data points as centers
    # remaining. This gives the provided sampler the chance to select all
    # remaining centers from a single batch.
    with ops.control_dependencies(
        [check_ops.assert_greater_equal(self._num_data, self._num_remaining)]):
      return sampler()

  def _choose_initial_centers(self):
    if isinstance(self._initial_clusters, str):
      if self._initial_clusters == RANDOM_INIT:
        return self._greedy_batch_sampler(self._random)
      else:  # self._initial_clusters == KMEANS_PLUS_PLUS_INIT
        return self._single_batch_sampler(self._kmeans_plus_plus)
    elif callable(self._initial_clusters):
      return self._initial_clusters(self._inputs, self._num_remaining)
    else:
      with ops.control_dependencies([
          check_ops.assert_equal(self._num_remaining,
                                 array_ops.shape(self._initial_clusters)[0])
      ]):
        return self._initial_clusters

  def _add_new_centers(self):
    """Adds some centers and returns the number of centers remaining."""
    new_centers = self._choose_initial_centers()
    if self._distance_metric == COSINE_DISTANCE:
      new_centers = nn_impl.l2_normalize(new_centers, dim=1)
    # If cluster_centers is empty, it doesn't have the right shape for concat.
    all_centers = control_flow_ops.cond(
        math_ops.equal(self._num_selected, 0), lambda: new_centers,
        lambda: array_ops.concat([self._cluster_centers, new_centers], 0))
    # TODO(ccolby): De-dupe all_centers?
    a = state_ops.assign(
        self._cluster_centers, all_centers, validate_shape=False)
    if self._cluster_centers_updated is not self._cluster_centers:
      a = state_ops.assign(
          self._cluster_centers_updated, a, validate_shape=False)
    return self._num_clusters - array_ops.shape(a)[0]

  def _initialize(self):
    with ops.control_dependencies([
        check_ops.assert_positive(self._num_remaining),
    ]):
      if self._initial_clusters == KMC2_INIT:
        num_now_remaining = self._kmc2_multiple_centers()
      else:
        num_now_remaining = self._add_new_centers()
      return control_flow_ops.cond(
          math_ops.equal(num_now_remaining, 0),
          lambda: state_ops.assign(self._cluster_centers_initialized, True),
          control_flow_ops.no_op)

  def op(self):
    """Returns the cluster initializer op."""
    return control_flow_ops.cond(
        math_ops.equal(self._num_remaining, 0),
        lambda: check_ops.assert_equal(self._cluster_centers_initialized, True),
        self._initialize)
