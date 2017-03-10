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

from tensorflow.contrib.factorization.python.ops import gen_clustering_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.contrib.factorization.python.ops.gen_clustering_ops import *
# pylint: enable=wildcard-import
from tensorflow.contrib.util import loader
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.embedding_ops import embedding_lookup
from tensorflow.python.platform import resource_loader

_clustering_ops = loader.load_op_library(
    resource_loader.get_path_to_datafile('_clustering_ops.so'))

# Euclidean distance between vectors U and V is defined as ||U - V||_F which is
# the square root of the sum of the absolute squares of the elements difference.
SQUARED_EUCLIDEAN_DISTANCE = 'squared_euclidean'
# Cosine distance between vectors U and V is defined as
# 1 - (U \dot V) / (||U||_F ||V||_F)
COSINE_DISTANCE = 'cosine'

RANDOM_INIT = 'random'
KMEANS_PLUS_PLUS_INIT = 'kmeans_plus_plus'


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
               kmeans_plus_plus_num_retries=2):
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
      inputs: An input tensor or list of input tensors
      num_clusters: number of clusters.
      initial_clusters: Specifies the clusters used during initialization.  Can
        be a tensor or numpy array, or a function that generates the clusters.
        Can also be "random" to specify that clusters should be chosen randomly
        from input data.
      distance_metric: distance metric used for clustering.
      use_mini_batch: If true, use the mini-batch k-means algorithm. Else assume
        full batch.
      mini_batch_steps_per_iteration: number of steps after which the updated
        cluster centers are synced back to a master copy.
      random_seed: Seed for PRNG used to initialize seeds.
      kmeans_plus_plus_num_retries: For each point that is sampled during
        kmeans++ initialization, this parameter specifies the number of
        additional points to draw from the current distribution before selecting
        the best. If a negative value is specified, a heuristic is used to
        sample O(log(num_to_sample)) additional points.
    """
    self._inputs = inputs if isinstance(inputs, list) else [inputs]
    assert num_clusters > 0, num_clusters
    self._num_clusters = num_clusters
    if initial_clusters is None:
      initial_clusters = RANDOM_INIT
    self._initial_clusters = initial_clusters
    assert distance_metric in [SQUARED_EUCLIDEAN_DISTANCE, COSINE_DISTANCE]
    self._distance_metric = distance_metric
    self._use_mini_batch = use_mini_batch
    self._mini_batch_steps_per_iteration = int(mini_batch_steps_per_iteration)
    self._random_seed = random_seed
    self._kmeans_plus_plus_num_retries = kmeans_plus_plus_num_retries

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
      assert False, ('Unsupported distance metric passed to Kmeans %s' %
                     str(distance_metric))

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
      with ops.colocate_with(inp):
        # Computes Euclidean distance. Note the first and third terms are
        # broadcast additions.
        squared_distance = (math_ops.reduce_sum(
            math_ops.square(inp), 1, keep_dims=True) - 2 * math_ops.matmul(
                inp, clusters, transpose_b=True) + array_ops.transpose(
                    math_ops.reduce_sum(
                        math_ops.square(clusters), 1, keep_dims=True)))
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
      with ops.colocate_with(clusters):
        clusters = nn_impl.l2_normalize(clusters, dim=1)
    for inp in inputs:
      with ops.colocate_with(inp):
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
      # 2 * squared_euclidian_distance. We are using this fact and reusing the
      # nearest_neighbors op.
      # TODO(ands): Support COSINE distance in nearest_neighbors and remove
      # this.
      with ops.colocate_with(clusters):
        clusters = nn_impl.l2_normalize(clusters, dim=1)
    for inp, score in zip(inputs, scores):
      with ops.colocate_with(inp):
        (indices,
         distances) = gen_clustering_ops.nearest_neighbors(inp, clusters, 1)
        if self._distance_metric == COSINE_DISTANCE:
          distances *= 0.5
        output.append(
            (score, array_ops.squeeze(distances), array_ops.squeeze(indices)))
    return zip(*output)

  def _init_clusters_random(self):
    """Does random initialization of clusters.

    Returns:
      Tensor of randomly initialized clusters.
    """
    num_data = math_ops.add_n([array_ops.shape(inp)[0] for inp in self._inputs])
    # Note that for mini-batch k-means, we should ensure that the batch size of
    # data used during initialization is sufficiently large to avoid duplicated
    # clusters.
    with ops.control_dependencies(
        [check_ops.assert_less_equal(self._num_clusters, num_data)]):
      indices = random_ops.random_uniform(
          array_ops.reshape(self._num_clusters, [-1]),
          minval=0,
          maxval=math_ops.cast(num_data, dtypes.int64),
          seed=self._random_seed,
          dtype=dtypes.int64)
      clusters_init = embedding_lookup(
          self._inputs, indices, partition_strategy='div')
      return clusters_init

  def _clusters_l2_normalized(self):
    """Returns True if clusters centers are kept normalized."""
    return (self._distance_metric == COSINE_DISTANCE and
            (not self._use_mini_batch or
             self._mini_batch_steps_per_iteration > 1))

  def _initialize_clusters(self,
                           cluster_centers,
                           cluster_centers_initialized,
                           cluster_centers_updated):
    """Returns an op to initialize the cluster centers."""

    init = self._initial_clusters
    if init == RANDOM_INIT:
      clusters_init = self._init_clusters_random()
    elif init == KMEANS_PLUS_PLUS_INIT:
      # Points from only the first shard are used for initializing centers.
      # TODO(ands): Use all points.
      inp = self._inputs[0]
      if self._distance_metric == COSINE_DISTANCE:
        inp = nn_impl.l2_normalize(inp, dim=1)
      clusters_init = gen_clustering_ops.kmeans_plus_plus_initialization(
          inp, self._num_clusters, self._random_seed,
          self._kmeans_plus_plus_num_retries)
    elif callable(init):
      clusters_init = init(self._inputs, self._num_clusters)
    elif not isinstance(init, str):
      clusters_init = init
    else:
      assert False, 'Unsupported init passed to Kmeans %s' % str(init)
    if self._distance_metric == COSINE_DISTANCE and clusters_init is not None:
      clusters_init = nn_impl.l2_normalize(clusters_init, dim=1)

    with ops.colocate_with(cluster_centers_initialized):
      initialized = control_flow_ops.with_dependencies(
          [clusters_init],
          array_ops.identity(cluster_centers_initialized))
    with ops.colocate_with(cluster_centers):
      assign_centers = state_ops.assign(cluster_centers, clusters_init,
                                        validate_shape=False)
      if cluster_centers_updated != cluster_centers:
        assign_centers = control_flow_ops.group(
            assign_centers,
            state_ops.assign(cluster_centers_updated, clusters_init,
                             validate_shape=False))
      assign_centers = control_flow_ops.with_dependencies(
          [assign_centers],
          state_ops.assign(cluster_centers_initialized, True))
      return control_flow_ops.cond(initialized,
                                   control_flow_ops.no_op,
                                   lambda: assign_centers).op

  def _create_variables(self):
    """Creates variables.

    Returns:
    Tuple with following elements:
      cluster_centers: a Tensor for storing cluster centers
      cluster_centers_initialized: bool Variable indicating whether clusters
        are initialized.
      cluster_counts: a Tensor for storing counts of points assigned to this
        cluster. This is used by mini-batch training.
      cluster_centers_updated: Tensor representing copy of cluster centers that
        are updated every step.
      update_in_steps: numbers of steps left before we sync
        cluster_centers_updated back to cluster_centers.
    """
    init_value = array_ops.constant([], dtype=dtypes.float32)
    cluster_centers = variables.Variable(init_value,
                                         name='clusters',
                                         validate_shape=False)
    cluster_centers_initialized = variables.Variable(False,
                                                     dtype=dtypes.bool,
                                                     name='initialized')

    if self._use_mini_batch and self._mini_batch_steps_per_iteration > 1:
      # Copy of cluster centers actively updated each step according to
      # mini-batch update rule.
      cluster_centers_updated = variables.Variable(init_value,
                                                   name='clusters_updated',
                                                   validate_shape=False)
      # How many steps till we copy the updated clusters to cluster_centers.
      update_in_steps = variables.Variable(self._mini_batch_steps_per_iteration,
                                           dtype=dtypes.int64,
                                           name='update_in_steps')
      # Count of points assigned to cluster_centers_updated.
      cluster_counts = variables.Variable(array_ops.zeros([self._num_clusters],
                                                          dtype=dtypes.int64))
    else:
      cluster_centers_updated = cluster_centers
      update_in_steps = None
      cluster_counts = (variables.Variable(array_ops.ones([self._num_clusters],
                                                          dtype=dtypes.int64))
                        if self._use_mini_batch else None)
    return (cluster_centers,
            cluster_centers_initialized,
            cluster_counts,
            cluster_centers_updated,
            update_in_steps)

  @classmethod
  def _l2_normalize_data(cls, inputs):
    """Normalized the input data."""
    output = []
    for inp in inputs:
      with ops.colocate_with(inp):
        output.append(nn_impl.l2_normalize(inp, dim=1))
    return output

  def training_graph(self):
    """Generate a training graph for kmeans algorithm.

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
    inputs = self._inputs
    (cluster_centers_var,
     cluster_centers_initialized,
     total_counts,
     cluster_centers_updated,
     update_in_steps) = self._create_variables()
    init_op = self._initialize_clusters(cluster_centers_var,
                                        cluster_centers_initialized,
                                        cluster_centers_updated)
    cluster_centers = cluster_centers_var

    if self._distance_metric == COSINE_DISTANCE:
      inputs = self._l2_normalize_data(inputs)
      if not self._clusters_l2_normalized():
        cluster_centers = nn_impl.l2_normalize(cluster_centers, dim=1)

    all_scores, scores, cluster_idx = self._infer_graph(inputs, cluster_centers)
    if self._use_mini_batch:
      sync_updates_op = self._mini_batch_sync_updates_op(
          update_in_steps,
          cluster_centers_var, cluster_centers_updated,
          total_counts)
      assert sync_updates_op is not None
      with ops.control_dependencies([sync_updates_op]):
        training_op = self._mini_batch_training_op(
            inputs, cluster_idx, cluster_centers_updated, total_counts)
    else:
      assert cluster_centers == cluster_centers_var
      training_op = self._full_batch_training_op(inputs, cluster_idx,
                                                 cluster_centers_var)

    return (all_scores, cluster_idx, scores,
            cluster_centers_initialized, init_op, training_op)

  def _mini_batch_sync_updates_op(self, update_in_steps,
                                  cluster_centers_var, cluster_centers_updated,
                                  total_counts):
    if self._use_mini_batch and self._mini_batch_steps_per_iteration > 1:
      assert update_in_steps is not None
      with ops.colocate_with(update_in_steps):
        def _f():
          # Note that there is a race condition here, so we do a best effort
          # updates here. We reset update_in_steps first so that other workers
          # don't duplicate the updates. Also we update cluster_center_vars
          # before resetting total_counts to avoid large updates to
          # cluster_centers_updated based on partially updated
          # cluster_center_vars.
          with ops.control_dependencies([state_ops.assign(
              update_in_steps,
              self._mini_batch_steps_per_iteration - 1)]):
            with ops.colocate_with(cluster_centers_updated):
              if self._distance_metric == COSINE_DISTANCE:
                cluster_centers = nn_impl.l2_normalize(cluster_centers_updated,
                                                       dim=1)
              else:
                cluster_centers = cluster_centers_updated
            with ops.colocate_with(cluster_centers_var):
              with ops.control_dependencies([state_ops.assign(
                  cluster_centers_var,
                  cluster_centers)]):
                with ops.colocate_with(cluster_centers_var):
                  with ops.control_dependencies([
                      state_ops.assign(total_counts,
                                       array_ops.zeros_like(total_counts))]):
                    return array_ops.identity(update_in_steps)
        return control_flow_ops.cond(
            update_in_steps <= 0,
            _f,
            lambda: state_ops.assign_sub(update_in_steps, 1))
    else:
      return control_flow_ops.no_op()

  def _mini_batch_training_op(self, inputs, cluster_idx_list,
                              cluster_centers, total_counts):
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
      with ops.colocate_with(inp):
        assert total_counts is not None
        cluster_idx = array_ops.reshape(cluster_idx, [-1])
        # Dedupe the unique ids of cluster_centers being updated so that updates
        # can be locally aggregated.
        unique_ids, unique_idx = array_ops.unique(cluster_idx)
        num_unique_cluster_idx = array_ops.size(unique_ids)
        # Fetch the old values of counts and cluster_centers.
        with ops.colocate_with(total_counts):
          old_counts = array_ops.gather(total_counts, unique_ids)
        # TODO(agarwal): This colocation seems to run into problems. Fix it.
        # with ops.colocate_with(cluster_centers):
        old_cluster_centers = array_ops.gather(cluster_centers, unique_ids)
        # Locally aggregate the increment to counts.
        count_updates = math_ops.unsorted_segment_sum(
            array_ops.ones_like(
                unique_idx, dtype=total_counts.dtype),
            unique_idx,
            num_unique_cluster_idx)
        # Locally compute the sum of inputs mapped to each id.
        # For a cluster with old cluster value x, old count n, and with data
        # d_1,...d_k newly assigned to it, we recompute the new value as
        # x += (sum_i(d_i) - k * x) / (n + k).
        # Compute sum_i(d_i), see comment above.
        cluster_center_updates = math_ops.unsorted_segment_sum(
            inp, unique_idx, num_unique_cluster_idx)
        # Shape to enable broadcasting count_updates and learning_rate to inp.
        # It extends the shape with 1's to match the rank of inp.
        broadcast_shape = array_ops.concat(
            [
                array_ops.reshape(num_unique_cluster_idx, [1]), array_ops.ones(
                    array_ops.reshape(array_ops.rank(inp) - 1, [1]),
                    dtype=dtypes.int32)
            ],
            0)
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
      update_counts = state_ops.scatter_add(
          total_counts,
          unique_ids,
          count_updates)
      update_cluster_centers = state_ops.scatter_add(
          cluster_centers,
          unique_ids,
          cluster_center_updates)
      update_ops.extend([update_counts, update_cluster_centers])
    return control_flow_ops.group(*update_ops)

  def _full_batch_training_op(self, inputs, cluster_idx_list, cluster_centers):
    """Creates an op for training for full batch case.

    Args:
      inputs: list of input Tensors.
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
      with ops.colocate_with(inp):
        cluster_sums.append(
            math_ops.unsorted_segment_sum(inp, cluster_idx, self._num_clusters))
        cluster_counts.append(
            math_ops.unsorted_segment_sum(
                array_ops.reshape(
                    array_ops.ones(
                        array_ops.reshape(array_ops.shape(inp)[0], [-1])),
                    [-1, 1]), cluster_idx, self._num_clusters))
    with ops.colocate_with(cluster_centers):
      new_clusters_centers = math_ops.add_n(cluster_sums) / (math_ops.cast(
          math_ops.add_n(cluster_counts), cluster_sums[0].dtype) + epsilon)
      if self._clusters_l2_normalized():
        new_clusters_centers = nn_impl.l2_normalize(new_clusters_centers, dim=1)
    return state_ops.assign(cluster_centers, new_clusters_centers)
