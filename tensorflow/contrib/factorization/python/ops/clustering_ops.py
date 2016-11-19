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

import tensorflow as tf

from tensorflow.contrib.factorization.python.ops import gen_clustering_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.contrib.factorization.python.ops.gen_clustering_ops import *
# pylint: enable=wildcard-import
from tensorflow.contrib.util import loader
from tensorflow.python.framework import ops
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
               random_seed=0,
               kmeans_plus_plus_num_retries=2):
    """Creates an object for generating KMeans clustering graph.

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
    self._initial_clusters = initial_clusters
    assert distance_metric in [SQUARED_EUCLIDEAN_DISTANCE, COSINE_DISTANCE]
    self._distance_metric = distance_metric
    self._use_mini_batch = use_mini_batch
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
      return cls._compute_cosine_distance(inputs, clusters,
                                          inputs_normalized=True)
    else:
      assert False, ('Unsupported distance metric passed to Kmeans %s'
                     % str(distance_metric))

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
        squared_distance = (tf.reduce_sum(tf.square(inp), 1, keep_dims=True) -
                            2 * tf.matmul(inp, clusters, transpose_b=True) +
                            tf.transpose(tf.reduce_sum(tf.square(clusters),
                                                       1,
                                                       keep_dims=True)))
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
        clusters = tf.nn.l2_normalize(clusters, dim=1)
    for inp in inputs:
      with ops.colocate_with(inp):
        if not inputs_normalized:
          inp = tf.nn.l2_normalize(inp, dim=1)
        output.append(1 - tf.matmul(inp, clusters, transpose_b=True))
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
        clusters = tf.nn.l2_normalize(clusters, dim=1)
    for inp, score in zip(inputs, scores):
      with ops.colocate_with(inp):
        (indices,
         distances) = gen_clustering_ops.nearest_neighbors(inp, clusters, 1)
        if self._distance_metric == COSINE_DISTANCE:
          distances *= 0.5
        output.append((score, tf.squeeze(distances), tf.squeeze(indices)))
    return zip(*output)

  def _init_clusters_random(self):
    """Does random initialization of clusters.

    Returns:
      Tensor of randomly initialized clusters.
    """
    num_data = tf.add_n([tf.shape(inp)[0] for inp in self._inputs])
    # Note that for mini-batch k-means, we should ensure that the batch size of
    # data used during initialization is sufficiently large to avoid duplicated
    # clusters.
    with tf.control_dependencies(
        [tf.assert_less_equal(self._num_clusters, num_data)]):
      indices = tf.random_uniform(tf.reshape(self._num_clusters, [-1]),
                                  minval=0,
                                  maxval=tf.cast(num_data, tf.int64),
                                  seed=self._random_seed,
                                  dtype=tf.int64)
      clusters_init = embedding_lookup(self._inputs, indices,
                                       partition_strategy='div')
      return clusters_init

  def _clusters_l2_normalized(self):
    """Returns True if clusters centers are kept normalized."""
    return self._distance_metric == COSINE_DISTANCE and not self._use_mini_batch

  def _init_clusters(self):
    """Initialization of clusters.

    Returns:
    Tuple with following elements:
      cluster_centers: a Tensor for storing cluster centers
      cluster_counts: a Tensor for storing counts of points assigned to this
        cluster. This is used by mini-batch training.
    """
    init = self._initial_clusters
    if init == RANDOM_INIT:
      clusters_init = self._init_clusters_random()
    elif init == KMEANS_PLUS_PLUS_INIT:
      # Points from only the first shard are used for initializing centers.
      # TODO(ands): Use all points.
      clusters_init = gen_clustering_ops.kmeans_plus_plus_initialization(
          self._inputs[0], self._num_clusters, self._random_seed,
          self._kmeans_plus_plus_num_retries)
    elif callable(init):
      clusters_init = init(self._inputs, self._num_clusters)
    elif not isinstance(init, str):
      clusters_init = init
    else:
      assert False, 'Unsupported init passed to Kmeans %s' % str(init)
    if self._distance_metric == COSINE_DISTANCE and clusters_init is not None:
      clusters_init = tf.nn.l2_normalize(clusters_init, dim=1)
    clusters_init = clusters_init if clusters_init is not None else []
    cluster_centers = tf.Variable(clusters_init,
                                  name='clusters',
                                  validate_shape=False)
    cluster_counts = (tf.Variable(tf.zeros([self._num_clusters],
                                           dtype=tf.int64))
                      if self._use_mini_batch else None)
    return cluster_centers, cluster_counts

  @classmethod
  def _l2_normalize_data(cls, inputs):
    """Normalized the input data."""
    output = []
    for inp in inputs:
      with ops.colocate_with(inp):
        output.append(tf.nn.l2_normalize(inp, dim=1))
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
      training_op: an op that runs an iteration of training.
    """
    # Implementation of kmeans.
    inputs = self._inputs
    cluster_centers_var, total_counts = self._init_clusters()
    cluster_centers = cluster_centers_var

    if self._distance_metric == COSINE_DISTANCE:
      inputs = self._l2_normalize_data(inputs)
      if not self._clusters_l2_normalized():
        cluster_centers = tf.nn.l2_normalize(cluster_centers, dim=1)

    all_scores, scores, cluster_idx = self._infer_graph(inputs, cluster_centers)
    if self._use_mini_batch:
      training_op = self._mini_batch_training_op(
          inputs, cluster_idx, cluster_centers, cluster_centers_var,
          total_counts)
    else:
      assert cluster_centers == cluster_centers_var
      training_op = self._full_batch_training_op(inputs, cluster_idx,
                                                 cluster_centers_var)
    return all_scores, cluster_idx, scores, training_op

  def _mini_batch_training_op(self, inputs, cluster_idx_list,
                              cluster_centers, cluster_centers_var,
                              total_counts):
    """Creates an op for training for mini batch case.

    Args:
      inputs: list of input Tensors.
      cluster_idx_list: A vector (or list of vectors). Each element in the
        vector corresponds to an input row in 'inp' and specifies the cluster id
        corresponding to the input.
      cluster_centers: Tensor of cluster centers, possibly normalized.
      cluster_centers_var: Tensor Ref of cluster centers.
      total_counts: Tensor Ref of cluster counts.

    Returns:
      An op for doing an update of mini-batch k-means.
    """
    update_ops = []
    for inp, cluster_idx in zip(inputs, cluster_idx_list):
      with ops.colocate_with(inp):
        assert total_counts is not None
        cluster_idx = tf.reshape(cluster_idx, [-1])
        # Dedupe the unique ids of cluster_centers being updated so that updates
        # can be locally aggregated.
        unique_ids, unique_idx = tf.unique(cluster_idx)
        num_unique_cluster_idx = tf.size(unique_ids)
        # Fetch the old values of counts and cluster_centers.
        with ops.colocate_with(total_counts):
          old_counts = tf.gather(total_counts, unique_ids)
        with ops.colocate_with(cluster_centers):
          old_cluster_centers = tf.gather(cluster_centers, unique_ids)
        # Locally aggregate the increment to counts.
        count_updates = tf.unsorted_segment_sum(
            tf.ones_like(unique_idx, dtype=total_counts.dtype),
            unique_idx,
            num_unique_cluster_idx)
        # Locally compute the sum of inputs mapped to each id.
        # For a cluster with old cluster value x, old count n, and with data
        # d_1,...d_k newly assigned to it, we recompute the new value as
        # x += (sum_i(d_i) - k * x) / (n + k).
        # Compute sum_i(d_i), see comment above.
        cluster_center_updates = tf.unsorted_segment_sum(
            inp,
            unique_idx,
            num_unique_cluster_idx)
        # Shape to enable broadcasting count_updates and learning_rate to inp.
        # It extends the shape with 1's to match the rank of inp.
        broadcast_shape = tf.concat(
            0,
            [tf.reshape(num_unique_cluster_idx, [1]),
             tf.ones(tf.reshape(tf.rank(inp) - 1, [1]), dtype=tf.int32)])
        # Subtract k * x, see comment above.
        cluster_center_updates -= tf.cast(
            tf.reshape(count_updates, broadcast_shape),
            inp.dtype) * old_cluster_centers
        learning_rate = tf.reciprocal(tf.cast(old_counts + count_updates,
                                              inp.dtype))
        learning_rate = tf.reshape(learning_rate, broadcast_shape)
        # scale by 1 / (n + k), see comment above.
        cluster_center_updates *= learning_rate
        # Apply the updates.
      update_counts = tf.scatter_add(
          total_counts,
          unique_ids,
          count_updates)
      update_cluster_centers = tf.scatter_add(
          cluster_centers_var,
          unique_ids,
          cluster_center_updates)
      update_ops.extend([update_counts, update_cluster_centers])
    return tf.group(*update_ops)

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
    epsilon = tf.constant(1e-6, dtype=inputs[0].dtype)
    for inp, cluster_idx in zip(inputs, cluster_idx_list):
      with ops.colocate_with(inp):
        cluster_sums.append(tf.unsorted_segment_sum(inp,
                                                    cluster_idx,
                                                    self._num_clusters))
        cluster_counts.append(tf.unsorted_segment_sum(
            tf.reshape(tf.ones(tf.reshape(tf.shape(inp)[0], [-1])), [-1, 1]),
            cluster_idx,
            self._num_clusters))
    with ops.colocate_with(cluster_centers):
      new_clusters_centers = tf.add_n(cluster_sums) / (
          tf.cast(tf.add_n(cluster_counts), cluster_sums[0].dtype) + epsilon)
      if self._clusters_l2_normalized():
        new_clusters_centers = tf.nn.l2_normalize(new_clusters_centers, dim=1)
    return tf.assign(cluster_centers, new_clusters_centers)
