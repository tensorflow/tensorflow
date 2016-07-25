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

"""Implementation of k-means clustering on top of learn (aka skflow) API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf

from tensorflow.contrib.factorization.python.ops import clustering_ops
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators._sklearn import TransformerMixin
from tensorflow.contrib.learn.python.learn.learn_io import data_feeder
from tensorflow.contrib.learn.python.learn.monitors import BaseMonitor
from tensorflow.contrib.learn.python.learn.utils import checkpoints
from tensorflow.python.ops.control_flow_ops import with_dependencies

SQUARED_EUCLIDEAN_DISTANCE = clustering_ops.SQUARED_EUCLIDEAN_DISTANCE
COSINE_DISTANCE = clustering_ops.COSINE_DISTANCE
RANDOM_INIT = clustering_ops.RANDOM_INIT
KMEANS_PLUS_PLUS_INIT = clustering_ops.KMEANS_PLUS_PLUS_INIT


# TODO(agarwal,ands): support sharded input.
# TODO(agarwal,ands): enable stopping criteria based on improvements to cost.
# TODO(agarwal,ands): support random restarts.
class KMeansClustering(estimator.Estimator,
                       TransformerMixin):
  """K-Means clustering."""
  SCORES = 'scores'
  CLUSTER_IDX = 'cluster_idx'
  CLUSTERS = 'clusters'
  ALL_SCORES = 'all_scores'

  def __init__(self,
               num_clusters,
               model_dir=None,
               initial_clusters=clustering_ops.RANDOM_INIT,
               distance_metric=clustering_ops.SQUARED_EUCLIDEAN_DISTANCE,
               random_seed=0,
               use_mini_batch=True,
               kmeans_plus_plus_num_retries=2,
               config=None):
    """Creates a model for running KMeans training and inference.

    Args:
      num_clusters: number of clusters to train.
      model_dir: the directory to save the model results and log files.
      initial_clusters: specifies how to initialize the clusters for training.
        See clustering_ops.kmeans for the possible values.
      distance_metric: the distance metric used for clustering.
        See clustering_ops.kmeans for the possible values.
      random_seed: Python integer. Seed for PRNG used to initialize centers.
      use_mini_batch: If true, use the mini-batch k-means algorithm. Else assume
        full batch.
      kmeans_plus_plus_num_retries: For each point that is sampled during
        kmeans++ initialization, this parameter specifies the number of
        additional points to draw from the current distribution before selecting
        the best. If a negative value is specified, a heuristic is used to
        sample O(log(num_to_sample)) additional points.
      config: See Estimator
    """
    super(KMeansClustering, self).__init__(
        model_dir=model_dir,
        config=config)
    self.kmeans_plus_plus_num_retries = kmeans_plus_plus_num_retries
    self._num_clusters = num_clusters
    self._training_initial_clusters = initial_clusters
    self._training_graph = None
    self._distance_metric = distance_metric
    self._use_mini_batch = use_mini_batch
    self._random_seed = random_seed
    self._initialized = False

# pylint: disable=protected-access
  class _StopWhenConverged(BaseMonitor):
    """Stops when the change in loss goes below a tolerance."""

    def __init__(self, tolerance):
      """Initializes a '_StopWhenConverged' monitor.

      Args:
        tolerance: A relative tolerance of change between iterations.
      """
      super(KMeansClustering._StopWhenConverged, self).__init__()
      self._tolerance = tolerance

    def begin(self, max_steps):
      super(KMeansClustering._StopWhenConverged, self).begin(max_steps)
      self._prev_loss = None

    def step_begin(self, step):
      super(KMeansClustering._StopWhenConverged, self).step_begin(step)
      return [self._estimator._loss]

    def step_end(self, step, output):
      super(KMeansClustering._StopWhenConverged, self).step_end(step, output)
      loss = output[self._estimator._loss]

      if self._prev_loss is None:
        self._prev_loss = loss
        return False

      relative_change = (abs(loss - self._prev_loss)
                         / (1 + abs(self._prev_loss)))
      self._prev_loss = loss
      return relative_change < self._tolerance
# pylint: enable=protected-access

  def fit(self, x, y=None, monitors=None, logdir=None, steps=None, batch_size=128,
          relative_tolerance=None):
    """Trains a k-means clustering on x.

    Note: See Estimator for logic for continuous training and graph
      construction across multiple calls to fit.

    Args:
      x: training input matrix of shape [n_samples, n_features].
      y: labels. Should be None.
      monitors: Monitor object to print training progress and invoke early
        stopping
      logdir: the directory to save the log file that can be used for optional
        visualization.
      steps: number of training steps. If not None, overrides the value passed
        in constructor.
      batch_size: mini-batch size to use. Requires `use_mini_batch=True`.
      relative_tolerance: A relative tolerance of change in the loss between
        iterations.  Stops learning if the loss changes less than this amount.
        Note that this may not work correctly if use_mini_batch=True.

    Returns:
      Returns self.
    """
    assert y is None
    if logdir is not None:
      self._model_dir = logdir
    self._data_feeder = data_feeder.setup_train_data_feeder(
        x, None, self._num_clusters, batch_size if self._use_mini_batch else None)
    if relative_tolerance is not None:
      if monitors is not None:
        monitors += [self._StopWhenConverged(relative_tolerance)]
      else:
        monitors = [self._StopWhenConverged(relative_tolerance)]
    # Make sure that we will eventually terminate.
    assert ((monitors is not None and len(monitors)) or (steps is not None)
            or (self.steps is not None))
    self._train_model(input_fn=self._data_feeder.input_builder,
                      feed_fn=self._data_feeder.get_feed_dict_fn(),
                      steps=steps,
                      monitors=monitors,
                      init_feed_fn=self._data_feeder.get_feed_dict_fn())
    return self

  def predict(self, x, batch_size=None):
    """Predict cluster id for each element in x.

    Args:
      x: 2-D matrix or iterator.
      batch_size: size to use for batching up x for querying the model.

    Returns:
      Array with same number of rows as x, containing cluster ids.
    """
    return super(KMeansClustering, self).predict(
        x=x, batch_size=batch_size)[KMeansClustering.CLUSTER_IDX]

  def score(self, x, batch_size=None):
    """Predict total sum of distances to nearest clusters.

    Note that this function is different from the corresponding one in sklearn
    which returns the negative of the sum of distances.

    Args:
      x: 2-D matrix or iterator.
      batch_size: size to use for batching up x for querying the model.

    Returns:
      Total sum of distances to nearest clusters.
    """
    return np.sum(
        self.evaluate(x=x, batch_size=batch_size)[KMeansClustering.SCORES])

  def transform(self, x, batch_size=None):
    """Transforms each element in x to distances to cluster centers.

    Note that this function is different from the corresponding one in sklearn.
    For SQUARED_EUCLIDEAN distance metric, sklearn transform returns the
    EUCLIDEAN distance, while this function returns the SQUARED_EUCLIDEAN
    distance.

    Args:
      x: 2-D matrix or iterator.
      batch_size: size to use for batching up x for querying the model.

    Returns:
      Array with same number of rows as x, and num_clusters columns, containing
      distances to the cluster centers.
    """
    return super(KMeansClustering, self).predict(
        x=x, batch_size=batch_size)[KMeansClustering.ALL_SCORES]

  def clusters(self):
    """Returns cluster centers."""
    return checkpoints.load_variable(self.model_dir, self.CLUSTERS)

  def _get_train_ops(self, features, _):
    (_,
     _,
     losses,
     training_op) = clustering_ops.KMeans(
         features,
         self._num_clusters,
         self._training_initial_clusters,
         self._distance_metric,
         self._use_mini_batch,
         random_seed=self._random_seed,
         kmeans_plus_plus_num_retries=self.kmeans_plus_plus_num_retries
     ).training_graph()
    incr_step = tf.assign_add(tf.contrib.framework.get_global_step(), 1)
    self._loss = tf.reduce_sum(losses)
    training_op = with_dependencies([training_op, incr_step], self._loss)
    return training_op, self._loss

  def _get_predict_ops(self, features):
    (all_scores,
     model_predictions,
     _,
     _) = clustering_ops.KMeans(
         features,
         self._num_clusters,
         self._training_initial_clusters,
         self._distance_metric,
         self._use_mini_batch,
         random_seed=self._random_seed,
         kmeans_plus_plus_num_retries=self.kmeans_plus_plus_num_retries
     ).training_graph()
    return {
        KMeansClustering.ALL_SCORES: all_scores[0],
        KMeansClustering.CLUSTER_IDX: model_predictions[0]
    }

  def _get_eval_ops(self, features, _, unused_metrics):
    (_,
     _,
     losses,
     _) = clustering_ops.KMeans(
         features,
         self._num_clusters,
         self._training_initial_clusters,
         self._distance_metric,
         self._use_mini_batch,
         random_seed=self._random_seed,
         kmeans_plus_plus_num_retries=self.kmeans_plus_plus_num_retries
     ).training_graph()
    return {
        KMeansClustering.SCORES: tf.reduce_sum(losses),
    }
