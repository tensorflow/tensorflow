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
"""A canned Estimator for k-means clustering."""

# TODO(ccolby): Move clustering_ops.py into this file and streamline the code.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from tensorflow.contrib.factorization.python.ops import clustering_ops
from tensorflow.python.estimator import estimator
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator.export import export_output
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.summary import summary
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util


class _LossRelativeChangeHook(session_run_hook.SessionRunHook):
  """Stops when the change in loss goes below a tolerance."""

  def __init__(self, loss_tensor, tolerance):
    """Creates a _LossRelativeChangeHook.

    Args:
      loss_tensor: A scalar tensor of the loss value.
      tolerance: A relative tolerance of loss change between iterations.
    """
    self._loss_tensor = loss_tensor
    self._tolerance = tolerance
    self._prev_loss = None

  def before_run(self, run_context):
    del run_context  # unused
    return session_run_hook.SessionRunArgs(self._loss_tensor)

  def after_run(self, run_context, run_values):
    loss = run_values.results
    assert loss is not None
    if self._prev_loss:
      relative_change = (
          abs(loss - self._prev_loss) / (1 + abs(self._prev_loss)))
      if relative_change < self._tolerance:
        run_context.request_stop()
    self._prev_loss = loss


class _InitializeClustersHook(session_run_hook.SessionRunHook):
  """Initializes the cluster centers.

  The chief repeatedly invokes an initialization op until all cluster centers
  are initialized. The workers wait for the initialization phase to complete.
  """

  def __init__(self, init_op, is_initialized_var, is_chief):
    """Creates an _InitializeClustersHook.

    Args:
      init_op: An op that, when run, will choose some initial cluster centers.
          This op may need to be run multiple times to choose all the centers.
      is_initialized_var: A boolean variable reporting whether all initial
          centers have been chosen.
      is_chief: A boolean specifying whether this task is the chief.
    """
    self._init_op = init_op
    self._is_initialized_var = is_initialized_var
    self._is_chief = is_chief

  def after_create_session(self, session, coord):
    del coord  # unused
    assert self._init_op.graph is ops.get_default_graph()
    assert self._is_initialized_var.graph is self._init_op.graph
    while True:
      try:
        if session.run(self._is_initialized_var):
          break
        elif self._is_chief:
          session.run(self._init_op)
        else:
          time.sleep(1)
      except RuntimeError as e:
        logging.info(e)


def _parse_tensor_or_dict(features):
  """Helper function to convert the input points into a usable format.

  Args:
    features: The input points.

  Returns:
    If `features` is a dict of `k` features, each of which is a vector of `n`
    scalars, the return value is a Tensor of shape `(n, k)` representing `n`
    input points, where the items in the `k` dimension are sorted
    lexicographically by `features` key. If `features` is not a dict, it is
    returned unmodified.
  """
  if isinstance(features, dict):
    keys = sorted(features.keys())
    with ops.colocate_with(features[keys[0]]):
      features = array_ops.concat([features[k] for k in keys], axis=1)
  return features


class _ModelFn(object):
  """Model function for the estimator."""

  def __init__(self, num_clusters, initial_clusters, distance_metric,
               random_seed, use_mini_batch, mini_batch_steps_per_iteration,
               kmeans_plus_plus_num_retries, relative_tolerance):
    self._num_clusters = num_clusters
    self._initial_clusters = initial_clusters
    self._distance_metric = distance_metric
    self._random_seed = random_seed
    self._use_mini_batch = use_mini_batch
    self._mini_batch_steps_per_iteration = mini_batch_steps_per_iteration
    self._kmeans_plus_plus_num_retries = kmeans_plus_plus_num_retries
    self._relative_tolerance = relative_tolerance

  def model_fn(self, features, mode, config):
    """Model function for the estimator.

    Note that this does not take a `labels` arg. This works, but `input_fn` must
    return either `features` or, equivalently, `(features, None)`.

    Args:
      features: The input points. See @{tf.estimator.Estimator}.
      mode: See @{tf.estimator.Estimator}.
      config: See @{tf.estimator.Estimator}.

    Returns:
      A @{tf.estimator.EstimatorSpec} (see @{tf.estimator.Estimator}) specifying
      this behavior:
        * `train_op`: Execute one mini-batch or full-batch run of Lloyd's
             algorithm.
        * `loss`: The sum of the squared distances from each input point to its
             closest center.
        * `eval_metric_ops`: Maps `SCORE` to `loss`.
        * `predictions`: Maps `ALL_DISTANCES` to the distance from each input
             point to each cluster center; maps `CLUSTER_INDEX` to the index of
             the closest cluster center for each input point.
    """
    # input_points is a single Tensor. Therefore, the sharding functionality
    # in clustering_ops is unused, and some of the values below are lists of a
    # single item.
    input_points = _parse_tensor_or_dict(features)

    # Let N = the number of input_points.
    # all_distances: A list of one matrix of shape (N, num_clusters). Each value
    #   is the distance from an input point to a cluster center.
    # model_predictions: A list of one vector of shape (N). Each value is the
    #   cluster id of an input point.
    # losses: Similar to cluster_idx but provides the distance to the cluster
    #   center.
    # is_initialized: scalar indicating whether the initial cluster centers
    #   have been chosen; see init_op.
    # cluster_centers_var: a Variable containing the cluster centers.
    # init_op: an op to choose the initial cluster centers. A single worker
    #   repeatedly executes init_op until is_initialized becomes True.
    # training_op: an op that runs an iteration of training, either an entire
    #   Lloyd iteration or a mini-batch of a Lloyd iteration. Multiple workers
    #   may execute this op, but only after is_initialized becomes True.
    (all_distances, model_predictions, losses, is_initialized, init_op,
     training_op) = clustering_ops.KMeans(
         inputs=input_points,
         num_clusters=self._num_clusters,
         initial_clusters=self._initial_clusters,
         distance_metric=self._distance_metric,
         use_mini_batch=self._use_mini_batch,
         mini_batch_steps_per_iteration=self._mini_batch_steps_per_iteration,
         random_seed=self._random_seed,
         kmeans_plus_plus_num_retries=self._kmeans_plus_plus_num_retries
     ).training_graph()

    loss = math_ops.reduce_sum(losses)
    summary.scalar('loss/raw', loss)

    incr_step = state_ops.assign_add(training_util.get_global_step(), 1)
    training_op = control_flow_ops.with_dependencies([training_op, incr_step],
                                                     loss)

    training_hooks = [
        _InitializeClustersHook(init_op, is_initialized, config.is_chief)
    ]
    if self._relative_tolerance is not None:
      training_hooks.append(
          _LossRelativeChangeHook(loss, self._relative_tolerance))

    export_outputs = {
        KMeansClustering.ALL_DISTANCES:
            export_output.PredictOutput(all_distances[0]),
        KMeansClustering.CLUSTER_INDEX:
            export_output.PredictOutput(model_predictions[0]),
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            export_output.PredictOutput(model_predictions[0])
    }

    return model_fn_lib.EstimatorSpec(
        mode=mode,
        predictions={
            KMeansClustering.ALL_DISTANCES: all_distances[0],
            KMeansClustering.CLUSTER_INDEX: model_predictions[0],
        },
        loss=loss,
        train_op=training_op,
        eval_metric_ops={KMeansClustering.SCORE: metrics.mean(loss)},
        training_hooks=training_hooks,
        export_outputs=export_outputs)


# TODO(agarwal,ands): support sharded input.
class KMeansClustering(estimator.Estimator):
  """An Estimator for K-Means clustering.

  Example:
  ```
  import numpy as np
  import tensorflow as tf

  num_points = 100
  dimensions = 2
  points = np.random.uniform(0, 1000, [num_points, dimensions])

  def input_fn():
    return tf.train.limit_epochs(
        tf.convert_to_tensor(points, dtype=tf.float32), num_epochs=1)

  num_clusters = 5
  kmeans = tf.contrib.factorization.KMeansClustering(
      num_clusters=num_clusters, use_mini_batch=False)

  # train
  num_iterations = 10
  previous_centers = None
  for _ in xrange(num_iterations):
    kmeans.train(input_fn)
    cluster_centers = kmeans.cluster_centers()
    if previous_centers is not None:
      print 'delta:', cluster_centers - previous_centers
    previous_centers = cluster_centers
    print 'score:', kmeans.score(input_fn)
  print 'cluster centers:', cluster_centers

  # map the input points to their clusters
  cluster_indices = list(kmeans.predict_cluster_index(input_fn))
  for i, point in enumerate(points):
    cluster_index = cluster_indices[i]
    center = cluster_centers[cluster_index]
    print 'point:', point, 'is in cluster', cluster_index, 'centered at', center
  ```

  The `SavedModel` saved by the `export_savedmodel` method does not include the
  cluster centers. However, the cluster centers may be retrieved by the
  latest checkpoint saved during training. Specifically,
  ```
  kmeans.cluster_centers()
  ```
  is equivalent to
  ```
  tf.train.load_variable(
      kmeans.model_dir, KMeansClustering.CLUSTER_CENTERS_VAR_NAME)
  ```
  """

  # Valid values for the distance_metric constructor argument.
  SQUARED_EUCLIDEAN_DISTANCE = clustering_ops.SQUARED_EUCLIDEAN_DISTANCE
  COSINE_DISTANCE = clustering_ops.COSINE_DISTANCE

  # Values for initial_clusters constructor argument.
  RANDOM_INIT = clustering_ops.RANDOM_INIT
  KMEANS_PLUS_PLUS_INIT = clustering_ops.KMEANS_PLUS_PLUS_INIT

  # Metric returned by evaluate(): The sum of the squared distances from each
  # input point to its closest center.
  SCORE = 'score'

  # Keys returned by predict().
  # ALL_DISTANCES: The distance from each input  point to each cluster center.
  # CLUSTER_INDEX: The index of the closest cluster center for each input point.
  CLUSTER_INDEX = 'cluster_index'
  ALL_DISTANCES = 'all_distances'

  # Variable name used by cluster_centers().
  CLUSTER_CENTERS_VAR_NAME = clustering_ops.CLUSTERS_VAR_NAME

  def __init__(self,
               num_clusters,
               model_dir=None,
               initial_clusters=RANDOM_INIT,
               distance_metric=SQUARED_EUCLIDEAN_DISTANCE,
               random_seed=0,
               use_mini_batch=True,
               mini_batch_steps_per_iteration=1,
               kmeans_plus_plus_num_retries=2,
               relative_tolerance=None,
               config=None):
    """Creates an Estimator for running KMeans training and inference.

    This Estimator implements the following variants of the K-means algorithm:

    If `use_mini_batch` is False, it runs standard full batch K-means. Each
    training step runs a single iteration of K-Means and must process the full
    input at once. To run in this mode, the `input_fn` passed to `train` must
    return the entire input dataset.

    If `use_mini_batch` is True, it runs a generalization of the mini-batch
    K-means algorithm. It runs multiple iterations, where each iteration is
    composed of `mini_batch_steps_per_iteration` steps. Each training step
    accumulates the contribution from one mini-batch into temporary storage.
    Every `mini_batch_steps_per_iteration` steps, the cluster centers are
    updated and the temporary storage cleared for the next iteration. Note
    that:
      * If `mini_batch_steps_per_iteration=1`, the algorithm reduces to the
        standard K-means mini-batch algorithm.
      * If `mini_batch_steps_per_iteration = num_inputs / batch_size`, the
        algorithm becomes an asynchronous version of the full-batch algorithm.
        However, there is no guarantee by this implementation that each input
        is seen exactly once per iteration. Also, different updates are applied
        asynchronously without locking. So this asynchronous version may not
        behave exactly like a full-batch version.

    Args:
      num_clusters: An integer tensor specifying the number of clusters. This
        argument is ignored if `initial_clusters` is a tensor or numpy array.
      model_dir: The directory to save the model results and log files.
      initial_clusters: Specifies how the initial cluster centers are chosen.
        One of the following:
        * a tensor or numpy array with the initial cluster centers.
        * a callable `f(inputs, k)` that selects and returns up to `k` centers
              from an input batch. `f` is free to return any number of centers
              from `0` to `k`. It will be invoked on successive input batches
              as necessary until all `num_clusters` centers are chosen.
        * `KMeansClustering.RANDOM_INIT`: Choose centers randomly from an input
              batch. If the batch size is less than `num_clusters` then the
              entire batch is chosen to be initial cluster centers and the
              remaining centers are chosen from successive input batches.
        * `KMeansClustering.KMEANS_PLUS_PLUS_INIT`: Use kmeans++ to choose
              centers from the first input batch. If the batch size is less
              than `num_clusters`, a TensorFlow runtime error occurs.
      distance_metric: The distance metric used for clustering. One of:
        * `KMeansClustering.SQUARED_EUCLIDEAN_DISTANCE`: Euclidean distance
             between vectors `u` and `v` is defined as `||u - v||_2` which is
             the square root of the sum of the absolute squares of the elements'
             difference.
        * `KMeansClustering.COSINE_DISTANCE`: Cosine distance between vectors
             `u` and `v` is defined as `1 - (u . v) / (||u||_2 ||v||_2)`.
      random_seed: Python integer. Seed for PRNG used to initialize centers.
      use_mini_batch: A boolean specifying whether to use the mini-batch k-means
        algorithm. See explanation above.
      mini_batch_steps_per_iteration: The number of steps after which the
        updated cluster centers are synced back to a master copy. Used only if
        `use_mini_batch=True`. See explanation above.
      kmeans_plus_plus_num_retries: For each point that is sampled during
        kmeans++ initialization, this parameter specifies the number of
        additional points to draw from the current distribution before selecting
        the best. If a negative value is specified, a heuristic is used to
        sample `O(log(num_to_sample))` additional points. Used only if
        `initial_clusters=KMeansClustering.KMEANS_PLUS_PLUS_INIT`.
      relative_tolerance: A relative tolerance of change in the loss between
        iterations. Stops learning if the loss changes less than this amount.
        This may not work correctly if `use_mini_batch=True`.
      config: See @{tf.estimator.Estimator}.

    Raises:
      ValueError: An invalid argument was passed to `initial_clusters` or
        `distance_metric`.
    """
    if isinstance(initial_clusters, str) and initial_clusters not in [
        KMeansClustering.RANDOM_INIT, KMeansClustering.KMEANS_PLUS_PLUS_INIT
    ]:
      raise ValueError(
          "Unsupported initialization algorithm '%s'" % initial_clusters)
    if distance_metric not in [
        KMeansClustering.SQUARED_EUCLIDEAN_DISTANCE,
        KMeansClustering.COSINE_DISTANCE
    ]:
      raise ValueError("Unsupported distance metric '%s'" % distance_metric)
    super(KMeansClustering, self).__init__(
        model_fn=_ModelFn(
            num_clusters, initial_clusters, distance_metric, random_seed,
            use_mini_batch, mini_batch_steps_per_iteration,
            kmeans_plus_plus_num_retries, relative_tolerance).model_fn,
        model_dir=model_dir,
        config=config)

  def _predict_one_key(self, input_fn, predict_key):
    for result in self.predict(input_fn=input_fn, predict_keys=[predict_key]):
      yield result[predict_key]

  def predict_cluster_index(self, input_fn):
    """Finds the index of the closest cluster center to each input point.

    Args:
      input_fn: Input points. See @{tf.estimator.Estimator.predict}.

    Yields:
      The index of the closest cluster center for each input point.
    """
    for index in self._predict_one_key(input_fn,
                                       KMeansClustering.CLUSTER_INDEX):
      yield index

  def score(self, input_fn):
    """Returns the sum of squared distances to nearest clusters.

    Note that this function is different from the corresponding one in sklearn
    which returns the negative sum.

    Args:
      input_fn: Input points. See @{tf.estimator.Estimator.evaluate}. Only one
          batch is retrieved.

    Returns:
      The sum of the squared distance from each point in the first batch of
      inputs to its nearest cluster center.
    """
    return self.evaluate(input_fn=input_fn, steps=1)[KMeansClustering.SCORE]

  def transform(self, input_fn):
    """Transforms each input point to its distances to all cluster centers.

    Note that if `distance_metric=KMeansClustering.SQUARED_EUCLIDEAN_DISTANCE`,
    this
    function returns the squared Euclidean distance while the corresponding
    sklearn function returns the Euclidean distance.

    Args:
      input_fn: Input points. See @{tf.estimator.Estimator.predict}.

    Yields:
      The distances from each input point to each cluster center.
    """
    for distances in self._predict_one_key(input_fn,
                                           KMeansClustering.ALL_DISTANCES):
      yield distances

  def cluster_centers(self):
    """Returns the cluster centers."""
    return self.get_variable_value(KMeansClustering.CLUSTER_CENTERS_VAR_NAME)
