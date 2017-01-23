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
"""Implementation of Gaussian mixture model (GMM) clustering.

This goes on top of skflow API.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np

from tensorflow.contrib import framework
from tensorflow.contrib.factorization.python.ops import gmm_ops
from tensorflow.contrib.framework.python.framework import checkpoint_utils
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.learn.python.learn import graph_actions
from tensorflow.contrib.learn.python.learn import monitors as monitor_lib
from tensorflow.contrib.learn.python.learn.estimators import estimator as estimator_lib
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib.learn.python.learn.estimators._sklearn import TransformerMixin
from tensorflow.contrib.learn.python.learn.learn_io import data_feeder
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed as random_seed_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops.control_flow_ops import with_dependencies
from tensorflow.python.platform import tf_logging as logging


def _streaming_sum(scalar_tensor):
  """Create a sum metric and update op."""
  sum_metric = framework.local_variable(constant_op.constant(0.0))
  sum_update = sum_metric.assign_add(scalar_tensor)
  return sum_metric, sum_update


class GMM(estimator_lib.Estimator, TransformerMixin):
  """GMM clustering."""
  SCORES = 'scores'
  ASSIGNMENTS = 'assignments'
  ALL_SCORES = 'all_scores'

  def __init__(self,
               num_clusters,
               model_dir=None,
               random_seed=0,
               params='wmc',
               initial_clusters='random',
               covariance_type='full',
               batch_size=128,
               steps=10,
               continue_training=False,
               config=None,
               verbose=1):
    """Creates a model for running GMM training and inference.

    Args:
      num_clusters: number of clusters to train.
      model_dir: the directory to save the model results and log files.
      random_seed: Python integer. Seed for PRNG used to initialize centers.
      params: Controls which parameters are updated in the training process.
        Can contain any combination of "w" for weights, "m" for means,
        and "c" for covars.
      initial_clusters: specifies how to initialize the clusters for training.
        See gmm_ops.gmm for the possible values.
      covariance_type: one of "full", "diag".
      batch_size: See Estimator
      steps: See Estimator
      continue_training: See Estimator
      config: See Estimator
      verbose: See Estimator
    """
    super(GMM, self).__init__(model_dir=model_dir, config=config)
    self.batch_size = batch_size
    self.steps = steps
    self.continue_training = continue_training
    self.verbose = verbose
    self._num_clusters = num_clusters
    self._params = params
    self._training_initial_clusters = initial_clusters
    self._covariance_type = covariance_type
    self._training_graph = None
    self._random_seed = random_seed

  def fit(self, x, y=None, monitors=None, logdir=None, steps=None):
    """Trains a GMM clustering on x.

    Note: See Estimator for logic for continuous training and graph
      construction across multiple calls to fit.

    Args:
      x: training input matrix of shape [n_samples, n_features].
      y: labels. Should be None.
      monitors: List of `Monitor` objects to print training progress and
        invoke early stopping.
      logdir: the directory to save the log file that can be used for optional
        visualization.
      steps: number of training steps. If not None, overrides the value passed
        in constructor.

    Returns:
      Returns self.
    """
    if logdir is not None:
      self._model_dir = logdir
    self._data_feeder = data_feeder.setup_train_data_feeder(x, None,
                                                            self._num_clusters,
                                                            self.batch_size)
    _legacy_train_model(  # pylint: disable=protected-access
        self,
        input_fn=self._data_feeder.input_builder,
        feed_fn=self._data_feeder.get_feed_dict_fn(),
        steps=steps or self.steps,
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
    return np.array([
        prediction[GMM.ASSIGNMENTS]
        for prediction in super(GMM, self).predict(
            x=x, batch_size=batch_size, as_iterable=True)
    ])

  def score(self, x, batch_size=None):
    """Predict total sum of distances to nearest clusters.

    Args:
      x: 2-D matrix or iterator.
      batch_size: size to use for batching up x for querying the model.

    Returns:
      Total score.
    """
    return np.sum(self.evaluate(x=x, batch_size=batch_size)[GMM.SCORES])

  def transform(self, x, batch_size=None):
    """Transforms each element in x to distances to cluster centers.

    Args:
      x: 2-D matrix or iterator.
      batch_size: size to use for batching up x for querying the model.

    Returns:
      Array with same number of rows as x, and num_clusters columns, containing
      distances to the cluster centers.
    """
    return np.array([
        prediction[GMM.ALL_SCORES]
        for prediction in super(GMM, self).predict(
            x=x, batch_size=batch_size, as_iterable=True)
    ])

  def clusters(self):
    """Returns cluster centers."""
    clusters = checkpoint_utils.load_variable(
        self.model_dir, gmm_ops.GmmAlgorithm.CLUSTERS_VARIABLE)
    return np.squeeze(clusters, 1)

  def covariances(self):
    """Returns the covariances."""
    return checkpoint_utils.load_variable(
        self.model_dir, gmm_ops.GmmAlgorithm.CLUSTERS_COVS_VARIABLE)

  def _parse_tensor_or_dict(self, features):
    if isinstance(features, dict):
      return array_ops.concat([features[k] for k in sorted(features.keys())], 1)
    return features

  def _get_train_ops(self, features, _):
    (_, _, losses, training_op) = gmm_ops.gmm(
        self._parse_tensor_or_dict(features), self._training_initial_clusters,
        self._num_clusters, self._random_seed, self._covariance_type,
        self._params)
    incr_step = state_ops.assign_add(variables.get_global_step(), 1)
    loss = math_ops.reduce_sum(losses)
    training_op = with_dependencies([training_op, incr_step], loss)
    return training_op, loss

  def _get_predict_ops(self, features):
    (all_scores, model_predictions, _, _) = gmm_ops.gmm(
        self._parse_tensor_or_dict(features), self._training_initial_clusters,
        self._num_clusters, self._random_seed, self._covariance_type,
        self._params)
    return {
        GMM.ALL_SCORES: all_scores[0],
        GMM.ASSIGNMENTS: model_predictions[0][0],
    }

  def _get_eval_ops(self, features, _, unused_metrics):
    (_,
     _,
     losses,
     _) = gmm_ops.gmm(
         self._parse_tensor_or_dict(features),
         self._training_initial_clusters,
         self._num_clusters,
         self._random_seed,
         self._covariance_type,
         self._params)
    return {GMM.SCORES: _streaming_sum(math_ops.reduce_sum(losses))}


# TODO(xavigonzalvo): delete this after implementing model-fn based Estimator.
def _legacy_train_model(estimator,
                        input_fn,
                        steps,
                        feed_fn=None,
                        init_op=None,
                        init_feed_fn=None,
                        init_fn=None,
                        device_fn=None,
                        monitors=None,
                        log_every_steps=100,
                        fail_on_nan_loss=True,
                        max_steps=None):
  """Legacy train function of Estimator."""
  if hasattr(estimator.config, 'execution_mode'):
    if estimator.config.execution_mode not in ('all', 'train'):
      return

    # Stagger startup of worker sessions based on task id.
    sleep_secs = min(
        estimator.config.training_worker_max_startup_secs,
        estimator.config.task_id *
        estimator.config.training_worker_session_startup_stagger_secs)
    if sleep_secs:
      logging.info('Waiting %d secs before starting task %d.', sleep_secs,
                   estimator.config.task_id)
      time.sleep(sleep_secs)

  # Device allocation
  device_fn = device_fn or estimator._device_fn  # pylint: disable=protected-access

  with ops.Graph().as_default() as g, g.device(device_fn):
    random_seed_lib.set_random_seed(estimator.config.tf_random_seed)
    global_step = framework.create_global_step(g)
    features, labels = input_fn()
    estimator._check_inputs(features, labels)  # pylint: disable=protected-access

    # The default return type of _get_train_ops is ModelFnOps. But there are
    # some subclasses of tf.contrib.learn.Estimator which override this
    # method and use the legacy signature, namely _get_train_ops returns a
    # (train_op, loss) tuple. The following else-statement code covers these
    # cases, but will soon be deleted after the subclasses are updated.
    # TODO(b/32664904): Update subclasses and delete the else-statement.
    train_ops = estimator._get_train_ops(features, labels)  # pylint: disable=protected-access
    if isinstance(train_ops, model_fn_lib.ModelFnOps):  # Default signature
      train_op = train_ops.train_op
      loss_op = train_ops.loss
      if estimator.config.is_chief:
        hooks = train_ops.training_chief_hooks + train_ops.training_hooks
      else:
        hooks = train_ops.training_hooks
    else:  # Legacy signature
      if len(train_ops) != 2:
        raise ValueError('Expected a tuple of train_op and loss, got {}'.format(
            train_ops))
      train_op = train_ops[0]
      loss_op = train_ops[1]
      hooks = []

    hooks += monitor_lib.replace_monitors_with_hooks(monitors, estimator)

    ops.add_to_collection(ops.GraphKeys.LOSSES, loss_op)
    return graph_actions._monitored_train(  # pylint: disable=protected-access
        graph=g,
        output_dir=estimator.model_dir,
        train_op=train_op,
        loss_op=loss_op,
        global_step_tensor=global_step,
        init_op=init_op,
        init_feed_dict=init_feed_fn() if init_feed_fn is not None else None,
        init_fn=init_fn,
        log_every_steps=log_every_steps,
        supervisor_is_chief=estimator.config.is_chief,
        supervisor_master=estimator.config.master,
        supervisor_save_model_secs=estimator.config.save_checkpoints_secs,
        supervisor_save_model_steps=estimator.config.save_checkpoints_steps,
        supervisor_save_summaries_steps=estimator.config.save_summary_steps,
        keep_checkpoint_max=estimator.config.keep_checkpoint_max,
        keep_checkpoint_every_n_hours=(
            estimator.config.keep_checkpoint_every_n_hours),
        feed_fn=feed_fn,
        steps=steps,
        fail_on_nan_loss=fail_on_nan_loss,
        hooks=hooks,
        max_steps=max_steps)
