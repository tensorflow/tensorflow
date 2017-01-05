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

import numpy as np

from tensorflow.contrib import framework
from tensorflow.contrib.factorization.python.ops import gmm_ops
from tensorflow.contrib.framework.python.framework import checkpoint_utils
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators._sklearn import TransformerMixin
from tensorflow.contrib.learn.python.learn.learn_io import data_feeder
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops.control_flow_ops import with_dependencies


def _streaming_sum(scalar_tensor):
  """Create a sum metric and update op."""
  sum_metric = framework.local_variable(constant_op.constant(0.0))
  sum_update = sum_metric.assign_add(scalar_tensor)
  return sum_metric, sum_update


class GMM(estimator.Estimator, TransformerMixin):
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
    self._train_model(
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
      return array_ops.concat_v2([features[k] for k in sorted(features.keys())],
                                 1)
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
