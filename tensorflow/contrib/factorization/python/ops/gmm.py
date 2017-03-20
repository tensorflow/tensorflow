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
"""Implementation of Gaussian mixture model (GMM) clustering using tf.Learn."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib import framework
from tensorflow.contrib.factorization.python.ops import gmm_ops
from tensorflow.contrib.framework.python.framework import checkpoint_utils
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
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


class GMM(estimator.Estimator):
  """An estimator for GMM clustering."""
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
               config=None):
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
      config: See Estimator
    """
    self._num_clusters = num_clusters
    self._params = params
    self._training_initial_clusters = initial_clusters
    self._covariance_type = covariance_type
    self._training_graph = None
    self._random_seed = random_seed
    super(GMM, self).__init__(
        model_fn=self._model_builder(), model_dir=model_dir, config=config)

  def predict_assignments(self, input_fn=None, batch_size=None, outputs=None):
    """See BaseEstimator.predict."""
    results = self.predict(input_fn=input_fn,
                           batch_size=batch_size,
                           outputs=outputs)
    for result in results:
      yield result[GMM.ASSIGNMENTS]

  def score(self, input_fn=None, batch_size=None, steps=None):
    """Predict total sum of distances to nearest clusters.

    Note that this function is different from the corresponding one in sklearn
    which returns the negative of the sum of distances.

    Args:
      input_fn: see predict.
      batch_size: see predict.
      steps: see predict.

    Returns:
      Total sum of distances to nearest clusters.
    """
    results = self.evaluate(input_fn=input_fn, batch_size=batch_size,
                            steps=steps)
    return np.sum(results[GMM.SCORES])
  
  def weights(self):
    """Returns the cluster weights."""
    return checkpoint_utils.load_variable(
        self.model_dir, gmm_ops.GmmAlgorithm.CLUSTERS_WEIGHT)
    
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
      return array_ops.concat([features[k] for k in sorted(features.keys())],
                              1)
    return features

  def _model_builder(self):
    """Creates a model function."""

    def _model_fn(features, labels, mode):
      """Model function."""
      assert labels is None, labels
      (all_scores, model_predictions, losses, training_op) = gmm_ops.gmm(
          self._parse_tensor_or_dict(features), self._training_initial_clusters,
          self._num_clusters, self._random_seed, self._covariance_type,
          self._params)
      incr_step = state_ops.assign_add(variables.get_global_step(), 1)
      loss = math_ops.reduce_sum(losses)
      training_op = with_dependencies([training_op, incr_step], loss)
      predictions = {
          GMM.ALL_SCORES: all_scores[0],
          GMM.ASSIGNMENTS: model_predictions[0][0],
      }
      eval_metric_ops = {
          GMM.SCORES: _streaming_sum(loss),
      }
      return model_fn_lib.ModelFnOps(mode=mode, predictions=predictions,
                                     eval_metric_ops=eval_metric_ops,
                                     loss=loss, train_op=training_op)

    return _model_fn
