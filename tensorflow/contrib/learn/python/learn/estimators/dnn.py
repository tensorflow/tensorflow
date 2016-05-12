"""Deep Neural Network estimators."""
#  Copyright 2015-present The Scikit Flow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.learn.python.learn.estimators import _sklearn
from tensorflow.contrib.learn.python.learn.estimators.base import TensorFlowEstimator
from tensorflow.contrib.learn.python.learn import models


class TensorFlowDNNClassifier(TensorFlowEstimator, _sklearn.ClassifierMixin):
  """TensorFlow DNN Classifier model.

  Parameters:
    hidden_units: List of hidden units per layer.
    n_classes: Number of classes in the target.
    batch_size: Mini batch size.
    steps: Number of steps to run over data.
    optimizer: Optimizer name (or class), for example "SGD", "Adam", "Adagrad".
    learning_rate: If this is constant float value, no decay function is used.
      Instead, a customized decay function can be passed that accepts
      global_step as parameter and returns a Tensor.
      e.g. exponential decay function:
      def exp_decay(global_step):
          return tf.train.exponential_decay(
              learning_rate=0.1, global_step,
              decay_steps=2, decay_rate=0.001)
    class_weight: None or list of n_classes floats. Weight associated with
      classes for loss computation. If not given, all classes are
      supposed to have weight one.
    continue_training: when continue_training is True, once initialized
      model will be continuely trained on every call of fit.
    config: RunConfig object that controls the configurations of the
      session, e.g. num_cores, gpu_memory_fraction, etc.
    dropout: When not None, the probability we will drop out a given coordinate.
  """

  def __init__(self,
               hidden_units,
               n_classes,
               batch_size=32,
               steps=200,
               optimizer='Adagrad',
               learning_rate=0.1,
               class_weight=None,
               clip_gradients=5.0,
               continue_training=False,
               config=None,
               verbose=1,
               dropout=None):
    self.hidden_units = hidden_units
    self.dropout = dropout
    super(TensorFlowDNNClassifier, self).__init__(
        model_fn=self._model_fn,
        n_classes=n_classes,
        batch_size=batch_size,
        steps=steps,
        optimizer=optimizer,
        learning_rate=learning_rate,
        class_weight=class_weight,
        clip_gradients=clip_gradients,
        continue_training=continue_training,
        config=config,
        verbose=verbose)

  def _model_fn(self, X, y):
    return models.get_dnn_model(self.hidden_units,
                                models.logistic_regression,
                                dropout=self.dropout)(X, y)

  @property
  def weights_(self):
    """Returns weights of the DNN weight layers."""
    return [self._session.run(w)
            for w in self._graph.get_collection('dnn_weights')
           ] + [self.get_tensor_value('logistic_regression/weights:0')]

  @property
  def bias_(self):
    """Returns bias of the DNN's bias layers."""
    return [self._session.run(b)
            for b in self._graph.get_collection('dnn_biases')
           ] + [self.get_tensor_value('logistic_regression/bias:0')]


class TensorFlowDNNRegressor(TensorFlowEstimator, _sklearn.RegressorMixin):
  """TensorFlow DNN Regressor model.

  Parameters:
    hidden_units: List of hidden units per layer.
    batch_size: Mini batch size.
    steps: Number of steps to run over data.
    optimizer: Optimizer name (or class), for example "SGD", "Adam", "Adagrad".
    learning_rate: If this is constant float value, no decay function is
      used. Instead, a customized decay function can be passed that accepts
      global_step as parameter and returns a Tensor.
      e.g. exponential decay function:
      def exp_decay(global_step):
          return tf.train.exponential_decay(
              learning_rate=0.1, global_step,
              decay_steps=2, decay_rate=0.001)
    continue_training: when continue_training is True, once initialized
      model will be continuely trained on every call of fit.
    config: RunConfig object that controls the configurations of the session,
      e.g. num_cores, gpu_memory_fraction, etc.
    verbose: Controls the verbosity, possible values:
      0: the algorithm and debug information is muted.
      1: trainer prints the progress.
      2: log device placement is printed.
    dropout: When not None, the probability we will drop out a given coordinate.
  """

  def __init__(self,
               hidden_units,
               n_classes=0,
               batch_size=32,
               steps=200,
               optimizer='Adagrad',
               learning_rate=0.1,
               clip_gradients=5.0,
               continue_training=False,
               config=None,
               verbose=1,
               dropout=None):
    self.hidden_units = hidden_units
    self.dropout = dropout
    super(TensorFlowDNNRegressor, self).__init__(
        model_fn=self._model_fn,
        n_classes=n_classes,
        batch_size=batch_size,
        steps=steps,
        optimizer=optimizer,
        learning_rate=learning_rate,
        clip_gradients=clip_gradients,
        continue_training=continue_training,
        config=config,
        verbose=verbose)

  def _model_fn(self, X, y):
    return models.get_dnn_model(self.hidden_units,
                                models.linear_regression,
                                dropout=self.dropout)(X, y)

  @property
  def weights_(self):
    """Returns weights of the DNN weight layers."""
    return [self._session.run(w)
            for w in self._graph.get_collection('dnn_weights')
           ] + [self.get_tensor_value('linear_regression/weights:0')]

  @property
  def bias_(self):
    """Returns bias of the DNN's bias layers."""
    return [self._session.run(b)
            for b in self._graph.get_collection('dnn_biases')
           ] + [self.get_tensor_value('linear_regression/bias:0')]
