# pylint: disable=g-bad-file-header
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

"""Deep Neural Network estimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers
from tensorflow.contrib.learn.python.learn import models
from tensorflow.contrib.learn.python.learn.estimators import _sklearn
from tensorflow.contrib.learn.python.learn.estimators import dnn_linear_combined
from tensorflow.contrib.learn.python.learn.estimators.base import TensorFlowEstimator
from tensorflow.python.ops import nn


# TODO(ipolosukhin): Merge thirdparty DNN with this.
class DNNClassifier(dnn_linear_combined.DNNLinearCombinedClassifier):
  """A classifier for TensorFlow DNN models.

    Example:
      ```
      installed_app_id = sparse_column_with_hash_bucket("installed_id", 1e6)
      impression_app_id = sparse_column_with_hash_bucket("impression_id", 1e6)

      installed_emb = embedding_column(installed_app_id, dimension=16,
                                       combiner="sum")
      impression_emb = embedding_column(impression_app_id, dimension=16,
                                        combiner="sum")

      estimator = DNNClassifier(
          feature_columns=[installed_emb, impression_emb],
          hidden_units=[1024, 512, 256])

      # Input builders
      def input_fn_train: # returns X, Y
        pass
      estimator.train(input_fn_train)

      def input_fn_eval: # returns X, Y
        pass
      estimator.evaluate(input_fn_eval)
      estimator.predict(x)
      ```

    Input of `fit`, `train`, and `evaluate` should have following features,
      otherwise there will be a `KeyError`:
        if `weight_column_name` is not `None`, a feature with
          `key=weight_column_name` whose value is a `Tensor`.
        for each `column` in `feature_columns`:
        - if `column` is a `SparseColumn`, a feature with `key=column.name`
          whose `value` is a `SparseTensor`.
        - if `column` is a `RealValuedColumn, a feature with `key=column.name`
          whose `value` is a `Tensor`.
        - if `feauture_columns` is None, then `input` must contains only real
          valued `Tensor`.

  Parameters:
    hidden_units: List of hidden units per layer. All layers are fully
      connected. Ex. [64, 32] means first layer has 64 nodes and second one has
      32.
    feature_columns: An iterable containing all the feature columns used by the
      model. All items in the set should be instances of classes derived from
      `FeatureColumn`.
    model_dir: Directory to save model parameters, graph and etc.
    n_classes: number of target classes. Default is binary classification.
      It must be greater than 1.
    weight_column_name: A string defining feature column name representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example.
    optimizer: An instance of `tf.Optimizer` used to train the model. If `None`,
      will use an Adagrad optimizer.
    activation_fn: Activation function applied to each layer. If `None`, will
      use `tf.nn.relu`.
  """

  def __init__(self,
               hidden_units,
               feature_columns=None,
               model_dir=None,
               n_classes=2,
               weight_column_name=None,
               optimizer=None,
               activation_fn=nn.relu):
    super(DNNClassifier, self).__init__(n_classes=n_classes,
                                        weight_column_name=weight_column_name,
                                        dnn_feature_columns=feature_columns,
                                        dnn_optimizer=optimizer,
                                        dnn_hidden_units=hidden_units,
                                        dnn_activation_fn=activation_fn)

  def _get_train_ops(self, features, targets):
    """See base class."""
    if self._dnn_feature_columns is None:
      self._dnn_feature_columns = layers.infer_real_valued_columns(features)
    return super(DNNClassifier, self)._get_train_ops(features, targets)


class DNNRegressor(dnn_linear_combined.DNNLinearCombinedRegressor):
  """A regressor for TensorFlow DNN models.

    Example:
      ```
      installed_app_id = sparse_column_with_hash_bucket("installed_id", 1e6)
      impression_app_id = sparse_column_with_hash_bucket("impression_id", 1e6)

      installed_emb = embedding_column(installed_app_id, dimension=16,
                                       combiner="sum")
      impression_emb = embedding_column(impression_app_id, dimension=16,
                                        combiner="sum")

      estimator = DNNRegressor(
          feature_columns=[installed_emb, impression_emb],
          hidden_units=[1024, 512, 256])

      # Input builders
      def input_fn_train: # returns X, Y
        pass
      estimator.train(input_fn_train)

      def input_fn_eval: # returns X, Y
        pass
      estimator.evaluate(input_fn_eval)
      estimator.predict(x)
      ```

    Input of `fit`, `train`, and `evaluate` should have following features,
      otherwise there will be a `KeyError`:
        if `weight_column_name` is not `None`, a feature with
          `key=weight_column_name` whose value is a `Tensor`.
        for each `column` in `feature_columns`:
        - if `column` is a `SparseColumn`, a feature with `key=column.name`
          whose `value` is a `SparseTensor`.
        - if `column` is a `RealValuedColumn, a feature with `key=column.name`
          whose `value` is a `Tensor`.
        - if `feauture_columns` is None, then `input` must contains only real
          valued `Tensor`.



  Parameters:
    hidden_units: List of hidden units per layer. All layers are fully
      connected. Ex. [64, 32] means first layer has 64 nodes and second one has
      32.
    feature_columns: An iterable containing all the feature columns used by the
      model. All items in the set should be instances of classes derived from
      `FeatureColumn`.
    model_dir: Directory to save model parameters, graph and etc.
    weight_column_name: A string defining feature column name representing
      weights. It is used to down weight or boost examples during training. It
      will be multiplied by the loss of the example.
    optimizer: An instance of `tf.Optimizer` used to train the model. If `None`,
      will use an Adagrad optimizer.
    activation_fn: Activation function applied to each layer. If `None`, will
      use `tf.nn.relu`.
  """

  def __init__(self,
               hidden_units,
               feature_columns=None,
               model_dir=None,
               weight_column_name=None,
               optimizer=None,
               activation_fn=nn.relu):
    super(DNNRegressor, self).__init__(weight_column_name=weight_column_name,
                                       dnn_feature_columns=feature_columns,
                                       dnn_optimizer=optimizer,
                                       dnn_hidden_units=hidden_units,
                                       dnn_activation_fn=activation_fn)

  def _get_train_ops(self, features, targets):
    """See base class."""
    if self._dnn_feature_columns is None:
      self._dnn_feature_columns = layers.infer_real_valued_columns(features)
    return super(DNNRegressor, self)._get_train_ops(features, targets)


# TODO(ipolosukhin): Deprecate this class in favor of DNNClassifier.
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
    return [self.get_tensor_value(w.name)
            for w in self._graph.get_collection('dnn_weights')
           ] + [self.get_tensor_value('logistic_regression/weights')]

  @property
  def bias_(self):
    """Returns bias of the DNN's bias layers."""
    return [self.get_tensor_value(b.name)
            for b in self._graph.get_collection('dnn_biases')
           ] + [self.get_tensor_value('logistic_regression/bias')]


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
    return [self.get_tensor_value(w.name)
            for w in self._graph.get_collection('dnn_weights')
           ] + [self.get_tensor_value('linear_regression/weights')]

  @property
  def bias_(self):
    """Returns bias of the DNN's bias layers."""
    return [self.get_tensor_value(b.name)
            for b in self._graph.get_collection('dnn_biases')
           ] + [self.get_tensor_value('linear_regression/bias')]
