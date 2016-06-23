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
from tensorflow.contrib.learn.python.learn.estimators import _sklearn
from tensorflow.contrib.learn.python.learn.estimators import dnn_linear_combined
from tensorflow.contrib.learn.python.learn.estimators.base import DeprecatedMixin
from tensorflow.python.ops import nn


class DNNClassifier(dnn_linear_combined.DNNLinearCombinedClassifier):
  """A classifier for TensorFlow DNN models.

  Example:

  ```python
  education = sparse_column_with_hash_bucket(column_name="education",
                                             hash_bucket_size=1000)
  occupation = sparse_column_with_hash_bucket(column_name="occupation",
                                              hash_bucket_size=1000)

  education_emb = embedding_column(sparse_id_column=education, dimension=16,
                                   combiner="sum")
  occupation_emb = embedding_column(sparse_id_column=occupation, dimension=16,
                                   combiner="sum")

  estimator = DNNClassifier(
      feature_columns=[education_emb, occupation_emb],
      hidden_units=[1024, 512, 256])

  # Or estimator using the ProximalAdagradOptimizer optimizer with
  # regularization.
  estimator = DNNClassifier(
      feature_columns=[education_emb, occupation_emb],
      hidden_units=[1024, 512, 256],
      optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.001
      ))

  # Input builders
  def input_fn_train: # returns x, Y
    pass
  estimator.fit(input_fn=input_fn_train)

  def input_fn_eval: # returns x, Y
    pass
  estimator.evaluate(input_fn=input_fn_eval)
  estimator.predict(x=x)
  ```

  Input of `fit` and `evaluate` should have following features,
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
  """

  def __init__(self,
               hidden_units,
               feature_columns=None,
               model_dir=None,
               n_classes=2,
               weight_column_name=None,
               optimizer=None,
               activation_fn=nn.relu,
               dropout=None,
               config=None):
    super(DNNClassifier, self).__init__(model_dir=model_dir,
                                        n_classes=n_classes,
                                        weight_column_name=weight_column_name,
                                        dnn_feature_columns=feature_columns,
                                        dnn_optimizer=optimizer,
                                        dnn_hidden_units=hidden_units,
                                        dnn_activation_fn=activation_fn,
                                        dnn_dropout=dropout,
                                        config=config)

  def _get_train_ops(self, features, targets):
    """See base class."""
    if self._dnn_feature_columns is None:
      self._dnn_feature_columns = layers.infer_real_valued_columns(features)
    return super(DNNClassifier, self)._get_train_ops(features, targets)

  @property
  def weights_(self):
    return self.dnn_weights_

  @property
  def bias_(self):
    return self.dnn_bias_


class DNNRegressor(dnn_linear_combined.DNNLinearCombinedRegressor):
  """A regressor for TensorFlow DNN models.

  Example:

  ```python
  education = sparse_column_with_hash_bucket(column_name="education",
                                             hash_bucket_size=1000)
  occupation = sparse_column_with_hash_bucket(column_name="occupation",
                                              hash_bucket_size=1000)

  education_emb = embedding_column(sparse_id_column=education, dimension=16,
                                   combiner="sum")
  occupation_emb = embedding_column(sparse_id_column=occupation, dimension=16,
                                   combiner="sum")

  estimator = DNNRegressor(
      feature_columns=[education_emb, occupation_emb],
      hidden_units=[1024, 512, 256])

  # Or estimator using the ProximalAdagradOptimizer optimizer with
  # regularization.
  estimator = DNNRegressor(
      feature_columns=[education_emb, occupation_emb],
      hidden_units=[1024, 512, 256],
      optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.001
      ))

  # Input builders
  def input_fn_train: # returns x, Y
    pass
  estimator.fit(input_fn=input_fn_train)

  def input_fn_eval: # returns x, Y
    pass
  estimator.evaluate(input_fn=input_fn_eval)
  estimator.predict(x=x)
  ```

  Input of `fit` and `evaluate` should have following features,
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
  """

  def __init__(self,
               hidden_units,
               feature_columns=None,
               model_dir=None,
               weight_column_name=None,
               optimizer=None,
               activation_fn=nn.relu,
               dropout=None,
               config=None):
    super(DNNRegressor, self).__init__(model_dir=model_dir,
                                       weight_column_name=weight_column_name,
                                       dnn_feature_columns=feature_columns,
                                       dnn_optimizer=optimizer,
                                       dnn_hidden_units=hidden_units,
                                       dnn_activation_fn=activation_fn,
                                       dnn_dropout=dropout,
                                       config=config)

  def _get_train_ops(self, features, targets):
    """See base class."""
    if self._dnn_feature_columns is None:
      self._dnn_feature_columns = layers.infer_real_valued_columns(features)
    return super(DNNRegressor, self)._get_train_ops(features, targets)

  @property
  def weights_(self):
    return self.dnn_weights_

  @property
  def bias_(self):
    return self.dnn_bias_


# TensorFlowDNNClassifier and TensorFlowDNNRegressor are deprecated.
class TensorFlowDNNClassifier(DeprecatedMixin, DNNClassifier,
                              _sklearn.ClassifierMixin):
  pass


class TensorFlowDNNRegressor(DeprecatedMixin, DNNRegressor,
                             _sklearn.RegressorMixin):
  pass
