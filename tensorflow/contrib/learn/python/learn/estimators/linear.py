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

"""Linear Estimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.learn.python.learn.estimators import _sklearn
from tensorflow.contrib.learn.python.learn.estimators import dnn_linear_combined
from tensorflow.contrib.learn.python.learn.estimators.base import DeprecatedMixin
from tensorflow.contrib.linear_optimizer.python import sdca_optimizer
from tensorflow.python.framework import ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.platform import tf_logging as logging


# TODO(b/29580537): Replace with @changing decorator.
def _changing(feature_columns):
  if feature_columns is not None:
    return
  logging.warn(
      "Change warning: `feature_columns` will be required after 2016-08-01.\n"
      "Instructions for updating:\n"
      "Pass `tf.contrib.learn.infer_real_valued_columns_from_input(x)` or"
      " `tf.contrib.learn.infer_real_valued_columns_from_input_fn(input_fn)`"
      " as `feature_columns`, where `x` or `input_fn` is your argument to"
      " `fit`, `evaluate`, or `predict`.")


class LinearClassifier(dnn_linear_combined.DNNLinearCombinedClassifier):
  """Linear classifier model.

  Train a linear model to classify instances into one of multiple possible
  classes. When number of possible classes is 2, this is binary classification.

  Example:

  ```python
  education = sparse_column_with_hash_bucket(column_name="education",
                                             hash_bucket_size=1000)
  occupation = sparse_column_with_hash_bucket(column_name="occupation",
                                              hash_bucket_size=1000)

  education_x_occupation = crossed_column(columns=[education, occupation],
                                          hash_bucket_size=10000)

  # Estimator using the default optimizer.
  estimator = LinearClassifier(
      feature_columns=[occupation, education_x_occupation])

  # Or estimator using the FTRL optimizer with regularization.
  estimator = LinearClassifier(
      feature_columns=[occupation, education_x_occupation],
      optimizer=tf.train.FtrlOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.001
      ))

  # Or estimator using the SDCAOptimizer.
  estimator = LinearClassifier(
     feature_columns=[occupation, education_x_occupation],
     optimizer=tf.contrib.linear_optimizer.SDCAOptimizer(
       example_id_column='example_id',
       symmetric_l2_regularization=2.0
     ))

  # Input builders
  def input_fn_train: # returns x, y
    ...
  def input_fn_eval: # returns x, y
    ...
  estimator.fit(input_fn=input_fn_train)
  estimator.evaluate(input_fn=input_fn_eval)
  estimator.predict(x=x)
  ```

  Input of `fit` and `evaluate` should have following features,
    otherwise there will be a `KeyError`:

  * if `weight_column_name` is not `None`, a feature with
    `key=weight_column_name` whose value is a `Tensor`.
  * for each `column` in `feature_columns`:
    - if `column` is a `SparseColumn`, a feature with `key=column.name`
      whose `value` is a `SparseTensor`.
    - if `column` is a `WeightedSparseColumn`, two features: the first with
      `key` the id column name, the second with `key` the weight column name.
      Both features' `value` must be a `SparseTensor`.
    - if `column` is a `RealValuedColumn`, a feature with `key=column.name`
      whose `value` is a `Tensor`.
    - if `feature_columns` is `None`, then `input` must contains only real
      valued `Tensor`.
  """

  def __init__(self,
               feature_columns=None,
               model_dir=None,
               n_classes=2,
               weight_column_name=None,
               optimizer=None,
               gradient_clip_norm=None,
               enable_centered_bias=True,
               config=None):
    """Construct a `LinearClassifier` estimator object.

    Args:
      feature_columns: An iterable containing all the feature columns used by
        the model. All items in the set should be instances of classes derived
        from `FeatureColumn`.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator
        to continue training a previously saved model.
      n_classes: number of target classes. Default is binary classification.
      weight_column_name: A string defining feature column name representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example.
      optimizer: The optimizer used to train the model. If specified, it should
        be either an instance of `tf.Optimizer` or the SDCAOptimizer. If `None`,
        the Ftrl optimizer will be used.
      gradient_clip_norm: A `float` > 0. If provided, gradients are clipped
        to their global norm with this clipping ratio. See
        `tf.clip_by_global_norm` for more details.
      enable_centered_bias: A bool. If True, estimator will learn a centered
        bias variable for each class. Rest of the model structure learns the
        residual after centered bias.
      config: `RunConfig` object to configure the runtime settings.

    Returns:
      A `LinearClassifier` estimator.
    """
    _changing(feature_columns)
    super(LinearClassifier, self).__init__(
        model_dir=model_dir,
        n_classes=n_classes,
        weight_column_name=weight_column_name,
        linear_feature_columns=feature_columns,
        linear_optimizer=optimizer,
        gradient_clip_norm=gradient_clip_norm,
        enable_centered_bias=enable_centered_bias,
        config=config)
    self._feature_columns_inferred = False

  # TODO(b/29580537): Remove feature_columns inference.
  def _validate_linear_feature_columns(self, features):
    if self._linear_feature_columns is None:
      self._linear_feature_columns = layers.infer_real_valued_columns(features)
      self._feature_columns_inferred = True
    elif self._feature_columns_inferred:
      this_dict = {c.name: c for c in self._linear_feature_columns}
      that_dict = {
          c.name: c for c in layers.infer_real_valued_columns(features)
      }
      if this_dict != that_dict:
        raise ValueError(
            "Feature columns, expected %s, got %s.", (this_dict, that_dict))

  def _get_train_ops(self, features, targets):
    """See base class."""
    self._validate_linear_feature_columns(features)
    if not isinstance(self._linear_optimizer, sdca_optimizer.SDCAOptimizer):
      return super(LinearClassifier, self)._get_train_ops(features, targets)

    # SDCA currently supports binary classification only.
    if self._target_column.num_label_columns > 2:
      raise ValueError(
          "SDCA does not currently support multi-class classification.")
    global_step = contrib_variables.get_global_step()
    assert global_step

    logits, columns_to_variables, _ = layers.weighted_sum_from_feature_columns(
        columns_to_tensors=features,
        feature_columns=self._linear_feature_columns,
        num_outputs=self._target_column.num_label_columns,
        weight_collections=[self._linear_model.get_scope_name()],
        scope=self._linear_model.get_scope_name())
    with ops.control_dependencies([self._centered_bias()]):
      loss = self._target_column.loss(logits, targets, features)
    logging_ops.scalar_summary("loss", loss)

    train_ops = self._linear_optimizer.get_train_step(
        self._linear_feature_columns, self._target_column.weight_column_name,
        self._loss_type(), features, targets, columns_to_variables, global_step)

    return train_ops, loss

  def _get_eval_ops(self, features, targets, metrics=None):
    self._validate_linear_feature_columns(features)
    return super(LinearClassifier, self)._get_eval_ops(features, targets,
                                                       metrics)

  def _get_predict_ops(self, features):
    """See base class."""
    self._validate_linear_feature_columns(features)
    return super(LinearClassifier, self)._get_predict_ops(features)

  def _loss_type(self):
    return "logistic_loss"

  @property
  def weights_(self):
    return self.linear_weights_

  @property
  def bias_(self):
    return self.linear_bias_


class LinearRegressor(dnn_linear_combined.DNNLinearCombinedRegressor):
  """Linear regressor model.

  Train a linear regression model to predict target variable value given
  observation of feature values.

  Example:

  ```python
  education = sparse_column_with_hash_bucket(column_name="education",
                                             hash_bucket_size=1000)
  occupation = sparse_column_with_hash_bucket(column_name="occupation",
                                              hash_bucket_size=1000)

  education_x_occupation = crossed_column(columns=[education, occupation],
                                          hash_bucket_size=10000)

  estimator = LinearRegressor(
      feature_columns=[occupation, education_x_occupation])

  # Input builders
  def input_fn_train: # returns x, y
    ...
  def input_fn_eval: # returns x, y
    ...
  estimator.fit(input_fn=input_fn_train)
  estimator.evaluate(input_fn=input_fn_eval)
  estimator.predict(x=x)
  ```

  Input of `fit` and `evaluate` should have following features,
    otherwise there will be a KeyError:

  * if `weight_column_name` is not `None`:
    key=weight_column_name, value=a `Tensor`
  * for column in `feature_columns`:
    - if isinstance(column, `SparseColumn`):
        key=column.name, value=a `SparseTensor`
    - if isinstance(column, `WeightedSparseColumn`):
        {key=id column name, value=a `SparseTensor`,
         key=weight column name, value=a `SparseTensor`}
    - if isinstance(column, `RealValuedColumn`):
        key=column.name, value=a `Tensor`
    - if `feature_columns` is `None`:
        input must contains only real valued `Tensor`.
  """

  def __init__(self,
               feature_columns=None,
               model_dir=None,
               weight_column_name=None,
               optimizer=None,
               gradient_clip_norm=None,
               enable_centered_bias=True,
               target_dimension=1,
               config=None):
    """Construct a `LinearRegressor` estimator object.

    Args:
      feature_columns: An iterable containing all the feature columns used by
        the model. All items in the set should be instances of classes derived
        from `FeatureColumn`.
      model_dir: Directory to save model parameters, graph, etc. This can
        also be used to load checkpoints from the directory into a estimator
        to continue training a previously saved model.
      weight_column_name: A string defining feature column name representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example.
      optimizer: An instance of `tf.Optimizer` used to train the model. If
        `None`, will use an Ftrl optimizer.
      gradient_clip_norm: A `float` > 0. If provided, gradients are clipped
        to their global norm with this clipping ratio. See
        `tf.clip_by_global_norm` for more details.
      enable_centered_bias: A bool. If True, estimator will learn a centered
        bias variable for each class. Rest of the model structure learns the
        residual after centered bias.
      target_dimension: dimension of the target for multilabels.
      config: `RunConfig` object to configure the runtime settings.

    Returns:
      A `LinearRegressor` estimator.
    """
    _changing(feature_columns)
    super(LinearRegressor, self).__init__(
        model_dir=model_dir,
        weight_column_name=weight_column_name,
        linear_feature_columns=feature_columns,
        linear_optimizer=optimizer,
        gradient_clip_norm=gradient_clip_norm,
        enable_centered_bias=enable_centered_bias,
        target_dimension=target_dimension,
        config=config)
    self._feature_columns_inferred = False

  # TODO(b/29580537): Remove feature_columns inference.
  def _validate_linear_feature_columns(self, features):
    if self._linear_feature_columns is None:
      self._linear_feature_columns = layers.infer_real_valued_columns(features)
      self._feature_columns_inferred = True
    elif self._feature_columns_inferred:
      this_dict = {c.name: c for c in self._linear_feature_columns}
      that_dict = {
          c.name: c for c in layers.infer_real_valued_columns(features)
      }
      if this_dict != that_dict:
        raise ValueError(
            "Feature columns, expected %s, got %s.", (this_dict, that_dict))

  def _get_train_ops(self, features, targets):
    """See base class."""
    if isinstance(self._linear_optimizer, sdca_optimizer.SDCAOptimizer):
      raise ValueError("SDCAOptimizer does not currently support regression.")
    self._validate_linear_feature_columns(features)
    return super(LinearRegressor, self)._get_train_ops(features, targets)

  def _get_eval_ops(self, features, targets, metrics=None):
    self._validate_linear_feature_columns(features)
    return super(LinearRegressor, self)._get_eval_ops(features, targets,
                                                      metrics)

  def _get_predict_ops(self, features):
    """See base class."""
    self._validate_linear_feature_columns(features)
    return super(LinearRegressor, self)._get_predict_ops(features)

  @property
  def weights_(self):
    return self.linear_weights_

  @property
  def bias_(self):
    return self.linear_bias_


# TensorFlowLinearRegressor and TensorFlowLinearClassifier are deprecated.
class TensorFlowLinearRegressor(DeprecatedMixin, LinearRegressor,
                                _sklearn.RegressorMixin):
  pass


class TensorFlowLinearClassifier(DeprecatedMixin, LinearClassifier,
                                 _sklearn.ClassifierMixin):
  pass


TensorFlowRegressor = TensorFlowLinearRegressor
TensorFlowClassifier = TensorFlowLinearClassifier
