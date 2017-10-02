# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.python.estimator import estimator
from tensorflow.python.feature_column import feature_column as feature_column_lib #pylint: disable=line-too-long
from tensorflow.python.estimator.canned.utils import common_model_fn
from tensorflow.python.estimator.canned.utils import classifier_head
from tensorflow.python.estimator.canned.utils import regression_head
from tensorflow.python.estimator.canned.utils import add_layer_summary

# The default learning rate of 0.2 is a historical artifact of the initial
# implementation, but seems a reasonable choice.
_LEARNING_RATE = 0.2

def _linear_logit_fn_builder(units, feature_columns):
  """Function builder for a linear logit_fn.

  Args:
    units: An int indicating the dimension of the logit layer.
    feature_columns: An iterable containing all the feature columns used by
      the model.

  Returns:
    A logit_fn (see below).

  """

  def linear_logit_fn(features, mode=None, input_layer_partitioner=None): #pylint: disable=unused-argument
    """Linear model logit_fn.

    Args:
      features: This is the first item returned from the `input_fn`
                passed to `train`, `evaluate`, and `predict`. This should be a
                single `Tensor` or `dict` of same.

    Returns:
      A `Tensor` representing the logits.
    """
    logits = feature_column_lib.linear_model(
        features=features, feature_columns=feature_columns, units=units)
    add_layer_summary(logits, 'linear')
    return logits


  return linear_logit_fn


class _Linear(estimator.Estimator):
  """Common functionality for canned Linear models"""
  def __init__(self,
               head,
               feature_columns,
               model_dir,
               optimizer,
               config,
               partitioner):
    """Initializes a `_Linear` instance."""

    logit_fn = _linear_logit_fn_builder(
        units=head.logits_dimension,
        feature_columns=tuple(feature_columns or []))

    def _model_fn(features, labels, mode, config):
      return common_model_fn(
          name='linear',
          features=features,
          labels=labels,
          mode=mode,
          head=head,
          logit_fn=logit_fn,
          optimizer=optimizer,
          learning_rate=_LEARNING_RATE,
          partitioner=partitioner,
          config=config)

    super(_Linear, self).__init__(
        model_fn=_model_fn,
        model_dir=model_dir,
        config=config)


class LinearClassifier(_Linear):
  """Linear classifier model.

  Train a linear model to classify instances into one of multiple possible
  classes. When number of possible classes is 2, this is binary classification.

  Example:

  ```python
  sparse_column_a = sparse_column_with_hash_bucket(...)
  sparse_column_b = sparse_column_with_hash_bucket(...)

  sparse_feature_a_x_sparse_feature_b = crossed_column(...)

  # Estimator using the default optimizer.
  estimator = LinearClassifier(
      feature_columns=[sparse_column_a, sparse_feature_a_x_sparse_feature_b])

  # Or estimator using the FTRL optimizer with regularization.
  estimator = LinearClassifier(
      feature_columns=[sparse_column_a, sparse_feature_a_x_sparse_feature_b],
      optimizer=tf.train.FtrlOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.001
      ))

  # Input builders
  def input_fn_train: # returns x, y (where y represents label's class index).
    ...
  def input_fn_eval: # returns x, y (where y represents label's class index).
    ...
  estimator.train(input_fn=input_fn_train)
  estimator.evaluate(input_fn=input_fn_eval)
  estimator.predict(input_fn=input_fn_predict)
  ```

  Input of `train` and `evaluate` should have following features,
    otherwise there will be a `KeyError`:

  * if `weight_column` is not `None`, a feature with
    `key=weight_column` whose value is a `Tensor`.
  * for each `column` in `feature_columns`:
    - if `column` is a `SparseColumn`, a feature with `key=column.name`
      whose `value` is a `SparseTensor`.
    - if `column` is a `WeightedSparseColumn`, two features: the first with
      `key` the id column name, the second with `key` the weight column name.
      Both features' `value` must be a `SparseTensor`.
    - if `column` is a `RealValuedColumn`, a feature with `key=column.name`
      whose `value` is a `Tensor`.

  Loss is calculated by using softmax cross entropy.
  """

  def __init__(self,
               feature_columns,
               model_dir=None,
               n_classes=2,
               weight_column=None,
               label_vocabulary=None,
               optimizer='Ftrl',
               config=None,
               partitioner=None):
    """Construct a `LinearClassifier` estimator object.

    Args:
      feature_columns: An iterable containing all the feature columns used by
        the model. All items in the set should be instances of classes derived
        from `FeatureColumn`.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator
        to continue training a previously saved model.
      n_classes: number of label classes. Default is binary classification.
        Note that class labels are integers representing the class index (i.e.
        values from 0 to n_classes-1). For arbitrary label values (e.g. string
        labels), convert to class indices first.
      weight_column: A string or a `_NumericColumn` created by
        `tf.feature_column.numeric_column` defining feature column representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example. If it is a string, it is
        used as a key to fetch weight tensor from the `features`. If it is a
        `_NumericColumn`, raw tensor is fetched by key `weight_column.key`,
        then weight_column.normalizer_fn is applied on it to get weight tensor.
      label_vocabulary: A list of strings represents possible label values. If
        given, labels must be string type and have any value in
        `label_vocabulary`. If it is not given, that means labels are
        already encoded as integer or float within [0, 1] for `n_classes=2` and
        encoded as integer values in {0, 1,..., n_classes-1} for `n_classes`>2 .
        Also there will be errors if vocabulary is not provided and labels are
        string.
      optimizer: An instance of `tf.Optimizer` used to train the model. Defaults
        to FTRL optimizer.
      config: `RunConfig` object to configure the runtime settings.
      partitioner: Optional. Partitioner for input layer.

    Returns:
      A `LinearClassifier` estimator.

    Raises:
      ValueError: if n_classes < 2.
    """
    head = classifier_head(
        n_classes=n_classes,
        weight_column=weight_column,
        label_vocabulary=label_vocabulary)

    super(LinearClassifier, self).__init__(
        head=head,
        feature_columns=feature_columns,
        model_dir=model_dir,
        optimizer=optimizer,
        config=config,
        partitioner=partitioner)


class LinearRegressor(_Linear):
  """An estimator for TensorFlow Linear regression problems.

  Train a linear regression model to predict label value given observation of
  feature values.

  Example:

  ```python
  sparse_column_a = sparse_column_with_hash_bucket(...)
  sparse_column_b = sparse_column_with_hash_bucket(...)

  sparse_feature_a_x_sparse_feature_b = crossed_column(...)

  estimator = LinearRegressor(
      feature_columns=[sparse_column_a, sparse_feature_a_x_sparse_feature_b])

  # Input builders
  def input_fn_train: # returns x, y
    ...
  def input_fn_eval: # returns x, y
    ...
  estimator.train(input_fn=input_fn_train)
  estimator.evaluate(input_fn=input_fn_eval)
  estimator.predict(input_fn=input_fn_predict)
  ```

  Input of `train` and `evaluate` should have following features,
    otherwise there will be a KeyError:

  * if `weight_column` is not `None`:
    key=weight_column, value=a `Tensor`
  * for column in `feature_columns`:
    - if isinstance(column, `SparseColumn`):
        key=column.name, value=a `SparseTensor`
    - if isinstance(column, `WeightedSparseColumn`):
        {key=id column name, value=a `SparseTensor`,
         key=weight column name, value=a `SparseTensor`}
    - if isinstance(column, `RealValuedColumn`):
        key=column.name, value=a `Tensor`

  Loss is calculated by using mean squared error.
  """

  def __init__(self,
               feature_columns,
               model_dir=None,
               label_dimension=1,
               weight_column=None,
               optimizer='Ftrl',
               config=None,
               partitioner=None):
    """Initializes a `LinearRegressor` instance.

    Args:
      feature_columns: An iterable containing all the feature columns used by
        the model. All items in the set should be instances of classes derived
        from `FeatureColumn`.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator
        to continue training a previously saved model.
      label_dimension: Number of regression targets per example. This is the
        size of the last dimension of the labels and logits `Tensor` objects
        (typically, these have shape `[batch_size, label_dimension]`).
      weight_column: A string or a `_NumericColumn` created by
        `tf.feature_column.numeric_column` defining feature column representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example. If it is a string, it is
        used as a key to fetch weight tensor from the `features`. If it is a
        `_NumericColumn`, raw tensor is fetched by key `weight_column.key`,
        then weight_column.normalizer_fn is applied on it to get weight tensor.
      optimizer: An instance of `tf.Optimizer` used to train the model. Defaults
        to FTRL optimizer.
      config: `RunConfig` object to configure the runtime settings.
      partitioner: Optional. Partitioner for input layer.
    """
    head = regression_head(
        label_dimension=label_dimension,
        weight_column=weight_column)

    super(LinearRegressor, self).__init__(
        head=head,
        feature_columns=feature_columns,
        model_dir=model_dir,
        optimizer=optimizer,
        config=config,
        partitioner=partitioner)
