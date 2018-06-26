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
"""Linear estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.estimator import estimator
from tensorflow.python.estimator.canned import linear as linear_lib


class LinearEstimator(estimator.Estimator):
  """An estimator for TensorFlow linear models with user-specified head.

  Example:

  ```python
  categorical_column_a = categorical_column_with_hash_bucket(...)
  categorical_column_b = categorical_column_with_hash_bucket(...)

  categorical_feature_a_x_categorical_feature_b = crossed_column(...)

  # Estimator using the default optimizer.
  estimator = LinearEstimator(
      head=tf.contrib.estimator.multi_label_head(n_classes=3),
      feature_columns=[categorical_column_a,
                       categorical_feature_a_x_categorical_feature_b])

  # Or estimator using an optimizer with a learning rate decay.
  estimator = LinearEstimator(
      head=tf.contrib.estimator.multi_label_head(n_classes=3),
      feature_columns=[categorical_column_a,
                       categorical_feature_a_x_categorical_feature_b],
      optimizer=lambda: tf.train.FtrlOptimizer(
          learning_rate=tf.exponential_decay(
              learning_rate=0.1,
              global_step=tf.get_global_step(),
              decay_steps=10000,
              decay_rate=0.96))

  # Or estimator using the FTRL optimizer with regularization.
  estimator = LinearEstimator(
      head=tf.contrib.estimator.multi_label_head(n_classes=3),
      feature_columns=[categorical_column_a,
                       categorical_feature_a_x_categorical_feature_b])
      optimizer=tf.train.FtrlOptimizer(
          learning_rate=0.1,
          l1_regularization_strength=0.001
      ))

  def input_fn_train: # returns x, y (where y represents label's class index).
    ...
  estimator.train(input_fn=input_fn_train, steps=100)
  def input_fn_eval: # returns x, y (where y represents label's class index).
    ...
  metrics = estimator.evaluate(input_fn=input_fn_eval, steps=10)
  def input_fn_predict: # returns x, None
    ...
  predictions = estimator.predict(input_fn=input_fn_predict)
  ```

  Input of `train` and `evaluate` should have following features,
  otherwise there will be a `KeyError`:

  * if `weight_column` is not `None`, a feature with
    `key=weight_column` whose value is a `Tensor`.
  * for each `column` in `feature_columns`:
    - if `column` is a `_CategoricalColumn`, a feature with `key=column.name`
      whose `value` is a `SparseTensor`.
    - if `column` is a `_WeightedCategoricalColumn`, two features: the first
      with `key` the id column name, the second with `key` the weight column
      name. Both features' `value` must be a `SparseTensor`.
    - if `column` is a `_DenseColumn`, a feature with `key=column.name`
      whose `value` is a `Tensor`.

  Loss and predicted output are determined by the specified head.

  @compatibility(eager)
  Estimators are not compatible with eager execution.
  @end_compatibility
  """

  def __init__(self,
               head,
               feature_columns,
               model_dir=None,
               optimizer='Ftrl',
               config=None,
               partitioner=None):
    """Initializes a `LinearEstimator` instance.

    Args:
      head: A `_Head` instance constructed with a method such as
        `tf.contrib.estimator.multi_label_head`.
      feature_columns: An iterable containing all the feature columns used by
        the model. All items in the set should be instances of classes derived
        from `FeatureColumn`.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator
        to continue training a previously saved model.
      optimizer: An instance of `tf.Optimizer` used to train the model. Can also
        be a string (one of 'Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD'), or
        callable. Defaults to FTRL optimizer.
      config: `RunConfig` object to configure the runtime settings.
      partitioner: Optional. Partitioner for input layer.
    """
    def _model_fn(features, labels, mode, config):
      return linear_lib._linear_model_fn(  # pylint: disable=protected-access
          features=features,
          labels=labels,
          mode=mode,
          head=head,
          feature_columns=tuple(feature_columns or []),
          optimizer=optimizer,
          partitioner=partitioner,
          config=config)
    super(LinearEstimator, self).__init__(
        model_fn=_model_fn, model_dir=model_dir, config=config)
