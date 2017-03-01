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
"""Support Vector Machine (SVM) Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers
from tensorflow.contrib.framework import deprecated
from tensorflow.contrib.framework import deprecated_arg_values
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import head as head_lib
from tensorflow.contrib.learn.python.learn.estimators import linear
from tensorflow.contrib.learn.python.learn.estimators import prediction_key
from tensorflow.contrib.linear_optimizer.python import sdca_optimizer


def _as_iterable(preds, output):
  for pred in preds:
    yield pred[output]


class SVM(estimator.Estimator):
  """Support Vector Machine (SVM) model for binary classification.

  Currently, only linear SVMs are supported. For the underlying optimization
  problem, the `SDCAOptimizer` is used. For performance and convergence tuning,
  the num_loss_partitions parameter passed to `SDCAOptimizer` (see `__init__()`
  method), should be set to (#concurrent train ops per worker) x (#workers). If
  num_loss_partitions is larger or equal to this value, convergence is
  guaranteed but becomes slower as num_loss_partitions increases. If it is set
  to a smaller value, the optimizer is more aggressive in reducing the global
  loss but convergence is not guaranteed. The recommended value in tf.learn
  (where there is one process per worker) is the number of workers running the
  train steps. It defaults to 1 (single machine).

  Example:

  ```python
  real_feature_column = real_valued_column(...)
  sparse_feature_column = sparse_column_with_hash_bucket(...)

  estimator = SVM(
      example_id_column='example_id',
      feature_columns=[real_feature_column, sparse_feature_column],
      l2_regularization=10.0)

  # Input builders
  def input_fn_train: # returns x, y
    ...
  def input_fn_eval: # returns x, y
    ...

  estimator.fit(input_fn=input_fn_train)
  estimator.evaluate(input_fn=input_fn_eval)
  estimator.predict(x=x)
  ```

  Input of `fit` and `evaluate` should have following features, otherwise there
  will be a `KeyError`:
    a feature with `key=example_id_column` whose value is a `Tensor` of dtype
    string.
    if `weight_column_name` is not `None`, a feature with
    `key=weight_column_name` whose value is a `Tensor`.
    for each `column` in `feature_columns`:
      - if `column` is a `SparseColumn`, a feature with `key=column.name`
        whose `value` is a `SparseTensor`.
      - if `column` is a `RealValuedColumn, a feature with `key=column.name`
        whose `value` is a `Tensor`.
  """

  def __init__(self,
               example_id_column,
               feature_columns,
               weight_column_name=None,
               model_dir=None,
               l1_regularization=0.0,
               l2_regularization=0.0,
               num_loss_partitions=1,
               kernels=None,
               config=None,
               feature_engineering_fn=None):
    """Constructs an `SVM` estimator object.

    Args:
      example_id_column: A string defining the feature column name representing
        example ids. Used to initialize the underlying optimizer.
      feature_columns: An iterable containing all the feature columns used by
        the model. All items in the set should be instances of classes derived
        from `FeatureColumn`.
      weight_column_name: A string defining feature column name representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
      l1_regularization: L1-regularization parameter. Refers to global L1
        regularization (across all examples).
      l2_regularization: L2-regularization parameter. Refers to global L2
        regularization (across all examples).
      num_loss_partitions: number of partitions of the (global) loss function
        optimized by the underlying optimizer (SDCAOptimizer).
      kernels: A list of kernels for the SVM. Currently, no kernels are
        supported. Reserved for future use for non-linear SVMs.
      config: RunConfig object to configure the runtime settings.
      feature_engineering_fn: Feature engineering function. Takes features and
                        labels which are the output of `input_fn` and
                        returns features and labels which will be fed
                        into the model.

    Raises:
      ValueError: if kernels passed is not None.
    """
    if kernels is not None:
      raise ValueError("Kernel SVMs are not currently supported.")
    optimizer = sdca_optimizer.SDCAOptimizer(
        example_id_column=example_id_column,
        num_loss_partitions=num_loss_partitions,
        symmetric_l1_regularization=l1_regularization,
        symmetric_l2_regularization=l2_regularization)

    self._feature_columns = feature_columns
    chief_hook = linear._SdcaUpdateWeightsHook()  # pylint: disable=protected-access
    super(SVM, self).__init__(
        model_fn=linear.sdca_model_fn,
        model_dir=model_dir,
        config=config,
        params={
            "head": head_lib._binary_svm_head(  # pylint: disable=protected-access
                weight_column_name=weight_column_name,
                enable_centered_bias=False),
            "feature_columns": feature_columns,
            "optimizer": optimizer,
            "weight_column_name": weight_column_name,
            "update_weights_hook": chief_hook,
        },
        feature_engineering_fn=feature_engineering_fn)

  @deprecated_arg_values(
      estimator.AS_ITERABLE_DATE, estimator.AS_ITERABLE_INSTRUCTIONS,
      as_iterable=False)
  def predict_classes(self, x=None, input_fn=None, batch_size=None,
                      as_iterable=True):
    """Runs inference to determine the predicted class."""
    key = prediction_key.PredictionKey.CLASSES
    preds = super(SVM, self).predict(
        x=x,
        input_fn=input_fn,
        batch_size=batch_size,
        outputs=[key],
        as_iterable=as_iterable)
    if as_iterable:
      return _as_iterable(preds, output=key)
    return preds[key]

  @deprecated_arg_values(
      estimator.AS_ITERABLE_DATE, estimator.AS_ITERABLE_INSTRUCTIONS,
      as_iterable=False)
  def predict_proba(self, x=None, input_fn=None, batch_size=None, outputs=None,
                    as_iterable=True):
    """Runs inference to determine the class probability predictions."""
    key = prediction_key.PredictionKey.PROBABILITIES
    preds = super(SVM, self).predict(
        x=x,
        input_fn=input_fn,
        batch_size=batch_size,
        outputs=[key],
        as_iterable=as_iterable)
    if as_iterable:
      return _as_iterable(preds, output=key)
    return preds[key]
  # pylint: enable=protected-access

  @deprecated("2017-03-25", "Please use Estimator.export_savedmodel() instead.")
  def export(self, export_dir, signature_fn=None,
             input_fn=None, default_batch_size=1,
             exports_to_keep=None):
    """See BaseEstimator.export."""
    return self.export_with_defaults(
        export_dir=export_dir,
        signature_fn=signature_fn,
        input_fn=input_fn,
        default_batch_size=default_batch_size,
        exports_to_keep=exports_to_keep)

  @deprecated("2017-03-25", "Please use Estimator.export_savedmodel() instead.")
  def export_with_defaults(
      self,
      export_dir,
      signature_fn=None,
      input_fn=None,
      default_batch_size=1,
      exports_to_keep=None):
    """Same as BaseEstimator.export, but uses some defaults."""
    def default_input_fn(unused_estimator, examples):
      return layers.parse_feature_columns_from_examples(
          examples, self._feature_columns)
    return super(SVM, self).export(export_dir=export_dir,
                                   signature_fn=signature_fn,
                                   input_fn=input_fn or default_input_fn,
                                   default_batch_size=default_batch_size,
                                   exports_to_keep=exports_to_keep)
