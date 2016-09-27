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

import inspect
import tempfile

from tensorflow.contrib import layers
from tensorflow.contrib import metrics as metrics_lib
from tensorflow.contrib.layers.python.layers import target_column
from tensorflow.contrib.learn.python.learn import evaluable
from tensorflow.contrib.learn.python.learn import metric_spec
from tensorflow.contrib.learn.python.learn import trainable
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import linear
from tensorflow.contrib.learn.python.learn.utils import checkpoints
from tensorflow.contrib.linear_optimizer.python import sdca_optimizer


def _as_iterable(preds, output):
  for pred in preds:
    yield pred[output]


def _get_metric_args(metric):
  if hasattr(metric, "__code__"):
    return inspect.getargspec(metric).args
  elif hasattr(metric, "func") and hasattr(metric, "keywords"):
    return [arg for arg in inspect.getargspec(metric.func).args
            if arg not in metric.keywords.keys()]


class SVM(trainable.Trainable, evaluable.Evaluable):
  """Support Vector Machine (SVM) model for binary classification.

  Currently, only linear SVMs are supported. For the underlying optimization
  problem, the `SDCAOptimizer` is used. For performance and convergence tuning,
  the num_loss_partitions parameter passed to `SDCAOptimizer` (see `__init__()`
  method), should be set to (#concurrent train ops per worker) x (#workers). If
  num_loss_partitions is larger or equal to this value, convergence is
  guaranteed but becomes slower as num_loss_partitions increases. If it is set
  to a smaller value, the optimizer is more agressive in reducing the global
  loss but convergence is not guaranteed. The recommended value in tf.learn
  (where there is one process per worker) is the number of workers running the
  train steps. It defaults to 1 (single machine).

  Example Usage:
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
               config=None):
    """Constructs a `SVM~ estimator object.

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

    Raises:
      ValueError: if kernels passed is not None.
    """
    if kernels is not None:
      raise ValueError("Kernel SVMs are not currently supported.")
    self._optimizer = sdca_optimizer.SDCAOptimizer(
        example_id_column=example_id_column,
        num_loss_partitions=num_loss_partitions,
        symmetric_l1_regularization=l1_regularization,
        symmetric_l2_regularization=l2_regularization)

    self._feature_columns = feature_columns
    self._model_dir = model_dir or tempfile.mkdtemp()
    self._estimator = estimator.Estimator(
        model_fn=linear.sdca_classifier_model_fn,
        model_dir=self._model_dir,
        config=config,
        params={
            "feature_columns": feature_columns,
            "optimizer": self._optimizer,
            "weight_column_name": weight_column_name,
            "loss_type": "hinge_loss",
        })

  def fit(self, x=None, y=None, input_fn=None, steps=None, batch_size=None,
          monitors=None, max_steps=None):
    """See trainable.Trainable."""
    return self._estimator.fit(x=x, y=y, input_fn=input_fn, steps=steps,
                               batch_size=batch_size, monitors=monitors,
                               max_steps=max_steps)

  # pylint: disable=protected-access
  def evaluate(self, x=None, y=None, input_fn=None, feed_fn=None,
               batch_size=None, steps=None, metrics=None, name=None):
    """See evaluable.Evaluable."""
    if not metrics:
      metrics = {}
      metrics["accuracy"] = metric_spec.MetricSpec(
          metric_fn=metrics_lib.streaming_accuracy,
          prediction_key=linear._CLASSES)
    additional_metrics = (
        target_column.get_default_binary_metrics_for_eval([0.5]))
    additional_metrics = {
        name: metric_spec.MetricSpec(metric_fn=metric,
                                     prediction_key=linear._LOGISTIC)
        for name, metric in additional_metrics.items()
    }
    metrics.update(additional_metrics)

    # TODO(b/31229024): Remove this loop
    for metric_name, metric in metrics.items():
      if isinstance(metric, metric_spec.MetricSpec):
        continue

      if isinstance(metric_name, tuple):
        if len(metric_name) != 2:
          raise ValueError("Ignoring metric %s. It returned a tuple with len  "
                           "%s, expected 2." % (metric_name, len(metric_name)))

        valid_keys = {linear._CLASSES, linear._LOGISTIC, linear._PROBABILITIES}
        if metric_name[1] not in valid_keys:
          raise ValueError("Ignoring metric %s. The 2nd element of its name "
                           "should be in %s" % (metric_name, valid_keys))
      metrics[metric_name] = linear._wrap_metric(metric)
    return self._estimator.evaluate(x=x, y=y, input_fn=input_fn,
                                    feed_fn=feed_fn, batch_size=batch_size,
                                    steps=steps, metrics=metrics, name=name)

  def predict(self, x=None, input_fn=None, batch_size=None, as_iterable=False):
    """Runs inference to determine the predicted class."""
    preds = self._estimator.predict(x=x, input_fn=input_fn,
                                    batch_size=batch_size,
                                    outputs=[linear._CLASSES],
                                    as_iterable=as_iterable)
    if as_iterable:
      return _as_iterable(preds, output=linear._CLASSES)
    return preds[linear._CLASSES]

  def predict_proba(self, x=None, input_fn=None, batch_size=None, outputs=None,
                    as_iterable=False):
    """Runs inference to determine the class probability predictions."""
    preds = self._estimator.predict(x=x, input_fn=input_fn,
                                    batch_size=batch_size,
                                    outputs=[linear._PROBABILITIES],
                                    as_iterable=as_iterable)
    if as_iterable:
      return _as_iterable(preds, output=linear._PROBABILITIES)
    return preds[linear._PROBABILITIES]
  # pylint: enable=protected-access

  def get_variable_names(self):
    return [name for name, _ in checkpoints.list_variables(self._model_dir)]

  def export(self, export_dir, signature_fn=None,
             input_fn=None, default_batch_size=1,
             exports_to_keep=None):
    """See BaseEstimator.export."""
    def default_input_fn(unused_estimator, examples):
      return layers.parse_feature_columns_from_examples(
          examples, self._feature_columns)
    return self._estimator.export(export_dir=export_dir,
                                  signature_fn=signature_fn,
                                  input_fn=input_fn or default_input_fn,
                                  default_batch_size=default_batch_size,
                                  exports_to_keep=exports_to_keep)

  @property
  def weights_(self):
    values = {}
    optimizer_regex = r".*/"+self._optimizer.get_name() + r"(_\d)?$"
    for name, _ in checkpoints.list_variables(self._model_dir):
      if (name.startswith("linear/") and
          name != "linear/bias_weight" and
          not re.match(optimizer_regex, name)):
        values[name] = checkpoints.load_variable(self._model_dir, name)
    if len(values) == 1:
      return values[list(values.keys())[0]]
    return values

  @property
  def bias_(self):
    return checkpoints.load_variable(self._model_dir,
                                     name="linear/bias_weight")
