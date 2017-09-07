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
"""Extenders of tf.estimator.Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.estimator import estimator as estimator_lib
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator import util as estimator_util
from tensorflow.python.util import tf_inspect

_VALID_METRIC_FN_ARGS = set(['features', 'labels', 'predictions', 'config'])


def add_metrics(estimator, metric_fn):
  """Creates new ${tf.estimator.Estimator} which has given metrics.

  Example:

  ```python
    def my_auc(labels, predictions):
      return {'auc': tf.metrics.auc(labels, predictions['logistic'])}

    estimator = tf.estimator.DNNClassifier(...)
    estimator = tf.contrib.estimator.add_metrics(estimator, my_auc)
    estimator.train(...)
    estimator.evaluate(...)
  ```
  Example usage of custom metric which uses features:

  ```python
    def my_auc(features, labels, predictions):
      return {'auc': tf.metrics.auc(
        labels, predictions['logistic'], weights=features['weight'])}

    estimator = tf.estimator.DNNClassifier(...)
    estimator = tf.contrib.estimator.add_metrics(estimator, my_auc)
    estimator.train(...)
    estimator.evaluate(...)
  ```

  Args:
    estimator: A ${tf.estimator.Estimator} object.
    metric_fn: A function which should obey the following signature:
      - Args: can only have following four arguments in any order:
        * predictions: Predictions `Tensor` or dict of `Tensor` created by given
          `estimator`.
        * features: Input `dict` of `Tensor` objects created by `input_fn` which
          is given to `estimator.evaluate` as an argument.
        * labels:  Labels `Tensor` or dict of `Tensor` created by `input_fn`
          which is given to `estimator.evaluate` as an argument.
        * config: config attribute of the `estimator`.
       - Returns:
         Dict of metric results keyed by name. Final metrics are a union of this
         and `estimator's` existing metrics. If there is a name conflict between
         this and `estimator`s existing metrics, this will override the existing
         one. The values of the dict are the results of calling a metric
         function, namely a `(metric_tensor, update_op)` tuple.

  Returns:
      A new ${tf.estimator.Estimator} which has a union of original metrics with
        given ones.
  """
  _verify_metric_fn_args(metric_fn)

  def new_model_fn(features, labels, mode):
    spec = _get_model_fn(estimator)(features, labels, mode)
    if mode != model_fn_lib.ModeKeys.EVAL:
      return spec
    new_metrics = _call_metric_fn(metric_fn, features, labels, spec.predictions,
                                  estimator.config)
    all_metrics = spec.eval_metric_ops or {}
    all_metrics.update(new_metrics)
    return spec._replace(eval_metric_ops=all_metrics)

  return estimator_lib.Estimator(
      model_fn=new_model_fn,
      model_dir=estimator.model_dir,
      config=estimator.config)


# TODO(ispir): Move this to tf.estimator.Estimator.
def _get_model_fn(estimator):
  return estimator._call_model_fn  # pylint: disable=protected-access


def _verify_metric_fn_args(metric_fn):
  args = set(estimator_util.fn_args(metric_fn))
  if tf_inspect.ismethod(metric_fn):
    if 'self' in args:
      args.remove('self')
  invalid_args = list(args - _VALID_METRIC_FN_ARGS)
  if invalid_args:
    raise ValueError('metric_fn (%s) has following not expected args: %s' %
                     (metric_fn, invalid_args))


def _call_metric_fn(metric_fn, features, labels, predictions, config):
  """Calls metric fn with proper arguments."""
  metric_fn_args = estimator_util.fn_args(metric_fn)
  kwargs = {}
  if 'features' in metric_fn_args:
    kwargs['features'] = features
  if 'labels' in metric_fn_args:
    kwargs['labels'] = labels
  if 'predictions' in metric_fn_args:
    kwargs['predictions'] = predictions
  if 'config' in metric_fn_args:
    kwargs['config'] = config
  return metric_fn(**kwargs)
