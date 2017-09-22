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
from tensorflow.python.ops import clip_ops
from tensorflow.python.training import optimizer as optimizer_lib
from tensorflow.python.util import tf_inspect

_VALID_METRIC_FN_ARGS = set(['features', 'labels', 'predictions', 'config'])


def add_metrics(estimator, metric_fn):
  """Creates a new ${tf.estimator.Estimator} which has given metrics.

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

  def new_model_fn(features, labels, mode, config):
    spec = estimator.model_fn(features, labels, mode, config)
    if mode != model_fn_lib.ModeKeys.EVAL:
      return spec
    new_metrics = _call_metric_fn(metric_fn, features, labels, spec.predictions,
                                  config)
    all_metrics = spec.eval_metric_ops or {}
    all_metrics.update(new_metrics)
    return spec._replace(eval_metric_ops=all_metrics)

  return estimator_lib.Estimator(
      model_fn=new_model_fn,
      model_dir=estimator.model_dir,
      config=estimator.config)


def clip_gradients_by_norm(optimizer, clip_norm):
  """Returns an optimizer which clips gradients before appliying them.

  Example:

  ```python
  optimizer = tf.train.ProximalAdagradOptimizer(
      learning_rate=0.1,
      l1_regularization_strength=0.001)
  optimizer = tf.contrib.estimator.clip_gradients_by_norm(
      optimizer, clip_norm)
  estimator = tf.estimator.DNNClassifier(
      feature_columns=[...],
      hidden_units=[1024, 512, 256],
      optimizer=optimizer)
  ```

  Args:
    optimizer: An `tf.Optimizer` object to apply gradients.
    clip_norm: A 0-D (scalar) `Tensor` > 0. The clipping ratio.

  Returns:
    A `tf.Optimizer`.
  """

  def clip_grads(grads_and_vars):
    gradients, variables = zip(*grads_and_vars)
    gradients = clip_ops.clip_by_global_norm(gradients, clip_norm)[0]
    grads_and_vars = list(zip(gradients, variables))
    return grads_and_vars

  return _TransformGradients(
      optimizer=optimizer,
      transform_grads_fn=clip_grads,
      name='ClipByNorm' + optimizer.get_name())


class _TransformGradients(optimizer_lib.Optimizer):
  """Add given gradient transformation to the optimizer."""

  def __init__(self, optimizer, transform_grads_fn, name=None):
    """Construct an `tf.Optimizer` wrapper to apply given transformations.

    Example:

    ```python
    optimizer = tf.train.ProximalAdagradOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.001)
    def clip_grads(grads_and_vars):
      gradients, variables = zip(*grads_and_vars)
      gradients = tf.clip_by_global_norm(grads, my_norm)[0]
      grads_and_vars = list(zip(gradients, variables))
      return grads_and_vars
    optimizer = _TransformGradients(
        opt=optimizer, transform_grads_fn=clip_grads)
    estimator = tf.estimator.DNNClassifier(
        feature_columns=[...],
        hidden_units=[1024, 512, 256],
        optimizer=optimizer)
    ```

    Args:
      optimizer: An `tf.Optimizer` object to apply gradients.
      transform_grads_fn: A function which takes a single argument, a list of
        gradient to variable pairs (tuples), performs any requested gradient
        updates, such as gradient clipping or multipliers, and returns the
        updated list.
      name: A string which will be used for debugging purposes.
    """
    super(_TransformGradients, self).__init__(
        use_locking=False, name=name or optimizer.get_name())
    self._optimizer = optimizer
    self._transform_grads_fn = transform_grads_fn

  def compute_gradients(self, *args, **kwargs):
    """See `tf.Optimizer`."""
    return self._optimizer.compute_gradients(*args, **kwargs)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Apply gradients to variables.

    Calls `transform_grads_fn`, and then applies the real optimizer.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        compute_gradients().
      global_step: Optional Variable to increment by one after the
        variables have been updated.
      name: Optional name for the returned operation.  Default to the
        name passed to the Optimizer constructor.

    Returns:
      An `Operation` that applies the gradients. If `global_step` was not None,
      that operation also increments `global_step`.

    Raises:
      ValueError: If the grads_and_vars is malformed.
    """
    grads_and_vars = self._transform_grads_fn(grads_and_vars)
    return self._optimizer.apply_gradients(grads_and_vars, global_step, name)

  def get_slot(self, *args, **kwargs):
    """See `tf.Optimizer`."""
    return self._optimizer.get_slot(*args, **kwargs)

  def get_slot_names(self, *args, **kwargs):
    """See `tf.Optimizer`."""
    return self._optimizer.get_slot_names(*args, **kwargs)


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
