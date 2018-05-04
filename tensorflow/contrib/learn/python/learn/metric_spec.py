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
"""The metric spec class to flexibly connect models and metrics (deprecated).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.deprecation import deprecated


def _assert_named_args(sentinel):
  if sentinel is not None:
    raise ValueError(
        '`metric_fn` requires named args: '
        '`labels`, `predictions`, and optionally `weights`.')


def _args(fn):
  """Get argument names for function-like object.

  Args:
    fn: Function, or function-like object (e.g., result of `functools.partial`).

  Returns:
    `tuple` of string argument names.
  """
  if hasattr(fn, 'func') and hasattr(fn, 'keywords'):
    # Handle functools.partial and similar objects.
    return tuple(
        [arg for arg in _args(fn.func) if arg not in set(fn.keywords.keys())])
  # Handle function.
  return tuple(tf_inspect.getargspec(fn).args)


_CANONICAL_LABELS_ARG = 'labels'
_LABELS_ARGS = set((_CANONICAL_LABELS_ARG, 'label', 'targets', 'target'))
_CANONICAL_PREDICTIONS_ARG = 'predictions'
_PREDICTIONS_ARGS = set((_CANONICAL_PREDICTIONS_ARG, 'prediction',
                         'logits', 'logit'))
_CANONICAL_WEIGHTS_ARG = 'weights'
_WEIGHTS_ARGS = set((_CANONICAL_WEIGHTS_ARG, 'weight'))


def _matching_arg(
    fn_name, fn_args, candidate_args, canonical_arg, is_required=False):
  """Find single argument in `args` from `candidate_args`.

  Args:
    fn_name: Function name, only used for error string.
    fn_args: String argument names to `fn_name` function.
    candidate_args: Candidate argument names to find in `args`.
    canonical_arg: Canonical argument name in `candidate_args`. This is only
      used to log a warning if a non-canonical match is found.
    is_required: Whether function is required to have an arg in
      `candidate_args`.

  Returns:
    String argument name if found, or `None` if not found.

  Raises:
    ValueError: if 2 candidates are found, or 0 are found and `is_required` is
      set.
  """
  assert canonical_arg in candidate_args  # Sanity check.
  matching_args = candidate_args.intersection(fn_args)
  if len(matching_args) > 1:
    raise ValueError(
        'Ambiguous arguments %s, must provide only one of %s.' % (
            matching_args, candidate_args))
  matching_arg = matching_args.pop() if matching_args else None
  if matching_arg:
    if matching_arg != canonical_arg:
      logging.warning(
          'Canonical arg %s missing from %s(%s), using %s.',
          canonical_arg, fn_name, fn_args, matching_arg)
  elif is_required:
    raise ValueError(
        '%s missing from %s(%s).' % (candidate_args, fn_name, fn_args))
  return matching_arg


def _fn_name(fn):
  if hasattr(fn, '__name__'):
    return fn.__name__
  if hasattr(fn, 'func') and hasattr(fn.func, '__name__'):
    return fn.func.__name__  # If it's a functools.partial.
  return str(fn)


def _adapt_metric_fn(
    metric_fn, metric_fn_name, is_labels_required, is_weights_required):
  """Adapt `metric_fn` to take only named args.

  This returns a function that takes only named args `labels`, `predictions`,
  and `weights`, and invokes `metric_fn` according to the following rules:
  - If `metric_fn` args include exactly one of `_LABELS_ARGS`, that arg is
    passed (usually by name, but positionally if both it and `predictions` need
    to be passed positionally). Otherwise, `labels` are omitted.
  - If `metric_fn` args include exactly one of `_PREDICTIONS_ARGS`, that arg is
    passed by name. Otherwise, `predictions` are passed positionally as the
    first non-label argument.
  - If exactly one of `_WEIGHTS_ARGS` is provided, that arg is passed by
    name.

  Args:
    metric_fn: Metric function to be wrapped.
    metric_fn_name: `metric_fn` name, only used for logging.
    is_labels_required: Whether `labels` is a required arg.
    is_weights_required: Whether `weights` is a required arg.

  Returns:
    Function accepting only named args `labels, `predictions`, and `weights`,
    and passing those to `metric_fn`.

  Raises:
    ValueError: if one of the following is true:
    - `metric_fn` has more than one arg of `_LABELS_ARGS`, `_PREDICTIONS_ARGS`,
      or `_WEIGHTS_ARGS`
    - `is_labels_required` is true, and `metric_fn` has no arg from
      `_LABELS_ARGS`.
    - `is_weights_required` is true, and `metric_fn` has no arg from
      `_WEIGHTS_ARGS`.
  """
  args = _args(metric_fn)

  labels_arg = _matching_arg(
      metric_fn_name, args, _LABELS_ARGS, _CANONICAL_LABELS_ARG,
      is_labels_required)
  predictions_arg = _matching_arg(
      metric_fn_name, args, _PREDICTIONS_ARGS, _CANONICAL_PREDICTIONS_ARG)
  weights_arg = _matching_arg(
      metric_fn_name, args, _WEIGHTS_ARGS, _CANONICAL_WEIGHTS_ARG,
      is_weights_required)

  # pylint: disable=invalid-name
  if labels_arg:
    if predictions_arg:
      # Both labels and predictions are named args.
      def _named_metric_fn(
          _sentinel=None, labels=None, predictions=None, weights=None):
        _assert_named_args(_sentinel)
        kwargs = {
            labels_arg: labels,
            predictions_arg: predictions,
        }
        if weights is not None:
          kwargs[weights_arg] = weights
        return metric_fn(**kwargs)
      return _named_metric_fn

    if labels_arg == args[0]:
      # labels is a named arg, and first. predictions is not a named arg, so we
      # want to pass it as the 2nd positional arg (i.e., the first non-labels
      # position), which means passing both positionally.
      def _positional_metric_fn(
          _sentinel=None, labels=None, predictions=None, weights=None):
        _assert_named_args(_sentinel)
        # TODO(ptucker): Should we support metrics that take only labels?
        # Currently, if you want streaming mean of a label, you have to wrap it
        # in a fn that takes discards predictions.
        if weights is None:
          return metric_fn(labels, predictions)
        return metric_fn(labels, predictions, **{weights_arg: weights})
      return _positional_metric_fn

    # labels is a named arg, and not first, so we pass predictions positionally
    # and labels by name.
    def _positional_predictions_metric_fn(
        _sentinel=None, labels=None, predictions=None, weights=None):
      _assert_named_args(_sentinel)
      kwargs = {
          labels_arg: labels,
      }
      if weights is not None:
        kwargs[weights_arg] = weights
      return metric_fn(predictions, **kwargs)
    return _positional_predictions_metric_fn

  if predictions_arg:
    # No labels, and predictions is named, so we pass the latter as a named arg.
    def _named_no_labels_metric_fn(
        _sentinel=None, labels=None, predictions=None, weights=None):
      del labels
      _assert_named_args(_sentinel)
      kwargs = {
          predictions_arg: predictions,
      }
      # TODO(ptucker): Should we allow weights with no labels?
      if weights is not None:
        kwargs[weights_arg] = weights
      return metric_fn(**kwargs)
    return _named_no_labels_metric_fn

  # Neither labels nor predictions are named, so we just pass predictions as the
  # first arg.
  def _positional_no_labels_metric_fn(
      _sentinel=None, labels=None, predictions=None, weights=None):
    del labels
    _assert_named_args(_sentinel)
    if weights is None:
      return metric_fn(predictions)
    # TODO(ptucker): Should we allow weights with no labels?
    return metric_fn(predictions, **{weights_arg: weights})
  return _positional_no_labels_metric_fn


class MetricSpec(object):
  """MetricSpec connects a model to metric functions.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  The MetricSpec class contains all information necessary to connect the
  output of a `model_fn` to the metrics (usually, streaming metrics) that are
  used in evaluation.

  It is passed in the `metrics` argument of `Estimator.evaluate`. The
  `Estimator` then knows which predictions, labels, and weight to use to call a
  given metric function.

  When building the ops to run in evaluation, an `Estimator` will call
  `create_metric_ops`, which will connect the given `metric_fn` to the model
  as detailed in the docstring for `create_metric_ops`, and return the metric.

  Example:

  Assuming a model has an input function which returns inputs containing
  (among other things) a tensor with key "input_key", and a labels dictionary
  containing "label_key". Let's assume that the `model_fn` for this model
  returns a prediction with key "prediction_key".

  In order to compute the accuracy of the "prediction_key" prediction, we
  would add

  ```
  "prediction accuracy": MetricSpec(metric_fn=prediction_accuracy_fn,
                                    prediction_key="prediction_key",
                                    label_key="label_key")
  ```

  to the metrics argument to `evaluate`. `prediction_accuracy_fn` can be either
  a predefined function in metric_ops (e.g., `streaming_accuracy`) or a custom
  function you define.

  If we would like the accuracy to be weighted by "input_key", we can add that
  as the `weight_key` argument.

  ```
  "prediction accuracy": MetricSpec(metric_fn=prediction_accuracy_fn,
                                    prediction_key="prediction_key",
                                    label_key="label_key",
                                    weight_key="input_key")
  ```

  An end-to-end example is as follows:

  ```
  estimator = tf.contrib.learn.Estimator(...)
  estimator.fit(...)
  _ = estimator.evaluate(
      input_fn=input_fn,
      steps=1,
      metrics={
          'prediction accuracy':
              metric_spec.MetricSpec(
                  metric_fn=prediction_accuracy_fn,
                  prediction_key="prediction_key",
                  label_key="label_key")
      })
  ```

  """

  @deprecated(None, 'Use tf.estimator.EstimatorSpec.eval_metric_ops.')
  def __init__(self,
               metric_fn,
               prediction_key=None,
               label_key=None,
               weight_key=None):
    """Constructor.

    Creates a MetricSpec.

    Args:
      metric_fn: A function to use as a metric. See `_adapt_metric_fn` for
        rules on how `predictions`, `labels`, and `weights` are passed to this
        function. This must return either a single `Tensor`, which is
        interpreted as a value of this metric, or a pair
        `(value_op, update_op)`, where `value_op` is the op to call to
        obtain the value of the metric, and `update_op` should be run for
        each batch to update internal state.
      prediction_key: The key for a tensor in the `predictions` dict (output
        from the `model_fn`) to use as the `predictions` input to the
        `metric_fn`. Optional. If `None`, the `model_fn` must return a single
        tensor or a dict with only a single entry as `predictions`.
      label_key: The key for a tensor in the `labels` dict (output from the
        `input_fn`) to use as the `labels` input to the `metric_fn`.
        Optional. If `None`, the `input_fn` must return a single tensor or a
        dict with only a single entry as `labels`.
      weight_key: The key for a tensor in the `inputs` dict (output from the
        `input_fn`) to use as the `weights` input to the `metric_fn`.
        Optional. If `None`, no weights will be passed to the `metric_fn`.
    """
    self._metric_fn_name = _fn_name(metric_fn)
    self._metric_fn = _adapt_metric_fn(
        metric_fn=metric_fn,
        metric_fn_name=self._metric_fn_name,
        is_labels_required=label_key is not None,
        is_weights_required=weight_key is not None)
    self._prediction_key = prediction_key
    self._label_key = label_key
    self._weight_key = weight_key

  @property
  def prediction_key(self):
    return self._prediction_key

  @property
  def label_key(self):
    return self._label_key

  @property
  def weight_key(self):
    return self._weight_key

  @property
  def metric_fn(self):
    """Metric function.

    This function accepts named args: `predictions`, `labels`, `weights`. It
    returns a single `Tensor` or `(value_op, update_op)` pair. See `metric_fn`
    constructor argument for more details.

    Returns:
      Function, see `metric_fn` constructor argument for more details.
    """
    return self._metric_fn

  def __str__(self):
    return ('MetricSpec(metric_fn=%s, ' % self._metric_fn_name +
            'prediction_key=%s, ' % self.prediction_key +
            'label_key=%s, ' % self.label_key +
            'weight_key=%s)' % self.weight_key
           )

  def create_metric_ops(self, inputs, labels, predictions):
    """Connect our `metric_fn` to the specified members of the given dicts.

    This function will call the `metric_fn` given in our constructor as follows:

    ```
      metric_fn(predictions[self.prediction_key],
                labels[self.label_key],
                weights=weights[self.weight_key])
    ```

    And returns the result. The `weights` argument is only passed if
    `self.weight_key` is not `None`.

    `predictions` and `labels` may be single tensors as well as dicts. If
    `predictions` is a single tensor, `self.prediction_key` must be `None`. If
    `predictions` is a single element dict, `self.prediction_key` is allowed to
    be `None`. Conversely, if `labels` is a single tensor, `self.label_key` must
    be `None`. If `labels` is a single element dict, `self.label_key` is allowed
    to be `None`.

    Args:
      inputs: A dict of inputs produced by the `input_fn`
      labels: A dict of labels or a single label tensor produced by the
        `input_fn`.
      predictions: A dict of predictions or a single tensor produced by the
        `model_fn`.

    Returns:
      The result of calling `metric_fn`.

    Raises:
      ValueError: If `predictions` or `labels` is a single `Tensor` and
        `self.prediction_key` or `self.label_key` is not `None`; or if
        `self.label_key` is `None` but `labels` is a dict with more than one
        element, or if `self.prediction_key` is `None` but `predictions` is a
        dict with more than one element.
    """
    def _get_dict(name, dict_or_tensor, key):
      """Get a single tensor or an element of a dict or raise ValueError."""
      if key:
        if not isinstance(dict_or_tensor, dict):
          raise ValueError('MetricSpec with ' + name + '_key specified'
                           ' requires ' +
                           name + 's dict, got %s.\n' % dict_or_tensor +
                           'You must not provide a %s_key if you ' % name +
                           'only have a single Tensor as %ss.' % name)
        if key not in dict_or_tensor:
          raise KeyError(
              'Key \'%s\' missing from %s.' % (key, dict_or_tensor.keys()))
        return dict_or_tensor[key]
      else:
        if isinstance(dict_or_tensor, dict):
          if len(dict_or_tensor) != 1:
            raise ValueError('MetricSpec without specified ' + name + '_key'
                             ' requires ' + name + 's tensor or single element'
                             ' dict, got %s' % dict_or_tensor)
          return six.next(six.itervalues(dict_or_tensor))
        return dict_or_tensor

    # Get the predictions.
    prediction = _get_dict('prediction', predictions, self.prediction_key)

    # Get the labels.
    label = _get_dict('label', labels, self.label_key)

    try:
      return self.metric_fn(
          labels=label,
          predictions=prediction,
          weights=inputs[self.weight_key] if self.weight_key else None)
    except Exception as ex:
      logging.error('Could not create metric ops for %s, %s.' % (self, ex))
      raise
