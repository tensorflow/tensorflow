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

"""Classes and methods related to model_fn."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import six

from tensorflow.python.estimator.export.export_output import ExportOutput
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.training import monitored_session
from tensorflow.python.training import session_run_hook
from tensorflow.python.util import nest


class ModeKeys(object):
  """Standard names for model modes.

  The following standard keys are defined:

  * `TRAIN`: training mode.
  * `EVAL`: evaluation mode.
  * `PREDICT`: inference mode.
  """

  TRAIN = 'train'
  EVAL = 'eval'
  PREDICT = 'infer'


class MetricKeys(object):
  """Metric key strings."""
  LOSS = 'loss'
  AVERAGE_LOSS = 'average_loss'


class EstimatorSpec(
    collections.namedtuple('EstimatorSpec', [
        'predictions', 'loss', 'train_op', 'eval_metric_ops',
        'export_outputs', 'training_chief_hooks', 'training_hooks',
        'scaffold'
    ])):
  """Ops and objects returned from a `model_fn` and passed to `Estimator`.

  `EstimatorSpec` fully defines the model to be run by `Estimator`.
  """

  def __new__(cls,
              mode,
              predictions=None,
              loss=None,
              train_op=None,
              eval_metric_ops=None,
              export_outputs=None,
              training_chief_hooks=None,
              training_hooks=None,
              scaffold=None):
    """Creates a validated `EstimatorSpec` instance.

    Depending on the value of `mode`, different arguments are required. Namely
    * For `mode == ModeKeys.TRAIN`: required fields are `loss` and `train_op`.
    * For `mode == ModeKeys.EVAL`: required field is`loss`.
    * For `mode == ModeKeys.PREDICT`: required fields are `predictions`.

    model_fn can populate all arguments independent of mode. In this case, some
    arguments will be ignored by `Estimator`. E.g. `train_op` will be ignored
    in eval and infer modes. Example:

    ```python
    def my_model_fn(mode, features, labels):
      predictions = ...
      loss = ...
      train_op = ...
      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=predictions,
          loss=loss,
          train_op=train_op)
    ```

    Alternatively, model_fn can just populate the arguments appropriate to the
    given mode. Example:

    ```python
    def my_model_fn(mode, features, labels):
      if (mode == tf.estimator.ModeKeys.TRAIN or
          mode == tf.estimator.ModeKeys.EVAL):
        loss = ...
      else:
        loss = None
      if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = ...
      else:
        train_op = None
      if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = ...
      else:
        predictions = None

      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=predictions,
          loss=loss,
          train_op=train_op)
    ```

    Args:
      mode: A `ModeKeys`. Specifies if this is training, evaluation or
        prediction.
      predictions: Predictions `Tensor` or dict of `Tensor`.
      loss: Training loss `Tensor`. Must be either scalar, or with shape `[1]`.
      train_op: Op for the training step.
      eval_metric_ops: Dict of metric results keyed by name. The values of the
        dict are the results of calling a metric function, namely a
        `(metric_tensor, update_op)` tuple.
      export_outputs: Describes the output signatures to be exported to
        `SavedModel` and used during serving.
        A dict `{name: output}` where:
        * name: An arbitrary name for this output.
        * output: an `ExportOutput` object such as `ClassificationOutput`,
            `RegressionOutput`, or `PredictOutput`.
        Single-headed models only need to specify one entry in this dictionary.
        Multi-headed models should specify one entry for each head, one of
        which must be named using
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY.
      training_chief_hooks: Iterable of `tf.train.SessionRunHook` objects to
        run on the chief worker during training.
      training_hooks: Iterable of `tf.train.SessionRunHook` objects that to run
        on all workers during training.
      scaffold: A `tf.train.Scaffold` object that can be used to set
        initialization, saver, and more to be used in training.

    Returns:
      A validated `EstimatorSpec` object.

    Raises:
      ValueError: If validation fails.
      TypeError: If any of the arguments is not the expected type.
    """
    # Validate train_op.
    if train_op is None:
      if mode == ModeKeys.TRAIN:
        raise ValueError('Missing train_op.')
    else:
      _check_is_tensor_or_operation(train_op, 'train_op')

    # Validate loss.
    if loss is None:
      if mode in (ModeKeys.TRAIN, ModeKeys.EVAL):
        raise ValueError('Missing loss.')
    else:
      loss = _check_is_tensor(loss, 'loss')
      loss_shape = loss.get_shape()
      if loss_shape.num_elements() not in (None, 1):
        raise ValueError('Loss must be scalar, given: {}'.format(loss))
      if not loss_shape.is_compatible_with(tensor_shape.scalar()):
        loss = array_ops.reshape(loss, [])

    # Validate predictions.
    if predictions is None:
      if mode == ModeKeys.PREDICT:
        raise ValueError('Missing predictions.')
      predictions = {}
    else:
      if isinstance(predictions, dict):
        predictions = {
            k: _check_is_tensor(v, 'predictions[{}]'.format(k))
            for k, v in six.iteritems(predictions)
        }
      else:
        predictions = _check_is_tensor(predictions, 'predictions')

    # Validate eval_metric_ops.
    if eval_metric_ops is None:
      eval_metric_ops = {}
    else:
      if not isinstance(eval_metric_ops, dict):
        raise TypeError(
            'eval_metric_ops must be a dict, given: {}'.format(eval_metric_ops))
      for key, metric_value_and_update in six.iteritems(eval_metric_ops):
        if (not isinstance(metric_value_and_update, tuple) or
            len(metric_value_and_update) != 2):
          raise TypeError(
              'Values of eval_metric_ops must be (metric_value, update_op) '
              'tuples, given: {} for key: {}'.format(
                  metric_value_and_update, key))
        metric_value, metric_update = metric_value_and_update
        for metric_value_member in nest.flatten(metric_value):
          # Allow (possibly nested) tuples for metric values, but require that
          # each of them be Tensors or Operations.
          _check_is_tensor_or_operation(metric_value_member,
                                        'eval_metric_ops[{}]'.format(key))
        _check_is_tensor_or_operation(metric_update,
                                      'eval_metric_ops[{}]'.format(key))

    # Validate export_outputs.
    if export_outputs is not None:
      if not isinstance(export_outputs, dict):
        raise TypeError('export_outputs must be dict, given: {}'.format(
            export_outputs))
      for v in six.itervalues(export_outputs):
        if not isinstance(v, ExportOutput):
          raise TypeError(
              'Values in export_outputs must be ExportOutput objects. '
              'Given: {}'.format(export_outputs))
      # Note export_outputs is allowed to be empty.
      if len(export_outputs) == 1:
        (key, value), = export_outputs.items()
        if key != signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
          export_outputs[
              signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = value
      if len(export_outputs) > 1:
        if (signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
            not in export_outputs):
          raise ValueError(
              'Multiple export_outputs were provided, but none of them is '
              'specified as the default.  Do this by naming one of them with '
              'signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY.')

    # Validate that all tensors and ops are from the default graph.
    default_graph = ops.get_default_graph()
    for value in _prediction_values(predictions):
      if value.graph is not default_graph:
        raise ValueError('prediction values must be from the default graph.')
    if loss is not None and loss.graph is not default_graph:
      raise ValueError('loss must be from the default graph.')
    if train_op is not None and train_op.graph is not default_graph:
      raise ValueError('train_op must be from the default graph.')
    for value in nest.flatten(list(eval_metric_ops.values())):
      if value.graph is not default_graph:
        raise ValueError(
            'eval_metric_ops values must be from the default graph.')

    # Validate hooks.
    training_chief_hooks = tuple(training_chief_hooks or [])
    training_hooks = tuple(training_hooks or [])
    for hook in training_hooks + training_chief_hooks:
      if not isinstance(hook, session_run_hook.SessionRunHook):
        raise TypeError(
            'All hooks must be SessionRunHook instances, given: {}'.format(
                hook))

    scaffold = scaffold or monitored_session.Scaffold()
    # Validate scaffold.
    if not isinstance(scaffold, monitored_session.Scaffold):
      raise TypeError(
          'scaffold must be tf.train.Scaffold. Given: {}'.format(scaffold))

    return super(EstimatorSpec, cls).__new__(
        cls,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        export_outputs=export_outputs,
        training_chief_hooks=training_chief_hooks,
        training_hooks=training_hooks,
        scaffold=scaffold)


def _check_is_tensor_or_operation(x, name):
  if not (isinstance(x, ops.Operation) or isinstance(x, ops.Tensor)):
    raise TypeError('{} must be Operation or Tensor, given: {}'.format(name, x))


def _check_is_tensor(x, tensor_name):
  """Returns `x` if it is a `Tensor`, raises TypeError otherwise."""
  if not isinstance(x, ops.Tensor):
    raise TypeError('{} must be Tensor, given: {}'.format(tensor_name, x))
  return x


def _prediction_values(predictions):
  """Returns the values of the given predictions dict or `Tensor`."""
  if predictions is None:
    return []
  if isinstance(predictions, dict):
    return list(six.itervalues(predictions))
  return [predictions]
