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

from tensorflow.python.estimator.export import export_output as export_output_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.training import monitored_session
from tensorflow.python.training import session_run_hook
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import estimator_export


@estimator_export('estimator.ModeKeys')
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


LOSS_METRIC_KEY = 'loss'
AVERAGE_LOSS_METRIC_KEY = 'average_loss'

# Mapping of the modes to appropriate tag_constants that are used for saving.
EXPORT_TAG_MAP = {
    ModeKeys.PREDICT: [tag_constants.SERVING],
    ModeKeys.TRAIN: [tag_constants.TRAINING],
    ModeKeys.EVAL: [tag_constants.EVAL],
}


@estimator_export('estimator.EstimatorSpec')
class EstimatorSpec(
    collections.namedtuple('EstimatorSpec', [
        'mode', 'predictions', 'loss', 'train_op', 'eval_metric_ops',
        'export_outputs', 'training_chief_hooks', 'training_hooks', 'scaffold',
        'evaluation_hooks', 'prediction_hooks'
    ])):
  """Ops and objects returned from a `model_fn` and passed to an `Estimator`.

  `EstimatorSpec` fully defines the model to be run by an `Estimator`.
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
              scaffold=None,
              evaluation_hooks=None,
              prediction_hooks=None):
    """Creates a validated `EstimatorSpec` instance.

    Depending on the value of `mode`, different arguments are required. Namely

    * For `mode == ModeKeys.TRAIN`: required fields are `loss` and `train_op`.
    * For `mode == ModeKeys.EVAL`: required field is `loss`.
    * For `mode == ModeKeys.PREDICT`: required fields are `predictions`.

    model_fn can populate all arguments independent of mode. In this case, some
    arguments will be ignored by an `Estimator`. E.g. `train_op` will be
    ignored in eval and infer modes. Example:

    ```python
    def my_model_fn(features, labels, mode):
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
    def my_model_fn(features, labels, mode):
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
        `(metric_tensor, update_op)` tuple. `metric_tensor` should be evaluated
        without any impact on state (typically is a pure computation results
        based on variables.). For example, it should not trigger the `update_op`
        or requires any input fetching.
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
        If no entry is provided, a default `PredictOutput` mapping to
        `predictions` will be created.
      training_chief_hooks: Iterable of `tf.train.SessionRunHook` objects to
        run on the chief worker during training.
      training_hooks: Iterable of `tf.train.SessionRunHook` objects to run
        on all workers during training.
      scaffold: A `tf.train.Scaffold` object that can be used to set
        initialization, saver, and more to be used in training.
      evaluation_hooks: Iterable of `tf.train.SessionRunHook` objects to
        run during evaluation.
      prediction_hooks: Iterable of `tf.train.SessionRunHook` objects to
        run during predictions.

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

    # Validate the passed export outputs, or generate defaults.
    if mode == ModeKeys.PREDICT:
      export_outputs = _get_export_outputs(export_outputs, predictions)

    # Validate that all tensors and ops are from the default graph.
    default_graph = ops.get_default_graph()

    # We enumerate possible error causes here to aid in debugging.
    error_message_template = (
        '{0} with "{1}" must be from the default graph. '
        'Possible causes of this error include: \n\n'
        '1) {0} was created outside the context of the default graph.'
        '\n\n'
        '2) The object passed through to EstimatorSpec was not created '
        'in the most recent call to "model_fn".')

    if isinstance(predictions, dict):
      for key, value in six.iteritems(predictions):
        if value.graph is not default_graph:
          raise ValueError(error_message_template.format(
              'prediction values',
              '{0}: {1}'.format(key, value.name)))
    elif predictions is not None:
      # 'predictions' must be a single Tensor.
      if predictions.graph is not default_graph:
        raise ValueError(error_message_template.format(
            'prediction values', predictions.name))

    if loss is not None and loss.graph is not default_graph:
      raise ValueError(error_message_template.format('loss', loss.name))
    if train_op is not None and train_op.graph is not default_graph:
      raise ValueError(error_message_template.format('train_op', train_op.name))
    for key, value in list(six.iteritems(eval_metric_ops)):
      values = nest.flatten(value)
      for val in values:
        if val.graph is not default_graph:
          raise ValueError(error_message_template.format(
              'eval_metric_ops',
              '{0}: {1}'.format(key, val.name)))

    # Validate hooks.
    training_chief_hooks = tuple(training_chief_hooks or [])
    training_hooks = tuple(training_hooks or [])
    evaluation_hooks = tuple(evaluation_hooks or [])
    prediction_hooks = tuple(prediction_hooks or [])

    for hook in (training_hooks + training_chief_hooks + evaluation_hooks +
                 prediction_hooks):
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
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        export_outputs=export_outputs,
        training_chief_hooks=training_chief_hooks,
        training_hooks=training_hooks,
        scaffold=scaffold,
        evaluation_hooks=evaluation_hooks,
        prediction_hooks=prediction_hooks)

  def _replace(self, **kwds):
    """Return a new EstimatorSpec replacing specified fields with new values."""
    if 'mode' in kwds:
      if self.mode != kwds['mode']:
        raise ValueError('mode of EstimatorSpec cannot be changed.')
    new_fields = map(kwds.pop, self._fields, list(self))
    return EstimatorSpec(*new_fields)


def _get_export_outputs(export_outputs, predictions):
  """Validate export_outputs or create default export_outputs.

  Args:
    export_outputs: Describes the output signatures to be exported to
      `SavedModel` and used during serving. Should be a dict or None.
    predictions:  Predictions `Tensor` or dict of `Tensor`.

  Returns:
    Valid export_outputs dict

  Raises:
    TypeError: if export_outputs is not a dict or its values are not
      ExportOutput instances.
  """
  if export_outputs is None:
    default_output = export_output_lib.PredictOutput(predictions)
    export_outputs = {
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: default_output}

  if not isinstance(export_outputs, dict):
    raise TypeError('export_outputs must be dict, given: {}'.format(
        export_outputs))
  for v in six.itervalues(export_outputs):
    if not isinstance(v, export_output_lib.ExportOutput):
      raise TypeError(
          'Values in export_outputs must be ExportOutput objects. '
          'Given: {}'.format(export_outputs))

  _maybe_add_default_serving_output(export_outputs)

  return export_outputs


def _maybe_add_default_serving_output(export_outputs):
  """Add a default serving output to the export_outputs if not present.

  Args:
    export_outputs: Describes the output signatures to be exported to
      `SavedModel` and used during serving. Should be a dict.

  Returns:
    export_outputs dict with default serving signature added if necessary

  Raises:
    ValueError: if multiple export_outputs were provided without a default
      serving key.
  """
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

  return export_outputs


class _TPUEstimatorSpec(collections.namedtuple('TPUEstimatorSpec', [
    'mode',
    'predictions',
    'loss',
    'train_op',
    'eval_metrics',
    'export_outputs',
    'scaffold_fn',
    'host_call'])):
  """Ops and objects returned from a `model_fn` and passed to `TPUEstimator`.

  This is a simplified implementation of `tf.contrib.tpu.EstimatorSpec`. See
  tensorflow/contrib/tpu/python/tpu/tpu_estimator.py for more detailed
  documentation.
  """

  def __new__(cls,
              mode,
              predictions=None,
              loss=None,
              train_op=None,
              eval_metrics=None,
              export_outputs=None,
              scaffold_fn=None,
              host_call=None):
    """Creates a `_TPUEstimatorSpec` instance."""
    return super(_TPUEstimatorSpec, cls).__new__(cls,
                                                 mode=mode,
                                                 predictions=predictions,
                                                 loss=loss,
                                                 train_op=train_op,
                                                 eval_metrics=eval_metrics,
                                                 export_outputs=export_outputs,
                                                 scaffold_fn=scaffold_fn,
                                                 host_call=host_call)

  def as_estimator_spec(self):
    """Creates an equivalent `EstimatorSpec` used by CPU train/eval."""
    if not self.eval_metrics:
      eval_metric_ops = None
    else:
      metric_fn, tensors = self.eval_metrics
      eval_metric_ops = metric_fn(**tensors)
    return EstimatorSpec(mode=self.mode,
                         predictions=self.predictions,
                         loss=self.loss,
                         train_op=self.train_op,
                         eval_metric_ops=eval_metric_ops,
                         export_outputs=self.export_outputs)


def _check_is_tensor_or_operation(x, name):
  if not (isinstance(x, ops.Operation) or isinstance(x, ops.Tensor)):
    raise TypeError('{} must be Operation or Tensor, given: {}'.format(name, x))


def _check_is_tensor(x, tensor_name):
  """Returns `x` if it is a `Tensor`, raises TypeError otherwise."""
  if not isinstance(x, ops.Tensor):
    raise TypeError('{} must be Tensor, given: {}'.format(tensor_name, x))
  return x
