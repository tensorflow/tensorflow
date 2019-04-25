# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""xla is an experimental library that provides XLA support APIs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.compiler.xla import xla
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_decorator


compile = xla.compile  # pylint: disable=redefined-builtin
check_function_argument_count = xla.check_function_argument_count

class _CapturedObject(object):
  """A placeholder to capture an object."""

  def __init__(self):
    self._object = None

  def capture(self, o):
    if self._object:
      raise RuntimeError(
          'InternalError: _CapturedObject can capture only once. Please file '
          'bug.')

    self._object = o

  def get(self):
    return self._object


def _get_scaffold(captured_scaffold_fn):
  """Retrieves the Scaffold from `captured_scaffold_fn`."""
  scaffold_fn = captured_scaffold_fn.get()

  if not scaffold_fn:
    return None

  scaffold = scaffold_fn()
  if scaffold is None:
    raise ValueError(
        'TPUEstimatorSpec.scaffold_fn returns None, which is not allowed')

  return scaffold


class _ModelFnWrapper(object):
  """_ModelFnWrapper supports executing model_fn with XLA."""

  def __init__(self, function):
    self._model_fn = function

  def __call__(self, features, labels, mode, params):

    # TPUEstimator compiles model_fn when use_tpu=True. To avoid double
    # compilation, we use this params['use_tpu'] as a hint. When it is set to
    # True, model_fn is called without compilation.
    # Note that this condition isn't accurate for the case of exporting a model.
    # In that case we should ideally not compile so that user can see detailed
    # graph. However, we don't have enough information to tell whether model_fn
    # is being called for export mode or not.
    # TODO(ycao): Make this condition more accurate when implementing PREDICT
    # mode.
    if params.get('use_tpu'):
      return self._call_model_fn(features, labels, mode, params)

    if mode == model_fn_lib.ModeKeys.TRAIN:
      train_step, captured_scaffold_fn = self._make_train_step(
          features, labels, params)
      (loss,) = compile(train_step)
      return model_fn_lib.EstimatorSpec(
          mode=mode,
          loss=loss,
          train_op=array_ops.identity(loss),
          scaffold=_get_scaffold(captured_scaffold_fn))
    elif mode == model_fn_lib.ModeKeys.EVAL:
      eval_step, captured_eval_metric_fn, captured_scaffold_fn = (
          self._make_eval_step(features, labels, params))
      outputs = compile(eval_step)
      loss = outputs[0]

      # Calculate eval_metric_ops if eval_metric_fn is set and captured.
      eval_metric_fn = captured_eval_metric_fn.get()
      if eval_metric_fn:
        eval_metric_fn_tensors = outputs[1:]
        eval_metric_ops = eval_metric_fn(*eval_metric_fn_tensors)
      else:
        eval_metric_ops = None

      return model_fn_lib.EstimatorSpec(
          mode=mode,
          loss=loss,
          eval_metric_ops=eval_metric_ops,
          scaffold=_get_scaffold(captured_scaffold_fn))
    else:
      raise NotImplementedError('%s is not implemented, only TRAIN and EVAL are'
                                ' supported' % mode)

  def _make_train_step(self, features, labels, params):
    """Creates a single step of training for xla.compile()."""
    captured_scaffold_fn = _CapturedObject()

    def train_step():
      """A single step of training."""
      estimator_spec = self._call_model_fn(features, labels,
                                           model_fn_lib.ModeKeys.TRAIN, params)

      try:
        captured_scaffold_fn.capture(estimator_spec.scaffold_fn)
      except AttributeError:
        captured_scaffold_fn.capture(None)

      # train_step will be run by xla.compile(). xla.compile() only supports
      # tensor output while train_op can be either an operation or a tensor.
      # Even though xla.compile() automatically adds operation-typed train_op as
      # control dependency of other tensor outputs, it doesn't do so for
      # tensor-typed train_op. Thus, we need to set it explicitly here.
      with ops.control_dependencies([estimator_spec.train_op]):
        return array_ops.identity(estimator_spec.loss)

    return train_step, captured_scaffold_fn

  def _make_eval_step(self, features, labels, params):
    """Creates a single step of evaluation for xla.compile()."""
    captured_eval_metric_fn = _CapturedObject()
    captured_scaffold_fn = _CapturedObject()

    def eval_step():
      """A single step of evaluation."""
      estimator_spec = self._call_model_fn(features, labels,
                                           model_fn_lib.ModeKeys.EVAL, params)

      try:
        captured_scaffold_fn.capture(estimator_spec.scaffold_fn)
      except AttributeError:
        captured_scaffold_fn.capture(None)

      eval_metric_fn = None
      eval_metric_fn_tensors = []
      try:
        if estimator_spec.eval_metrics:
          (eval_metric_fn, eval_metric_fn_tensors) = estimator_spec.eval_metrics
      except AttributeError:
        pass

      # If a dictionary is provided, we need to convert it into a list sorted
      # according to order of eval_metric_fn positional arguments.
      if isinstance(eval_metric_fn_tensors, dict):
        eval_metric_fn_args = function_utils.fn_args(eval_metric_fn)
        eval_metric_fn_tensors = [
            eval_metric_fn_tensors[i] for i in eval_metric_fn_args
        ]

      captured_eval_metric_fn.capture(eval_metric_fn)

      return tuple([estimator_spec.loss] + eval_metric_fn_tensors)

    return eval_step, captured_eval_metric_fn, captured_scaffold_fn

  def _call_model_fn(self, features, labels, mode, params):
    """Calls the model_fn with required parameters."""
    model_fn_args = function_utils.fn_args(self._model_fn)
    kwargs = {}

    if 'labels' in model_fn_args:
      kwargs['labels'] = labels
    elif labels is not None:
      raise ValueError(
          'model_fn does not take labels, but input_fn returns labels.')
    if 'mode' in model_fn_args:
      kwargs['mode'] = mode

    if 'params' in model_fn_args:
      kwargs['params'] = params

    return self._verify_estimator_spec(
        self._model_fn(features=features, **kwargs))

  def _verify_estimator_spec(self, estimator_spec):
    """Verifies estimator spec contains correct data."""
    # TODO(ycao): Implement estimator spec verification for other modes.

    try:
      if estimator_spec.scaffold:
        logging.warning('EstimatorSpec.scaffold is ignored with XLA compilation'
                        '. Please use TPUEstimatorSpec.scaffold_fn instead.')
    except AttributeError:
      pass

    try:
      if estimator_spec.eval_metric_ops:
        raise ValueError('EstimatorSpec.eval_metric_ops is not supported with '
                         'XLA compilation. Please use '
                         'TPUEstimatorSpec.eval_metrics instead.')
    except AttributeError:
      pass

    if estimator_spec.mode == model_fn_lib.ModeKeys.EVAL:
      # If estimator_spec is of type TPUEstimatorSpec and contains eval_metrics,
      # check that eval_metrics contains eval_metric_fn and
      # eval_metric_fn_tensors with matching arguments.
      try:
        eval_metrics = estimator_spec.eval_metrics
      except AttributeError:
        eval_metrics = None

      if eval_metrics:
        (eval_metric_fn, eval_metric_fn_tensors) = eval_metrics
        eval_metric_fn_args = function_utils.fn_args(eval_metric_fn)

        if isinstance(eval_metric_fn_tensors, dict):
          missing_tensors = [
              i for i in eval_metric_fn_args if i not in eval_metric_fn_tensors
          ]
          additional_tensors = [
              i for i in eval_metric_fn_tensors if i not in eval_metric_fn_args
          ]

          if missing_tensors:
            raise ValueError('Arguments %s are needed by metric_fn (first '
                             'element of TPUEstimatorSpec.eval_metrics) but '
                             'they are not provided by evaluation tensors '
                             '(second element of TPUEstimatorSpec.eval_metrics)'
                             '.' % missing_tensors)

          if additional_tensors:
            raise ValueError('Arguments %s are provided by evaluation tensors '
                             '(second element of TPUEstimatorSpec.eval_metrics)'
                             ' but they are not needed by metric_fn (first '
                             'element of TPUEstimatorSpec.eval_metrics).' %
                             additional_tensors)

    return estimator_spec


def estimator_model_fn(target_model_fn=None):
  """estimator_model_fn decorates a model_fn to be compiled for execution.

  Currently it only works with `TPUEstimator`. If you need to use it with base
  `Estimator`, please add `tf.compat.v1.enable_resource_variables()` at the
  beginning of your program.

  Example 1, decorating model_fn:
  ```
  @xla.estimator_model_fn()
  def model_fn(features, labels, mode, params):
    ...
    return EstimatorSpec(...)


  est = Estimator(model_fn=model_fn, ...)
  est.train(...)

  ```

  Example 2, decorator as function:
  ```
  def model_fn(features, labels, mode, params):
    ...
    return EstimatorSpec(...)

  est = Estimator(model_fn=xla.estimator_model_fn(model_fn), ...)
  est.train(...)
  ```

  Args:
    target_model_fn: model_fn to be decorated. This is only needed when
      decorator is used in function call form (example 2).

  Returns:
    Decorated target_model_fn.
  """

  def decorated(function):
    return tf_decorator.make_decorator(function, _ModelFnWrapper(function))

  return decorated(target_model_fn) if target_model_fn else decorated
