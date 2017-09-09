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

"""Classes and functions related to train_and_evaluate."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import six

from tensorflow.python.training import session_run_hook


def _validate_input_fn(input_fn):
  """Validates the `input_fn`."""
  if not callable(input_fn):
    raise TypeError(
        '`input_fn` must be callable, given: {}'.format(input_fn))


def _validate_hooks(hooks):
  """Validates the `hooks`."""
  hooks = tuple(hooks or [])
  for hook in hooks:
    if not isinstance(hook, session_run_hook.SessionRunHook):
      raise TypeError(
          'All hooks must be `SessionRunHook` instances, given: {}'.format(
              hook))
  return hooks


class TrainSpec(
    collections.namedtuple('TrainSpec', ['input_fn', 'max_steps', 'hooks'])):
  """Objects passed to `train_and_evaluate`.

  `TrainSpec` fully defines the objects to be run by `Estimator.train`.
  """

  def __new__(cls,
              input_fn,
              max_steps=None,
              hooks=None):
    """Creates a validated `TrainSpec` instance.

    Args:
      input_fn: Training input function returning a tuple of:
          features - `Tensor` or dictionary of string feature name to `Tensor`.
          labels - `Tensor` or dictionary of `Tensor` with labels.
      max_steps: Int. Number of total steps for which to train model. If `None`,
        train forever or train until `input_fn` generates the `OutOfRange` error
        or `StopIteration` exception. See `Estimator.train` for details.
      hooks: Iterable of `tf.train.SessionRunHook` objects to run
        on all workers (including chief) during training.

    Returns:
      A validated `TrainSpec` object.

    Raises:
      ValueError: If validation fails.
      TypeError: If any of the arguments is not the expected type.
    """
    # Validate input_fn.
    _validate_input_fn(input_fn)

    # Validate max_steps.
    if max_steps is not None and max_steps <= 0:
      raise ValueError(
          'Must specify max_steps > 0, given: {}'.format(max_steps))

    # Validate hooks.
    hooks = _validate_hooks(hooks)

    return super(TrainSpec, cls).__new__(
        cls,
        input_fn=input_fn,
        max_steps=max_steps,
        hooks=hooks)


class EvalSpec(
    collections.namedtuple('EvalSpec', [
        'input_fn', 'steps', 'name', 'hooks', 'export_strategies',
        'delay_secs', 'throttle_secs'
    ])):
  """Objects passed to `train_and_evaluate`.

  `EvalSpec` fully defines the objects to be run by `Estimator.evaluate` and
  `Estimator.export_savedmodel`.
  """

  def __new__(cls,
              input_fn,
              steps=100,
              name=None,
              hooks=None,
              export_strategies=None,
              delay_secs=120,
              throttle_secs=600):
    """Creates a validated `EvalSpec` instance.

    Args:
      input_fn: Training input function returning a tuple of:
          features - `Tensor` or dictionary of string feature name to `Tensor`.
          labels - `Tensor` or dictionary of `Tensor` with labels.
      steps: Int. Number of total steps for which to train model. If `None`,
        train forever or train until `input_fn` generates the `OutOfRange` error
        or `StopIteration` exception. See `Estimator.train` for details.
      name: String. Name of the evaluation if user needs to run multiple
        evaluations on different data sets. Metrics for different evaluations
        are saved in separate folders, and appear separately in tensorboard.
      hooks: Iterable of `tf.train.SessionRunHook` objects to run
        on all workers (including chief) during training.
      export_strategies: Iterable of `ExportStrategy`s, or a single one, or
        `None`. `export_strategies` will be invoked after each evaluation.
      delay_secs: Int. Start evaluating after waiting for this many seconds.
      throttle_secs: Int. Do not re-evaluate unless the last evaluation was
        started at least this many seconds ago. Of course, evaluation does not
        occur if no new checkpoint is available, hence, this is the minimum.

    Returns:
      A validated `TrainSpec` object.

    Raises:
      ValueError: If validation fails.
      TypeError: If any of the arguments is not the expected type.
    """
    # Validate input_fn.
    _validate_input_fn(input_fn)

    # Validate steps.
    if steps is not None and steps <= 0:
      raise ValueError('Must specify steps > 0, given: {}'.format(steps))

    # Validate name.
    if name is not None and not isinstance(name, six.string_types):
      raise TypeError('`name` must be string, given: {}'.format(name))

    # Validate hooks.
    hooks = _validate_hooks(hooks)

    # Validate export_strategies.
    export_strategies = tuple(export_strategies or [])
    # TODO(b/65169058): Validate export_strategies once `ExportStratey` defined.

    # Validate delay_secs.
    if delay_secs < 0:
      raise ValueError(
          'Must specify delay_secs >= 0, given: {}'.format(delay_secs))

    # Validate throttle_secs.
    if throttle_secs < 0:
      raise ValueError(
          'Must specify throttle_secs >= 0, given: {}'.format(throttle_secs))

    return super(EvalSpec, cls).__new__(
        cls,
        input_fn=input_fn,
        steps=steps,
        name=name,
        hooks=hooks,
        export_strategies=export_strategies,
        delay_secs=delay_secs,
        throttle_secs=throttle_secs)

