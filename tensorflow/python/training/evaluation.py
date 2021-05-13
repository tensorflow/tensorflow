# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains functions for evaluation and summarization of metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import monitored_session
from tensorflow.python.training import session_run_hook


def _get_or_create_eval_step():
  """Gets or creates the eval step `Tensor`.

  Returns:
    A `Tensor` representing a counter for the evaluation step.

  Raises:
    ValueError: If multiple `Tensors` have been added to the
      `tf.GraphKeys.EVAL_STEP` collection.
  """
  graph = ops.get_default_graph()
  eval_steps = graph.get_collection(ops.GraphKeys.EVAL_STEP)
  if len(eval_steps) == 1:
    return eval_steps[0]
  elif len(eval_steps) > 1:
    raise ValueError('Multiple tensors added to tf.GraphKeys.EVAL_STEP')
  else:
    counter = variable_scope.get_variable(
        'eval_step',
        shape=[],
        dtype=dtypes.int64,
        initializer=init_ops.zeros_initializer(),
        trainable=False,
        collections=[ops.GraphKeys.LOCAL_VARIABLES, ops.GraphKeys.EVAL_STEP])
    return counter


def _get_latest_eval_step_value(update_ops):
  """Gets the eval step `Tensor` value after running `update_ops`.

  Args:
    update_ops: A list of `Tensors` or a dictionary of names to `Tensors`, which
      are run before reading the eval step value.

  Returns:
    A `Tensor` representing the value for the evaluation step.
  """
  if isinstance(update_ops, dict):
    update_ops = list(update_ops.values())

  with ops.control_dependencies(update_ops):
    return array_ops.identity(_get_or_create_eval_step().read_value())


class _MultiStepStopAfterNEvalsHook(session_run_hook.SessionRunHook):
  """Run hook used by the evaluation routines to run the `eval_ops` N times."""

  def __init__(self, num_evals, steps_per_run=1):
    """Constructs the run hook.

    Args:
      num_evals: The number of evaluations to run for. if set to None, will
        iterate the dataset until all inputs are exhausted.
      steps_per_run: Number of steps executed per run call.
    """
    self._num_evals = num_evals
    self._evals_completed = None
    self._steps_per_run_initial_value = steps_per_run

  def _set_evals_completed_tensor(self, updated_eval_step):
    self._evals_completed = updated_eval_step

  def begin(self):
    self._steps_per_run_variable = \
        basic_session_run_hooks.get_or_create_steps_per_run_variable()

  def after_create_session(self, session, coord):
    # Update number of steps to run in the first run call
    if self._num_evals is None:
      steps = self._steps_per_run_initial_value
    else:
      steps = min(self._steps_per_run_initial_value, self._num_evals)
    self._steps_per_run_variable.load(steps, session=session)

  def before_run(self, run_context):
    return session_run_hook.SessionRunArgs(
        {'evals_completed': self._evals_completed})

  def after_run(self, run_context, run_values):
    evals_completed = run_values.results['evals_completed']
    # Update number of steps to run in the next iteration
    if self._num_evals is None:
      steps = self._steps_per_run_initial_value
    else:
      steps = min(self._num_evals - evals_completed,
                  self._steps_per_run_initial_value)
    self._steps_per_run_variable.load(steps, session=run_context.session)

    if self._num_evals is None:
      logging.info('Evaluation [%d]', evals_completed)
    else:
      logging.info('Evaluation [%d/%d]', evals_completed, self._num_evals)
    if self._num_evals is not None and evals_completed >= self._num_evals:
      run_context.request_stop()


class _StopAfterNEvalsHook(session_run_hook.SessionRunHook):
  """Run hook used by the evaluation routines to run the `eval_ops` N times."""

  def __init__(self, num_evals, log_progress=True):
    """Constructs the run hook.

    Args:
      num_evals: The number of evaluations to run for. if set to None, will
        iterate the dataset until all inputs are exhausted.
      log_progress: Whether to log evaluation progress, defaults to True.
    """
    # The number of evals to run for.
    self._num_evals = num_evals
    self._evals_completed = None
    self._log_progress = log_progress
    # Reduce logging frequency if there are 20 or more evaluations.
    self._log_frequency = (1 if (num_evals is None or num_evals < 20) else
                           math.floor(num_evals / 10.))

  def _set_evals_completed_tensor(self, updated_eval_step):
    self._evals_completed = updated_eval_step

  def before_run(self, run_context):
    return session_run_hook.SessionRunArgs(
        {'evals_completed': self._evals_completed})

  def after_run(self, run_context, run_values):
    evals_completed = run_values.results['evals_completed']
    if self._log_progress:
      if self._num_evals is None:
        logging.info('Evaluation [%d]', evals_completed)
      else:
        if ((evals_completed % self._log_frequency) == 0 or
            (self._num_evals == evals_completed)):
          logging.info('Evaluation [%d/%d]', evals_completed, self._num_evals)
    if self._num_evals is not None and evals_completed >= self._num_evals:
      run_context.request_stop()


def _evaluate_once(checkpoint_path,
                   master='',
                   scaffold=None,
                   eval_ops=None,
                   feed_dict=None,
                   final_ops=None,
                   final_ops_feed_dict=None,
                   hooks=None,
                   config=None):
  """Evaluates the model at the given checkpoint path.

  During a single evaluation, the `eval_ops` is run until the session is
  interrupted or requested to finish. This is typically requested via a
  `tf.contrib.training.StopAfterNEvalsHook` which results in `eval_ops` running
  the requested number of times.

  Optionally, a user can pass in `final_ops`, a single `Tensor`, a list of
  `Tensors` or a dictionary from names to `Tensors`. The `final_ops` is
  evaluated a single time after `eval_ops` has finished running and the fetched
  values of `final_ops` are returned. If `final_ops` is left as `None`, then
  `None` is returned.

  One may also consider using a `tf.contrib.training.SummaryAtEndHook` to record
  summaries after the `eval_ops` have run. If `eval_ops` is `None`, the
  summaries run immediately after the model checkpoint has been restored.

  Note that `evaluate_once` creates a local variable used to track the number of
  evaluations run via `tf.contrib.training.get_or_create_eval_step`.
  Consequently, if a custom local init op is provided via a `scaffold`, the
  caller should ensure that the local init op also initializes the eval step.

  Args:
    checkpoint_path: The path to a checkpoint to use for evaluation.
    master: The BNS address of the TensorFlow master.
    scaffold: An tf.compat.v1.train.Scaffold instance for initializing variables
      and restoring variables. Note that `scaffold.init_fn` is used by the
      function to restore the checkpoint. If you supply a custom init_fn, then
      it must also take care of restoring the model from its checkpoint.
    eval_ops: A single `Tensor`, a list of `Tensors` or a dictionary of names to
      `Tensors`, which is run until the session is requested to stop, commonly
      done by a `tf.contrib.training.StopAfterNEvalsHook`.
    feed_dict: The feed dictionary to use when executing the `eval_ops`.
    final_ops: A single `Tensor`, a list of `Tensors` or a dictionary of names
      to `Tensors`.
    final_ops_feed_dict: A feed dictionary to use when evaluating `final_ops`.
    hooks: List of `tf.estimator.SessionRunHook` callbacks which are run inside
      the evaluation loop.
    config: An instance of `tf.compat.v1.ConfigProto` that will be used to
      configure the `Session`. If left as `None`, the default will be used.

  Returns:
    The fetched values of `final_ops` or `None` if `final_ops` is `None`.
  """
  eval_step = _get_or_create_eval_step()

  # Prepare the run hooks.
  hooks = list(hooks or [])

  if eval_ops is not None:
    if any(isinstance(h, _MultiStepStopAfterNEvalsHook) for h in hooks):
      steps_per_run_variable = \
          basic_session_run_hooks.get_or_create_steps_per_run_variable()
      update_eval_step = state_ops.assign_add(
          eval_step,
          math_ops.cast(steps_per_run_variable, dtype=eval_step.dtype),
          use_locking=True)
    else:
      update_eval_step = state_ops.assign_add(eval_step, 1, use_locking=True)

    if isinstance(eval_ops, dict):
      eval_ops['update_eval_step'] = update_eval_step
    elif isinstance(eval_ops, (tuple, list)):
      eval_ops = list(eval_ops) + [update_eval_step]
    else:
      eval_ops = [eval_ops, update_eval_step]

    eval_step_value = _get_latest_eval_step_value(eval_ops)

    for h in hooks:
      if isinstance(h, (_StopAfterNEvalsHook, _MultiStepStopAfterNEvalsHook)):
        h._set_evals_completed_tensor(eval_step_value)  # pylint: disable=protected-access

  logging.info('Starting evaluation at ' +
               time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime()))
  start = time.time()
  # Prepare the session creator.
  session_creator = monitored_session.ChiefSessionCreator(
      scaffold=scaffold,
      checkpoint_filename_with_path=checkpoint_path,
      master=master,
      config=config)

  final_ops_hook = basic_session_run_hooks.FinalOpsHook(final_ops,
                                                        final_ops_feed_dict)
  hooks.append(final_ops_hook)

  with monitored_session.MonitoredSession(
      session_creator=session_creator, hooks=hooks) as session:
    if eval_ops is not None:
      while not session.should_stop():
        session.run(eval_ops, feed_dict)
  logging.info('Inference Time : {:0.5f}s'.format(time.time() - start))

  logging.info('Finished evaluation at ' +
               time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime()))
  return final_ops_hook.final_ops_values
