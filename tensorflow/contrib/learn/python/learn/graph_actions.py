#  Copyright 2016 Google Inc. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""High level operations on graphs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections as py_collections
import itertools
import sys
import time

import numpy as np

from six import reraise
from tensorflow.contrib.framework.python.ops import ops as contrib_ops
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.layers.python.layers import summaries
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.client import session as tf_session
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.training import summary_io
from tensorflow.python.training import supervisor as tf_supervisor


# pylint: disable=invalid-name
Supervisor = tf_supervisor.Supervisor
Coordinator = coordinator.Coordinator
SummaryWriter = summary_io.SummaryWriter


class NanLossDuringTrainingError(RuntimeError):

  def __str__(self):
    return 'NaN loss during training.'


def _make_saver(graph):
  vars_to_save = graph.get_collection(ops.GraphKeys.VARIABLES)
  if vars_to_save:
    return tf_saver.Saver(vars_to_save, sharded=True)
  else:
    return None


def _restore_from_checkpoint(session, graph, checkpoint_path, saver=None):
  logging.info('Loading model from checkpoint: %s.', checkpoint_path)
  assert gfile.Glob(checkpoint_path)
  saver = saver or _make_saver(graph)
  if saver:
    saver.restore(session, checkpoint_path)
  else:
    logging.info('No variables found in graph, not creating Saver() object.')


def _run_dict(session, run_dict, feed_dict=None):
  """Convenience function to run a session on each item in a dict of tensors.

  Args:
    session: The session to evaluate.
    run_dict: A dict of tensors to be run in the session.
    feed_dict: Feed dict to be used in running the session.
  Returns:
    A dict containing the result of evaluating the tensors.
  Raises:
    ValueError: if `run_dict` is missing or empty.
  """
  if run_dict is None:
    raise ValueError('Invalid run_dict %s.', run_dict)
  keys = run_dict.keys()
  values = session.run([run_dict[key] for key in keys], feed_dict=feed_dict)
  return dict(zip(keys, values))


class SupervisorParams(py_collections.namedtuple(
    'SupervisorParams',
    ['is_chief', 'master', 'save_model_secs', 'save_summaries_secs'])):
  """Parameters required to configure supervisor for training.

  Fields:
    is_chief: Whether the current process is the chief supervisor in charge of
      restoring the model and running standard services.
    master: The master string to use when preparing the session.
    save_model_secs: Save a checkpoint every `save_model_secs` seconds when
      training.
    save_summaries_secs: Save summaries every `save_summaries_secs` seconds when
      training.
  """
SupervisorParams.__new__.__defaults__ = (True, '', 600, 10)


def _prepare_session(graph,
                     output_dir,
                     start_services,
                     global_step_tensor,
                     init_op=None,
                     init_fn=None,
                     supervisor_is_chief=True,
                     supervisor_master='',
                     supervisor_save_model_secs=600,
                     supervisor_save_summaries_secs=10):
  """Starts a session using the supervisor."""
  if global_step_tensor is None:
    global_step_tensor = Supervisor.USE_DEFAULT
  supervisor = Supervisor(
      graph,
      init_op=init_op or Supervisor.USE_DEFAULT,
      is_chief=supervisor_is_chief,
      logdir=output_dir,
      saver=_make_saver(graph),
      global_step=global_step_tensor,
      save_model_secs=supervisor_save_model_secs,
      save_summaries_secs=supervisor_save_summaries_secs,
      init_fn=init_fn)
  session = supervisor.PrepareSession(master=supervisor_master,
                                      start_standard_services=start_services)
  supervisor.StartQueueRunners(session)
  return supervisor, session


# TODO(ptucker): Add unit test.
# TODO(wicke): switch to forced named kwargs
def train(graph,
          output_dir,
          train_op,
          loss_op,
          global_step_tensor=None,
          init_op=None,
          init_fn=None,
          log_every_steps=10,
          supervisor_is_chief=True,
          supervisor_master='',
          supervisor_save_model_secs=600,
          supervisor_save_summaries_secs=10,
          feed_fn=None,
          max_steps=None,
          fail_on_nan_loss=True):
  """Train a model.

  Given `graph`, a directory to write outputs to (`output_dir`), and some ops,
  run a training loop. The given `train_op` performs one step of training on the
  model. The `loss_op` represents the objective function of the training. It is
  expected to increment the `global_step_tensor`, a scalar integer tensor
  counting training steps. This function uses `Supervisor` to initialize the
  graph (from a checkpoint if one is available in `output_dir`), write summaries
  defined in the graph, and write regular checkpoints as defined by
  `supervisor_save_model_secs`.

  Training continues until `global_step_tensor` evaluates to `max_steps`, or, if
  `fail_on_nan_loss`, until `loss_op` evaluates to `NaN`. In that case the
  program is terminated with exit code 1.

  Args:
    graph: A graph to train. It is expected that this graph is not in use
      elsewhere.
    output_dir: A directory to write outputs to.
    train_op: An op that performs one training step when run.
    loss_op: A scalar loss tensor.
    global_step_tensor: A tensor representing the global step. If none is given,
      one is extracted from the graph using the same logic as in `Supervisor`.
    init_op: An op that initializes the graph. If `None`, use `Supervisor`'s
      default.
    init_fn: Optional callable passed to Supervisor to initialize the model.
    log_every_steps: Output logs regularly. The logs contain timing data and the
      current loss.
    supervisor_is_chief: Whether the current process is the chief supervisor in
      charge of restoring the model and running standard services.
    supervisor_master: The master string to use when preparing the session.
    supervisor_save_model_secs: Save a checkpoint every
      `supervisor_save_model_secs` seconds when training.
    supervisor_save_summaries_secs: Save summaries every
      `supervisor_save_summaries_secs` seconds when training.
    feed_fn: A function that is called every iteration to produce a `feed_dict`
      passed to `session.run` calls. Optional.
    max_steps: Train until `global_step_tensor` evaluates to this value.
    fail_on_nan_loss: If true, raise `NanLossDuringTrainingError` if `loss_op`
      evaluates to `NaN`. If false, continue training as if nothing happened.

  Returns:
    The final loss value.

  Raises:
    ValueError: If `global_step_tensor` is not provided. See
        `tf.contrib.framework.get_global_step` for how we look it up if not
        provided explicitly.
    NanLossDuringTrainingError: If `fail_on_nan_loss` is `True`, and loss ever
        evaluates to `NaN`.
  """
  global_step_tensor = contrib_variables.assert_or_get_global_step(
      graph, global_step_tensor)
  if global_step_tensor is None:
    raise ValueError('No "global_step" was provided or found in the graph.')
  supervisor, session = _prepare_session(
      graph=graph,
      output_dir=output_dir,
      start_services=True,
      global_step_tensor=global_step_tensor,
      init_op=init_op,
      init_fn=init_fn,
      supervisor_is_chief=supervisor_is_chief,
      supervisor_master=supervisor_master,
      supervisor_save_model_secs=supervisor_save_model_secs,
      supervisor_save_summaries_secs=supervisor_save_summaries_secs)

  with session:
    get_current_step = lambda: session.run(global_step_tensor)

    start_step = get_current_step()
    last_step = start_step
    last_log_step = start_step
    loss_value = None
    logging.info('Training steps [%d,%s)', last_step, 'inf'
                 if max_steps is None else str(max_steps))
    excinfo = None
    try:
      while not supervisor.ShouldStop() and (
          (max_steps is None) or (last_step < max_steps)):
        start_time = time.time()
        feed_dict = feed_fn() if feed_fn is not None else None
        _, loss_value = session.run([train_op, loss_op], feed_dict=feed_dict)
        if np.isnan(loss_value):
          failure_message = 'Model diverged with loss = NaN.'
          if fail_on_nan_loss:
            logging.error(failure_message)
            raise NanLossDuringTrainingError()
          else:
            logging.warning(failure_message)

        this_step = get_current_step()

        if this_step <= last_step:
          logging.error(
              'Global step was not incremented by train op at step %s'
              ': new step %d' % (last_step, this_step))

        last_step = this_step
        is_last_step = (max_steps is not None) and (last_step >= max_steps)
        if is_last_step or (last_step - last_log_step >= log_every_steps):
          logging.info(
              'training step %d, loss = %.5f (%.3f sec/batch).',
              last_step, loss_value, float(time.time() - start_time))
          last_log_step = last_step
    except errors.OutOfRangeError as e:
      logging.warn('Got exception during tf.learn training loop possibly '
                   'due to exhausted input queue %s.', e)
    except BaseException as e:  # pylint: disable=broad-except
      # Hold on to any other exceptions while we try recording a final
      # checkpoint and summary.
      excinfo = sys.exc_info()
    finally:
      try:
        # Call supervisor.Stop() from within a try block because it re-raises
        # exceptions thrown by the supervised threads.
        supervisor.Stop(close_summary_writer=False)

        # Save one last checkpoint and summaries
        # TODO(wicke): This should be handled by Supervisor

        # In case we encountered an exception in the try block before we updated
        # last_step, update it here (again).
        last_step = get_current_step()
        if supervisor_is_chief:
          ckpt_path = supervisor.save_path
          logging.info('Saving checkpoint for step %d to checkpoint: %s.' % (
              last_step, ckpt_path))
          supervisor.saver.save(session, ckpt_path, global_step=last_step)
          if supervisor.summary_op is not None:
            summary_strs = session.run(supervisor.summary_op)
            supervisor.summary_writer.add_summary(summary_strs, last_step)
            supervisor.summary_writer.add_session_log(
                SessionLog(status=SessionLog.STOP), last_step)
            supervisor.summary_writer.close()
      # catch OutOfRangeError which is thrown when queue is out of data (and for
      # other reasons as well).
      except errors.OutOfRangeError as e:
        logging.warn('OutOfRangeError in tf.learn final checkpoint possibly '
                     'due to exhausted input queue. Note: summary_op is not '
                     'expected to trigger dequeues. %s.', e)
      except BaseException as e:  # pylint: disable=broad-except
        # If we don't already have an exception to re-raise, raise this one.
        if not excinfo:
          raise
        # Otherwise, log this one and raise the other in the finally block.
        logging.error('Got exception during tf.learn final checkpoint %s.', e)
      finally:
        if excinfo:
          reraise(*excinfo)
    return loss_value


# TODO(ptucker): Add unit test.
def evaluate(graph,
             output_dir,
             checkpoint_path,
             eval_dict,
             global_step_tensor=None,
             init_op=None,
             supervisor_master='',
             log_every_steps=10,
             feed_fn=None,
             max_steps=None):
  """Evaluate a model loaded from a checkpoint.

  Given `graph`, a directory to write summaries to (`output_dir`), a checkpoint
  to restore variables from, and a `dict` of `Tensor`s to evaluate, run an eval
  loop for `max_steps` steps.

  In each step of evaluation, all tensors in the `eval_dict` are evaluated, and
  every `log_every_steps` steps, they are logged. At the very end of evaluation,
  a summary is evaluated (finding the summary ops using `Supervisor`'s logic)
  and written to `output_dir`.

  Args:
    graph: A `Graph` to train. It is expected that this graph is not in use
      elsewhere.
    output_dir: A string containing the directory to write a summary to.
    checkpoint_path: A string containing the path to a checkpoint to restore.
      Can be `None` if the graph doesn't require loading any variables.
    eval_dict: A `dict` mapping string names to tensors to evaluate for in every
      eval step.
    global_step_tensor: A `Variable` containing the global step. If `None`,
      one is extracted from the graph using the same logic as in `Supervisor`.
      Used to place eval summaries on training curves.
    init_op: An op that initializes the graph. If `None`, use `Supervisor`'s
      default.
    supervisor_master: The master string to use when preparing the session.
    log_every_steps: Integer. Output logs every `log_every_steps` evaluation
      steps. The logs contain the `eval_dict` and timing information.
    feed_fn: A function that is called every iteration to produce a `feed_dict`
      passed to `session.run` calls. Optional.
    max_steps: Integer. Evaluate `eval_dict` this many times.

  Returns:
    A tuple `(eval_results, global_step)`:
    eval_results: A `dict` mapping `string` to numeric values (`int`, `float`)
      that are the eval results from the last step of the eval.  None if no
      eval steps were run.
    global_step: The global step this evaluation corresponds to.
  """
  global_step_tensor = contrib_variables.assert_or_get_global_step(
      graph, global_step_tensor)

  # Add scalar summaries for every tensor in evaluation dict if there is not
  # one existing already or it's a string.
  existing_tags = [tensor_util.constant_value(summary.op.inputs[0])
                   for summary in ops.get_collection(ops.GraphKeys.SUMMARIES)]
  for key, value in eval_dict.items():
    if key in existing_tags:
      continue
    if isinstance(value, ops.Tensor):
      summaries.summarize_tensor(value, tag=key)

  # TODO(wicke): Don't use supervisor here, or switch to output_dir=eval_dir.
  supervisor, session = _prepare_session(
      graph=graph,
      output_dir=None,  # Must be None to avoid writing an event file
      start_services=False,
      global_step_tensor=global_step_tensor,
      init_op=init_op,
      supervisor_is_chief=True,
      supervisor_master=supervisor_master,
      supervisor_save_model_secs=None,
      supervisor_save_summaries_secs=None)
  global_step_tensor = supervisor.global_step

  with session:
    if checkpoint_path:
      _restore_from_checkpoint(
          session, graph, checkpoint_path, supervisor.saver)

    current_global_step = session.run(global_step_tensor)
    eval_results = None
    # TODO(amodei): Fix this to run through the eval set exactly once.
    step = 0
    logging.info('Eval steps [%d,%s)', step, 'inf' if max_steps is None
                 else str(max_steps))
    try:
      try:
        while not supervisor.ShouldStop() and (
            (max_steps is None) or (step < max_steps)):
          start_time = time.time()
          feed_dict = feed_fn() if feed_fn is not None else None
          eval_results = _run_dict(session, eval_dict, feed_dict=feed_dict)
          # TODO(wicke): We should assert that the global step hasn't changed.
          step += 1
          if step % log_every_steps == 0:
            duration = time.time() - start_time
            logging.info('Results after %d steps (%.3f sec/batch): %s.',
                         step, float(duration),
                         ', '.join('%s = %s' % (k, v)
                                   for k, v in eval_results.items()))
      finally:
        # Make our own summary writer and write a summary to the eval dir
        if supervisor.summary_op is not None:
          summary_writer = None
          try:
            summary_writer = SummaryWriter(output_dir,
                                           graph_def=session.graph_def)

            summary_str = session.run(supervisor.summary_op)
            if summary_str:
              summary_writer.add_summary(summary_str, current_global_step)
          finally:
            if summary_writer:
              summary_writer.close()

        # Call supervisor.Stop() from within a try block because it re-raises
        # exceptions thrown by the supervised threads.
        supervisor.Stop()
    # catch OutOfRangeError which is thrown when queue is out of data (and for
    # other reasons as well).
    except errors.OutOfRangeError as e:
      logging.warn('Input queue exhausted: %s.', e)

  return eval_results, current_global_step


def run_n(output_dict, feed_dict=None, restore_checkpoint_path=None, n=1):
  """Run `output_dict` tensors `n` times, with the same `feed_dict` each run.

  Args:
    output_dict: A `dict` mapping string names to tensors to run. Must all be
      from the same graph.
    feed_dict: `dict` of input values to feed each run.
    restore_checkpoint_path: A string containing the path to a checkpoint to
      restore.
    n: Number of times to repeat.

  Returns:
    A list of `n` `dict` objects, each containing values read from `output_dict`
    tensors.
  """
  return run_feeds(
      output_dict=output_dict,
      feed_dicts=itertools.repeat(feed_dict, n),
      restore_checkpoint_path=restore_checkpoint_path)


# TODO(ptucker): Add save_checkpoint_path.
def run_feeds(output_dict, feed_dicts, restore_checkpoint_path=None):
  """Run `output_dict` tensors with each input in `feed_dicts`.

  If `checkpoint_path` is supplied, restore from checkpoint. Otherwise, init all
  variables.

  Args:
    output_dict: A `dict` mapping string names to `Tensor` objects to run.
      Tensors must all be from the same graph.
    feed_dicts: Iterable of `dict` objects of input values to feed.
    restore_checkpoint_path: A string containing the path to a checkpoint to
      restore.

  Returns:
    A list of dicts of values read from `output_dict` tensors, one item in the
    list for each item in `feed_dicts`. Keys are the same as `output_dict`,
    values are the results read from the corresponding `Tensor` in
    `output_dict`.

  Raises:
    ValueError: if `output_dict` or `feed_dicts` is None or empty.
  """
  if not output_dict:
    raise ValueError('output_dict is invalid: %s.' % output_dict)
  if not feed_dicts:
    raise ValueError('feed_dicts is invalid: %s.' % feed_dicts)

  graph = contrib_ops.get_graph_from_inputs(output_dict.values())

  with graph.as_default() as g:
    with tf_session.Session('') as session:
      if restore_checkpoint_path:
        _restore_from_checkpoint(session, g, restore_checkpoint_path)
      else:
        session.run(variables.initialize_all_variables())
      session.run(variables.initialize_local_variables())
      session.run(data_flow_ops.initialize_all_tables())
      coord = Coordinator()
      try:
        queue_runner.start_queue_runners(session, coord=coord)
        return [_run_dict(session, output_dict, f) for f in feed_dicts]
      finally:
        coord.request_stop()


def infer(restore_checkpoint_path, output_dict, feed_dict=None):
  return run_feeds(output_dict=output_dict,
                   feed_dicts=[feed_dict] if feed_dict is not None else [None],
                   restore_checkpoint_path=restore_checkpoint_path)[0]
