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

"""High level operations on graphs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import sys
import threading
import time

import numpy as np

from six import reraise

from tensorflow.contrib.framework.python.ops import ops as contrib_ops
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.layers.python.layers import summaries
from tensorflow.contrib.learn.python.learn import monitors as monitors_lib
from tensorflow.python.client import session as tf_session
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.training import session_manager as session_manager_lib
from tensorflow.python.training import summary_io
from tensorflow.python.training import supervisor as tf_supervisor

# pylint: disable=invalid-name
Supervisor = tf_supervisor.Supervisor
Coordinator = coordinator.Coordinator
SummaryWriter = summary_io.SummaryWriter

# Singletone for SummaryWriter per logdir folder.
_SUMMARY_WRITERS = {}

# Lock protecting _SUMMARY_WRITERS
_summary_writer_lock = threading.Lock()


def get_summary_writer(logdir):
  """Returns single SummaryWriter per logdir in current run.

  Args:
    logdir: str, folder to write summaries.

  Returns:
    Existing `SummaryWriter` object or new one if never wrote to given
    directory.
  """
  _summary_writer_lock.acquire()
  if logdir not in _SUMMARY_WRITERS:
    _SUMMARY_WRITERS[logdir] = SummaryWriter(logdir,
                                             graph=ops.get_default_graph())
  _summary_writer_lock.release()
  return _SUMMARY_WRITERS[logdir]


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


def _run_with_monitors(session, step, tensors, feed_dict, monitors):
  """Runs session for given tensors with monitor callbacks."""
  for monitor in monitors:
    tensors += monitor.step_begin(step)
  tensors = list(set(tensors))

  outputs = session.run(tensors, feed_dict=feed_dict)
  outputs = dict(zip(
      [t.name if isinstance(t, ops.Tensor) else t for t in tensors],
      outputs))

  should_stop = False
  for monitor in monitors:
    induce_stop = monitor.step_end(step, outputs)
    should_stop = should_stop or induce_stop
  return outputs, should_stop


# TODO(ptucker): Add unit test.
# TODO(wicke): switch to forced named kwargs
def train(graph,
          output_dir,
          train_op,
          loss_op,
          global_step_tensor=None,
          init_op=None,
          init_feed_dict=None,
          init_fn=None,
          log_every_steps=10,
          supervisor_is_chief=True,
          supervisor_master='',
          supervisor_save_model_secs=600,
          supervisor_save_summaries_steps=100,
          feed_fn=None,
          max_steps=None,
          fail_on_nan_loss=True,
          monitors=None):
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
    init_feed_dict: A dictionary that maps `Tensor` objects to feed values.
      This feed dictionary will be used when `init_op` is evaluated.
    init_fn: Optional callable passed to Supervisor to initialize the model.
    log_every_steps: Output logs regularly. The logs contain timing data and the
      current loss.
    supervisor_is_chief: Whether the current process is the chief supervisor in
      charge of restoring the model and running standard services.
    supervisor_master: The master string to use when preparing the session.
    supervisor_save_model_secs: Save a checkpoint every
      `supervisor_save_model_secs` seconds when training.
    supervisor_save_summaries_steps: Save summaries every
      `supervisor_save_summaries_steps` seconds when training.
    feed_fn: A function that is called every iteration to produce a `feed_dict`
      passed to `session.run` calls. Optional.
    max_steps: Train until `global_step_tensor` evaluates to this value.
    fail_on_nan_loss: If true, raise `NanLossDuringTrainingError` if `loss_op`
      evaluates to `NaN`. If false, continue training as if nothing happened.
    monitors: List of `BaseMonitor` subclass instances. Used for callbacks
      inside the training loop.

  Returns:
    The final loss value.

  Raises:
    ValueError: If `global_step_tensor` is not provided. See
        `tf.contrib.framework.get_global_step` for how we look it up if not
        provided explicitly.
    NanLossDuringTrainingError: If `fail_on_nan_loss` is `True`, and loss ever
        evaluates to `NaN`.
  """
  if not output_dir:
    raise ValueError('Output directory should be non-empty.')

  global_step_tensor = contrib_variables.assert_or_get_global_step(
      graph, global_step_tensor)
  if global_step_tensor is None:
    raise ValueError('No "global_step" was provided or found in the graph.')

  summary_writer = (get_summary_writer(output_dir)
                    if supervisor_is_chief else None)

  # TODO(ipolosukhin): Replace all functionality of Supervisor with Monitors.
  if not supervisor_is_chief:
    # monitors should run only on the chief.
    monitors = []
  elif not monitors:
    monitors = monitors_lib.get_default_monitors(
        loss_op=loss_op,
        summary_op=logging_ops.get_summary_op(),
        save_summary_steps=supervisor_save_summaries_steps,
        summary_writer=summary_writer)

  # Start monitors, can create graph parts.
  for monitor in monitors:
    monitor.begin(max_steps=max_steps)

  supervisor = Supervisor(
      graph,
      init_op=init_op or Supervisor.USE_DEFAULT,
      init_feed_dict=init_feed_dict,
      is_chief=supervisor_is_chief,
      logdir=output_dir,
      saver=_make_saver(graph),
      global_step=global_step_tensor,
      summary_op=None,
      summary_writer=summary_writer,
      save_model_secs=supervisor_save_model_secs,
      init_fn=init_fn)
  session = supervisor.PrepareSession(master=supervisor_master,
                                      start_standard_services=True)
  supervisor.StartQueueRunners(session)

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

        outputs, should_stop = _run_with_monitors(
            session, last_step + 1, [train_op, loss_op], feed_dict, monitors)

        loss_value = outputs[loss_op.name]
        if np.isnan(loss_value):
          failure_message = 'Model diverged with loss = NaN.'
          if fail_on_nan_loss:
            logging.error(failure_message)
            raise NanLossDuringTrainingError()
          else:
            logging.warning(failure_message)

        if should_stop:
          break

        this_step = get_current_step()

        if this_step <= last_step:
          logging.error(
              'Global step was not incremented by train op at step %s'
              ': new step %d', last_step, this_step)

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
          logging.info('Saving checkpoint for step %d to checkpoint: %s.',
                       last_step, ckpt_path)
          supervisor.saver.save(session, ckpt_path, global_step=last_step)

          # Finish monitors.
          for monitor in monitors:
            monitor.end()

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


def _get_first_op_from_collection(collection_name):
  elements = ops.get_collection(collection_name)
  if elements is not None:
    if elements:
      return elements[0]
  return None


def _get_saver():
  """Lazy init and return saver."""
  saver = _get_first_op_from_collection(ops.GraphKeys.SAVERS)
  if saver is not None:
    if saver:
      saver = saver[0]
    else:
      saver = None
  if saver is None and variables.all_variables():
    saver = tf_saver.Saver()
    ops.add_to_collection(ops.GraphKeys.SAVERS, saver)
  return saver


def _get_ready_op():
  ready_op = _get_first_op_from_collection(ops.GraphKeys.READY_OP)
  if ready_op is None:
    ready_op = variables.report_uninitialized_variables()
    ops.add_to_collection(ops.GraphKeys.READY_OP, ready_op)
  return ready_op


def _get_local_init_op():
  local_init_op = _get_first_op_from_collection(
      ops.GraphKeys.LOCAL_INIT_OP)
  if local_init_op is None:
    op_list = [variables.initialize_local_variables(),
               data_flow_ops.initialize_all_tables()]
    if op_list:
      local_init_op = control_flow_ops.group(*op_list)
      ops.add_to_collection(ops.GraphKeys.LOCAL_INIT_OP, local_init_op)
  return local_init_op


def _start_queue_runners(session, coord):
  queue_runners = ops.get_collection(ops.GraphKeys.QUEUE_RUNNERS)
  threads = []
  for qr in queue_runners:
    threads.extend(qr.create_threads(session, coord=coord, daemon=True,
                                     start=True))
  return threads


# TODO(ptucker): Add unit test.
def evaluate(graph,
             output_dir,
             checkpoint_path,
             eval_dict,
             update_op=None,
             global_step_tensor=None,
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
    eval_dict: A `dict` mapping string names to tensors to evaluate. It is
      evaluated in every logging step. The result of the final evaluation is
      returned. If update_op is None, then it's evaluated in every step.
    update_op: A `Tensor` which is run in every step.
    global_step_tensor: A `Variable` containing the global step. If `None`,
      one is extracted from the graph using the same logic as in `Supervisor`.
      Used to place eval summaries on training curves.
    supervisor_master: The master string to use when preparing the session.
    log_every_steps: Integer. Output logs every `log_every_steps` evaluation
      steps. The logs contain the `eval_dict` and timing information.
    feed_fn: A function that is called every iteration to produce a `feed_dict`
      passed to `session.run` calls. Optional.
    max_steps: Integer. Evaluate `eval_dict` this many times.

  Returns:
    A tuple `(eval_results, global_step)`:
    eval_results: A `dict` mapping `string` to numeric values (`int`, `float`)
      that are the result of running eval_dict in the last step. `None` if no
      eval steps were run.
    global_step: The global step this evaluation corresponds to.
  """
  global_step_tensor = contrib_variables.assert_or_get_global_step(
      graph, global_step_tensor)

  for key, value in eval_dict.items():
    if not summaries.is_summary_tag_unique(key):
      continue
    if isinstance(value, ops.Tensor):
      summaries.summarize_tensor(value, tag=key)

  # Create or get summary op, global_step and saver.
  summary_op = logging_ops.get_summary_op()
  saver = _get_saver()
  local_init_op = _get_local_init_op()
  ready_op = _get_ready_op()

  session_manager = session_manager_lib.SessionManager(
      local_init_op=local_init_op,
      ready_op=ready_op)
  session, initialized = session_manager.recover_session(
      master=supervisor_master,
      saver=saver,
      checkpoint_dir=checkpoint_path)

  # Start queue runners.
  coord = coordinator.Coordinator()
  threads = _start_queue_runners(session, coord)

  with session:
    if not initialized:
      logging.warning('Failed to initialize from %s.', checkpoint_path)
      # TODO(ipolosukhin): This should be failing, but old code relies on that.
      session.run(variables.initialize_all_variables())
      if checkpoint_path:
        _restore_from_checkpoint(session, graph, checkpoint_path, saver)

    current_global_step = session.run(global_step_tensor)
    eval_results = None
    # TODO(amodei): Fix this to run through the eval set exactly once.
    step = 0
    eval_step = None
    feed_dict = None
    logging.info('Eval steps [%d,%s) for training step %d.', step,
                 'inf' if max_steps is None
                 else str(max_steps), current_global_step)
    try:
      try:
        while (max_steps is None) or (step < max_steps):
          step += 1
          start_time = time.time()
          feed_dict = feed_fn() if feed_fn is not None else None
          if update_op is not None:
            session.run(update_op, feed_dict=feed_dict)
          else:
            eval_results = session.run(eval_dict, feed_dict=feed_dict)
            eval_step = step

          # TODO(wicke): We should assert that the global step hasn't changed.
          if step % log_every_steps == 0:
            if eval_step is None or step != eval_step:
              eval_results = session.run(eval_dict, feed_dict=feed_dict)
              eval_step = step
            duration = time.time() - start_time
            logging.info('Results after %d steps (%.3f sec/batch): %s.',
                         step, float(duration),
                         ', '.join('%s = %s' % (k, v)
                                   for k, v in eval_results.items()))
      finally:
        if eval_results is None or step != eval_step:
          eval_results = session.run(eval_dict, feed_dict=feed_dict)
          eval_step = step
        # Stop queue runners.
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=120)

        # Make our own summary writer and write a summary to the eval dir.
        # Only is feed_fn is not provided.
        # TODO(ipolosukhin): Convert evaluation to use streaming_metrics,
        # then we can save for non feed_fn as well.
        if summary_op is not None and feed_fn is None:
          summary_writer = None
          try:
            summary_writer = get_summary_writer(output_dir)
            summary_str = session.run(summary_op)
            if summary_str:
              summary_writer.add_summary(summary_str, current_global_step)
          finally:
            if summary_writer:
              summary_writer.close()
    # catch OutOfRangeError which is thrown when queue is out of data (and for
    # other reasons as well).
    except errors.OutOfRangeError as e:
      if max_steps is None:
        logging.info('Input queue is exhausted.')
      else:
        logging.warn('Input queue is exhausted: %s.', e)
    # catch StopIteration which is thrown is DataReader is out of data.
    except StopIteration as e:
      if max_steps is None:
        logging.info('Input iterator is exhausted.')
      else:
        logging.warn('Input iterator is exhausted: %s.', e)

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
        return [session.run(output_dict, f) for f in feed_dicts]
      finally:
        coord.request_stop()


def infer(restore_checkpoint_path, output_dict, feed_dict=None):
  return run_feeds(output_dict=output_dict,
                   feed_dicts=[feed_dict] if feed_dict is not None else [None],
                   restore_checkpoint_path=restore_checkpoint_path)[0]
