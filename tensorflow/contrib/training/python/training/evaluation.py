# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Contains functions for evaluation and summarization of metrics.

The evaluation.py module contains helper functions for evaluating TensorFlow
modules using a variety of metrics and summarizing the results.

****************************************
* Evaluating a Checkpointed Model Once *
****************************************

Once we've trained a model, we'll want to evaluate it. The simplest way to do
this is to evaluate the performance of a saved model a single time. In order
to do this, we can specify a number of metrics we'll want to evaluate as well
as specify the summaries we want to save to disk. Furthermore, we can print
out the metrics values to stdout:

  # Specify where the checkpoint is stored:
  checkpoint_path = ...

  # Create model and obtain the predictions:
  images, labels = LoadData(...)
  predictions = MyModel(images)

  # Choose the metrics to compute:
  names_to_values, names_to_updates = tf.contrib.metrics.aggregate_metric_map({
      "accuracy": tf.contrib.metrics.streaming_accuracy(predictions, labels),
      "mse": tf.contrib.metrics.streaming_mean_squared_error(
        predictions, labels),
  })

  # Define the summaries to write:
  for metric_name, metric_value in metrics_to_values.iteritems():
    tf.summary.scalar(metric_name, metric_value)

  checkpoint_dir = '/tmp/my_model_dir/'
  log_dir = '/tmp/my_model_eval/'

  # We'll evaluate 1000 batches:
  num_evals = 1000

  names_to_values = evaluate_once(
      checkpoint_path=checkpoint_path,
      eval_ops=names_to_updates.values(),
      final_ops=names_to_values,
      hooks=[
            tf.contrib.training.StopAfterNEvalsHook(num_evals),
            tf.contrib.training.SummaryAtEndHook(logdir),
      ],
      config=None)

  for name in names_to_values:
    print('Metric %s has value %f.' % (name, names_to_values[name]))


************************************************
* Evaluating a Checkpointed Model with Metrics *
************************************************

Often, one wants to evaluate a model checkpoint saved on disk. This can be
performed once or repeatedly on a set schedule.

To evaluate a particular model, users define zero or more metrics and zero or
more summaries and call the evaluate_repeatedly method:

  # Create model and obtain the predictions:
  images, labels = LoadData(...)
  predictions = MyModel(images)

  # Choose the metrics to compute:
  names_to_values, names_to_updates = tf.contrib.metrics.aggregate_metric_map({
      "accuracy": tf.contrib.metrics.streaming_accuracy(predictions, labels),
      "mse": tf.contrib.metrics.streaming_mean_squared_error(
          predictions, labels),
  })

  # Define the summaries to write:
  for metric_name, metric_value in metrics_to_values.iteritems():
    tf.summary.scalar(metric_name, metric_value)

  checkpoint_dir = '/tmp/my_model_dir/'
  log_dir = '/tmp/my_model_eval/'

  # We'll evaluate 1000 batches:
  num_evals = 1000

  # Evaluate every 10 minutes:
  tf.contrib.training.evaluate_repeatedly(
      checkpoint_dir,
      eval_ops=names_to_updates.values(),
      hooks=[
            tf.contrib.training.StopAfterNEvalsHook(num_evals),
            tf.contrib.training.SummaryAtEndHook(logdir),
      ],
      eval_interval_secs=600)

*******************************************************
* Evaluating a Checkpointed Model with Summaries Only *
*******************************************************

At times, an evaluation can be performed without metrics at all but rather
with only summaries. The user need only leave out the 'eval_ops' argument:

  # Create model and obtain the predictions:
  images, labels = LoadData(...)
  predictions = MyModel(images)

  # Define the summaries to write:
  tf.summary.scalar(...)
  tf.summary.histogram(...)

  checkpoint_dir = '/tmp/my_model_dir/'
  log_dir = '/tmp/my_model_eval/'

  # Evaluate once every 10 minutes.
  tf.contrib.training.evaluate_repeatedly(
      checkpoint_dir,
      hooks=[
          tf.contrib.training.SummaryAtEndHook(logdir),
      ],
      eval_interval_secs=600)

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import evaluation
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util

__all__ = [
    'StopAfterNEvalsHook',
    'SummaryAtEndHook',
    'checkpoints_iterator',
    'evaluate_once',
    'evaluate_repeatedly',
    'get_or_create_eval_step',
    'wait_for_new_checkpoint',
]

# pylint: disable=protected-access
# pylint: disable=invalid-name
StopAfterNEvalsHook = evaluation._StopAfterNEvalsHook
evaluate_once = evaluation._evaluate_once
get_or_create_eval_step = evaluation._get_or_create_eval_step
# pylint: enable=invalid-name
# pylint: enable=protected-access


def wait_for_new_checkpoint(checkpoint_dir,
                            last_checkpoint=None,
                            seconds_to_sleep=1,
                            timeout=None):
  """Waits until a new checkpoint file is found.

  Args:
    checkpoint_dir: The directory in which checkpoints are saved.
    last_checkpoint: The last checkpoint path used or `None` if we're expecting
      a checkpoint for the first time.
    seconds_to_sleep: The number of seconds to sleep for before looking for a
      new checkpoint.
    timeout: The maximum amount of time to wait. If left as `None`, then the
      process will wait indefinitely.

  Returns:
    a new checkpoint path, or None if the timeout was reached.
  """
  logging.info('Waiting for new checkpoint at %s', checkpoint_dir)
  stop_time = time.time() + timeout if timeout is not None else None
  while True:
    checkpoint_path = tf_saver.latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None or checkpoint_path == last_checkpoint:
      if stop_time is not None and time.time() + seconds_to_sleep > stop_time:
        return None
      time.sleep(seconds_to_sleep)
    else:
      logging.info('Found new checkpoint at %s', checkpoint_path)
      return checkpoint_path


def checkpoints_iterator(checkpoint_dir, min_interval_secs=0, timeout=None):
  """Continuously yield new checkpoint files as they appear.

  The iterator only checks for new checkpoints when control flow has been
  reverted to it. This means it can miss checkpoints if your code takes longer
  to run between iterations than `min_interval_secs` or the interval at which
  new checkpoints are written.

  Args:
    checkpoint_dir: The directory in which checkpoints are saved.
    min_interval_secs: The minimum number of seconds between yielding
      checkpoints.
    timeout: The maximum amount of time to wait between checkpoints. If left as
      `None`, then the process will wait indefinitely.

  Yields:
    String paths to latest checkpoint files as they arrive. Stops yielding only
    if/when waiting for a checkpoint times out.
  """
  checkpoint_path = None
  while True:
    checkpoint_path = wait_for_new_checkpoint(
        checkpoint_dir, checkpoint_path, timeout=timeout)
    if checkpoint_path is None:
      # timed out
      return
    start = time.time()
    yield checkpoint_path
    time_to_next_eval = start + min_interval_secs - time.time()
    if time_to_next_eval > 0:
      time.sleep(time_to_next_eval)


class SummaryAtEndHook(session_run_hook.SessionRunHook):
  """A run hook that saves a summary with the results of evaluation."""

  def __init__(self, log_dir=None, summary_writer=None,
               summary_op=None, feed_dict=None):
    """Constructs the Summary Hook.

    Args:
      log_dir: The directory where the summary events are saved to.  Used only
        when `summary_writer` is not specified.
      summary_writer: A `tf.summary.FileWriter` to write summary events with.
      summary_op: The summary op to run. If left as `None`, then all summaries
        in the tf.GraphKeys.SUMMARIES collection are used.
      feed_dict: An optional feed dictionary to use when evaluating the
        summaries.

    Raises:
      ValueError: If both `log_dir` and `summary_writer` are `None`.
    """
    self._summary_op = summary_op
    self._feed_dict = feed_dict
    self._summary_writer = summary_writer
    self._log_dir = log_dir
    self._summary_writer = summary_writer
    if self._log_dir is None and self._summary_writer is None:
      raise ValueError('One of log_dir or summary_writer should be used.')
    self._global_step = variables.get_or_create_global_step()

  def begin(self):
    if self._summary_writer is None and self._log_dir:
      self._summary_writer = summary.FileWriterCache.get(self._log_dir)
    if self._summary_op is None:
      self._summary_op = summary.merge_all()

  def end(self, session):
    global_step = training_util.global_step(session, self._global_step)
    summary_str = session.run(self._summary_op, self._feed_dict)
    if self._summary_writer:
      self._summary_writer.add_summary(summary_str, global_step)
      self._summary_writer.flush()


def _scaffold_with_init(scaffold, saver, checkpoint_path):
  """Creates a scaffold that loads the given checkpoint using an init_fn.

  Args:
    scaffold: The scaffold to copy.
    saver: The saver to use when restoring the checkpoint.
    checkpoint_path: An absolute path to a checkpoint.

  Returns:
    A scaffold with an init_fn that loads the given checkpoint. If the scaffold
    provided already has an init_fn, the scaffold is returned unchanged.
  """

  def restore_checkpoint(_, session):
    saver.restore(session, checkpoint_path)

  if not scaffold.init_fn:
    scaffold = monitored_session.Scaffold(
        init_op=scaffold.init_op,
        init_feed_dict=scaffold.init_feed_dict,
        init_fn=restore_checkpoint,
        ready_op=scaffold.ready_op,
        local_init_op=scaffold.local_init_op,
        summary_op=scaffold.summary_op,
        saver=scaffold.saver)
  return scaffold


def evaluate_repeatedly(checkpoint_dir,
                        master='',
                        scaffold=None,
                        eval_ops=None,
                        feed_dict=None,
                        final_ops=None,
                        final_ops_feed_dict=None,
                        eval_interval_secs=60,
                        hooks=None,
                        config=None,
                        max_number_of_evaluations=None,
                        timeout=None):
  """Repeatedly searches for a checkpoint in `checkpoint_dir` and evaluates it.

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
  summaries run immedietly after the model checkpoint has been restored.

  Note that `evaluate_once` creates a local variable used to track the number of
  evaluations run via `tf.contrib.training.get_or_create_eval_step`.
  Consequently, if a custom local init op is provided via a `scaffold`, the
  caller should ensure that the local init op also initializes the eval step.

  Args:
    checkpoint_dir: The directory where checkpoints are stored.
    master: The BNS address of the TensorFlow master.
    scaffold: An tf.train.Scaffold instance for initializing variables and
      restoring variables. Note that `scaffold.init_fn` is used by the function
      to restore the checkpoint. If you supply a custom init_fn, then it must
      also take care of restoring the model from its checkpoint.
    eval_ops: A single `Tensor`, a list of `Tensors` or a dictionary of names
      to `Tensors`, which is run until the session is requested to stop,
      commonly done by a `tf.contrib.training.StopAfterNEvalsHook`.
    feed_dict: The feed dictionary to use when executing the `eval_ops`.
    final_ops: A single `Tensor`, a list of `Tensors` or a dictionary of names
      to `Tensors`.
    final_ops_feed_dict: A feed dictionary to use when evaluating `final_ops`.
    eval_interval_secs: The minimum number of seconds between evaluations.
    hooks: List of `tf.train.SessionRunHook` callbacks which are run inside the
      evaluation loop.
    config: An instance of `tf.ConfigProto` that will be used to
      configure the `Session`. If left as `None`, the default will be used.
    max_number_of_evaluations: The maximum times to run the evaluation. If left
      as `None`, then evaluation runs indefinitely.
    timeout: The maximum amount of time to wait between checkpoints. If left as
      `None`, then the process will wait indefinitely.

  Returns:
    The fetched values of `final_ops` or `None` if `final_ops` is `None`.
  """
  eval_step = get_or_create_eval_step()

  # Prepare the run hooks.
  hooks = hooks or []

  if eval_ops is not None:
    update_eval_step = state_ops.assign_add(eval_step, 1)

    for h in hooks:
      if isinstance(h, StopAfterNEvalsHook):
        h._set_evals_completed_tensor(update_eval_step)  # pylint: disable=protected-access

    if isinstance(eval_ops, dict):
      eval_ops['update_eval_step'] = update_eval_step
    elif isinstance(eval_ops, (tuple, list)):
      eval_ops = list(eval_ops) + [update_eval_step]
    else:
      eval_ops = [eval_ops, update_eval_step]

  final_ops_hook = basic_session_run_hooks.FinalOpsHook(
      final_ops, final_ops_feed_dict)
  hooks.append(final_ops_hook)

  num_evaluations = 0
  for checkpoint_path in checkpoints_iterator(checkpoint_dir,
                                              eval_interval_secs, timeout):

    session_creator = monitored_session.ChiefSessionCreator(
        scaffold=scaffold,
        checkpoint_filename_with_path=checkpoint_path,
        master=master,
        config=config)

    with monitored_session.MonitoredSession(
        session_creator=session_creator, hooks=hooks) as session:
      logging.info('Starting evaluation at ' + time.strftime(
          '%Y-%m-%d-%H:%M:%S', time.gmtime()))
      if eval_ops is not None:
        while not session.should_stop():
          session.run(eval_ops, feed_dict)

      logging.info('Finished evaluation at ' + time.strftime(
          '%Y-%m-%d-%H:%M:%S', time.gmtime()))
    num_evaluations += 1

    if max_number_of_evaluations is not None and num_evaluations >= max_number_of_evaluations:
      return final_ops_hook.final_ops_values

  logging.info('Timed-out waiting for a checkpoint.')
  return final_ops_hook.final_ops_values
