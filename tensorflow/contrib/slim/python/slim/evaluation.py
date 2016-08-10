# Copyright 2016 Google Inc. All Rights Reserved.
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

**********************
* Evaluating Metrics *
**********************

In the simplest use case, we use a model to create the predictions, then specify
the metrics and finally call the `evaluation` method:

  # Create model and obtain the predictions:
  images, labels = LoadData(...)
  predictions = MyModel(images)

  # Choose the metrics to compute:
  names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
      "accuracy": slim.metrics.accuracy(predictions, labels),
      "mse": slim.metrics.mean_squared_error(predictions, labels),
  })

  init_op = tf.group(
      tf.initialize_all_variables(),
      tf.initialize_local_variables())

  with tf.Session() as sess:
    metric_values = slim.evaluation(
        sess,
        num_evals=1,
        init_op=init_op,
        eval_op=names_to_updates.values(),
        final_op=name_to_values.values())

    for metric, value in zip(names_to_values.keys(), metric_values):
      logging.info('Metric %s has value: %f', metric, value)

************************************************
* Evaluating a Checkpointed Model with Metrics *
************************************************

Often, one wants to evaluate a model checkpoint saved on disk. This can be
performed once or repeatedly on a set schedule.

To evaluate a particular model, users define zero or more metrics and zero or
more summaries and call the evaluation_loop method:

  # Create model and obtain the predictions:
  images, labels = LoadData(...)
  predictions = MyModel(images)

  # Choose the metrics to compute:
  names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
      "accuracy": slim.metrics.accuracy(predictions, labels),
      "mse": slim.metrics.mean_squared_error(predictions, labels),
  })

  # Define the summaries to write:
  for metric_name, metric_value in metrics_to_values.iteritems():
    tf.scalar_summary(metric_name, metric_value)

  checkpoint_dir = '/tmp/my_model_dir/'
  log_dir = '/tmp/my_model_eval/'

  # We'll evaluate 1000 batches:
  num_evals = 1000

  # Evaluate every 10 minutes:
  slim.evaluation_loop(
      master='',
      checkpoint_dir,
      logdir,
      num_evals=num_evals,
      eval_op=names_to_updates.values(),
      summary_op=tf.merge_summary(summary_ops),
      eval_interval_secs=600)

**************************************************
* Evaluating a Checkpointed Model with Summaries *
**************************************************

At times, an evaluation can be performed without metrics at all but rather
with only summaries. The user need only leave out the 'eval_op' argument:

  # Create model and obtain the predictions:
  images, labels = LoadData(...)
  predictions = MyModel(images)

  # Define the summaries to write:
  tf.scalar_summary(...)
  tf.histogram_summary(...)

  checkpoint_dir = '/tmp/my_model_dir/'
  log_dir = '/tmp/my_model_eval/'

  # Evaluate once every 10 minutes.
  slim.evaluation_loop(
      master='',
      checkpoint_dir,
      logdir,
      num_evals=1,
      summary_op=tf.merge_summary(summary_ops),
      eval_interval_secs=600)

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.training import summary_io
from tensorflow.python.training import supervisor
from tensorflow.python.training import training_util

__all__ = ['evaluation', 'evaluation_loop', 'wait_for_new_checkpoint']


def wait_for_new_checkpoint(checkpoint_dir,
                            last_checkpoint,
                            seconds_to_sleep=1,
                            timeout=None):
  """Waits until a new checkpoint file is found.

  Args:
    checkpoint_dir: The directory in which checkpoints are saved.
    last_checkpoint: The last checkpoint path used.
    seconds_to_sleep: The number of seconds to sleep for before looking for a
      new checkpoint.
    timeout: The maximum amount of time to wait. If left as `None`, then the
      process will wait indefinitely.

  Returns:
    a new checkpoint path, or None if the timeout was reached.
  """
  stop_time = time.time() + timeout if timeout is not None else None
  while True:
    checkpoint_path = tf_saver.latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None or checkpoint_path == last_checkpoint:
      if stop_time is not None and time.time() + seconds_to_sleep > stop_time:
        return None
      time.sleep(seconds_to_sleep)
    else:
      return checkpoint_path


def evaluation(sess,
               num_evals=1,
               init_op=None,
               init_op_feed_dict=None,
               eval_op=None,
               eval_op_feed_dict=None,
               final_op=None,
               final_op_feed_dict=None,
               summary_op=None,
               summary_op_feed_dict=None,
               summary_writer=None,
               global_step=None):
  """Performs a single evaluation run.

  A single evaluation consists of several steps run in the following order:
  (1) an initialization op, (2) an evaluation op which is executed `num_evals`
  times (3) a finalization op and (4) the execution of a summary op which is
  written out using a summary writer.

  Args:
    sess: The current TensorFlow `Session`.
    num_evals: The number of times to execute `eval_op`.
    init_op: An operation run at the beginning of evaluation.
    init_op_feed_dict: A feed dictionary to use when executing `init_op`.
    eval_op: A operation run `num_evals` times.
    eval_op_feed_dict: The feed dictionary to use when executing the `eval_op`.
    final_op: An operation to execute after all of the `eval_op` executions. The
      value of `final_op` is returned.
    final_op_feed_dict: A feed dictionary to use when executing `final_op`.
    summary_op: A summary op executed after `eval_op` and `finalize_op`.
    summary_op_feed_dict: An optional feed dictionary to use when executing the
      `summary_op`.
    summary_writer: The summery writer used if `summary_op` is provided.
    global_step: the global step variable. If left as `None`, then
      slim.variables.global_step() is used.

  Returns:
    The value of `final_op` or `None` if `final_op` is `None`.

  Raises:
    ValueError: if `summary_op` is provided but `global_step` is `None`.
  """
  if init_op is not None:
    logging.info('Executing init op')
    sess.run(init_op, init_op_feed_dict)

  if eval_op is not None:
    logging.info('Executing eval ops')
    for i in range(int(num_evals)):
      logging.info('Executing eval_op %d/%d', i + 1, num_evals)
      sess.run(eval_op, eval_op_feed_dict)

  if final_op is not None:
    logging.info('Executing final op')
    final_op_value = sess.run(final_op, final_op_feed_dict)
  else:
    final_op_value = None

  if summary_op is not None:
    logging.info('Executing summary op')
    if global_step is None:
      global_step = variables.get_or_create_global_step()

    global_step = training_util.global_step(sess, global_step)
    summary = sess.run(summary_op, summary_op_feed_dict)
    summary_writer.add_summary(summary, global_step)
    summary_writer.flush()

  return final_op_value


_USE_DEFAULT = 0


def evaluation_loop(master,
                    checkpoint_dir,
                    logdir,
                    num_evals=1,
                    eval_op=None,
                    eval_op_feed_dict=None,
                    final_op=None,
                    final_op_feed_dict=None,
                    summary_op=_USE_DEFAULT,
                    summary_op_feed_dict=None,
                    variables_to_restore=None,
                    eval_interval_secs=60,
                    max_number_of_evaluations=None,
                    session_config=None):
  """Runs TF-Slim's Evaluation Loop.

  Args:
    master: The BNS address of the TensorFlow master.
    checkpoint_dir: The directory where checkpoints are stored.
    logdir: The directory where the TensorFlow summaries are written to.
    num_evals: The number of times to run `eval_op`.
    eval_op: A operation run `num_evals` times.
    eval_op_feed_dict: The feed dictionary to use when executing the `eval_op`.
    final_op: An operation to execute after all of the `eval_op` executions. The
      value of `final_op` is returned.
    final_op_feed_dict: A feed dictionary to use when executing `final_op`.
    summary_op: The summary_op to evaluate after running TF-Slims metric ops. By
      default the summary_op is set to tf.merge_all_summaries().
    summary_op_feed_dict: An optional feed dictionary to use when running the
      `summary_op`.
    variables_to_restore: A list of TensorFlow variables to restore during
      evaluation. If the argument is left as `None` then
      slim.variables.GetVariablesToRestore() is used.
    eval_interval_secs: The minimum number of seconds between evaluations.
    max_number_of_evaluations: the max number of iterations of the evaluation.
      If the value is left as 'None', the evaluation continues indefinitely.
    session_config: An instance of `tf.ConfigProto` that will be used to
      configure the `Session`. If left as `None`, the default will be used.
  """
  if summary_op == _USE_DEFAULT:
    summary_op = logging_ops.merge_all_summaries()

  global_step = variables.get_or_create_global_step()

  init_op = control_flow_ops.group(tf_variables.initialize_all_variables(),
                                   tf_variables.initialize_local_variables(),
                                   data_flow_ops.initialize_all_tables())

  saver = tf_saver.Saver(variables_to_restore or
                         variables.get_variables_to_restore())

  summary_writer = summary_io.SummaryWriter(logdir)

  sv = supervisor.Supervisor(graph=ops.get_default_graph(),
                             logdir=logdir,
                             init_op=init_op,
                             summary_op=None,
                             summary_writer=None,
                             global_step=None,
                             saver=saver)

  last_checkpoint = None
  number_of_evaluations = 0
  while True:
    last_checkpoint = wait_for_new_checkpoint(checkpoint_dir, last_checkpoint)
    start = time.time()
    logging.info('Starting evaluation at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                           time.gmtime()))

    with sv.managed_session(
        master, start_standard_services=False, config=session_config) as sess:
      sv.saver.restore(sess, last_checkpoint)
      sv.start_queue_runners(sess)
      evaluation(sess,
                 num_evals=num_evals,
                 eval_op=eval_op,
                 eval_op_feed_dict=eval_op_feed_dict,
                 final_op=final_op,
                 final_op_feed_dict=final_op_feed_dict,
                 summary_op=summary_op,
                 summary_op_feed_dict=summary_op_feed_dict,
                 summary_writer=summary_writer,
                 global_step=global_step)

    logging.info('Finished evaluation at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                           time.gmtime()))
    number_of_evaluations += 1
    if (max_number_of_evaluations and
        number_of_evaluations >= max_number_of_evaluations):
      logging.info('Reached max_number_of_evaluations=%s. Exit',
                   max_number_of_evaluations)
      break

    time_to_next_eval = start + eval_interval_secs - time.time()
    if time_to_next_eval > 0:
      time.sleep(time_to_next_eval)
