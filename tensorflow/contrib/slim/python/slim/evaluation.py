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

  inital_op = tf.group(
      tf.global_variables_initializer(),
      tf.local_variables_initializer())

  with tf.Session() as sess:
    metric_values = slim.evaluation(
        sess,
        num_evals=1,
        inital_op=initial_op,
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
    tf.summary.scalar(metric_name, metric_value)

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
      summary_op=tf.contrib.deprecated.merge_summary(summary_ops),
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
  tf.summary.scalar(...)
  tf.summary.histogram(...)

  checkpoint_dir = '/tmp/my_model_dir/'
  log_dir = '/tmp/my_model_eval/'

  # Evaluate once every 10 minutes.
  slim.evaluation_loop(
      master='',
      checkpoint_dir,
      logdir,
      num_evals=1,
      summary_op=tf.contrib.deprecated.merge_summary(summary_ops),
      eval_interval_secs=600)

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.training.python.training import evaluation
from tensorflow.python.summary import summary
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver as tf_saver

__all__ = [
    'evaluate_once',
    'evaluation_loop',
    'wait_for_new_checkpoint',
    'checkpoints_iterator',
]

wait_for_new_checkpoint = evaluation.wait_for_new_checkpoint
checkpoints_iterator = evaluation.checkpoints_iterator

_USE_DEFAULT = 0


def evaluate_once(master,
                  checkpoint_path,
                  logdir,
                  num_evals=1,
                  initial_op=None,
                  initial_op_feed_dict=None,
                  eval_op=None,
                  eval_op_feed_dict=None,
                  final_op=None,
                  final_op_feed_dict=None,
                  summary_op=_USE_DEFAULT,
                  summary_op_feed_dict=None,
                  variables_to_restore=None,
                  session_config=None):
  """Evaluates the model at the given checkpoint path.

  Args:
    master: The BNS address of the TensorFlow master.
    checkpoint_path: The path to a checkpoint to use for evaluation.
    logdir: The directory where the TensorFlow summaries are written to.
    num_evals: The number of times to run `eval_op`.
    initial_op: An operation run at the beginning of evaluation.
    initial_op_feed_dict: A feed dictionary to use when executing `initial_op`.
    eval_op: A operation run `num_evals` times.
    eval_op_feed_dict: The feed dictionary to use when executing the `eval_op`.
    final_op: An operation to execute after all of the `eval_op` executions. The
      value of `final_op` is returned.
    final_op_feed_dict: A feed dictionary to use when executing `final_op`.
    summary_op: The summary_op to evaluate after running TF-Slims metric ops. By
      default the summary_op is set to tf.summary.merge_all().
    summary_op_feed_dict: An optional feed dictionary to use when running the
      `summary_op`.
    variables_to_restore: A list of TensorFlow variables to restore during
      evaluation. If the argument is left as `None` then
      slim.variables.GetVariablesToRestore() is used.
    session_config: An instance of `tf.ConfigProto` that will be used to
      configure the `Session`. If left as `None`, the default will be used.

  Returns:
    The value of `final_op` or `None` if `final_op` is `None`.
  """
  if summary_op == _USE_DEFAULT:
    summary_op = summary.merge_all()

  hooks = [evaluation.StopAfterNEvalsHook(num_evals),]

  if summary_op is not None:
    hooks.append(
        evaluation.SummaryAtEndHook(logdir, summary_op, summary_op_feed_dict))

  saver = None
  if variables_to_restore is not None:
    saver = tf_saver.Saver(variables_to_restore)

  return evaluation.evaluate_once(
      checkpoint_path,
      master=master,
      scaffold=monitored_session.Scaffold(
          init_op=initial_op, init_feed_dict=initial_op_feed_dict, saver=saver),
      eval_ops=eval_op,
      feed_dict=eval_op_feed_dict,
      final_ops=final_op,
      final_ops_feed_dict=final_op_feed_dict,
      hooks=hooks,
      config=session_config)


def evaluation_loop(master,
                    checkpoint_dir,
                    logdir,
                    num_evals=1,
                    initial_op=None,
                    initial_op_feed_dict=None,
                    eval_op=None,
                    eval_op_feed_dict=None,
                    final_op=None,
                    final_op_feed_dict=None,
                    summary_op=_USE_DEFAULT,
                    summary_op_feed_dict=None,
                    variables_to_restore=None,
                    eval_interval_secs=60,
                    max_number_of_evaluations=None,
                    session_config=None,
                    timeout=None):
  """Runs TF-Slim's Evaluation Loop.

  Args:
    master: The BNS address of the TensorFlow master.
    checkpoint_dir: The directory where checkpoints are stored.
    logdir: The directory where the TensorFlow summaries are written to.
    num_evals: The number of times to run `eval_op`.
    initial_op: An operation run at the beginning of evaluation.
    initial_op_feed_dict: A feed dictionary to use when executing `initial_op`.
    eval_op: A operation run `num_evals` times.
    eval_op_feed_dict: The feed dictionary to use when executing the `eval_op`.
    final_op: An operation to execute after all of the `eval_op` executions. The
      value of `final_op` is returned.
    final_op_feed_dict: A feed dictionary to use when executing `final_op`.
    summary_op: The summary_op to evaluate after running TF-Slims metric ops. By
      default the summary_op is set to tf.summary.merge_all().
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
    timeout: The maximum amount of time to wait between checkpoints. If left as
      `None`, then the process will wait indefinitely.

  Returns:
    The value of `final_op` or `None` if `final_op` is `None`.
  """
  if summary_op == _USE_DEFAULT:
    summary_op = summary.merge_all()

  hooks = [evaluation.StopAfterNEvalsHook(num_evals),]

  if summary_op is not None:
    hooks.append(
        evaluation.SummaryAtEndHook(logdir, summary_op, summary_op_feed_dict))

  saver = None
  if variables_to_restore is not None:
    saver = tf_saver.Saver(variables_to_restore)

  return evaluation.evaluate_repeatedly(
      checkpoint_dir,
      master=master,
      scaffold=monitored_session.Scaffold(
          init_op=initial_op, init_feed_dict=initial_op_feed_dict, saver=saver),
      eval_ops=eval_op,
      feed_dict=eval_op_feed_dict,
      final_ops=final_op,
      final_ops_feed_dict=final_op_feed_dict,
      eval_interval_secs=eval_interval_secs,
      hooks=hooks,
      config=session_config,
      max_number_of_evaluations=max_number_of_evaluations,
      timeout=timeout)
