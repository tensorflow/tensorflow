# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

This file contains helper functions for evaluating TensorFlow models.

**********************
* Evaluating Metrics *
**********************

In the most simple use case, a session has already been created along with an
initialized (and perhaps trained) model. In this case, one need only instantiate
a set of metrics and call their update operations:

  # Create the session and model:
  sess = ...
  labels = ...
  predictions = ...

  # Specify the metrics:
  accuracy, accuracy_update_op = metrics.streaming_accuracy(labels, predictions)
  mean_absolute_diff, mean_absolute_diff_update_op = metrics.MeanAbsoluteDiff(
      labels, predictions)

  # Aggregate data from 100 batches of data:
  with tf.Session() as sess:
    sess.run(tf.initialize_local_variables())
    slim.evaluation.evaluate_once(
      sess,
      update_ops=[accuracy_update_op, mean_absolute_diff_update_op],
      num_evals=100)

    # Print the results:
    print 'Accuracy = %f' % sess.run(accuracy)
    print 'Mean Absolute Diff = %f' % sess.run(mean_absolute_diff)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.training import summary_io
from tensorflow.python.training import supervisor
from tensorflow.python.training import training_util


def wait_for_new_checkpoint(checkpoint_dir, last_checkpoint,
                            seconds_to_sleep=1):
  """Waits until a new checkpoint file is found.

  Args:
    checkpoint_dir: The directory in which checkpoints are saved.
    last_checkpoint: The last checkpoint path used.
    seconds_to_sleep: The number of seconds to sleep for before looking for a
      new checkpoint.

  Returns:
    a new checkpoint path.
  """
  while True:
    checkpoint_path = tf_saver.latest_checkpoint(checkpoint_dir)
    if checkpoint_path == last_checkpoint:
      time.sleep(seconds_to_sleep)
    else:
      return checkpoint_path


def evaluation(
    sess,
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

  A single evaluation consistents of several steps run in the following order:
  (1) an initialization op, (2) an evaluation op which is executed `num_evals`
  times (3) a finalization op and (4) the execution of a summary op which is
  written out using a summary writer.

  Args:
    sess: The current Tensorflow `Session`.
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
    sess.run(init_op, init_op_feed_dict)

  if eval_op is not None:
    for i in range(int(num_evals)):
      logging.info('Executing eval_op %d/%d', i+1, num_evals)
      sess.run(eval_op, eval_op_feed_dict)

  if final_op:
    final_op_value = sess.run(final_op, final_op_feed_dict)
  else:
    final_op_value = None

  if summary_op is not None:
    if global_step is None:
      global_step = variables.global_step()

    global_step = training_util.global_step(sess, global_step)
    summary = sess.run(summary_op, summary_op_feed_dict)
    summary_writer.add_summary(summary, global_step)
    summary_writer.flush()

  return final_op_value


def evaluation_loop(master, checkpoint_dir, logdir, num_evals=1,
                    eval_op=None, eval_op_feed_dict=None,
                    final_op=None, final_op_feed_dict=None, summary_op=None,
                    summary_op_feed_dict=None, variables_to_restore=None,
                    eval_interval_secs=60):
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
    summary_op: The summary_op to evaluate after running TF-Slims metric ops.
    summary_op_feed_dict: An optional feed dictionary to use when running the
      `summary_op`.
    variables_to_restore: A list of TensorFlow variables to restore during
      evaluation. If the argument is left as `None` then
      slim.variables.GetVariablesToRestore() is used.
    eval_interval_secs: The minimum number of seconds between evaluations.
  """
  global_step = variables.get_or_create_global_step()

  init_op = control_flow_ops.group(
      tf_variables.initialize_all_variables(),
      tf_variables.initialize_local_variables(),
      tf_variables.initialize_all_tables())

  saver = tf_saver.Saver(
      variables_to_restore or variables.get_variables_to_restore())

  summary_writer = summary_io.SummaryWriter(logdir)

  sv = supervisor.Supervisor(
      graph=ops.get_default_graph(),
      logdir=logdir,
      init_op=init_op,
      summary_op=None,
      summary_writer=None,
      global_step=None,
      saver=saver)

  last_checkpoint = None
  while True:
    last_checkpoint = wait_for_new_checkpoint(checkpoint_dir, last_checkpoint)
    start = time.time()
    logging.info(
        'Starting evaluation at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                  time.gmtime()))

    with sv.managed_session(master, start_standard_services=False) as sess:
      sv.start_queue_runners(sess)
      sv.saver.restore(sess, last_checkpoint)
      evaluation(
          sess,
          num_evals=num_evals,
          eval_op=eval_op,
          eval_op_feed_dict=eval_op_feed_dict,
          final_op=final_op,
          final_op_feed_dict=final_op_feed_dict,
          summary_op=summary_op,
          summary_op_feed_dict=summary_op_feed_dict,
          summary_writer=summary_writer,
          global_step=global_step)

    logging.info(
        'Finished evaluation at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                  time.gmtime()))
    time_to_next_eval = start + eval_interval_secs - time.time()
    if time_to_next_eval > 0:
      time.sleep(time_to_next_eval)
