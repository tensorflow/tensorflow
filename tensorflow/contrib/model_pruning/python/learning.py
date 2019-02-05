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
"""Wrapper around tf-slim's training code contrib/slim/python/slim/learning.py
to support training of pruned models

*******************************************************************
* A simple working training script with support for model pruning *
*******************************************************************

  # Load data and create the model:
  images, labels = LoadData(...)
  predictions = MyModel(images)

  # Define the loss:
  slim.losses.log_loss(predictions, labels)
  total_loss = slim.losses.get_total_loss()

  # Define the optimizer:
  optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate, FLAGS.momentum)

  # Create the train_op
  train_op = slim.learning.create_train_op(total_loss, optimizer)

  # Parse pruning hyperparameters
  pruning_hparams = pruning.get_pruning_hparams().parse(FLAGS.pruning_hparams)

  # Create a pruning object using the pruning_hparams
  p = pruning.Pruning(pruning_hparams)

  # Add mask update ops to the graph
  mask_update_op = p.conditional_mask_update_op()

  # Run training.
  learning.train(train_op,
                 my_log_dir,
                 mask_update_op)
  see contrib/slim/python/slim/learning.py for additional examples
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import slim as _slim

_USE_DEFAULT = 0
train_step = _slim.learning.train_step


def train(train_op,
          logdir,
          mask_update_op,
          train_step_fn=train_step,
          train_step_kwargs=_USE_DEFAULT,
          log_every_n_steps=1,
          graph=None,
          master='',
          is_chief=True,
          global_step=None,
          number_of_steps=None,
          init_op=_USE_DEFAULT,
          init_feed_dict=None,
          local_init_op=_USE_DEFAULT,
          init_fn=None,
          ready_op=_USE_DEFAULT,
          summary_op=_USE_DEFAULT,
          save_summaries_secs=600,
          summary_writer=_USE_DEFAULT,
          startup_delay_steps=0,
          saver=None,
          save_interval_secs=600,
          sync_optimizer=None,
          session_config=None,
          trace_every_n_steps=None):
  """Wrapper around tf-slim's train function.

  Runs a training loop using a TensorFlow supervisor.
  When the sync_optimizer is supplied, gradient updates are applied
  synchronously. Otherwise, gradient updates are applied asynchronous.

  Args:
    train_op: A `Tensor` that, when executed, will apply the gradients and
      return the loss value.
    logdir: The directory where training logs are written to. If None, model
      checkpoints and summaries will not be written.
    mask_update_op: Operation that upon execution updates the weight masks and
      thresholds.
    train_step_fn: The function to call in order to execute a single gradient
      step. The function must have take exactly four arguments: the current
      session, the `train_op` `Tensor`, a global step `Tensor` and a dictionary.
    train_step_kwargs: A dictionary which is passed to the `train_step_fn`. By
      default, two `Boolean`, scalar ops called "should_stop" and "should_log"
      are provided.
    log_every_n_steps: The frequency, in terms of global steps, that the loss
      and global step and logged.
    graph: The graph to pass to the supervisor. If no graph is supplied the
      default graph is used.
    master: The address of the tensorflow master.
    is_chief: Specifies whether or not the training is being run by the primary
      replica during replica training.
    global_step: The `Tensor` representing the global step. If left as `None`,
      then slim.variables.get_or_create_global_step() is used.
    number_of_steps: The max number of gradient steps to take during training,
      as measured by 'global_step': training will stop if global_step is
      greater than 'number_of_steps'. If the value is left as None, training
      proceeds indefinitely.
    init_op: The initialization operation. If left to its default value, then
      the session is initialized by calling `tf.global_variables_initializer()`.
    init_feed_dict: A feed dictionary to use when executing the `init_op`.
    local_init_op: The local initialization operation. If left to its default
      value, then the session is initialized by calling
      `tf.local_variables_initializer()` and `tf.tables_initializer()`.
    init_fn: An optional callable to be executed after `init_op` is called. The
      callable must accept one argument, the session being initialized.
    ready_op: Operation to check if the model is ready to use. If left to its
      default value, then the session checks for readiness by calling
      `tf.report_uninitialized_variables()`.
    summary_op: The summary operation.
    save_summaries_secs: How often, in seconds, to save summaries.
    summary_writer: `SummaryWriter` to use.  Can be `None`
      to indicate that no summaries should be written. If unset, we
      create a SummaryWriter.
    startup_delay_steps: The number of steps to wait for before beginning. Note
      that this must be 0 if a sync_optimizer is supplied.
    saver: Saver to save checkpoints. If None, a default one will be created
      and used.
    save_interval_secs: How often, in seconds, to save the model to `logdir`.
    sync_optimizer: an instance of tf.train.SyncReplicasOptimizer, or a list of
      them. If the argument is supplied, gradient updates will be synchronous.
      If left as `None`, gradient updates will be asynchronous.
    session_config: An instance of `tf.ConfigProto` that will be used to
      configure the `Session`. If left as `None`, the default will be used.
    trace_every_n_steps: produce and save a `Timeline` in Chrome trace format
      and add it to the summaries every `trace_every_n_steps`. If None, no trace
      information will be produced or saved.

  Returns:
    the value of the loss function after training.

  Raises:
    ValueError: if `train_op` is empty or if `startup_delay_steps` is
      non-zero when `sync_optimizer` is supplied, if `number_of_steps` is
      negative, or if `trace_every_n_steps` is not `None` and no `logdir` is
      provided.
  """

  def train_step_with_pruning_fn(sess, train_op, global_step,
                                 train_step_kwargs):
    total_loss, should_stop = train_step_fn(sess, train_op, global_step,
                                            train_step_kwargs)
    sess.run(mask_update_op)
    return total_loss, should_stop

  total_loss, _ = _slim.learning.train(
      train_op,
      logdir,
      train_step_fn=train_step_with_pruning_fn,
      train_step_kwargs=train_step_kwargs,
      log_every_n_steps=log_every_n_steps,
      graph=graph,
      master=master,
      is_chief=is_chief,
      global_step=global_step,
      number_of_steps=number_of_steps,
      init_op=init_op,
      init_feed_dict=init_feed_dict,
      local_init_op=local_init_op,
      init_fn=init_fn,
      ready_op=ready_op,
      summary_op=summary_op,
      save_summaries_secs=save_summaries_secs,
      summary_writer=summary_writer,
      startup_delay_steps=startup_delay_steps,
      saver=saver,
      save_interval_secs=save_interval_secs,
      sync_optimizer=sync_optimizer,
      session_config=session_config,
      trace_every_n_steps=trace_every_n_steps)

  return total_loss
