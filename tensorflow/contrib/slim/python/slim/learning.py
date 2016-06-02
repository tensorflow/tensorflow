# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Contains TF-Slim code for training models.

This script contains various functions for training models which includes
creating train tensors, altering gradients and optimizing loss functions. The
following presents an example of how to use the learning module:

  g = tf.Graph()

  # Setup the model and losses
  images, labels = LoadData(...)
  predictions = CreateNetwork(images)
  total_loss = slim.losses.log_loss(predictions, labels)

  # Define the optimizer.
  optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate, FLAGS.momentum)

  # Set up training gradients and dependencies.
  train_tensor = slim.learning.create_train_tensor(total_loss, optimizer)

  # Run training.
  learning.train(train_tensor, my_log_dir)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time

from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import optimizer as tf_optimizer
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.training import supervisor
from tensorflow.python.training import sync_replicas_optimizer
from tensorflow.python.training import training_util

__all__ = [
    'clip_gradient_norms',
    'multiply_gradients',
    'create_train_op',
    'train'
]


def clip_gradient_norms(gradients_to_variables, max_norm):
  """Clips the gradients by the given value.

  Args:
    gradients_to_variables: A list of gradient to variable pairs (tuples).
    max_norm: the maximum norm value.

  Returns:
    A list of clipped gradient to variable pairs.
  """
  clipped_grads_and_vars = []
  for grad, var in gradients_to_variables:
    if grad is not None:
      if isinstance(grad, ops.IndexedSlices):
        tmp = clip_ops.clip_by_norm(grad.values, max_norm)
        grad = ops.IndexedSlices(tmp, grad.indices, grad.dense_shape)
      else:
        grad = clip_ops.clip_by_norm(grad, max_norm)
    clipped_grads_and_vars.append((grad, var))
  return clipped_grads_and_vars


def multiply_gradients(grads_and_vars, gradient_multipliers):
  """Multiply specified gradients.

  Args:
    grads_and_vars: A list of gradient to variable pairs (tuples).
    gradient_multipliers: A map from either `Variables` or `Variable` op names
      to the coefficient by which the associated gradient should be scaled.

  Returns:
    The updated list of gradient to variable pairs.

  Raises:
    ValueError: If `grads_and_vars` is not a list or if `gradient_multipliers`
    is empty or None or if `gradient_multipliers` is not a dictionary.
  """
  if not isinstance(grads_and_vars, list):
    raise ValueError('`grads_and_vars` must be a list.')
  if not gradient_multipliers:
    raise ValueError('`gradient_multipliers` is empty.')
  if not isinstance(gradient_multipliers, dict):
    raise ValueError('`gradient_multipliers` must be a dict.')

  multiplied_grads_and_vars = []
  for grad, var in grads_and_vars:
    if var in gradient_multipliers or var.op.name in gradient_multipliers:
      key = var if var in gradient_multipliers else var.op.name
      if grad is None:
        raise ValueError('Requested multiple of `None` gradient.')

      if isinstance(grad, ops.IndexedSlices):
        tmp = grad.values * constant_op.constant(gradient_multipliers[key],
                                                 dtype=grad.dtype)
        grad = ops.IndexedSlices(tmp, grad.indices, grad.dense_shape)
      else:
        grad *= constant_op.constant(gradient_multipliers[key],
                                     dtype=grad.dtype)
    multiplied_grads_and_vars.append((grad, var))
  return multiplied_grads_and_vars


def create_train_op(
    total_loss,
    optimizer,
    global_step=None,
    update_ops=None,
    variables_to_train=None,
    clip_gradient_norm=0,
    summarize_gradients=False,
    gate_gradients=tf_optimizer.Optimizer.GATE_OP,
    aggregation_method=None,
    colocate_gradients_with_ops=False):
  """Creates an `Operation` that evaluates the gradients and returns the loss.

  Args:
    total_loss: A `Tensor` representing the total loss.
    optimizer: A tf.Optimizer to use for computing the gradients.
    global_step: A `Tensor` representing the global step variable. If left as
      `None`, then slim.variables.global_step() is used.
    update_ops: an optional list of updates to execute. Note that the update_ops
      that are used are the union of those update_ops passed to the function and
      the value of slim.ops.GetUpdateOps(). Therefore, if `update_ops` is None,
      then the value of slim.ops.GetUpdateOps() is still used.
    variables_to_train: an optional list of variables to train. If None, it will
      default to all tf.trainable_variables().
    clip_gradient_norm: If greater than 0 then the gradients would be clipped
      by it.
    summarize_gradients: Whether or not add summaries for each gradient.
    gate_gradients: How to gate the computation of gradients. See tf.Optimizer.
    aggregation_method: Specifies the method used to combine gradient terms.
      Valid values are defined in the class `AggregationMethod`.
    colocate_gradients_with_ops: Whether or not to try colocating the gradients
      with the ops that generated them.

  Returns:
    A `Tensor` that when evaluated, computes the gradients and returns the total
      loss value.
  """
  if global_step is None:
    global_step = variables.get_or_create_global_step()

  update_ops = set(update_ops or [])

  # Make sure update_ops are computed before total_loss.
  if update_ops:
    with control_flow_ops.control_dependencies(update_ops):
      barrier = control_flow_ops.no_op(name='update_barrier')
    total_loss = control_flow_ops.with_dependencies([barrier], total_loss)

  if variables_to_train is None:
    # Default to tf.trainable_variables()
    variables_to_train = tf_variables.trainable_variables()
  else:
    # Make sure that variables_to_train are in tf.trainable_variables()
    for v in variables_to_train:
      assert v in tf_variables.trainable_variables()

  assert variables_to_train

  # Create the gradients. Note that apply_gradients adds the gradient
  # computation to the current graph.
  grads = optimizer.compute_gradients(
      total_loss, variables_to_train, gate_gradients=gate_gradients,
      aggregation_method=aggregation_method,
      colocate_gradients_with_ops=colocate_gradients_with_ops)

  # Clip gradients.
  if clip_gradient_norm > 0:
    grads = clip_gradient_norms(grads, clip_gradient_norm)

  # Summarize gradients.
  if summarize_gradients:
    for grad, var in grads:
      if grad is not None:
        if isinstance(grad, ops.IndexedSlices):
          grad_values = grad.values
        else:
          grad_values = grad
        logging_ops.histogram_summary(var.op.name + ':gradient', grad_values)
        logging_ops.histogram_summary(var.op.name + ':gradient_norm',
                                      clip_ops.global_norm([grad_values]))
      else:
        logging.info('Var %s has no gradient', var.op.name)

  # Create gradient updates.
  grad_updates = optimizer.apply_gradients(grads, global_step=global_step)

  # Make sure total_loss is valid.
  total_loss = array_ops.check_numerics(total_loss, 'LossTensor is inf or nan')

  # Ensure the train_tensor computes grad_updates.
  return control_flow_ops.with_dependencies([grad_updates], total_loss)


def _wait_for_step(sess, global_step, step):
  """Wait till the global step has reached at least 'step'.

  Args:
    sess: A session.
    global_step: A Tensor.
    step: Int.  The global step to reach.
  """
  while True:
    if training_util.global_step(sess, global_step) >= step:
      break
    time.sleep(1.0)


def train_loop(sv,
               sess,
               train_op,
               should_stop_op,
               should_log_op,
               global_step,
               cleanup_op=None):
  """Runs the training loop.

  Args:
    sv: The supervisor instance.
    sess: The session.
    train_op: An `Operation` that evaluates the gradients and returns the
      total loss.
    should_stop_op: A boolean `Tensor` that signals the end of training.
    should_log_op: A boolean `Tensor` that signals whether or not we should log
      the global step and current loss.
    global_step: A `Tensor` representing the global training step.
    cleanup_op: An operation to run if an exception is thrown.

  Returns:
    total_loss: The total loss value after training.
  """
  total_loss = 0

  try:
    while not sv.should_stop():
      start_time = time.time()
      total_loss, np_global_step, np_should_log, np_should_stop = sess.run(
          [train_op, global_step, should_log_op, should_stop_op])
      time_elapsed = time.time() - start_time

      if np_should_log:
        logging.info('global step %d: loss = %.4f (%.2f sec)',
                     np_global_step, total_loss, time_elapsed)
      if np_should_stop:
        break
  finally:
    if sv.is_chief and cleanup_op is not None:
      sess.run(cleanup_op)
  return total_loss

_USE_DEFAULT = 0


def train(
    train_op,
    logdir,
    log_every_n_steps=1,
    graph=None,
    master='',
    is_chief=True,
    global_step=None,
    number_of_steps=None,
    init_op=_USE_DEFAULT,
    init_feed_dict=None,
    init_fn=None,
    summary_op=_USE_DEFAULT,
    save_summaries_secs=600,
    startup_delay_steps=0,
    saver=None,
    save_interval_secs=600,
    sync_optimizer=None):
  """Runs a training loop using a TensorFlow supervisor.

  When the sync_optimizer is supplied, gradient updates are applied
  synchronously. Otherwise, gradient updates are applied asynchronous.

  Args:
    train_op: A `Tensor` that, when executed, will apply the gradients and
      return the loss value.
    logdir: the directory where training logs are written to.
    log_every_n_steps: The frequency, in terms of global steps, that the loss
      and global step and logged.
    graph: The graph to pass to the supervisor. If no graph is supplied the
      default graph is used.
    master: The BNS name of the tensorflow master.
    is_chief: Specifies whether or not the training is being run by the primary
      replica during replica training.
    global_step: The `Tensor` representing the global step. If left as `None`,
      then slim.variables.get_or_create_global_step() is used.
    number_of_steps: The max number of gradient steps to take during training.
      If the value is left as None, training proceeds indefinitely.
    init_op: The initialization operation.
    init_feed_dict: A feed dictionary to use when executing the `init_op`.
    init_fn: An optional callable to be executed after `init_op` is called. The
      callable must accept one argument, the session being initialized.
    summary_op: The summary operation.
    save_summaries_secs: How often, in seconds, to save summaries.
    startup_delay_steps: The number of steps to wait for before beginning. Note
      that this must be 0 if a sync_optimizer is supplied.
    saver: Saver to save checkpoints. If none, a default one will be created
      and used.
    save_interval_secs: How often, in seconds, to save the model to `logdir`.
    sync_optimizer: an instance of tf.train.SyncReplicasOptimizer. If the
      argument is supplied, gradient updates will be synchronous. If left as
      `None`, gradient updates will be asynchronous.

  Returns:
    the value of the loss function after training.

  Raises:
    ValueError: if `train_op` is empty or if `startup_delay_steps` is
      non-zero when `sync_optimizer` is supplied, or if `number_of_steps` is
      negative.
  """
  if train_op is None:
    raise ValueError('train_op cannot be None.')

  if sync_optimizer and startup_delay_steps > 0:
    raise ValueError(
        'startup_delay_steps must be zero when sync_optimizer is supplied.')

  if number_of_steps is not None and number_of_steps <= 0:
    raise ValueError(
        '`number_of_steps` must be either None or a positive number.')

  graph = graph or ops.get_default_graph()
  if global_step is None:
    global_step = variables.get_or_create_global_step()
  saver = saver or tf_saver.Saver()

  if init_op is None:
    init_op = control_flow_ops.group(
        tf_variables.initialize_all_variables(),
        tf_variables.initialize_local_variables(),
        tf_variables.initialize_all_tables())

  if summary_op == _USE_DEFAULT:
    summary_op = logging_ops.merge_all_summaries()

  local_init_op = None
  cleanup_op = None

  if is_chief and sync_optimizer:
    if not isinstance(sync_optimizer,
                      sync_replicas_optimizer.SyncReplicasOptimizer):
      raise ValueError(
          '`sync_optimizer` must be a tf.train.SyncReplicasOptimizer')

    # Need to create these BEFORE the supervisor finalizes the graph:
    local_init_op = sync_optimizer.get_init_tokens_op()
    chief_queue_runner = sync_optimizer.get_chief_queue_runner()
    cleanup_op = sync_optimizer.get_clean_up_op()

  if number_of_steps:
    # Need to subtract 1 since the check for greater/equality is done
    # concurrently with the increment of global_step.
    # TODO(nsilberman): add a dependency to ensure the order of operations.
    should_stop_op = math_ops.greater_equal(global_step, number_of_steps-1)
  else:
    should_stop_op = constant_op.constant(False)

  should_log_op = math_ops.equal(math_ops.mod(global_step, log_every_n_steps),
                                 0)

  sv = supervisor.Supervisor(
      graph=graph,
      is_chief=is_chief,
      logdir=logdir,
      init_op=init_op,
      init_feed_dict=init_feed_dict,
      local_init_op=local_init_op,
      summary_op=summary_op,
      global_step=global_step,
      saver=saver,
      save_summaries_secs=save_summaries_secs,
      save_model_secs=save_interval_secs,
      init_fn=init_fn)

  with sv.managed_session(master, start_standard_services=False) as sess:
    if is_chief:
      sv.start_standard_services(sess)
    elif not is_chief and startup_delay_steps > 0:
      _wait_for_step(sess, global_step,
                     min(startup_delay_steps, number_of_steps or sys.maxint))
    sv.start_queue_runners(sess)
    if is_chief and sync_optimizer:
      sv.start_queue_runners(sess, [chief_queue_runner])

    total_loss = train_loop(
        sv, sess, train_op, should_stop_op, should_log_op, global_step,
        cleanup_op)

    # This waits for service threads to finish.
    sv.Stop()

    if sv.is_chief:
      logging.info('Finished training! Saving model to disk.')
      sv.saver.save(sess, sv.save_path, global_step=sv.global_step)

    return total_loss
