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
"""Contains TF-Slim code for training models.

This script contains various functions for training models. These include
manipulating gradients, creating a `train_op` (an operation that computes the
loss and applies the gradients) and a training loop function. The training loop
allows the user to pass in the `train_op` and runs the optimization according
to user-specified arguments. Note that the training loop uses the tf.Supervisor
and its managed_session in its implementation to ensure the ability of worker
processes to recover from failures.

************************************
* A simple working training script *
************************************

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

  # Run training.
  slim.learning.train(train_op, my_log_dir)

*************************
* Creating the train_op *
*************************

In order to train, TF-Slim's train loop needs a train_op: an `Operation` that
(a) computes the loss, (b) applies the gradients to update the weights and
(c) returns the value of the loss. slim.learning.create_train_op creates
such an `Operation`. This function also provides the ability to manipulate
the gradients using a few arguments:

  # Create the train_op and clip the gradient norms:
  train_op = slim.learning.create_train_op(
      total_loss,
      optimizer,
      clip_gradient_norm=4)

  # Create the train_op and scale the gradients by providing a map from variable
  # name (or variable) to a scaling coefficient:
  gradient_multipliers = {
    'conv0/weights': 1.2,
    'fc8/weights': 3.4,
  }
  train_op = slim.learning.create_train_op(
      total_loss,
      optimizer,
      gradient_multipliers=gradient_multipliers)

****************************************************************
* Performing additional (non-gradient) updates during training *
****************************************************************

Many networks utilize modules, like BatchNorm, that require performing a series
of non-gradient updates during training. slim.learning.create_train_op allows
a user to pass in a list of update_ops to call along with the gradient updates.

  train_op = slim.learning.create_train_op(total_loss, optimizer, update_ops)

By default, slim.learning.create_train_op includes all update ops that are
part of the `tf.GraphKeys.UPDATE_OPS` collection. Additionally, TF-Slim's
slim.batch_norm function adds the moving mean and moving variance updates to
this collection. Consequently, users who want to use slim.batch_norm will not
need to take any additional steps in order to have the moving mean and moving
variance updates be computed.

However, users with additional, specialized updates can either override the
default update ops or simply add additional update ops to the
`tf.GraphKeys.UPDATE_OPS` collection:

  # Force TF-Slim NOT to use ANY update_ops:
  train_op = slim.learning.create_train_op(
     total_loss,
     optimizer,
     update_ops=[])

  # Use an alternative set of update ops:
  train_op = slim.learning.create_train_op(
     total_loss,
     optimizer,
     update_ops=my_other_update_ops)

  # Use an alternative set of update ops in addition to the default updates:
  tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, my_update0)
  tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, my_update1)

  train_op = slim.learning.create_train_op(
     total_loss,
     optimizer)

  # Which is the same as:
  train_op = slim.learning.create_train_op(
     total_loss,
     optimizer,
     update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS))

******************************************
* Initializing a model from a checkpoint *
******************************************

It is common to want to 'warm-start' a model from a pre-trained checkpoint.
TF-Slim provides a convenient mechanism for doing so:

  ...

  # Create the train_op
  train_op = slim.learning.create_train_op(total_loss, optimizer)

  # Create the initial assignment op
  checkpoint_path = '/path/to/old_model_checkpoint'
  variables_to_restore = slim.get_model_variables()
  init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
      checkpoint_path, variables_to_restore)

  # Create an initial assignment function.
  def InitAssignFn(sess):
      sess.run(init_assign_op, init_feed_dict)

  # Run training.
  slim.learning.train(train_op, my_log_dir, init_fn=InitAssignFn)

***************************************************************************
* Initializing a model from a checkpoint whose variable names don't match *
***************************************************************************

At times, a user may want to initialize a new model with values from a
checkpoint whose variable names do not match those of the current model. In this
case, one needs to create a mapping from the checkpoint variable names to the
current model variables. This requires only a small modification of the code
above:
  ...
  # Creates a model with two variables, var0 and var1
  predictions = MyModel(images)
  ...

  # Create the train_op
  train_op = slim.learning.create_train_op(total_loss, optimizer)

  checkpoint_path = '/path/to/old_model_checkpoint'

  # Create the mapping:
  variables_to_restore = {
      'name_var_0_in_checkpoint': slim.get_unique_variable('var0'),
      'name_var_1_in_checkpoint': slim.get_unique_variable('var1')
  }
  init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
      checkpoint_path, variables_to_restore)

  # Create an initial assignment function.
  def InitAssignFn(sess):
      sess.run(init_assign_op, init_feed_dict)

  # Run training.
  slim.learning.train(train_op, my_log_dir, init_fn=InitAssignFn)


*************************************************
* Fine-Tuning Part of a model from a checkpoint *
*************************************************

Rather than initializing all of the weights of a given model, we sometimes
only want to restore some of the weights from a checkpoint. To do this, one
need only filter those variables to initialize as follows:

  ...

  # Create the train_op
  train_op = slim.learning.create_train_op(total_loss, optimizer)

  checkpoint_path = '/path/to/old_model_checkpoint'

  # Specify the variables to restore via a list of inclusion or exclusion
  # patterns:
  variables_to_restore = slim.get_variables_to_restore(
      include=["conv"], exclude=["fc8", "fc9])
  # or
  variables_to_restore = slim.get_variables_to_restore(exclude=["conv"])

  init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
      checkpoint_path, variables_to_restore)

  # Create an initial assignment function.
  def InitAssignFn(sess):
      sess.run(init_assign_op, init_feed_dict)

  # Run training.
  slim.learning.train(train_op, my_log_dir, init_fn=InitAssignFn)

******************************************************
* Initializing model variables from values in memory *
******************************************************

One may want to initialize the weights of a model from values from an arbitrary
source (a text document, matlab file, etc). While this is technically feasible
using plain TensorFlow, it also results in the values of your weights being
stored in the graph. For large models, this becomes prohibitively large. TF-Slim
allows you to perform this initial assignment without having to store the values
of the initial model in the graph itself by using placeholders and a feed
dictionary:

  ...

  # Create the train_op
  train_op = slim.learning.create_train_op(total_loss, optimizer)

  # Create the mapping from variable names to values:
  var0_initial_value = ReadFromDisk(...)
  var1_initial_value = ReadFromDisk(...)

  var_names_to_values = {
    'var0': var0_initial_value,
    'var1': var1_initial_value,
  }
  init_assign_op, init_feed_dict = slim.assign_from_values(var_names_to_values)

  # Create an initial assignment function.
  def InitAssignFn(sess):
      sess.run(init_assign_op, init_feed_dict)

  # Run training.
  slim.learning.train(train_op, my_log_dir, init_fn=InitAssignFn)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

from tensorflow.contrib.framework.python.ops import variables
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import timeline
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import optimizer as tf_optimizer
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.training import supervisor
from tensorflow.python.training import sync_replicas_optimizer
from tensorflow.python.training import training_util

__all__ = [
    'add_gradients_summaries', 'clip_gradient_norms', 'multiply_gradients',
    'create_train_op', 'train_step', 'train'
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
        tmp = grad.values * constant_op.constant(
            gradient_multipliers[key], dtype=grad.dtype)
        grad = ops.IndexedSlices(tmp, grad.indices, grad.dense_shape)
      else:
        grad *= constant_op.constant(
            gradient_multipliers[key], dtype=grad.dtype)
    multiplied_grads_and_vars.append((grad, var))
  return multiplied_grads_and_vars


def add_gradients_summaries(grads_and_vars):
  """Add summaries to gradients.

  Args:
    grads_and_vars: A list of gradient to variable pairs (tuples).

  Returns:
    The list of created summaries.
  """
  summaries = []
  for grad, var in grads_and_vars:
    if grad is not None:
      if isinstance(grad, ops.IndexedSlices):
        grad_values = grad.values
      else:
        grad_values = grad
      summaries.append(
          summary.histogram(var.op.name + '/gradient', grad_values))
      summaries.append(
          summary.histogram(var.op.name + '/gradient_norm',
                            clip_ops.global_norm([grad_values])))
    else:
      logging.info('Var %s has no gradient', var.op.name)

  return summaries


_USE_GLOBAL_STEP = 0


def create_train_op(total_loss,
                    optimizer,
                    global_step=_USE_GLOBAL_STEP,
                    update_ops=None,
                    variables_to_train=None,
                    clip_gradient_norm=0,
                    summarize_gradients=False,
                    gate_gradients=tf_optimizer.Optimizer.GATE_OP,
                    aggregation_method=None,
                    colocate_gradients_with_ops=False,
                    gradient_multipliers=None):
  """Creates an `Operation` that evaluates the gradients and returns the loss.

  Args:
    total_loss: A `Tensor` representing the total loss.
    optimizer: A tf.Optimizer to use for computing the gradients.
    global_step: A `Tensor` representing the global step variable. If left as
      `_USE_GLOBAL_STEP`, then slim.variables.global_step() is used.
    update_ops: An optional list of updates to execute. If `update_ops` is
      `None`, then the update ops are set to the contents of the
      `tf.GraphKeys.UPDATE_OPS` collection. If `update_ops` is not `None`, but
      it doesn't contain all of the update ops in `tf.GraphKeys.UPDATE_OPS`,
      a warning will be displayed.
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
    gradient_multipliers: A dictionary of either `Variables` or `Variable` op
      names to the coefficient by which the associated gradient should be
      scaled.
  Returns:
    A `Tensor` that when evaluated, computes the gradients and returns the total
      loss value.
  """
  if global_step is _USE_GLOBAL_STEP:
    global_step = variables.get_or_create_global_step()

  # Update ops use GraphKeys.UPDATE_OPS collection if update_ops is None.
  global_update_ops = set(ops.get_collection(ops.GraphKeys.UPDATE_OPS))
  if update_ops is None:
    update_ops = global_update_ops
  else:
    update_ops = set(update_ops)
  if not global_update_ops.issubset(update_ops):
    logging.warning('update_ops in create_train_op does not contain all the '
                    ' update_ops in GraphKeys.UPDATE_OPS')

  # Make sure update_ops are computed before total_loss.
  if update_ops:
    with ops.control_dependencies(update_ops):
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
      total_loss,
      variables_to_train,
      gate_gradients=gate_gradients,
      aggregation_method=aggregation_method,
      colocate_gradients_with_ops=colocate_gradients_with_ops)

  # Scale gradients.
  if gradient_multipliers:
    with ops.name_scope('multiply_grads'):
      grads = multiply_gradients(grads, gradient_multipliers)

  # Clip gradients.
  if clip_gradient_norm > 0:
    with ops.name_scope('clip_grads'):
      grads = clip_gradient_norms(grads, clip_gradient_norm)

  # Summarize gradients.
  if summarize_gradients:
    with ops.name_scope('summarize_grads'):
      add_gradients_summaries(grads)

  # Create gradient updates.
  grad_updates = optimizer.apply_gradients(grads, global_step=global_step)

  with ops.name_scope('train_op'):
    # Make sure total_loss is valid.
    total_loss = array_ops.check_numerics(total_loss,
                                          'LossTensor is inf or nan')

    # Ensure the train_tensor computes grad_updates.
    train_op = control_flow_ops.with_dependencies([grad_updates], total_loss)

  # Add the operation used for training to the 'train_op' collection
  train_ops = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
  if train_op not in train_ops:
    train_ops.append(train_op)

  return train_op


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


def train_step(sess, train_op, global_step, train_step_kwargs):
  """Function that takes a gradient step and specifies whether to stop.

  Args:
    sess: The current session.
    train_op: An `Operation` that evaluates the gradients and returns the
      total loss.
    global_step: A `Tensor` representing the global training step.
    train_step_kwargs: A dictionary of keyword arguments.

  Returns:
    The total loss and a boolean indicating whether or not to stop training.

  Raises:
    ValueError: if 'should_trace' is in `train_step_kwargs` but `logdir` is not.
  """
  start_time = time.time()

  trace_run_options = None
  run_metadata = None
  if 'should_trace' in train_step_kwargs:
    if 'logdir' not in train_step_kwargs:
      raise ValueError('logdir must be present in train_step_kwargs when '
                       'should_trace is present')
    if sess.run(train_step_kwargs['should_trace']):
      trace_run_options = config_pb2.RunOptions(
          trace_level=config_pb2.RunOptions.FULL_TRACE)
      run_metadata = config_pb2.RunMetadata()

  total_loss, np_global_step = sess.run([train_op, global_step],
                                        options=trace_run_options,
                                        run_metadata=run_metadata)
  time_elapsed = time.time() - start_time

  if run_metadata is not None:
    tl = timeline.Timeline(run_metadata.step_stats)
    trace = tl.generate_chrome_trace_format()
    trace_filename = os.path.join(train_step_kwargs['logdir'],
                                  'tf_trace-%d.json' % np_global_step)
    logging.info('Writing trace to %s', trace_filename)
    file_io.write_string_to_file(trace_filename, trace)
    if 'summary_writer' in train_step_kwargs:
      train_step_kwargs['summary_writer'].add_run_metadata(run_metadata,
                                                           'run_metadata-%d' %
                                                           np_global_step)

  if 'should_log' in train_step_kwargs:
    if sess.run(train_step_kwargs['should_log']):
      logging.info('global step %d: loss = %.4f (%.2f sec/step)',
                   np_global_step, total_loss, time_elapsed)

  # TODO(nsilberman): figure out why we can't put this into sess.run. The
  # issue right now is that the stop check depends on the global step. The
  # increment of global step often happens via the train op, which used
  # created using optimizer.apply_gradients.
  #
  # Since running `train_op` causes the global step to be incremented, one
  # would expected that using a control dependency would allow the
  # should_stop check to be run in the same session.run call:
  #
  #   with ops.control_dependencies([train_op]):
  #     should_stop_op = ...
  #
  # However, this actually seems not to work on certain platforms.
  if 'should_stop' in train_step_kwargs:
    should_stop = sess.run(train_step_kwargs['should_stop'])
  else:
    should_stop = False

  return total_loss, should_stop


_USE_DEFAULT = 0


def train(train_op,
          logdir,
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
  """Runs a training loop using a TensorFlow supervisor.

  When the sync_optimizer is supplied, gradient updates are applied
  synchronously. Otherwise, gradient updates are applied asynchronous.

  Args:
    train_op: A `Tensor` that, when executed, will apply the gradients and
      return the loss value.
    logdir: The directory where training logs are written to. If None, model
      checkpoints and summaries will not be written.
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
    number_of_steps: The max number of gradient steps to take during training.
      If the value is left as None, training proceeds indefinitely.
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
    sync_optimizer: an instance of tf.train.SyncReplicasOptimizer. If the
      argument is supplied, gradient updates will be synchronous. If left as
      `None`, gradient updates will be asynchronous.
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
  if train_op is None:
    raise ValueError('train_op cannot be None.')

  if logdir is None:
    if summary_op != _USE_DEFAULT:
      raise ValueError('Cannot provide summary_op because logdir=None')
    if saver is not None:
      raise ValueError('Cannot provide saver because logdir=None')
    if trace_every_n_steps is not None:
      raise ValueError('Cannot provide trace_every_n_steps because '
                       'logdir=None')

  if sync_optimizer is not None and startup_delay_steps > 0:
    raise ValueError(
        'startup_delay_steps must be zero when sync_optimizer is supplied.')

  if number_of_steps is not None and number_of_steps <= 0:
    raise ValueError(
        '`number_of_steps` must be either None or a positive number.')

  graph = graph or ops.get_default_graph()
  with graph.as_default():
    if global_step is None:
      global_step = variables.get_or_create_global_step()
    saver = saver or tf_saver.Saver()

    with ops.name_scope('init_ops'):
      if init_op == _USE_DEFAULT:
        init_op = tf_variables.global_variables_initializer()

      if ready_op == _USE_DEFAULT:
        ready_op = tf_variables.report_uninitialized_variables()

      if local_init_op == _USE_DEFAULT:
        local_init_op = control_flow_ops.group(
            tf_variables.local_variables_initializer(),
            data_flow_ops.tables_initializer())

      if sync_optimizer is not None and isinstance(
          sync_optimizer, sync_replicas_optimizer.SyncReplicasOptimizer):
        with ops.control_dependencies([local_init_op] if local_init_op is
                                      not None else []):
          if is_chief:
            local_init_op = sync_optimizer.chief_init_op
          else:
            local_init_op = sync_optimizer.local_step_init_op
        ready_for_local_init_op = sync_optimizer.ready_for_local_init_op
      else:
        ready_for_local_init_op = None

    if summary_op == _USE_DEFAULT:
      summary_op = summary.merge_all()

    if summary_writer == _USE_DEFAULT:
      summary_writer = supervisor.Supervisor.USE_DEFAULT

    cleanup_op = None

    if is_chief and sync_optimizer is not None:
      if not isinstance(sync_optimizer,
                        (sync_replicas_optimizer.SyncReplicasOptimizer)):
        raise ValueError(
            '`sync_optimizer` must be a tf.train.SyncReplicasOptimizer.')

      # Need to create these BEFORE the supervisor finalizes the graph:
      init_tokens_op = sync_optimizer.get_init_tokens_op()
      chief_queue_runner = sync_optimizer.get_chief_queue_runner()
      if isinstance(sync_optimizer,
                    sync_replicas_optimizer.SyncReplicasOptimizer):
        cleanup_op = sync_optimizer.get_clean_up_op()

    if train_step_kwargs == _USE_DEFAULT:
      with ops.name_scope('train_step'):
        train_step_kwargs = {}

        if number_of_steps:
          should_stop_op = math_ops.greater_equal(global_step, number_of_steps)
        else:
          should_stop_op = constant_op.constant(False)
        train_step_kwargs['should_stop'] = should_stop_op
        train_step_kwargs['should_log'] = math_ops.equal(
            math_ops.mod(global_step, log_every_n_steps), 0)
        if is_chief and trace_every_n_steps is not None:
          train_step_kwargs['should_trace'] = math_ops.equal(
              math_ops.mod(global_step, trace_every_n_steps), 0)
          train_step_kwargs['logdir'] = logdir

  sv = supervisor.Supervisor(
      graph=graph,
      is_chief=is_chief,
      logdir=logdir,
      init_op=init_op,
      init_feed_dict=init_feed_dict,
      local_init_op=local_init_op,
      ready_for_local_init_op=ready_for_local_init_op,
      ready_op=ready_op,
      summary_op=summary_op,
      summary_writer=summary_writer,
      global_step=global_step,
      saver=saver,
      save_summaries_secs=save_summaries_secs,
      save_model_secs=save_interval_secs,
      init_fn=init_fn)

  if summary_writer is not None:
    train_step_kwargs['summary_writer'] = sv.summary_writer

  should_retry = True
  while should_retry:
    try:
      should_retry = False
      with sv.managed_session(
          master, start_standard_services=False, config=session_config) as sess:
        logging.info('Starting Session.')
        if is_chief:
          if logdir:
            sv.start_standard_services(sess)
        elif startup_delay_steps > 0:
          _wait_for_step(sess, global_step,
                         min(startup_delay_steps, number_of_steps or
                             sys.maxint))
        sv.start_queue_runners(sess)
        logging.info('Starting Queues.')
        if is_chief and sync_optimizer is not None:
          sv.start_queue_runners(sess, [chief_queue_runner])
          sess.run(init_tokens_op)
        try:
          try:
            while not sv.should_stop():
              total_loss, should_stop = train_step_fn(sess, train_op, global_step,
                                                      train_step_kwargs)
              if should_stop:
                logging.info('Stopping Training.')
                break
          except errors.OutOfRangeError:
            # OutOfRangeError is thrown when epoch limit per 
            # tf.train.limit_epochs is reached.
            logging.info('Caught OutOfRangeError. Stopping Training.')
          if logdir and sv.is_chief:
            logging.info('Finished training! Saving model to disk.')
            sv.saver.save(sess, sv.save_path, global_step=sv.global_step)
        except:
          if sv.is_chief and cleanup_op is not None:
            logging.info('About to execute sync_clean_up_op!')
            sess.run(cleanup_op)
          raise

    except errors.AbortedError:
      # Always re-run on AbortedError as it indicates a restart of one of the
      # distributed tensorflow servers.
      logging.info('Retrying training!')
      should_retry = True

  return total_loss
