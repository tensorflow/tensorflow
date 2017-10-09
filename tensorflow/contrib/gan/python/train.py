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
"""The TFGAN project provides a lightweight GAN training/testing framework.

This file contains the core helper functions to create and train a GAN model.
See the README or examples in `tensorflow_models` for details on how to use.

TFGAN training occurs in four steps:
1) Create a model
2) Add a loss
3) Create train ops
4) Run the train ops

The functions in this file are organized around these four steps. Each function
corresponds to one of the steps.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.framework.python.ops import variables as variables_lib
from tensorflow.contrib.gan.python import losses as tfgan_losses
from tensorflow.contrib.gan.python import namedtuples
from tensorflow.contrib.slim.python.slim import learning as slim_learning
from tensorflow.contrib.training.python.training import training
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.distributions import distribution as ds
from tensorflow.python.ops.losses import losses
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import sync_replicas_optimizer
from tensorflow.python.training import training_util


__all__ = [
    'gan_model',
    'infogan_model',
    'acgan_model',
    'gan_loss',
    'gan_train_ops',
    'gan_train',
    'get_sequential_train_hooks',
    'get_joint_train_hooks',
    'get_sequential_train_steps',
]


def gan_model(
    # Lambdas defining models.
    generator_fn,
    discriminator_fn,
    # Real data and conditioning.
    real_data,
    generator_inputs,
    # Optional scopes.
    generator_scope='Generator',
    discriminator_scope='Discriminator',
    # Options.
    check_shapes=True):
  """Returns GAN model outputs and variables.

  Args:
    generator_fn: A python lambda that takes `generator_inputs` as inputs and
      returns the outputs of the GAN generator.
    discriminator_fn: A python lambda that takes `real_data`/`generated data`
      and `generator_inputs`. Outputs a Tensor in the range [-inf, inf].
    real_data: A Tensor representing the real data.
    generator_inputs: A Tensor or list of Tensors to the generator. In the
      vanilla GAN case, this might be a single noise Tensor. In the conditional
      GAN case, this might be the generator's conditioning.
    generator_scope: Optional generator variable scope. Useful if you want to
      reuse a subgraph that has already been created.
    discriminator_scope: Optional discriminator variable scope. Useful if you
      want to reuse a subgraph that has already been created.
    check_shapes: If `True`, check that generator produces Tensors that are the
      same shape as real data. Otherwise, skip this check.

  Returns:
    A GANModel namedtuple.

  Raises:
    ValueError: If the generator outputs a Tensor that isn't the same shape as
      `real_data`.
  """
  # Create models
  with variable_scope.variable_scope(generator_scope) as gen_scope:
    generator_inputs = _convert_tensor_or_l_or_d(generator_inputs)
    generated_data = generator_fn(generator_inputs)
  with variable_scope.variable_scope(discriminator_scope) as dis_scope:
    discriminator_gen_outputs = discriminator_fn(generated_data,
                                                 generator_inputs)
  with variable_scope.variable_scope(dis_scope, reuse=True):
    real_data = ops.convert_to_tensor(real_data)
    discriminator_real_outputs = discriminator_fn(real_data, generator_inputs)

  if check_shapes:
    if not generated_data.shape.is_compatible_with(real_data.shape):
      raise ValueError(
          'Generator output shape (%s) must be the same shape as real data '
          '(%s).' % (generated_data.shape, real_data.shape))

  # Get model-specific variables.
  generator_variables = variables_lib.get_trainable_variables(gen_scope)
  discriminator_variables = variables_lib.get_trainable_variables(dis_scope)

  return namedtuples.GANModel(
      generator_inputs,
      generated_data,
      generator_variables,
      gen_scope,
      generator_fn,
      real_data,
      discriminator_real_outputs,
      discriminator_gen_outputs,
      discriminator_variables,
      dis_scope,
      discriminator_fn)


def infogan_model(
    # Lambdas defining models.
    generator_fn,
    discriminator_fn,
    # Real data and conditioning.
    real_data,
    unstructured_generator_inputs,
    structured_generator_inputs,
    # Optional scopes.
    generator_scope='Generator',
    discriminator_scope='Discriminator'):
  """Returns an InfoGAN model outputs and variables.

  See https://arxiv.org/abs/1606.03657 for more details.

  Args:
    generator_fn: A python lambda that takes a list of Tensors as inputs and
      returns the outputs of the GAN generator.
    discriminator_fn: A python lambda that takes `real_data`/`generated data`
      and `generator_inputs`. Outputs a 2-tuple of (logits, distribution_list).
      `logits` are in the range [-inf, inf], and `distribution_list` is a list
      of Tensorflow distributions representing the predicted noise distribution
      of the ith structure noise.
    real_data: A Tensor representing the real data.
    unstructured_generator_inputs: A list of Tensors to the generator.
      These tensors represent the unstructured noise or conditioning.
    structured_generator_inputs: A list of Tensors to the generator.
      These tensors must have high mutual information with the recognizer.
    generator_scope: Optional generator variable scope. Useful if you want to
      reuse a subgraph that has already been created.
    discriminator_scope: Optional discriminator variable scope. Useful if you
      want to reuse a subgraph that has already been created.

  Returns:
    An InfoGANModel namedtuple.

  Raises:
    ValueError: If the generator outputs a Tensor that isn't the same shape as
      `real_data`.
    ValueError: If the discriminator output is malformed.
  """
  # Create models
  with variable_scope.variable_scope(generator_scope) as gen_scope:
    unstructured_generator_inputs = _convert_tensor_or_l_or_d(
        unstructured_generator_inputs)
    structured_generator_inputs = _convert_tensor_or_l_or_d(
        structured_generator_inputs)
    generator_inputs = (
        unstructured_generator_inputs + structured_generator_inputs)
    generated_data = generator_fn(generator_inputs)
  with variable_scope.variable_scope(discriminator_scope) as disc_scope:
    dis_gen_outputs, predicted_distributions = discriminator_fn(
        generated_data, generator_inputs)
  _validate_distributions(predicted_distributions, structured_generator_inputs)
  with variable_scope.variable_scope(disc_scope, reuse=True):
    real_data = ops.convert_to_tensor(real_data)
    dis_real_outputs, _ = discriminator_fn(real_data, generator_inputs)

  if not generated_data.get_shape().is_compatible_with(real_data.get_shape()):
    raise ValueError(
        'Generator output shape (%s) must be the same shape as real data '
        '(%s).' % (generated_data.get_shape(), real_data.get_shape()))

  # Get model-specific variables.
  generator_variables = variables_lib.get_trainable_variables(gen_scope)
  discriminator_variables = variables_lib.get_trainable_variables(
      disc_scope)

  return namedtuples.InfoGANModel(
      generator_inputs,
      generated_data,
      generator_variables,
      gen_scope,
      generator_fn,
      real_data,
      dis_real_outputs,
      dis_gen_outputs,
      discriminator_variables,
      disc_scope,
      lambda x, y: discriminator_fn(x, y)[0],  # conform to non-InfoGAN API
      structured_generator_inputs,
      predicted_distributions)


def acgan_model(
    # Lambdas defining models.
    generator_fn,
    discriminator_fn,
    # Real data and conditioning.
    real_data,
    generator_inputs,
    one_hot_labels,
    # Optional scopes.
    generator_scope='Generator',
    discriminator_scope='Discriminator',
    # Options.
    check_shapes=True):
  """Returns an ACGANModel contains all the pieces needed for ACGAN training.

  The `acgan_model` is the same as the `gan_model` with the only difference
  being that the discriminator additionally outputs logits to classify the input
  (real or generated).
  Therefore, an explicit field holding one_hot_labels is necessary, as well as a
  discriminator_fn that outputs a 2-tuple holding the logits for real/fake and
  classification.

  See https://arxiv.org/abs/1610.09585 for more details.

  Args:
    generator_fn: A python lambda that takes `generator_inputs` as inputs and
      returns the outputs of the GAN generator.
    discriminator_fn: A python lambda that takes `real_data`/`generated data`
      and `generator_inputs`. Outputs a tuple consisting of two Tensors:
        (1) real/fake logits in the range [-inf, inf]
        (2) classification logits in the range [-inf, inf]
    real_data: A Tensor representing the real data.
    generator_inputs: A Tensor or list of Tensors to the generator. In the
      vanilla GAN case, this might be a single noise Tensor. In the conditional
      GAN case, this might be the generator's conditioning.
    one_hot_labels: A Tensor holding one-hot-labels for the batch. Needed by
      acgan_loss.
    generator_scope: Optional generator variable scope. Useful if you want to
      reuse a subgraph that has already been created.
    discriminator_scope: Optional discriminator variable scope. Useful if you
      want to reuse a subgraph that has already been created.
    check_shapes: If `True`, check that generator produces Tensors that are the
      same shape as real data. Otherwise, skip this check.

  Returns:
    A ACGANModel namedtuple.

  Raises:
    ValueError: If the generator outputs a Tensor that isn't the same shape as
      `real_data`.
    TypeError: If the discriminator does not output a tuple consisting of
    (discrimination logits, classification logits).
  """
  # Create models
  with variable_scope.variable_scope(generator_scope) as gen_scope:
    generator_inputs = _convert_tensor_or_l_or_d(generator_inputs)
    generated_data = generator_fn(generator_inputs)
  with variable_scope.variable_scope(discriminator_scope) as dis_scope:
    (discriminator_gen_outputs, discriminator_gen_classification_logits
    ) = _validate_acgan_discriminator_outputs(
        discriminator_fn(generated_data, generator_inputs))
  with variable_scope.variable_scope(dis_scope, reuse=True):
    real_data = ops.convert_to_tensor(real_data)
    (discriminator_real_outputs, discriminator_real_classification_logits
    ) = _validate_acgan_discriminator_outputs(
        discriminator_fn(real_data, generator_inputs))
  if check_shapes:
    if not generated_data.shape.is_compatible_with(real_data.shape):
      raise ValueError(
          'Generator output shape (%s) must be the same shape as real data '
          '(%s).' % (generated_data.shape, real_data.shape))

  # Get model-specific variables.
  generator_variables = variables_lib.get_trainable_variables(gen_scope)
  discriminator_variables = variables_lib.get_trainable_variables(
      dis_scope)

  return namedtuples.ACGANModel(
      generator_inputs, generated_data, generator_variables, gen_scope,
      generator_fn, real_data, discriminator_real_outputs,
      discriminator_gen_outputs, discriminator_variables, dis_scope,
      discriminator_fn, one_hot_labels,
      discriminator_real_classification_logits,
      discriminator_gen_classification_logits)


def _validate_aux_loss_weight(aux_loss_weight, name='aux_loss_weight'):
  if isinstance(aux_loss_weight, ops.Tensor):
    aux_loss_weight.shape.assert_is_compatible_with([])
    with ops.control_dependencies(
        [check_ops.assert_greater_equal(aux_loss_weight, 0.0)]):
      aux_loss_weight = array_ops.identity(aux_loss_weight)
  elif aux_loss_weight is not None and aux_loss_weight < 0:
    raise ValueError('`%s` must be greater than 0. Instead, was %s' %
                     (name, aux_loss_weight))
  return aux_loss_weight


def _use_aux_loss(aux_loss_weight):
  if aux_loss_weight is not None:
    if not isinstance(aux_loss_weight, ops.Tensor):
      return aux_loss_weight > 0
    else:
      return True
  else:
    return False


def gan_loss(
    # GANModel.
    model,
    # Loss functions.
    generator_loss_fn=tfgan_losses.wasserstein_generator_loss,
    discriminator_loss_fn=tfgan_losses.wasserstein_discriminator_loss,
    # Auxiliary losses.
    gradient_penalty_weight=None,
    gradient_penalty_epsilon=1e-10,
    mutual_information_penalty_weight=None,
    aux_cond_generator_weight=None,
    aux_cond_discriminator_weight=None,
    # Options.
    add_summaries=True):
  """Returns losses necessary to train generator and discriminator.

  Args:
    model: A GANModel tuple.
    generator_loss_fn: The loss function on the generator. Takes a GANModel
      tuple.
    discriminator_loss_fn: The loss function on the discriminator. Takes a
      GANModel tuple.
    gradient_penalty_weight: If not `None`, must be a non-negative Python number
      or Tensor indicating how much to weight the gradient penalty. See
      https://arxiv.org/pdf/1704.00028.pdf for more details.
    gradient_penalty_epsilon: If `gradient_penalty_weight` is not None, the
      small positive value used by the gradient penalty function for numerical
      stability. Note some applications will need to increase this value to
      avoid NaNs.
    mutual_information_penalty_weight: If not `None`, must be a non-negative
      Python number or Tensor indicating how much to weight the mutual
      information penalty. See https://arxiv.org/abs/1606.03657 for more
      details.
    aux_cond_generator_weight: If not None: add a classification loss as in
      https://arxiv.org/abs/1610.09585
    aux_cond_discriminator_weight: If not None: add a classification loss as in
      https://arxiv.org/abs/1610.09585
    add_summaries: Whether or not to add summaries for the losses.

  Returns:
    A GANLoss 2-tuple of (generator_loss, discriminator_loss). Includes
    regularization losses.

  Raises:
    ValueError: If any of the auxiliary loss weights is provided and negative.
    ValueError: If `mutual_information_penalty_weight` is provided, but the
      `model` isn't an `InfoGANModel`.
  """
  # Validate arguments.
  gradient_penalty_weight = _validate_aux_loss_weight(gradient_penalty_weight,
                                                      'gradient_penalty_weight')
  mutual_information_penalty_weight = _validate_aux_loss_weight(
      mutual_information_penalty_weight, 'infogan_weight')
  aux_cond_generator_weight = _validate_aux_loss_weight(
      aux_cond_generator_weight, 'aux_cond_generator_weight')
  aux_cond_discriminator_weight = _validate_aux_loss_weight(
      aux_cond_discriminator_weight, 'aux_cond_discriminator_weight')

  # Verify configuration for mutual information penalty
  if (_use_aux_loss(mutual_information_penalty_weight) and
      not isinstance(model, namedtuples.InfoGANModel)):
    raise ValueError(
        'When `mutual_information_penalty_weight` is provided, `model` must be '
        'an `InfoGANModel`. Instead, was %s.' % type(model))

  # Verify configuration for mutual auxiliary condition loss (ACGAN).
  if ((_use_aux_loss(aux_cond_generator_weight) or
       _use_aux_loss(aux_cond_discriminator_weight)) and
      not isinstance(model, namedtuples.ACGANModel)):
    raise ValueError(
        'When `aux_cond_generator_weight` or `aux_cond_discriminator_weight` '
        'is provided, `model` must be an `ACGANModel`. Instead, was %s.' %
        type(model))

  # Create standard losses.
  gen_loss = generator_loss_fn(model, add_summaries=add_summaries)
  dis_loss = discriminator_loss_fn(model, add_summaries=add_summaries)

  # Add optional extra losses.
  if _use_aux_loss(gradient_penalty_weight):
    gp_loss = tfgan_losses.wasserstein_gradient_penalty(
        model, epsilon=gradient_penalty_epsilon, add_summaries=add_summaries)
    dis_loss += gradient_penalty_weight * gp_loss
  if _use_aux_loss(mutual_information_penalty_weight):
    info_loss = tfgan_losses.mutual_information_penalty(
        model, add_summaries=add_summaries)
    dis_loss += mutual_information_penalty_weight * info_loss
    gen_loss += mutual_information_penalty_weight * info_loss
  if _use_aux_loss(aux_cond_generator_weight):
    ac_gen_loss = tfgan_losses.acgan_generator_loss(
        model, add_summaries=add_summaries)
    gen_loss += aux_cond_generator_weight * ac_gen_loss
  if _use_aux_loss(aux_cond_discriminator_weight):
    ac_disc_loss = tfgan_losses.acgan_discriminator_loss(
        model, add_summaries=add_summaries)
    dis_loss += aux_cond_discriminator_weight * ac_disc_loss
  # Gathers auxilliary losses.
  if model.generator_scope:
    gen_reg_loss = losses.get_regularization_loss(model.generator_scope.name)
  else:
    gen_reg_loss = 0
  if model.discriminator_scope:
    dis_reg_loss = losses.get_regularization_loss(
        model.discriminator_scope.name)
  else:
    dis_reg_loss = 0

  return namedtuples.GANLoss(gen_loss + gen_reg_loss, dis_loss + dis_reg_loss)


def _get_update_ops(kwargs, gen_scope, dis_scope, check_for_unused_ops=True):
  """Gets generator and discriminator update ops.

  Args:
    kwargs: A dictionary of kwargs to be passed to `create_train_op`.
      `update_ops` is removed, if present.
    gen_scope: A scope for the generator.
    dis_scope: A scope for the discriminator.
    check_for_unused_ops: A Python bool. If `True`, throw Exception if there are
      unused update ops.

  Returns:
    A 2-tuple of (generator update ops, discriminator train ops).

  Raises:
    ValueError: If there are update ops outside of the generator or
      discriminator scopes.
  """
  if 'update_ops' in kwargs:
    update_ops = set(kwargs['update_ops'])
    del kwargs['update_ops']
  else:
    update_ops = set(ops.get_collection(ops.GraphKeys.UPDATE_OPS))

  all_gen_ops = set(ops.get_collection(ops.GraphKeys.UPDATE_OPS, gen_scope))
  all_dis_ops = set(ops.get_collection(ops.GraphKeys.UPDATE_OPS, dis_scope))

  if check_for_unused_ops:
    unused_ops = update_ops - all_gen_ops - all_dis_ops
    if unused_ops:
      raise ValueError('There are unused update ops: %s' % unused_ops)

  gen_update_ops = list(all_gen_ops & update_ops)
  dis_update_ops = list(all_dis_ops & update_ops)

  return gen_update_ops, dis_update_ops


def gan_train_ops(
    model,
    loss,
    generator_optimizer,
    discriminator_optimizer,
    check_for_unused_update_ops=True,
    # Optional args to pass directly to the `create_train_op`.
    **kwargs):
  """Returns GAN train ops.

  The highest-level call in TFGAN. It is composed of functions that can also
  be called, should a user require more control over some part of the GAN
  training process.

  Args:
    model: A GANModel.
    loss: A GANLoss.
    generator_optimizer: The optimizer for generator updates.
    discriminator_optimizer: The optimizer for the discriminator updates.
    check_for_unused_update_ops: If `True`, throws an exception if there are
      update ops outside of the generator or discriminator scopes.
    **kwargs: Keyword args to pass directly to
      `training.create_train_op` for both the generator and
      discriminator train op.

  Returns:
    A GANTrainOps tuple of (generator_train_op, discriminator_train_op) that can
    be used to train a generator/discriminator pair.
  """
  # Create global step increment op.
  global_step = training_util.get_or_create_global_step()
  global_step_inc = global_step.assign_add(1)

  # Get generator and discriminator update ops. We split them so that update
  # ops aren't accidentally run multiple times. For now, throw an error if
  # there are update ops that aren't associated with either the generator or
  # the discriminator. Might modify the `kwargs` dictionary.
  gen_update_ops, dis_update_ops = _get_update_ops(
      kwargs, model.generator_scope.name, model.discriminator_scope.name,
      check_for_unused_update_ops)

  generator_global_step = None
  if isinstance(generator_optimizer,
                sync_replicas_optimizer.SyncReplicasOptimizer):
    # TODO(joelshor): Figure out a way to get this work without including the
    # dummy global step in the checkpoint.
    # WARNING: Making this variable a local variable causes sync replicas to
    # hang forever.
    generator_global_step = variable_scope.get_variable(
        'dummy_global_step_generator',
        shape=[],
        dtype=global_step.dtype.base_dtype,
        initializer=init_ops.zeros_initializer(),
        trainable=False,
        collections=[ops.GraphKeys.GLOBAL_VARIABLES])
    gen_update_ops += [generator_global_step.assign(global_step)]
  with ops.name_scope('generator_train'):
    gen_train_op = training.create_train_op(
        total_loss=loss.generator_loss,
        optimizer=generator_optimizer,
        variables_to_train=model.generator_variables,
        global_step=generator_global_step,
        update_ops=gen_update_ops,
        **kwargs)

  discriminator_global_step = None
  if isinstance(discriminator_optimizer,
                sync_replicas_optimizer.SyncReplicasOptimizer):
    # See comment above `generator_global_step`.
    discriminator_global_step = variable_scope.get_variable(
        'dummy_global_step_discriminator',
        shape=[],
        dtype=global_step.dtype.base_dtype,
        initializer=init_ops.zeros_initializer(),
        trainable=False,
        collections=[ops.GraphKeys.GLOBAL_VARIABLES])
    dis_update_ops += [discriminator_global_step.assign(global_step)]
  with ops.name_scope('discriminator_train'):
    disc_train_op = training.create_train_op(
        total_loss=loss.discriminator_loss,
        optimizer=discriminator_optimizer,
        variables_to_train=model.discriminator_variables,
        global_step=discriminator_global_step,
        update_ops=dis_update_ops,
        **kwargs)

  return namedtuples.GANTrainOps(gen_train_op, disc_train_op, global_step_inc)


# TODO(joelshor): Implement a dynamic GAN train loop, as in `Real-Time Adaptive
# Image Compression` (https://arxiv.org/abs/1705.05823)
class RunTrainOpsHook(session_run_hook.SessionRunHook):
  """A hook to run train ops a fixed number of times."""

  def __init__(self, train_ops, train_steps):
    """Run train ops a certain number of times.

    Args:
      train_ops: A train op or iterable of train ops to run.
      train_steps: The number of times to run the op(s).
    """
    if not isinstance(train_ops, (list, tuple)):
      train_ops = [train_ops]
    self._train_ops = train_ops
    self._train_steps = train_steps

  def before_run(self, run_context):
    for _ in range(self._train_steps):
      run_context.session.run(self._train_ops)


def get_sequential_train_hooks(train_steps=namedtuples.GANTrainSteps(1, 1)):
  """Returns a hooks function for sequential GAN training.

  Args:
    train_steps: A `GANTrainSteps` tuple that determines how many generator
      and discriminator training steps to take.

  Returns:
    A function that takes a GANTrainOps tuple and returns a list of hooks.
  """
  def get_hooks(train_ops):
    generator_hook = RunTrainOpsHook(train_ops.generator_train_op,
                                     train_steps.generator_train_steps)
    discriminator_hook = RunTrainOpsHook(train_ops.discriminator_train_op,
                                         train_steps.discriminator_train_steps)
    return [generator_hook, discriminator_hook]
  return get_hooks


def get_joint_train_hooks(train_steps=namedtuples.GANTrainSteps(1, 1)):
  """Returns a hooks function for sequential GAN training.

  When using these train hooks, IT IS RECOMMENDED TO USE `use_locking=True` ON
  ALL OPTIMIZERS TO AVOID RACE CONDITIONS.

  The order of steps taken is:
  1) Combined generator and discriminator steps
  2) Generator only steps, if any remain
  3) Discriminator only steps, if any remain

  **NOTE**: Unlike `get_sequential_train_hooks`, this method performs updates
  for the generator and discriminator simultaneously whenever possible. This
  reduces the number of `tf.Session` calls, and can also change the training
  semantics.

  To illustrate the difference look at the following example:

  `train_steps=namedtuples.GANTrainSteps(3, 5)` will cause
  `get_sequential_train_hooks` to make 8 session calls:
    1) 3 generator steps
    2) 5 discriminator steps

  In contrast, `get_joint_train_steps` will make 5 session calls:
  1) 3 generator + discriminator steps
  2) 2 discriminator steps

  Args:
    train_steps: A `GANTrainSteps` tuple that determines how many generator
      and discriminator training steps to take.

  Returns:
    A function that takes a GANTrainOps tuple and returns a list of hooks.
  """
  g_steps = train_steps.generator_train_steps
  d_steps = train_steps.discriminator_train_steps
  # Get the number of each type of step that should be run.
  num_d_and_g_steps = min(g_steps, d_steps)
  num_g_steps = g_steps - num_d_and_g_steps
  num_d_steps = d_steps - num_d_and_g_steps

  def get_hooks(train_ops):
    g_op = train_ops.generator_train_op
    d_op = train_ops.discriminator_train_op

    joint_hook = RunTrainOpsHook([g_op, d_op], num_d_and_g_steps)
    g_hook = RunTrainOpsHook(g_op, num_g_steps)
    d_hook = RunTrainOpsHook(d_op, num_d_steps)

    return [joint_hook, g_hook, d_hook]
  return get_hooks


# TODO(joelshor): This function currently returns the global step. Find a
# good way for it to return the generator, discriminator, and final losses.
def gan_train(
    train_ops,
    logdir,
    get_hooks_fn=get_sequential_train_hooks(),
    master='',
    is_chief=True,
    scaffold=None,
    hooks=None,
    chief_only_hooks=None,
    save_checkpoint_secs=600,
    save_summaries_steps=100,
    config=None):
  """A wrapper around `contrib.training.train` that uses GAN hooks.

  Args:
    train_ops: A GANTrainOps named tuple.
    logdir: The directory where the graph and checkpoints are saved.
    get_hooks_fn: A function that takes a GANTrainOps tuple and returns a list
      of hooks.
    master: The URL of the master.
    is_chief: Specifies whether or not the training is being run by the primary
      replica during replica training.
    scaffold: An tf.train.Scaffold instance.
    hooks: List of `tf.train.SessionRunHook` callbacks which are run inside the
      training loop.
    chief_only_hooks: List of `tf.train.SessionRunHook` instances which are run
      inside the training loop for the chief trainer only.
    save_checkpoint_secs: The frequency, in seconds, that a checkpoint is saved
      using a default checkpoint saver. If `save_checkpoint_secs` is set to
      `None`, then the default checkpoint saver isn't used.
    save_summaries_steps: The frequency, in number of global steps, that the
      summaries are written to disk using a default summary saver. If
      `save_summaries_steps` is set to `None`, then the default summary saver
      isn't used.
    config: An instance of `tf.ConfigProto`.

  Returns:
    Output of the call to `training.train`.
  """
  new_hooks = get_hooks_fn(train_ops)
  if hooks is not None:
    hooks = list(hooks) + list(new_hooks)
  else:
    hooks = new_hooks
  return training.train(
      train_ops.global_step_inc_op,
      logdir,
      master=master,
      is_chief=is_chief,
      scaffold=scaffold,
      hooks=hooks,
      chief_only_hooks=chief_only_hooks,
      save_checkpoint_secs=save_checkpoint_secs,
      save_summaries_steps=save_summaries_steps,
      config=config)


def get_sequential_train_steps(
    train_steps=namedtuples.GANTrainSteps(1, 1)):
  """Returns a thin wrapper around slim.learning.train_step, for GANs.

  This function is to provide support for the Supervisor. For new code, please
  use `MonitoredSession` and `get_sequential_train_hooks`.

  Args:
    train_steps: A `GANTrainSteps` tuple that determines how many generator
      and discriminator training steps to take.

  Returns:
    A function that can be used for `train_step_fn` for GANs.
  """

  def sequential_train_steps(sess, train_ops, global_step, train_step_kwargs):
    """A thin wrapper around slim.learning.train_step, for GANs.

    Args:
      sess: A Tensorflow session.
      train_ops: A GANTrainOps tuple of train ops to run.
      global_step: The global step.
      train_step_kwargs: Dictionary controlling `train_step` behavior.

    Returns:
      A scalar final loss and a bool whether or not the train loop should stop.
    """
    # Only run `should_stop` at the end, if required. Make a local copy of
    # `train_step_kwargs`, if necessary, so as not to modify the caller's
    # dictionary.
    should_stop_op, train_kwargs = None, train_step_kwargs
    if 'should_stop' in train_step_kwargs:
      should_stop_op = train_step_kwargs['should_stop']
      train_kwargs = train_step_kwargs.copy()
      del train_kwargs['should_stop']

    # Run generator training steps.
    gen_loss = 0
    for _ in range(train_steps.generator_train_steps):
      cur_gen_loss, _ = slim_learning.train_step(
          sess, train_ops.generator_train_op, global_step, train_kwargs)
      gen_loss += cur_gen_loss

    # Run discriminator training steps.
    dis_loss = 0
    for _ in range(train_steps.discriminator_train_steps):
      cur_dis_loss, _ = slim_learning.train_step(
          sess, train_ops.discriminator_train_op, global_step, train_kwargs)
      dis_loss += cur_dis_loss

    sess.run(train_ops.global_step_inc_op)

    # Run the `should_stop` op after the global step has been incremented, so
    # that the `should_stop` aligns with the proper `global_step` count.
    if should_stop_op is not None:
      should_stop = sess.run(should_stop_op)
    else:
      should_stop = False

    return gen_loss + dis_loss, should_stop

  return sequential_train_steps


# Helpers


def _convert_tensor_or_l_or_d(tensor_or_l_or_d):
  """Convert input, list of inputs, or dictionary of inputs to Tensors."""
  if isinstance(tensor_or_l_or_d, (list, tuple)):
    return [ops.convert_to_tensor(x) for x in tensor_or_l_or_d]
  elif isinstance(tensor_or_l_or_d, dict):
    return {k: ops.convert_to_tensor(v) for k, v in tensor_or_l_or_d.items()}
  else:
    return ops.convert_to_tensor(tensor_or_l_or_d)


def _validate_distributions(distributions_l, noise_l):
  if not isinstance(distributions_l, (tuple, list)):
    raise ValueError('`predicted_distributions` must be a list. Instead, found '
                     '%s.' % type(distributions_l))
  for dist in distributions_l:
    if not isinstance(dist, ds.Distribution):
      raise ValueError('Every element in `predicted_distributions` must be a '
                       '`tf.Distribution`. Instead, found %s.' % type(dist))
  if len(distributions_l) != len(noise_l):
    raise ValueError('Length of `predicted_distributions` %i must be the same '
                     'as the length of structured noise %i.' %
                     (len(distributions_l), len(noise_l)))


def _validate_acgan_discriminator_outputs(discriminator_output):
  try:
    a, b = discriminator_output
  except (TypeError, ValueError):
    raise TypeError(
        'A discriminator function for ACGAN must output a tuple '
        'consisting of (discrimination logits, classification logits).')
  return a, b
