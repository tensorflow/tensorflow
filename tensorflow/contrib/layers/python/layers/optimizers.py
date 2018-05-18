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
"""Optimizer ops for use in layers and tf.learn."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.contrib import framework as contrib_framework
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as vars_
from tensorflow.python.summary import summary
from tensorflow.python.training import moving_averages
from tensorflow.python.training import optimizer as optimizer_
from tensorflow.python.training import training as train

OPTIMIZER_CLS_NAMES = {
    "Adagrad": train.AdagradOptimizer,
    "Adam": train.AdamOptimizer,
    "Ftrl": train.FtrlOptimizer,
    "Momentum": lambda learning_rate: train.MomentumOptimizer(learning_rate, momentum=0.9),  # pylint: disable=line-too-long
    "RMSProp": train.RMSPropOptimizer,
    "SGD": train.GradientDescentOptimizer,
}

OPTIMIZER_SUMMARIES = [
    "learning_rate",
    "loss",
    "gradients",
    "gradient_norm",
    "global_gradient_norm",
]


def optimize_loss(loss,
                  global_step,
                  learning_rate,
                  optimizer,
                  gradient_noise_scale=None,
                  gradient_multipliers=None,
                  clip_gradients=None,
                  learning_rate_decay_fn=None,
                  update_ops=None,
                  variables=None,
                  name=None,
                  summaries=None,
                  colocate_gradients_with_ops=False,
                  increment_global_step=True):
  """Given loss and parameters for optimizer, returns a training op.

  Various ways of passing optimizers include:

  - by string specifying the name of the optimizer. See OPTIMIZER_CLS_NAMES
      for full list. E.g. `optimize_loss(..., optimizer='Adam')`.
  - by function taking learning rate `Tensor` as argument and returning an
      `Optimizer` instance. E.g. `optimize_loss(...,
      optimizer=lambda lr: tf.train.MomentumOptimizer(lr, momentum=0.5))`.
    Alternatively, if `learning_rate` is `None`, the function takes no
    arguments. E.g. `optimize_loss(..., learning_rate=None,
      optimizer=lambda: tf.train.MomentumOptimizer(0.5, momentum=0.5))`.
  - by a subclass of `Optimizer` having a single-argument constructor
      (the argument is the learning rate), such as AdamOptimizer or
      AdagradOptimizer. E.g. `optimize_loss(...,
      optimizer=tf.train.AdagradOptimizer)`.
  - by an instance of a subclass of `Optimizer`.
      E.g., `optimize_loss(..., optimizer=tf.train.AdagradOptimizer(0.5))`.

  Args:
    loss: Scalar `Tensor`.
    global_step: Scalar int `Tensor`, step counter to update on each step
                 unless `increment_global_step` is `False`. If not supplied,
                 it will be fetched from the default graph (see
                 `tf.train.get_global_step` for details). If it has
                 not been created, no step will be incremented with each weight
                 update. `learning_rate_decay_fn` requires `global_step`.
    learning_rate: float or `Tensor`, magnitude of update per each training
                   step. Can be `None`.
    optimizer: string, class or optimizer instance, used as trainer.
               string should be name of optimizer, like 'SGD',
                 'Adam', 'Adagrad'. Full list in OPTIMIZER_CLS_NAMES constant.
               class should be sub-class of `tf.Optimizer` that implements
                 `compute_gradients` and `apply_gradients` functions.
               optimizer instance should be instantiation of `tf.Optimizer`
                 sub-class and have `compute_gradients` and `apply_gradients`
                 functions.
    gradient_noise_scale: float or None, adds 0-mean normal noise scaled by this
                          value.
    gradient_multipliers: dict of variables or variable names to floats.
                          If present, gradients for specified
                          variables will be multiplied by given constant.
    clip_gradients: float, callable or `None`. If float, is provided, a global
      clipping is applied to prevent the norm of the gradient to exceed this
      value. Alternatively, a callable can be provided e.g.: adaptive_clipping.
      This callable takes a `list` of `(gradients, variables)` `tuple`s and
      returns the same thing with the gradients modified.
    learning_rate_decay_fn: function, takes `learning_rate` and `global_step`
                            `Tensor`s, returns `Tensor`.
                            Can be used to implement any learning rate decay
                            functions.
                            For example: `tf.train.exponential_decay`.
                            Ignored if `learning_rate` is not supplied.
    update_ops: list of update `Operation`s to execute at each step. If `None`,
                uses elements of UPDATE_OPS collection. The order of execution
                between `update_ops` and `loss` is non-deterministic.
    variables: list of variables to optimize or
               `None` to use all trainable variables.
    name: The name for this operation is used to scope operations and summaries.
    summaries: List of internal quantities to visualize on tensorboard. If not
               set, the loss, the learning rate, and the global norm of the
               gradients will be reported. The complete list of possible values
               is in OPTIMIZER_SUMMARIES.
    colocate_gradients_with_ops: If True, try colocating gradients with the
                                 corresponding op.
    increment_global_step: Whether to increment `global_step`. If your model
      calls `optimize_loss` multiple times per training step (e.g. to optimize
      different parts of the model), use this arg to avoid incrementing
      `global_step` more times than necessary.

  Returns:
    Training op.

  Raises:
    ValueError: if:
        * `loss` is an invalid type or shape.
        * `global_step` is an invalid type or shape.
        * `learning_rate` is an invalid type or value.
        * `optimizer` has the wrong type.
        * `clip_gradients` is neither float nor callable.
        * `learning_rate` and `learning_rate_decay_fn` are supplied, but no
          `global_step` is available.
        * `gradients` is empty.
  """
  loss = ops.convert_to_tensor(loss)
  contrib_framework.assert_scalar(loss)
  if global_step is None:
    global_step = train.get_global_step()
  else:
    train.assert_global_step(global_step)
  with vs.variable_scope(name, "OptimizeLoss", [loss, global_step]):
    # Update ops take UPDATE_OPS collection if not provided.
    if update_ops is None:
      update_ops = set(ops.get_collection(ops.GraphKeys.UPDATE_OPS))
    # Make sure update ops are ran before computing loss.
    if update_ops:
      loss = control_flow_ops.with_dependencies(list(update_ops), loss)

    # Learning rate variable, with possible decay.
    lr = None
    if learning_rate is not None:
      if (isinstance(learning_rate, ops.Tensor) and
          learning_rate.get_shape().ndims == 0):
        lr = learning_rate
      elif isinstance(learning_rate, float):
        if learning_rate < 0.0:
          raise ValueError("Invalid learning_rate %s.", learning_rate)
        lr = vs.get_variable(
            "learning_rate", [],
            trainable=False,
            initializer=init_ops.constant_initializer(learning_rate))
      else:
        raise ValueError("Learning rate should be 0d Tensor or float. "
                         "Got %s of type %s" % (str(learning_rate),
                                                str(type(learning_rate))))
    if summaries is None:
      summaries = ["loss", "learning_rate", "global_gradient_norm"]
    else:
      for summ in summaries:
        if summ not in OPTIMIZER_SUMMARIES:
          raise ValueError("Summaries should be one of [%s], you provided %s." %
                           (", ".join(OPTIMIZER_SUMMARIES), summ))
    if learning_rate is not None and learning_rate_decay_fn is not None:
      if global_step is None:
        raise ValueError("global_step is required for learning_rate_decay_fn.")
      lr = learning_rate_decay_fn(lr, global_step)
      if "learning_rate" in summaries:
        summary.scalar("learning_rate", lr)

    # Create optimizer, given specified parameters.
    if isinstance(optimizer, six.string_types):
      if lr is None:
        raise ValueError("Learning rate is None, but should be specified if "
                         "optimizer is string (%s)." % optimizer)
      if optimizer not in OPTIMIZER_CLS_NAMES:
        raise ValueError(
            "Optimizer name should be one of [%s], you provided %s." %
            (", ".join(OPTIMIZER_CLS_NAMES), optimizer))
      opt = OPTIMIZER_CLS_NAMES[optimizer](learning_rate=lr)
    elif (isinstance(optimizer, type) and
          issubclass(optimizer, optimizer_.Optimizer)):
      if lr is None:
        raise ValueError("Learning rate is None, but should be specified if "
                         "optimizer is class (%s)." % optimizer)
      opt = optimizer(learning_rate=lr)
    elif isinstance(optimizer, optimizer_.Optimizer):
      opt = optimizer
    elif callable(optimizer):
      if learning_rate is not None:
        opt = optimizer(lr)
      else:
        opt = optimizer()
      if not isinstance(opt, optimizer_.Optimizer):
        raise ValueError("Unrecognized optimizer: function should return "
                         "subclass of Optimizer. Got %s." % str(opt))
    else:
      raise ValueError("Unrecognized optimizer: should be string, "
                       "subclass of Optimizer, instance of "
                       "subclass of Optimizer or function with one argument. "
                       "Got %s." % str(optimizer))

    # All trainable variables, if specific variables are not specified.
    if variables is None:
      variables = vars_.trainable_variables()

    # Compute gradients.
    gradients = opt.compute_gradients(
        loss,
        variables,
        colocate_gradients_with_ops=colocate_gradients_with_ops)

    # Optionally add gradient noise.
    if gradient_noise_scale is not None:
      gradients = _add_scaled_noise_to_gradients(gradients,
                                                 gradient_noise_scale)

    # Multiply some gradients.
    if gradient_multipliers is not None:
      gradients = _multiply_gradients(gradients, gradient_multipliers)
      if not gradients:
        raise ValueError(
            "Empty list of (gradient, var) pairs encountered. This is most "
            "likely to be caused by an improper value of gradient_multipliers.")

    if "global_gradient_norm" in summaries or "gradient_norm" in summaries:
      summary.scalar("global_norm/gradient_norm",
                     clip_ops.global_norm(list(zip(*gradients))[0]))

    # Optionally clip gradients by global norm.
    if isinstance(clip_gradients, float):
      gradients = _clip_gradients_by_norm(gradients, clip_gradients)
    elif callable(clip_gradients):
      gradients = clip_gradients(gradients)
    elif clip_gradients is not None:
      raise ValueError(
          "Unknown type %s for clip_gradients" % type(clip_gradients))

    # Add scalar summary for loss.
    if "loss" in summaries:
      summary.scalar("loss", loss)

    # Add histograms for variables, gradients and gradient norms.
    for gradient, variable in gradients:
      if isinstance(gradient, ops.IndexedSlices):
        grad_values = gradient.values
      else:
        grad_values = gradient

      if grad_values is not None:
        var_name = variable.name.replace(":", "_")
        if "gradients" in summaries:
          summary.histogram("gradients/%s" % var_name, grad_values)
        if "gradient_norm" in summaries:
          summary.scalar("gradient_norm/%s" % var_name,
                         clip_ops.global_norm([grad_values]))

    if clip_gradients is not None and ("global_gradient_norm" in summaries or
                                       "gradient_norm" in summaries):
      summary.scalar("global_norm/clipped_gradient_norm",
                     clip_ops.global_norm(list(zip(*gradients))[0]))

    # Create gradient updates.
    grad_updates = opt.apply_gradients(
        gradients,
        global_step=global_step if increment_global_step else None,
        name="train")

    # Ensure the train_tensor computes grad_updates.
    train_tensor = control_flow_ops.with_dependencies([grad_updates], loss)

    return train_tensor


def _clip_gradients_by_norm(grads_and_vars, clip_gradients):
  """Clips gradients by global norm."""
  gradients, variables = zip(*grads_and_vars)
  clipped_gradients, _ = clip_ops.clip_by_global_norm(gradients, clip_gradients)
  return list(zip(clipped_gradients, variables))


def _adaptive_max_norm(norm, std_factor, decay, global_step, epsilon, name):
  """Find max_norm given norm and previous average."""
  with vs.variable_scope(name, "AdaptiveMaxNorm", [norm]):
    log_norm = math_ops.log(norm + epsilon)

    def moving_average(name, value, decay):
      moving_average_variable = vs.get_variable(
          name,
          shape=value.get_shape(),
          dtype=value.dtype,
          initializer=init_ops.zeros_initializer(),
          trainable=False)
      return moving_averages.assign_moving_average(
          moving_average_variable, value, decay, zero_debias=False)

    # quicker adaptation at the beginning
    if global_step is not None:
      n = math_ops.to_float(global_step)
      decay = math_ops.minimum(decay, n / (n + 1.))

    # update averages
    mean = moving_average("mean", log_norm, decay)
    sq_mean = moving_average("sq_mean", math_ops.square(log_norm), decay)

    variance = sq_mean - math_ops.square(mean)
    std = math_ops.sqrt(math_ops.maximum(epsilon, variance))
    max_norms = math_ops.exp(mean + std_factor * std)
    return max_norms, mean


def adaptive_clipping_fn(std_factor=2.,
                         decay=0.95,
                         static_max_norm=None,
                         global_step=None,
                         report_summary=False,
                         epsilon=1e-8,
                         name=None):
  """Adapt the clipping value using statistics on the norms.

  Implement adaptive gradient as presented in section 3.2.1 of
  https://arxiv.org/abs/1412.1602.

  Keeps a moving average of the mean and std of the log(norm) of the gradient.
  If the norm exceeds `exp(mean + std_factor*std)` then all gradients will be
  rescaled such that the global norm becomes `exp(mean)`.

  Args:
    std_factor: Python scaler (or tensor).
      `max_norm = exp(mean + std_factor*std)`
    decay: The smoothing factor of the moving averages.
    static_max_norm: If provided, will threshold the norm to this value as an
      extra safety.
    global_step: Optional global_step. If provided, `decay = decay*n/(n+1)`.
      This provides a quicker adaptation of the mean for the first steps.
    report_summary: If `True`, will add histogram summaries of the `max_norm`.
    epsilon: Small value chosen to avoid zero variance.
    name: The name for this operation is used to scope operations and summaries.

  Returns:
    A function for applying gradient clipping.
  """

  def gradient_clipping(grads_and_vars):
    """Internal function for adaptive clipping."""
    grads, variables = zip(*grads_and_vars)

    norm = clip_ops.global_norm(grads)

    max_norm, log_mean = _adaptive_max_norm(norm, std_factor, decay,
                                            global_step, epsilon, name)

    # reports the max gradient norm for debugging
    if report_summary:
      summary.scalar("global_norm/adaptive_max_gradient_norm", max_norm)

    # factor will be 1. if norm is smaller than max_norm
    factor = array_ops.where(norm < max_norm,
                             array_ops.ones_like(norm),
                             math_ops.exp(log_mean) / norm)

    if static_max_norm is not None:
      factor = math_ops.minimum(static_max_norm / norm, factor)

    # apply factor
    clipped_grads = []
    for grad in grads:
      if grad is None:
        clipped_grads.append(None)
      elif isinstance(grad, ops.IndexedSlices):
        clipped_grads.append(
            ops.IndexedSlices(grad.values * factor, grad.indices,
                              grad.dense_shape))
      else:
        clipped_grads.append(grad * factor)

    return list(zip(clipped_grads, variables))

  return gradient_clipping


def _add_scaled_noise_to_gradients(grads_and_vars, gradient_noise_scale):
  """Adds scaled noise from a 0-mean normal distribution to gradients."""
  gradients, variables = zip(*grads_and_vars)
  noisy_gradients = []
  for gradient in gradients:
    if gradient is None:
      noisy_gradients.append(None)
      continue
    if isinstance(gradient, ops.IndexedSlices):
      gradient_shape = gradient.dense_shape
    else:
      gradient_shape = gradient.get_shape()
    noise = random_ops.truncated_normal(gradient_shape) * gradient_noise_scale
    noisy_gradients.append(gradient + noise)
  return list(zip(noisy_gradients, variables))


def _multiply_gradients(grads_and_vars, gradient_multipliers):
  """Multiply specified gradients."""
  multiplied_grads_and_vars = []
  for grad, var in grads_and_vars:
    if (grad is not None and
        (var in gradient_multipliers or var.name in gradient_multipliers)):
      key = var if var in gradient_multipliers else var.name
      multiplier = constant_op.constant(
          gradient_multipliers[key], dtype=dtypes.float32)
      if isinstance(grad, ops.IndexedSlices):
        grad_values = grad.values * multiplier
        grad = ops.IndexedSlices(grad_values, grad.indices, grad.dense_shape)
      else:
        grad *= multiplier
    multiplied_grads_and_vars.append((grad, var))
  return multiplied_grads_and_vars
