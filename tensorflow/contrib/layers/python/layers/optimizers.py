# Copyright 2015 Google Inc. All Rights Reserved.
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

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as vars_
from tensorflow.python.training import optimizer as optimizer_
from tensorflow.python.training import training as train

OPTIMIZER_CLS_NAMES = {
    "Adagrad": train.AdagradOptimizer,
    "Adam": train.AdamOptimizer,
    "Ftrl": train.FtrlOptimizer,
    "Momentum": train.MomentumOptimizer,
    "RMSProp": train.RMSPropOptimizer,
    "SGD": train.GradientDescentOptimizer,
}


def optimize_loss(loss,
                  global_step,
                  learning_rate,
                  optimizer,
                  clip_gradients=None,
                  moving_average_decay=0.9,
                  learning_rate_decay_fn=None,
                  variables=None):
  """Given loss and parameters for optimizer, returns a training op.

  Args:
    loss: Tensor, 0 dimensional.
    global_step: Tensor, step counter for each update.
    learning_rate: float or Tensor, magnitude of update per each training step.
    optimizer: string, class or optimizer instance, used as trainer.
               string should be name of optimizer, like 'SGD',
                 'Adam', 'Adagrad'. Full list in OPTIMIZER_CLS_NAMES constant.
               class should be sub-class of tf.Optimizer that implements
                 `compute_gradients` and `apply_gradients` functions.
               optimizer instance should be instantion of tf.Optimizer sub-class
                 and have `compute_gradients` and `apply_gradients` functions.
    clip_gradients: float or None, clips gradients by this value.
    moving_average_decay: float or None, takes into account previous loss
                          to make learning smoother due to outliers.
    learning_rate_decay_fn: function, takes learning_rate and global_step
                            Tensors, returns Tensor. Can be used to implement
                            any learning rate decay funcitons.
                            For example: tf.train.exponential_decay.
    variables: list of variables to optimizer or none.

  Returns:
    Training op.

  Raises:
    ValueError: if optimizer is wrong type.
  """
  # Moving average of the loss with decay.
  if moving_average_decay is not None:
    # Generate moving averages of the loss.
    loss_averages = train.ExponentialMovingAverage(moving_average_decay,
                                                   name="avg")
    loss_averages_op = loss_averages.apply([loss])
    logging_ops.scalar_summary("loss/mean", loss_averages.average(loss))
    loss = control_flow_ops.with_dependencies([loss_averages_op], loss)

  # Learning rate variable, with possible decay.
  if isinstance(learning_rate, ops.Tensor) and len(learning_rate.get_shape()) == 0:
    lr = learning_rate
  elif isinstance(learning_rate, float):
    lr = vs.get_variable("learning_rate",
                         [],
                         trainable=False,
                         initializer=init_ops.constant_initializer(learning_rate))
  else:
    raise ValueError("Learning rate should be 0d Tensor or float. Got %s" %
        str(learning_rate))
  if learning_rate_decay_fn is not None:
    lr = learning_rate_decay_fn(lr, global_step)

  # Create optimizer, given specified parameters.
  if isinstance(optimizer, six.string_types):
    if optimizer not in OPTIMIZER_CLS_NAMES:
      raise ValueError("Optimizer name should be one of [%s], you provided %s."
                       % (", ".join(OPTIMIZER_CLS_NAMES), optimizer))
    opt = OPTIMIZER_CLS_NAMES[optimizer](learning_rate=lr)
  elif isinstance(optimizer, type) and issubclass(optimizer,
                                                  optimizer_.Optimizer):
    opt = optimizer(learning_rate=lr)
  elif isinstance(optimizer, optimizer_.Optimizer):
    opt = optimizer
  else:
    raise ValueError("Unrecognized optimizer: should be string, "
                     "subclass of Optimizer or instance of "
                     "subclass of Optimizer. Got %s." % str(optimizer))

  # All trainable variables, if specific variables are not specified.
  if variables is None:
    variables = vars_.trainable_variables()

  # Compute gradients and clip them if provided.
  gradients = opt.compute_gradients(loss, variables)
  if clip_gradients is not None:
    gradients, variables = zip(*gradients)
    clipped_gradients, _ = clip_ops.clip_by_global_norm(gradients,
                                                        clip_gradients)
    gradients = list(zip(clipped_gradients, variables))

  # Add scalar summary for loss.
  logging_ops.scalar_summary("loss", loss)

  # Add histograms for variables, gradients and gradient norms.
  for gradient, variable in gradients:
    if isinstance(gradient, ops.IndexedSlices):
      grad_values = gradient.values
    else:
      grad_values = gradient

    if grad_values is not None:
      logging_ops.histogram_summary(variable.name, variable)
      logging_ops.histogram_summary(variable.name + "/gradients", grad_values)
      logging_ops.histogram_summary(variable.name + "/gradient_norm",
                                    clip_ops.global_norm([grad_values]))

  # Create gradient updates.
  grad_updates = opt.apply_gradients(gradients,
                                     global_step=global_step,
                                     name="train")
  # Make sure total_loss is valid.
  final_loss = array_ops.check_numerics(loss, "Loss is inf or nan")

  # Ensure the train_tensor computes grad_updates.
  train_tensor = control_flow_ops.with_dependencies([grad_updates], final_loss)

  return train_tensor

