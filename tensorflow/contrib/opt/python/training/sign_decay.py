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
"""Implementation of the sign decay functions used in PowerSign and AddSign.

See [Bello et al., ICML 2017] Neural Optimizer Search with Reinforcement
Learning for details.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


def get_linear_decay_fn(decay_steps):
  """Returns a function that computes a linear decay.

  This decay computes linear annealing:
    max(0, (decay_steps - global_step) / decay_steps)

  Example usage:
  ```
  decay_steps = 1000
  linear_decay_fn = get_linear_decay_fn(decay_steps)
  decayed = linear_decay_fn(global_step)
  x *= decayed
  ```
  Args:
    decay_steps: number of steps to decay over.
  Returns:
    linear_decay_fn: a function that computes the linear decay.
  """
  # pylint:disable=missing-docstring
  def linear_decay_fn(global_step):
    if global_step is None:
      raise ValueError("global_step is required for linear_decay.")
    global_step = math_ops.minimum(global_step, decay_steps)
    remaining_steps = math_ops.cast(
        decay_steps, dtypes.int32) - math_ops.cast(global_step, dtypes.int32)
    decayed = (math_ops.cast(remaining_steps, dtypes.float32) /
               math_ops.cast(decay_steps, dtypes.float32))
    return math_ops.maximum(0.0, decayed)
  # pylint:enable=missing-docstring
  return linear_decay_fn


def get_cosine_decay_fn(decay_steps, num_periods=0.5, zero_after=None):
  """Returns a function that computes a cosine decay.

  This decay computes cosine annealing:
    0.5 * (1.0 + cos(2.0 * pi * num_periods * global_step / decay_steps))

  This decay can be used to decay the sign quantity in the AddSign and PowerSign
  optimizers discovered in
  [Bello et al., ICML 2017] Neural Optimizer Search with RL.

  Example usage:
  ```
  decay_steps = 1000
  num_periods = 2
  cosine_decay_fn = get_cosine_decay_fn(decay_steps, num_periods=num_periods)
  decayed = cosine_decay_fn(global_step)
  x *= decayed
  ```
  Args:
    decay_steps: number of steps to decay over.
    num_periods: number of periods for cosine signal. 0.5 by default,
      which maps the last decay step to 0.
    zero_after: if not None, number after which the decay function
      will just return 0.
  Returns:
    cosine_decay_fn: a function that computes the cosine decay.
  """
  # pylint:disable=missing-docstring
  def cosine_decay_fn(global_step):
    if global_step is None:
      raise ValueError("global_step is required for cosine_decay.")
    global_step = math_ops.minimum(global_step, decay_steps)
    completed_fraction = (math_ops.cast(global_step, dtypes.float32) /
                          math_ops.cast(decay_steps, dtypes.float32))
    fraction = 2.0 * num_periods * completed_fraction
    decayed = 0.5 * (
        1.0 + math_ops.cos(constant_op.constant(math.pi) * fraction))
    if zero_after is not None:
      decayed = array_ops.where(
          math_ops.greater_equal(fraction, 2 * zero_after), 0.0, decayed)
    return decayed
  # pylint:enable=missing-docstring
  return cosine_decay_fn


def get_restart_decay_fn(decay_steps, num_periods=1, zero_after=None):
  """Returns a function that computes a restart decay.

  This decay computes
    0.5 * (1.0 + cos(pi * (num_periods * global_step) % num_training_steps))

  This is a simplified version of the restart decay introduced in
  "SGDR: Stochastic Gradient Descent with Warm Restarts"
  by Ilya Loshchilov & Frank Hutter, Proceedings of
  ICLR'2017, available at https://arxiv.org/pdf/1608.03983.pdf

  This decay can be used to decay the sign quantity in the AddSign and PowerSign
  optimizers discovered in
  [Bello et al., ICML 2017] Neural Optimizer Search with RL.

  Example usage:
  ```
  decay_steps = 1000
  num_periods = 2.0
  restart_decay_fn = get_restart_decay_fn(decay_steps,
                                          num_periods=num_periods)
  decayed = restart_decay_fn(global_step)
  x *= decayed
  ```
  Args:
    decay_steps: number of steps to decay over.
    num_periods: number of periods for cosine signal. 1 by default,
      which maps the last decay step to 0.
    zero_after: if not None, number after which the decay function
      will return 0.
  Returns:
    restart_decay_fn: a function that computes the restart decay.
  """
  # pylint:disable=missing-docstring
  def restart_decay_fn(global_step):
    if global_step is None:
      raise ValueError("global_step is required for cosine_decay.")
    global_step = math_ops.minimum(global_step, decay_steps)
    num = math_ops.mod(num_periods * math_ops.cast(global_step, dtypes.float32),
                       decay_steps)
    fraction = num / math_ops.cast(decay_steps, dtypes.float32)
    decayed = 0.5 * (
        1.0 + math_ops.cos(constant_op.constant(math.pi) * fraction))
    if zero_after is not None:
      tmp = (math_ops.cast(num_periods * global_step, dtypes.float32) /
             math_ops.cast(decay_steps, dtypes.float32))
      decayed = array_ops.where(
          math_ops.greater_equal(tmp, zero_after), 0.0, decayed)
    return decayed
  # pylint:enable=missing-docstring
  return restart_decay_fn
