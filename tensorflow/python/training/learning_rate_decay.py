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
"""Various learning rate decay functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from tensorflow.python.eager import context
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["train.exponential_decay"])
def exponential_decay(learning_rate,
                      global_step,
                      decay_steps,
                      decay_rate,
                      staircase=False,
                      name=None):
  """Applies exponential decay to the learning rate.

  When training a model, it is often recommended to lower the learning rate as
  the training progresses.  This function applies an exponential decay function
  to a provided initial learning rate.  It requires a `global_step` value to
  compute the decayed learning rate.  You can just pass a TensorFlow variable
  that you increment at each training step.

  The function returns the decayed learning rate.  It is computed as:

  ```python
  decayed_learning_rate = learning_rate *
                          decay_rate ^ (global_step / decay_steps)
  ```

  If the argument `staircase` is `True`, then `global_step / decay_steps` is an
  integer division and the decayed learning rate follows a staircase function.

  Example: decay every 100000 steps with a base of 0.96:

  ```python
  ...
  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = 0.1
  learning_rate = tf.compat.v1.train.exponential_decay(starter_learning_rate,
  global_step,
                                             100000, 0.96, staircase=True)
  # Passing global_step to minimize() will increment it at each step.
  learning_step = (
      tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
      .minimize(...my loss..., global_step=global_step)
  )
  ```

  Args:
    learning_rate: A scalar `float32` or `float64` `Tensor` or a Python number.
      The initial learning rate.
    global_step: A scalar `int32` or `int64` `Tensor` or a Python number. Global
      step to use for the decay computation.  Must not be negative.
    decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number. Must
      be positive.  See the decay computation above.
    decay_rate: A scalar `float32` or `float64` `Tensor` or a Python number.
      The decay rate.
    staircase: Boolean.  If `True` decay the learning rate at discrete intervals
    name: String.  Optional name of the operation.  Defaults to
      'ExponentialDecay'.

  Returns:
    A scalar `Tensor` of the same type as `learning_rate`.  The decayed
    learning rate.

  Raises:
    ValueError: if `global_step` is not supplied.

  @compatibility(eager)
  When eager execution is enabled, this function returns a function which in
  turn returns the decayed learning rate Tensor. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  @end_compatibility
  """
  decayed_lr = learning_rate_schedule.ExponentialDecay(
      learning_rate, decay_steps, decay_rate, staircase=staircase, name=name)
  if not context.executing_eagerly():
    decayed_lr = decayed_lr(global_step)
  else:
    decayed_lr = functools.partial(decayed_lr, global_step)
  return decayed_lr


@tf_export(v1=["train.piecewise_constant_decay", "train.piecewise_constant"])
def piecewise_constant(x, boundaries, values, name=None):
  """Piecewise constant from boundaries and interval values.

  Example: use a learning rate that's 1.0 for the first 100001 steps, 0.5
    for the next 10000 steps, and 0.1 for any additional steps.

  ```python
  global_step = tf.Variable(0, trainable=False)
  boundaries = [100000, 110000]
  values = [1.0, 0.5, 0.1]
  learning_rate = tf.compat.v1.train.piecewise_constant(global_step, boundaries,
  values)

  # Later, whenever we perform an optimization step, we increment global_step.
  ```

  Args:
    x: A 0-D scalar `Tensor`. Must be one of the following types: `float32`,
      `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`.
    boundaries: A list of `Tensor`s or `int`s or `float`s with strictly
      increasing entries, and with all elements having the same type as `x`.
    values: A list of `Tensor`s or `float`s or `int`s that specifies the values
      for the intervals defined by `boundaries`. It should have one more element
      than `boundaries`, and all elements should have the same type.
    name: A string. Optional name of the operation. Defaults to
      'PiecewiseConstant'.

  Returns:
    A 0-D Tensor. Its value is `values[0]` when `x <= boundaries[0]`,
    `values[1]` when `x > boundaries[0]` and `x <= boundaries[1]`, ...,
    and values[-1] when `x > boundaries[-1]`.

  Raises:
    ValueError: if types of `x` and `boundaries` do not match, or types of all
        `values` do not match or
        the number of elements in the lists does not match.

  @compatibility(eager)
  When eager execution is enabled, this function returns a function which in
  turn returns the decayed learning rate Tensor. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  @end_compatibility
  """
  decayed_lr = learning_rate_schedule.PiecewiseConstantDecay(
      boundaries, values, name=name)
  if not context.executing_eagerly():
    decayed_lr = decayed_lr(x)
  else:
    decayed_lr = functools.partial(decayed_lr, x)
  return decayed_lr


@tf_export(v1=["train.polynomial_decay"])
def polynomial_decay(learning_rate,
                     global_step,
                     decay_steps,
                     end_learning_rate=0.0001,
                     power=1.0,
                     cycle=False,
                     name=None):
  """Applies a polynomial decay to the learning rate.

  It is commonly observed that a monotonically decreasing learning rate, whose
  degree of change is carefully chosen, results in a better performing model.
  This function applies a polynomial decay function to a provided initial
  `learning_rate` to reach an `end_learning_rate` in the given `decay_steps`.

  It requires a `global_step` value to compute the decayed learning rate.  You
  can just pass a TensorFlow variable that you increment at each training step.

  The function returns the decayed learning rate.  It is computed as:

  ```python
  global_step = min(global_step, decay_steps)
  decayed_learning_rate = (learning_rate - end_learning_rate) *
                          (1 - global_step / decay_steps) ^ (power) +
                          end_learning_rate

  ```

  If `cycle` is True then a multiple of `decay_steps` is used, the first one
  that is bigger than `global_steps`.

  ```python
  decay_steps = decay_steps * ceil(global_step / decay_steps)
  decayed_learning_rate = (learning_rate - end_learning_rate) *
                          (1 - global_step / decay_steps) ^ (power) +
                          end_learning_rate

  ```

  Example: decay from 0.1 to 0.01 in 10000 steps using sqrt (i.e. power=0.5):

  ```python
  ...
  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = 0.1
  end_learning_rate = 0.01
  decay_steps = 10000
  learning_rate = tf.compat.v1.train.polynomial_decay(starter_learning_rate,
  global_step,
                                            decay_steps, end_learning_rate,
                                            power=0.5)
  # Passing global_step to minimize() will increment it at each step.
  learning_step = (
      tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
      .minimize(...my loss..., global_step=global_step)
  )
  ```

  Args:
    learning_rate: A scalar `float32` or `float64` `Tensor` or a Python number.
      The initial learning rate.
    global_step: A scalar `int32` or `int64` `Tensor` or a Python number. Global
      step to use for the decay computation.  Must not be negative.
    decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number. Must
      be positive.  See the decay computation above.
    end_learning_rate: A scalar `float32` or `float64` `Tensor` or a Python
      number.  The minimal end learning rate.
    power: A scalar `float32` or `float64` `Tensor` or a Python number.  The
      power of the polynomial. Defaults to linear, 1.0.
    cycle: A boolean, whether or not it should cycle beyond decay_steps.
    name: String.  Optional name of the operation. Defaults to
      'PolynomialDecay'.

  Returns:
    A scalar `Tensor` of the same type as `learning_rate`.  The decayed
    learning rate.

  Raises:
    ValueError: if `global_step` is not supplied.

  @compatibility(eager)
  When eager execution is enabled, this function returns a function which in
  turn returns the decayed learning rate Tensor. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  @end_compatibility
  """
  decayed_lr = learning_rate_schedule.PolynomialDecay(
      learning_rate,
      decay_steps,
      end_learning_rate=end_learning_rate,
      power=power,
      cycle=cycle,
      name=name)

  if not context.executing_eagerly():
    decayed_lr = decayed_lr(global_step)
  else:
    decayed_lr = functools.partial(decayed_lr, global_step)
  return decayed_lr


@tf_export(v1=["train.natural_exp_decay"])
def natural_exp_decay(learning_rate,
                      global_step,
                      decay_steps,
                      decay_rate,
                      staircase=False,
                      name=None):
  """Applies natural exponential decay to the initial learning rate.

  When training a model, it is often recommended to lower the learning rate as
  the training progresses.  This function applies an exponential decay function
  to a provided initial learning rate.  It requires an `global_step` value to
  compute the decayed learning rate.  You can just pass a TensorFlow variable
  that you increment at each training step.

  The function returns the decayed learning rate.  It is computed as:

  ```python
  decayed_learning_rate = learning_rate * exp(-decay_rate * global_step /
  decay_step)
  ```

  or, if `staircase` is `True`, as:

  ```python
  decayed_learning_rate = learning_rate * exp(-decay_rate * floor(global_step /
  decay_step))
  ```

  Example: decay exponentially with a base of 0.96:

  ```python
  ...
  global_step = tf.Variable(0, trainable=False)
  learning_rate = 0.1
  decay_steps = 5
  k = 0.5
  learning_rate = tf.compat.v1.train.natural_exp_decay(learning_rate,
  global_step,
                                             decay_steps, k)

  # Passing global_step to minimize() will increment it at each step.
  learning_step = (
      tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
      .minimize(...my loss..., global_step=global_step)
  )
  ```

  Args:
    learning_rate: A scalar `float32` or `float64` `Tensor` or a Python number.
      The initial learning rate.
    global_step: A Python number. Global step to use for the decay computation.
      Must not be negative.
    decay_steps: How often to apply decay.
    decay_rate: A Python number.  The decay rate.
    staircase: Whether to apply decay in a discrete staircase, as opposed to
      continuous, fashion.
    name: String.  Optional name of the operation.  Defaults to
      'ExponentialTimeDecay'.

  Returns:
    A scalar `Tensor` of the same type as `learning_rate`.  The decayed
    learning rate.

  Raises:
    ValueError: if `global_step` is not supplied.

  @compatibility(eager)
  When eager execution is enabled, this function returns a function which in
  turn returns the decayed learning rate Tensor. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  @end_compatibility
  """
  natural_exp_rate = math_ops.exp(math_ops.negative(decay_rate))
  decayed_lr = learning_rate_schedule.ExponentialDecay(
      learning_rate,
      decay_steps,
      natural_exp_rate,
      staircase=staircase,
      name=name)

  if not context.executing_eagerly():
    decayed_lr = decayed_lr(global_step)
  else:
    decayed_lr = functools.partial(decayed_lr, global_step)
  return decayed_lr


@tf_export(v1=["train.inverse_time_decay"])
def inverse_time_decay(learning_rate,
                       global_step,
                       decay_steps,
                       decay_rate,
                       staircase=False,
                       name=None):
  """Applies inverse time decay to the initial learning rate.

  When training a model, it is often recommended to lower the learning rate as
  the training progresses.  This function applies an inverse decay function
  to a provided initial learning rate.  It requires an `global_step` value to
  compute the decayed learning rate.  You can just pass a TensorFlow variable
  that you increment at each training step.

  The function returns the decayed learning rate.  It is computed as:

  ```python
  decayed_learning_rate = learning_rate / (1 + decay_rate * global_step /
  decay_step)
  ```

  or, if `staircase` is `True`, as:

  ```python
  decayed_learning_rate = learning_rate / (1 + decay_rate * floor(global_step /
  decay_step))
  ```

  Example: decay 1/t with a rate of 0.5:

  ```python
  ...
  global_step = tf.Variable(0, trainable=False)
  learning_rate = 0.1
  decay_steps = 1.0
  decay_rate = 0.5
  learning_rate = tf.compat.v1.train.inverse_time_decay(learning_rate,
  global_step,
  decay_steps, decay_rate)

  # Passing global_step to minimize() will increment it at each step.
  learning_step = (
      tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
      .minimize(...my loss..., global_step=global_step)
  )
  ```

  Args:
    learning_rate: A scalar `float32` or `float64` `Tensor` or a Python number.
      The initial learning rate.
    global_step: A Python number. Global step to use for the decay computation.
      Must not be negative.
    decay_steps: How often to apply decay.
    decay_rate: A Python number.  The decay rate.
    staircase: Whether to apply decay in a discrete staircase, as opposed to
      continuous, fashion.
    name: String.  Optional name of the operation.  Defaults to
      'InverseTimeDecay'.

  Returns:
    A scalar `Tensor` of the same type as `learning_rate`.  The decayed
    learning rate.

  Raises:
    ValueError: if `global_step` is not supplied.

  @compatibility(eager)
  When eager execution is enabled, this function returns a function which in
  turn returns the decayed learning rate Tensor. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  @end_compatibility
  """
  decayed_lr = learning_rate_schedule.InverseTimeDecay(
      learning_rate, decay_steps, decay_rate, staircase=staircase, name=name)

  if not context.executing_eagerly():
    decayed_lr = decayed_lr(global_step)
  else:
    decayed_lr = functools.partial(decayed_lr, global_step)
  return decayed_lr


@tf_export(v1=["train.cosine_decay"])
def cosine_decay(learning_rate, global_step, decay_steps, alpha=0.0, name=None):
  """Applies cosine decay to the learning rate.

  See [Loshchilov & Hutter, ICLR2016], SGDR: Stochastic Gradient Descent
  with Warm Restarts. https://arxiv.org/abs/1608.03983

  When training a model, it is often recommended to lower the learning rate as
  the training progresses.  This function applies a cosine decay function
  to a provided initial learning rate.  It requires a `global_step` value to
  compute the decayed learning rate.  You can just pass a TensorFlow variable
  that you increment at each training step.

  The function returns the decayed learning rate.  It is computed as:
  ```python
  global_step = min(global_step, decay_steps)
  cosine_decay = 0.5 * (1 + cos(pi * global_step / decay_steps))
  decayed = (1 - alpha) * cosine_decay + alpha
  decayed_learning_rate = learning_rate * decayed
  ```

  Example usage:
  ```python
  decay_steps = 1000
  lr_decayed = cosine_decay(learning_rate, global_step, decay_steps)
  ```

  Args:
    learning_rate: A scalar `float32` or `float64` Tensor or a Python number.
      The initial learning rate.
    global_step: A scalar `int32` or `int64` `Tensor` or a Python number. Global
      step to use for the decay computation.
    decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number. Number
      of steps to decay over.
    alpha: A scalar `float32` or `float64` Tensor or a Python number. Minimum
      learning rate value as a fraction of learning_rate.
    name: String. Optional name of the operation.  Defaults to 'CosineDecay'.

  Returns:
    A scalar `Tensor` of the same type as `learning_rate`.  The decayed
    learning rate.
  Raises:
    ValueError: if `global_step` is not supplied.

  @compatibility(eager)
  When eager execution is enabled, this function returns a function which in
  turn returns the decayed learning rate Tensor. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  @end_compatibility
  """
  decayed_lr = learning_rate_schedule.CosineDecay(
      learning_rate, decay_steps, alpha=alpha, name=name)

  if not context.executing_eagerly():
    decayed_lr = decayed_lr(global_step)
  else:
    decayed_lr = functools.partial(decayed_lr, global_step)
  return decayed_lr


@tf_export(v1=["train.cosine_decay_restarts"])
def cosine_decay_restarts(learning_rate,
                          global_step,
                          first_decay_steps,
                          t_mul=2.0,
                          m_mul=1.0,
                          alpha=0.0,
                          name=None):
  """Applies cosine decay with restarts to the learning rate.

  See [Loshchilov & Hutter, ICLR2016], SGDR: Stochastic Gradient Descent
  with Warm Restarts. https://arxiv.org/abs/1608.03983

  When training a model, it is often recommended to lower the learning rate as
  the training progresses.  This function applies a cosine decay function with
  restarts to a provided initial learning rate.  It requires a `global_step`
  value to compute the decayed learning rate.  You can just pass a TensorFlow
  variable that you increment at each training step.

  The function returns the decayed learning rate while taking into account
  possible warm restarts. The learning rate multiplier first decays
  from 1 to `alpha` for `first_decay_steps` steps. Then, a warm
  restart is performed. Each new warm restart runs for `t_mul` times more steps
  and with `m_mul` times smaller initial learning rate.

  Example usage:
  ```python
  first_decay_steps = 1000
  lr_decayed = cosine_decay_restarts(learning_rate, global_step,
                                     first_decay_steps)
  ```

  Args:
    learning_rate: A scalar `float32` or `float64` Tensor or a Python number.
      The initial learning rate.
    global_step: A scalar `int32` or `int64` `Tensor` or a Python number. Global
      step to use for the decay computation.
    first_decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
      Number of steps to decay over.
    t_mul: A scalar `float32` or `float64` `Tensor` or a Python number. Used to
      derive the number of iterations in the i-th period
    m_mul: A scalar `float32` or `float64` `Tensor` or a Python number.
      Used to derive the initial learning rate of the i-th period:
    alpha: A scalar `float32` or `float64` Tensor or a Python number. Minimum
      learning rate value as a fraction of the learning_rate.
    name: String. Optional name of the operation.  Defaults to 'SGDRDecay'.

  Returns:
    A scalar `Tensor` of the same type as `learning_rate`.  The decayed
    learning rate.
  Raises:
    ValueError: if `global_step` is not supplied.

  @compatibility(eager)
  When eager execution is enabled, this function returns a function which in
  turn returns the decayed learning rate Tensor. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  @end_compatibility
  """
  decayed_lr = learning_rate_schedule.CosineDecayRestarts(
      learning_rate,
      first_decay_steps,
      t_mul=t_mul,
      m_mul=m_mul,
      alpha=alpha,
      name=name)

  if not context.executing_eagerly():
    decayed_lr = decayed_lr(global_step)
  else:
    decayed_lr = functools.partial(decayed_lr, global_step)
  return decayed_lr


@tf_export(v1=["train.linear_cosine_decay"])
def linear_cosine_decay(learning_rate,
                        global_step,
                        decay_steps,
                        num_periods=0.5,
                        alpha=0.0,
                        beta=0.001,
                        name=None):
  """Applies linear cosine decay to the learning rate.

  See [Bello et al., ICML2017] Neural Optimizer Search with RL.
  https://arxiv.org/abs/1709.07417

  For the idea of warm starts here controlled by `num_periods`,
  see [Loshchilov & Hutter, ICLR2016] SGDR: Stochastic Gradient Descent
  with Warm Restarts. https://arxiv.org/abs/1608.03983

  Note that linear cosine decay is more aggressive than cosine decay and
  larger initial learning rates can typically be used.

  When training a model, it is often recommended to lower the learning rate as
  the training progresses.  This function applies a linear cosine decay function
  to a provided initial learning rate.  It requires a `global_step` value to
  compute the decayed learning rate.  You can just pass a TensorFlow variable
  that you increment at each training step.

  The function returns the decayed learning rate.  It is computed as:
  ```python
  global_step = min(global_step, decay_steps)
  linear_decay = (decay_steps - global_step) / decay_steps)
  cosine_decay = 0.5 * (
      1 + cos(pi * 2 * num_periods * global_step / decay_steps))
  decayed = (alpha + linear_decay) * cosine_decay + beta
  decayed_learning_rate = learning_rate * decayed
  ```

  Example usage:
  ```python
  decay_steps = 1000
  lr_decayed = linear_cosine_decay(learning_rate, global_step, decay_steps)
  ```

  Args:
    learning_rate: A scalar `float32` or `float64` Tensor or a Python number.
      The initial learning rate.
    global_step: A scalar `int32` or `int64` `Tensor` or a Python number. Global
      step to use for the decay computation.
    decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number. Number
      of steps to decay over.
    num_periods: Number of periods in the cosine part of the decay. See
      computation above.
    alpha: See computation above.
    beta: See computation above.
    name: String.  Optional name of the operation.  Defaults to
      'LinearCosineDecay'.

  Returns:
    A scalar `Tensor` of the same type as `learning_rate`.  The decayed
    learning rate.
  Raises:
    ValueError: if `global_step` is not supplied.

  @compatibility(eager)
  When eager execution is enabled, this function returns a function which in
  turn returns the decayed learning rate Tensor. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  @end_compatibility
  """
  decayed_lr = learning_rate_schedule.LinearCosineDecay(
      learning_rate,
      decay_steps,
      num_periods=num_periods,
      alpha=alpha,
      beta=beta,
      name=name)

  if not context.executing_eagerly():
    decayed_lr = decayed_lr(global_step)
  else:
    decayed_lr = functools.partial(decayed_lr, global_step)
  return decayed_lr


@tf_export(v1=["train.noisy_linear_cosine_decay"])
def noisy_linear_cosine_decay(learning_rate,
                              global_step,
                              decay_steps,
                              initial_variance=1.0,
                              variance_decay=0.55,
                              num_periods=0.5,
                              alpha=0.0,
                              beta=0.001,
                              name=None):
  """Applies noisy linear cosine decay to the learning rate.

  See [Bello et al., ICML2017] Neural Optimizer Search with RL.
  https://arxiv.org/abs/1709.07417

  For the idea of warm starts here controlled by `num_periods`,
  see [Loshchilov & Hutter, ICLR2016] SGDR: Stochastic Gradient Descent
  with Warm Restarts. https://arxiv.org/abs/1608.03983

  Note that linear cosine decay is more aggressive than cosine decay and
  larger initial learning rates can typically be used.

  When training a model, it is often recommended to lower the learning rate as
  the training progresses.  This function applies a noisy linear
  cosine decay function to a provided initial learning rate.
  It requires a `global_step` value to compute the decayed learning rate.
  You can just pass a TensorFlow variable that you increment at each
  training step.

  The function returns the decayed learning rate.  It is computed as:
  ```python
  global_step = min(global_step, decay_steps)
  linear_decay = (decay_steps - global_step) / decay_steps)
  cosine_decay = 0.5 * (
      1 + cos(pi * 2 * num_periods * global_step / decay_steps))
  decayed = (alpha + linear_decay + eps_t) * cosine_decay + beta
  decayed_learning_rate = learning_rate * decayed
  ```
  where eps_t is 0-centered gaussian noise with variance
  initial_variance / (1 + global_step) ** variance_decay

  Example usage:
  ```python
  decay_steps = 1000
  lr_decayed = noisy_linear_cosine_decay(
    learning_rate, global_step, decay_steps)
  ```

  Args:
    learning_rate: A scalar `float32` or `float64` Tensor or a Python number.
      The initial learning rate.
    global_step: A scalar `int32` or `int64` `Tensor` or a Python number. Global
      step to use for the decay computation.
    decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number. Number
      of steps to decay over.
    initial_variance: initial variance for the noise. See computation above.
    variance_decay: decay for the noise's variance. See computation above.
    num_periods: Number of periods in the cosine part of the decay. See
      computation above.
    alpha: See computation above.
    beta: See computation above.
    name: String.  Optional name of the operation.  Defaults to
      'NoisyLinearCosineDecay'.

  Returns:
    A scalar `Tensor` of the same type as `learning_rate`.  The decayed
    learning rate.
  Raises:
    ValueError: if `global_step` is not supplied.

  @compatibility(eager)
  When eager execution is enabled, this function returns a function which in
  turn returns the decayed learning rate Tensor. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  @end_compatibility
  """
  decayed_lr = learning_rate_schedule.NoisyLinearCosineDecay(
      learning_rate,
      decay_steps,
      initial_variance=initial_variance,
      variance_decay=variance_decay,
      num_periods=num_periods,
      alpha=alpha,
      beta=beta,
      name=name)

  if not context.executing_eagerly():
    decayed_lr = decayed_lr(global_step)
  else:
    decayed_lr = functools.partial(decayed_lr, global_step)
  return decayed_lr
