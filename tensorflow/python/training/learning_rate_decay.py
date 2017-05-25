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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops


def exponential_decay(learning_rate, global_step, decay_steps, decay_rate,
                      staircase=False, name=None):
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
  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                             100000, 0.96, staircase=True)
  # Passing global_step to minimize() will increment it at each step.
  learning_step = (
      tf.train.GradientDescentOptimizer(learning_rate)
      .minimize(...my loss..., global_step=global_step)
  )
  ```

  Args:
    learning_rate: A scalar `float32` or `float64` `Tensor` or a
      Python number.  The initial learning rate.
    global_step: A scalar `int32` or `int64` `Tensor` or a Python number.
      Global step to use for the decay computation.  Must not be negative.
    decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
      Must be positive.  See the decay computation above.
    decay_rate: A scalar `float32` or `float64` `Tensor` or a
      Python number.  The decay rate.
    staircase: Boolean.  If `True` decay the learning rate at discrete intervals
    name: String.  Optional name of the operation.  Defaults to
      'ExponentialDecay'.

  Returns:
    A scalar `Tensor` of the same type as `learning_rate`.  The decayed
    learning rate.

  Raises:
    ValueError: if `global_step` is not supplied.
  """
  if global_step is None:
    raise ValueError("global_step is required for exponential_decay.")
  with ops.name_scope(name, "ExponentialDecay",
                      [learning_rate, global_step,
                       decay_steps, decay_rate]) as name:
    learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
    dtype = learning_rate.dtype
    global_step = math_ops.cast(global_step, dtype)
    decay_steps = math_ops.cast(decay_steps, dtype)
    decay_rate = math_ops.cast(decay_rate, dtype)
    p = global_step / decay_steps
    if staircase:
      p = math_ops.floor(p)
    return math_ops.multiply(learning_rate, math_ops.pow(decay_rate, p),
                             name=name)


def piecewise_constant(x, boundaries, values, name=None):
  """Piecewise constant from boundaries and interval values.

  Example: use a learning rate that's 1.0 for the first 100000 steps, 0.5
    for steps 100001 to 110000, and 0.1 for any additional steps.

  ```python
  global_step = tf.Variable(0, trainable=False)
  boundaries = [100000, 110000]
  values = [1.0, 0.5, 0.1]
  learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

  # Later, whenever we perform an optimization step, we increment global_step.
  ```

  Args:
    x: A 0-D scalar `Tensor`. Must be one of the following types: `float32`,
      `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`.
    boundaries: A list of `Tensor`s or `int`s or `float`s with strictly
      increasing entries, and with all elements having the same type as `x`.
    values: A list of `Tensor`s or float`s or `int`s that specifies the values
      for the intervals defined by `boundaries`. It should have one more element
      than `boundaries`, and all elements should have the same type.
    name: A string. Optional name of the operation. Defaults to
      'PiecewiseConstant'.

  Returns:
    A 0-D Tensor. Its value is `values[0]` when `x <= boundaries[0]`,
    `values[1]` when `x > boundaries[0]` and `x <= boundaries[1]`, ...,
    and values[-1] when `x > boundaries[-1]`.

  Raises:
    ValueError: if types of `x` and `buondaries` do not match, or types of all
        `values` do not match.
  """
  with ops.name_scope(name, "PiecewiseConstant",
                      [x, boundaries, values, name]) as name:
    x = ops.convert_to_tensor(x)
    # Avoid explicit conversion to x's dtype. This could result in faulty
    # comparisons, for example if floats are converted to integers.
    boundaries = ops.convert_n_to_tensor(boundaries)
    for b in boundaries:
      if b.dtype != x.dtype:
        raise ValueError(
            "Boundaries (%s) must have the same dtype as x (%s)." % (
                b.dtype, x.dtype))
    # TODO(rdipietro): Ensure that boundaries' elements are strictly increasing.
    values = ops.convert_n_to_tensor(values)
    for v in values[1:]:
      if v.dtype != values[0].dtype:
        raise ValueError(
            "Values must have elements all with the same dtype (%s vs %s)." % (
                values[0].dtype, v.dtype))

    pred_fn_pairs = {}
    pred_fn_pairs[x <= boundaries[0]] = lambda: values[0]
    pred_fn_pairs[x > boundaries[-1]] = lambda: values[-1]
    for low, high, v in zip(boundaries[:-1], boundaries[1:], values[1:-1]):
      # Need to bind v here; can do this with lambda v=v: ...
      pred = (x > low) & (x <= high)
      pred_fn_pairs[pred] = lambda v=v: v

    # The default isn't needed here because our conditions are mutually
    # exclusive and exhaustive, but tf.case requires it.
    default = lambda: values[0]
    return control_flow_ops.case(pred_fn_pairs, default, exclusive=True)


def polynomial_decay(learning_rate, global_step, decay_steps,
                     end_learning_rate=0.0001, power=1.0,
                     cycle=False, name=None):
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
  learning_rate = tf.train.polynomial_decay(starter_learning_rate, global_step,
                                            decay_steps, end_learning_rate,
                                            power=0.5)
  # Passing global_step to minimize() will increment it at each step.
  learning_step = (
      tf.train.GradientDescentOptimizer(learning_rate)
      .minimize(...my loss..., global_step=global_step)
  )
  ```

  Args:
    learning_rate: A scalar `float32` or `float64` `Tensor` or a
      Python number.  The initial learning rate.
    global_step: A scalar `int32` or `int64` `Tensor` or a Python number.
      Global step to use for the decay computation.  Must not be negative.
    decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
      Must be positive.  See the decay computation above.
    end_learning_rate: A scalar `float32` or `float64` `Tensor` or a
      Python number.  The minimal end learning rate.
    power: A scalar `float32` or `float64` `Tensor` or a
      Python number.  The power of the polynomial. Defaults to linear, 1.0.
    cycle: A boolean, whether or not it should cycle beyond decay_steps.
    name: String.  Optional name of the operation. Defaults to
      'PolynomialDecay'.

  Returns:
    A scalar `Tensor` of the same type as `learning_rate`.  The decayed
    learning rate.

  Raises:
    ValueError: if `global_step` is not supplied.
  """
  if global_step is None:
    raise ValueError("global_step is required for polynomial_decay.")
  with ops.name_scope(name, "PolynomialDecay",
                      [learning_rate, global_step,
                       decay_steps, end_learning_rate, power]) as name:
    learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
    dtype = learning_rate.dtype
    global_step = math_ops.cast(global_step, dtype)
    decay_steps = math_ops.cast(decay_steps, dtype)
    end_learning_rate = math_ops.cast(end_learning_rate, dtype)
    power = math_ops.cast(power, dtype)
    if cycle:
      # Find the first multiple of decay_steps that is bigger than global_step.
      decay_steps = math_ops.multiply(decay_steps,
                                      math_ops.ceil(global_step / decay_steps))
    else:
      # Make sure that the global_step used is not bigger than decay_steps.
      global_step = math_ops.minimum(global_step, decay_steps)

    p = math_ops.div(global_step, decay_steps)
    return math_ops.add(math_ops.multiply(learning_rate - end_learning_rate,
                                          math_ops.pow(1 - p, power)),
                        end_learning_rate, name=name)


def natural_exp_decay(learning_rate, global_step, decay_steps, decay_rate,
                      staircase=False, name=None):
  """Applies natural exponential decay to the initial learning rate.

  When training a model, it is often recommended to lower the learning rate as
  the training progresses.  This function applies an exponential decay function
  to a provided initial learning rate.  It requires an `global_step` value to
  compute the decayed learning rate.  You can just pass a TensorFlow variable
  that you increment at each training step.

  The function returns the decayed learning rate.  It is computed as:

  ```python
  decayed_learning_rate = learning_rate * exp(-decay_rate * global_step)
  ```

  Example: decay exponentially with a base of 0.96:

  ```python
  ...
  global_step = tf.Variable(0, trainable=False)
  learning_rate = 0.1
  k = 0.5
  learning_rate = tf.train.exponential_time_decay(learning_rate, global_step, k)

  # Passing global_step to minimize() will increment it at each step.
  learning_step = (
      tf.train.GradientDescentOptimizer(learning_rate)
      .minimize(...my loss..., global_step=global_step)
  )
  ```

  Args:
    learning_rate: A scalar `float32` or `float64` `Tensor` or a
      Python number.  The initial learning rate.
    global_step: A Python number.
      Global step to use for the decay computation.  Must not be negative.
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
  """
  if global_step is None:
    raise ValueError("global_step is required for natural_exp_decay.")
  with ops.name_scope(name, "NaturalExpDecay",
                      [learning_rate, global_step, decay_rate]) as name:
    learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
    dtype = learning_rate.dtype
    global_step = math_ops.cast(global_step, dtype)
    decay_steps = math_ops.cast(decay_steps, dtype)
    decay_rate = math_ops.cast(decay_rate, dtype)
    p = global_step / decay_steps
    if staircase:
      p = math_ops.floor(p)
    exponent = math_ops.exp(math_ops.multiply(math_ops.negative(decay_rate), p))
    return math_ops.multiply(learning_rate, exponent, name=name)


def inverse_time_decay(learning_rate, global_step, decay_steps, decay_rate,
                       staircase=False, name=None):
  """Applies inverse time decay to the initial learning rate.

  When training a model, it is often recommended to lower the learning rate as
  the training progresses.  This function applies an inverse decay function
  to a provided initial learning rate.  It requires an `global_step` value to
  compute the decayed learning rate.  You can just pass a TensorFlow variable
  that you increment at each training step.

  The function returns the decayed learning rate.  It is computed as:

  ```python
  decayed_learning_rate = learning_rate / (1 + decay_rate * t)
  ```

  Example: decay 1/t with a rate of 0.5:

  ```python
  ...
  global_step = tf.Variable(0, trainable=False)
  learning_rate = 0.1
  k = 0.5
  learning_rate = tf.train.inverse_time_decay(learning_rate, global_step, k)

  # Passing global_step to minimize() will increment it at each step.
  learning_step = (
      tf.train.GradientDescentOptimizer(learning_rate)
      .minimize(...my loss..., global_step=global_step)
  )
  ```

  Args:
    learning_rate: A scalar `float32` or `float64` `Tensor` or a
      Python number.  The initial learning rate.
    global_step: A Python number.
      Global step to use for the decay computation.  Must not be negative.
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
  """
  if global_step is None:
    raise ValueError("global_step is required for inverse_time_decay.")
  with ops.name_scope(name, "InverseTimeDecay",
                      [learning_rate, global_step, decay_rate]) as name:
    learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
    dtype = learning_rate.dtype
    global_step = math_ops.cast(global_step, dtype)
    decay_steps = math_ops.cast(decay_steps, dtype)
    decay_rate = math_ops.cast(decay_rate, dtype)
    p = global_step / decay_steps
    if staircase:
      p = math_ops.floor(p)
    const = math_ops.cast(constant_op.constant(1), learning_rate.dtype)
    denom = math_ops.add(const, math_ops.multiply(decay_rate, p))
    return math_ops.div(learning_rate, denom, name=name)
