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

import abc
import math

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.optimizers.schedules.LearningRateSchedule")
class LearningRateSchedule(object):
  """A serializable learning rate decay schedule.

  `LearningRateSchedule`s can be passed in as the learning rate of optimizers in
  `tf.keras.optimizers`. They can be serialized and deserialized using
  `tf.keras.optimizers.schedules.serialize` and
  `tf.keras.optimizers.schedules.deserialize`.
  """

  @abc.abstractmethod
  def __call__(self, step):
    raise NotImplementedError("Learning rate schedule must override __call__")

  @abc.abstractmethod
  def get_config(self):
    raise NotImplementedError("Learning rate schedule must override get_config")

  @classmethod
  def from_config(cls, config):
    """Instantiates a `LearningRateSchedule` from its config.

    Args:
        config: Output of `get_config()`.

    Returns:
        A `LearningRateSchedule` instance.
    """
    return cls(**config)


@keras_export("keras.optimizers.schedules.ExponentialDecay")
class ExponentialDecay(LearningRateSchedule):
  """A LearningRateSchedule that uses an exponential decay schedule.

  When training a model, it is often recommended to lower the learning rate as
  the training progresses. This schedule applies an exponential decay function
  to an optimizer step, given a provided initial learning rate.

  The schedule a 1-arg callable that produces a decayed learning
  rate when passed the current optimizer step. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  It is computed as:

  ```python
  def decayed_learning_rate(step):
    return initial_learning_rate * decay_rate ^ (step / decay_steps)
  ```

  If the argument `staircase` is `True`, then `step / decay_steps` is
  an integer division and the decayed learning rate follows a
  staircase function.

  You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
  as the learning rate.
  Example: When fitting a Keras model, decay every 100000 steps with a base
  of 0.96:

  ```python
  initial_learning_rate = 0.1
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate,
      decay_steps=100000,
      decay_rate=0.96,
      staircase=True)

  model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  model.fit(data, labels, epochs=5)
  ```

  The learning rate schedule is also serializable and deserializable using
  `tf.keras.optimizers.schedules.serialize` and
  `tf.keras.optimizers.schedules.deserialize`.

  Returns:
    A 1-arg callable learning rate schedule that takes the current optimizer
    step and outputs the decayed learning rate, a scalar `Tensor` of the same
    type as `initial_learning_rate`.
  """

  def __init__(
      self,
      initial_learning_rate,
      decay_steps,
      decay_rate,
      staircase=False,
      name=None):
    """Applies exponential decay to the learning rate.

    Args:
      initial_learning_rate: A scalar `float32` or `float64` `Tensor` or a
        Python number.  The initial learning rate.
      decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
        Must be positive.  See the decay computation above.
      decay_rate: A scalar `float32` or `float64` `Tensor` or a
        Python number.  The decay rate.
      staircase: Boolean.  If `True` decay the learning rate at discrete
        intervals
      name: String.  Optional name of the operation.  Defaults to
        'ExponentialDecay'.
    """
    super(ExponentialDecay, self).__init__()
    self.initial_learning_rate = initial_learning_rate
    self.decay_steps = decay_steps
    self.decay_rate = decay_rate
    self.staircase = staircase
    self.name = name

  def __call__(self, step):
    with ops.name_scope_v2(self.name or "ExponentialDecay") as name:
      initial_learning_rate = ops.convert_to_tensor_v2(
          self.initial_learning_rate, name="initial_learning_rate")
      dtype = initial_learning_rate.dtype
      decay_steps = math_ops.cast(self.decay_steps, dtype)
      decay_rate = math_ops.cast(self.decay_rate, dtype)

      global_step_recomp = math_ops.cast(step, dtype)
      p = global_step_recomp / decay_steps
      if self.staircase:
        p = math_ops.floor(p)
      return math_ops.multiply(
          initial_learning_rate, math_ops.pow(decay_rate, p), name=name)

  def get_config(self):
    return {
        "initial_learning_rate": self.initial_learning_rate,
        "decay_steps": self.decay_steps,
        "decay_rate": self.decay_rate,
        "staircase": self.staircase,
        "name": self.name
    }


@keras_export("keras.optimizers.schedules.PiecewiseConstantDecay")
class PiecewiseConstantDecay(LearningRateSchedule):
  """A LearningRateSchedule that uses a piecewise constant decay schedule.

  The function returns a 1-arg callable to compute the piecewise constant
  when passed the current optimizer step. This can be useful for changing the
  learning rate value across different invocations of optimizer functions.

  Example: use a learning rate that's 1.0 for the first 100001 steps, 0.5
    for the next 10000 steps, and 0.1 for any additional steps.

  ```python
  step = tf.Variable(0, trainable=False)
  boundaries = [100000, 110000]
  values = [1.0, 0.5, 0.1]
  learning_rate_fn = keras.optimizers.schedules.PiecewiseConstantDecay(
      boundaries, values)

  # Later, whenever we perform an optimization step, we pass in the step.
  learning_rate = learning_rate_fn(step)
  ```

  You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
  as the learning rate. The learning rate schedule is also serializable and
  deserializable using `tf.keras.optimizers.schedules.serialize` and
  `tf.keras.optimizers.schedules.deserialize`.

  Returns:
    A 1-arg callable learning rate schedule that takes the current optimizer
    step and outputs the decayed learning rate, a scalar `Tensor` of the same
    type as the boundary tensors.

    The output of the 1-arg function that takes the `step`
    is `values[0]` when `step <= boundaries[0]`,
    `values[1]` when `step > boundaries[0]` and `step <= boundaries[1]`, ...,
    and values[-1] when `step > boundaries[-1]`.
  """

  def __init__(
      self,
      boundaries,
      values,
      name=None):
    """Piecewise constant from boundaries and interval values.

    Args:
      boundaries: A list of `Tensor`s or `int`s or `float`s with strictly
        increasing entries, and with all elements having the same type as the
        optimizer step.
      values: A list of `Tensor`s or `float`s or `int`s that specifies the
        values for the intervals defined by `boundaries`. It should have one
        more element than `boundaries`, and all elements should have the same
        type.
      name: A string. Optional name of the operation. Defaults to
        'PiecewiseConstant'.

    Raises:
      ValueError: if the number of elements in the lists do not match.
    """
    super(PiecewiseConstantDecay, self).__init__()

    if len(boundaries) != len(values) - 1:
      raise ValueError(
          "The length of boundaries should be 1 less than the length of values")

    self.boundaries = boundaries
    self.values = values
    self.name = name

  def __call__(self, step):
    with ops.name_scope_v2(self.name or "PiecewiseConstant"):
      boundaries = ops.convert_n_to_tensor(self.boundaries)
      values = ops.convert_n_to_tensor(self.values)
      x_recomp = ops.convert_to_tensor_v2(step)
      for i, b in enumerate(boundaries):
        if b.dtype.base_dtype != x_recomp.dtype.base_dtype:
          # We cast the boundaries to have the same type as the step
          b = math_ops.cast(b, x_recomp.dtype.base_dtype)
          boundaries[i] = b
      pred_fn_pairs = []
      pred_fn_pairs.append((x_recomp <= boundaries[0], lambda: values[0]))
      pred_fn_pairs.append((x_recomp > boundaries[-1], lambda: values[-1]))
      for low, high, v in zip(boundaries[:-1], boundaries[1:], values[1:-1]):
        # Need to bind v here; can do this with lambda v=v: ...
        pred = (x_recomp > low) & (x_recomp <= high)
        pred_fn_pairs.append((pred, lambda v=v: v))

      # The default isn't needed here because our conditions are mutually
      # exclusive and exhaustive, but tf.case requires it.
      default = lambda: values[0]
      return control_flow_ops.case(pred_fn_pairs, default, exclusive=True)

  def get_config(self):
    return {
        "boundaries": self.boundaries,
        "values": self.values,
        "name": self.name
    }


@keras_export("keras.optimizers.schedules.PolynomialDecay")
class PolynomialDecay(LearningRateSchedule):
  """A LearningRateSchedule that uses a polynomial decay schedule.

  It is commonly observed that a monotonically decreasing learning rate, whose
  degree of change is carefully chosen, results in a better performing model.
  This schedule applies a polynomial decay function to an optimizer step,
  given a provided `initial_learning_rate`, to reach an `end_learning_rate`
  in the given `decay_steps`.

  It requires a `step` value to compute the decayed learning rate. You
  can just pass a TensorFlow variable that you increment at each training
  step.

  The schedule is a 1-arg callable that produces a decayed learning rate
  when passed the current optimizer step. This can be useful for changing the
  learning rate value across different invocations of optimizer functions.
  It is computed as:

  ```python
  def decayed_learning_rate(step):
    step = min(step, decay_steps)
    return ((initial_learning_rate - end_learning_rate) *
            (1 - step / decay_steps) ^ (power)
           ) + end_learning_rate
  ```

  If `cycle` is True then a multiple of `decay_steps` is used, the first one
  that is bigger than `step`.

  ```python
  def decayed_learning_rate(step):
    decay_steps = decay_steps * ceil(step / decay_steps)
    return ((initial_learning_rate - end_learning_rate) *
            (1 - step / decay_steps) ^ (power)
           ) + end_learning_rate
  ```

  You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
  as the learning rate.
  Example: Fit a model while decaying from 0.1 to 0.01 in 10000 steps using
  sqrt (i.e. power=0.5):

  ```python
  ...
  starter_learning_rate = 0.1
  end_learning_rate = 0.01
  decay_steps = 10000
  learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
      starter_learning_rate,
      decay_steps,
      end_learning_rate,
      power=0.5)

  model.compile(optimizer=tf.keras.optimizers.SGD(
                    learning_rate=learning_rate_fn),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  model.fit(data, labels, epochs=5)
  ```

  The learning rate schedule is also serializable and deserializable using
  `tf.keras.optimizers.schedules.serialize` and
  `tf.keras.optimizers.schedules.deserialize`.

  Returns:
    A 1-arg callable learning rate schedule that takes the current optimizer
    step and outputs the decayed learning rate, a scalar `Tensor` of the same
    type as `initial_learning_rate`.
  """

  def __init__(
      self,
      initial_learning_rate,
      decay_steps,
      end_learning_rate=0.0001,
      power=1.0,
      cycle=False,
      name=None):
    """Applies a polynomial decay to the learning rate.

    Args:
      initial_learning_rate: A scalar `float32` or `float64` `Tensor` or a
        Python number.  The initial learning rate.
      decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
        Must be positive.  See the decay computation above.
      end_learning_rate: A scalar `float32` or `float64` `Tensor` or a
        Python number.  The minimal end learning rate.
      power: A scalar `float32` or `float64` `Tensor` or a
        Python number.  The power of the polynomial. Defaults to linear, 1.0.
      cycle: A boolean, whether or not it should cycle beyond decay_steps.
      name: String.  Optional name of the operation. Defaults to
        'PolynomialDecay'.
    """
    super(PolynomialDecay, self).__init__()

    self.initial_learning_rate = initial_learning_rate
    self.decay_steps = decay_steps
    self.end_learning_rate = end_learning_rate
    self.power = power
    self.cycle = cycle
    self.name = name

  def __call__(self, step):
    with ops.name_scope_v2(self.name or "PolynomialDecay") as name:
      initial_learning_rate = ops.convert_to_tensor_v2(
          self.initial_learning_rate, name="initial_learning_rate")
      dtype = initial_learning_rate.dtype
      end_learning_rate = math_ops.cast(self.end_learning_rate, dtype)
      power = math_ops.cast(self.power, dtype)

      global_step_recomp = math_ops.cast(step, dtype)
      decay_steps_recomp = math_ops.cast(self.decay_steps, dtype)
      if self.cycle:
        # Find the first multiple of decay_steps that is bigger than
        # global_step. If global_step is zero set the multiplier to 1
        multiplier = control_flow_ops.cond(
            math_ops.equal(global_step_recomp, 0), lambda: 1.0,
            lambda: math_ops.ceil(global_step_recomp / self.decay_steps))
        decay_steps_recomp = math_ops.multiply(decay_steps_recomp, multiplier)
      else:
        # Make sure that the global_step used is not bigger than decay_steps.
        global_step_recomp = math_ops.minimum(global_step_recomp,
                                              decay_steps_recomp)

      p = math_ops.divide(global_step_recomp, decay_steps_recomp)
      return math_ops.add(
          math_ops.multiply(initial_learning_rate - end_learning_rate,
                            math_ops.pow(1 - p, power)),
          end_learning_rate,
          name=name)

  def get_config(self):
    return {
        "initial_learning_rate": self.initial_learning_rate,
        "decay_steps": self.decay_steps,
        "end_learning_rate": self.end_learning_rate,
        "power": self.power,
        "cycle": self.cycle,
        "name": self.name
    }


@keras_export("keras.optimizers.schedules.InverseTimeDecay")
class InverseTimeDecay(LearningRateSchedule):
  """A LearningRateSchedule that uses an inverse time decay schedule.

  When training a model, it is often recommended to lower the learning rate as
  the training progresses. This schedule applies the inverse decay function
  to an optimizer step, given a provided initial learning rate.
  It requires a `step` value to compute the decayed learning rate. You can
  just pass a TensorFlow variable that you increment at each training step.

  The schedule a 1-arg callable that produces a decayed learning
  rate when passed the current optimizer step. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  It is computed as:

  ```python
  def decayed_learning_rate(step):
    return initial_learning_rate / (1 + decay_rate * step / decay_step)
  ```

  or, if `staircase` is `True`, as:

  ```python
  def decayed_learning_rate(step):
    return initial_learning_rate / (1 + decay_rate * floor(step / decay_step))
  ```

  You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
  as the learning rate.
  Example: Fit a Keras model when decaying 1/t with a rate of 0.5:

  ```python
  ...
  initial_learning_rate = 0.1
  decay_steps = 1.0
  decay_rate = 0.5
  learning_rate_fn = keras.optimizers.schedules.InverseTimeDecay(
    initial_learning_rate, decay_steps, decay_rate)

  model.compile(optimizer=tf.keras.optimizers.SGD(
                    learning_rate=learning_rate_fn),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  model.fit(data, labels, epochs=5)
  ```

  Returns:
    A 1-arg callable learning rate schedule that takes the current optimizer
    step and outputs the decayed learning rate, a scalar `Tensor` of the same
    type as `initial_learning_rate`.
  """

  def __init__(
      self,
      initial_learning_rate,
      decay_steps,
      decay_rate,
      staircase=False,
      name=None):
    """Applies inverse time decay to the initial learning rate.

    Args:
      initial_learning_rate: A scalar `float32` or `float64` `Tensor` or a
        Python number.  The initial learning rate.
      decay_steps: How often to apply decay.
      decay_rate: A Python number.  The decay rate.
      staircase: Whether to apply decay in a discrete staircase, as opposed to
        continuous, fashion.
      name: String.  Optional name of the operation.  Defaults to
        'InverseTimeDecay'.
    """
    super(InverseTimeDecay, self).__init__()

    self.initial_learning_rate = initial_learning_rate
    self.decay_steps = decay_steps
    self.decay_rate = decay_rate
    self.staircase = staircase
    self.name = name

  def __call__(self, step):
    with ops.name_scope_v2(self.name or "InverseTimeDecay") as name:
      initial_learning_rate = ops.convert_to_tensor_v2(
          self.initial_learning_rate, name="initial_learning_rate")
      dtype = initial_learning_rate.dtype
      decay_steps = math_ops.cast(self.decay_steps, dtype)
      decay_rate = math_ops.cast(self.decay_rate, dtype)

      global_step_recomp = math_ops.cast(step, dtype)
      p = global_step_recomp / decay_steps
      if self.staircase:
        p = math_ops.floor(p)
      const = math_ops.cast(constant_op.constant(1), dtype)
      denom = math_ops.add(const, math_ops.multiply(decay_rate, p))
      return math_ops.divide(initial_learning_rate, denom, name=name)

  def get_config(self):
    return {
        "initial_learning_rate": self.initial_learning_rate,
        "decay_steps": self.decay_steps,
        "decay_rate": self.decay_rate,
        "staircase": self.staircase,
        "name": self.name
    }


@keras_export("keras.experimental.CosineDecay")
class CosineDecay(LearningRateSchedule):
  """A LearningRateSchedule that uses a cosine decay schedule.

  See [Loshchilov & Hutter, ICLR2016], SGDR: Stochastic Gradient Descent
  with Warm Restarts. https://arxiv.org/abs/1608.03983

  When training a model, it is often recommended to lower the learning rate as
  the training progresses. This schedule applies a cosine decay function
  to an optimizer step, given a provided initial learning rate.
  It requires a `step` value to compute the decayed learning rate. You can
  just pass a TensorFlow variable that you increment at each training step.

  The schedule a 1-arg callable that produces a decayed learning
  rate when passed the current optimizer step. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  It is computed as:

  ```python
  def decayed_learning_rate(step):
    step = min(step, decay_steps)
    cosine_decay = 0.5 * (1 + cos(pi * step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    return initial_learning_rate * decayed
  ```

  Example usage:
  ```python
  decay_steps = 1000
  lr_decayed_fn = tf.keras.experimental.CosineDecay(
      initial_learning_rate, decay_steps)
  ```

  You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
  as the learning rate. The learning rate schedule is also serializable and
  deserializable using `tf.keras.optimizers.schedules.serialize` and
  `tf.keras.optimizers.schedules.deserialize`.

  Returns:
    A 1-arg callable learning rate schedule that takes the current optimizer
    step and outputs the decayed learning rate, a scalar `Tensor` of the same
    type as `initial_learning_rate`.
  """

  def __init__(
      self,
      initial_learning_rate,
      decay_steps,
      alpha=0.0,
      name=None):
    """Applies cosine decay to the learning rate.

    Args:
      initial_learning_rate: A scalar `float32` or `float64` Tensor or a
        Python number. The initial learning rate.
      decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
        Number of steps to decay over.
      alpha: A scalar `float32` or `float64` Tensor or a Python number.
        Minimum learning rate value as a fraction of initial_learning_rate.
      name: String. Optional name of the operation.  Defaults to 'CosineDecay'.
    """
    super(CosineDecay, self).__init__()

    self.initial_learning_rate = initial_learning_rate
    self.decay_steps = decay_steps
    self.alpha = alpha
    self.name = name

  def __call__(self, step):
    with ops.name_scope_v2(self.name or "CosineDecay"):
      initial_learning_rate = ops.convert_to_tensor_v2(
          self.initial_learning_rate, name="initial_learning_rate")
      dtype = initial_learning_rate.dtype
      decay_steps = math_ops.cast(self.decay_steps, dtype)

      global_step_recomp = math_ops.cast(step, dtype)
      global_step_recomp = math_ops.minimum(global_step_recomp, decay_steps)
      completed_fraction = global_step_recomp / decay_steps
      cosine_decayed = 0.5 * (1.0 + math_ops.cos(
          constant_op.constant(math.pi) * completed_fraction))

      decayed = (1 - self.alpha) * cosine_decayed + self.alpha
      return math_ops.multiply(initial_learning_rate, decayed)

  def get_config(self):
    return {
        "initial_learning_rate": self.initial_learning_rate,
        "decay_steps": self.decay_steps,
        "alpha": self.alpha,
        "name": self.name
    }


@keras_export("keras.experimental.CosineDecayRestarts")
class CosineDecayRestarts(LearningRateSchedule):
  """A LearningRateSchedule that uses a cosine decay schedule with restarts.

  See [Loshchilov & Hutter, ICLR2016], SGDR: Stochastic Gradient Descent
  with Warm Restarts. https://arxiv.org/abs/1608.03983

  When training a model, it is often recommended to lower the learning rate as
  the training progresses. This schedule applies a cosine decay function with
  restarts to an optimizer step, given a provided initial learning rate.
  It requires a `step` value to compute the decayed learning rate. You can
  just pass a TensorFlow variable that you increment at each training step.

  The schedule a 1-arg callable that produces a decayed learning
  rate when passed the current optimizer step. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.

  The learning rate multiplier first decays
  from 1 to `alpha` for `first_decay_steps` steps. Then, a warm
  restart is performed. Each new warm restart runs for `t_mul` times more
  steps and with `m_mul` times smaller initial learning rate.

  Example usage:
  ```python
  first_decay_steps = 1000
  lr_decayed_fn = (
    tf.keras.experimental.CosineDecayRestarts(
        initial_learning_rate,
        first_decay_steps))
  ```

  You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
  as the learning rate. The learning rate schedule is also serializable and
  deserializable using `tf.keras.optimizers.schedules.serialize` and
  `tf.keras.optimizers.schedules.deserialize`.

  Returns:
    A 1-arg callable learning rate schedule that takes the current optimizer
    step and outputs the decayed learning rate, a scalar `Tensor` of the same
    type as `initial_learning_rate`.
  """

  def __init__(
      self,
      initial_learning_rate,
      first_decay_steps,
      t_mul=2.0,
      m_mul=1.0,
      alpha=0.0,
      name=None):
    """Applies cosine decay with restarts to the learning rate.

    Args:
      initial_learning_rate: A scalar `float32` or `float64` Tensor or a Python
        number. The initial learning rate.
      first_decay_steps: A scalar `int32` or `int64` `Tensor` or a Python
        number. Number of steps to decay over.
      t_mul: A scalar `float32` or `float64` `Tensor` or a Python number.
        Used to derive the number of iterations in the i-th period
      m_mul: A scalar `float32` or `float64` `Tensor` or a Python number.
        Used to derive the initial learning rate of the i-th period:
      alpha: A scalar `float32` or `float64` Tensor or a Python number.
        Minimum learning rate value as a fraction of the initial_learning_rate.
      name: String. Optional name of the operation.  Defaults to 'SGDRDecay'.
    """
    super(CosineDecayRestarts, self).__init__()

    self.initial_learning_rate = initial_learning_rate
    self.first_decay_steps = first_decay_steps
    self._t_mul = t_mul
    self._m_mul = m_mul
    self.alpha = alpha
    self.name = name

  def __call__(self, step):
    with ops.name_scope_v2(self.name or "SGDRDecay") as name:
      initial_learning_rate = ops.convert_to_tensor_v2(
          self.initial_learning_rate, name="initial_learning_rate")
      dtype = initial_learning_rate.dtype
      first_decay_steps = math_ops.cast(self.first_decay_steps, dtype)
      alpha = math_ops.cast(self.alpha, dtype)
      t_mul = math_ops.cast(self._t_mul, dtype)
      m_mul = math_ops.cast(self._m_mul, dtype)

      global_step_recomp = math_ops.cast(step, dtype)
      completed_fraction = global_step_recomp / first_decay_steps

      def compute_step(completed_fraction, geometric=False):
        """Helper for `cond` operation."""
        if geometric:
          i_restart = math_ops.floor(
              math_ops.log(1.0 - completed_fraction * (1.0 - t_mul)) /
              math_ops.log(t_mul))

          sum_r = (1.0 - t_mul**i_restart) / (1.0 - t_mul)
          completed_fraction = (completed_fraction - sum_r) / t_mul**i_restart

        else:
          i_restart = math_ops.floor(completed_fraction)
          completed_fraction -= i_restart

        return i_restart, completed_fraction

      i_restart, completed_fraction = control_flow_ops.cond(
          math_ops.equal(t_mul, 1.0),
          lambda: compute_step(completed_fraction, geometric=False),
          lambda: compute_step(completed_fraction, geometric=True))

      m_fac = m_mul**i_restart
      cosine_decayed = 0.5 * m_fac * (1.0 + math_ops.cos(
          constant_op.constant(math.pi) * completed_fraction))
      decayed = (1 - alpha) * cosine_decayed + alpha

      return math_ops.multiply(initial_learning_rate, decayed, name=name)

  def get_config(self):
    return {
        "initial_learning_rate": self.initial_learning_rate,
        "first_decay_steps": self.first_decay_steps,
        "t_mul": self._t_mul,
        "m_mul": self._m_mul,
        "alpha": self.alpha,
        "name": self.name
    }


@keras_export("keras.experimental.LinearCosineDecay")
class LinearCosineDecay(LearningRateSchedule):
  """A LearningRateSchedule that uses a linear cosine decay schedule.

  See [Bello et al., ICML2017] Neural Optimizer Search with RL.
  https://arxiv.org/abs/1709.07417

  For the idea of warm starts here controlled by `num_periods`,
  see [Loshchilov & Hutter, ICLR2016] SGDR: Stochastic Gradient Descent
  with Warm Restarts. https://arxiv.org/abs/1608.03983

  Note that linear cosine decay is more aggressive than cosine decay and
  larger initial learning rates can typically be used.

  When training a model, it is often recommended to lower the learning rate as
  the training progresses. This schedule applies a linear cosine decay
  function to an optimizer step, given a provided initial learning rate.
  It requires a `step` value to compute the decayed learning rate. You can
  just pass a TensorFlow variable that you increment at each training step.

  The schedule a 1-arg callable that produces a decayed learning
  rate when passed the current optimizer step. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  It is computed as:

  ```python
  def decayed_learning_rate(step):
    step = min(step, decay_steps)
    linear_decay = (decay_steps - step) / decay_steps
    cosine_decay = 0.5 * (
        1 + cos(pi * 2 * num_periods * step / decay_steps))
    decayed = (alpha + linear_decay) * cosine_decay + beta
    return initial_learning_rate * decayed
  ```

  Example usage:
  ```python
  decay_steps = 1000
  lr_decayed_fn = (
    tf.keras.experimental.LinearCosineDecay(
      initial_learning_rate, decay_steps))
  ```

  You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
  as the learning rate. The learning rate schedule is also serializable and
  deserializable using `tf.keras.optimizers.schedules.serialize` and
  `tf.keras.optimizers.schedules.deserialize`.

  Returns:
    A 1-arg callable learning rate schedule that takes the current optimizer
    step and outputs the decayed learning rate, a scalar `Tensor` of the same
    type as `initial_learning_rate`.
  """

  def __init__(
      self,
      initial_learning_rate,
      decay_steps,
      num_periods=0.5,
      alpha=0.0,
      beta=0.001,
      name=None):
    """Applies linear cosine decay to the learning rate.

    Args:
      initial_learning_rate: A scalar `float32` or `float64` Tensor or a Python
        number. The initial learning rate.
      decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
        Number of steps to decay over.
      num_periods: Number of periods in the cosine part of the decay.
        See computation above.
      alpha: See computation above.
      beta: See computation above.
      name: String.  Optional name of the operation.  Defaults to
        'LinearCosineDecay'.
    """
    super(LinearCosineDecay, self).__init__()

    self.initial_learning_rate = initial_learning_rate
    self.decay_steps = decay_steps
    self.num_periods = num_periods
    self.alpha = alpha
    self.beta = beta
    self.name = name

  def __call__(self, step):
    with ops.name_scope_v2(self.name or "LinearCosineDecay") as name:
      initial_learning_rate = ops.convert_to_tensor_v2(
          self.initial_learning_rate, name="initial_learning_rate")
      dtype = initial_learning_rate.dtype
      decay_steps = math_ops.cast(self.decay_steps, dtype)
      num_periods = math_ops.cast(self.num_periods, dtype)
      alpha = math_ops.cast(self.alpha, dtype)
      beta = math_ops.cast(self.beta, dtype)

      global_step_recomp = math_ops.cast(step, dtype)
      global_step_recomp = math_ops.minimum(global_step_recomp, decay_steps)
      linear_decayed = (decay_steps - global_step_recomp) / decay_steps
      completed_fraction = global_step_recomp / decay_steps
      fraction = 2.0 * num_periods * completed_fraction
      cosine_decayed = 0.5 * (
          1.0 + math_ops.cos(constant_op.constant(math.pi) * fraction))

      linear_cosine_decayed = (alpha + linear_decayed) * cosine_decayed + beta
      return math_ops.multiply(initial_learning_rate, linear_cosine_decayed,
                               name=name)

  def get_config(self):
    return {
        "initial_learning_rate": self.initial_learning_rate,
        "decay_steps": self.decay_steps,
        "num_periods": self.num_periods,
        "alpha": self.alpha,
        "beta": self.beta,
        "name": self.name
    }


@keras_export("keras.experimental.NoisyLinearCosineDecay")
class NoisyLinearCosineDecay(LearningRateSchedule):
  """A LearningRateSchedule that uses a noisy linear cosine decay schedule.

  See [Bello et al., ICML2017] Neural Optimizer Search with RL.
  https://arxiv.org/abs/1709.07417

  For the idea of warm starts here controlled by `num_periods`,
  see [Loshchilov & Hutter, ICLR2016] SGDR: Stochastic Gradient Descent
  with Warm Restarts. https://arxiv.org/abs/1608.03983

  Note that linear cosine decay is more aggressive than cosine decay and
  larger initial learning rates can typically be used.

  When training a model, it is often recommended to lower the learning rate as
  the training progresses. This schedule applies a noisy linear cosine decay
  function to an optimizer step, given a provided initial learning rate.
  It requires a `step` value to compute the decayed learning rate. You can
  just pass a TensorFlow variable that you increment at each training step.

  The schedule a 1-arg callable that produces a decayed learning
  rate when passed the current optimizer step. This can be useful for changing
  the learning rate value across different invocations of optimizer functions.
  It is computed as:

  ```python
  def decayed_learning_rate(step):
    step = min(step, decay_steps)
    linear_decay = (decay_steps - step) / decay_steps)
    cosine_decay = 0.5 * (
        1 + cos(pi * 2 * num_periods * step / decay_steps))
    decayed = (alpha + linear_decay + eps_t) * cosine_decay + beta
    return initial_learning_rate * decayed
  ```
  where eps_t is 0-centered gaussian noise with variance
  initial_variance / (1 + global_step) ** variance_decay

  Example usage:
  ```python
  decay_steps = 1000
  lr_decayed_fn = (
    tf.keras.experimental.NoisyLinearCosineDecay(
      initial_learning_rate, decay_steps))
  ```

  You can pass this schedule directly into a `tf.keras.optimizers.Optimizer`
  as the learning rate. The learning rate schedule is also serializable and
  deserializable using `tf.keras.optimizers.schedules.serialize` and
  `tf.keras.optimizers.schedules.deserialize`.

  Returns:
    A 1-arg callable learning rate schedule that takes the current optimizer
    step and outputs the decayed learning rate, a scalar `Tensor` of the same
    type as `initial_learning_rate`.
  """

  def __init__(
      self,
      initial_learning_rate,
      decay_steps,
      initial_variance=1.0,
      variance_decay=0.55,
      num_periods=0.5,
      alpha=0.0,
      beta=0.001,
      name=None):
    """Applies noisy linear cosine decay to the learning rate.

    Args:
      initial_learning_rate: A scalar `float32` or `float64` Tensor or a Python
        number. The initial learning rate.
      decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number.
        Number of steps to decay over.
      initial_variance: initial variance for the noise. See computation above.
      variance_decay: decay for the noise's variance. See computation above.
      num_periods: Number of periods in the cosine part of the decay.
        See computation above.
      alpha: See computation above.
      beta: See computation above.
      name: String.  Optional name of the operation.  Defaults to
        'NoisyLinearCosineDecay'.
    """
    super(NoisyLinearCosineDecay, self).__init__()

    self.initial_learning_rate = initial_learning_rate
    self.decay_steps = decay_steps
    self.initial_variance = initial_variance
    self.variance_decay = variance_decay
    self.num_periods = num_periods
    self.alpha = alpha
    self.beta = beta
    self.name = name

  def __call__(self, step):
    with ops.name_scope_v2(self.name or "NoisyLinearCosineDecay") as name:
      initial_learning_rate = ops.convert_to_tensor_v2(
          self.initial_learning_rate, name="initial_learning_rate")
      dtype = initial_learning_rate.dtype
      decay_steps = math_ops.cast(self.decay_steps, dtype)
      initial_variance = math_ops.cast(self.initial_variance, dtype)
      variance_decay = math_ops.cast(self.variance_decay, dtype)
      num_periods = math_ops.cast(self.num_periods, dtype)
      alpha = math_ops.cast(self.alpha, dtype)
      beta = math_ops.cast(self.beta, dtype)

      global_step_recomp = math_ops.cast(step, dtype)
      global_step_recomp = math_ops.minimum(global_step_recomp, decay_steps)
      linear_decayed = (decay_steps - global_step_recomp) / decay_steps
      variance = initial_variance / (
          math_ops.pow(1.0 + global_step_recomp, variance_decay))
      std = math_ops.sqrt(variance)
      noisy_linear_decayed = (
          linear_decayed + random_ops.random_normal(
              linear_decayed.shape, stddev=std))

      completed_fraction = global_step_recomp / decay_steps
      fraction = 2.0 * num_periods * completed_fraction
      cosine_decayed = 0.5 * (
          1.0 + math_ops.cos(constant_op.constant(math.pi) * fraction))
      noisy_linear_cosine_decayed = (
          (alpha + noisy_linear_decayed) * cosine_decayed + beta)

      return math_ops.multiply(
          initial_learning_rate, noisy_linear_cosine_decayed, name=name)

  def get_config(self):
    return {
        "initial_learning_rate": self.initial_learning_rate,
        "decay_steps": self.decay_steps,
        "initial_variance": self.initial_variance,
        "variance_decay": self.variance_decay,
        "num_periods": self.num_periods,
        "alpha": self.alpha,
        "beta": self.beta,
        "name": self.name
    }


@keras_export("keras.optimizers.schedules.serialize")
def serialize(learning_rate_schedule):
  return generic_utils.serialize_keras_object(learning_rate_schedule)


@keras_export("keras.optimizers.schedules.deserialize")
def deserialize(config, custom_objects=None):
  return generic_utils.deserialize_keras_object(
      config,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name="decay")
