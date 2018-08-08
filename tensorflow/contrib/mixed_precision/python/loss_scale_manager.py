# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""LossScaleManager classes for mixed precision training."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope


@six.add_metaclass(abc.ABCMeta)
class LossScaleManager(object):
  """Abstract loss scale manager class.

  Loss scale managers with a different strategy should subclass this class.
  Loss scaling is a process that:

  1) Applies a multiplier on the loss before computing gradients, and
  2) Applies the reciprocal of the multiplier on the gradients before they are
     applied on variables.

  This class is used together with
  @{tf.contrib.mixed_precision.LossScaleOptimizer} for mixed precision training
  (float32 variables and float16 ops) on Nvidia GPUs in order to achieve the
  same model quality as single precision training, with the benefits of
  potential higher throughput.

  See @{tf.contrib.mixed_precision.LossScaleOptimizer} for more details.
  """

  @abc.abstractmethod
  def get_loss_scale(self):
    """Returns the loss scale as a scalar `float32` tensor."""
    pass

  @abc.abstractmethod
  def update_loss_scale(self, finite_grads):
    """Updates loss scale based on if gradients are finite in current step.

    Args:
      finite_grads: bool scalar tensor indicating if all gradients are
        finite (i.e., not inf or nan).

    Returns:
      An op, when executed updates the loss scale. If eager execution is
      enabled, does not return anything.
    """
    del finite_grads
    return


class FixedLossScaleManager(LossScaleManager):
  """Loss scale manager with a fixed loss scale.

  The loss scale is not updated for the lifetime of the class.
  """

  def __init__(self, loss_scale):
    """Creates the fixed loss scale manager.

    Args:
      loss_scale: A Python float. Its ideal value varies depending on models to
        run. Choosing a too small loss_scale might affect model quality; a too
        big loss_scale might cause inf or nan. There is no single right
        loss_scale to apply. There is no harm choosing a relatively big number
        as long as no nan or inf is encountered in training.

    Raises:
      ValueError: If loss_scale is less than 1.
    """
    if loss_scale < 1:
      raise ValueError("loss scale must be at least 1.")
    self._loss_scale = ops.convert_to_tensor(loss_scale, dtype=dtypes.float32)

  def get_loss_scale(self):
    return self._loss_scale

  def update_loss_scale(self, finite_grads):
    del finite_grads
    return gen_control_flow_ops.no_op()


class ExponentialUpdateLossScaleManager(LossScaleManager):
  """Loss scale manager uses an exponential update strategy.

  In general, the strategy increases loss scale by a greater-than-one factor
  after encountering a consecutive series of steps with finite gradients;
  Similarly, it decreases the loss scale by a factor when the accumulated number
  of steps with non-finite (nan or inf) gradients are met. An update is not
  applied if its result is less than 1 or overflows the float32 dynamic range.

  The number of finite and non-finite steps are cleared every time the loss
  scale is changed. The condition to decrease the loss scale is looser than to
  increase it since the former does not require the steps to be consecutive.
  """

  def __init__(self,
               init_loss_scale,
               incr_every_n_steps,
               decr_every_n_nan_or_inf=2,
               incr_ratio=2,
               decr_ratio=0.8):
    """Constructor of exponential-update loss scale manager.

    Args:
      init_loss_scale: A Python float.  The loss scale to use at the beginning.
      incr_every_n_steps: Increases loss scale every n consecutive steps with
        finite gradients.
      decr_every_n_nan_or_inf: Decreases loss scale every n accumulated steps
        with nan or inf gradients.
      incr_ratio: The multiplier to use when increasing the loss scale.
      decr_ratio: The less-than-one-multiplier to use when decreasing the loss
        scale.
    """
    self._incr_every_n_steps = incr_every_n_steps
    self._decr_every_n_nan_or_inf = decr_every_n_nan_or_inf
    self._incr_ratio = incr_ratio
    self._decr_ratio = decr_ratio
    self._loss_scale = variable_scope.variable(
        name="loss_scale",
        initial_value=ops.convert_to_tensor(init_loss_scale, dtypes.float32),
        dtype=dtypes.float32,
        trainable=False)
    self._num_good_steps = variable_scope.variable(
        name="good_steps", initial_value=0, dtype=dtypes.int32, trainable=False)
    self._num_bad_steps = variable_scope.variable(
        name="bad_steps", initial_value=0, dtype=dtypes.int32, trainable=False)

  def _reset_stats(self):
    return control_flow_ops.group(
        state_ops.assign(self._num_good_steps, 0),
        state_ops.assign(self._num_bad_steps, 0))

  def get_loss_scale(self):
    """Returns the loss scale."""
    return self._loss_scale

  def update_loss_scale(self, finite_grads):
    """Updates loss scale based on if gradients are finite in current step."""

    def update_if_finite_grads():
      """Branch function when grads are all finite."""

      def incr_loss_scale():
        new_loss_scale = control_flow_ops.cond(
            gen_math_ops.is_finite(self._loss_scale * self._incr_ratio),
            lambda: self._loss_scale * self._incr_ratio,
            lambda: self._loss_scale)
        update_op = state_ops.assign(self._loss_scale, new_loss_scale)
        # When loss_scale is updated, both good and bad steps are reset.
        return control_flow_ops.group(update_op, self._reset_stats())

      return control_flow_ops.cond(
          self._num_good_steps + 1 >= self._incr_every_n_steps,
          incr_loss_scale,
          lambda: state_ops.assign_add(self._num_good_steps, 1).op)

    def update_if_not_finite_grads():
      """Branch function when any grad is not finite."""

      def decr_loss_scale():
        update_op = state_ops.assign(
            self._loss_scale,
            gen_math_ops.maximum(1., self._loss_scale * self._decr_ratio))
        # When loss_scale is updated, both good and bad steps are reset.
        return control_flow_ops.group(update_op, self._reset_stats())

      def just_update_steps():
        # When bad_steps is incremented, good_step is reset.
        return control_flow_ops.group(
            state_ops.assign_add(self._num_bad_steps, 1),
            state_ops.assign(self._num_good_steps, 0))

      return control_flow_ops.cond(
          self._num_bad_steps + 1 >= self._decr_every_n_nan_or_inf,
          decr_loss_scale, just_update_steps)

    return control_flow_ops.cond(finite_grads, update_if_finite_grads,
                                 update_if_not_finite_grads)
