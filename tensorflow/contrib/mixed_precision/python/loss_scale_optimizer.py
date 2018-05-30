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
"""Loss scaling optimizer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer


class LossScaleOptimizer(optimizer.Optimizer):
  """An optimizer that applies loss scaling in backprop.

  This class is useful for mixed precision training on GPUs (or other potential
  accelerators), which is an approach to improve compute throughput without loss
  of model quality.

  The commmon configuration of mixed precision models is the following:
  * variables are kept in high precision (e.g. float32).
  * computations are done in lower precision (e.g. float16). variables are
    casted to lower precision before they're used.
  * (in training), final gradients are casted back to variable precision and get
    applied.

  Because computations happen in lower precision, gradients in the backprop pass
  might underflow in the smaller dynamic range, causing a model to converge at a
  suboptimal level. This optimizer multiplies the loss by a factor before
  backprop starts to prevent underflow. Before gradients are applied, they are
  casted to higher precision and down-scaled by the same factor, so
  mathematically the variable updates are no different from regular
  same-precision training.

  See [Nvidia's manual on mixed precision training](
  https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html)
  for more details.

  To use loss scale optimizer, one only needs choose a loss scale strategy and
  wrap a regular optimizer. See examples below.

  ```
  loss = loss_fn()
  opt = tf.AdamOptimizer(learning_rate=...)

  # Choose a loss scale manager which decides how to pick the right loss scale
  # throughout the training process.
  loss_scale_manger = tf.contrib.mixed_precision.FixedLossScaleManager(5000)

  # Wraps the original optimizer in a LossScaleOptimizer.
  loss_scale_optimizer = LossScaleOptimizer(opt, loss_scale_manager)

  # Call minimize() on the loss scale optimizer.
  train_op = loss_scale_optimizer.minimize(loss)
  ```

  If gradients clipping is applied, one can call
  `optimizer.compute_gradients()` and `optimizer.apply_gradients()`
  seperately.

  Notice the following way of using LossScaleOptimizer is not intended. Always
  use `loss_scale_optimizer.compute_gradients()` to compute gradients instead of
  `tf.gradients()` if doing mixed precision training.

  ```
  # The following is a wrong way to use LossScaleOptimizer along with
  # tf.gradients().

  # Always use loss_scale_optimizer.compute_gradients() to compute grads, or
  # loss scale is not correctly applied.
  grads = tf.gradients(loss, ...)

  # Do some custom grad clipping.
  grads = clip_grads(grads, ...)

  loss_scale_optimizer.apply(grads_and_vars)
  ```
  """

  def __init__(self, opt, loss_scale_manager):
    """Construct a loss scaling optimizer.

    Args:
      opt: The actual optimizer that will be used to compute and apply the
        gradients. Must be an implementation of the @{tf.train.Optimizer}
        interface.
      loss_scale_manager: A LossScaleManager object.
    """
    self._opt = opt
    self._loss_scale_manager = loss_scale_manager

  def compute_gradients(self,
                        loss,
                        var_list=None,
                        gate_gradients=optimizer.Optimizer.GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
    """Compute gradients. See base class @{tf.train.Optimizer}."""
    loss_scale = self._loss_scale_manager.get_loss_scale()
    if context.executing_eagerly():

      def scaled_loss():
        loss_val = loss()
        return loss_val * math_ops.cast(loss_scale, loss_val.dtype.base_dtype)
    else:
      if callable(loss):
        loss_val = loss()
      else:
        loss_val = loss
      scaled_loss = loss_val * math_ops.cast(loss_scale,
                                             loss_val.dtype.base_dtype)
    grads_and_vars = self._opt.compute_gradients(
        scaled_loss,
        var_list=var_list,
        gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops,
        grad_loss=grad_loss)
    return self._down_scale(grads_and_vars, loss_scale)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Apply gradients. See base class @{tf.train.Optimizer}."""
    grads = [g for (g, _) in grads_and_vars]

    is_finite_grad = []
    for g in grads:
      is_finite_grad.append(math_ops.reduce_all(gen_math_ops.is_finite(g)))
    is_overall_finite = math_ops.reduce_all(is_finite_grad)

    # Only update gradients when all grads are finite.
    def true_apply_gradients_fn():
      return self._opt.apply_gradients(grads_and_vars, global_step, name)

    update_vars = control_flow_ops.cond(
        is_overall_finite, true_apply_gradients_fn, gen_control_flow_ops.no_op)
    # Potentially adjust gradient scale in case of finite gradients.
    return control_flow_ops.group(
        update_vars,
        self._loss_scale_manager.update_loss_scale(is_overall_finite))

  def _down_scale(self, grads_vars, loss_scale):
    # Down scale grads by the loss_scale.
    gv = []
    inv_loss_scale = gen_math_ops.reciprocal(loss_scale)
    for g, v in grads_vars:
      if g is not None:
        gv.append((g * math_ops.cast(inv_loss_scale, g.dtype.base_dtype), v))
      else:
        gv.append((g, v))
    return gv
