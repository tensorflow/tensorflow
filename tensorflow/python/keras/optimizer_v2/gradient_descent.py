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
"""Momentum for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.optimizers.SGD")
class SGD(optimizer_v2.OptimizerV2):
  """Stochastic gradient descent and momentum optimizer.

  Computes:
  ```
  theta(t+1) = theta(t) - learning_rate * gradient
  gradient is evaluated at theta(t).
  ```

  or Computes (if `nesterov = False`):
  ```
  v(t+1) = momentum * v(t) - learning_rate * gradient
  theta(t+1) = theta(t) + v(t+1)
  if `nesterov` is False, gradient is evaluated at theta(t).
  if `nesterov` is True, gradient is evaluated at theta(t) + momentum * v(t),
    and the variables always store theta + m v instead of theta
  ```

  Some of the args below are hyperparameters, where a hyperparameter is
  defined as a scalar Tensor, a regular Python value, or a callable (which
  will be evaluated when `apply_gradients` is called) returning a scalar
  Tensor or a Python value.

  @compatibility(eager)
  When eager execution is enabled, learning_rate can be a callable that takes
  no arguments and returns the actual value to use. This can be useful for
  changing these values across different invocations of optimizer functions.
  @end_compatibility

  # References
      nesterov = True, See [Sutskever et al., 2013](
        http://jmlr.org/proceedings/papers/v28/sutskever13.pdf).
  """

  def __init__(self,
               learning_rate=0.01,
               momentum=0.0,
               nesterov=False,
               name="SGD",
               **kwargs):
    """Construct a new Stochastic Gradient Descent or Momentum optimizer.

    Arguments:
      learning_rate: float hyperparameter >= 0. Learning rate.
      momentum: float hyperparameter >= 0 that accelerates SGD in the relevant
        direction and dampens oscillations.
      nesterov: boolean. Whether to apply Nesterov momentum.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to 'SGD'.
      **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
        `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
        gradients by value, `decay` is included for backward compatibility to
        allow time inverse decay of learning rate. `lr` is included for backward
        compatibility, recommended to use `learning_rate` instead.
    """
    super(SGD, self).__init__(name, **kwargs)
    self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
    self._set_hyper("decay", self._initial_decay)

    self._momentum = False
    if isinstance(momentum, ops.Tensor) or callable(momentum) or momentum > 0:
      self._momentum = True
    if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
      raise ValueError("`momentum` must be between [0, 1].")
    self._set_hyper("momentum", momentum)

    self.nesterov = nesterov

  def _create_slots(self, var_list):
    if self._momentum:
      for var in var_list:
        self.add_slot(var, "momentum")

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(SGD, self)._prepare_local(var_device, var_dtype, apply_state)
    apply_state[(var_device, var_dtype)]["momentum"] = array_ops.identity(
        self._get_hyper("momentum", var_dtype))

  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    if self._momentum:
      momentum_var = self.get_slot(var, "momentum")
      return training_ops.resource_apply_keras_momentum(
          var.handle,
          momentum_var.handle,
          coefficients["lr_t"],
          grad,
          coefficients["momentum"],
          use_locking=self._use_locking,
          use_nesterov=self.nesterov)
    else:
      return training_ops.resource_apply_gradient_descent(
          var.handle, coefficients["lr_t"], grad, use_locking=self._use_locking)

  def _resource_apply_sparse_duplicate_indices(self, grad, var, indices,
                                               **kwargs):
    if self._momentum:
      return super(SGD, self)._resource_apply_sparse_duplicate_indices(
          grad, var, indices, **kwargs)
    else:
      var_device, var_dtype = var.device, var.dtype.base_dtype
      coefficients = (kwargs.get("apply_state", {}).get((var_device, var_dtype))
                      or self._fallback_apply_state(var_device, var_dtype))

      return resource_variable_ops.resource_scatter_add(
          var.handle, indices, -grad * coefficients["lr_t"])

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    # This method is only needed for momentum optimization.
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))

    momentum_var = self.get_slot(var, "momentum")
    return training_ops.resource_sparse_apply_keras_momentum(
        var.handle,
        momentum_var.handle,
        coefficients["lr_t"],
        grad,
        indices,
        coefficients["momentum"],
        use_locking=self._use_locking,
        use_nesterov=self.nesterov)

  def get_config(self):
    config = super(SGD, self).get_config()
    config.update({
        "learning_rate": self._serialize_hyperparameter("learning_rate"),
        "decay": self._serialize_hyperparameter("decay"),
        "momentum": self._serialize_hyperparameter("momentum"),
        "nesterov": self.nesterov,
    })
    return config
