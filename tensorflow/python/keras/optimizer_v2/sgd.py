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

"""Momentum for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import training_ops


class SGD(optimizer_v2.OptimizerV2):
  """Stochastic gradient descent optimizer.

  Includes support for momentum and Nesterov momentum.

  Computes (if `nesterov = False`):

  ```
  accumulation = momentum * accumulation + gradient
  variable -= learning_rate * accumulation
  ```

  Some of the args below are hyperparameters, where a hyperparameter is
  defined as a scalar Tensor, a regular Python value, or a callable (which
  will be evaluated when `apply_gradients` is called) returning a scalar
  Tensor or a Python value.

  Note that in the dense version of this algorithm, `accumulation` is updated
  and applied regardless of a gradient's value, whereas the sparse version (when
  the gradient is an `IndexedSlices`, typically because of `tf.gather` or an
  embedding) only updates variable slices and corresponding `accumulation` terms
  when that part of the variable was used in the forward pass.

  @compatibility(eager)
  When eager execution is enabled, learning_rate and momentum can each be a
  callable that takes no arguments and returns the actual value to use. This
  can be useful for changing these values across different invocations of
  optimizer functions.
  @end_compatibility

  Arguments:
      learning_rate: float hyperparameter >= 0. Learning rate.
      momentum: float hyperparameter >= 0 or None. Parameter that accelerates
        SGD in the relevant direction and dampens oscillations.
      nesterov: boolean. Whether to apply Nesterov momentum. See [Sutskever et
        al., 2013](http://jmlr.org/proceedings/papers/v28/sutskever13.pdf). This
          implementation always computes gradients at the value of the
          variable(s) passed to the optimizer. Using Nesterov Momentum makes the
          variable(s) track the values called `theta_t + mu*v_t` in the paper.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to 'SGD'.
  """

  def __init__(self,
               learning_rate=0.001,
               momentum=None,
               nesterov=False,
               name="SGD"):
    super(SGD, self).__init__(name)
    self._set_hyper("learning_rate", learning_rate)
    # Only create momentum variables and use momentum ops if needed.
    if momentum is not None:
      self._set_hyper("momentum", momentum)
      self._use_nesterov = nesterov
      self._use_momentum = True
    else:
      self._use_momentum = False

  def _create_vars(self, var_list, state):
    if self._use_momentum:
      for v in var_list:
        state.zeros_slot(v, "momentum")

  def _apply_dense(self, grad, var, state):
    if self._use_momentum:
      mom = state.get_slot(var, "momentum")
      return training_ops.apply_momentum(
          var,
          mom,
          state.get_hyper("learning_rate", var.dtype.base_dtype),
          grad,
          state.get_hyper("momentum", var.dtype.base_dtype),
          use_locking=self._use_locking,
          use_nesterov=self._use_nesterov).op
    else:
      return training_ops.apply_gradient_descent(
          var,
          state.get_hyper("learning_rate", var.dtype.base_dtype),
          grad,
          use_locking=self._use_locking).op

  def _resource_apply_dense(self, grad, var, state):
    if self._use_momentum:
      mom = state.get_slot(var, "momentum")
      return training_ops.resource_apply_momentum(
          var.handle,
          mom.handle,
          state.get_hyper("learning_rate", var.dtype.base_dtype),
          grad,
          state.get_hyper("momentum", var.dtype.base_dtype),
          use_locking=self._use_locking,
          use_nesterov=self._use_nesterov)
    else:
      lr = state.get_hyper("learning_rate", grad.dtype.base_dtype)
      return training_ops.resource_apply_gradient_descent(
          var.handle, lr, grad, use_locking=self._use_locking)

  def _apply_sparse(self, grad, var, state):
    if self._use_momentum:
      mom = state.get_slot(var, "momentum")
      return training_ops.sparse_apply_momentum(
          var,
          mom,
          state.get_hyper("learning_rate", var.dtype.base_dtype),
          grad.values,
          grad.indices,
          state.get_hyper("momentum", var.dtype.base_dtype),
          use_locking=self._use_locking,
          use_nesterov=self._use_nesterov).op
    else:
      return super(SGD, self)._apply_sparse(grad, var, state)

  def _resource_apply_sparse(self, grad, var, indices, state):
    if self._use_momentum:
      mom = state.get_slot(var, "momentum")
      return training_ops.resource_sparse_apply_momentum(
          var.handle,
          mom.handle,
          state.get_hyper("learning_rate", var.dtype.base_dtype),
          grad,
          indices,
          state.get_hyper("momentum", var.dtype.base_dtype),
          use_locking=self._use_locking,
          use_nesterov=self._use_nesterov)
    else:
      return super(SGD, self)._resource_apply_sparse(grad, var, indices, state)

  def _resource_apply_sparse_duplicate_indices(self, grad, var, indices, state):
    if self._use_momentum:
      return super(SGD, self)._resource_apply_sparse_duplicate_indices(
          grad, var, indices, state)
    else:
      lr = state.get_hyper("learning_rate", grad.dtype.base_dtype)
      return resource_variable_ops.resource_scatter_add(var.handle, indices,
                                                        -grad * lr)

  def _apply_sparse_duplicate_indices(self, grad, var, state):
    if self._use_momentum:
      return super(SGD, self)._apply_sparse_duplicate_indices(grad, var, state)
    else:
      delta = ops.IndexedSlices(
          grad.values * state.get_hyper("learning_rate", var.dtype.base_dtype),
          grad.indices, grad.dense_shape)
      return var.scatter_sub(delta, use_locking=self._use_locking)

  def get_config(self):
    config = super(SGD, self).get_config()
    # Control whether momentum variables are created.
    if not self._use_momentum:
      momentum = None
    else:
      momentum = self._serializer_hyperparameter("momentum")
    config.update({
        "learning_rate": self._serialize_hyperparameter("learning_rate"),
        "momentum": momentum,
        "nesterov": self._use_nesterov
    })
    return config
