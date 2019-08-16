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
"""Layer-wise Adaptive Rate Scaling optimizer for large-batch training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops


class LARSOptimizer(optimizer.Optimizer):
  """Layer-wise Adaptive Rate Scaling for large batch training.

  Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
  I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)

  Implements the LARS learning rate scheme presented in the paper above. This
  optimizer is useful when scaling the batch size to up to 32K without
  significant performance degradation. It is recommended to use the optimizer
  in conjunction with:
      - Gradual learning rate warm-up
      - Linear learning rate scaling
      - Poly rule learning rate decay

  Note, LARS scaling is currently only enabled for dense tensors. Sparse tensors
  use the default momentum optimizer.
  """

  def __init__(
      self,
      learning_rate,
      momentum=0.9,
      weight_decay=0.0001,
      # The LARS coefficient is a hyperparameter
      eeta=0.001,
      epsilon=0.0,
      name="LARSOptimizer",
      # Enable skipping variables from LARS scaling.
      # TODO(sameerkm): Enable a direct mechanism to pass a
      # subset of variables to the optimizer.
      skip_list=None,
      use_nesterov=False):
    """Construct a new LARS Optimizer.

    Args:
      learning_rate: A `Tensor` or floating point value. The base learning rate.
      momentum: A floating point value. Momentum hyperparameter.
      weight_decay: A floating point value. Weight decay hyperparameter.
      eeta: LARS coefficient as used in the paper. Dfault set to LARS
        coefficient from the paper. (eeta / weight_decay) determines the highest
        scaling factor in LARS.
      epsilon: Optional epsilon parameter to be set in models that have very
        small gradients. Default set to 0.0.
      name: Optional name prefix for variables and ops created by LARSOptimizer.
      skip_list: List of strings to enable skipping variables from LARS scaling.
        If any of the strings in skip_list is a subset of var.name, variable
        'var' is skipped from LARS scaling. For a typical classification model
        with batch normalization, the skip_list is ['batch_normalization',
        'bias']
      use_nesterov: when set to True, nesterov momentum will be enabled

    Raises:
      ValueError: If a hyperparameter is set to a non-sensical value.
    """
    if momentum < 0.0:
      raise ValueError("momentum should be positive: %s" % momentum)
    if weight_decay < 0.0:
      raise ValueError("weight_decay should be positive: %s" % weight_decay)
    super(LARSOptimizer, self).__init__(use_locking=False, name=name)

    self._learning_rate = learning_rate
    self._momentum = momentum
    self._weight_decay = weight_decay
    self._eeta = eeta
    self._epsilon = epsilon
    self._name = name
    self._skip_list = skip_list
    self._use_nesterov = use_nesterov

  def _create_slots(self, var_list):
    for v in var_list:
      self._zeros_slot(v, "momentum", self._name)

  def compute_lr(self, grad, var):
    scaled_lr = self._learning_rate
    if self._skip_list is None or not any(v in var.name
                                          for v in self._skip_list):
      w_norm = linalg_ops.norm(var, ord=2)
      g_norm = linalg_ops.norm(grad, ord=2)
      trust_ratio = array_ops.where(
          math_ops.greater(w_norm, 0),
          array_ops.where(
              math_ops.greater(g_norm, 0),
              (self._eeta * w_norm /
               (g_norm + self._weight_decay * w_norm + self._epsilon)), 1.0),
          1.0)
      scaled_lr = self._learning_rate * trust_ratio
      # Add the weight regularization gradient
      grad = grad + self._weight_decay * var
    return scaled_lr, grad

  def _apply_dense(self, grad, var):
    scaled_lr, grad = self.compute_lr(grad, var)
    mom = self.get_slot(var, "momentum")
    return training_ops.apply_momentum(
        var,
        mom,
        math_ops.cast(1.0, var.dtype.base_dtype),
        grad * scaled_lr,
        self._momentum,
        use_locking=False,
        use_nesterov=self._use_nesterov)

  def _resource_apply_dense(self, grad, var):
    scaled_lr, grad = self.compute_lr(grad, var)
    mom = self.get_slot(var, "momentum")
    return training_ops.resource_apply_momentum(
        var.handle,
        mom.handle,
        math_ops.cast(1.0, var.dtype.base_dtype),
        grad * scaled_lr,
        self._momentum,
        use_locking=False,
        use_nesterov=self._use_nesterov)

  # Fallback to momentum optimizer for sparse tensors
  def _apply_sparse(self, grad, var):
    mom = self.get_slot(var, "momentum")
    return training_ops.sparse_apply_momentum(
        var,
        mom,
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        grad.values,
        grad.indices,
        math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
        use_locking=self._use_locking,
        use_nesterov=self._use_nesterov).op

  def _resource_apply_sparse(self, grad, var, indices):
    mom = self.get_slot(var, "momentum")
    return training_ops.resource_sparse_apply_momentum(
        var.handle,
        mom.handle,
        math_ops.cast(self._learning_rate_tensor, grad.dtype),
        grad,
        indices,
        math_ops.cast(self._momentum_tensor, grad.dtype),
        use_locking=self._use_locking,
        use_nesterov=self._use_nesterov)

  def _prepare(self):
    learning_rate = self._learning_rate
    if callable(learning_rate):
      learning_rate = learning_rate()
    self._learning_rate_tensor = ops.convert_to_tensor(
        learning_rate, name="learning_rate")
    momentum = self._momentum
    if callable(momentum):
      momentum = momentum()
    self._momentum_tensor = ops.convert_to_tensor(momentum, name="momentum")
