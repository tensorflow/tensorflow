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

"""One-line documentation for psgld module.

PSGLD algorithm [Preconditioned Stochastic Gradient Langevin Dynamics for Deep Neural Networks]

A detailed description of PSGLD.

- SGLD is derived from Stochastic Gradient Langevin Dynamics (SGLD) and SGD
- require only the gradient on mini-batches of data
- incorperating adaptive preconditioners from RMSProp

mean_square = decay * mean_square + (1-decay) * gradient ** 2
PCDer = epsilon + sqrt(mean_square)
Delta = momentum * Delta + learning_rate * gradient / PCDer +
        normal(0, sqrt(2 * learning_rate / PCDer))

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops



class PSGLDOptimizer(optimizer.Optimizer):
    """Optimizer that implements the PSGLD algorithm.

  See the [paper]
  (http://arxiv.org/pdf/1512.07666v1.pdf).

  @@__init__
  """

    def __init__(self,
                 learning_rate,
                 decay=0.9,
                 momentum=0.0,
                 epsilon=1e-10,
                 use_locking=False,
                 name="PSGLD"):
        """Construct a new PSGLD optimizer.

    Note that in dense implement of this algorithm, ms and mom will
    update even if grad is zero, but in sparse implement, ms and mom
    will not update in iterations grad is zero.

    Args:
      learning_rate: A Tensor or a floating point value.  The learning rate.
      decay: Discounting factor for the history/coming gradient
      momentum: A scalar tensor.
      epsilon: Small value to avoid zero denominator.
      use_locking: If True use locks for update operation.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "PSGLD".
    """
        super(PSGLDOptimizer, self).__init__(use_locking, name)
        self._learning_rate = learning_rate
        self._decay = decay
        self._momentum = momentum
        self._epsilon = epsilon

        # Tensors for learning rate and momentum.  Created in _prepare.
        self._learning_rate_tensor = None
        self._decay_tensor = None
        self._momentum_tensor = None
        self._epsilon_tensor = None

    def _create_slots(self, var_list):
        for v in var_list:
            val = constant_op.constant(1.0, dtype=v.dtype, shape=v.get_shape())
            self._get_or_make_slot(v, val, "ms", self._name)  # ms: mean_square           
            self._zeros_slot(v, "momentum", self._name)
            self._zeros_slot(v, "rnd", self._name) # slot for random tensor           

    def _prepare(self):
        self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate, name="learning_rate")
        self._decay_tensor = ops.convert_to_tensor(self._decay, name="decay")
        self._momentum_tensor = ops.convert_to_tensor(self._momentum, name="momentum")
        self._epsilon_tensor = ops.convert_to_tensor(self._epsilon, name="epsilon")

    def _apply_dense(self, grad, var):
        ms = self.get_slot(var, "ms")
        mom = self.get_slot(var, "momentum")		
        # generate randoms following normal distribution
        rnd = self.get_slot(var, "rnd")
        rnd = variables.Variable(random_ops.random_normal(rnd.get_shape(), 1.0), name="rnd")
        return training_ops.apply_psgld(
            var,
            ms,
            mom,
            rnd,
            math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),  # Cast a tensor to a new type
            math_ops.cast(self._decay_tensor, var.dtype.base_dtype),
            math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
            math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype),
            grad, use_locking=self._use_locking).op

    # sparse implementation
    def _apply_sparse(self, grad, var):
        ms = self.get_slot(var, "ms")
        mom = self.get_slot(var, "momentum")		
        # generate randoms following normal distribution
        rnd = self.get_slot(var, "rnd")
        rnd = variables.Variable(random_ops.random_normal(rnd.get_shape(), 1.0), name="rnd")
        return training_ops.sparse_apply_psgld(
            var,
            ms,
            mom,
            rnd,
            math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
            math_ops.cast(self._decay_tensor, var.dtype.base_dtype),
            math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
            math_ops.cast(self._epsilon_tensor, var.dtype.base_dtype),
            grad.values,
            grad.indices,
            use_locking=self._use_locking)
