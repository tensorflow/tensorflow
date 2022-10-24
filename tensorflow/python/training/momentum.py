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
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["train.MomentumOptimizer"])
class MomentumOptimizer(optimizer.Optimizer):
  """Optimizer that implements the Momentum algorithm.

  Computes (if `use_nesterov = False`):

  ```
  accumulation = momentum * accumulation + gradient
  variable -= learning_rate * accumulation
  ```

  Note that in the dense version of this algorithm, `accumulation` is updated
  and applied regardless of a gradient's value, whereas the sparse version (when
  the gradient is an `IndexedSlices`, typically because of `tf.gather` or an
  embedding) only updates variable slices and corresponding `accumulation` terms
  when that part of the variable was used in the forward pass.

  @compatibility(TF2)
  tf.compat.v1.train.MomentumOptimizer is compatible with eager mode and
  `tf.function`.
  When eager execution is enabled, `learning_rate`,`momentum`, can each be a
  callable that takes no arguments and returns the actual value to use. This
  can be useful for changing these values across different invocations of
  optimizer functions.

  To switch to native TF2 style, please directly use
  [`tf.keras.optimizers.SGD`]
  (https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD)
  with the `momentum` argument.

  #### Structural mapping to native TF2

  Before:

  ```python
  optimizer = tf.compat.v1.train.MomentumOptimizer(
    learning_rate=learning_rate,
    momentum=momentum,
    use_nesterov=use_nesterov)
  ```

  After:

  ```python
  optimizer = tf.keras.optimizers.SGD(
    learning_rate=learning_rate,
    momentum=momentum,
    nesterov=use_nesterov)
  ```

  #### How to map arguments
  | TF1 Arg Name       | TF2 Arg Name   | Note                             |
  | ------------------ | -------------  | -------------------------------  |
  | `learning_rate`    | `learning_rate`| Be careful of setting           |
  : : : learning_rate tensor value computed from the global step.          :
  : : : In TF1 this was usually meant to imply a dynamic learning rate and :
  : : : would recompute in each step. In TF2 (eager + function) it will    :
  : : : treat it as a scalar value that only gets computed once instead of :
  : : : a symbolic placeholder to be computed each time.                   :
  | `momentum`         | `momentum`     | -                                |
  | `use_locking`      | -              | Not applicable in TF2.           |
  | `use_nesterov`     | `nesterov`     | -                                |

  #### Before & after usage example
  Before:

  ```python
  x = tf.Variable([1,2,3], dtype=tf.float32)
  grad = tf.constant([0.1, 0.2, 0.3])
  optimizer = tf.compat.v1.train.MomentumOptimizer(
    learning_rate=0.001,
    momentum=0.9,
    use_nesterov=False)
  optimizer.apply_gradients(zip([grad], [x]))
  ```

  After:

  ```python
  x = tf.Variable([1,2,3], dtype=tf.float32)
  grad = tf.constant([0.1, 0.2, 0.3])
  optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.001,
    momentum=0.9,
    nesterov=False)
  optimizer.apply_gradients(zip([grad], [x]))
  ```

  @end_compatibility

  """

  def __init__(self, learning_rate, momentum,
               use_locking=False, name="Momentum", use_nesterov=False):
    """Construct a new Momentum optimizer.

    Args:
      learning_rate: A `Tensor` or a floating point value.  The learning rate.
      momentum: A `Tensor` or a floating point value.  The momentum.
      use_locking: If `True` use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "Momentum".
      use_nesterov: If `True` use Nesterov Momentum.
        See (Sutskever et al., 2013).
        This implementation always computes gradients at the value of the
        variable(s) passed to the optimizer. Using Nesterov Momentum makes the
        variable(s) track the values called `theta_t + mu*v_t` in the paper.
        This implementation is an approximation of the original formula, valid
        for high values of momentum. It will compute the "adjusted gradient"
        in NAG by assuming that the new gradient will be estimated by the
        current average gradient plus the product of momentum and the change
        in the average gradient.

    References:
      On the importance of initialization and momentum in deep learning:
        [Sutskever et al., 2013]
        (http://proceedings.mlr.press/v28/sutskever13.html)
        ([pdf](http://proceedings.mlr.press/v28/sutskever13.pdf))


    """
    super(MomentumOptimizer, self).__init__(use_locking, name)
    self._learning_rate = learning_rate
    self._momentum = momentum
    self._use_nesterov = use_nesterov

  def _create_slots(self, var_list):
    for v in var_list:
      self._zeros_slot(v, "momentum", self._name)

  def _prepare(self):
    learning_rate = self._learning_rate
    if callable(learning_rate):
      learning_rate = learning_rate()
    self._learning_rate_tensor = ops.convert_to_tensor(learning_rate,
                                                       name="learning_rate")
    momentum = self._momentum
    if callable(momentum):
      momentum = momentum()
    self._momentum_tensor = ops.convert_to_tensor(momentum, name="momentum")

  def _apply_dense(self, grad, var):
    mom = self.get_slot(var, "momentum")
    return training_ops.apply_momentum(
        var, mom,
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        grad,
        math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
        use_locking=self._use_locking,
        use_nesterov=self._use_nesterov).op

  def _resource_apply_dense(self, grad, var):
    mom = self.get_slot(var, "momentum")
    return training_ops.resource_apply_momentum(
        var.handle, mom.handle,
        math_ops.cast(self._learning_rate_tensor, grad.dtype.base_dtype),
        grad,
        math_ops.cast(self._momentum_tensor, grad.dtype.base_dtype),
        use_locking=self._use_locking,
        use_nesterov=self._use_nesterov)

  def _apply_sparse(self, grad, var):
    mom = self.get_slot(var, "momentum")
    return training_ops.sparse_apply_momentum(
        var, mom,
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        grad.values, grad.indices,
        math_ops.cast(self._momentum_tensor, var.dtype.base_dtype),
        use_locking=self._use_locking,
        use_nesterov=self._use_nesterov).op

  def _resource_apply_sparse(self, grad, var, indices):
    mom = self.get_slot(var, "momentum")
    return training_ops.resource_sparse_apply_momentum(
        var.handle, mom.handle,
        math_ops.cast(self._learning_rate_tensor, grad.dtype),
        grad, indices,
        math_ops.cast(self._momentum_tensor, grad.dtype),
        use_locking=self._use_locking,
        use_nesterov=self._use_nesterov)
