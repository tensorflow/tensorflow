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

"""Adagrad for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["train.AdagradOptimizer"])
class AdagradOptimizer(optimizer.Optimizer):
  """Optimizer that implements the Adagrad algorithm.

  References:
    Adaptive Subgradient Methods for Online Learning and Stochastic Optimization
      :[Duchi et al., 2011](http://jmlr.org/papers/v12/duchi11a.html)
      ([pdf](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf))

  @compatibility(TF2)
  tf.compat.v1.train.AdagradOptimizer is compatible with eager mode and
  `tf.function`.
  When eager execution is enabled, `learning_rate`,
  `initial_accumulator_value`, and `epsilon` can each be a callable that
  takes no arguments and returns the actual value to use. This can be useful
  for changing these values across different invocations of optimizer
  functions.

  To switch to native TF2 style, use [`tf.keras.optimizers.Adagrad`]
  (https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adagrad)
  instead. Please notice that due to the implementation differences,
  `tf.keras.optimizers.Adagrad` and
  `tf.compat.v1.train.AdagradOptimizer` may have slight differences in
  floating point numerics even though the formula used for the variable
  updates still matches.

  #### Structural mapping to native TF2

  Before:

  ```python
  optimizer = tf.compat.v1.train.AdagradOptimizer(
    learning_rate=learning_rate,
    initial_accumulator_value=initial_accumulator_value)
  ```

  After:

  ```python
  optimizer = tf.keras.optimizers.Adagrad(
    learning_rate=learning_rate,
    initial_accumulator_value=initial_accumulator_value,
    epsilon=1e-07)
  ```

  #### How to map arguments
  | TF1 Arg Name       | TF2 Arg Name   | Note                             |
  | ------------------ | -------------  | -------------------------------  |
  | `learning_rate`    | `learning_rate` | Be careful of setting           |
  : : : learning_rate tensor value computed from the global step.          :
  : : : In TF1 this was usually meant to imply a dynamic learning rate and :
  : : : would recompute in each step. In TF2 (eager + function) it will    :
  : : : treat it as a scalar value that only gets computed once instead of :
  : : : a symbolic placeholder to be computed each time.                   :
  | `initial_accumulator_value` | `initial_accumulator_value` | The        |
  : : : argument can be value of zero in TF2, which is not accepted in TF1.|
  | - | `epsilon`      | `epsilon` is become configurable in TF2. The      |
  : : : defualt value is changed from 1e-8 to 1e-7                         :
  | `use_locking`      | -             | Not applicable in TF2.            |

  #### Before & after usage example
  Before:

  ```python
  x = tf.Variable([1,2,3], dtype=tf.float32)
  grad = tf.constant([0.1, 0.2, 0.3])
  optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=0.001)
  optimizer.apply_gradients(zip([grad], [x]))
  ```

  After:

  ```python
  x = tf.Variable([1,2,3], dtype=tf.float32)
  grad = tf.constant([0.1, 0.2, 0.3])
  optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.001)
  optimizer.apply_gradients(zip([grad], [x]))
  ```

  @end_compatibility
  """

  def __init__(self, learning_rate, initial_accumulator_value=0.1,
               use_locking=False, name="Adagrad"):
    """Construct a new Adagrad optimizer.

    Args:
      learning_rate: A `Tensor` or a floating point value.  The learning rate.
      initial_accumulator_value: A floating point value.
        Starting value for the accumulators, must be positive.
      use_locking: If `True` use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "Adagrad".

    Raises:
      ValueError: If the `initial_accumulator_value` is invalid.

    """
    if initial_accumulator_value <= 0.0:
      raise ValueError("initial_accumulator_value must be positive: %s" %
                       initial_accumulator_value)
    super(AdagradOptimizer, self).__init__(use_locking, name)
    self._learning_rate = learning_rate
    self._initial_accumulator_value = initial_accumulator_value
    # Created in Initialize.
    self._learning_rate_tensor = None

  def _create_slots(self, var_list):
    for v in var_list:
      dtype = v.dtype.base_dtype
      if v.get_shape().is_fully_defined():
        init = init_ops.constant_initializer(self._initial_accumulator_value,
                                             dtype=dtype)
      else:
        init = self._init_constant_op(v, dtype)
      self._get_or_make_slot_with_initializer(v, init, v.get_shape(), dtype,
                                              "accumulator", self._name)

  def _init_constant_op(self, v, dtype):
    def init():
      # Use a Tensor instead of initializer if variable does not have
      # static shape.
      init_constant = gen_array_ops.fill(array_ops.shape(v),
                                         self._initial_accumulator_value)
      return math_ops.cast(init_constant, dtype)
    return init

  def _prepare(self):
    learning_rate = self._call_if_callable(self._learning_rate)
    self._learning_rate_tensor = ops.convert_to_tensor(
        learning_rate, name="learning_rate")

  def _apply_dense(self, grad, var):
    acc = self.get_slot(var, "accumulator")
    return training_ops.apply_adagrad(
        var,
        acc,
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        grad,
        use_locking=self._use_locking)

  def _resource_apply_dense(self, grad, var):
    acc = self.get_slot(var, "accumulator")
    return training_ops.resource_apply_adagrad(
        var.handle,
        acc.handle,
        math_ops.cast(self._learning_rate_tensor, grad.dtype.base_dtype),
        grad,
        use_locking=self._use_locking)

  def _apply_sparse(self, grad, var):
    acc = self.get_slot(var, "accumulator")
    return training_ops.sparse_apply_adagrad(
        var,
        acc,
        math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype),
        grad.values,
        grad.indices,
        use_locking=self._use_locking)

  def _resource_apply_sparse(self, grad, var, indices):
    acc = self.get_slot(var, "accumulator")
    return training_ops.resource_sparse_apply_adagrad(
        var.handle,
        acc.handle,
        math_ops.cast(self._learning_rate_tensor, grad.dtype),
        grad,
        indices,
        use_locking=self._use_locking)
