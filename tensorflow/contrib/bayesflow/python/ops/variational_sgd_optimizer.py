# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""An optimizer module for constant stochastic gradient descent."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope as varscope_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops


class VariationalSGDOptimizer(optimizer.Optimizer):
  """An optimizer module for constant stochastic gradient descent.

  This implements an optimizer module for the constant stochastic gradient
  descent algorithm [1].  The optimization variable is regarded as an
  approximate sample from the posterior .

  Note: If a prior is included in the loss, it should be scaled by
  `1/num_pseudo_batches`, where num_pseudo_batches is the number of minibatches
  in the data.  I.e., it should be divided by the `num_pseudo_batches` term
  described below.

  [1]: "Stochastic Gradient Descent as Approximate Bayesian Inference
       Stephan Mandt, Matthew D. Hoffman, David M. Blei.
       ArXiv:1704.04289, 2017. https://arxiv.org/abs/1704.04289

  Args:
    batch_size: Scalar `int`-like `Tensor`. The number of examples in a
      minibatch in the data set. Note: Assumes the loss is taken as the mean
      over a minibatch. Otherwise if the sum was taken set this to 1.
    total_num_examples: Scalar `int`-like `Tensor`. The total number of examples
      in the data set.
    max_learning_rate: Scalar `float`-like `Tensor`. A maximum allowable
      effective coordinate-wise learning rate. The algorithm scales down any
      effective learning rate (i.e. after preconditioning) that is larger than
      this. (Default: `1`)
    preconditioner_decay_rate: Scalar `float`-like `Tensor`. The exponential
      decay rate of the rescaling of the preconditioner (RMSprop). (This is
      "alpha" in [1]). Should be smaller than but nearly `1` to approximate
      sampling from the posterior. (Default: `0.95`)
    burnin: Scalar `int`-like `Tensor`. The number of iterations to collect
      gradient statistics to update the preconditioner before starting to draw
      noisy samples. (Default: `25`)
    burnin_max_learning_rate: Scalar `float`-like `Tensor`. Maximum learning
      rate to use during the burnin period.
      (Default: `1e-8`)
    use_single_learning_rate: Boolean Indicates whether one single learning
      rate is used or coordinate_wise learning rates are used.
      (Default: `False`)
    name: Python `str` describing ops managed by this function.
      (Default: `"VariationalSGDOptimizer"`)
    variable_scope: Variable scope used for calls to `tf.get_variable`.
      If `None`, a new variable scope is created using name
      `ops.get_default_graph().unique_name(name or default_name)`.

  Raises:
    InvalidArgumentError: If preconditioner_decay_rate is a `Tensor` not in
      `(0,1]`.
  """

  def __init__(self,
               batch_size,
               total_num_examples,
               max_learning_rate=1.0,
               preconditioner_decay_rate=0.95,
               burnin=25,
               burnin_max_learning_rate=1e-6,
               use_single_learning_rate=False,
               name=None,
               variable_scope=None):
    default_name = 'VariationalSGDOptimizer'
    with ops.name_scope(name, default_name, [
        max_learning_rate, preconditioner_decay_rate, batch_size, burnin,
        burnin_max_learning_rate
    ]):
      if variable_scope is None:
        var_scope_name = ops.get_default_graph().unique_name(
            name or default_name)
        with varscope_ops.variable_scope(var_scope_name) as scope:
          self._variable_scope = scope
      else:
        self._variable_scope = variable_scope

      self._preconditioner_decay_rate = ops.convert_to_tensor(
          preconditioner_decay_rate, name='preconditioner_decay_rate')
      self._batch_size = ops.convert_to_tensor(batch_size, name='batch_size')
      self._total_num_examples = ops.convert_to_tensor(
          total_num_examples, name='total_num_examples')
      self._burnin = ops.convert_to_tensor(burnin, name='burnin')
      self._burnin_max_learning_rate = ops.convert_to_tensor(
          burnin_max_learning_rate, name='burnin_max_learning_rate')
      self._max_learning_rate = ops.convert_to_tensor(
          max_learning_rate, name='max_learning_rate')
      self._use_single_learning_rate = use_single_learning_rate

      with varscope_ops.variable_scope(self._variable_scope):
        self._counter = varscope_ops.get_variable(
            'counter', initializer=0, trainable=False)

      self._preconditioner_decay_rate = control_flow_ops.with_dependencies([
          check_ops.assert_non_negative(
              self._preconditioner_decay_rate,
              message='`preconditioner_decay_rate` must be non-negative'),
          check_ops.assert_less_equal(
              self._preconditioner_decay_rate,
              1.,
              message='`preconditioner_decay_rate` must be at most 1.'),
      ], self._preconditioner_decay_rate)

      self._batch_size = control_flow_ops.with_dependencies([
          check_ops.assert_greater(
              self._batch_size,
              0,
              message='`batch_size` must be greater than zero')
      ], self._batch_size)

      self._total_num_examples = control_flow_ops.with_dependencies([
          check_ops.assert_greater(
              self._total_num_examples,
              0,
              message='`total_num_examples` must be greater than zero')
      ], self._total_num_examples)

      self._burnin = control_flow_ops.with_dependencies([
          check_ops.assert_non_negative(
              self._burnin, message='`burnin` must be non-negative'),
          check_ops.assert_integer(
              self._burnin, message='`burnin` must be an integer')
      ], self._burnin)

      self._burnin_max_learning_rate = control_flow_ops.with_dependencies([
          check_ops.assert_non_negative(
              self._burnin_max_learning_rate,
              message='`burnin_max_learning_rate` must be non-negative')
      ], self._burnin_max_learning_rate)

      self._max_learning_rate = control_flow_ops.with_dependencies([
          check_ops.assert_non_negative(
              self._max_learning_rate,
              message='`max_learning_rate` must be non-negative')
      ], self._max_learning_rate)

      super(VariationalSGDOptimizer, self).__init__(
          use_locking=False, name=name or default_name)

  def _create_slots(self, var_list):
    for v in var_list:
      init_moment = init_ops.zeros_initializer(dtype=v.dtype)
      self._get_or_make_slot_with_initializer(
          v, init_moment, v.get_shape(), v.dtype, 'first_moment', self._name)
      self._get_or_make_slot_with_initializer(
          v, init_moment, v.get_shape(), v.dtype, 'second_moment', self._name)

  def _prepare(self):
    self._decay_tensor = ops.convert_to_tensor(
        self._preconditioner_decay_rate, name='preconditioner_decay_rate')
    self._batch_size_tensor = ops.convert_to_tensor(
        self._batch_size, name='batch_size_tensor')

    super(VariationalSGDOptimizer, self)._prepare()

  def _get_coordinatewise_learning_rate(self, grad, var):
    # Compute the learning rate using a moving average for the diagonal of BB^T
    avg_first = self.get_slot(var, 'first_moment')
    avg_second = self.get_slot(var, 'second_moment')
    decay_tensor = math_ops.cast(self._decay_tensor, var.dtype)
    batch_size = math_ops.cast(self._batch_size_tensor, var.dtype)

    # Create an estimator for the moving average of gradient mean and variance
    # via Welford's algorithm
    if isinstance(grad, ops.Tensor):
      delta = grad - avg_first
      first_moment_update = avg_first.assign_add(
          array_ops.where(self._counter < 1, math_ops.cast(1, var.dtype),
                          1. - decay_tensor) * delta)

      with ops.control_dependencies([first_moment_update]):
        second_moment_update = avg_second.assign_add(
            math_ops.cast(self._counter < 1, var.dtype) *
            -(1. - decay_tensor) * (
                avg_second - decay_tensor  * math_ops.square(delta)))
      diag_preconditioner = control_flow_ops.with_dependencies(
          [second_moment_update],
          clip_ops.clip_by_value(avg_second, 1e-12, 1e12))
    elif isinstance(grad, ops.IndexedSlices):
      delta = grad.values - array_ops.gather_nd(avg_first, grad.indices)
      first_moment_update = state_ops.scatter_add(
          avg_first,
          grad.indices,
          array_ops.where(self._counter < 1,
                          math_ops.cast(1., var.dtype),
                          1. - decay_tensor) * delta)

      with ops.control_dependencies([first_moment_update]):
        avg_second = state_ops.scatter_add(
            avg_second,
            grad.indices,
            math_ops.cast(self._counter < 1, var.dtype) *
            -(1. - decay_tensor) * (
                array_ops.gather_nd(avg_second, grad.indices) - decay_tensor *
                math_ops.square(delta)))
        avg_second = array_ops.gather_nd(avg_second, grad.indices)
        # TODO(b/70783772)
        diag_preconditioner = clip_ops.clip_by_value(avg_second, 1e-12, 1e12)
    else:
      raise errors.InvalidArgumentError(
          None, None, 'grad must of type Tensor or IndexedSlice')

    diag_preconditioner *= batch_size

    if self._use_single_learning_rate:
      diag_preconditioner = math_ops.reduce_mean(diag_preconditioner)

    # From Theorem 2 Corollary 1 of Mandt et al. 2017
    return 2. * batch_size / (
        math_ops.cast(self._total_num_examples, var.dtype.base_dtype) *
        diag_preconditioner)

  def _apply_dense(self, grad, var):

    max_learning_rate = array_ops.where(self._counter < self._burnin,
                                        self._burnin_max_learning_rate,
                                        self._max_learning_rate)

    learn_rates = clip_ops.clip_by_value(
        self._get_coordinatewise_learning_rate(grad, var), 0.0,
        math_ops.cast(max_learning_rate, var.dtype.base_dtype))

    newgrad = grad * learn_rates
    return training_ops.apply_gradient_descent(
        var,
        math_ops.cast(1.0, var.dtype),
        newgrad,
        use_locking=self._use_locking).op

  def _apply_sparse(self, grad, var):

    max_learning_rate = array_ops.where(self._counter < self._burnin,
                                        self._burnin_max_learning_rate,
                                        self._max_learning_rate)

    learn_rate = clip_ops.clip_by_value(
        self._get_coordinatewise_learning_rate(grad, var), 0.0,
        math_ops.cast(max_learning_rate, var.dtype))
    delta = grad.values * learn_rate

    return state_ops.scatter_sub(var, grad.indices, delta,
                                 use_locking=self._use_locking)

  def _finish(self, update_ops, name_scope):
    update_ops.append([self._counter.assign_add(1)])
    return control_flow_ops.group(*update_ops, name=name_scope)

  @property
  def variable_scope(self):
    """Variable scope of all calls to `tf.get_variable`."""
    return self._variable_scope
