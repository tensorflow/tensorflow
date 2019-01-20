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

"""Base class to make optimizers weight decay ready."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.opt.python.training import shampoo
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import adam
from tensorflow.python.training import momentum as momentum_opt
from tensorflow.python.training import optimizer
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.ops import array_ops


class DecoupledWeightDecayExtension(object):
  """This class allows to extend optimizers with decoupled weight decay.

  It implements the decoupled weight decay described by Loshchilov & Hutter
  (https://arxiv.org/pdf/1711.05101.pdf), in which the weight decay is
  decoupled from the optimization steps w.r.t. to the loss function.
  For SGD variants, this simplifies hyperparameter search since it decouples
  the settings of weight decay and learning rate.
  For adaptive gradient algorithms, it regularizes variables with large
  gradients more than L2 regularization would, which was shown to yield better
  training loss and generalization error in the paper above.

  This class alone is not an optimizer but rather extends existing
  optimizers with decoupled weight decay. We explicitly define the two examples
  used in the above paper (SGDW and AdamW), but in general this can extend
  any OptimizerX by using
  `extend_with_weight_decay(OptimizerX, weight_decay=weight_decay)`.
  In order for it to work, it must be the first class the Optimizer with
  weight decay inherits from, e.g.

  ```python
  class AdamWOptimizer(DecoupledWeightDecayExtension, adam.AdamOptimizer):
    def __init__(self, weight_decay, *args, **kwargs):
      super(AdamWOptimizer, self).__init__(weight_decay, *args, **kwargs).
  ```

  Note that this extension decays weights BEFORE applying the update based
  on the gradient, i.e. this extension only has the desired behaviour for
  optimizers which do not depend on the value of'var' in the update step!
  
  Note: when applying a decay to the learning rate, be sure to manually apply
  the decay to the `weight_decay` as well. For example:

  ```python
    schedule = tf.train.piecewise_constant(tf.train.get_global_step(), 
                                           [10000, 15000], [1e-0, 1e-1, 1e-2])
    lr = 1e-1 * schedule()
    wd = lambda: 1e-4 * schedule()

    # ...

    optimizer = tf.contrib.opt.MomentumWOptimizer(learning_rate=lr,
                                                  weight_decay=wd,
                                                  momentum=0.9,
                                                  use_nesterov=True)
  ```
  """

  def __init__(self, weight_decay, **kwargs):
    """Construct the extension class that adds weight decay to an optimizer.

    Args:
      weight_decay: A `Tensor` or a floating point value, the factor by which
        a variable is decayed in the update step.
      **kwargs: Optional list or tuple or set of `Variable` objects to
        decay.
    """
    self._decay_var_list = None  # is set in minimize or apply_gradients
    self._weight_decay = weight_decay
    # The tensors are initialized in call to _prepare
    self._weight_decay_tensor = None
    super(DecoupledWeightDecayExtension, self).__init__(**kwargs)

  def minimize(self, loss, global_step=None, var_list=None,
               gate_gradients=optimizer.Optimizer.GATE_OP,
               aggregation_method=None, colocate_gradients_with_ops=False,
               name=None, grad_loss=None, decay_var_list=None):
    """Add operations to minimize `loss` by updating `var_list` with decay.

    This function is the same as Optimizer.minimize except that it allows to
    specify the variables that should be decayed using decay_var_list.
    If decay_var_list is None, all variables in var_list are decayed.

    For more information see the documentation of Optimizer.minimize.

    Args:
      loss: A `Tensor` containing the value to minimize.
      global_step: Optional `Variable` to increment by one after the
        variables have been updated.
      var_list: Optional list or tuple of `Variable` objects to update to
        minimize `loss`.  Defaults to the list of variables collected in
        the graph under the key `GraphKeys.TRAINABLE_VARIABLES`.
      gate_gradients: How to gate the computation of gradients.  Can be
        `GATE_NONE`, `GATE_OP`, or  `GATE_GRAPH`.
      aggregation_method: Specifies the method used to combine gradient terms.
        Valid values are defined in the class `AggregationMethod`.
      colocate_gradients_with_ops: If True, try colocating gradients with
        the corresponding op.
      name: Optional name for the returned operation.
      grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.
      decay_var_list: Optional list of decay variables.

    Returns:
      An Operation that updates the variables in `var_list`.  If `global_step`
      was not `None`, that operation also increments `global_step`.

    """
    self._decay_var_list = set(decay_var_list) if decay_var_list else False
    return super(DecoupledWeightDecayExtension, self).minimize(
        loss, global_step=global_step, var_list=var_list,
        gate_gradients=gate_gradients, aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops, name=name,
        grad_loss=grad_loss)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None,
                      decay_var_list=None):
    """Apply gradients to variables and decay the variables.

    This function is the same as Optimizer.apply_gradients except that it
    allows to specify the variables that should be decayed using
    decay_var_list. If decay_var_list is None, all variables in var_list
    are decayed.

    For more information see the documentation of Optimizer.apply_gradients.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        `compute_gradients()`.
      global_step: Optional `Variable` to increment by one after the
        variables have been updated.
      name: Optional name for the returned operation.  Default to the
        name passed to the `Optimizer` constructor.
      decay_var_list: Optional list of decay variables.

    Returns:
      An `Operation` that applies the specified gradients. If `global_step`
      was not None, that operation also increments `global_step`.
    """
    self._decay_var_list = set(decay_var_list) if decay_var_list else False
    return super(DecoupledWeightDecayExtension, self).apply_gradients(
        grads_and_vars, global_step=global_step, name=name)

  def _prepare(self):
    weight_decay = self._weight_decay
    if callable(weight_decay):
      weight_decay = weight_decay()
    self._weight_decay_tensor = ops.convert_to_tensor(
        weight_decay, name="weight_decay")
    # Call the optimizers _prepare function.
    super(DecoupledWeightDecayExtension, self)._prepare()

  def _decay_weights_op(self, var):
    if not self._decay_var_list or var in self._decay_var_list:
      return var.assign_sub(self._weight_decay * var, self._use_locking)
    return control_flow_ops.no_op()

  def _decay_weights_sparse_op(self, var, indices, scatter_add):
    if not self._decay_var_list or var in self._decay_var_list:
      update = -self._weight_decay * array_ops.gather(var, indices)
      return scatter_add(var, indices, update, self._use_locking)
    return control_flow_ops.no_op()

  # Here, we overwrite the apply functions that the base optimizer calls.
  # super().apply_x resolves to the apply_x function of the BaseOptimizer.
  def _apply_dense(self, grad, var):
    with ops.control_dependencies([self._decay_weights_op(var)]):
      return super(DecoupledWeightDecayExtension, self)._apply_dense(grad, var)

  def _resource_apply_dense(self, grad, var):
    with ops.control_dependencies([self._decay_weights_op(var)]):
      return super(DecoupledWeightDecayExtension, self)._resource_apply_dense(
          grad, var)

  def _apply_sparse(self, grad, var):
    scatter_add = state_ops.scatter_add
    decay_op = self._decay_weights_sparse_op(var, grad.indices, scatter_add)
    with ops.control_dependencies([decay_op]):
      return super(DecoupledWeightDecayExtension, self)._apply_sparse(
          grad, var)

  def _resource_scatter_add(self, x, i, v, _=None):
    # last argument allows for one overflow argument, to have the same function
    # signature as state_ops.scatter_add
    with ops.control_dependencies(
        [resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
      return x.value()

  def _resource_apply_sparse(self, grad, var, indices):
    scatter_add = self._resource_scatter_add
    decay_op = self._decay_weights_sparse_op(var, indices, scatter_add)
    with ops.control_dependencies([decay_op]):
      return super(DecoupledWeightDecayExtension, self)._resource_apply_sparse(
          grad, var, indices)


def extend_with_decoupled_weight_decay(base_optimizer):
  """Factory function returning an optimizer class with decoupled weight decay.

  Returns an optimizer class. An instance of the returned class computes the
  update step of `base_optimizer` and additionally decays the weights.
  E.g., the class returned by
  `extend_with_decoupled_weight_decay(tf.train.AdamOptimizer)` is equivalent to
  `tf.contrib.opt.AdamWOptimizer`.

  The API of the new optimizer class slightly differs from the API of the
  base optimizer:
  - The first argument to the constructor is the weight decay rate.
  - `minimize` and `apply_gradients` accept the optional keyword argument
    `decay_var_list`, which specifies the variables that should be decayed.
    If `None`, all variables that are optimized are decayed.

  Usage example:
  ```python
  # MyAdamW is a new class
  MyAdamW = extend_with_decoupled_weight_decay(tf.train.AdamOptimizer)
  # Create a MyAdamW object
  optimizer = MyAdamW(weight_decay=0.001, learning_rate=0.001)
  sess.run(optimizer.minimize(loss, decay_variables=[var1, var2]))

  Note that this extension decays weights BEFORE applying the update based
  on the gradient, i.e. this extension only has the desired behaviour for
  optimizers which do not depend on the value of'var' in the update step!
  ```

  Args:
    base_optimizer: An optimizer class that inherits from tf.train.Optimizer.

  Returns:
    A new optimizer class that inherits from DecoupledWeightDecayExtension
    and base_optimizer.
  """

  class OptimizerWithDecoupledWeightDecay(DecoupledWeightDecayExtension,
                                          base_optimizer):
    """Base_optimizer with decoupled weight decay.

    This class computes the update step of `base_optimizer` and
    additionally decays the variable with the weight decay being decoupled from
    the optimization steps w.r.t. to the loss function, as described by
    Loshchilov & Hutter (https://arxiv.org/pdf/1711.05101.pdf).
    For SGD variants, this simplifies hyperparameter search since
    it decouples the settings of weight decay and learning rate.
    For adaptive gradient algorithms, it regularizes variables with large
    gradients more than L2 regularization would, which was shown to yield
    better training loss and generalization error in the paper above.
    """

    def __init__(self, weight_decay, *args, **kwargs):
      # super delegation is necessary here
      # pylint: disable=useless-super-delegation
      super(OptimizerWithDecoupledWeightDecay, self).__init__(
          weight_decay, *args, **kwargs)
      # pylint: enable=useless-super-delegation

  return OptimizerWithDecoupledWeightDecay


@tf_export("contrib.opt.MomentumWOptimizer")
class MomentumWOptimizer(DecoupledWeightDecayExtension,
                         momentum_opt.MomentumOptimizer):
  """Optimizer that implements the Momentum algorithm with weight_decay.

  This is an implementation of the SGDW optimizer described in "Fixing
  Weight Decay Regularization in Adam" by Loshchilov & Hutter
  (https://arxiv.org/abs/1711.05101)
  ([pdf])(https://arxiv.org/pdf/1711.05101.pdf).
  It computes the update step of `train.MomentumOptimizer` and additionally
  decays the variable. Note that this is different from adding
  L2 regularization on the variables to the loss. Decoupling the weight decay
  from other hyperparameters (in particular the learning rate) simplifies
  hyperparameter search.

  For further information see the documentation of the Momentum Optimizer.

  Note that this optimizer can also be instantiated as
  ```python
  extend_with_weight_decay(tf.train.MomentumOptimizer,
                           weight_decay=weight_decay)
  ```
  """

  def __init__(self, weight_decay, learning_rate, momentum,
               use_locking=False, name="MomentumW", use_nesterov=False):
    """Construct a new MomentumW optimizer.

    For further information see the documentation of the Momentum Optimizer.

    Args:
      weight_decay:  A `Tensor` or a floating point value.  The weight decay.
      learning_rate: A `Tensor` or a floating point value.  The learning rate.
      momentum: A `Tensor` or a floating point value.  The momentum.
      use_locking: If `True` use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to "Momentum".
      use_nesterov: If `True` use Nesterov Momentum.
        See [Sutskever et al., 2013](
        http://jmlr.org/proceedings/papers/v28/sutskever13.pdf).
        This implementation always computes gradients at the value of the
        variable(s) passed to the optimizer. Using Nesterov Momentum makes the
        variable(s) track the values called `theta_t + mu*v_t` in the paper.

    @compatibility(eager)
    When eager execution is enabled, learning_rate, weight_decay and momentum
    can each be a callable that takes no arguments and returns the actual value
    to use. This can be useful for changing these values across different
    invocations of optimizer functions.
    @end_compatibility
    """
    super(MomentumWOptimizer, self).__init__(
        weight_decay, learning_rate=learning_rate, momentum=momentum,
        use_locking=use_locking, name=name, use_nesterov=use_nesterov)


@tf_export("contrib.opt.AdamWOptimizer")
class AdamWOptimizer(DecoupledWeightDecayExtension, adam.AdamOptimizer):
  """Optimizer that implements the Adam algorithm with weight decay.

  This is an implementation of the AdamW optimizer described in "Fixing
  Weight Decay Regularization in Adam" by Loshchilov & Hutter
  (https://arxiv.org/abs/1711.05101)
  ([pdf])(https://arxiv.org/pdf/1711.05101.pdf).

  It computes the update step of `train.AdamOptimizer` and additionally decays
  the variable. Note that this is different from adding L2 regularization on
  the variables to the loss: it regularizes variables with large
  gradients more than L2 regularization would, which was shown to yield better
  training loss and generalization error in the paper above.

  For further information see the documentation of the Adam Optimizer.

  Note that this optimizer can also be instantiated as
  ```python
  extend_with_weight_decay(tf.train.AdamOptimizer, weight_decay=weight_decay)
  ```
  """

  def __init__(self, weight_decay, learning_rate=0.001, beta1=0.9, beta2=0.999,
               epsilon=1e-8, use_locking=False, name="AdamW"):
    """Construct a new AdamW optimizer.

    For further information see the documentation of the Adam Optimizer.

    Args:
      weight_decay:  A `Tensor` or a floating point value.  The weight decay.
      learning_rate: A Tensor or a floating point value.  The learning rate.
      beta1: A float value or a constant float tensor.
        The exponential decay rate for the 1st moment estimates.
      beta2: A float value or a constant float tensor.
        The exponential decay rate for the 2nd moment estimates.
      epsilon: A small constant for numerical stability. This epsilon is
        "epsilon hat" in the Kingma and Ba paper (in the formula just before
        Section 2.1), not the epsilon in Algorithm 1 of the paper.
      use_locking: If True use locks for update operations.
      name: Optional name for the operations created when applying gradients.
        Defaults to "Adam".
    """
    super(AdamWOptimizer, self).__init__(
        weight_decay, learning_rate=learning_rate, beta1=beta1, beta2=beta2,
        epsilon=epsilon, use_locking=use_locking, name=name)


@tf_export("contrib.opt.ShampooWOptimizer")
class ShampooWOptimizer(DecoupledWeightDecayExtension,
                        shampoo.ShampooOptimizer):
  """Optimizer that implements the Shampoo algorithm with weight decay.

  For further information see the documentation of the Shampoo Optimizer.
  """

  def __init__(self,
               weight_decay,
               global_step,
               max_matrix_size=768,
               gbar_decay=0.0,
               gbar_weight=1.0,
               mat_gbar_decay=1.0,
               mat_gbar_weight=1.0,
               learning_rate=1.0,
               svd_interval=1,
               precond_update_interval=1,
               epsilon=1e-4,
               alpha=0.5,
               use_iterative_root=False,
               use_locking=False,
               name="ShampooW"):
    """Construct a new ShampooW optimizer.

    For further information see the documentation of the Shampoo Optimizer.

    Args:
      weight_decay:  A `Tensor` or a floating point value.  The weight decay.
      global_step: tensorflow variable indicating the step.
      max_matrix_size: We do not perform SVD for matrices larger than this.
      gbar_decay:
      gbar_weight:  Used to update gbar: gbar[t] = gbar_decay[t] * gbar[t-1] +
        gbar_weight[t] * g[t]
      mat_gbar_decay:
      mat_gbar_weight:  Used to update mat_gbar: mat_gbar_j[t] =
        mat_gbar_decay[t] * mat_gbar_j[t-1] + mat_gbar_weight[t] * gg_j[t]
      learning_rate: Similar to SGD
      svd_interval: We should do SVD after this many steps. Default = 1, i.e.
        every step. Usually 20 leads to no loss of accuracy, and 50 or 100 is
        also OK. May also want more often early,
                    and less often later - set in caller as for example:
                    "svd_interval = lambda(T): tf.cond(
                        T < 2000, lambda: 20.0, lambda: 1000.0)"
      precond_update_interval: We should update the preconditioners after this
        many steps. Default = 1. Usually less than svd_interval.
      epsilon:  epsilon * I_n is added to each mat_gbar_j for stability
      alpha:  total power of the preconditioners.
      use_iterative_root: should the optimizer use SVD (faster) or the iterative
        root method (for TPU) for finding the roots of PSD matrices.
      use_locking: If `True` use locks for update operations.
      name: name of optimizer.
    """
    super(ShampooWOptimizer, self).__init__(
        weight_decay,
        global_step=global_step,
        max_matrix_size=max_matrix_size,
        gbar_decay=gbar_decay,
        gbar_weight=gbar_weight,
        mat_gbar_decay=mat_gbar_weight,
        learning_rate=learning_rate,
        svd_interval=svd_interval,
        precond_update_interval=precond_update_interval,
        epsilon=epsilon,
        alpha=alpha,
        use_iterative_root=use_iterative_root,
        use_locking=use_locking,
        name=name)
