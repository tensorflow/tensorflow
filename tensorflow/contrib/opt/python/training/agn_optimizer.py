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
# ============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import optimizer
from tensorflow.python.training import session_run_hook

GLOBAL_VARIABLE_NAME = 'global_center_variable'
GRAD_VARIABLE_NAME = 'grad_variable'


class AGNCustomGetter(object):
  """Custom_getter class is used to do:

  1. Change trainable variables to local collection and place them at worker
    device
  2. Generate global variables(global center variables)
  3. Generate grad variables(gradients) which record the gradients sum
    and place them at worker device
    Notice that the class should be used with tf.replica_device_setter,
    so that the global center variables and global step variable can be placed
    at ps device.
  """

  def __init__(self, worker_device):
    """
      Args:
        worker_device: put the grad_variables on worker device
    """
    self._worker_device = worker_device
    self._global_map = {}
    self._grad_map = {}

  def __call__(self, getter, name, trainable, collections, *args, **kwargs):
    if trainable:
      with ops.device(self._worker_device):
        local_var = getter(
            name,
            trainable=True,
            collections=[ops.GraphKeys.LOCAL_VARIABLES],
            *args,
            **kwargs)
      if kwargs['reuse'] == True:
        return local_var
      global_center_variable = getter(
          name='%s/%s' % (GLOBAL_VARIABLE_NAME, name),
          trainable=False,
          collections=[ops.GraphKeys.GLOBAL_VARIABLES],
          *args,
          **kwargs)

      with ops.device(self._worker_device):
        grad_variable = getter(
            name='%s/%s' % (GRAD_VARIABLE_NAME, name),
            trainable=False,
            collections=[ops.GraphKeys.LOCAL_VARIABLES],
            *args,
            **kwargs)
      if kwargs['partitioner'] is None:
        self._grad_map[local_var] = grad_variable
        self._global_map[local_var] = global_center_variable
      else:
        v_list = list(local_var)
        for i in range(len(v_list)):
          self._grad_map[v_list[i]] = list(grad_variable)[i]
          self._global_map[v_list[i]] = list(global_center_variable)[i]
      return local_var
    else:
      return getter(
          name, trainable=trainable, collections=collections, *args, **kwargs)


class AGNOptimizer(optimizer.Optimizer):
  """Wrapper that implements the Accumulated GradientNormalization algorithm.

  Reference:
    Accumulated Gradient Normalization: Joeri Hermans ACML2017
    https://arxiv.org/abs/1710.02368
  """

  def __init__(self,
               optimizer,
               num_worker,
               custom_getter,
               communication_period=10,
               use_locking=True,
               name='AGNOptimizer'):
    """Construct a new AGN optimizer.

    Args:
      optimizer: input optimizer, can be sgd/momentum/adam etc.
      num_worker: The number of workers
      custom_getter: The AGNCustomGetter
      communication_period: An int point value to controls the frequency of the
        communication between every worker and the ps.
      use_locking: If True use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "AGNOptimizer".
    """
    super(AGNOptimizer, self).__init__(use_locking, name)
    self._opt = optimizer
    self._num_worker = num_worker
    self._period = communication_period
    self._global_map = custom_getter._global_map
    self._grad_map = custom_getter._grad_map
    self._local_step = variable_scope.get_variable(
        initializer=0,
        trainable=False,
        collections=[ops.GraphKeys.LOCAL_VARIABLES],
        name='local_step')
    self._opt._prepare()

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Apply gradients to global variables.

    This is the second part of `minimize()`. It returns an `Operation` that
    applies gradients.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        `compute_gradients()`.
      global_step: Optional `Variable` to increment by one after the variables
        have been updated.
      name: Optional name for the returned operation.  Default to the name
        passed to the `Optimizer` constructor.

    Returns:
      An `Operation` that applies the specified gradients. If `global_step`
      was not None, that operation also increments `global_step`.
    """
    local_vars = [v for g, v in grads_and_vars if g is not None]
    grads = [g for g, v in grads_and_vars if g is not None]

    def _variable_creator(next_creator, collections, **kwargs):
      if not collections:
        collections = [ops.GraphKeys.LOCAL_VARIABLES]
      elif ops.GraphKeys.GLOBAL_VARIABLES in collections:
        collections = list(collections)
        collections.append(ops.GraphKeys.LOCAL_VARIABLES)
        collections.remove(ops.GraphKeys.GLOBAL_VARIABLES)
      return next_creator(collections=collections, **kwargs)

    # theta = theta - lr * grad
    with variable_scope.variable_creator_scope(_variable_creator):
      local_update_op = self._opt.apply_gradients(grads_and_vars)

    # a = a + grad
    update_ops = []
    update_ops.append(local_update_op)
    grad_vars = [self._grad_map[var] for var in local_vars]
    for g, grad_var in zip(grads, grad_vars):
      update_ops.append(state_ops.assign_add(grad_var, g))

    global_center_vars = [self._global_map[var] for var in local_vars]

    # update global variables.
    def _Update_global_variables():
      global_norm = []
      # a = a / t
      for g in grad_vars:
        global_norm.append(state_ops.assign(g, g / self._period))
      # apply
      with ops.control_dependencies(global_norm):
        apply_global_op = self._opt.apply_gradients(
            zip(grad_vars, global_center_vars))

      # pull
      with ops.control_dependencies([apply_global_op]):
        update_ops = []
        if global_step:
          with ops.colocate_with(global_step):
            update_ops.append(state_ops.assign_add(global_step, 1))

        for lvar in local_vars:
          g_val = self._global_map[lvar].read_value()
          update_ops.append(state_ops.assign(lvar, g_val))
        for grad_var in grad_vars:
          update_ops.append(
              state_ops.assign(grad_var, array_ops.zeros_like(grad_var)))
        variable_update = control_flow_ops.group(*(update_ops))
      return variable_update

    local_update = state_ops.assign_add(
        self._local_step, 1, name='local_step_update').op

    with ops.control_dependencies([local_update]):
      condition = math_ops.equal(
          math_ops.mod(self._local_step, self._period), 0)
    with ops.control_dependencies(update_ops):
      conditional_update = control_flow_ops.cond(
          condition, _Update_global_variables, control_flow_ops.no_op)
    return conditional_update

  def get_init_op(self, task_index):
    """Returns the op to let all the local variables and local center

    variables equal to the global center variables before the training begins
    """
    init_ops = []
    local_vars = variables.trainable_variables()
    global_center_vars = [self._global_map[var] for var in local_vars]
    grad_vars = [self._grad_map[var] for var in local_vars]
    if not (local_vars and global_center_vars and grad_vars):
      raise ValueError('The lists of local_variables, global_center_variables,'
                       'grad_center_variables should not be empty')
    for lvar, gc_var in zip(local_vars, global_center_vars):
      init_ops.append(state_ops.assign(lvar, gc_var))
    for g in grad_vars:
      init_ops.append(state_ops.assign(g, array_ops.zeros_like(g)))
    init_op = control_flow_ops.group(*(init_ops))
    return init_op

  def make_session_run_hook(self, is_chief, task_index):
    """Creates a hook to handle AGNOptimizerHook ops such as initialization."""
    return _AGNOptimizerHook(self, is_chief, task_index)


class _AGNOptimizerHook(session_run_hook.SessionRunHook):

  def __init__(self, agn_optimizer, is_chief, task_index):
    """Creates hook to handle AGNOptimizer initialization ops.

    Args:
      agn_optimizer: `AGNOptimizer` which this hook will initialize.
      is_chief: `Bool`, whether is this a chief replica or not.
      task_index: int, task_index of worker
    """
    self._agn_optimizer = agn_optimizer
    self._is_chief = is_chief
    self._task_index = task_index

  def begin(self):
    self._local_init_op = variables.local_variables_initializer()
    self._global_init_op = None
    if self._is_chief:
      self._global_init_op = variables.global_variables_initializer()
    self._variable_init_op = self._agn_optimizer.get_init_op(self._task_index)

  def after_create_session(self, session, coord):
    """Run initialization ops"""
    session.run(self._variable_init_op)
