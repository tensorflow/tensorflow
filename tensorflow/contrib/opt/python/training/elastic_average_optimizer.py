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
"""Wrapper optimizer for Elastic Average SGD """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import optimizer
from tensorflow.python.training import session_run_hook
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import constant_op

LOCAL_VARIABLE_NAME = 'local_center_variable'
GLOBAL_VARIABLE_NAME = 'global_center_variable'


class ElasticAverageCustomGetter(object):
  """Custom_getter class is used to do:
  1. Change trainable variables to local collection and place them at worker
    device
  2. Generate global variables(global center variables)
  3. Generate local variables(local center variables) which record the global
    variables and place them at worker device
    Notice that the class should be used with tf.replica_device_setter,
    so that the global center variables and global step variable can be placed
    at ps device. Besides, use 'tf.get_variable' instead of 'tf.Variable' to
    use this custom getter.

  For example,
  ea_custom_getter = ElasticAverageCustomGetter(worker_device)
  with tf.device(
    tf.train.replica_device_setter(
      worker_device=worker_device,
      ps_device="/job:ps/cpu:0",
      cluster=cluster)),
    tf.variable_scope('',custom_getter=ea_custom_getter):
    hid_w = tf.get_variable(
      initializer=tf.truncated_normal(
          [IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
          stddev=1.0 / IMAGE_PIXELS),
      name="hid_w")
    hid_b = tf.get_variable(initializer=tf.zeros([FLAGS.hidden_units]),
                            name="hid_b")
  """

  def __init__(self, worker_device):
    """Create a new `ElasticAverageCustomGetter`.

    Args:
      worker_device: String.  Name of the `worker` job.
    """
    self._worker_device = worker_device
    self._local_map = {}
    self._global_map = {}

  def __call__(self, getter, name, trainable, collections, *args, **kwargs):
    if trainable:
      with ops.device(self._worker_device):
        local_var = getter(
            name,
            trainable=True,
            collections=[ops.GraphKeys.LOCAL_VARIABLES],
            *args,
            **kwargs)
      global_center_variable = variable_scope.variable(
          name='%s/%s' % (GLOBAL_VARIABLE_NAME, name),
          initial_value=local_var.initialized_value(),
          trainable=False,
          collections=[ops.GraphKeys.GLOBAL_VARIABLES])

      with ops.device(self._worker_device):
        local_center_variable = variable_scope.variable(
            name='%s/%s' % (LOCAL_VARIABLE_NAME, name),
            initial_value=local_var.initialized_value(),
            trainable=False,
            collections=[ops.GraphKeys.LOCAL_VARIABLES])

      self._local_map[local_var] = local_center_variable
      self._global_map[local_var] = global_center_variable
      return local_var
    else:
      return getter(name, trainable, collections, *args, **kwargs)


class ElasticAverageOptimizer(optimizer.Optimizer):
  """Wrapper optimizer that implements the Elastic Average SGD algorithm.
  This is an async optimizer. During the training, Each worker will update
  the local variables and maintains its own local_step, which starts from 0
  and is incremented by 1 after each update of local variables. Whenever
  the communication period divides the local step, the worker requests
  the current global center variables and then computed the elastic difference
  between global center variables and local variables. The elastic difference
  then be used to update both local variables and global variables.
  """

  # Default value as paper described
  BETA = 0.9

  def __init__(self,
               opt,
               num_worker,
               ea_custom_getter,
               communication_period=10,
               moving_rate=None,
               rho=None,
               use_locking=True,
               name='ElasticAverageOptimizer'):
    """Construct a new gradient descent optimizer.

    Args:
      opt: The actual optimizer that will be used to update local variables.
        Must be one of the Optimizer classes.
      num_worker: The number of workers
      ea_custom_getter: The ElasticAverageCustomGetter
      communication_period: An int point value to controls the frequency
        of the communication between every worker and the ps.
      moving_rate: A floating point value to control the elastic difference.
      rho: the amount of exploration we allow ine the model. The default
        value is moving_rate/learning_rate
      use_locking: If True use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "ElasticAverageOptimizer".
    """
    super(ElasticAverageOptimizer, self).__init__(use_locking, name)
    self._opt = opt
    self._num_worker = num_worker
    self._period = communication_period
    self._local_map = ea_custom_getter._local_map
    self._global_map = ea_custom_getter._global_map

    if moving_rate is None:
      self._moving_rate = BETA / communication_period / num_worker
    else:
      self._moving_rate = moving_rate
    if rho is None:
      self._rho = self._moving_rate / self._opt._learning_rate
    else:
      self._rho = rho

    self._local_step = variable_scope.get_variable(
        initializer=0,
        trainable=False,
        collections=[ops.GraphKeys.LOCAL_VARIABLES],
        name='local_step')
    self._opt._prepare()

  def compute_gradients(self,
                        loss,
                        var_list=None,
                        gate_gradients=optimizer.Optimizer.GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
    """Compute gradients of `loss` for the variables in `var_list`.

    Add rho*elastic_difference to loss to control the exploration
    This is the first part of `minimize()`.  It returns a list
    of (gradient, variable) pairs where "gradient" is the gradient
    for "variable".  Note that "gradient" can be a `Tensor`, an
    `IndexedSlices`, or `None` if there is no gradient for the
    given variable.

    Args:
      loss: A Tensor containing the value to minimize.
      var_list: Optional list or tuple of `tf.Variable` to update to minimize
        `loss`.  Defaults to the list of variables collected in the graph
        under the key `GraphKey.TRAINABLE_VARIABLES`.
      gate_gradients: How to gate the computation of gradients.  Can be
        `GATE_NONE`, `GATE_OP`, or `GATE_GRAPH`.
      aggregation_method: Specifies the method used to combine gradient terms.
        Valid values are defined in the class `AggregationMethod`.
      colocate_gradients_with_ops: If True, try colocating gradients with
        the corresponding op.
      grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.

    Returns:
      A list of (gradient, variable) pairs. Variable is always present, but
      gradient can be `None`.

    Raises:
      TypeError: If `var_list` contains anything else than `Variable` objects.
      ValueError: If some arguments are invalid.
    """
    if not var_list:
      var_list = variables.trainable_variables()

    elastic_difference = [
        math_ops.subtract(v, lv)
        for v, lv in zip(variables.trainable_variables(),
                         [self._local_map[var] for var in var_list])
    ]

    distance_loss = self._rho * math_ops.add_n(
        [gen_nn_ops.l2_loss(ed) for ed in elastic_difference])

    total_loss = loss + distance_loss
    return self._opt.compute_gradients(total_loss, var_list, gate_gradients,
                                       aggregation_method,
                                       colocate_gradients_with_ops, grad_loss)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Apply gradients to global variables.

    This is the second part of `minimize()`. It returns an `Operation` that
    applies gradients.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        `compute_gradients()`.
      global_step: Optional `Variable` to increment by one after the
        variables have been updated.
      name: Optional name for the returned operation.  Default to the
        name passed to the `Optimizer` constructor.

    Returns:
      An `Operation` that applies the specified gradients. If `global_step`
      was not None, that operation also increments `global_step`.

    Raises:
      TypeError: If `grads_and_vars` is malformed.
      ValueError: If none of the variables have gradients.
    """
    apply_updates = self._opt.apply_gradients(grads_and_vars)
    with ops.control_dependencies([apply_updates]):
      local_update = state_ops.assign_add(
          self._local_step, 1, name='local_step_update').op

    # update global variables.
    def _Update_global_variables():
      local_vars = [v for g, v in grads_and_vars if g is not None]
      global_center_vars = [self._global_map[var] for var in local_vars]
      local_center_vars = [self._local_map[var] for var in local_vars]
      local_center_vars_update = []
      for lvar, var in zip(local_center_vars, global_center_vars):
        local_center_vars_update.append(lvar.assign(var))
      update_ops = []
      differences = []
      with ops.control_dependencies(local_center_vars_update):
        for v, lv in zip(local_vars, local_center_vars):
          with ops.device(v.device):
            differences.append(math_ops.subtract(v, lv))
        for lvar, diff in zip(local_vars, differences):
          with ops.device(lvar.device):
            update_ops.append(
                state_ops.assign_sub(lvar,
                                     math_ops.multiply(self._moving_rate,
                                                       diff)))
        for var, diff in zip(global_center_vars, differences):
          with ops.device(var.device):
            update_ops.append(
                state_ops.assign_add(var,
                                     math_ops.multiply(self._moving_rate,
                                                       diff)))
        if global_step:
          with ops.colocate_with(global_step):
            update_ops.append(state_ops.assign_add(global_step, 1))
      variable_update = control_flow_ops.group(*(update_ops))
      return variable_update

    with ops.control_dependencies([local_update]):
      condition = math_ops.equal(
          math_ops.mod(self._local_step, self._period), 0)
      conditional_update = control_flow_ops.cond(
          condition, _Update_global_variables, control_flow_ops.no_op)
    return conditional_update

  def get_init_op(self, task_index):
    """Returns the op to let all the local variables and local center
    variables equal to the global center variables before the training begins"""

    def _Add_sync_queues_and_barrier(enqueue_after_list):
      """Adds ops to enqueu on all worker queues"""
      sync_queues = [
          data_flow_ops.FIFOQueue(
              self._num_worker, [dtypes.bool],
              shapes=[[]],
              shared_name='%s%s' % ('variable_init_sync_queue', i))
          for i in range(self._num_worker)
      ]
      queue_ops = []
      # For each other worker, add an entry in a queue
      token = constant_op.constant(False)
      with ops.control_dependencies(enqueue_after_list):
        for i, q in enumerate(sync_queues):
          if i == task_index:
            queue_ops.append(control_flow_ops.no_op())
          else:
            queue_ops.append(q.enqueue(token))
      queue_ops.append(
          sync_queues[task_index].dequeue_many(len(sync_queues) - 1))
      return control_flow_ops.group(*queue_ops)

    init_ops = []
    local_vars = variables.trainable_variables()
    global_center_vars = [self._global_map[var] for var in local_vars]
    local_center_vars = [self._local_map[var] for var in local_vars]
    if not (local_vars and global_center_vars and local_center_vars):
      raise ValueError('The lists of local_variables, global_center_variables, '
                       'local_center_variables should not be empty  ')
    for lvar, gc_var, lc_var in zip(local_vars, global_center_vars,
                                    local_center_vars):
      init_ops.append(state_ops.assign(lvar, gc_var))
      init_ops.append(state_ops.assign(lc_var, gc_var))

    init_op = control_flow_ops.group(*(init_ops))
    sync_queue_op = _Add_sync_queues_and_barrier([init_op])
    return sync_queue_op

  def make_session_run_hook(self, is_chief, task_index):
    """Creates a hook to handle ElasticAverageOptimizerHook ops such as initialization."""
    return _ElasticAverageOptimizerHook(self, is_chief, task_index)


class _ElasticAverageOptimizerHook(session_run_hook.SessionRunHook):

  def __init__(self, ea_optimizer, is_chief, task_index):
    """Creates hook to handle ElasticAverageOptimizer initialization ops.

    Args:
      ea_optimizer: `ElasticAverageOptimizer` which this hook will initialize.
      is_chief: `Bool`, whether is this a chief replica or not.
    """
    self._ea_optimizer = ea_optimizer
    self._is_chief = is_chief
    self._task_index = task_index

  def begin(self):
    self._local_init_op = variables.local_variables_initializer()
    self._global_init_op = None
    if self._is_chief:
      self._global_init_op = variables.global_variables_initializer()
    self._variable_init_op = self._ea_optimizer.get_init_op(self._task_index)
