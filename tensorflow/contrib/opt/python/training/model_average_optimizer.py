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

"""Wrapper optimizer for Model Average """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import constant_op
from tensorflow.python.training import optimizer
from tensorflow.python.training import session_run_hook
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops

GLOBAL_VARIABLE_NAME = 'global_center_variable'


class ModelAverageCustomGetter(object):
  """Custom_getter class is used to do:
  1. Change trainable variables to local collection and place them at worker
    device
  2. Generate global variables
    Notice that the class should be used with tf.replica_device_setter,
    so that the global center variables and global step variable can be placed
    at ps device. Besides, use 'tf.get_variable' instead of 'tf.Variable' to
    use this custom getter.

  For example,
  ma_custom_getter = ModelAverageCustomGetter(worker_device)
  with tf.device(
    tf.train.replica_device_setter(
      worker_device=worker_device,
      ps_device="/job:ps/cpu:0",
      cluster=cluster)),
    tf.variable_scope('',custom_getter=ma_custom_getter):
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
    self._local_2_global = {}

  def __call__(self, getter, name, trainable, collections, *args, **kwargs):
    if trainable:
      with ops.device(self._worker_device):
        local_var = getter(name, trainable=True,
                           collections=[ops.GraphKeys.LOCAL_VARIABLES],
                           *args, **kwargs)

      global_variable = variable_scope.variable(
        name='%s/%s' % (GLOBAL_VARIABLE_NAME, name),
        initial_value=local_var.initialized_value(),
        trainable=False,
        collections=[ops.GraphKeys.GLOBAL_VARIABLES])

      self._local_2_global[local_var] = global_variable
      return local_var
    else:
      return getter(name, trainable, collections, *args, **kwargs)


class ModelAverageOptimizer(optimizer.Optimizer):
  """Wrapper optimizer that implements the Model Average algorithm.
  This is a sync optimizer. During the training, each worker will update
  the local variables and maintains its own local_step, which starts from 0
  and is incremented by 1 after each update of local variables. Whenever the
  interval_steps divides the local step, the local variables from all the
  workers will be averaged and assigned to global center variables. Then the
  local variables will be assigned by global center variables.
  """

  def __init__(
      self,
      opt,
      num_worker,
      is_chief,
      ma_custom_getter,
      interval_steps=100,
      use_locking=True,
      name="ModelAverageOptimizer"):
    """Construct a new model average optimizer.

    Args:
      opt: The actual optimizer that will be used to update local variables
      num_worker: The number of workers
      is_chief: whether chief worker
      ma_custom_getter: ModelAverageCustomGetter
      interval_steps: An int point value to controls the frequency of the
        average of local variables
      use_locking: If True use locks for update operations
      name: string. Optional name of the returned operation
    """
    super(ModelAverageOptimizer, self).__init__(use_locking, name)
    self._opt = opt
    self._num_worker = num_worker
    self._is_chief = is_chief
    self._local_2_global = ma_custom_getter._local_2_global
    self._interval_steps = interval_steps
    self._accumulator_list = []
    self._chief_init_op = None

    self._local_step = variable_scope.get_variable(
      initializer=0,
      trainable=False,
      collections=[ops.GraphKeys.LOCAL_VARIABLES],
      name="local_step")

    self._opt._prepare()

  def compute_gradients(self, *args, **kwargs):
    """Compute gradients of "loss" for the variables in "var_list".

    This simply wraps the compute_gradients() from the real optimizer.

    Args:
      *args: Arguments for compute_gradients().
      **kwargs: Keyword arguments for compute_gradients().

    Returns:
      A list of (gradient, variable) pairs.
    """
    return self._opt.compute_gradients(*args, **kwargs)

  def _local_vars_update(self, var_list):
    """Get the update ops for the local variables in "var_list".

    Args:
      var_list: Optional list or tuple of 'tf.Variable' to update

    Returns:
      An update op
    """
    if not var_list:
      raise ValueError(
        'The list of local_variables should not be empty')
    update_ops = []
    global_center_vars = [self._local_2_global[var] for var in var_list]
    for lvar, gvar in zip(var_list, global_center_vars):
      with ops.device(lvar.device):
        update_ops.append(state_ops.assign(lvar, gvar.read_value()))
    return control_flow_ops.group(*(update_ops))

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Apply gradients to variables.

    This contains most of the synchronization implementation and also wraps the
    apply_gradients() from the real optimizer. The chief work updates global
    variables.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        compute_gradients().
      global_step: Optional Variable to increment by one after the
        variables have been updated.
      name: Optional name for the returned operation.  Default to the
        name passed to the Optimizer constructor.

    Returns:
      A conditional 'Operation' that update both local and global variables or
      just local variables

    Raises:
      ValueError: If the grads_and_vars is empty.
      ValueError: If global step is not provided, the staleness cannot be
        checked.
    """

    # update local variables
    if not grads_and_vars:
      raise ValueError("Must supply at least one variable")
    if global_step is None:
      raise ValueError("Global step is required")

    apply_updates = self._opt.apply_gradients(grads_and_vars)
    with ops.control_dependencies([apply_updates]):
      local_update = state_ops.assign_add(
        self._local_step, 1, name='local_step_update').op

    # update global variables.
    def _Update_global_variables():
      local_vars = [v for g, v in grads_and_vars if g is not None]
      global_vars = [self._local_2_global[v] for v in local_vars]
      # sync queue
      with ops.colocate_with(global_step):
        sync_queue = data_flow_ops.FIFOQueue(-1, [dtypes.bool], shapes=[[]],
                                             shared_name='sync_queue')
      train_ops = []
      aggregated_vars = []
      with ops.name_scope(None, self._name + '/global'):
        for var, gvar in zip(local_vars, global_vars):
          with ops.device(gvar.device):
            if isinstance(var._ref(), ops.Tensor):
              var_accum = data_flow_ops.ConditionalAccumulator(
                var.dtype,
                shape=var.get_shape(),
                shared_name=gvar.name + "/var_accum")
              train_ops.append(
                var_accum.apply_grad(var._ref(), local_step=global_step))
              aggregated_vars.append(var_accum.take_grad(self._num_worker))
            else:
              raise ValueError("Unknown local variable type!")
            self._accumulator_list.append((var_accum, gvar.device))
      # chief worker updates global vars and enqueues tokens to the sync queue
      if self._is_chief:
        update_ops = []
        with ops.control_dependencies(train_ops):
          for avg_var, gvar in zip(aggregated_vars, global_vars):
            with ops.device(gvar.device):
              update_ops.append(state_ops.assign(gvar, avg_var))
          with ops.device(global_step.device):
            update_ops.append(state_ops.assign_add(global_step, 1))
        with ops.control_dependencies(update_ops), ops.device(
            global_step.device):
          tokens = array_ops.fill([self._num_worker - 1],
                                  constant_op.constant(False))
          sync_op = sync_queue.enqueue_many(tokens)
      else:
        with ops.control_dependencies(train_ops), ops.device(
            global_step.device):
          sync_op = sync_queue.dequeue()

      with ops.control_dependencies([sync_op]):
        local_update_op = self._local_vars_update(local_vars)
      return local_update_op

    with ops.control_dependencies([local_update]):
      condition = math_ops.equal(math_ops.mod(
        self._local_step, self._interval_steps), 0)
      conditional_update = control_flow_ops.cond(
        condition, _Update_global_variables, control_flow_ops.no_op)

    chief_init_ops = []
    for accum, dev in self._accumulator_list:
      with ops.device(dev):
        chief_init_ops.append(
          accum.set_global_step(
            global_step, name="SetGlobalStep"))
    self._chief_init_op = control_flow_ops.group(*(chief_init_ops))

    return conditional_update

  def get_init_op(self):
    """Returns the op to let all the local variables equal to the global
     variables before the training begins"""
    return self._local_vars_update(variables.trainable_variables())

  def make_session_run_hook(self):
    """Creates a hook to handle ModelAverage ops such as initialization."""
    return _ModelAverageOptimizerHook(self, self._is_chief)


class _ModelAverageOptimizerHook(session_run_hook.SessionRunHook):
  def __init__(self, ma_optimizer, is_chief):
    """Creates hook to handle ModelAverageOptimizer initialization ops.

    Args:
      ea_optimizer: `ModelAverageOptimizer` which this hook will initialize.
      is_chief: `Bool`, whether is this a chief replica or not.
    """
    self._ma_optimizer = ma_optimizer
    self._is_chief = is_chief

  def begin(self):
    self._local_init_op = variables.local_variables_initializer()
    self._global_init_op = None
    if self._is_chief:
      self._global_init_op = variables.global_variables_initializer()
      self._chief_init_op = self._ma_optimizer._chief_init_op
    self._variable_init_op = self._ma_optimizer.get_init_op()
