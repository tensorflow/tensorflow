# 2017 Contrib.
# ==============================================================================

"""Synchronize replicas for model average training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import six

from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.training import server_lib
from tensorflow.python.training import device_setter
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import queue_runner
from tensorflow.python.training import session_manager
from tensorflow.python.training import session_run_hook

class _ModelAverageDeviceChooser(object):
  """Class to choose devices for Ops in a model average training setup.

  This class is not to be used directly by users.  See instead
  `model_average_device_setter()` below.
  """

  def __init__(self, ps_tasks, ps_device, worker_device, ps_ops,
               ps_strategy):
    """Create a new `_ReplicaDeviceChooser`.

    Args:
      ps_tasks: Number of tasks in the `ps` job.
      ps_device: String.  Name of the `ps` job.
      worker_device: String.  Name of the `worker` job.
      ps_ops: List of strings representing `Operation` types that need to be
        placed on `ps` devices.
      ps_strategy: A callable invoked for every ps `Operation` (i.e. matched by
        `ps_ops`), that takes the `Operation` and returns the ps task index to
        use.
    """
    self._ps_tasks = ps_tasks
    self._ps_device = ps_device
    self._worker_device = worker_device
    self._ps_ops = ps_ops
    self._ps_strategy = ps_strategy

  def device_function(self, op):
    """Choose a device for `op`.

    Args:
      op: an `Operation`.

    Returns:
      The device to use for the `Operation`.
    """

    current_device = pydev.DeviceSpec.from_string(op.device or "")

    node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
    # TODO(chen meng): only global variables should be placed on ps. Now we just
    # do this according to substring in var name.
    # put global variables on ps
    if self._ps_tasks and self._ps_device and node_def.op in self._ps_ops \
       and node_def.name.find('modelAverage') != -1:
      ps_device = pydev.DeviceSpec.from_string(self._ps_device)

      current_job, ps_job = current_device.job, ps_device.job
      if ps_job and (not current_job or current_job == ps_job):
        ps_device.task = self._ps_strategy(op)

      ps_device.merge_from(current_device)
      return ps_device.to_string()

    # put internal model average variables on worker
    worker_device = pydev.DeviceSpec.from_string(self._worker_device or "")
    worker_device.merge_from(current_device)
    return worker_device.to_string()

def model_average_device_setter(ps_tasks=0, ps_device="/job:ps",
                                worker_device="/job:worker",
                                cluster=None, ps_ops=None, ps_strategy=None):
  """Return a `device function` to use when building a Graph for model average.

  There is only one difference between model_average_device_setter and
  replica_device_setter : replica_device_setter placed all variables
  (including global/local variables) on ps, while in model average, each worker
  own its local variables (local model parameters), these local variables
  should be placed in each worker.
  Args:
    ps_tasks: Number of tasks in the `ps` job.  Ignored if `cluster` is
      provided.
    ps_device: String.  Device of the `ps` job.  If empty no `ps` job is used.
      Defaults to `ps`.
    worker_device: String.  Device of the `worker` job.  If empty no `worker`
      job is used.
    cluster: `ClusterDef` proto or `ClusterSpec`.
    ps_ops: List of strings representing `Operation` types that need to be
      placed on `ps` devices.  If `None`, defaults to `["Variable"]`.
    ps_strategy: A callable invoked for every ps `Operation` (i.e. matched by
      `ps_ops`), that takes the `Operation` and returns the ps task index to
      use.  If `None`, defaults to a round-robin strategy across all `ps`
      devices.

  Returns:
    A function to pass to `tf.device()`.

  Raises:
    TypeError if `cluster` is not a dictionary or `ClusterDef` protocol buffer,
    or if `ps_strategy` is provided but not a callable.
  """
  if cluster is not None:
    if isinstance(cluster, server_lib.ClusterSpec):
      cluster_spec = cluster.as_dict()
    else:
      cluster_spec = server_lib.ClusterSpec(cluster).as_dict()
    # Get ps_job_name from ps_device by striping "/job:".
    ps_job_name = pydev.DeviceSpec.from_string(ps_device).job
    if ps_job_name not in cluster_spec or cluster_spec[ps_job_name] is None:
      return None
    ps_tasks = len(cluster_spec[ps_job_name])

  if ps_tasks == 0:
    return None

  if ps_ops is None:
    ps_ops = ["Variable", "VariableV2", "VarHandleOp"]

  if ps_strategy is None:
    # pylint: disable=protected-access
    ps_strategy = device_setter._RoundRobinStrategy(ps_tasks)
  if not six.callable(ps_strategy):
    raise TypeError("ps_strategy must be callable")
  chooser = _ModelAverageDeviceChooser(
      ps_tasks, ps_device, worker_device, ps_ops, ps_strategy)
  return chooser.device_function

class ModelAverageOptimizer(object):
  """Class to synchronize, aggregate model params.

  In a typical synchronous training environment (N-replica synchronous training)
  , gradients will be averaged each step, and then applying them to the
  variables in one shot, after which replicas can fetch the new variables and
  continue. In a model average training environment, model variables will be
  averaged (or with momentum) every 'interval_steps' steps, and then fetch the
  new variables and continue training in local worker. In the interval between
  two "average operation", there are no data transfer at all, which can
  accerlate training.

  The following accumulators/queue are created:
  <empty line>
  * N `model-variable accumulators`, one per variable for train model. local
  variables are pushed to them and the chief worker will wait until enough
  variables are collected and then average them. The accumulator will drop all
  stale variables (more details in the accumulator op).
  * 1 `token` queue where the optimizer pushes the new global_step value after
    all variables are updated.

  The following local variable is created:
  * `sync_rep_local_step`, one per replica. Compared against the global_step in
    each accumulator to check for staleness of the variables.

  The optimizer adds nodes to the graph to collect local variables and pause
  the trainers until variables are updated.
  For the Parameter Server job:
  <empty line>
  1. An accumulator is created for each variable, and each replica pushes the
     local variables into the accumulators.
  2. Each accumulator averages once enough variables (replicas_to_aggregate)
     have been accumulated.
  3. apply the averaged variables to global variables.
  4. Only after all variables have been updated, increment the global step.
  5. Only after step 4, pushes `global_step` in the `token_queue`, once for
     each worker replica. The workers can now fetch the global step, use it to
     update its local_step variable and start the next batch.

  For the replicas:
  <empty line>
  1. Start a training block: fetch variables, finish "interval_steps" steps
     training.
  2. Once current training block has been finished, push local variables into
     accumulators. Each accumulator will check the staleness and drop the
     stale ones.
  3. After pushing all the variables, dequeue an updated value of global_step
     from the token queue and record that step to its local_step variable. Note
     that this is effectively a barrier.
  4. fetch new variables, Start the next block.

  ### Usage

  ```python
  # Create any optimizer to update the variables, say a simple SGD:
  opt = GradientDescentOptimizer(learning_rate=0.1)

  # Create a ModelAverageOptimizer to update the global variables:
  # Note that if you want to have 2 backup replicas, you can change
  # total_num_replicas=52 and make sure this number matches how many physical
  # replicas you started in your job.
  ma = tf.contrib.model_average.ModelAverageOptimizer(replicas_to_aggregate=50,
                                                      interval_steps=100)

  # You can create the hook which handles model average operations.
  ma_hook = ma.make_ma_run_hook()
  # And also, create the hook which handles initialization and queues.
  ma_replicas_hook = ma.make_session_run_hook(is_chief)
  ```

  In the training program, every worker will run the train_op as if not
  model_average or synchronized. Note that if you want to run other ops like
  test op, you should use common session instead of monitoredSession:

  ```python
  with training.MonitoredTrainingSession(
      master=workers[worker_id].target, is_chief=is_chief,
      hooks=[ma_replicas_hook, ma_hook]) as mon_sess:
    while not mon_sess.should_stop():
      mon_sess.run(training_op)

      sess = mon_sess._tf_sess()
      sess.run(testing_op)
  ```
  """

  def __init__(self,
               replicas_to_aggregate,
               interval_steps,
               total_num_replicas=None,
               block_momentum_rate=0.0,
               use_nesterov=True,
               block_learning_rate=1.0):
    """Construct a model_average optimizer.

    Args:
      replicas_to_aggregate: number of replicas to aggregate for each variable
        update.
      interval_steps: number of steps between two "average op", which specifies
        how frequent a model synchronization is performed.
      total_num_replicas: Total number of tasks/workers/replicas, could be
        different from replicas_to_aggregate.
        total_num_replicas > replicas_to_aggregate: it is backup_replicas +
        replicas_to_aggregate.
      block_momentum_rate: It brings in the historical blockwise gradients.
        The default value is 0.0. When using default value, the naive
        ModelAverage method is applied, and the original learning rate of
        local optimizer should be multiply by num_of_workers.
        If using BMUF algorithm, the block momentum_rate is usually set
        according to the number of workers: block_momentum_rate =
        1.0 - 1.0/num_of_workers, the learning rate of local optimizer can be
        unchanged.
        For details, see Ref: K. Chen and Q. Huo, "Scalable training of deep
        learning machines by incremental block training with intra-block
        parallel optimization and blockwise model-update filtering," in
        Proceedings of ICASSP, 2016.
      use_nesterov: means the Nesterov-style momentum update is applied on the
        block level. The default value is true. This can accelerate training
        with non-zero block_momentum_rate.
      block_learning_rate: block_learning_rate is always 1.0 or slightly higher
        than 1.0
    """
    if total_num_replicas is None:
      total_num_replicas = replicas_to_aggregate
    logging.info(
        "ModelAverageV1: replicas_to_aggregate=%s; total_num_replicas=%s",
        replicas_to_aggregate, total_num_replicas)
    self._replicas_to_aggregate = replicas_to_aggregate
    self._block_momentum_rate = block_momentum_rate
    self._block_learning_rate = block_learning_rate
    self._interval_steps = interval_steps
    self._use_nesterov = use_nesterov
    self._gradients_applied = False
    self._total_num_replicas = total_num_replicas
    self._tokens_per_step = max(total_num_replicas, replicas_to_aggregate)
    self._global_step = None
    self._sync_token_queue = None

    # The synchronization op will be executed in a queue runner which should
    # only be executed by one of the replicas (usually the chief).
    self._chief_queue_runner = None

    # Remember which accumulator is on which device to set the initial step in
    # the accumulator to be global step. This list contains list of the
    # following format: (accumulator, device).
    self._accumulator_list = []
    # ModelAverageHook should be called before ReplicasHook.
    self._ma_run_hook = False
    # name: string. Name of the global variables and related operation on ps.
    self._name = 'modelAverage'

    self._generate_local_and_global_variables()

  def _generate_local_and_global_variables(self):
    """Change all variables to local variables and generate a global-version
       placed on ps for each.
    """
    # Change all variables to local variables.
    for v in variables.global_variables():
      if v.op.name.find(self._name) != -1:
        raise AssertionError('%s: cannot use \'%s\' as a substr of any name for'
                             ' ops or variables when calling an '
                             'ModelAverageOptimizer.' % (v.op.name, self._name))
      ops.add_to_collection(ops.GraphKeys.LOCAL_VARIABLES, v)
    # Clear global_variables list.
    ops.get_default_graph().clear_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    # Generate new global variables dependent on trainable variables.
    for v in variables.trainable_variables():
      if v.op.name.find(self._name) != -1:
        raise AssertionError('%s: cannot use \'%s\' as a substr of a name for '
                             'any ops or variables when calling an '
                             'ModelAverageOptimizer.' % (v.op.name, self._name))
      # v_g is the global-variable version of each user-defined trainable
      # variable.  They are supposed to be placed on PS device. v_g is used to
      # store averaged model parameters.
      v_g = variable_scope.get_variable(
          name='%s/g/%s' % (self._name, v.op.name),
          initializer=v.initialized_value(),
          trainable=False,
          collections=[ops.GraphKeys.GLOBAL_VARIABLES, 'global_model'])
      v_block_grad = variable_scope.get_variable(
          name='%s/block_grad/%s' % (self._name, v.op.name), shape=v.get_shape(),
          initializer=init_ops.constant_initializer(0.0, dtype=v.dtype),
          trainable=False,
          collections=[ops.GraphKeys.GLOBAL_VARIABLES, 'block_grad'])
      v_nesterov = variable_scope.get_variable(
          name='%s/nesterov/%s' % (self._name, v.op.name),
          initializer=v_g.initialized_value(),
          trainable=False,
          collections=[ops.GraphKeys.GLOBAL_VARIABLES, 'nesterov'])
    self._super_global_step = variables.Variable(0, name="%s_global_step" %
                                                 self._name, trainable=False)
    self._num_of_global_variables = len(variables.global_variables())
    self._num_of_trainable_variables = len(variables.trainable_variables())

  def _apply_model_average(self,
                           lvars_and_gvars,
                           global_vars,
                           block_grad_vars,
                           nesterov_vars,
                           global_step=None,
                           name=None):
    """Apply local weights to global variables.

    Args:
      lvars_and_gvars: List of (local_vars, global_vars) pairs.
      global_vars: The averaged weights.
      block_grad_vars: The historical blockwise gradients.
      nesterov_vars: The Nesterov-style momentum updated weights.
      global_step: Optional Variable to increment by one after the
        variables have been updated.
      name: Optional name for the returned operation.  Default to the
        name passed to the Optimizer constructor.

    Returns:
      train_op: The op to dequeue a token so the replicas can exit this batch
      and start the next one. This is executed by each replica.

    Raises:
      ValueError: If the lvars_and_gvars is empty.
      ValueError: If global step is not provided, the staleness cannot be
        checked.
    """
    if not lvars_and_gvars:
      raise ValueError("Must supply at least one variable")

    if global_step is None:
      raise ValueError("Global step is required to check staleness")

    self._global_step = global_step
    train_ops = []
    aggregated_lvars = []
    var_list = []

    model_reassign_ops = []
    nesterov_reassign_ops = []
    bg_reassign_ops = []


    self._local_step = variables.Variable(
        initial_value=0,
        trainable=False,
        collections=[ops.GraphKeys.LOCAL_VARIABLES],
        dtype=global_step.dtype.base_dtype,
        name="ma_local_step")
    self.local_step_init_op = state_ops.assign(self._local_step, global_step)
    chief_init_ops = [self.local_step_init_op]
    self.ready_for_local_init_op = variables.report_uninitialized_variables(
        variables.global_variables())

    with ops.name_scope(None, self._name):
      for lvar, var in lvars_and_gvars:
        lvar = ops.convert_to_tensor(lvar)
        var_list.append(var)
        with ops.device(var.device):
          if lvar is None:
            aggregated_lvars.append(None)  # pass-through.
            continue
          elif isinstance(lvar, ops.Tensor):
            lvar_accum = data_flow_ops.ConditionalAccumulator(
                lvar.dtype,
                shape=var.get_shape(),
                shared_name=var.name + "/lvar_accum")
            train_ops.append(lvar_accum.apply_grad(
                lvar, local_step=self._local_step))
            aggregated_lvars.append(lvar_accum.take_grad(
                self._replicas_to_aggregate))
          else:
            if not isinstance(lvar, ops.IndexedSlices):
              raise ValueError("Unknown model variable type!")
            lvar_accum = data_flow_ops.SparseConditionalAccumulator(
                lvar.dtype, shape=(),
                shared_name=var.name + "/model_variable_accum")
            train_ops.append(lvar_accum.apply_indexed_slices_grad(
                lvar, local_step=self._local_step))
            aggregated_lvars.append(lvar_accum.take_indexed_slices_grad(
                self._replicas_to_aggregate))

          self._accumulator_list.append((lvar_accum, var.device))

      aggregated_lvars_and_gvars = zip(aggregated_lvars, var_list)

      # sync_op will be assigned to the same device as the global step.
      with ops.device(global_step.device), ops.name_scope(""):
        for (avg_var, init_var), bg_var in zip(aggregated_lvars_and_gvars,
                                               block_grad_vars):
          gk_avg = math_ops.subtract(init_var, avg_var)
          gk_new = gk_avg
          block_grad = math_ops.multiply(self._block_learning_rate, gk_new)
          his_bg = math_ops.multiply(self._block_momentum_rate, bg_var)
          bg_new = math_ops.add(his_bg, block_grad)
          bg_reassign_ops.append(state_ops.assign(bg_var, bg_new))
        bg_op = control_flow_ops.group(*(bg_reassign_ops))
        with ops.control_dependencies([bg_op]):
          for global_var, bg_var in zip(global_vars, block_grad_vars):
            model_reassign_ops.append(state_ops.assign_sub(global_var, bg_var))
          g_update_op = control_flow_ops.group(*(model_reassign_ops))
        with ops.control_dependencies([g_update_op]):
          for n_var, global_var, bg_var in zip(nesterov_vars, global_vars,
                                               block_grad_vars):
            momentum = math_ops.multiply(self._block_momentum_rate, bg_var)
            nesterov = math_ops.subtract(global_var, momentum)
            nesterov_reassign_ops.append(state_ops.assign(n_var, nesterov))
          nesterov_reassign_ops.append(state_ops.assign_add(global_step, 1))
          update_op = control_flow_ops.group(*(nesterov_reassign_ops))


      self._local_step = self._local_step
      self._train_ops = train_ops
      # Create token queue.
      with ops.device(global_step.device), ops.name_scope(""):
        sync_token_queue = (
            data_flow_ops.FIFOQueue(-1,
                                    global_step.dtype.base_dtype,
                                    shapes=(),
                                    name="sync_token_q",
                                    shared_name="sync_token_q"))
        self._sync_token_queue = sync_token_queue

        # dummy_queue is passed to the queue runner. Don't use the real queues
        # because the queue runner doesn't automatically reopen it once it
        # closed queues in PS devices.
        dummy_queue = (
            data_flow_ops.FIFOQueue(1,
                                    types_pb2.DT_INT32,
                                    shapes=(),
                                    name="dummy_queue",
                                    shared_name="dummy_queue"))

      with ops.device(global_step.device), ops.name_scope(""):
        # Replicas have to wait until they can get a token from the token queue.
        with ops.control_dependencies(train_ops):
          token = sync_token_queue.dequeue()
        train_op = state_ops.assign(self._local_step, token)

        with ops.control_dependencies([update_op]):
          # Sync_op needs to insert tokens to the token queue at the end of the
          # step so the replicas can fetch them to start the next step.
          tokens = array_ops.fill([self._tokens_per_step], global_step)
          sync_op = sync_token_queue.enqueue_many((tokens,))

        self._chief_queue_runner = queue_runner.QueueRunner(dummy_queue,
                                                            [sync_op])
      for accum, dev in self._accumulator_list:
        with ops.device(dev):
          chief_init_ops.append(
              accum.set_global_step(
                  global_step, name="SetGlobalStep"))
      self.chief_init_op = control_flow_ops.group(*(chief_init_ops))
      self._gradients_applied = True
      return train_op

  def assign_vars(self, local_vars, global_vars):
    reassign_ops = []
    for local_var, global_var in zip(local_vars, global_vars):
      reassign_ops.append(state_ops.assign(local_var, global_var))
    refresh_ops = control_flow_ops.group(*(reassign_ops))
    return refresh_ops


  def get_chief_queue_runner(self):
    """Returns the QueueRunner for the chief to execute.

    This includes the operations to synchronize replicas: local weights,
    apply to variables, increment global step, insert tokens to token queue.

    Note that this can only be called after calling apply_gradients() which
    actually generates this queuerunner.

    Returns:
      A `QueueRunner` for chief to execute.

    Raises:
      ValueError: If this is called before apply_gradients().
    """
    if self._gradients_applied is False:
      raise ValueError("Should be called after apply_gradients().")

    return self._chief_queue_runner

  def get_init_tokens_op(self, num_tokens=0):
    """Returns the op to fill the sync_token_queue with the tokens.

    This is supposed to be executed in the beginning of the chief/sync thread
    so that even if the total_num_replicas is less than replicas_to_aggregate,
    the model can still proceed as the replicas can compute multiple steps per
    variable update. Make sure:
    `num_tokens >= replicas_to_aggregate - total_num_replicas`.

    Args:
      num_tokens: Number of tokens to add to the queue.

    Returns:
      An op for the chief/sync replica to fill the token queue.

    Raises:
      ValueError: If this is called before apply_gradients().
      ValueError: If num_tokens are smaller than replicas_to_aggregate -
        total_num_replicas.
    """
    if self._gradients_applied is False:
      raise ValueError(
          "get_init_tokens_op() should be called after apply_gradients().")

    tokens_needed = self._replicas_to_aggregate - self._total_num_replicas
    if num_tokens == -1:
      num_tokens = self._replicas_to_aggregate
    elif num_tokens < tokens_needed:
      raise ValueError(
          "Too few tokens to finish the first step: %d (given) vs %d (needed)" %
          (num_tokens, tokens_needed))

    if num_tokens > 0:
      with ops.device(self._global_step.device), ops.name_scope(""):
        tokens = array_ops.fill([num_tokens], self._global_step)
        init_tokens = self._sync_token_queue.enqueue_many((tokens,))
    else:
      init_tokens = control_flow_ops.no_op(name="no_init_tokens")

    return init_tokens

  def make_ma_run_hook(self):
    local_vars = variables.trainable_variables()
    global_vars = ops.get_collection_ref('global_model')
    block_grad_vars = ops.get_collection_ref('block_grad')
    nesterov_vars = ops.get_collection_ref('nesterov')

    if self._use_nesterov:
      self._refresh_local_vars_op = self.assign_vars(local_vars, nesterov_vars)
      local_and_init_vars = list(zip(local_vars, nesterov_vars))
      self._init_nesterov_vars_op = self.assign_vars(nesterov_vars, global_vars)
    else:
      self._refresh_local_vars_op = self.assign_vars(local_vars, global_vars)
      local_and_init_vars = list(zip(local_vars, global_vars))
      self._init_nesterov_vars_op = None

    self._apply_ma_op = self._apply_model_average(local_and_init_vars,
                                                  global_vars,
                                                  block_grad_vars,
                                                  nesterov_vars,
                                                  self._super_global_step)
    self._ma_run_hook = True

    return self._ModelAverageHook(self, self._refresh_local_vars_op,
                                  self._init_nesterov_vars_op,
                                  self._apply_ma_op,
                                  self._interval_steps,
                                  self._super_global_step)

  def make_session_run_hook(self, is_chief, num_tokens=0):
    """Creates a hook to handle ReplicasHook ops such as initialization."""
    if self._ma_run_hook is False:
      raise ValueError("make_session_run_hook Should be "
                       "called after make_ma_run_hook.")

    if is_chief:
      return self._ReplicasHook(self.chief_init_op,
                                self.ready_for_local_init_op,
                                self.get_chief_queue_runner(),
                                self.get_init_tokens_op(num_tokens))

    return self._ReplicasHook(self.local_step_init_op,
                              self.ready_for_local_init_op, None, None)

  class _ModelAverageHook(session_run_hook.SessionRunHook):
    def __init__(self, parent, refresh_local_vars_op, init_nesterov_vars_op,
                 apply_ma_op, interval_steps, super_global_step):
      self._refresh_local_vars_op = refresh_local_vars_op
      self._init_nesterov_vars_op = init_nesterov_vars_op
      self._apply_ma_op = apply_ma_op
      self._interval_steps = interval_steps
      self._super_global_step = super_global_step
      self._parent = parent

    def after_create_session(self, session, coord):
      # Initialized current iteration step.
      self._curr_iter = 0
      # pylint: disable=protected-access
      num_global_vars = self._parent._num_of_global_variables
      num_trainable_vars = self._parent._num_of_trainable_variables
      if num_global_vars != len(variables.global_variables()):
        raise AssertionError('cannot declare global variables after calling an '
                             'ModelAverageOptimizer! Please set '
                             'ModelAverageOptimizer at the last.')
      if num_trainable_vars != len(variables.trainable_variables()):
        raise AssertionError('cannot declare trainable variables after calling '
                             'an ModelAverageOptimizer! Please set '
                             'ModelAverageOptimizer at the last.')

    def before_run(self, run_context):
      if self._curr_iter == 0:
        session = run_context.session
        if self._init_nesterov_vars_op is not None:
          session.run(self._init_nesterov_vars_op)
        session.run(self._refresh_local_vars_op)

    def after_run(self, run_context, run_values):
      ''' Model Average Distributed Training '''
      session = run_context.session
      if (self._curr_iter + 1) % self._interval_steps == 0:
        # Apply model_average op before pulling.
        cur_time = time.time()
        _, super_global_step = session.run([self._apply_ma_op,
                                            self._super_global_step])
        elapsed_ma_time = time.time() - cur_time
        logging.info("Model Average %s: super_global_step: %d, _step:%d,"
                     "model average time: %.4f sec."
                     % (type(self).__name__, super_global_step,
                        self._curr_iter, elapsed_ma_time))
        # Pull new model params after model average op.
        _ = session.run(self._refresh_local_vars_op)
      self._curr_iter = self._curr_iter + 1

  class _ReplicasHook(session_run_hook.SessionRunHook):
    """A SessionRunHook handles ops related to ModelAverageOptimizer."""

    def __init__(self, local_init_op, ready_for_local_init_op, q_runner,
                 init_tokens_op):
      """Creates hook to handle ModelAverageOptimizer initialization ops.

      Args:
        local_init_op: Either `ModelAverageOptimizer.chief_init_op` or
          `ModelAverageOptimizer.local_step_init_op`.
        ready_for_local_init_op: `ModelAverageOptimizer.ready_for_local_init_op`
        q_runner: Either `ModelAverageOptimizer.get_chief_queue_runner` or
                  `None`
        init_tokens_op: `ModelAverageOptimizer.get_init_tokens_op` or None
      """
      self._local_init_op = local_init_op
      self._ready_for_local_init_op = ready_for_local_init_op
      self._q_runner = q_runner
      self._init_tokens_op = init_tokens_op

    def after_create_session(self, session, coord):
      """Runs ModelAverageOptimizer initialization ops."""
      # pylint: disable=protected-access
      local_init_success, msg = session_manager._ready(
          self._ready_for_local_init_op, session,
          "Model is not ready for ModelAverageOptimizer local init.")
      if not local_init_success:
        raise RuntimeError(
            "Init operations did not make model ready for "
            "ModelAverageOptimizer local_init. Init op: %s, error: %s" %
            (self._local_init_op.name, msg))
      session.run(self._local_init_op)
      if self._init_tokens_op is not None:
        session.run(self._init_tokens_op)
      if self._q_runner is not None:
        self._q_runner.create_threads(
            session, coord=coord, daemon=True, start=True)
