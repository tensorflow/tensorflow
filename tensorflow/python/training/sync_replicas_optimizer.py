# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Synchronize replicas for training."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import optimizer
from tensorflow.python.training import queue_runner


class SyncReplicasOptimizer(optimizer.Optimizer):
  """Class to synchronize, aggregate gradients and pass them to the optimizer.

  In a typical asynchronous training environment, it's common to have some
  stale gradients. For example, with a N-replica asynchronous training,
  gradients will be applied to the variables N times independently. Depending
  on each replica's training speed, some gradients might be calculated from
  copies of the variable from several steps back (N-1 steps on average). This
  optimizer avoids stale gradients by collecting gradients from all replicas,
  summing them, then applying them to the variables in one shot, after
  which replicas can fetch the new variables and continue.

  The following queues are created:
  <empty line>
  * N `gradient` queues, one per variable to train. Gradients are pushed to
    these queues and the chief worker will dequeue_many and then sum them
    before applying to variables.
  * 1 `token` queue where the optimizer pushes the new global_step value after
    all gradients have been applied.

  The following variables are created:
  * N `local_step`, one per replica. Compared against global step to check for
    staleness of the gradients.

  This adds nodes to the graph to collect gradients and pause the trainers until
  variables are updated.
  For the PS:
  <empty line>
  1. A queue is created for each variable, and each replica now pushes the
    gradients into the queue instead of directly applying them to the
    variables.
  2. For each gradient_queue, pop and sum the gradients once enough
    replicas (replicas_to_aggregate) have pushed gradients to the queue.
  3. Apply the aggregated gradients to the variables.
  4. Only after all variables have been updated, increment the global step.
  5. Only after step 4, clear all the gradients in the queues as they are
    stale now (could happen when replicas are restarted and push to the queues
    multiple times, or from the backup replicas).
  6. Only after step 5, pushes `global_step` in the `token_queue`, once for
    each worker replica. The workers can now fetch it to its local_step variable
    and start the next batch.

  For the replicas:
  <empty line>
  1. Start a step: fetch variables and compute gradients.
  2. Once the gradients have been computed, push them into `gradient_queue` only
    if local_step equals global_step, otherwise the gradients are just dropped.
    This avoids stale gradients.
  3. After pushing all the gradients, dequeue an updated value of global_step
    from the token queue and record that step to its local_step variable. Note
    that this is effectively a barrier.
  4. Start the next batch.

  ### Usage

  ```python
  # Create any optimizer to update the variables, say a simple SGD:
  opt = GradientDescentOptimizer(learning_rate=0.1)

  # Wrap the optimizer with sync_replicas_optimizer with 50 replicas: at each
  # step the optimizer collects 50 gradients before applying to variables.
  opt = tf.SyncReplicasOptimizer(opt, replicas_to_aggregate=50,
            replica_id=task_id, total_num_replicas=50)
  # Note that if you want to have 2 backup replicas, you can change
  # total_num_replicas=52 and make sure this number matches how many physical
  # replicas you started in your job.

  # Some models have startup_delays to help stablize the model but when using
  # sync_replicas training, set it to 0.

  # Now you can call `minimize()` or `compute_gradients()` and
  # `apply_gradients()` normally
  grads = opt.minimize(total_loss, global_step=self.global_step)


  # You can now call get_init_tokens_op() and get_chief_queue_runner().
  # Note that get_init_tokens_op() must be called before creating session
  # because it modifies the graph.
  init_token_op = opt.get_init_tokens_op()
  chief_queue_runner = opt.get_chief_queue_runner()
  ```

  In the training program, every worker will run the train_op as if not
  synchronized. But one worker (usually the chief) will need to execute the
  chief_queue_runner and get_init_tokens_op generated from this optimizer.

  ```python
  # After the session is created by the superviser and before the main while
  # loop:
  if is_chief and FLAGS.sync_replicas:
    sv.start_queue_runners(sess, [chief_queue_runner])
    # Insert initial tokens to the queue.
    sess.run(init_token_op)
  ```

  @@__init__
  @@compute_gradients
  @@apply_gradients
  @@get_chief_queue_runner
  @@get_init_tokens_op
  """

  def __init__(self,
               opt,
               replicas_to_aggregate,
               variable_averages=None,
               variables_to_average=None,
               replica_id=None,
               total_num_replicas=0,
               use_locking=False,
               name="sync_replicas"):
    """Construct a sync_replicas optimizer.

    Args:
      opt: The actual optimizer that will be used to compute and apply the
        gradients. Must be one of the Optimizer classes.
      replicas_to_aggregate: number of replicas to aggregate for each variable
        update.
      variable_averages: Optional `ExponentialMovingAverage` object, used to
        maintain moving averages for the variables passed in
        `variables_to_average`.
      variables_to_average: a list of variables that need to be averaged. Only
        needed if variable_averages is passed in.
      replica_id: This is the task/worker/replica ID. Needed as index to access
        local_steps to check staleness. Must be in the interval:
        [0, total_num_replicas)
      total_num_replicas: Total number of tasks/workers/replicas, could be
        different from replicas_to_aggregate.
        If total_num_replicas > replicas_to_aggregate: it is backup_replicas +
        replicas_to_aggregate.
        If total_num_replicas < replicas_to_aggregate: Replicas compute
        multiple batches per update to variables.
      use_locking: If True use locks for update operation.
      name: string. Optional name of the returned operation.
    """
    if total_num_replicas == 0:
      total_num_replicas = replicas_to_aggregate

    super(SyncReplicasOptimizer, self).__init__(use_locking, name)
    logging.info(
        "SyncReplicas enabled: replicas_to_aggregate=%s; total_num_replicas=%s",
        replicas_to_aggregate, total_num_replicas)
    self._opt = opt
    self._replicas_to_aggregate = replicas_to_aggregate
    self._gradients_applied = False
    self._variable_averages = variable_averages
    self._variables_to_average = variables_to_average
    self._replica_id = replica_id
    self._total_num_replicas = total_num_replicas
    self._tokens_per_step = max(total_num_replicas, replicas_to_aggregate)
    self._global_step = None
    self._sync_token_queue = None

    # This will be executed in a queue runner and includes the synchronization
    # operations done by the chief.
    self._chief_queue_runner = None

    # Remember which queue is on which device for the "clear" operation.
    # This list contains list of the following format: (grad_queue, device).
    self._one_element_queue_list = []
    # Sparse gradients queue has both value and index
    self._sparse_grad_queues_and_devs = []

    # clean_up_op will be executed when the chief is about to restart.
    # If chief restarts, it is possible that some variables have already been
    # updated before and when chief comes back, these variables will not be
    # updated again as the workers have already computed the gradients for
    # them.
    # But chief still waits for all variables to be updated, which will hang
    # the training.
    # To avoid such hang, every time the chief is about to die, it will call
    # abort_op to kill the PS with the token_queue so all replicas will also
    # restart.
    # TODO(jmchen): When training restarts, the variables are restored from the
    # previous checkpoint. As such all the gradients in all the queues should be
    # removed as they are computed from potentially different variables.
    # Currently this is not done.
    self._clean_up_op = None

  def compute_gradients(self, *args, **kwargs):
    """Compute gradients of "loss" for the variables in "var_list".

    This simply wraps the compute_gradients() from the real optimizer. The
    gradients will be aggregated in the apply_gradients() so that user can
    modify the gradients like clipping with per replica global norm if needed.
    The global norm with aggregated gradients can be bad as one replica's huge
    gradients can hurt the gradients from other replicas.

    Args:
      *args: Arguments for compute_gradients().
      **kwargs: Keyword arguments for compute_gradients().

    Returns:
      A list of (gradient, variable) pairs.
    """
    return self._opt.compute_gradients(*args, **kwargs)

  def _aggregate_sparse_grad(self, grad, var, train_ops):
    """Aggregate sparse gradients.

    Args:
      grad: The sparse gradient to aggregate.
      var: The variable to apply this gradient to.
      train_ops: The train_ops for the worker to run.

    Returns:
      aggregated_grad: Aggregated grad.
    """
    # Sparse gradients have to be inserted as one pair of (value,
    # indice) as an element instead of the whole "indexedslice" because
    # their shapes are not deterministic.
    sparse_grad_queue = (data_flow_ops.FIFOQueue(
        -1,
        (grad.values.dtype, grad.indices.dtype),
        shapes=(var.get_shape().as_list()[1:], ()),
        shared_name="sparse_grad_q_%s" % var.name))
    self._sparse_grad_queues_and_devs.append((sparse_grad_queue, var.device))

    # Sparse token is inserted after the "enqueue_many" finishes. This
    # is needed to make sure enough sparse gradients have been enqueued
    # before applying them to the variables.
    sparse_token_queue = (data_flow_ops.FIFOQueue(
        self._replicas_to_aggregate * 2,
        types_pb2.DT_INT32,
        shapes=(),
        shared_name="sparse_token_q_%s" % var.name))
    self._one_element_queue_list.append((sparse_token_queue, var.device))

    enqueue_spares_op = sparse_grad_queue.enqueue_many([grad.values,
                                                        grad.indices])
    with ops.control_dependencies([enqueue_spares_op]):
      train_ops.append(sparse_token_queue.enqueue((1,)))

    with ops.control_dependencies([sparse_token_queue.dequeue_many(
        self._replicas_to_aggregate)]):
      values, indices = sparse_grad_queue.dequeue_many(sparse_grad_queue.size())
      concat_grad = ops.IndexedSlices(values, indices, grad.dense_shape)

      # Sum the gradients of the same variables in the sparse layers so
      # that each variable is only updated once. Note that with 2
      # gradients g1 and g2 from 2 replicas for the same variable,
      # apply(g1+g2) is different from apply(g1) and then apply(g2) when
      # the optimizer is complex like Momentum or Adagrad.
      values = concat_grad.values
      indices = concat_grad.indices
      new_indices, indx = array_ops.unique(indices)
      num_indices = array_ops.shape(new_indices)[0]
      sum_values = math_ops.unsorted_segment_sum(values, indx, num_indices)
      return ops.IndexedSlices(sum_values, new_indices, concat_grad.dense_shape)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """Apply gradients to variables.

    This contains most of the synchronization implementation and also wraps the
    apply_gradients() from the real optimizer.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        compute_gradients().
      global_step: Optional Variable to increment by one after the
        variables have been updated.
      name: Optional name for the returned operation.  Default to the
        name passed to the Optimizer constructor.

    Returns:
      train_op: The op to dequeue a token so the replicas can exit this batch
      and start the next one. This is executed by each replica.

    Raises:
      ValueError: If the grads_and_vars is empty.
      ValueError: If global step is not provided, the staleness cannot be
        checked.
    """
    if not grads_and_vars:
      raise ValueError("Must supply at least one variable")

    if global_step is None:
      raise ValueError("Global step is required to check staleness")

    self._global_step = global_step
    train_ops = []
    aggregated_grad = []
    inputs = []
    var_list = []
    for x in grads_and_vars:
      inputs.extend(list(x))

    with ops.device(global_step.device):
      self._local_steps = variables.Variable(
          array_ops.zeros(
              [self._total_num_replicas],
              dtype=global_step.dtype),
          trainable=False,
          name="local_steps")

    # Check staleness. Note that this has to be ref(), otherwise identity will
    # be accessed and it will be old values.
    local_step = array_ops.slice(self._local_steps.ref(),
                                 array_ops.reshape(self._replica_id, (1,)),
                                 [1],
                                 name="get_local_step")
    local_step = array_ops.reshape(local_step, ())
    is_stale = math_ops.less(local_step, global_step)

    with ops.op_scope(inputs, None, self._name):
      for grad, var in grads_and_vars:
        var_list.append(var)
        with ops.device(var.device):
          if isinstance(grad, ops.Tensor):
            gradient_queue = (data_flow_ops.FIFOQueue(self._tokens_per_step * 2,
                                                      grad.dtype,
                                                      shapes=var.get_shape(),
                                                      shared_name=var.name))
            self._one_element_queue_list.append((gradient_queue, var.device))
            train_ops.append(gradient_queue.enqueue([grad]))

            # Aggregate all gradients
            gradients = gradient_queue.dequeue_many(
                self._replicas_to_aggregate)
            aggregated_grad.append(math_ops.reduce_sum(gradients, [0]))
          elif grad is None:
            aggregated_grad.append(None)  # pass-through.
          else:
            if not isinstance(grad, ops.IndexedSlices):
              raise ValueError("Unknown grad type!")
            aggregated_grad.append(self._aggregate_sparse_grad(grad, var,
                                                               train_ops))

      aggregated_grads_and_vars = zip(aggregated_grad, var_list)

      # sync_op will be assigned to the same device as the global step.
      with ops.device(global_step.device), ops.name_scope(""):
        update_op = self._opt.apply_gradients(aggregated_grads_and_vars,
                                              global_step)

      # Create token queue.
      with ops.device(global_step.device), ops.name_scope(""):
        sync_token_queue = (
            data_flow_ops.FIFOQueue(-1,
                                    global_step.dtype.base_dtype,
                                    shapes=(),
                                    shared_name="sync_token_q"))
        self._sync_token_queue = sync_token_queue

        # dummy_queue is passed to the queue runner. Don't use the real queues
        # because the queue runner doesn't automatically reopen it once it
        # closed queues in PS devices.
        dummy_queue = (
            data_flow_ops.FIFOQueue(1,
                                    types_pb2.DT_INT32,
                                    shapes=(),
                                    shared_name="dummy_queue"))
      # Clear all the gradients queues in case there are stale gradients.
      clear_queue_ops = []
      with ops.control_dependencies([update_op]):
        for queue, dev in self._one_element_queue_list:
          with ops.device(dev):
            stale_grads = queue.dequeue_many(queue.size())
            clear_queue_ops.append(stale_grads)

        for queue, dev in self._sparse_grad_queues_and_devs:
          with ops.device(dev):
            _, stale_indices = queue.dequeue_many(queue.size())
            clear_queue_ops.append(stale_indices)

      with ops.device(global_step.device):
        self._clean_up_op = control_flow_ops.abort(
            error_msg="From sync_replicas")

      # According to the staleness, select between the enqueue op (real_grad)
      # or no-op (no_op_grad). Effectively dropping all the stale gradients.
      no_op_grad = lambda: [control_flow_ops.no_op(name="no_grad_enqueue")]
      real_grad = lambda: [control_flow_ops.group(*train_ops)]
      final_train_ops = control_flow_ops.cond(is_stale, no_op_grad, real_grad)

      with ops.device(global_step.device), ops.name_scope(""):
        # Replicas have to wait until they can get a token from the token queue.
        with ops.control_dependencies([final_train_ops]):
          token = sync_token_queue.dequeue()
          train_op = state_ops.scatter_update(self._local_steps,
                                              self._replica_id, token)

        with ops.control_dependencies(clear_queue_ops):
          # Sync_op needs to insert tokens to the token queue at the end of the
          # step so the replicas can fetch them to start the next step.
          # Note that ref() is used to avoid reading from the identity with old
          # the step.
          tokens = array_ops.fill([self._tokens_per_step], global_step.ref())
          sync_op = sync_token_queue.enqueue_many((tokens,))

        if self._variable_averages is not None:
          with ops.control_dependencies([sync_op]), ops.name_scope(""):
            sync_op = self._variable_averages.apply(
                self._variables_to_average)

        self._chief_queue_runner = queue_runner.QueueRunner(dummy_queue,
                                                            [sync_op])
        self._gradients_applied = True
        return train_op

  def get_chief_queue_runner(self):
    """Returns the QueueRunner for the chief to execute.

    This includes the operations to synchronize replicas: aggregate gradients,
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

  def get_slot(self, *args, **kwargs):
    """Return a slot named "name" created for "var" by the Optimizer.

    This simply wraps the get_slot() from the actual optimizer.

    Args:
      *args: Arguments for get_slot().
      **kwargs: Keyword arguments for get_slot().

    Returns:
      The `Variable` for the slot if it was created, `None` otherwise.
    """
    return self._opt.get_slot(*args, **kwargs)

  def get_slot_names(self, *args, **kwargs):
    """Return a list of the names of slots created by the `Optimizer`.

    This simply wraps the get_slot_names() from the actual optimizer.

    Args:
      *args: Arguments for get_slot().
      **kwargs: Keyword arguments for get_slot().

    Returns:
      A list of strings.
    """
    return self._opt.get_slot_names(*args, **kwargs)

  def get_clean_up_op(self):
    """Returns the clean up op for the chief to execute before exit.

    This includes the operation to abort the device with the token queue so all
    other replicas can also restart. This can avoid potential hang when chief
    restarts.

    Note that this can only be called after calling apply_gradients().

    Returns:
      A clean_up_op for chief to execute before exits.

    Raises:
      ValueError: If this is called before apply_gradients().
    """
    if self._gradients_applied is False:
      raise ValueError(
          "get_clean_up_op() should be called after apply_gradients().")

    return self._clean_up_op

  def get_init_tokens_op(self, num_tokens=-1):
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
        tokens = array_ops.fill([num_tokens],
                                self._global_step.ref())
        init_tokens = self._sync_token_queue.enqueue_many((tokens,))
    else:
      init_tokens = control_flow_ops.no_op(name="no_init_tokens")

    return init_tokens
