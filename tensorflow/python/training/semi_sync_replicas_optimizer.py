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
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import optimizer
from tensorflow.python.training import queue_runner
from tensorflow.python.training import session_manager
from tensorflow.python.training import session_run_hook

class SemiSyncReplicasOptimizer(optimizer.Optimizer):

  def __init__(self,
               opt,
               batch_sync_num,
               use_locking=False,
               name="semi_sync_replicas"):

    super(SemiSyncReplicasOptimizer, self).__init__(use_locking, name)
    logging.info(
        "SemiSyncReplicasOptimizer: batch_sync_num=%s",
        batch_sync_num)
    self._opt = opt
    self._batch_sync_num = batch_sync_num
    self._gradients_applied = False
    self._global_step = None
    self._sync_token_queue = None
    self._reverse_sync_token_queue = None

    # The synchronization op will be executed in a queue runner which should
    # only be executed by one of the replicas (usually the chief).
    self._chief_queue_runner = None


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

    self._local_step = variables.Variable(
        initial_value=0,
        trainable=False,
        collections=[ops.GraphKeys.LOCAL_VARIABLES],
        dtype=global_step.dtype.base_dtype,
        name="sync_rep_local_step")
    self.local_step_init_op = state_ops.assign(self._local_step, global_step)
    chief_init_ops = [self.local_step_init_op]
    self.ready_for_local_init_op = variables.report_uninitialized_variables(
        variables.global_variables())

    with ops.name_scope(None, self._name):
      # sync_op will be assigned to the same device as the global step.
      # Create token queue.
      with ops.device(global_step.device), ops.name_scope(""):
        sync_token_queue = (
            data_flow_ops.FIFOQueue(-1,
                                    global_step.dtype.base_dtype,
                                    shapes=(),
                                    name="sync_token_q",
                                    shared_name="sync_token_q"))
        self._sync_token_queue = sync_token_queue

        reverse_sync_token_queue = (
            data_flow_ops.FIFOQueue(-1,
                                    global_step.dtype.base_dtype,
                                    shapes=(),
                                    name="reverse_sync_token_q",
                                    shared_name="reverse_sync_token_q"))
        self._reverse_sync_token_queue = reverse_sync_token_queue
        # dummy_queue is passed to the queue runner. Don't use the real queues
        # because the queue runner doesn't automatically reopen it once it
        # closed queues in PS devices.
        dummy_queue = (
            data_flow_ops.FIFOQueue(1,
                                    types_pb2.DT_INT32,
                                    shapes=(),
                                    name="dummy_queue",
                                    shared_name="dummy_queue"))
      train_ops=[]
      with ops.device(global_step.device), ops.name_scope(""):
        # Replicas have to wait until they can get a token from the token queue.
        token = sync_token_queue.dequeue()
        local_step_assign_op = state_ops.assign(self._local_step, token)

      with ops.control_dependencies([local_step_assign_op]):
        update_op = self._opt.apply_gradients(grads_and_vars, global_step=global_step)

      with ops.device(global_step.device), ops.name_scope(""):
        with ops.control_dependencies([update_op]):
          rsync_op = reverse_sync_token_queue.enqueue(global_step)

        train_ops.append(rsync_op)

        # Sync_op needs to insert tokens to the token queue at the end of the
        # step so the replicas can fetch them to start the next step.
        #tokens = array_ops.fill([self._batch_sync_num], global_step)
        tokens = reverse_sync_token_queue.dequeue_many(self._batch_sync_num)
        sync_op = sync_token_queue.enqueue_many((tokens,))

        self._chief_queue_runner = queue_runner.QueueRunner(dummy_queue,
                                                            [sync_op])
      self.chief_init_op = control_flow_ops.group(*(chief_init_ops))
      train_op = control_flow_ops.group(*(train_ops))
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

  def get_init_tokens_op(self, num_tokens=-1):
    """Returns the op to fill the sync_token_queue with the tokens.

    This is supposed to be executed in the beginning of the chief/sync thread
    to provide the initial tokens

    Args:
      num_tokens: Number of tokens to add to the queue.

    Returns:
      An op for the chief/sync replica to fill the token queue.

    Raises:
      ValueError: If this is called before apply_gradients().
      ValueError: If num_tokens are smaller than batch_sync_num -
        total_num_replicas.
    """
    if self._gradients_applied is False:
      raise ValueError(
          "get_init_tokens_op() should be called after apply_gradients().")

    if num_tokens == -1:
      num_tokens = self._batch_sync_num

    if num_tokens > 0:
      with ops.device(self._global_step.device), ops.name_scope(""):
        tokens = array_ops.fill([num_tokens], self._global_step)
        init_tokens = self._sync_token_queue.enqueue_many((tokens,))
    else:
      init_tokens = control_flow_ops.no_op(name="no_init_tokens")

    return init_tokens

  def make_session_run_hook(self, is_chief, num_tokens=-1):
    """Creates a hook to handle SyncReplicasHook ops such as initialization."""
    return _SemiSyncReplicasOptimizerHook(self, is_chief, num_tokens)


class _SemiSyncReplicasOptimizerHook(session_run_hook.SessionRunHook):
  """A SessionRunHook handles ops related to SyncReplicasOptimizer."""

  def __init__(self, semi_sync_optimizer, is_chief, num_tokens):
    """Creates hook to handle SyncReplicaOptimizer initialization ops.

    Args:
      sync_optimizer: `SemiSyncReplicasOptimizer` which this hook will initialize.
      is_chief: `Bool`, whether is this a chief replica or not.
      num_tokens: Number of tokens to add to the queue.
    """
    self._semi_sync_optimizer = semi_sync_optimizer
    self._is_chief = is_chief
    self._num_tokens = num_tokens

  def begin(self):
    if self._semi_sync_optimizer._gradients_applied is False:  # pylint: disable=protected-access
      raise ValueError(
          "SemiSyncReplicasOptimizer.apply_gradient should be called before using "
          "the hook.")
    if self._is_chief:
      self._local_init_op = self._semi_sync_optimizer.chief_init_op
      self._ready_for_local_init_op = (
          self._semi_sync_optimizer.ready_for_local_init_op)
      self._q_runner = self._semi_sync_optimizer.get_chief_queue_runner()
      self._init_tokens_op = self._semi_sync_optimizer.get_init_tokens_op(
          self._num_tokens)
    else:
      self._local_init_op = self._semi_sync_optimizer.local_step_init_op
      self._ready_for_local_init_op = (
          self._semi_sync_optimizer.ready_for_local_init_op)
      self._q_runner = None
      self._init_tokens_op = None

  def after_create_session(self, session, coord):
    """Runs SemiSyncReplicasOptimizer initialization ops."""
    local_init_success, msg = session_manager._ready(  # pylint: disable=protected-access
        self._ready_for_local_init_op, session,
        "Model is not ready for SemiSyncReplicasOptimizer local init.")
    if not local_init_success:
      raise RuntimeError(
          "Init operations did not make model ready for SemiSyncReplicasOptimizer "
          "local_init. Init op: %s, error: %s" %
          (self._local_init_op.name, msg))
    session.run(self._local_init_op)
    if self._init_tokens_op is not None:
      session.run(self._init_tokens_op)
    if self._q_runner is not None:
      self._q_runner.create_threads(
          session, coord=coord, daemon=True, start=True)


