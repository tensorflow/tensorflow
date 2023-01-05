# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for saving/loading Trackable objects asynchronously."""

import atexit
import collections
import copy
import threading
import time
import weakref

from absl import logging

from tensorflow.python.distribute import device_util
from tensorflow.python.distribute.sharded_variable import ShardedVariable
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import executor
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import ops
from tensorflow.python.ops.resource_variable_ops import UninitializedVariable
from tensorflow.python.ops.variables import Variable
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.tpu.tpu_embedding_v2 import TPUEmbedding
from tensorflow.python.training import optimizer as optimizer_v1
from tensorflow.python.util import object_identity

# Captures the timestamp of the first Checkpoint instantiation or end of a write
# operation. Can be accessed by multiple Checkpoint instances.
_END_TIME_OF_LAST_ASYNC_WRITE = None
_END_TIME_OF_LAST_ASYNC_WRITE_LOCK = threading.Lock()

# API label for cell names used in async checkpoint metrics.
_ASYNC_CHECKPOINT = "async_checkpoint"


def _get_duration_microseconds(start_time_seconds, end_time_seconds):
  """Calculate the duration between start and end time.

  Args:
    start_time_seconds: The start time in seconds.
    end_time_seconds: The end time in seconds.

  Returns:
    The duration between the start and the end time. Return 0 if
    end_time_seconds < start_time_seconds.
  """
  if end_time_seconds < start_time_seconds:
    # Avoid returning negative value in case of clock skew.
    return 0
  return round((end_time_seconds - start_time_seconds) * 1000000)


class AsyncCheckpointHelper:
  """Helper class for async checkpoint."""

  def __init__(self, checkpointer_impl, root=None, **kwargs):
    """Initialize AsyncCheckpoint.

    Args:
      checkpointer_impl: The Checkpoint class to power the AsyncCheckpoint.
      root: The root object to checkpoint. `root` may be a trackable object or
        `WeakRef` of a trackable object.
      **kwargs: The keyword arguments representing the checkpointed variables.
    """
    # TODO(chienchunh): Make sure the processing for the root object is
    #   consistent when integrating with the public API, e.g., adding all kwarg
    #   items as the child of the root object.
    if root:
      trackable_root = root() if isinstance(root, weakref.ref) else root
      kwargs["root"] = trackable_root
      trackable_root._maybe_initialize_trackable()

    self._checkpointer_impl = checkpointer_impl
    self._checkpoint_items = kwargs

    # The underlying Checkpoint instance and its items.
    self._checkpoint = None
    self._checkpoint_options = None

    # The callback function that needs to be executed after checkpoint write.
    # Currently this is only applied to the scenario where CheckpointManager is
    # used, which triggers the _write() method.
    self._async_write_done_callback = None

    # The list of all nodes from the original checkpoint items.
    # TODO(chienchunh): Consider changing this to local variable.
    self._original_nodes = None
    # The mapping between the original and the copied resource variables.
    # The copied variables are used for the underlying checkpointing.
    self._object_map = None
    # A list of TPUEmbedding objects included in the checkpoint items.
    self._tpu_embedding_objects = None

    self._default_device = device_util.current() or "CPU:0"
    self._default_device = device_util.canonicalize(self._default_device)

    self._save_file_prefix = None
    self._use_checkpoint_save = False
    self._async_save_thread = None
    self._async_save_thread_shutdown = False
    # Semaphores for writing/reading the cpu-copied variables (self._var_pairs)
    # TODO(chienchunh): Consider Queue/Condition instead of Semaphore.
    self._writer_sem = threading.Semaphore(1)
    self._reader_sem = threading.Semaphore(0)

    # Register to join the async save thread upon exit.
    atexit.register(self._join_async_save_thread)

    global _END_TIME_OF_LAST_ASYNC_WRITE
    with _END_TIME_OF_LAST_ASYNC_WRITE_LOCK:
      if _END_TIME_OF_LAST_ASYNC_WRITE is None:
        _END_TIME_OF_LAST_ASYNC_WRITE = time.time()

  @def_function.function
  def _copy_from_cpu(self):
    """Copy the checkpointed variables from the host CPU to the accelerator.

    TODO(chienchunh): Get the concrete function before firstly called to avoid
                      hangining the accelerators idle during function tracing.
    """
    for accelerator_var, cpu_var in self._object_map.items():
      if isinstance(accelerator_var, (ShardedVariable, TPUEmbedding)):
        # Skip for SharededVariable and TPUEmbedding as their sub-variables will
        # be copied over separately through other entries in the object map.
        continue
      with ops.device(accelerator_var.device):
        accelerator_var.assign(cpu_var.read_value())

  @def_function.function
  def _copy_to_cpu(self):
    """Copy the checkpointed variables from the accelerator to the host CPU.

    TODO(chienchunh): Get the concrete function before firstly called to avoid
                      hangining the accelerators idle during function tracing.
    """
    for accelerator_var, cpu_var in self._object_map.items():
      if isinstance(accelerator_var, (ShardedVariable, TPUEmbedding)):
        # Skip for SharededVariable and TPUEmbedding as their sub-variables will
        # be copied over separately through other entries in the object map.
        continue
      with ops.device(cpu_var.device):
        cpu_var.assign(accelerator_var.read_value())
    for tpu_embedding in self._tpu_embedding_objects:
      tpu_embedding._retrieve_variables()  # pylint: disable=protected-access

  def _traverse_variables(self, to_traverse, visited):
    """Create the copied nodes and variables while traversing the nodes.

    This method performs a BFS to traverse the nodes while avoiding duplicated
    visits. Throughout the process, self._mapping, self._original_nodes, and
    self._var_pairs are populated.

    Args:
      to_traverse: A deque that stores the nodes to be traversed.
      visited: A list of nodes that have been visited.
    """
    # pylint: disable=protected-access
    while to_traverse:
      current_trackable = to_traverse.popleft()
      self._original_nodes.append(current_trackable)

      if isinstance(current_trackable, (Variable, ShardedVariable)):
        self._copy_trackable(current_trackable)
      if isinstance(current_trackable, TPUEmbedding):
        self._handle_tpu_embedding(current_trackable)

      for child in current_trackable._trackable_children().values():
        if child in visited:
          continue
        visited.add(child)
        to_traverse.append(child)
    # pylint: enable=protected-access

  def _ensure_initialized(self):
    """Initialize the async checkpoint internal state."""
    if self._checkpoint is not None:
      return

    self._original_nodes = []
    self._object_map = object_identity.ObjectIdentityDictionary()
    self._tpu_embedding_objects = []

    # Add the top-level checkpoint items to be traversed,
    to_traverse = collections.deque([])
    visited = object_identity.ObjectIdentitySet()
    for v in self._checkpoint_items.values():
      if isinstance(v, (Variable, ShardedVariable)):
        self._copy_trackable(v)
      elif isinstance(v, TPUEmbedding):
        self._handle_tpu_embedding(v)
      to_traverse.append(v)
      visited.add(v)
    self._traverse_variables(to_traverse, visited)

    # Copy for the slot variables.
    for current_trackable in self._original_nodes:
      if (isinstance(current_trackable, optimizer_v1.Optimizer)
          # Note: dir() is used rather than hasattr() here to avoid triggering
          # custom __getattr__ code, see b/152031870 for context.
          or "get_slot_names" in dir(current_trackable)):
        slot_names = current_trackable.get_slot_names()
        for slot_name in slot_names:
          for original_variable in self._original_nodes:
            if not isinstance(original_variable, Variable):
              continue
            try:
              original_slot_variable = current_trackable.get_slot(
                  original_variable, slot_name)
            except (AttributeError, KeyError):
              continue
            if isinstance(original_slot_variable, (Variable, ShardedVariable)):
              self._copy_trackable(original_slot_variable)

    # Initiate the underlying Checkpoint instance with the copied items.
    self._checkpoint = self._checkpointer_impl(**self._checkpoint_items)

    # Pass the object map of the copied variables to the underlying Checkpoint.
    self._checkpoint._saver._object_map = self._object_map  # pylint: disable=protected-access

    # Initiate the async thread for checkpoint saving.
    self._async_save_thread = threading.Thread(
        target=self._async_save, daemon=True)
    self._async_save_thread.start()

  def _join_async_save_thread(self):
    """Join the async save thread.

    The steps for terminating the async save thread:
    1). Wait until the last async save event is done.
    2). Set _async_save_thread_shutdown flag to false to indicate termination.
    3). Trigger the async save thread to check and fail the while-predicate.
    4). Join the async save thread. (The thread may finish before joining.)
    """
    if self._writer_sem.acquire(timeout=3600):  # Step-1.
      self._async_save_thread_shutdown = True  # Step-2.
      self._reader_sem.release()  # Step-3.
      logging.info("Joining the async save thread.")
      if self._async_save_thread is not None:
        self._async_save_thread.join()  # Step-4.
    else:
      logging.error("Timeout waiting for the async save thread; terminating the"
                    " thread instead. The last checkpoint may be incomeplete.")

  def _async_save(self):
    """The thread function for the async checkpoint save."""
    with context.executor_scope(
        executor.new_executor(
            enable_async=False, enable_streaming_enqueue=False)):
      while self._reader_sem.acquire() and not self._async_save_thread_shutdown:
        logging.info("Starting async checkpoint save on the device: %s",
                     self._default_device)

        async_save_start_time = time.time()

        # Specify the ops placement on the worker if running with
        # coordinator-worker mode. This is required as launching a new thread
        # would clear the placement policy and make localhost the default
        # placement, while the main thread's default placement would be the
        # master worker's CPU:0.
        with ops.device(self._default_device):
          if self._use_checkpoint_save:
            self._checkpoint.save(self._save_file_prefix,
                                  self._checkpoint_options)
          else:
            self._checkpoint._write(  # pylint: disable=protected-access
                self._save_file_prefix,
                options=self._checkpoint_options,
                write_done_callback=self._async_write_done_callback)
        # Allow the next checkpoint event to overwrite the cpu-copied variables.
        self._writer_sem.release()

        async_save_end_time = time.time()
        metrics.AddAsyncCheckpointWriteDuration(
            api_label=_ASYNC_CHECKPOINT,
            microseconds=_get_duration_microseconds(async_save_start_time,
                                                    async_save_end_time))

        # Measure the elapsed time since the last checkpoint.
        # Due to the nature of async checkpoint, here it actually captures the
        # duration between the start_time of the previous checkpoint and the
        # start time of this checkpoint. As a result, the duration of the final
        # async checkpoint is excluded, which is fine since it does not take
        # much time.
        global _END_TIME_OF_LAST_ASYNC_WRITE
        with _END_TIME_OF_LAST_ASYNC_WRITE_LOCK:
          metrics.AddTrainingTimeSaved(
              api_label=_ASYNC_CHECKPOINT,
              microseconds=_get_duration_microseconds(
                  _END_TIME_OF_LAST_ASYNC_WRITE, async_save_start_time))
          _END_TIME_OF_LAST_ASYNC_WRITE = async_save_start_time
    logging.info("Async save thread reached the end of the execution.")

  def _copy_for_variable(self, original_var):
    """Create a new instance for the input trackable.

    Args:
      original_var: Input Variable object to be copied.
    """
    op_device = pydev.DeviceSpec.from_string(original_var.device).replace(
        device_type="CPU", device_index=0).to_string()
    with ops.device(op_device):
      new_var = UninitializedVariable(
          trainable=original_var.trainable,
          shape=original_var.shape,
          dtype=original_var.dtype,
          name=original_var._shared_name)  # pylint: disable=protected-access
    self._object_map[original_var] = new_var

  def _copy_for_sharded_variable(self, original_var):
    """Create a new instance for the input ShardedVariable.

    Args:
      original_var: Input ShardedVariable object to be copied.
    """
    copied_vars = []
    for v in original_var._variables:  # pylint: disable=protected-access
      self._copy_for_variable(v)
      copied_vars.append(self._object_map[v])
    self._object_map[original_var] = ShardedVariable(
        copied_vars, name=original_var.name)

  def _copy_trackable(self, original_trackable):
    """Create a new instance for the input trackable.

    Args:
      original_trackable: The trackable instance to be copied.

    Raises:
      AttributeError: if the input trackable is not Variable or ShardedVariable.
    """
    if isinstance(original_trackable, ShardedVariable):
      self._copy_for_sharded_variable(original_trackable)
    elif isinstance(original_trackable, Variable):
      self._copy_for_variable(original_trackable)
    else:
      raise AttributeError("Only Variable or ShardedVariable can be copied.")

  def _handle_tpu_embedding(self, tpu_embedding):
    """Handle TPUEmbedding.

    Args:
      tpu_embedding: TPUEmbedding object to be handled.

    Raises:
      AttributeError: if the input trackable is not TPUEmbedding type.
    """
    if not isinstance(tpu_embedding, TPUEmbedding):
      raise AttributeError("Expecting TPUEmbedding type; got %s" %
                           type(tpu_embedding))

    # Create a dummy TPUEmbedding object and add it to the object_map. This is
    # to prevent the TPUEmbedding's save_callback from being triggered because
    # the embedding values have already being retrieved by AsyncCheckpoint.
    # pylint: disable=protected-access
    new_embedding = TPUEmbedding(
        feature_config=tpu_embedding._feature_config,
        optimizer=tpu_embedding._table_config[0].optimizer,
        pipeline_execution_with_tensor_core=tpu_embedding
        ._pipeline_execution_with_tensor_core)
    self._object_map[tpu_embedding] = new_embedding
    # pylint: enable=protected-access

    if tpu_embedding not in self._tpu_embedding_objects:
      self._tpu_embedding_objects.append(tpu_embedding)

  @property
  def save_counter(self):
    """An integer variable numbering the checkpoint events.

    This is maintained by the underlying tf.train.Checkpoing object employed by
    AsyncCheckpoint class. The number starts at 0 and gets incremented for each
    checkpoint event.

    Returns:
      The save counter variable.
    """
    self._ensure_initialized()
    return self._checkpoint.save_counter

  def write(self, save_path, options=None):
    """Save the checkpointed variables.

    Args:
      save_path: The file prefix of the checkpoint file.
      options: Optional CheckpointOption instance.

    Returns:
      The full path of the checkpoint file.
    """
    self._write(save_path, options)

  def _write(self, save_path, options=None, write_done_callback=None):
    """Save the checkpointed variables.

    This method has exactly the same logic as save(), except it does not
    increment the underlying save_counter, which is done by the caller, e.g.,
    CheckpointManager.

    Args:
      save_path: The file prefix of the checkpoint file.
      options: Optional CheckpointOption instance.
      write_done_callback: Optional callback function executed after the async
        write is done.

    Returns:
      The full path of the checkpoint file.
    """
    self._ensure_initialized()

    write_start_time = time.time()

    # Copy the variable values to the host CPU.
    if self._writer_sem.acquire():
      self._copy_to_cpu()

    # Trigger the async thread to checkpoint the cpu-copied variables.
    # Need to wait until the weight copying finishes before checkpoint save.
    context.async_wait()
    self._save_file_prefix = save_path
    self._use_checkpoint_save = False

    # Ensure that we do not request async checkpointing to the underlying
    # checkpointer as this could lead to an infinite loop.
    self._checkpoint_options = copy.copy(options) if options else None
    if self._checkpoint_options:
      self._checkpoint_options.experimental_enable_async_checkpoint = False

    self._async_write_done_callback = write_done_callback
    self._reader_sem.release()

    write_end_time = time.time()
    metrics.AddCheckpointWriteDuration(
        api_label=_ASYNC_CHECKPOINT,
        microseconds=_get_duration_microseconds(write_start_time,
                                                write_end_time))

    return save_path

  def save(self, save_path, options=None):
    """Save the checkpointed variables.

    Args:
      save_path: The file prefix of the checkpoint file.
      options: Optional CheckpointOption instance.

    Returns:
      The full path of the checkpoint file.
    """
    # If this is the first time that AsyncCheckpoint.save() is called,
    # initialize the cpu-copied variables and create the pair-wise mapping
    # between the original model variables and the cpu-copied variables.
    #
    # This is not performed in the initializer because some variables, e.g.,
    # slot variables of the optimizer, were not created until actually running
    # the train function, so we could only get the complete list of the
    # variables after some train steps were run.
    self._ensure_initialized()

    save_start_time = time.time()

    # Copy the variable values to the host CPU.
    if self._writer_sem.acquire():
      self._copy_to_cpu()

    # Retrieve the save counter from the underlying checkpoint object to
    # re-construct the full path of the checkpoint file.
    # This step has to happen before triggerting the underlying checkpoint;
    # otherwise, the save_counter value may or may not have been updated.
    save_counter = self._checkpoint.save_counter.numpy() + 1
    full_path = "{}-{}".format(save_path, save_counter)

    # Trigger the async thread to checkpoint the cpu-copied variables.
    # Need to wait until the weight copying finishes before checkpoint save.
    context.async_wait()
    self._save_file_prefix = save_path
    self._use_checkpoint_save = True

    # Ensure that we do not request async checkpointing to the underlying
    # checkpointer as this could lead to an infinite loop.
    self._checkpoint_options = copy.copy(options) if options else None
    if self._checkpoint_options:
      self._checkpoint_options.experimental_enable_async_checkpoint = False

    self._reader_sem.release()

    save_end_time = time.time()
    metrics.AddCheckpointWriteDuration(
        api_label=_ASYNC_CHECKPOINT,
        microseconds=_get_duration_microseconds(save_start_time, save_end_time))

    return full_path

  def read(self, save_path, options=None):
    """Restore the checkpointed variables.

    This method has exactly the same logic as restore(). This method is
    implemented only to fulfill the duty of subclassing tf.train.Checkpoint.

    Args:
      save_path: The full name of the checkpoint file to be restored.
      options: CheckpointOption instance.

    Returns:
      A load status object, which can be used to make assertions about the
      status of a checkpoint restoration. See tf.train.Checkpoint.restore()
      for more details.
    """
    return self.restore(save_path, options)

  def restore(self, save_path, options=None):
    """Restore the checkpointed variables.

    Args:
      save_path: The full name of the checkpoint file to be restored.
      options: CheckpointOption instance.

    Returns:
      A load status object, which can be used to make assertions about the
      status of a checkpoint restoration. See tf.train.Checkpoint.restore()
      for more details.
    """
    # Ensure that we do not request async checkpointing to the underlying
    # checkpointer as this could lead to an infinite loop.
    self._checkpoint_options = (
        copy.copy(options) if options else self._checkpoint_options)
    if self._checkpoint_options:
      self._checkpoint_options.experimental_enable_async_checkpoint = False

    # Wait for any ongoing checkpoint event to finish.
    with self._writer_sem:
      # If _checkpoint has not been initialized yet, it means the restore() is
      # called right after the coordinator is restarted. We directly restore
      # the checkpointed items through tf.train.Checkpoint.restore().
      if self._checkpoint is None:
        tmp_checkpoint = self._checkpointer_impl(**self._checkpoint_items)
        return tmp_checkpoint.restore(save_path, self._checkpoint_options)

      # Restore the values of the cpu-copied variables.
      status = self._checkpoint.restore(save_path, self._checkpoint_options)

      # Restore the values of the original model.
      self._copy_from_cpu()
      return status

  def sync(self):
    """Sync on any ongoing save or restore events."""
    with self._writer_sem:
      logging.info("Sync on ongoing save/restore.")
