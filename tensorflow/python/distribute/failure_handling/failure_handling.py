# Lint as: python3
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Module for `CoordinatedCheckpointManager`.

This is currently under development and the API is subject to change.
"""
import os
import signal
import sys
import threading
import time

from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute.failure_handling import gce_util
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_management

_PREEMPTION_KEY = 'TERMINATED_WORKER'
_RUN_COUNT_KEY = 'RUN_TO_CHECKPOINT'
_ACKNOWLEDGE_KEY = 'RECEIVED_SIGNAL'
_ITERATION_VARIABLE = 'checkpointed_runs'
# TODO(wxinyi): Move the hardcoded 42 to an internal module.
_RESTARTABLE_EXIT_CODE = 42
_STOP_WATCHING_CLUSTER_VALUE = 'STOP_WATCHER'


def _mwms_write_checkpoint_dir(checkpoint_dir, task_type, task_id,
                               cluster_spec):
  """Returns checkpoint_dir for chief and a temp dir for any other worker."""
  dirpath = os.path.dirname(checkpoint_dir)
  base = os.path.basename(checkpoint_dir)
  if not multi_worker_util.is_chief(
      cluster_spec=cluster_spec, task_type=task_type, task_id=task_id):
    base_dirpath = 'workertemp_' + str(task_id)
    dirpath = os.path.join(dirpath, base_dirpath)
    gfile.MakeDirs(dirpath)
  return os.path.join(dirpath, base)


class CoordinatedCheckpointManager(object):
  """Preemption and error handler for synchronous training.

  The API helps coordinate all workers to save a checkpoint upon receiving a
  preemption signal and helps propagate accurate error messages during training.
  When the program recovers from preemption, the checkpoint that is passed to
  initialize a `CoordinatedCheckpointManager` object will be loaded
  automatically.

  Right after the initialization, a thread starts to watch out for a preemption
  signal for any member in the cluster, but the signal will only be handled
  (which includes aligning the step to save a checkpoint, saving a checkpoint,
  and exiting with a platform recognized restart code) after entering a
  `CoordinatedCheckpointManager.run` call.

  Example usage:
  ```python
  strategy = tf.distribute.MultiWorkerMirroredStrategy()

  with strategy.scope():
    dataset, model, optimizer = ...

    fh_checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    coordinated_checkpoint_manager = tf.distribute.CoordinatedCheckpointManager(
        cluster_resolver, fh_checkpoint, checkpoint_directory)


    # `coordinated_checkpoint_manager.total_runs` will be restored to its
    # checkpointed value when training is restored after interruption.
    for epoch in range(coordinated_checkpoint_manager.total_runs //
                       STEPS_PER_EPOCH, num_epochs):
      for step in range(coordinated_checkpoint_manager.total_runs %
                        STEPS_PER_EPOCH, num_steps):
        # distributed_train_step is a function wrapped by strategy.run
        loss += coordinated_checkpoint_manager.run(distributed_train_step,
                                                   args=(next(dataset),))
  ```

  `CoordinatedCheckpointManager` will create a CheckpointManager to manage the
  checkpoint and only one CheckpointManager should be active in a particular
  directory at a time. Thus, if the user would like to save a checkpoint for
  purpose other than fault tolerance, e.g., for evaluation, they should save it
  in a directory different from the one passed to a
  `CoordinatedCheckpointManager`.

  This API targets multi-client distributed training, and right now only
  `tf.distribute.MultiWorkerMirroredStrategy` is supported.
  """

  def __init__(self, cluster_resolver, checkpoint, checkpoint_dir):
    """Creates the failure handler.

    Args:
      cluster_resolver: a `tf.distribute.cluster_resolver.ClusterResolver`. You
        may also get it through the `cluster_resolver` attribute of the
        strategy in use.
      checkpoint: a `tf.train.Checkpoint` that will be saved upon preemption and
        loaded upon restart by the `CoordinatedCheckpointManager` API
        automatically.
      checkpoint_dir: a directory for the `CoordinatedCheckpointManager` to play
        with checkpoints. `CoordinatedCheckpointManager` will create a
        `tf.train.CheckpointManager` to manage the passed-in `checkpoint`. Since
        only one `tf.train.CheckpointManager` should be active in a particular
        directory at a time, this `checkpoint_dir` arg should preferably be
        separated from where the user saves their checkpoint for non-fault
        tolerance purpose.
    """
    self._cluster_resolver = cluster_resolver
    self._checkpoint = checkpoint
    self._id_in_cluster = str(
        multi_worker_util.id_in_cluster(
            self._cluster_resolver.cluster_spec(),
            self._cluster_resolver.task_type,
            self._cluster_resolver.task_id))

    # The number of calls to `CoordinatedCheckpointManager.run` when the latest
    # checkpoint was saved.
    self._checkpointed_runs = variables.Variable(
        initial_value=constant_op.constant(0, dtype=dtypes.int64),
        trainable=False,
        name=_ITERATION_VARIABLE)
    if not hasattr(self._checkpoint,
                   _ITERATION_VARIABLE):
      setattr(self._checkpoint, _ITERATION_VARIABLE,
              self._checkpointed_runs)

    # Make CheckpointManagers. MultiWorkerMirroredStrategy requires different
    # setup on chief and on other workers.
    self._read_checkpoint_manager = checkpoint_management.CheckpointManager(
        checkpoint, directory=checkpoint_dir, max_to_keep=1)
    if multi_worker_util.is_chief(
        cluster_spec=cluster_resolver.cluster_spec(),
        task_type=cluster_resolver.task_type,
        task_id=cluster_resolver.task_id):
      self._write_checkpoint_manager = self._read_checkpoint_manager
    else:
      self._write_checkpoint_manager = checkpoint_management.CheckpointManager(
          checkpoint,
          _mwms_write_checkpoint_dir(checkpoint_dir, cluster_resolver.task_type,
                                     cluster_resolver.task_id,
                                     cluster_resolver.cluster_spec()),
          max_to_keep=1)

    self._read_checkpoint_manager.restore_or_initialize()

    # An internal step counter that's restored to checkpointed_iterations when
    # training is restored. It increments by one every time
    # `CoordinatedCheckpointManager.run` is called. Note that in this case, the
    # user must pass a single-step training function to
    # `CoordinatedCheckpointManager.run` instead of a multiple-step one.
    self._run_counter = self._checkpointed_runs.numpy()

    # The worker itself has received preeption signal.
    self._received_own_sigterm = threading.Event()

    # Some member (could be oneself) has received preemption signal, and the
    # step number to save a checkpoint has been aligned.
    self._received_sigterm_and_step = threading.Event()

    # When training is interrupted, we explicitly call the cleanup methods for
    # the thread watching for local worker's termination signal and the thread
    # watching for clusterwise information before we save a checkpoint and exit.
    # In the final chapter of the training where no interruption is encountered,
    # we rely on __del__ to clean up. However, there is no guarantee when or
    # whether __del__ is executed, thus we make the threads daemon to avoid it
    # preventing program from exit.
    self._cluster_wise_termination_watcher_thread = threading.Thread(
        target=self._wait_for_signal,
        name='PeerTerminationWatcher-%s' % self._id_in_cluster,
        daemon=True)
    self._cluster_wise_termination_watcher_thread.start()

    self._poll_gce_signal_thread = None
    self._platform_device = gce_util.detect_platform()
    if self._platform_device is gce_util.PlatformDevice.GCE_GPU:
      self._start_polling_for_gce_signal()
      self._exit_code = gce_util._RESTARTABLE_EXIT_CODE
    elif self._platform_device is gce_util.PlatformDevice.INTERNAL:
      self._start_watching_for_signal()
      self._exit_code = _RESTARTABLE_EXIT_CODE
    else:
      raise NotImplementedError('CoordinatedCheckpointManager is only supported'
                                ' for MultiWorkerMirroredStrategy with GPU.')

  def _start_watching_for_signal(self):
    signal.signal(signal.SIGTERM, self._sigterm_handler_fn)

  def _start_polling_for_gce_signal(self):
    self._poll_gce_signal_thread_should_stop = threading.Event()
    self._poll_gce_signal_thread = threading.Thread(
        target=self._poll_gce_signal,
        name='WorkerTerminationSignalWatcher-%s' % self._id_in_cluster,
        daemon=True)
    self._poll_gce_signal_thread.start()
    logging.info('Start polling for termination signal.')

  def _poll_gce_signal(self):
    """Poll maintenance notice from GCE and notify peers if receiving one."""
    while True:
      if self._poll_gce_signal_thread_should_stop.is_set():
        return
      if gce_util.signal_polling_fn():
        # For the countdown.
        self._signal_receipt_time = time.time()
        break
      time.sleep(1)
    try:
      context.context().set_config_key_value(_PREEMPTION_KEY,
                                             self._id_in_cluster)
      logging.info('Member %s has received termination notice.',
                   self._id_in_cluster)
      self._received_own_sigterm.set()

    # This is to handle the case that a worker has received termination
    # notice but hasn't come to the next step to set the step key. Other
    # workers might receive a termination notice too, and attempt to set the
    # config key again, which causes this error. This can be safely ignored
    # since checkpoint should be saved as early as the earliest call is made.
    except errors.AlreadyExistsError:
      logging.info('Member %s has received termination notice. But some other '
                   'worker has received it as well! Leaving'
                   ' it to them to decide when to checkpoint. ',
                   self._id_in_cluster)
      return

  def _stop_poll_gce_signal_thread(self):
    if self._poll_gce_signal_thread:
      self._poll_gce_signal_thread_should_stop.set()
      self._poll_gce_signal_thread.join()
      self._poll_gce_signal_thread = None
      logging.info('Shut down watcher for one\'s own termination signal')

  def _stop_cluster_wise_termination_watcher_thread(self):
    """Stop the thread that is _wait_for_signal."""
    if self._cluster_wise_termination_watcher_thread:
      try:
        if self._cluster_wise_termination_watcher_thread.is_alive():

          context.context().set_config_key_value(_PREEMPTION_KEY,
                                                 _STOP_WATCHING_CLUSTER_VALUE)
      except:  # pylint: disable=bare-except
        # We'll ignore any error in the process of setting this key. There
        # certainly will be a AlreadyExistError since all workers are trying to
        # push this key. Or some worker might have exited already, leading to a
        # errors.UnavailableError or errors.AbortedError. We'll also ignore
        # other errors since they are not important to the process.
        pass

      finally:

        self._cluster_wise_termination_watcher_thread.join()
        self._cluster_wise_termination_watcher_thread = None
        logging.info('Shut down watcher for peer\'s termination signal.')

  def __del__(self):
    self._stop_cluster_wise_termination_watcher_thread()
    self._stop_poll_gce_signal_thread()

  @property
  def total_runs(self):
    """Returns the number of times `CoordinatedCheckpointManager.run` is called.

    This value tracks the number of all calls to
    `CoordinatedCheckpointManager.run` including those before the program is
    restarted and the training is restored. The user can compute their total
    number of iterations by:
    `coordinated_checkpoint_manager.run * number_of_steps_in_train_function`,
    while for tf.distribute.MultiWorkerMirroredStrategy users,
    `number_of_steps_in_train_function` should be one.
    """
    return self._run_counter

  def run(self,
          distributed_train_function,
          *args,
          **kwargs):
    """Runs a training function with error and preemption handling.

    This function handles the preemption signal from any peer in the cluster by
    saving the training progress and exiting gracefully. (Specifically, when
    running on Borg, it exits with a special code so that the cluster
    automatically restarts the training after the down worker is back.) It will
    also propagate any program error encountered during execution of
    `distributed_train_function` to all workers so that they can raise the same
    error.

    The `distributed_train_function` argument should be a distributed train
    function (i.e., containing a call to `tf.distribute.Strategy.run`). For
    `tf.distribute.MultiWorkerMirroredStrategy` users, we recommend passing in a
    single-step `distributed_train_function` to
    `CoordinatedCheckpointManager.run` so that the checkpoint can be saved in
    time in case a preemption signal or maintenance notice is sent.

    Besides the preemption and error handling part,
    `CoordinatedCheckpointManager.run(distributed_train_function, *args,
    **kwargs)` has the same effect and output as
    `distributed_train_function(*args, **kwargs)`. `distributed_train_function`
    can return either some or no result. The following is a shortened example:

    ```python

    @tf.function
    def distributed_train_step(iterator):
      # A distributed single-step training function.

      def step_fn(inputs):
        # A per-replica single-step training function.
        x, y = inputs
        ...
        return loss

      per_replica_losses = strategy.run(step_fn, args=(next(iterator),))
      return strategy.reduce(
          tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    for epoch in range(coordinated_checkpoint_manager.total_runs //
                       STEPS_PER_EPOCH, EPOCHS_TO_RUN):
      iterator = iter(multi_worker_dataset)
      total_loss = 0.0
      num_batches = 0

      for step in range(coordinated_checkpoint_manager.total_runs %
                        STEPS_PER_EPOCH, STEPS_PER_EPOCH):
        total_loss += coordinated_checkpoint_manager.run(distributed_train_step)
        num_batches += 1

      train_loss = total_loss / num_batches
      print('Epoch: %d, train_loss: %f.' %(epoch.numpy(), train_loss))

      train_accuracy.reset_states()
    ```

    Args:
      distributed_train_function: A (single-step) distributed training function.
      *args: args for `distributed_train_function`.
      **kwargs: kwargs for `distributed_train_function`.

    Raises:
      Program error encountered by any member in the cluster encounters one
      while executing the `distributed_train_function`, or any error from the
      program error propagation process.

    Returns:
      Result of running the `distributed_train_function`.
    """
    # TODO(wxinyi): after we support use with TPUStrategy, we should expand the
    # API doc to state that `distributed_train_function` does not need to be a
    # single-step training function, since a multi-step host-training loop is
    # the dominant use case for TPU user. Besides, passing in a multi-step
    # `distributed_train_function` will require the user to track their own
    # training steps.
    try:
      self._checkpoint_if_preempted()
      result = distributed_train_function(*args, **kwargs)
      self._run_counter += 1

    except errors.OpError as e:
      logging.info('Propagating error to cluster: %r: %s', e, e)
      try:
        context.context().report_error_to_cluster(e.error_code, e.message)
      except Exception as ex:  # pylint: disable=broad-except
        logging.info('Ignoring error during error propagation: %r:%s', ex, ex)
      raise

    return result

  def _save_checkpoint(self):
    """Saves the checkpoint."""
    self._write_checkpoint_manager.save()
    # All workers need to participate in saving a checkpoint to avoid
    # deadlock. They need to write to different paths so that they would not
    # override each other. We make temporary directories for non-chief
    # workers to write to, and clean them up afterward.
    if not multi_worker_util.is_chief(
        cluster_spec=self._cluster_resolver.cluster_spec(),
        task_type=self._cluster_resolver.task_type,
        task_id=self._cluster_resolver.task_id):
      gfile.DeleteRecursively(
          os.path.dirname(self._write_checkpoint_manager.directory))

  def _checkpoint_if_preempted(self):
    """Checkpoint if any worker has received a preemption signal.

    This function handles preemption signal reported by any worker in the
    cluster. The current implementation relies on the fact that all workers in a
    MultiWorkerMirroredStrategy training cluster have a step number difference
    maximum of 1.
    - If the signal comes from the worker itself (i.e., where this failure
    handler sits), the worker will notify all peers to checkpoint after they
    finish CURRENT_STEP+1 steps, where CURRENT_STEP is the step this worker has
    just finished. And the worker will wait for all peers to acknowledge that
    they have received its preemption signal and the final-step number before
    the worker proceeds on training the final step.
    - If the signal comes from another member in the cluster but NO final-step
    info is available, proceed on training, because it will be available after
    finishing the next step.
    - If the signal comes from some other member in the cluster, and final-step
    info is available, if the worker has not finished these steps yet, keep
    training; otherwise, checkpoint and exit with a cluster-recognized restart
    code.
    """
    if self._received_sigterm_and_step.is_set():

      run_count_key = context.context().get_config_key_value(_RUN_COUNT_KEY)

      if run_count_key == str(self._run_counter):
        logging.info('Starting checkpoint and exit')

        self._checkpointed_runs.assign(self.total_runs)

        start_time = time.monotonic()
        self._save_checkpoint()
        end_time = time.monotonic()
        logging.info('Checkpoint finished at path %s',
                     self._write_checkpoint_manager.directory)
        logging.info('Checkpoint time: %f', end_time - start_time)
        self._stop_poll_gce_signal_thread()
        self._stop_cluster_wise_termination_watcher_thread()
        sys.exit(self._exit_code)

    elif (self._received_own_sigterm.is_set() and
          (context.context().get_config_key_value(_PREEMPTION_KEY)
           == self._id_in_cluster)):

      logging.info('Termination caught in main thread on preempted worker')

      step_to_save_at = str(self._run_counter + 1)
      context.context().set_config_key_value(_RUN_COUNT_KEY, step_to_save_at)
      logging.info('%s set to %s', _RUN_COUNT_KEY, step_to_save_at)

      n_workers = multi_worker_util.worker_count(
          self._cluster_resolver.cluster_spec(),
          self._cluster_resolver.task_type)
      for i in range(n_workers):
        context.context().get_config_key_value(f'{_ACKNOWLEDGE_KEY}_{i}')
        logging.info('Sigterm acknowledgement from replica %d received', i)

  def _sigterm_handler_fn(self, signum, frame):
    """Upload the to-be-preempted worker's id to coordination service."""
    del signum, frame
    logging.info('Member %s has received preemption signal.',
                 self._id_in_cluster)
    context.context().set_config_key_value(_PREEMPTION_KEY, self._id_in_cluster)
    self._received_own_sigterm.set()

  def _wait_for_signal(self):
    """Watch out for peer preemption signal and step-to-save and acknowledge."""
    preemption_key = context.context().get_config_key_value(_PREEMPTION_KEY)

    if preemption_key != _STOP_WATCHING_CLUSTER_VALUE:
      context.context().get_config_key_value(_RUN_COUNT_KEY)

      # This must be set before we set the ack key below, otherwise its value in
      # _checkpoint_if_preempted may be outdated.
      self._received_sigterm_and_step.set()

      ack_key = f'{_ACKNOWLEDGE_KEY}_{self._id_in_cluster}'
      context.context().set_config_key_value(ack_key, '1')
      logging.info('CoordinatedCheckpointManager._wait_for_signal: %s set, '
                   'preemption awareness acknowledged', ack_key)

