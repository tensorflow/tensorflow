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
"""Module for `WorkerPreemptionHandler`.

This is currently under development and the API is subject to change.

WorkerPreemptionHandler reduces loss of training progress caused by termination
(preemption or maintenance) of workers in multi-worker synchronous training and
avoid surfacing an error indistinguishable from application errors to the
job scheduler or users.
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

_INITIAL_RUN_COUNT_KEY = 'RUN_TO_CHECKPOINT'
_FINAL_RUN_COUNT_KEY = 'LAST_RUN_TO_CHECKPOINT'
# This key is used to guarantee that only one worker (and it's the earliest
# one that receives a preemption signal) sets _received_own_sigterm,
# leads the step resolution, and controls the grace period timeline.
_PREEMPTION_WORKER_KEY = 'TERMINATED_WORKER'
_ACKNOWLEDGE_KEY = 'RECEIVED_SIGNAL'
_ITERATION_VARIABLE = 'checkpointed_runs'
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


# TODO(wxinyi): rename time_till_termination to grace_period.
class TerminationConfig(object):
  """Configurations to customize for a platform other than Google's Borg or GCP.

  A TerminationConfig can be created and passed to the
  `WorkerPreemptionHandler` to provide customization based on the platform.
  It will deliver three pieces of information:

  * How to decide if there is a termination event soon

  The termination notification and how to fetch it varies across platforms. Thus
  we accept a user-defined function, `termination_watcher_function`, and execute
  it repeatedly to check for termination notification.
  `termination_watcher_function` should be a function that returns True if a
  termination notification has been made available and False otherwise. And the
  function should be lightweight and non-blocking so that we can clean up the
  resources properly if no termination signal is ever raised until training
  finishes.

  * How to exit the program

  We are asking for an `exit_fn` to execute after saving the checkpoint to exit
  the training program gracefully. For MultiWorkerMirroredStrategy, a restart is
  inevitable to reset the program's state. However, you can configure the
  `exit_fn` to facilitate the restart and make the training experience
  smooth. How so? Maybe your platform has an agreement to a RESTART_CODE that’s
  recognized as a program auto-restart signal, or you may have a coordinating
  script that starts up the training, in which you can configure the program to
  auto-restart if it ever exits with this RESTART_CODE. In both cases,
  you can configure `exit_fn` to be `sys.exit(RESTART_CODE)` and then wouldn’t
  even notice that the training has been interrupted and restarted.

  * How long do we have from receiving a termination event notice till the
  actual termination.

  Some platforms have the gap time as long as, say, one hour. In this case, you
  might want to utilize this time for training as much as possible until you
  have to save a checkpoint and exit. We can utilize this information if you
  pass it through the `time_till_termination` argument.


  *The default behavior*:

  If you are training with Google’s Borg system or GCP, we automatically detect
  the platform and make the right configuration for you. Besides these two
  platforms, the default behavior on an unrecognized platform is:

  * If `termination_event` is `None`, we will treat `signal.SIGTERM` as a
  termination event.

  * If `exit_fn` not configured, we exit the program with an arbitrary code 42.

  * If `time_till_termination` is not configured, the default is 0, and we will
  wrap up the current training step, save a checkpoint, and exit the program as
  soon as we receive the termination signal.
  """

  def __init__(self,
               termination_watcher_function=None,
               exit_fn=None,
               time_till_termination=None):
    self.termination_watcher_function = termination_watcher_function
    self.exit_fn = exit_fn
    self.time_till_termination = time_till_termination


# TODO(wxinyi): configure the exit function based on device type (GPU or TPU).
class GCPTerminationConfig(TerminationConfig):
  """Configurations for GCP GPU VM."""

  def __init__(  # pylint: disable=super-init-not-called
      self,
      termination_watcher_function=None,
      exit_fn=None,
      time_till_termination=None):
    self.termination_watcher_function = termination_watcher_function or gce_util.termination_watcher_function_gce
    self.exit_fn = exit_fn or gce_util.gce_exit_fn
    self.time_till_termination = (
        time_till_termination if time_till_termination or
        time_till_termination is 0 else gce_util.GRACE_PERIOD_GCE)  # pylint: disable=literal-comparison


class BorgTerminationConfig(TerminationConfig):
  """Configurations for Borg."""

  def __init__(  # pylint: disable=super-init-not-called
      self,
      termination_watcher_function=None,
      exit_fn=None,
      time_till_termination=None):
    self.termination_watcher_function = termination_watcher_function
    default_exit_fn = lambda: sys.exit(42)
    self.exit_fn = exit_fn or default_exit_fn
    self.time_till_termination = time_till_termination or 0


def _complete_config_for_environement(platform_device, termination_config):
  """Complete un-filled fields of TerminationConfig based on platform."""
  if platform_device is gce_util.PlatformDevice.GCE_GPU:
    return GCPTerminationConfig(termination_config.termination_watcher_function,
                                termination_config.exit_fn,
                                termination_config.time_till_termination)

  else:
    # The default we chose are the same as the ones used by Borg. So we just
    # return this.
    return BorgTerminationConfig(
        termination_config.termination_watcher_function,
        termination_config.exit_fn,
        termination_config.time_till_termination)


# Implementation:
# Each worker will create its own WorkerPreemptionHandler instance, and the
# instances communicate through coordination services. Each
# WorkerPreemptionHandler conduct three tasks in parallel:
# - Watches out for its own preemption signal. (_poll_termination_signal_thread)
# - Watches out for a step key from the coordination service made available
#   by any member in the cluster (_cluster_wise_termination_watcher_thread)
# - The main thread for training.
#
# The life cycle of a WorkerPreemptionHandler is as below:
#
# It starts two threads as two watcher as described above. And it starts
# training. Each time before it starts a training step, it will check if any
# information has been made available by the two watchers: The
# _poll_termination_signal_thread will be in charge of the _received_own_sigterm
# event, the _cluster_wise_termination_watcher_thread will be in charge of the
# _received_checkpoint_step event.
#
# If at any point the local worker receives a preemption signal,
# _poll_termination_signal_thread will set _received_own_sigterm.
# Next time before it attempts to run a training step, it will deal with the
# event, by setting its current finished step + 1 as the step after which a
# checkpoint should be saved and make it available to all the workers through
# the coordination service. It will then continue training.
#
# This step key will be picked up by the other watcher,
# _cluster_wise_termination_watcher_thread, both on the worker to be preempted
# and other workers. And it will set the _received_checkpoint_step event.
# Now, if there is a long grace period before the training
# has to terminate (e.g., an hour), we would like to keep training and save a
# checkpoint again right before the termination. Thus this watcher thread will
# move on to watch out for a final step-to-save key. Otherwise,
# it has finished all the task to do.
#
# Back to the main training thread. Again, before the next training step, the
# WorkerPreemptionHandler found that _received_checkpoint_step is set. If the
# local worker has not finished the required step after which to save a
# checkpoint, it will not do anything. Continue training and it will revisit
# after another step. If the step is met, then it will save a checkpoint,
# which requires participation of all workers.
#
# After this checkpoint is saved, if there is NO long grace period, all workers
# will just exit. If there is, all workers will enter a grace period countdown
# phase (_final_checkpoint_countdown) and clear the _received_checkpoint_step
# event. They will then continue training.
#
# For the worker to be preempted, during this countdown period, it will check
# whether the grace period is almost ending before its every step. If not,
# nothing needs to be done. If so, it will again set a step-to-save key and made
# it available to all workers. This is still watched by
# _cluster_wise_termination_watcher_thread and gestured by
# _received_checkpoint_step. A similar process is repeated: all workers save
# a checkpoint at an agreed step. And after they finish saving, they recognize
# that they have finished a countdown period for an extended grace period, and
# they all exit.
#
# When the program restarts and WorkerPreemptionHandler object is created, it
# will restore the checkpoint.
class WorkerPreemptionHandler(object):
  """Preemption and error handler for synchronous training.

  The API helps coordinate all workers to save a checkpoint upon receiving a
  preemption signal and helps propagate accurate error messages during training.
  When the program recovers from preemption, the checkpoint that is passed to
  initialize a `WorkerPreemptionHandler` object will be loaded
  automatically.

  Right after the initialization, a thread starts to watch out for a termination
  signal for any member in the cluster, but the signal will only be handled
  (which includes aligning the step to save a checkpoint, saving a checkpoint,
  and exiting with a platform recognized restart code) after entering a
  `WorkerPreemptionHandler.run` call.

  Example usage:
  ```python
  strategy = tf.distribute.MultiWorkerMirroredStrategy()

  with strategy.scope():
    dataset, model, optimizer = ...

    fh_checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    worker_preemption_watcher = tf.distribute.WorkerPreemptionHandler(
        cluster_resolver, fh_checkpoint, checkpoint_directory)


    # `worker_preemption_watcher.total_runs` will be restored to its
    # checkpointed value when training is restored after interruption.
    for epoch in range(worker_preemption_watcher.total_runs //
                       STEPS_PER_EPOCH, num_epochs):
      for step in range(worker_preemption_watcher.total_runs %
                        STEPS_PER_EPOCH, num_steps):
        # distributed_train_step is a function wrapped by strategy.run
        loss += worker_preemption_watcher.run(distributed_train_step,
                                                   args=(next(dataset),))
  ```

  `WorkerPreemptionHandler` will create a CheckpointManager to manage the
  checkpoint and only one CheckpointManager should be active in a particular
  directory at a time. Thus, if the user would like to save a checkpoint for
  purpose other than fault tolerance, e.g., for evaluation, they should save it
  in a directory different from the one passed to a
  `WorkerPreemptionHandler`.

  This API targets multi-client distributed training, and right now only
  `tf.distribute.MultiWorkerMirroredStrategy` is supported.
  """

  def __init__(self,
               cluster_resolver,
               checkpoint,
               checkpoint_dir,
               termination_config=TerminationConfig()):
    """Creates the failure handler.

    Args:
      cluster_resolver: a `tf.distribute.cluster_resolver.ClusterResolver`. You
        may also get it through the `cluster_resolver` attribute of the strategy
        in use.
      checkpoint: a `tf.train.Checkpoint` that will be saved upon preemption and
        loaded upon restart by the `WorkerPreemptionHandler` API automatically.
      checkpoint_dir: a directory for the `WorkerPreemptionHandler` to play with
        checkpoints. `WorkerPreemptionHandler` will create a
        `tf.train.CheckpointManager` to manage the passed-in `checkpoint`. Since
        only one `tf.train.CheckpointManager` should be active in a particular
        directory at a time, this `checkpoint_dir` arg should preferably be
        separated from where the user saves their checkpoint for non-fault
        tolerance purpose.
      termination_config: a `TerminationConfig` object to configure for a
        platform other than Google Borg or GCP.
    """
    self._cluster_resolver = cluster_resolver
    self._checkpoint = checkpoint
    self._id_in_cluster = str(
        multi_worker_util.id_in_cluster(
            self._cluster_resolver.cluster_spec(),
            self._cluster_resolver.task_type,
            self._cluster_resolver.task_id))

    # The number of calls to `WorkerPreemptionHandler.run` when the latest
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

    # grace period countdown. Set to True for all workers once they finish
    # timing saving a checkpoint. Once entering this phase, new
    # preemption/maintenance notice will not be handled, since the whole cluster
    # goes down as the worker who first initiates the grace period goes down.
    self._final_checkpoint_countdown = False

    self._estimated_run_time = 0

    # An internal step counter that's restored to checkpointed_iterations when
    # training is restored. It increments by one every time
    # `WorkerPreemptionHandler.run` is called. Note that in this case, the
    # user must pass a single-step training function to
    # `WorkerPreemptionHandler.run` instead of a multiple-step one.
    self._run_counter = self._checkpointed_runs.numpy()

    # The worker itself has received preeption signal.
    self._received_own_sigterm = threading.Event()

    # Some member (could be oneself) has received preemption signal, and the
    # step number to save a checkpoint has been aligned.
    self._received_checkpoint_step = threading.Event()

    self._platform_device = gce_util.detect_platform()

    if self._platform_device in (gce_util.PlatformDevice.GCE_TPU,
                                 gce_util.PlatformDevice.GCE_CPU):
      # While running MultiWorkerMirroredStrategy training with GPUs and CPUs
      # are the same on Borg, GCE CPU VM and GPU VM are different in terms
      # of live migration, grace period, etc. We can make it work upon request.
      raise NotImplementedError('WorkerPreemptionHandler does not support '
                                'training with TPU or CPU device on GCP.')

    completed_termination_config = _complete_config_for_environement(
        self._platform_device, termination_config)
    self._termination_watcher_function = completed_termination_config.termination_watcher_function
    self._exit_fn = completed_termination_config.exit_fn
    self._grace_period = completed_termination_config.time_till_termination

    # When training is interrupted, we explicitly call the cleanup methods for
    # the thread watching for local worker's termination signal and the thread
    # watching for clusterwise information before we save a checkpoint and exit.
    # In the final chapter of the training where no interruption is encountered,
    # we rely on __del__ to clean up. However, there is no guarantee when or
    # whether __del__ is executed, thus we make the threads daemon to avoid it
    # preventing program from exit.
    self._cluster_wise_termination_watcher_thread = threading.Thread(
        target=self._watch_step_to_save_key,
        name='PeerTerminationWatcher-%s' % self._id_in_cluster,
        daemon=True)
    logging.info('Start watcher for peer\'s signal.')
    self._cluster_wise_termination_watcher_thread.start()

    self._poll_termination_signal_thread = None

    if completed_termination_config.termination_watcher_function:
      self._start_polling_for_termination_signal()
    else:
      self._start_watching_for_signal()

  def _start_watching_for_signal(self):
    signal.signal(signal.SIGTERM, self._sigterm_handler_fn)

  def _start_polling_for_termination_signal(self):
    self._poll_termination_signal_thread_should_stop = threading.Event()
    self._poll_termination_signal_thread = threading.Thread(
        target=self._poll_termination_signal,
        name='WorkerTerminationSignalWatcher-%s' % self._id_in_cluster,
        daemon=True)
    logging.info('Start polling for termination signal.')
    self._poll_termination_signal_thread.start()

  def _poll_termination_signal(self):
    """Poll maintenance notice and notify peers if receiving one."""
    while True:
      if self._poll_termination_signal_thread_should_stop.is_set(
      ) or self._final_checkpoint_countdown:
        return
      if self._termination_watcher_function():
        break
      time.sleep(1)

    self._maybe_set_received_own_sigterm()

  def _maybe_set_received_own_sigterm(self):
    """Claim earliest preemption if no one else has done it before."""
    try:
      context.context().set_config_key_value(_PREEMPTION_WORKER_KEY,
                                             self._id_in_cluster)
      logging.info('Member %s has received termination notice.',
                   self._id_in_cluster)
      self._received_own_sigterm_time = time.time()
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

  def _stop_poll_termination_signal_thread(self):
    if self._poll_termination_signal_thread:

      self._poll_termination_signal_thread_should_stop.set()
      self._poll_termination_signal_thread.join()

      self._poll_termination_signal_thread = None
      logging.info('Shut down watcher for one\'s own termination signal')

  def _stop_cluster_wise_termination_watcher_thread(self):
    """Stop the thread that is _watch_step_to_save_key."""
    if self._cluster_wise_termination_watcher_thread:
      try:
        context.context().set_config_key_value(_INITIAL_RUN_COUNT_KEY,
                                               _STOP_WATCHING_CLUSTER_VALUE)
      except (errors.AlreadyExistsError, errors.UnavailableError):
        # We'll ignore any error in the process of setting this key. There
        # certainly will be a AlreadyExistError since all workers are trying to
        # push this key. Or some worker might have exited already, leading to a
        # errors.UnavailableError or errors.AbortedError.
        pass
      except Exception as e:  # pylint: disable=broad-except
        # We'll also ignore other errors since they are not important to the
        # process.
        logging.info('Ignoring error when shutting down '
                     '_stop_cluster_wise_termination_watcher_thread: ' + str(e))

      try:
        context.context().set_config_key_value(_FINAL_RUN_COUNT_KEY,
                                               _STOP_WATCHING_CLUSTER_VALUE)
      except (errors.AlreadyExistsError, errors.UnavailableError):
        pass

      except Exception as e:  # pylint: disable=broad-except
        logging.info('Ignoring error when shutting down '
                     '_stop_cluster_wise_termination_watcher_thread: ' + str(e))

      finally:
        self._cluster_wise_termination_watcher_thread.join()
        self._cluster_wise_termination_watcher_thread = None
        logging.info('Shut down watcher for peer\'s termination signal.')

  def __del__(self):
    self._stop_cluster_wise_termination_watcher_thread()
    self._stop_poll_termination_signal_thread()

  @property
  def total_runs(self):
    """Returns the number of times `WorkerPreemptionHandler.run` is called.

    This value tracks the number of all calls to
    `WorkerPreemptionHandler.run` including those before the program is
    restarted and the training is restored. The user can compute their total
    number of iterations by:
    `worker_preemption_watcher.run * number_of_steps_in_train_function`,
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
    `WorkerPreemptionHandler.run` so that the checkpoint can be saved in
    time in case a preemption signal or maintenance notice is sent.

    Besides the preemption and error handling part,
    `WorkerPreemptionHandler.run(distributed_train_function, *args,
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

    for epoch in range(worker_preemption_watcher.total_runs //
                       STEPS_PER_EPOCH, EPOCHS_TO_RUN):
      iterator = iter(multi_worker_dataset)
      total_loss = 0.0
      num_batches = 0

      for step in range(worker_preemption_watcher.total_runs %
                        STEPS_PER_EPOCH, STEPS_PER_EPOCH):
        total_loss += worker_preemption_watcher.run(distributed_train_step)
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
      run_begin_time = time.time()
      result = distributed_train_function(*args, **kwargs)
      new_run_time = time.time() - run_begin_time
      self._run_counter += 1
      # Update the average run time with the new run.
      self._estimated_run_time = self._estimated_run_time + (
          new_run_time - self._estimated_run_time) / self._run_counter

    except errors.OpError as e:
      logging.info('Propagating error to cluster: %r: %s', e, e)
      try:
        context.context().report_error_to_cluster(e.error_code, e.message)
      except Exception as ex:  # pylint: disable=broad-except
        logging.info('Ignoring error during error propagation: %r:%s', ex, ex)
      raise

    return result

  def _save_checkpoint(self):
    """Saves the checkpoint and exit program."""
    logging.info('WorkerPreemptionHandler: Starting saving a checkpoint.')
    self._checkpointed_runs.assign(self.total_runs)

    start_time = time.monotonic()

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

    end_time = time.monotonic()

    logging.info('Checkpoint finished at path %s',
                 self._write_checkpoint_manager.directory)
    self._checkpoint_time = end_time - start_time

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
    if self._final_checkpoint_countdown:
      run_count_config_key = _FINAL_RUN_COUNT_KEY

    else:
      run_count_config_key = _INITIAL_RUN_COUNT_KEY

    if self._received_checkpoint_step.is_set():

      run_count_key = context.context().get_config_key_value(
          run_count_config_key)

      if run_count_key == str(self._run_counter):
        self._save_checkpoint()

        if self._time_to_exit():
          self._stop_poll_termination_signal_thread()
          self._stop_cluster_wise_termination_watcher_thread()
          logging.info('WorkerPreemptionHandler: checkpoint saved. Exiting.')
          self._exit_fn()

        else:
          logging.info('Continue training for the grace period.')
          self._final_checkpoint_countdown = True
          self._received_checkpoint_step.clear()

    elif self._received_own_sigterm.is_set():
      # Only the worker who gets termination signal first among the cluster
      # will enter this branch. The following will happen in chronological
      # order:
      # 1. The worker just receives a preemption signal and enters this branch
      # for the first time. It will set a step-to-checkpoint and let the cluster
      # know.
      # 2. If there is a long grace period, it will also set
      # _final_checkpoint_countdown, so that during this grace period, it will
      # re-enter this branch to check if grace period is ending.
      # 3. If it is, set a step-to-checkpoint key again.

      if self._final_checkpoint_countdown:
        if self._target_time_for_termination < time.time():
          logging.info(
              'Grace period almost ended. Final call to save a checkpoint!')
        else:
          return

      step_to_save_at = str(self._run_counter + 1)

      logging.info('Termination caught in main thread on preempted worker')
      context.context().set_config_key_value(run_count_config_key,
                                             step_to_save_at)
      logging.info('%s set to %s', run_count_config_key, step_to_save_at)

      n_workers = multi_worker_util.worker_count(
          self._cluster_resolver.cluster_spec(),
          self._cluster_resolver.task_type)
      for i in range(n_workers):
        context.context().get_config_key_value(
            f'{_ACKNOWLEDGE_KEY}_{run_count_config_key}_{i}')
        logging.info('Sigterm acknowledgement from replica %d received', i)

      self._setup_countdown_if_has_grace_period_and_not_already_counting_down()

  def _time_to_exit(self):
    """Return whether to exit: exit if no grace period or grace period ends."""
    # we should directly exit in either of the two cases:
    # 1. if no grace period is provided;
    # 2. if there is a grace period, and we're in countdown period. This,
    # together with the fact that _received_checkpoint_step is set (again),
    # means it's time to exit: when there is a grace period, a worker
    # receives preemption signal and sets the step key. Then all workers
    # receive the step key and set their local _received_checkpoint_step
    # event, enters this branch in _checkpoint_if_preempted, make a
    # checkpoint. Then they set _final_checkpoint_countdown to True, clear
    # _received_checkpoint_step, and continue training. New preemption
    # signals anywhere in the cluster will not be handled, because
    # _PREEMPTION_WORKER_KEY is occupied. The only chance that
    # _received_checkpoint_step gets set again is when the worker who has
    # received the preemption signal earlier decide it's time to do a final
    # checkpoint (by checking if it already passes
    # _target_time_for_termination). It will upload a final step key. All
    # workers receive this key and again set _received_checkpoint_step. So,
    # if we found out that _received_checkpoint_step is set, and also
    # _final_checkpoint_countdown is true, it's checkpoint and exit time.
    return (self._grace_period <= 0) or self._final_checkpoint_countdown

  def _setup_countdown_if_has_grace_period_and_not_already_counting_down(self):
    """Set up at the beginning of a countdown period for long grace period."""
    if self._grace_period > 0 and not self._final_checkpoint_countdown:
      # A factor to provide more buffer / inaccuracy.
      # TODO(wxinyi): update buffer_factor as needed. Maybe deduct a constant.
      buffer_factor = 3
      # Timing by 2 since while the preempted worker needs to do 1 extra step
      # when time_till_final_call <=0, other workers might need to do x step
      # where 0<x<2
      self._target_time_for_termination = (
          self._received_own_sigterm_time + self._grace_period -
          buffer_factor * self._estimated_run_time * 2)

  def _sigterm_handler_fn(self, signum, frame):
    """Upload the to-be-preempted worker's id to coordination service."""
    del signum, frame
    self._maybe_set_received_own_sigterm()

  def _watch_step_to_save_key(self):
    """Watch out for step-to-save config key and acknowledge.

    All workers, including the one to be preempted, execute this function to get
    step-to-save.
    """

    step_value = context.context().get_config_key_value(_INITIAL_RUN_COUNT_KEY)

    # get_config_key_value does not return until it gets some result. Thus at
    # the time to clean up, we upload a _STOP_WATCHING_CLUSTER_VALUE as the
    # value so we can join the thread executing _watch_step_to_save_key.
    if step_value != _STOP_WATCHING_CLUSTER_VALUE:
      # This must be set before we set the ack key below, otherwise its value
      # in _checkpoint_if_preempted may be outdated.
      self._received_checkpoint_step.set()

      ack_key = f'{_ACKNOWLEDGE_KEY}_{_INITIAL_RUN_COUNT_KEY}_{self._id_in_cluster}'
      context.context().set_config_key_value(ack_key, '1')
      logging.info(
          'WorkerPreemptionHandler: %s set, '
          'preemption awareness acknowledged', ack_key)

      # If a positive grace_period is not configured, we get the
      # _INITIAL_RUN_COUNT_KEY and then we're done. _checkpoint_if_preempted
      # will save a checkpoint and then exit. Otherwise, we need to move on to
      # wait for the _FINAL_RUN_COUNT_KEY, the one that the preempted worker
      # will set after we utilize the extended grace period to train, so that
      # a final checkpoint should be made right before the termination.
      if self._grace_period > 0:
        # Continue to wait until a final call is made.
        final_step_value = context.context().get_config_key_value(
            _FINAL_RUN_COUNT_KEY)
        if final_step_value != _STOP_WATCHING_CLUSTER_VALUE:
          ack_key = f'{_ACKNOWLEDGE_KEY}_{_FINAL_RUN_COUNT_KEY}_{self._id_in_cluster}'
          context.context().set_config_key_value(ack_key, '1')
          logging.info('WorkerPreemptionHandler: %s acknowledged, final '
                       'checkpoint timing received.', ack_key)
          self._received_checkpoint_step.set()
