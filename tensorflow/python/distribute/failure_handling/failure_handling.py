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
"""Module for `PreemptionCheckpointHandler`.

This is currently under development and the API is subject to change.

PreemptionCheckpointHandler reduces loss of training progress caused by
termination (preemption or maintenance) of workers in multi-worker synchronous
training and avoid surfacing an error indistinguishable from application errors
to the job scheduler or users.
"""
import os
import signal
import sys
import threading
import time

from tensorflow.core.distributed_runtime.preemption import gen_check_preemption_op
from tensorflow.python.checkpoint import checkpoint as checkpoint_lib
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute.failure_handling import failure_handling_util
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls


_INITIAL_RUN_COUNT_KEY = 'RUN_TO_CHECKPOINT'
_FINAL_RUN_COUNT_KEY = 'LAST_RUN_TO_CHECKPOINT'
# This key is used to guarantee that only one worker (and it's the earliest
# one that receives a preemption signal) sets _received_own_sigterm,
# leads the step resolution, and controls the grace period timeline.
_PREEMPTION_WORKER_KEY = 'TERMINATED_WORKER'
_ACKNOWLEDGE_KEY = 'RECEIVED_SIGNAL'
_ITERATION_VARIABLE = 'checkpointed_runs'
_STOP_WATCHING_CLUSTER_VALUE = 'STOP_WATCHER'
PREEMPTION_KEY = 'TF_DEFAULT_PREEMPTION_NOTICE_KEY'


# TODO(wxinyi): add type annotations.
def _non_chief_checkpoint_dir(checkpoint_dir, task_id):
  """Returns a directory for non-chief worker to save checkpoint."""
  dirpath = os.path.dirname(checkpoint_dir)
  base = os.path.basename(checkpoint_dir)
  base_dirpath = 'workertemp_' + str(task_id)
  dirpath = os.path.join(dirpath, base_dirpath)
  file_io.recursive_create_dir_v2(dirpath)
  return os.path.join(dirpath, base)


@tf_export('distribute.experimental.TerminationConfig', v1=[])
class TerminationConfig(object):
  """Customization of `PreemptionCheckpointHandler` for various platforms.

  A `TerminationConfig` can be created and passed to a
  `tf.distribute.experimental.PreemptionCheckpointHandler` to provide
  customization based on the platform. It can deliver three pieces of
  information:

  * How to decide if there is a termination event soon

  The form of termination notification and how to fetch it vary across
  platforms. Thus `PreemptionCheckpointHandler` may take a user-defined
  function, `termination_watcher_fn`, and execute it repeatedly to check for
  termination notification. `termination_watcher_fn` should be a function
  that returns `True` if a termination notification is available and
  `False` otherwise. The function should be lightweight and non-blocking so that
  resources can be cleaned up properly if no termination signal is ever raised
  until training finishes.

  * How to exit the program

  A user can configure this through the `exit_fn`, which
  `PreemptionCheckpointHandler` executes after saving the checkpoint to exit the
  training program gracefully. For `tf.distribute.MultiWorkerMirroredStrategy`,
  a restart is necessary to reset the program's state. However, having a
  customized `exit_fn` may facilitate the restart and smoothen the training
  experience. How so? Maybe the platform has an agreement to a `RESTART_CODE`
  recognized as a program auto-restart signal, or maybe the user has a
  coordinating script that starts up the training, in which they can configure
  the program to auto-restart if it ever exits with this `RESTART_CODE`. In both
  cases, configuring the `exit_fn` to be `sys.exit(RESTART_CODE)` makes the
  training seamless.

  * How long does `PreemptionCheckpointHandler` have from receiving a
  termination event notice till the actual termination

  Some platforms have a gap time as long as one hour or so. In these cases,
  there is the option to utilize this gap time for training as much as possible
  before saving a checkpoint and exiting. This can be achieved by passing the
  `grace_period` argument a nonzero value. Note, for a user with a grace period
  that is not multiple times longer than their checkpoint writing time (e.g.,
  three times or more), we advise not to configure this argument, in which case
  `PreemptionCheckpointHandler` will directly save a checkpoint and exit.


  **The default behavior**:

  * For Google Borg Platform:
      * Automatically know how to detect preemption signal
      * Exit with a platform-recognized restart code
      * Save a checkpoint and exit immediately

  * For Google Cloud Platform:
      * Automatically know how to detect maintenance signal.
      * Exit with a code (User may configure this)
      * Automatically utilized the extended training period before save and exit

  * For Other platform:
      * If `termination_watcher_fn` is `None`, we will treat `signal.SIGTERM` as
      a termination signal.
      * If `exit_fn` is not configured, we exit the program with an arbitrary
      code.
      * If `grace_period` is not configured, we will wrap up the current
      training step, save a checkpoint, and exit the program as soon as we
      receive the termination signal.
  """

  def __init__(self,
               termination_watcher_fn=None,
               exit_fn=None,
               grace_period=None,
               save_fn=None):
    """Creates a `TerminationConfig` object.

    Args:
      termination_watcher_fn: a function to execute repeatedly that returns
        `True` if a preemption signal is available and False otherwise. The
        function cannot block until a preemption signal is available, which
        prevents proper cleanup of the program. A change is **NOT** recommended
        for users on Google Borg or Google Cloud Platform.
      exit_fn: a function to execute after a checkpoint is saved and before the
        preemption happens. Usually, it should be in the form of
        `lambda: sys.exit(RESTART_CODE)`, where `RESTART_CODE` varies by
        platform. A change is **NOT** recommended for users on Google Borg.
        Users on Google Cloud Platform may configure it to use a customized
        `RESTART_CODE`.
      grace_period: the length of time between receiving a preemption signal and
        the actual preemption. A change is **NOT** recommended for users on
        Google Borg, Google Cloud Platform, or users with a short grace period.
      save_fn: an optional function letting you configure how to save a
        checkpoint. This is useful if you'd like to pass extra argument to
        `tf.train.CheckpointManager.save` or `tf.train.Checkpoint.save`. By
        default, if not configured, the API will save checkpoint without extra
        arguments.
    """
    self.termination_watcher_fn = termination_watcher_fn
    self.exit_fn = exit_fn
    self.grace_period = grace_period
    self.save_fn = save_fn


# TODO(wxinyi): add some tests for TerminationConfig.
# TODO(wxinyi): configure the exit function based on device type (GPU or TPU).
class GcpGpuTerminationConfig(TerminationConfig):
  """Configurations for GCP GPU VM."""

  def __init__(  # pylint: disable=super-init-not-called
      self,
      termination_watcher_fn=None,
      exit_fn=None,
      grace_period=None,
      save_fn=None,
  ):
    self.termination_watcher_fn = (
        termination_watcher_fn
        or failure_handling_util.termination_watcher_function_gce
    )
    self.exit_fn = exit_fn or failure_handling_util.gce_exit_fn
    self.grace_period = (
        grace_period if grace_period or grace_period == 0 else
        failure_handling_util.GRACE_PERIOD_GCE)
    self.save_fn = save_fn


class GcpCpuTerminationConfig(TerminationConfig):
  """Configurations for GCP CPU VM."""

  def __init__(  # pylint: disable=super-init-not-called
      self,
      termination_watcher_fn=None,
      exit_fn=None,
      grace_period=None,
      save_fn=None):
    self.termination_watcher_fn = termination_watcher_fn or failure_handling_util.termination_watcher_function_gce
    self.exit_fn = exit_fn or failure_handling_util.gce_exit_fn
    self.grace_period = grace_period or 0
    self.save_fn = save_fn


class BorgTerminationConfig(TerminationConfig):
  """Configurations for Borg."""

  def __init__(  # pylint: disable=super-init-not-called
      self,
      termination_watcher_fn=None,
      exit_fn=None,
      grace_period=None,
      save_fn=None):
    self.termination_watcher_fn = termination_watcher_fn
    default_exit_fn = lambda: sys.exit(42)
    self.exit_fn = exit_fn or default_exit_fn
    self.grace_period = grace_period or 0
    self.save_fn = save_fn


class BorgTPUTerminationConfig(TerminationConfig):
  """Configurations for Borg."""

  def __init__(  # pylint: disable=super-init-not-called
      self,
      termination_watcher_fn=None,
      exit_fn=None,
      grace_period=None,
      save_fn=None):
    self.termination_watcher_fn = termination_watcher_fn
    self.exit_fn = exit_fn or failure_handling_util.default_tpu_exit_fn
    self.grace_period = grace_period or 0
    self.save_fn = save_fn


def _complete_config_for_environment(platform_device, termination_config):
  """Complete un-filled fields of TerminationConfig based on platform."""
  if not termination_config:
    termination_config = TerminationConfig()

  if platform_device is failure_handling_util.PlatformDevice.GCE_GPU:
    return GcpGpuTerminationConfig(termination_config.termination_watcher_fn,
                                   termination_config.exit_fn,
                                   termination_config.grace_period,
                                   termination_config.save_fn)

  elif platform_device is failure_handling_util.PlatformDevice.GCE_CPU:
    return GcpCpuTerminationConfig(termination_config.termination_watcher_fn,
                                   termination_config.exit_fn,
                                   termination_config.grace_period,
                                   termination_config.save_fn)

  elif platform_device is failure_handling_util.PlatformDevice.INTERNAL_TPU:
    return BorgTPUTerminationConfig(termination_config.termination_watcher_fn,
                                    termination_config.exit_fn,
                                    termination_config.grace_period,
                                    termination_config.save_fn)

  else:
    # The default we chose are the same as the ones used by Borg. So we just
    # return this.
    return BorgTerminationConfig(
        termination_config.termination_watcher_fn,
        termination_config.exit_fn, termination_config.grace_period,
        termination_config.save_fn)


# TODO(wxinyi): add release updates.
# Implementation:
# Each worker will create its own PreemptionCheckpointHandler instance, and the
# instances communicate through coordination services. Each
# PreemptionCheckpointHandler conduct three tasks in parallel:
# - Watches out for its own preemption signal. (_poll_termination_signal_thread)
# - Watches out for a step key from the coordination service made available
#   by any member in the cluster (_cluster_wise_termination_watcher_thread)
# - The main thread for training.
#
# The life cycle of a PreemptionCheckpointHandler is as below:
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
# PreemptionCheckpointHandler found that _received_checkpoint_step is set. If
# the local worker has not finished the required step after which to save a
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
# When the program restarts and PreemptionCheckpointHandler object is created,
# it will restore the checkpoint.
@tf_export('distribute.experimental.PreemptionCheckpointHandler', v1=[])
class PreemptionCheckpointHandler(object):
  # pylint: disable=line-too-long
  """Preemption and error handler for synchronous training.

  Note: This API only supports use with
  `tf.distribute.MultiWorkerMirroredStrategy` and `tf.distribute.TPUStrategy`.

  A `PreemptionCheckpointHandler` coordinates all workers to save a checkpoint
  upon receiving a preemption signal. It also helps disseminate application
  error messages accurately among the cluster. When a
  `PreemptionCheckpointHandler` object is created, it restores values from
  the latest checkpoint file if any exists.

  Right after the initialization, the object starts to watch out for termination
  signal for any member in the cluster. If receiving a signal, the next time the
  worker executes `PreemptionCheckpointHandler.run`, the
  `PreemptionCheckpointHandler` will align all workers to save a checkpoint.
  Then, if an `exit_fn` is configured via
  `tf.distribute.experimental.TerminationConfig`, it will be invoked. Otherwise,
  the process will simply exit and later the platform should restart it.

  Note: We advise users of `tf.distribute.MultiWorkerMirroredStrategy` who
  choose to configure their
  own `exit_fn` in `tf.distribute.experimental.TerminationConfig` to include a
  `sys.exit(CODE_OR_MESSAGE)` in the `exit_fn` so that after the restart, all
  workers can initialize communication services correctly. For users of
  `tf.distribute.TPUStrategy`, if they do not wish to do a cluster restart but
  would like an in-process restart (i.e., keep the coordinator alive and re-do
  the steps to connect to cluster, initialize TPU system, and make the
  `TPUStrategy` object), they could configure the `exit_fn` to a no-op.

  For users of `tf.distribute.MultiWorkerMirroredStrategy`, the core API is
  `PreemptionCheckpointHandler.run`:

  ```python
  strategy = tf.distribute.MultiWorkerMirroredStrategy()

  trained_epoch = tf.Variable(initial_value=tf.constant(0, dtype=tf.dtypes.int64), name='epoch')
  step_in_epoch = tf.Variable(initial_value=tf.constant(0, dtype=tf.dtypes.int64), name='step_in_epoch')

  with strategy.scope():
    dataset, model, optimizer = ...

    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     model=model,
                                     trained_epoch=trained_epoch,
                                     step_in_epoch=step_in_epoch)

    preemption_checkpoint_handler = tf.distribute.experimental.PreemptionCheckpointHandler(cluster_resolver, checkpoint, checkpoint_dir)

  while trained_epoch.numpy() < NUM_EPOCH:

    while step_in_epoch.numpy() < STEPS_PER_EPOCH:

      # distributed_train_function contains a call to strategy.run.
      loss += preemption_checkpoint_handler.run(distributed_train_function, args=(next(iterator),))
      # For users of MultiWorkerMirroredStrategy, usually
      # STEPS_PER_TRAIN_FUNCTION = 1.
      step_in_epoch.assign_add(STEPS_PER_TRAIN_FUNCTION)
      ...

    epoch.assign_add(1)
    step_in_epoch.assign(0)
  ```

  For users of `tf.distribute.TPUStrategy`, the core APIs are
  `PreemptionCheckpointHandler.run` and
  `PreemptionCheckpointHandler.watch_preemption_scope`:

  ```python

  strategy = tf.distribute.TPUStrategy(tpu_cluster_resolver)

  # Rest of TPU init omitted, see documentation for TPUSTrategy.

  with preemption_checkpoint_handler.watch_preemption_scope():
    while trained_epoch.numpy() < NUM_EPOCH:

      while step_in_epoch.numpy() < STEPS_PER_EPOCH:

        # distributed_train_function contains a call to strategy.run.
        loss += preemption_checkpoint_handler.run(distributed_train_function, args=(next(iterator),))

        # For users of TPUStrategy, usually STEPS_PER_TRAIN_FUNCTION >> 1 since
        # clustering multiple steps within a tf.function amortizes the overhead
        # of launching a multi-device function on TPU Pod.
        step_in_epoch.assign_add(STEPS_PER_TRAIN_FUNCTION)
        ...

      epoch.assign_add(1)
      step_in_epoch.assign(0)
  ```

  Not all interruptions come with advance notice so that the
  `PreemptionCheckpointHandler` can handle them, e.g., those caused by hardware
  failure. For a user who saves checkpoints for these cases themselves outside
  the `PreemptionCheckpointHandler`, if they are using a
  `tf.train.CheckpointManager`, pass it as the
  `checkpoint_or_checkpoint_manager` argument to the
  `PreemptionCheckpointHandler`. If they do not have a
  `tf.train.CheckpointManager` but are directly working with
  `tf.train.Checkpoint`, we advise saving the checkpoints in the directory
  that's passed as the `checkpoint_dir` argument. In this way, at the program
  beginning, `PreemptionCheckpointHandler` can restore the latest checkpoint
  from the directory, no matter it's saved by the user themselves or saved by
  the `PreemptionCheckpointHandler` before preemption happens.

  **A note on the platform:**

  `PreemptionCheckpointHandler` can only handle the kind of termination with
  advance notice. For now, the API recognizes the termination signal for CPU,
  GPU, and TPU on Google Borg and CPU and GPU on the Google Cloud Platform. In
  these cases, `PreemptionCheckpointHandler` will automatically adopt the
  correct preemption/maintenance notification detection mechanism. Users of
  other platforms can configure a detection monitoring behavior through the
  `tf.distribute.experimental.TerminationConfig`. Customization for the exit
  behavior and grace period length could also be done here.
  """
  # pylint: enable=line-too-long

  def __init__(self,
               cluster_resolver,
               checkpoint_or_checkpoint_manager,
               checkpoint_dir=None,
               termination_config=None):
    """Creates the `PreemptionCheckpointHandler`.

    Args:
      cluster_resolver: a `tf.distribute.cluster_resolver.ClusterResolver`
        object. You may also obtain it through the `cluster_resolver` attribute
        of the distribution strategy in use.
      checkpoint_or_checkpoint_manager: a `tf.train.CheckpointManager` or a
        `tf.train.Checkpoint`. If you are using a `tf.train.CheckpointManager`
        to manage checkpoints outside the `PreemptionCheckpointHandler` for
        backup purpose as well, pass it as `checkpoint_or_checkpoint_manager`
        argument. Otherwise, pass a `tf.train.Checkpoint` and the
        `PreemptionCheckpointHandler` will create
        a `tf.train.CheckpointManager` to manage it in the `checkpoint_dir`.
      checkpoint_dir: a directory where the `PreemptionCheckpointHandler` saves
        and restores checkpoints. When a `PreemptionCheckpointHandler` is
        created, the latest checkpoint in the `checkpoint_dir` will be restored.
        (This is not needed if a `tf.train.CheckpointManager` instead of a
        `tf.train.Checkpoint` is passed as the
        `checkpoint_or_checkpoint_manager` argument.)
      termination_config: optional, a
        `tf.distribute.experimental.TerminationConfig` object to configure for a
        platform other than Google Borg or GCP.
    """
    # TODO(wxinyi): Maybe make checkpoint_or_checkpoint_manager optional if
    # save_fn is passed. For now it's still useful for restore.
    if isinstance(checkpoint_or_checkpoint_manager,
                  checkpoint_lib.Checkpoint) and not checkpoint_dir:
      raise errors.InvalidArgumentError('When a checkpoint is passed, a '
                                        'checkpoint_dir must be passed as well'
                                        '.')

    self._cluster_resolver = cluster_resolver
    self._termination_config = termination_config
    self._checkpoint_or_checkpoint_manager = checkpoint_or_checkpoint_manager
    self._checkpoint_dir = checkpoint_dir

    self._platform_device = failure_handling_util.detect_platform()

    completed_termination_config = _complete_config_for_environment(
        self._platform_device, self._termination_config
    )
    self._termination_watcher_fn = (
        completed_termination_config.termination_watcher_fn
    )
    self._exit_fn = completed_termination_config.exit_fn
    self._grace_period = completed_termination_config.grace_period
    self._save_fn = completed_termination_config.save_fn

    if self._platform_device in (
        failure_handling_util.PlatformDevice.GCE_TPU,
        failure_handling_util.PlatformDevice.GCE_CPU,
    ):
      # While running MultiWorkerMirroredStrategy training with GPUs and CPUs
      # are the same on Borg, GCE CPU VM and GPU VM are different in terms
      # of live migration, grace period, etc. We can make it work upon request.
      raise NotImplementedError(
          'PreemptionCheckpointHandler does not support '
          'usage with TPU or CPU device on GCP.'
      )

    elif (
        self._platform_device
        == failure_handling_util.PlatformDevice.INTERNAL_TPU
    ):
      self._initialize_for_tpu_strategy()

    else:
      if cluster_resolver and 'ps' in cluster_resolver.cluster_spec().as_dict():
        raise NotImplementedError(
            'PreemptionCheckpointHandler does not support'
            'usage with tf.distribute.experimental.ParameterServerStrategy.'
        )

      self._initialize_for_mirrored_and_multi_worker_mirrored()

    logging.info('PreemptionCheckpointHandler initialized or restored.')

  def _initialize_for_tpu_strategy(self):
    """Makes configurations for using the handler with TPUStrategy."""
    self._is_chief = True
    self._poll_termination_signal_thread = None
    self._cluster_wise_termination_watcher_thread = None
    self._maybe_create_checkpoint_manager()
    self._read_checkpoint_manager.restore_or_initialize()
    self._run_counter = 0

  def _initialize_for_mirrored_and_multi_worker_mirrored(self):
    """Makes configurations and start watchers for MS, MWMS, or OneDevice."""
    if (
        not self._cluster_resolver
        or not self._cluster_resolver.cluster_spec().jobs
    ):
      # For MirroredStrategy, OneDeviceStrategy, and local-mode
      # MultiWorkerMirroredStrategy, an empty cluster spec is passed, and
      # coordination service is not enabled nor is it needed (since
      # it's used for cross-worker communication). Thus we will directly name
      # the worker id and is_chief properties and also skip the
      # uploading/reading from coordination service logic.
      self._local_mode = True
      self._id_in_cluster = 'single_worker'
      self._is_chief = True
    else:
      self._local_mode = False
      self._id_in_cluster = str(
          multi_worker_util.id_in_cluster(
              self._cluster_resolver.cluster_spec(),
              self._cluster_resolver.task_type,
              self._cluster_resolver.task_id))
      self._is_chief = multi_worker_util.is_chief(
          cluster_spec=self._cluster_resolver.cluster_spec(),
          task_type=self._cluster_resolver.task_type,
          task_id=self._cluster_resolver.task_id)
    # The number of calls to `PreemptionCheckpointHandler.run` when the latest
    # checkpoint was saved.
    self._checkpointed_runs = variables.Variable(
        initial_value=constant_op.constant(0, dtype=dtypes.int64),
        trainable=False,
        name=_ITERATION_VARIABLE)

    self._maybe_create_checkpoint_manager()

    if not hasattr(self._write_checkpoint_manager._checkpoint,  # pylint: disable=protected-access
                   _ITERATION_VARIABLE):
      setattr(self._write_checkpoint_manager._checkpoint, _ITERATION_VARIABLE,  # pylint: disable=protected-access
              self._checkpointed_runs)

    if not hasattr(self._read_checkpoint_manager._checkpoint,  # pylint: disable=protected-access
                   _ITERATION_VARIABLE):
      setattr(self._read_checkpoint_manager._checkpoint, _ITERATION_VARIABLE,  # pylint: disable=protected-access
              self._checkpointed_runs)

    self._read_checkpoint_manager.restore_or_initialize()

    # grace period countdown. Set to True for all workers once they finish
    # timing saving a checkpoint. Once entering this phase, new
    # preemption/maintenance notice will not be handled, since the whole cluster
    # goes down as the worker who first initiates the grace period goes down.
    self._final_checkpoint_countdown = False

    self._estimated_run_time = 0

    # An internal step counter that's restored to checkpointed_iterations when
    # training is restored. It increments by one every time
    # `PreemptionCheckpointHandler.run` is called. Note that in this case, the
    # user must pass a single-step training function to
    # `PreemptionCheckpointHandler.run` instead of a multiple-step one.
    self._run_counter = self._checkpointed_runs.numpy()

    # The worker itself has received preeption signal.
    self._received_own_sigterm = threading.Event()

    # Some member (could be oneself) has received preemption signal, and the
    # step number to save a checkpoint has been aligned.
    self._received_checkpoint_step = threading.Event()

    distribute_lib.distribution_strategy_input_api_counter.get_cell(
        self._platform_device.name,
        'PreemptionCheckpointHandler').increase_by(1)

    if not self._local_mode:
      # When training is interrupted, we explicitly call the cleanup methods for
      # the thread watching for local worker's termination signal and the thread
      # watching for clusterwise information before we save a checkpoint and
      # exit. In the final chapter of the training where no interruption is
      # encountered, we rely on __del__ to clean up. However, there is no
      # guarantee when or whether __del__ is executed, thus we make the threads
      # daemon to avoid it preventing program from exit.
      self._cluster_wise_termination_watcher_thread = threading.Thread(
          target=self._watch_step_to_save_key,
          name='PeerTerminationWatcher-%s' % self._id_in_cluster,
          daemon=True)
      logging.info('Start watcher for peer\'s signal.')
      self._cluster_wise_termination_watcher_thread.start()

    else:
      self._cluster_wise_termination_watcher_thread = None

    self._poll_termination_signal_thread = None

    if self._termination_watcher_fn:
      self._start_polling_for_termination_signal()
    else:
      self._start_watching_for_signal()

  def _maybe_create_checkpoint_manager(self):
    """Create CheckpointManager(s) if a checkpoint is passed else take it."""
    if isinstance(self._checkpoint_or_checkpoint_manager,
                  checkpoint_management.CheckpointManager):
      self._read_checkpoint_manager = self._checkpoint_or_checkpoint_manager
      self._write_checkpoint_manager = self._checkpoint_or_checkpoint_manager
      self._api_made_checkpoint_manager = False
    else:
      self._api_made_checkpoint_manager = True
      # Make CheckpointManagers. MultiWorkerMirroredStrategy requires different
      # setup on chief and on other workers.
      self._read_checkpoint_manager = checkpoint_management.CheckpointManager(
          self._checkpoint_or_checkpoint_manager,
          directory=self._checkpoint_dir,
          max_to_keep=1)

      if self._is_chief:
        self._write_checkpoint_manager = self._read_checkpoint_manager
      else:
        self._write_checkpoint_manager = (
            checkpoint_management.CheckpointManager(
                self._checkpoint_or_checkpoint_manager,
                _non_chief_checkpoint_dir(self._checkpoint_dir,
                                          self._cluster_resolver.task_id),
                max_to_keep=1))

  def _start_watching_for_signal(self):
    logging.info('Start watcher for local signal.')
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
      if self._termination_watcher_fn():
        break
      time.sleep(1)

    self._maybe_set_received_own_sigterm()

  def _maybe_set_received_own_sigterm(self):
    """Claim earliest preemption if no one else has done it before."""
    if self._local_mode:
      logging.info('Received termination notice.',
                   self._id_in_cluster)
      self._received_own_sigterm_time = time.time()
      self._received_own_sigterm.set()
      return

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
  @deprecated(None,
              'Track steps using a tf.Variable saved in checkpoint instead.')
  @doc_controls.do_not_generate_docs
  def total_run_calls(self):
    """Returns the number of times `PreemptionCheckpointHandler.run` is called.

    DEPRECATED: user should track total steps themselves, as this API provides
    little expressivity gain but could easily be misused and incurs extra
    synchronization cost for TPUStrategy users.

    This value tracks the number of all calls to
    `PreemptionCheckpointHandler.run` including those before the program is
    restarted and the training is restored, by saving and reading the value in
    the checkpoint. A user can compute their total number of iterations
    by `PreemptionCheckpointHandler.total_run_calls *
    number_of_steps_in_train_function`,
    while `number_of_steps_in_train_function` should be one for
    `tf.distribute.MultiWorkerMirroredStrategy` users. They can also use this
    value to infer the starting epoch and step after training restores, as shown
    in the example above.
    """
    if (self._platform_device ==
        failure_handling_util.PlatformDevice.INTERNAL_TPU):
      raise NotImplementedError('Please create variables saved in checkpoint '
                                'to keep track of steps and epochs.')
    return self._run_counter

  def run(self,
          distributed_train_function,
          *args,
          **kwargs):
    """Runs a training function with error and preemption handling.

    This function handles the preemption signal from any peer in the cluster by
    saving the training progress and exiting gracefully. It will
    also broadcase any program error encountered during the execution of
    `distributed_train_function` to all workers so that they can raise the same
    error.

    The `distributed_train_function` argument should be a distributed train
    function (i.e., containing a call to `tf.distribute.Strategy.run`). For
    `tf.distribute.MultiWorkerMirroredStrategy` users, we recommend passing in a
    single-step `distributed_train_function` to
    `PreemptionCheckpointHandler.run` so that the checkpoint can be saved in
    time in case a preemption signal or maintenance notice is sent.

    Besides the preemption and error handling part,
    `PreemptionCheckpointHandler.run(distributed_train_function, *args,
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

    for epoch in range(preemption_handler.total_run_calls // STEPS_PER_EPOCH,
                       EPOCHS_TO_RUN):
      iterator = iter(multi_worker_dataset)
      total_loss = 0.0
      num_batches = 0

      for step in range(preemption_handler.total_run_calls % STEPS_PER_EPOCH,
                        STEPS_PER_EPOCH):
        total_loss += preemption_handler.run(distributed_train_step)
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
      Program error encountered by any member in the cluster while executing the
      `distributed_train_function`, or any error from the program error
      propagation process.

    Returns:
      Result of running the `distributed_train_function`.
    """
    # TODO(wxinyi): after we support use with TPUStrategy, we should expand the
    # API doc to state that `distributed_train_function` does not need to be a
    # single-step training function, since a multi-step host-training loop is
    # the dominant use case for TPU user. Besides, passing in a multi-step
    # `distributed_train_function` will require the user to track their own
    # training steps.
    if (
        self._platform_device
        == failure_handling_util.PlatformDevice.INTERNAL_TPU
    ):
      return self._run_for_tpu(distributed_train_function, *args, **kwargs)
    else:
      return self._run_for_multi_worker_mirrored(
          distributed_train_function, *args, **kwargs
      )

  def _run_for_tpu(self, distributed_train_function, *args, **kwargs):
    """PreemptionCheckpointHandler.run implementation for TPUStrategy."""
    gen_check_preemption_op.check_preemption(preemption_key=PREEMPTION_KEY)
    return distributed_train_function(*args, **kwargs)

  def _run_for_multi_worker_mirrored(
      self, distributed_train_function, *args, **kwargs
  ):
    """PreemptionCheckpointHandler.run implementation for MWMS."""
    try:
      self._check_preemption_and_maybe_checkpoint()
      run_begin_time = time.time()
      result = distributed_train_function(*args, **kwargs)
      new_run_time = time.time() - run_begin_time
      self._run_counter += 1
      # Update the average run time with the new run.
      self._estimated_run_time = self._estimated_run_time + (
          new_run_time - self._estimated_run_time) / self._run_counter

    except errors.OpError as e:
      if not self._local_mode:
        logging.info('Propagating error to cluster: %r: %s', e, e)
        try:
          context.context().report_error_to_cluster(e.error_code, e.message)
        except Exception as ex:  # pylint: disable=broad-except
          logging.info('Ignoring error during error propagation: %r:%s', ex, ex)
      raise

    return result

  # Disabling line-too-long check since we do not want to break the line when
  # converted to public documentation.
  # pylint: disable=line-too-long
  def _save_checkpoint_if_preempted(self, *args, **kwargs):
    """Saves a checkpoint if a preemption signal has been made available.

    This is an alternative API for `PreemptionCheckpointHandler.run` and
    `PreemptionCheckpointHandler.watch_preemption_scope`. This method works for
    both `tf.distribute.MultiWorkerMirroredStrategy` and
    `tf.distribute.TPUStrategy`. However, **for TPUStrategy, this method will
    add a synchronization point between workers and the coordinator** and thus
    may have performance implication. If this is a concern, use the combination
    of `PreemptionCheckpointHandler.watch_preemption_scope` and
    `PreemptionCheckpointHandler.run` instead.

    ```python
    strategy = tf.distribute.TPUStrategy(tpu_cluster_resolver)
    # initialization omitted

    with strategy.scope():
      # Save in the checkpoint.
      trained_step = tf.Variable(initial_value=tf.constant(0, dtype=tf.dtypes.int64), name='trained_step', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

      checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory, max_to_keep=1)
      preemption_handler = tf.distribute.experimental.PreemptionCheckpointHandler(cluster_resolver, checkpoint_manager)

    while trained_step.numpy() < NUM_STEPS:
      train_multi_step_function()
      trained_step.assign_add(STEPS_IN_FUNCTION)
      preemption_handler._save_checkpoint_if_preempted()
    ```

    Args:
      *args: args for `tf.train.CheckpointManager.save()` to save checkpoint.
      **kwargs: kwargs for `tf.train.CheckpointManager.save()` to save.
    """
    # pylint: enable=line-too-long
    if (self._platform_device ==
        failure_handling_util.PlatformDevice.INTERNAL_TPU):

      try:
        with context.async_scope():
          gen_check_preemption_op.check_preemption(
              preemption_key=PREEMPTION_KEY)
      except errors.AbortedError as abort_error:
        if abort_error.experimental_payloads.get(
            b'type.googleapis.com/tensorflow.distributed_runtime.WorkerPreemption'
        ):
          logging.info('Clearing preemption error to save checkpoint...')

          context.async_clear_error()
          self._save_checkpoint(*args, **kwargs)

          # For TPU training, the default behavior is that it will block until
          # workers are down and returns with error.
          self._exit_fn()

        else:
          raise

    else:
      self._check_preemption_and_maybe_checkpoint(*args, **kwargs)
      self._run_counter += 1
      self._estimated_run_time = 0

  @tf_contextlib.contextmanager
  def watch_preemption_scope(self):
    """Syncs error and maybe save checkpoint for usage with TPUStrategy.

    Note: Usage with `tf.distribute.MultiWorkerMirroredStrategy` does not need
    this API.

    Example usage:

    ```python
    with preemption_checkpoint_handler.watch_preemption_scope():
      while trained_step.numpy() < NUM_STEPS:

        # distributed_train_function contains a call to strategy.run.
        loss += preemption_checkpoint_handler.run(distributed_train_function, args=(next(iterator),))
        trained_step.assign_add(STEPS_PER_TRAIN_FUNCTION)
    ```

    In this workflow, `PreemptionCheckpointHandler.run` will flag preemption
    signal received, and `watch_preemption_scope` will handle the preemption
    signal by saving a checkpoint and then either exit to restart or execute a
    user-passed `exit_fn` in `tf.distribute.experimental.TerminationConfig`. If
    no preemption signal is received during execution of ops and function inside
    the scope, `watch_preemption_scope` ensures the completion of all async op
    and function execution when exiting and will raises exceptions if async
    execution results in an error state.

    Yields:
      None
    """
    if self._platform_device == failure_handling_util.PlatformDevice.INTERNAL_TPU:
      try:
        with context.async_scope():
          yield
      except errors.AbortedError as abort_error:
        if abort_error.experimental_payloads.get(
            b'type.googleapis.com/tensorflow.distributed_runtime.WorkerPreemption'
        ):
          logging.info('Clearing preemption error to save checkpoint...')

          context.async_clear_error()
          self._save_checkpoint()

          self._exit_fn()

        else:
          raise
    else:
      try:
        yield
      except errors.OpError as e:
        if not self._local_mode:
          logging.info('Propagating error to cluster: %r: %s', e, e)
          try:
            context.context().report_error_to_cluster(e.error_code, e.message)
          except Exception as ex:  # pylint: disable=broad-except
            logging.info('Ignoring error during error propagation: %r:%s', ex, ex)
        raise

  def _save_checkpoint(self, *args, **kwargs):
    """Saves the checkpoint and exit program."""
    distribute_lib.distribution_strategy_input_api_counter.get_cell(
        self._platform_device.name,
        'PreemptionCheckpointHandler Saving Checkpoint').increase_by(1)
    logging.info('PreemptionCheckpointHandler: Starting saving a checkpoint.')

    if self._platform_device != failure_handling_util.PlatformDevice.INTERNAL_TPU:
      self._checkpointed_runs.assign(self.total_run_calls)

    start_time = time.monotonic()

    if self._save_fn:
      self._save_fn(*args, **kwargs)
    else:
      self._write_checkpoint_manager.save(*args, **kwargs)

    end_time = time.monotonic()

    logging.info('Checkpoint finished at path %s',
                 self._write_checkpoint_manager.directory)
    self._checkpoint_time = end_time - start_time

  def _check_preemption_and_maybe_checkpoint(self, *args, **kwargs):
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

    Args:
      *args: args for `tf.train.CheckpointManager.save()` to save checkpoint.
      **kwargs: kwargs for `tf.train.CheckpointManager.save()` to save.
    """
    if self._platform_device == failure_handling_util.PlatformDevice.INTERNAL_TPU:
      gen_check_preemption_op.check_preemption(preemption_key=PREEMPTION_KEY)
      return

    if self._final_checkpoint_countdown:
      run_count_config_key = _FINAL_RUN_COUNT_KEY

    else:
      run_count_config_key = _INITIAL_RUN_COUNT_KEY

    if self._received_checkpoint_step.is_set():

      if self._step_to_checkpoint == str(self._run_counter):
        self._save_checkpoint(*args, **kwargs)

        if self._time_to_exit():
          self._stop_poll_termination_signal_thread()
          self._stop_cluster_wise_termination_watcher_thread()
          if self._api_made_checkpoint_manager and not self._is_chief:
            gfile.DeleteRecursively(
                os.path.dirname(self._write_checkpoint_manager.directory))
          logging.info(
              'PreemptionCheckpointHandler: checkpoint saved. Exiting.')

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

      if self._local_mode:
        self._step_to_checkpoint = step_to_save_at
        self._received_checkpoint_step.set()

      else:
        context.context().set_config_key_value(run_count_config_key,
                                               step_to_save_at)
        logging.info('%s set to %s', run_count_config_key, step_to_save_at)

        if not self._local_mode:
          worker_count = multi_worker_util.worker_count(
              self._cluster_resolver.cluster_spec(),
              self._cluster_resolver.task_type)
          for i in range(worker_count):
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
    # event, enters this branch in _check_preemption_and_maybe_checkpoint, make
    # a checkpoint. Then they set _final_checkpoint_countdown to True, clear
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
      # in _check_preemption_and_maybe_checkpoint may be outdated.
      self._step_to_checkpoint = step_value
      self._received_checkpoint_step.set()

      ack_key = f'{_ACKNOWLEDGE_KEY}_{_INITIAL_RUN_COUNT_KEY}_{self._id_in_cluster}'
      context.context().set_config_key_value(ack_key, '1')
      logging.info(
          'PreemptionCheckpointHandler: %s set, '
          'preemption awareness acknowledged', ack_key)

      # If a positive grace_period is not configured, we get the
      # _INITIAL_RUN_COUNT_KEY and then we're done.
      # _check_preemption_and_maybe_checkpoint
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
          logging.info(
              'PreemptionCheckpointHandler: %s acknowledged, final '
              'checkpoint timing received.', ack_key)
          self._received_checkpoint_step.set()
          self._step_to_checkpoint = final_step_value

# TODO(wxinyi): remove this line after we move the Keras callback prototype and
# change gce test usage.
WorkerPreemptionHandler = PreemptionCheckpointHandler
