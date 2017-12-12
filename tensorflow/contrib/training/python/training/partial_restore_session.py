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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import time


from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver
from tensorflow.python.training import session_run_hook


class CheckpointRestorerHook(session_run_hook.SessionRunHook):
  """Restores checkpoints after tf.Session is created."""

  def __init__(self,
               checkpoint_dir=None,
               checkpoint_file=None,
               var_list=None,
               wait_for_checkpoint=False,
               max_wait_secs=7200,
               recovery_wait_secs=30,
               ):
    """Initializes a `CheckpointRestorerHook`.

    Args:
      checkpoint_dir: `str`, base directory for the checkpoint files.
      checkpoint_file: `str`, path name for the checkpoint file.
        only one of `checkpoint_file` and `checkpoint_dir` should be
        not None.
      var_list: `list`, optional, the list of variables to be restored.
        If None, all global variables defined would be restored.
      wait_for_checkpoint: Whether to wait for checkpoint to become available.
      max_wait_secs: Maximum time to wait for checkpoints to become available.
      recovery_wait_secs: Interval between checkpoint checks when waiting for
        checkpoint.

    Use `CheckpointRestorerHook` with MonitoredSession to restore a
    training session from checkpoint, especially when only part of checkpoint
    variables are intended to be restored.

    See `PartialRestoreSession` as a full example of usage.

    """
    logging.info("Create CheckpointRestorerHook.")
    super(CheckpointRestorerHook, self).__init__()
    self._dir, self._file, self._var_list = checkpoint_dir, checkpoint_file, var_list
    self._wait_for_checkpoint, self._max_wait_secs, self._recovery_wait_secs = \
        wait_for_checkpoint, max_wait_secs, recovery_wait_secs
    self._saver = saver.Saver(var_list=self._var_list)

  def after_create_session(self, session, coord):
    super(CheckpointRestorerHook, self).after_create_session(session, coord)

    if self._file:
      self._saver.restore(session, self._file)
      return
    wait_time = 0
    ckpt = saver.get_checkpoint_state(self._dir)
    while not ckpt or not ckpt.model_checkpoint_path:
      if self._wait_for_checkpoint and wait_time < self._max_wait_secs:
        logging.info("Waiting for checkpoint to be available.")
        time.sleep(self._recovery_wait_secs)
        wait_time += self._recovery_wait_secs
        ckpt = saver.get_checkpoint_state(self._dir)
      else:
        return

    # Loads the checkpoint.
    self._saver.restore(session, ckpt.model_checkpoint_path)
    self._saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)


def PartialRestoreSession(master='',  # pylint: disable=invalid-name
                          is_chief=True,
                          checkpoint_dir=None,
                          restore_var_list=None,
                          scaffold=None,
                          hooks=None,
                          chief_only_hooks=None,
                          save_checkpoint_secs=600,
                          save_summaries_steps=monitored_session.USE_DEFAULT,
                          save_summaries_secs=monitored_session.USE_DEFAULT,
                          config=None,
                          stop_grace_period_secs=120,
                          log_step_count_steps=100):
  """Creates a `MonitoredSession` for training.

    Supports partial restoration from checkpoints with parameter
    `restore_var_list`, by adding `CheckpointRestorerHook`.

  For a chief, this utility sets proper session initializer/restorer. It also
  creates hooks related to checkpoint and summary saving. For workers, this
  utility sets proper session creator which waits for the chief to
  initialize/restore. Please check `tf.train.MonitoredSession` for more
  information.


  Args:
    master: `String` the TensorFlow master to use.
    is_chief: If `True`, it will take care of initialization and recovery the
      underlying TensorFlow session. If `False`, it will wait on a chief to
      initialize or recover the TensorFlow session.
    checkpoint_dir: A string.  Optional path to a directory where to restore
      variables.
    restore_var_list: a list of variables, optional, if not all variables should
      be recovered from checkpoint.
      Useful when changing network structures during training, i.e., finetuning
      a pretrained model with new layers.
    scaffold: A `Scaffold` used for gathering or building supportive ops. If
      not specified, a default one is created. It's used to finalize the graph.
    hooks: Optional list of `SessionRunHook` objects.
    chief_only_hooks: list of `SessionRunHook` objects. Activate these hooks if
      `is_chief==True`, ignore otherwise.
    save_checkpoint_secs: The frequency, in seconds, that a checkpoint is saved
      using a default checkpoint saver. If `save_checkpoint_secs` is set to
      `None`, then the default checkpoint saver isn't used.
    save_summaries_steps: The frequency, in number of global steps, that the
      summaries are written to disk using a default summary saver. If both
      `save_summaries_steps` and `save_summaries_secs` are set to `None`, then
      the default summary saver isn't used. Default 100.
    save_summaries_secs: The frequency, in secs, that the summaries are written
      to disk using a default summary saver.  If both `save_summaries_steps` and
      `save_summaries_secs` are set to `None`, then the default summary saver
      isn't used. Default not enabled.
    config: an instance of `tf.ConfigProto` proto used to configure the session.
      It's the `config` argument of constructor of `tf.Session`.
    stop_grace_period_secs: Number of seconds given to threads to stop after
      `close()` has been called.
    log_step_count_steps: The frequency, in number of global steps, that the
      global step/sec is logged.

  Returns:
    A `MonitoredSession` object.
  """
  if save_summaries_steps == monitored_session.USE_DEFAULT \
          and save_summaries_secs == monitored_session.USE_DEFAULT:
    save_summaries_steps = 100
    save_summaries_secs = None
  elif save_summaries_secs == monitored_session.USE_DEFAULT:
    save_summaries_secs = None
  elif save_summaries_steps == monitored_session.USE_DEFAULT:
    save_summaries_steps = None

  scaffold = scaffold or monitored_session.Scaffold()
  if not is_chief:
    session_creator = monitored_session.WorkerSessionCreator(
        scaffold=scaffold, master=master, config=config)
    return monitored_session.MonitoredSession(
      session_creator=session_creator, hooks=hooks or [],
      stop_grace_period_secs=stop_grace_period_secs)

  all_hooks = []
  if chief_only_hooks:
    all_hooks.extend(chief_only_hooks)
  if restore_var_list is None:
    restore_checkpoint_dir = checkpoint_dir
  else:
    restore_checkpoint_dir = None
    all_hooks.append(CheckpointRestorerHook(
        checkpoint_dir, var_list=restore_var_list))
    all_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    missing_vars = filter(lambda v: not (v in restore_var_list), all_vars)
    logging.warning("MonitoredTrainingSession not restoring %s", missing_vars)
  session_creator = monitored_session.ChiefSessionCreator(
      scaffold=scaffold,
      checkpoint_dir=restore_checkpoint_dir,
      master=master,
      config=config)

  if checkpoint_dir:
    all_hooks.append(basic_session_run_hooks.StepCounterHook(
        output_dir=checkpoint_dir, every_n_steps=log_step_count_steps))

    if (save_summaries_steps and save_summaries_steps > 0) or (
        save_summaries_secs and save_summaries_secs > 0):
      all_hooks.append(basic_session_run_hooks.SummarySaverHook(
          scaffold=scaffold,
          save_steps=save_summaries_steps,
          save_secs=save_summaries_secs,
          output_dir=checkpoint_dir))
    if save_checkpoint_secs and save_checkpoint_secs > 0:
      all_hooks.append(basic_session_run_hooks.CheckpointSaverHook(
          checkpoint_dir, save_secs=save_checkpoint_secs, scaffold=scaffold))

  if hooks:
    all_hooks.extend(hooks)
  return monitored_session.MonitoredSession(
    session_creator=session_creator, hooks=all_hooks,
    stop_grace_period_secs=stop_grace_period_secs)
