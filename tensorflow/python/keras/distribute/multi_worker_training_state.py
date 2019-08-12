# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Training state management in multi-worker distributed training."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os
import uuid
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import mode_keys
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.training.tracking import tracking

# Constant for `tf.keras.Model` attribute to store the epoch at which the most
# recently saved checkpoint was saved.
CKPT_SAVED_EPOCH = '_ckpt_saved_epoch'

CKPT_SAVED_EPOCH_UNUSED_VALUE = -1


def checkpoint_exists(filepath):
  """Returns whether the checkpoint `filepath` refers to exists."""
  if filepath.endswith('.h5'):
    return file_io.file_exists(filepath)
  tf_saved_model_exists = file_io.file_exists(filepath)
  tf_weights_only_checkpoint_exists = file_io.file_exists(filepath + '.index')
  return tf_saved_model_exists or tf_weights_only_checkpoint_exists


def remove_checkpoint_if_exists(ckpt_dir, filepath):
  """Removes the checkpoint if it exists and returns whether it has removed."""
  if checkpoint_exists(filepath):
    _remove_dir(ckpt_dir)
    return True
  return False


def _remove_dir(dir_to_remove):
  file_io.delete_recursively(dir_to_remove)


class MultiWorkerTrainingState(object):
  """Training state management class in multi-worker distributed training.

  In multi-worker training, model weights and epoch information are saved
  periodically for fault-tolerance, also known as preemption-recovery purpose.
  This class provides apis for backing up and restoring the training state.
  """

  def __init__(self, model, original_filepath):
    self._model = model

    # The directory and filepath that store the training state backup file.
    self._backup_dir, self._backup_filepath = self._get_backup_filepath(
        original_filepath)

    # For those who should not checkpoint (e.g. non-chief worker in sync
    # training), create a temporary directory to write to (that will be
    # removed later).
    if not multi_worker_util.should_save_checkpoint():
      self._temp_dir, self._temp_filepath = self._get_temp_filepath(
          original_filepath)

    # The epoch at which the checkpoint is saved. Used for fault-tolerance.
    # GPU device only has int64 dtype registered VarHandleOp.
    self._ckpt_saved_epoch = variables.Variable(
        initial_value=constant_op.constant(
            CKPT_SAVED_EPOCH_UNUSED_VALUE, dtype=dtypes.int64),
        name='ckpt_saved_epoch')

    # Variable initialization.
    K.set_value(self._ckpt_saved_epoch, CKPT_SAVED_EPOCH_UNUSED_VALUE)

    # Calling `AutoTrackable.__setattr__` to avoid getting added as a weight of
    # model (which is done in `Layer.__setattr__`), which breaks saving/loading
    # in hdf5 format. Once becomes an attr of `model`, _ckpt_saved_epoch gets
    # tracked and will be included in the checkpoint file when backing up.
    tracking.AutoTrackable.__setattr__(self._model, CKPT_SAVED_EPOCH,
                                       self._ckpt_saved_epoch)

  def back_up(self, epoch):
    """Back up the current state of training into a checkpoint file.

    Arguments:
      epoch: The current epoch information to be saved.
    """
    # pylint: disable=protected-access
    self._assert_in_multi_worker_mode()

    # Update `_ckpt_saved_epoch`.
    K.set_value(self._ckpt_saved_epoch, epoch)

    # If this is multi-worker training, and this worker should not
    # save checkpoint, we replace the filepath with a dummy filepath so
    # it writes to a file that will be removed at the end of _save_model()
    # call. This is because the SyncOnReadVariable needs to be synced across
    # all the workers in order to be read, and all workers need to initiate
    # that.
    if multi_worker_util.should_save_checkpoint():
      save_filepath = self._backup_filepath
    else:
      save_filepath = self._temp_filepath

    # Save the weights plus CKPT_SAVED_EPOCH variable.
    self._model.save_weights(save_filepath, overwrite=True)

    if not multi_worker_util.should_save_checkpoint():
      # Remove the file in multi-worker training where this worker should
      # not checkpoint. It is a dummy file previously saved for sync distributed
      # training.
      _remove_dir(self._temp_dir)

  def restore(self):
    """Restore the training state from the backed up checkpoint file.

    Returns:
      True if the training state is successfully restored. False if the training
      state doesn't need to be restored, or error occurred so it can't.
    """
    self._assert_in_multi_worker_mode()
    if not multi_worker_util.should_load_checkpoint():
      # For multi-worker training, it should not restore a model in certain
      # worker setting (e.g. non-chief worker in ParameterServerStrategy).
      return False
    if file_io.file_exists(self._backup_dir):
      try:
        # Load the weights plus CKPT_SAVED_EPOCH variable.
        self._model.load_weights(self._backup_filepath)
        return True

      except (IOError, ValueError) as e:
        raise ValueError('Error loading file from {}. Reason: {}'.format(
            self._backup_filepath, e))
    return False

  def delete_backup(self):
    """Delete the backup directories.

    Delete the backup directories which should not exist after `fit()`
    successfully finishes.
    """
    self._assert_in_multi_worker_mode()
    tracking.AutoTrackable.__delattr__(self._model, CKPT_SAVED_EPOCH)
    if multi_worker_util.should_save_checkpoint():
      _remove_dir(self._backup_dir)
    else:
      assert not file_io.file_exists(self._temp_dir)

  def maybe_load_initial_epoch_from_ckpt(self, initial_epoch, mode):
    """Maybe load initial epoch from ckpt considering possible worker recovery.

    When `_ckpt_saved_epoch` attribute exists and is not
    `CKPT_SAVED_EPOCH_UNUSED_VALUE`, this is under multi-worker training setting
    and indicates the worker is recovering from previous failure. In this case,
    infer `initial_epoch` from `self._ckpt_saved_epoch` to continue previous
    unfinished training from certain epoch.

    Arguments:
      initial_epoch: The original initial_epoch user passes in in `fit()`.
      mode: The mode for running `model.fit()`.

    Returns:
      If the training is recovering from previous failure under multi-worker
      training setting, return the epoch the training is supposed to continue
      at. Otherwise, return the `initial_epoch` the user passes in.
    """
    self._assert_in_multi_worker_mode()

    # TODO(rchao): Add recovery for validation case
    # (when mode == ModeKeys.TEST).
    epoch = K.eval(self._ckpt_saved_epoch)
    if mode == mode_keys.ModeKeys.TRAIN and epoch >= 0:
      # The most recently saved epoch is one epoch prior to the epoch it
      # failed at, so return the value of 'self._ckpt_saved_epoch' plus one.
      return epoch + 1
    return initial_epoch

  @contextlib.contextmanager
  def untrack_vars(self):
    """Provides a scope within which training state variables are untracked.

    Regular checkpoint file saved by `ModelCheckpoint` callback that the user
    requests should not contain training state variables such as
    `CKPT_SAVED_EPOCH`, or the epoch the checkpoint is most recently saved at.

    Yields:
      None.
    """
    tracking.AutoTrackable.__delattr__(self._model, CKPT_SAVED_EPOCH)
    yield
    tracking.AutoTrackable.__setattr__(self._model, CKPT_SAVED_EPOCH,
                                       self._ckpt_saved_epoch)

  def _get_backup_filepath(self, original_filepath):
    backup_dir = os.path.join(os.path.dirname(original_filepath), 'backup')
    return backup_dir, os.path.join(backup_dir, 'training_state')

  def _get_temp_filepath(self, original_filepath):
    temp_dir = os.path.join(
        os.path.dirname(original_filepath), 'temp_training_states',
        str(uuid.uuid4()))
    return temp_dir, os.path.join(temp_dir, 'training_state')

  def _assert_in_multi_worker_mode(self):
    # pylint: disable=protected-access
    if not self._model._in_multi_worker_mode():
      raise ValueError('MultiWorkerTrainingState is only supposed to be used '
                       'in multi-worker training. This indicates some error '
                       'that needs to be fixed. Please submit a bug issue to '
                       'tf.keras team.')
