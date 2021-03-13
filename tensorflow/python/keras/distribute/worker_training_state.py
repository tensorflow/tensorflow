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
"""Training state management."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.distribute import distributed_file_utils
from tensorflow.python.keras.utils import mode_keys
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training.tracking import util as trackable_util

# Constant for `tf.keras.Model` attribute to store the epoch at which the most
# recently saved checkpoint was saved.
CKPT_SAVED_EPOCH = '_ckpt_saved_epoch'

CKPT_SAVED_EPOCH_UNUSED_VALUE = -1


class WorkerTrainingState(object):
  """Training state management class.

  This class provides apis for backing up and restoring the training state.
  This allows model and epoch information to be saved periodically and restore
  for fault-tolerance, also known as preemption-recovery purpose.
  """

  def __init__(self, model, checkpoint_dir):
    self._model = model

    # The epoch at which the checkpoint is saved. Used for fault-tolerance.
    # GPU device only has int64 dtype registered VarHandleOp.
    self._ckpt_saved_epoch = variables.Variable(
        initial_value=constant_op.constant(
            CKPT_SAVED_EPOCH_UNUSED_VALUE, dtype=dtypes.int64),
        name='ckpt_saved_epoch')

    # Variable initialization.
    K.set_value(self._ckpt_saved_epoch, CKPT_SAVED_EPOCH_UNUSED_VALUE)

    # _ckpt_saved_epoch gets tracked and is included in the checkpoint file
    # when backing up.
    checkpoint = trackable_util.Checkpoint(
        model=self._model, ckpt_saved_epoch=self._ckpt_saved_epoch)

    # If this is single-worker training, checkpoint_dir are the same for
    # write_checkpoint_manager and read_checkpoint_manager.
    #
    # If this is multi-worker training, and this worker should not
    # save checkpoint, we replace the write_checkpoint_manager's checkpoint_dir
    # with a temp filepath, so it writes to a file that will be removed at the
    # end of back_up() call. This is necessary because the SyncOnReadVariable
    # needs to be synced across all the workers in order to be read, and all
    # workers need to perform `save()`.
    # But all workers should restore from the same checkpoint_dir as passed in
    # read_checkpoint_manager.
    self.read_checkpoint_manager = checkpoint_management.CheckpointManager(
        checkpoint,
        directory=os.path.join(checkpoint_dir, 'chief'),
        max_to_keep=1)
    write_checkpoint_dir = distributed_file_utils.write_dirpath(
        checkpoint_dir, self._model.distribute_strategy)
    if self._model.distribute_strategy.extended.should_checkpoint:
      self.write_checkpoint_manager = self.read_checkpoint_manager
    else:
      self.write_checkpoint_manager = checkpoint_management.CheckpointManager(
          checkpoint, directory=write_checkpoint_dir, max_to_keep=1)

  def back_up(self, epoch):
    """Back up the current state of training into a checkpoint file.

    Args:
      epoch: The current epoch information to be saved.
    """
    K.set_value(self._ckpt_saved_epoch, epoch)
    # Save the model plus CKPT_SAVED_EPOCH variable.
    if self.write_checkpoint_manager.save():
      distributed_file_utils.remove_temp_dirpath(
          self.write_checkpoint_manager.directory,
          self._model.distribute_strategy)

  def restore(self):
    """Restore the training state from the backed up checkpoint file.

    Returns:
      True if the training state is successfully restored. False if the training
      state doesn't need to be restored, or error occurred so it can't.
    """
    self.read_checkpoint_manager.restore_or_initialize()

  def delete_backup(self):
    """Delete the backup directories.

    Delete the backup directories which should not exist after `fit()`
    successfully finishes.
    """
    if self.write_checkpoint_manager is self.read_checkpoint_manager:
      try:
        file_io.delete_recursively_v2(self.write_checkpoint_manager.directory)
      except errors.NotFoundError:
        pass

  def maybe_load_initial_epoch_from_ckpt(self, initial_epoch, mode):
    """Maybe load initial epoch from ckpt considering possible worker recovery.

    When `_ckpt_saved_epoch` attribute exists and is not
    `CKPT_SAVED_EPOCH_UNUSED_VALUE`, this is under multi-worker training setting
    and indicates the worker is recovering from previous failure. In this case,
    infer `initial_epoch` from `self._ckpt_saved_epoch` to continue previous
    unfinished training from certain epoch.

    Args:
      initial_epoch: The original initial_epoch user passes in in `fit()`.
      mode: The mode for running `model.fit()`.

    Returns:
      If the training is recovering from previous failure under multi-worker
      training setting, return the epoch the training is supposed to continue
      at. Otherwise, return the `initial_epoch` the user passes in.
    """

    epoch = K.eval(self._ckpt_saved_epoch)
    if mode == mode_keys.ModeKeys.TRAIN and epoch >= 0:
      # The most recently saved epoch is one epoch prior to the epoch it
      # failed at, so return the value of 'self._ckpt_saved_epoch' plus one.
      return epoch + 1
    return initial_epoch
