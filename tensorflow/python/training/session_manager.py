# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Training helper that checkpoints models and creates session."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_management
from tensorflow.python.util.tf_export import tf_export


def _maybe_name(obj):
  """Returns object name if it has one, or a message otherwise.

  This is useful for names that apper in error messages.
  Args:
    obj: Object to get the name of.
  Returns:
    name, "None", or a "no name" message.
  """
  if obj is None:
    return "None"
  elif hasattr(obj, "name"):
    return obj.name
  else:
    return "<no name for %s>" % type(obj)


@tf_export(v1=["train.SessionManager"])
class SessionManager(object):
  """Training helper that restores from checkpoint and creates session.

  This class is a small wrapper that takes care of session creation and
  checkpoint recovery. It also provides functions that to facilitate
  coordination among multiple training threads or processes.

  * Checkpointing trained variables as the training progresses.
  * Initializing variables on startup, restoring them from the most recent
    checkpoint after a crash, or wait for checkpoints to become available.

  ### Usage:

  ```python
  with tf.Graph().as_default():
     ...add operations to the graph...
    # Create a SessionManager that will checkpoint the model in '/tmp/mydir'.
    sm = SessionManager()
    sess = sm.prepare_session(master, init_op, saver, checkpoint_dir)
    # Use the session to train the graph.
    while True:
      sess.run(<my_train_op>)
  ```

  `prepare_session()` initializes or restores a model. It requires `init_op`
  and `saver` as an argument.

  A second process could wait for the model to be ready by doing the following:

  ```python
  with tf.Graph().as_default():
     ...add operations to the graph...
    # Create a SessionManager that will wait for the model to become ready.
    sm = SessionManager()
    sess = sm.wait_for_session(master)
    # Use the session to train the graph.
    while True:
      sess.run(<my_train_op>)
  ```

  `wait_for_session()` waits for a model to be initialized by other processes.

  """

  def __init__(self,
               local_init_op=None,
               ready_op=None,
               ready_for_local_init_op=None,
               graph=None,
               recovery_wait_secs=30,
               local_init_run_options=None):
    """Creates a SessionManager.

    The `local_init_op` is an `Operation` that is run always after a new session
    was created. If `None`, this step is skipped.

    The `ready_op` is an `Operation` used to check if the model is ready.  The
    model is considered ready if that operation returns an empty 1D string
    tensor. If the operation returns a non empty 1D string tensor, the elements
    are concatenated and used to indicate to the user why the model is not
    ready.

    The `ready_for_local_init_op` is an `Operation` used to check if the model
    is ready to run local_init_op.  The model is considered ready if that
    operation returns an empty 1D string tensor. If the operation returns a non
    empty 1D string tensor, the elements are concatenated and used to indicate
    to the user why the model is not ready.

    If `ready_op` is `None`, the model is not checked for readiness.

    `recovery_wait_secs` is the number of seconds between checks that
    the model is ready.  It is used by processes to wait for a model to
    be initialized or restored.  Defaults to 30 seconds.

    Args:
      local_init_op: An `Operation` run immediately after session creation.
         Usually used to initialize tables and local variables.
      ready_op: An `Operation` to check if the model is initialized.
      ready_for_local_init_op: An `Operation` to check if the model is ready
         to run local_init_op.
      graph: The `Graph` that the model will use.
      recovery_wait_secs: Seconds between checks for the model to be ready.
      local_init_run_options: RunOptions to be passed to session.run when
        executing the local_init_op.

    Raises:
      ValueError: If ready_for_local_init_op is not None but local_init_op is
        None
    """
    # Sets default values of arguments.
    if graph is None:
      graph = ops.get_default_graph()
    self._local_init_op = local_init_op
    self._ready_op = ready_op
    self._ready_for_local_init_op = ready_for_local_init_op
    self._graph = graph
    self._recovery_wait_secs = recovery_wait_secs
    self._target = None
    self._local_init_run_options = local_init_run_options
    if ready_for_local_init_op is not None and local_init_op is None:
      raise ValueError("If you pass a ready_for_local_init_op "
                       "you must also pass a local_init_op "
                       ", ready_for_local_init_op [%s]" %
                       ready_for_local_init_op)

  def _restore_checkpoint(self,
                          master,
                          saver=None,
                          checkpoint_dir=None,
                          checkpoint_filename_with_path=None,
                          wait_for_checkpoint=False,
                          max_wait_secs=7200,
                          config=None):
    """Creates a `Session`, and tries to restore a checkpoint.


    Args:
      master: `String` representation of the TensorFlow master to use.
      saver: A `Saver` object used to restore a model.
      checkpoint_dir: Path to the checkpoint files. The latest checkpoint in the
        dir will be used to restore.
      checkpoint_filename_with_path: Full file name path to the checkpoint file.
      wait_for_checkpoint: Whether to wait for checkpoint to become available.
      max_wait_secs: Maximum time to wait for checkpoints to become available.
      config: Optional `ConfigProto` proto used to configure the session.

    Returns:
      A pair (sess, is_restored) where 'is_restored' is `True` if
      the session could be restored, `False` otherwise.

    Raises:
      ValueError: If both checkpoint_dir and checkpoint_filename_with_path are
        set.
    """
    self._target = master
    sess = session.Session(self._target, graph=self._graph, config=config)

    if checkpoint_dir and checkpoint_filename_with_path:
      raise ValueError("Can not provide both checkpoint_dir and "
                       "checkpoint_filename_with_path.")
    # If either saver or checkpoint_* is not specified, cannot restore. Just
    # return.
    if not saver or not (checkpoint_dir or checkpoint_filename_with_path):
      return sess, False

    if checkpoint_filename_with_path:
      saver.restore(sess, checkpoint_filename_with_path)
      return sess, True

    # Waits up until max_wait_secs for checkpoint to become available.
    wait_time = 0
    ckpt = checkpoint_management.get_checkpoint_state(checkpoint_dir)
    while not ckpt or not ckpt.model_checkpoint_path:
      if wait_for_checkpoint and wait_time < max_wait_secs:
        logging.info("Waiting for checkpoint to be available.")
        time.sleep(self._recovery_wait_secs)
        wait_time += self._recovery_wait_secs
        ckpt = checkpoint_management.get_checkpoint_state(checkpoint_dir)
      else:
        return sess, False

    # Loads the checkpoint.
    saver.restore(sess, ckpt.model_checkpoint_path)
    saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)
    return sess, True

  def prepare_session(self,
                      master,
                      init_op=None,
                      saver=None,
                      checkpoint_dir=None,
                      checkpoint_filename_with_path=None,
                      wait_for_checkpoint=False,
                      max_wait_secs=7200,
                      config=None,
                      init_feed_dict=None,
                      init_fn=None):
    """Creates a `Session`. Makes sure the model is ready to be used.

    Creates a `Session` on 'master'. If a `saver` object is passed in, and
    `checkpoint_dir` points to a directory containing valid checkpoint
    files, then it will try to recover the model from checkpoint. If
    no checkpoint files are available, and `wait_for_checkpoint` is
    `True`, then the process would check every `recovery_wait_secs`,
    up to `max_wait_secs`, for recovery to succeed.

    If the model cannot be recovered successfully then it is initialized by
    running the `init_op` and calling `init_fn` if they are provided.
    The `local_init_op` is also run after init_op and init_fn, regardless of
    whether the model was recovered successfully, but only if
    `ready_for_local_init_op` passes.

    If the model is recovered from a checkpoint it is assumed that all
    global variables have been initialized, in particular neither `init_op`
    nor `init_fn` will be executed.

    It is an error if the model cannot be recovered and no `init_op`
    or `init_fn` or `local_init_op` are passed.

    Args:
      master: `String` representation of the TensorFlow master to use.
      init_op: Optional `Operation` used to initialize the model.
      saver: A `Saver` object used to restore a model.
      checkpoint_dir: Path to the checkpoint files. The latest checkpoint in the
        dir will be used to restore.
      checkpoint_filename_with_path: Full file name path to the checkpoint file.
      wait_for_checkpoint: Whether to wait for checkpoint to become available.
      max_wait_secs: Maximum time to wait for checkpoints to become available.
      config: Optional `ConfigProto` proto used to configure the session.
      init_feed_dict: Optional dictionary that maps `Tensor` objects to feed
        values.  This feed dictionary is passed to the session `run()` call when
        running the init op.
      init_fn: Optional callable used to initialize the model. Called after the
        optional `init_op` is called.  The callable must accept one argument,
        the session being initialized.

    Returns:
      A `Session` object that can be used to drive the model.

    Raises:
      RuntimeError: If the model cannot be initialized or recovered.
      ValueError: If both checkpoint_dir and checkpoint_filename_with_path are
        set.
    """

    sess, is_loaded_from_checkpoint = self._restore_checkpoint(
        master,
        saver,
        checkpoint_dir=checkpoint_dir,
        checkpoint_filename_with_path=checkpoint_filename_with_path,
        wait_for_checkpoint=wait_for_checkpoint,
        max_wait_secs=max_wait_secs,
        config=config)
    if not is_loaded_from_checkpoint:
      if init_op is None and not init_fn and self._local_init_op is None:
        raise RuntimeError("Model is not initialized and no init_op or "
                           "init_fn or local_init_op was given")
      if init_op is not None:
        sess.run(init_op, feed_dict=init_feed_dict)
      if init_fn:
        init_fn(sess)

    local_init_success, msg = self._try_run_local_init_op(sess)
    if not local_init_success:
      raise RuntimeError(
          "Init operations did not make model ready for local_init.  "
          "Init op: %s, init fn: %s, error: %s" % (_maybe_name(init_op),
                                                   init_fn,
                                                   msg))

    is_ready, msg = self._model_ready(sess)
    if not is_ready:
      raise RuntimeError(
          "Init operations did not make model ready.  "
          "Init op: %s, init fn: %s, local_init_op: %s, error: %s" %
          (_maybe_name(init_op), init_fn, self._local_init_op, msg))
    return sess

  def recover_session(self,
                      master,
                      saver=None,
                      checkpoint_dir=None,
                      checkpoint_filename_with_path=None,
                      wait_for_checkpoint=False,
                      max_wait_secs=7200,
                      config=None):
    """Creates a `Session`, recovering if possible.

    Creates a new session on 'master'.  If the session is not initialized
    and can be recovered from a checkpoint, recover it.

    Args:
      master: `String` representation of the TensorFlow master to use.
      saver: A `Saver` object used to restore a model.
      checkpoint_dir: Path to the checkpoint files. The latest checkpoint in the
        dir will be used to restore.
      checkpoint_filename_with_path: Full file name path to the checkpoint file.
      wait_for_checkpoint: Whether to wait for checkpoint to become available.
      max_wait_secs: Maximum time to wait for checkpoints to become available.
      config: Optional `ConfigProto` proto used to configure the session.

    Returns:
      A pair (sess, initialized) where 'initialized' is `True` if
      the session could be recovered and initialized, `False` otherwise.

    Raises:
      ValueError: If both checkpoint_dir and checkpoint_filename_with_path are
        set.
    """

    sess, is_loaded_from_checkpoint = self._restore_checkpoint(
        master,
        saver,
        checkpoint_dir=checkpoint_dir,
        checkpoint_filename_with_path=checkpoint_filename_with_path,
        wait_for_checkpoint=wait_for_checkpoint,
        max_wait_secs=max_wait_secs,
        config=config)

    # Always try to run local_init_op
    local_init_success, msg = self._try_run_local_init_op(sess)

    if not is_loaded_from_checkpoint:
      # Do not need to run checks for readiness
      return sess, False

    restoring_file = checkpoint_dir or checkpoint_filename_with_path
    if not local_init_success:
      logging.info(
          "Restoring model from %s did not make model ready for local init:"
          " %s", restoring_file, msg)
      return sess, False

    is_ready, msg = self._model_ready(sess)
    if not is_ready:
      logging.info("Restoring model from %s did not make model ready: %s",
                   restoring_file, msg)
      return sess, False

    logging.info("Restored model from %s", restoring_file)
    return sess, is_loaded_from_checkpoint

  def wait_for_session(self, master, config=None, max_wait_secs=float("Inf")):
    """Creates a new `Session` and waits for model to be ready.

    Creates a new `Session` on 'master'.  Waits for the model to be
    initialized or recovered from a checkpoint.  It's expected that
    another thread or process will make the model ready, and that this
    is intended to be used by threads/processes that participate in a
    distributed training configuration where a different thread/process
    is responsible for initializing or recovering the model being trained.

    NB: The amount of time this method waits for the session is bounded
    by max_wait_secs. By default, this function will wait indefinitely.

    Args:
      master: `String` representation of the TensorFlow master to use.
      config: Optional ConfigProto proto used to configure the session.
      max_wait_secs: Maximum time to wait for the session to become available.

    Returns:
      A `Session`. May be None if the operation exceeds the timeout
      specified by config.operation_timeout_in_ms.

    Raises:
      tf.DeadlineExceededError: if the session is not available after
        max_wait_secs.
    """
    self._target = master

    if max_wait_secs is None:
      max_wait_secs = float("Inf")
    timer = _CountDownTimer(max_wait_secs)

    while True:
      sess = session.Session(self._target, graph=self._graph, config=config)
      not_ready_msg = None
      not_ready_local_msg = None
      local_init_success, not_ready_local_msg = self._try_run_local_init_op(
          sess)
      if local_init_success:
        # Successful if local_init_op is None, or ready_for_local_init_op passes
        is_ready, not_ready_msg = self._model_ready(sess)
        if is_ready:
          return sess

      self._safe_close(sess)

      # Do we have enough time left to try again?
      remaining_ms_after_wait = (
          timer.secs_remaining() - self._recovery_wait_secs)
      if remaining_ms_after_wait < 0:
        raise errors.DeadlineExceededError(
            None, None,
            "Session was not ready after waiting %d secs." % (max_wait_secs,))

      logging.info("Waiting for model to be ready.  "
                   "Ready_for_local_init_op:  %s, ready: %s",
                   not_ready_local_msg, not_ready_msg)
      time.sleep(self._recovery_wait_secs)

  def _safe_close(self, sess):
    """Closes a session without raising an exception.

    Just like sess.close() but ignores exceptions.

    Args:
      sess: A `Session`.
    """
    # pylint: disable=broad-except
    try:
      sess.close()
    except Exception:
      # Intentionally not logging to avoid user complaints that
      # they get cryptic errors.  We really do not care that Close
      # fails.
      pass
    # pylint: enable=broad-except

  def _model_ready(self, sess):
    """Checks if the model is ready or not.

    Args:
      sess: A `Session`.

    Returns:
      A tuple (is_ready, msg), where is_ready is True if ready and False
      otherwise, and msg is `None` if the model is ready, a `String` with the
      reason why it is not ready otherwise.
    """
    return _ready(self._ready_op, sess, "Model not ready")

  def _model_ready_for_local_init(self, sess):
    """Checks if the model is ready to run local_init_op.

    Args:
      sess: A `Session`.

    Returns:
      A tuple (is_ready, msg), where is_ready is True if ready to run
      local_init_op and False otherwise, and msg is `None` if the model is
      ready to run local_init_op, a `String` with the reason why it is not ready
      otherwise.
    """
    return _ready(self._ready_for_local_init_op, sess,
                  "Model not ready for local init")

  def _try_run_local_init_op(self, sess):
    """Tries to run _local_init_op, if not None, and is ready for local init.

    Args:
      sess: A `Session`.

    Returns:
      A tuple (is_successful, msg), where is_successful is True if
      _local_init_op is None, or we ran _local_init_op, and False otherwise;
      and msg is a `String` with the reason why the model was not ready to run
      local init.
    """
    if self._local_init_op is not None:
      is_ready_for_local_init, msg = self._model_ready_for_local_init(sess)
      if is_ready_for_local_init:
        logging.info("Running local_init_op.")
        sess.run(self._local_init_op, options=self._local_init_run_options)
        logging.info("Done running local_init_op.")
        return True, None
      else:
        return False, msg
    return True, None


def _ready(op, sess, msg):
  """Checks if the model is ready or not, as determined by op.

  Args:
    op: An op, either _ready_op or _ready_for_local_init_op, which defines the
      readiness of the model.
    sess: A `Session`.
    msg: A message to log to warning if not ready

  Returns:
    A tuple (is_ready, msg), where is_ready is True if ready and False
    otherwise, and msg is `None` if the model is ready, a `String` with the
    reason why it is not ready otherwise.
  """
  if op is None:
    return True, None
  else:
    try:
      ready_value = sess.run(op)
      # The model is considered ready if ready_op returns an empty 1-D tensor.
      # Also compare to `None` and dtype being int32 for backward
      # compatibility.
      if (ready_value is None or ready_value.dtype == np.int32 or
          ready_value.size == 0):
        return True, None
      else:
        # TODO(sherrym): If a custom ready_op returns other types of tensor,
        # or strings other than variable names, this message could be
        # confusing.
        non_initialized_varnames = ", ".join(
            [i.decode("utf-8") for i in ready_value])
        return False, "Variables not initialized: " + non_initialized_varnames
    except errors.FailedPreconditionError as e:
      if "uninitialized" not in str(e):
        logging.warning("%s : error [%s]", msg, str(e))
        raise e
      return False, str(e)


class _CountDownTimer(object):

  def __init__(self, duration_secs):
    self._start_time_secs = time.time()
    self._duration_secs = duration_secs

  def secs_remaining(self):
    diff = self._duration_secs - (time.time() - self._start_time_secs)
    return max(0, diff)
