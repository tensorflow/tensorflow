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
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver as saver_mod


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

  def __init__(self, local_init_op=None, ready_op=None,
               graph=None, recovery_wait_secs=30):
    """Creates a SessionManager.

    The `local_init_op` is an `Operation` that is run always after a new session
    was created. If `None`, this step is skipped.

    The `ready_op` is an `Operation` used to check if the model is ready.  The
    model is considered ready if that operation returns an empty string tensor.
    If the operation returns non empty string tensor, the elements are
    concatenated and used to indicate to the user why the model is not ready.

    If `ready_op` is `None`, the model is not checked for readiness.

    `recovery_wait_secs` is the number of seconds between checks that
    the model is ready.  It is used by processes to wait for a model to
    be initialized or restored.  Defaults to 30 seconds.

    Args:
      local_init_op: An `Operation` run immediately after session creation.
         Usually used to initialize tables and local variables.
      ready_op: An `Operation` to check if the model is initialized.
      graph: The `Graph` that the model will use.
      recovery_wait_secs: Seconds between checks for the model to be ready.
    """
    # Sets default values of arguments.
    if graph is None:
      graph = ops.get_default_graph()
    self._local_init_op = local_init_op
    self._ready_op = ready_op
    self._graph = graph
    self._recovery_wait_secs = recovery_wait_secs
    self._target = None

  def prepare_session(self, master, init_op=None, saver=None,
                      checkpoint_dir=None, wait_for_checkpoint=False,
                      max_wait_secs=7200, config=None, init_feed_dict=None,
                      init_fn=None):
    """Creates a `Session`. Makes sure the model is ready to be used.

    Creates a `Session` on 'master'. If a `saver` object is passed in, and
    `checkpoint_dir` points to a directory containing valid checkpoint
    files, then it will try to recover the model from checkpoint. If
    no checkpoint files are available, and `wait_for_checkpoint` is
    `True`, then the process would check every `recovery_wait_secs`,
    up to `max_wait_secs`, for recovery to succeed.

    If the model cannot be recovered successfully then it is initialized by
    either running the provided `init_op`, or calling the provided `init_fn`.
    It is an error if the model cannot be recovered and neither an `init_op`
    or an `init_fn` are passed.

    This is a convenient function for the following, with a few error checks
    added:

    ```python
    sess, initialized = self.recover_session(master)
    if not initialized:
      if init_op:
        sess.run(init_op, feed_dict=init_feed_dict)
      if init_fn;
        init_fn(sess)
    return sess
    ```

    Args:
      master: `String` representation of the TensorFlow master to use.
      init_op: Optional `Operation` used to initialize the model.
      saver: A `Saver` object used to restore a model.
      checkpoint_dir: Path to the checkpoint files.
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
    """
    sess, initialized = self.recover_session(
        master, saver, checkpoint_dir=checkpoint_dir,
        wait_for_checkpoint=wait_for_checkpoint,
        max_wait_secs=max_wait_secs, config=config)
    if not initialized:
      if not init_op and not init_fn:
        raise RuntimeError("Model is not initialized and no init_op or "
                           "init_fn was given")
      if init_op:
        sess.run(init_op, feed_dict=init_feed_dict)
      if init_fn:
        init_fn(sess)
      not_ready = self._model_not_ready(sess)
      if not_ready:
        raise RuntimeError("Init operations did not make model ready.  "
                           "Init op: %s, init fn: %s, error: %s"
                           % (init_op.name, init_fn, not_ready))
    return sess

  def recover_session(self, master, saver=None, checkpoint_dir=None,
                      wait_for_checkpoint=False, max_wait_secs=7200,
                      config=None):
    """Creates a `Session`, recovering if possible.

    Creates a new session on 'master'.  If the session is not initialized
    and can be recovered from a checkpoint, recover it.

    Args:
      master: `String` representation of the TensorFlow master to use.
      saver: A `Saver` object used to restore a model.
      checkpoint_dir: Path to the checkpoint files.
      wait_for_checkpoint: Whether to wait for checkpoint to become available.
      max_wait_secs: Maximum time to wait for checkpoints to become available.
      config: Optional `ConfigProto` proto used to configure the session.

    Returns:
      A pair (sess, initialized) where 'initialized' is `True` if
      the session could be recovered, `False` otherwise.
    """
    self._target = master
    sess = session.Session(self._target, graph=self._graph, config=config)
    if self._local_init_op:
      sess.run([self._local_init_op])

    # If either saver or checkpoint_dir is not specified, cannot restore. Just
    # return.
    if not saver or not checkpoint_dir:
      not_ready = self._model_not_ready(sess)
      return sess, not_ready is None

    # Waits up until max_wait_secs for checkpoint to become available.
    wait_time = 0
    ckpt = saver_mod.get_checkpoint_state(checkpoint_dir)
    while not ckpt or not ckpt.model_checkpoint_path:
      if wait_for_checkpoint and wait_time < max_wait_secs:
        logging.info("Waiting for checkpoint to be available.")
        time.sleep(self._recovery_wait_secs)
        wait_time += self._recovery_wait_secs
        ckpt = saver_mod.get_checkpoint_state(checkpoint_dir)
      else:
        return sess, False

    # Loads the checkpoint and verifies that it makes the model ready.
    saver.restore(sess, ckpt.model_checkpoint_path)
    last_checkpoints = []
    for fname in ckpt.all_model_checkpoint_paths:
      fnames = gfile.Glob(fname)
      if fnames:
        mtime = gfile.Stat(fnames[0]).mtime
        last_checkpoints.append((fname, mtime))
    saver.set_last_checkpoints_with_time(last_checkpoints)
    not_ready = self._model_not_ready(sess)
    if not_ready:
      logging.info("Restoring model from %s did not make model ready: %s",
                   ckpt.model_checkpoint_path, not_ready)
      return sess, False
    else:
      logging.info("Restored model from %s", ckpt.model_checkpoint_path)
      return sess, True

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
      if self._local_init_op:
        sess.run([self._local_init_op])
      not_ready = self._model_not_ready(sess)
      if not not_ready:
        return sess

      self._safe_close(sess)

      # Do we have enough time left to try again?
      remaining_ms_after_wait = (
          timer.secs_remaining() - self._recovery_wait_secs)
      if remaining_ms_after_wait < 0:
        raise errors.DeadlineExceededError(
            None, None,
            "Session was not ready after waiting %d secs." % (max_wait_secs,))

      logging.info("Waiting for model to be ready: %s", not_ready)
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

  def _model_not_ready(self, sess):
    """Checks if the model is ready or not.

    Args:
      sess: A `Session`.

    Returns:
      `None` if the model is ready, a `String` with the reason why it is not
      ready otherwise.
    """
    if self._ready_op is None:
      return None
    else:
      try:
        ready_value = sess.run(self._ready_op)
        # The model is considered ready if ready_op returns an empty 1-D tensor.
        # Also compare to `None` and dtype being int32 for backward
        # compatibility.
        if (ready_value is None or ready_value.dtype == np.int32 or
            ready_value.size == 0):
          return None
        else:
          # TODO(sherrym): If a custom ready_op returns other types of tensor,
          # or strings other than variable names, this message could be
          # confusing.
          non_initialized_varnames = ", ".join(
              [i.decode("utf-8") for i in ready_value])
          return "Variables not initialized: " + non_initialized_varnames
      except errors.FailedPreconditionError as e:
        if "uninitialized" not in str(e):
          logging.warning("Model not ready raised: %s", str(e))
          raise  e
        return str(e)


class _CountDownTimer(object):

  def __init__(self, duration_secs):
    self._start_time_secs = time.time()
    self._duration_secs = duration_secs

  def secs_remaining(self):
    diff = self._duration_secs - (time.time() - self._start_time_secs)
    return max(0, diff)
