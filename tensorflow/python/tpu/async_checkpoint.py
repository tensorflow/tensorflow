# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ======================================
"""Hook for asynchronous checkpointing.

This hook dispatches checkpoint writing operations in a separate thread to
allow execution to continue on the main thread.
"""

import os
import threading
import time
from typing import Any, List, Optional, Text

from tensorflow.core.util import event_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training.summary_io import SummaryWriterCache


# Captures the timestamp of the first Saver object instantiation or end of a
# save operation. Can be accessed by multiple Saver instances.
_END_TIME_OF_LAST_WRITE = None
_END_TIME_OF_LAST_WRITE_LOCK = threading.Lock()

# API label for cell names used in TF1 async checkpoint metrics.
_ASYNC_CHECKPOINT_V1 = "async_checkpoint_v1"


def _get_duration_microseconds(start_time_seconds, end_time_seconds) -> int:
  """Returns the duration between start and end time in microseconds."""
  return max(int((end_time_seconds - start_time_seconds) * 1000000), 0)


class AsyncCheckpointSaverHook(basic_session_run_hooks.CheckpointSaverHook):
  """Saves checkpoints every N steps or seconds."""

  def __init__(self,
               checkpoint_dir: Text,
               save_secs: Optional[int] = None,
               save_steps: Optional[int] = None,
               saver: Optional[saver_lib.Saver] = None,
               checkpoint_basename: Text = "model.ckpt",
               scaffold: Optional[monitored_session.Scaffold] = None,
               listeners: Optional[List[
                   basic_session_run_hooks.CheckpointSaverListener]] = None):
    """Initializes a `CheckpointSaverHook`.

    Args:
      checkpoint_dir: `str`, base directory for the checkpoint files.
      save_secs: `int`, save every N secs.
      save_steps: `int`, save every N steps.
      saver: `Saver` object, used for saving.
      checkpoint_basename: `str`, base name for the checkpoint files.
      scaffold: `Scaffold`, use to get saver object.
      listeners: List of `CheckpointSaverListener` subclass instances. Used for
        callbacks that run immediately before or after this hook saves the
        checkpoint.

    Raises:
      ValueError: One of `save_steps` or `save_secs` should be set.
      ValueError: At most one of `saver` or `scaffold` should be set.
    """
    save_path = os.path.join(checkpoint_dir, checkpoint_basename)
    logging.info("Create AsyncCheckpointSaverHook saving to path\n%s",
                 save_path)
    if listeners:
      logging.info(" with %d listener(s).", len(listeners))
    if saver is not None and scaffold is not None:
      raise ValueError("You cannot provide both saver and scaffold.")
    self._saver = saver
    self._save_thread = None
    self._write_graph_thread = None
    self._checkpoint_dir = checkpoint_dir
    self._save_path = save_path
    self._scaffold = scaffold
    self._timer = basic_session_run_hooks.SecondOrStepTimer(
        every_secs=save_secs, every_steps=save_steps)
    self._listeners = listeners or []
    self._steps_per_run = 1
    self._summary_writer = None
    self._global_step_tensor = None

    self._last_checkpoint_step = None

    # Initialize the first timestamp for _END_TIME_OF_LAST_WRITE.
    global _END_TIME_OF_LAST_WRITE
    with _END_TIME_OF_LAST_WRITE_LOCK:
      if _END_TIME_OF_LAST_WRITE is None:
        _END_TIME_OF_LAST_WRITE = time.time()

  def _set_steps_per_run(self, steps_per_run):
    self._steps_per_run = steps_per_run

  def begin(self):
    self._summary_writer = SummaryWriterCache.get(self._checkpoint_dir)
    self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
    if self._global_step_tensor is None:
      raise RuntimeError(
          "Global step should be created to use CheckpointSaverHook.")
    for l in self._listeners:
      l.begin()

  def after_create_session(self, session: session_lib.Session, coord: Any):
    global_step = session.run(self._global_step_tensor)

    # We do write graph and saver_def at the first call of before_run.
    # We cannot do this in begin, since we let other hooks to change graph and
    # add variables in begin. Graph is finalized after all begin calls.
    def _write_graph_fn(self):
      training_util.write_graph(
          ops.get_default_graph().as_graph_def(add_shapes=True),
          self._checkpoint_dir, "graph.pbtxt")
    self._write_graph_thread = threading.Thread(target=_write_graph_fn,
                                                args=[self])
    self._write_graph_thread.start()

    saver_def = self._get_saver().saver_def if self._get_saver() else None
    graph = ops.get_default_graph()
    meta_graph_def = meta_graph.create_meta_graph_def(
        graph_def=graph.as_graph_def(add_shapes=True), saver_def=saver_def)
    self._summary_writer.add_graph(graph)
    self._summary_writer.add_meta_graph(meta_graph_def)
    # The checkpoint saved here is the state at step "global_step".
    self._save(session, global_step)
    self._timer.update_last_triggered_step(global_step)

  def before_run(self, run_context: Any):  # pylint: disable=unused-argument
    return session_run_hook.SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context: session_run_hook.SessionRunContext,
                run_values: Any):
    global_step = run_context.session.run(self._global_step_tensor)
    if self._timer.should_trigger_for_step(global_step):
      self._timer.update_last_triggered_step(global_step)
      logging.info("Triggering checkpoint. %s", global_step)
      if self._save(run_context.session, global_step):
        run_context.request_stop()

  def end(self, session: session_lib.Session):
    if self._save_thread:
      logging.info("Waiting for any pending checkpoints to finish.")
      self._save_thread.join()
    if self._write_graph_thread:
      logging.info("Waiting for any pending write_graph to finish.")
      self._write_graph_thread.join()

    last_step = session.run(self._global_step_tensor)

    if self._last_checkpoint_step != last_step:
      self._save(session, last_step, asynchronous=False)

    for l in self._listeners:
      l.end(session, last_step)

  def _save(self, session, step, asynchronous=True):
    """Saves the latest checkpoint, returns should_stop."""

    def _save_fn():
      """Run the saver process."""
      logging.info("Saving checkpoints for %d into %s.", step, self._save_path)

      start_time = time.time()
      for l in self._listeners:
        l.before_save(session, step)

      self._get_saver().save(session, self._save_path, global_step=step)
      self._summary_writer.add_session_log(
          event_pb2.SessionLog(
              status=event_pb2.SessionLog.CHECKPOINT,
              checkpoint_path=self._save_path), step)

      for l in self._listeners:
        l.after_save(session, step)

      # Measure the async checkpoint write duration, i.e., non-blocking time.
      end_time = time.time()
      metrics.AddAsyncCheckpointWriteDuration(
          api_label=_ASYNC_CHECKPOINT_V1,
          microseconds=_get_duration_microseconds(start_time, end_time))

      # Measure the elapsed time since the last checkpoint.
      # Due to the nature of async checkpoint, here it actually captures the
      # duration between the start_time of the previous checkpoint and the start
      # time of this checkpoint. As a result, the duration of the final async
      # checkpoint is excluded, which is fine since it does not take much time.
      global _END_TIME_OF_LAST_WRITE
      with _END_TIME_OF_LAST_WRITE_LOCK:
        metrics.AddTrainingTimeSaved(
            api_label=_ASYNC_CHECKPOINT_V1,
            microseconds=_get_duration_microseconds(_END_TIME_OF_LAST_WRITE,
                                                    start_time))
      _END_TIME_OF_LAST_WRITE = start_time

      logging.info("Checkpoint actual writing time: (%.3f sec)",
                   end_time - start_time)
      logging.info("Checkpoint finished for %d into %s.", step, self._save_path)

    # Measure the checkpoint write duration that is blocking the main thread.
    blocking_start_time = time.time()
    def end_of_blocking_time():
      blocking_end_time = time.time()
      metrics.AddCheckpointWriteDuration(
          api_label=_ASYNC_CHECKPOINT_V1,
          microseconds=_get_duration_microseconds(blocking_start_time,
                                                  blocking_end_time))

    if not asynchronous:
      self._last_checkpoint_step = step
      _save_fn()
      end_of_blocking_time()
      return

    if self._save_thread is not None:
      self._save_thread.join(timeout=0.1)
      if self._save_thread.is_alive():
        logging.info("Saver thread still in progress, skipping checkpoint.")
        end_of_blocking_time()
        return

    self._last_checkpoint_step = step
    self._save_thread = threading.Thread(target=_save_fn)
    self._save_thread.start()
    end_of_blocking_time()

  def _get_saver(self):
    if self._saver is not None:
      return self._saver
    elif self._scaffold is not None:
      return self._scaffold.saver

    # Get saver from the SAVERS collection if present.
    collection_key = ops.GraphKeys.SAVERS
    savers = ops.get_collection(collection_key)
    if not savers:
      raise RuntimeError(
          "No items in collection {}. Please add a saver to the collection "
          "or provide a saver or scaffold.".format(collection_key))
    elif len(savers) > 1:
      raise RuntimeError(
          "More than one item in collection {}. "
          "Please indicate which one to use by passing it to the constructor."
          .format(collection_key))

    self._saver = savers[0]
    return savers[0]
