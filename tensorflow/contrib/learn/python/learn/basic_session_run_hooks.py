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
"""Some common SessionRunHook classes.

@@
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import six

from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.learn.python.learn import session_run_hook
from tensorflow.contrib.learn.python.learn.session_run_hook import SessionRunArgs
from tensorflow.contrib.learn.python.learn.summary_writer_cache import SummaryWriterCache
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import training_util


class LoggingTensorHook(session_run_hook.SessionRunHook):
  """Prints given tensors every N iteration.

  The tensors will be printed to the log, with `INFO` severity.
  """

  def __init__(self, tensors, every_n_iter=100):
    """Initializes a LoggingHook monitor.

    Args:
      tensors: `dict` of tag to tensors/names or
          `iterable` of tensors/names.
      every_n_iter: `int`, print every N iteration.
    """
    if not isinstance(tensors, dict):
      tensors = {item: item for item in tensors}
    self._tensors = tensors
    self._every_n_iter = every_n_iter

  def begin(self):
    self._iter_count = 0
    # Convert names to tensors if given
    self._current_tensors = {tag: _as_graph_element(tensor)
                             for (tag, tensor) in self._tensors.items()}

  def before_run(self, run_context):  # pylint: disable=unused-argument
    if self._iter_count % self._every_n_iter == 0:
      return SessionRunArgs(self._current_tensors)
    else:
      return None

  def after_run(self, run_context, run_values):
    _ = run_context
    if self._iter_count % self._every_n_iter == 0:
      stats = []
      for tag in sorted(self._current_tensors.keys()):
        stats.append("%s = %s" % (tag, run_values.results[tag]))
      logging.info("%s", ", ".join(stats))
    self._iter_count += 1


class StopAtStepHook(session_run_hook.SessionRunHook):
  """Monitor to request stop at a specified step."""

  def __init__(self, num_steps=None, last_step=None):
    """Create a StopAtStep Hook.

    This hook requests stop after either a number of steps have been
    executed or a last step has been reached.  Only of the two options can be
    specified.

    if `num_steps` is specified, it indicates the number of steps to execute
    after `begin()` is called.  If instead `last_step` is specified, it
    indicates the last step we want to execute, as passed to the `after_run()`
    call.

    Args:
      num_steps: Number of steps to execute.
      last_step: Step after which to stop.

    Raises:
      ValueError: If one of the arguments is invalid.
    """
    if num_steps is None and last_step is None:
      raise ValueError("One of num_steps or last_step must be specified.")
    if num_steps is not None and last_step is not None:
      raise ValueError("Only one of num_steps or last_step can be specified.")
    self._num_steps = num_steps
    self._last_step = last_step

  def begin(self):
    self._global_step_tensor = contrib_variables.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError("Global step should be created to use StopAtStepHook.")

  def before_run(self, run_context):  # pylint: disable=unused-argument
    return SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context, run_values):
    global_step = run_values.results
    if self._last_step is None:
      self._last_step = global_step + self._num_steps - 1
    if global_step >= self._last_step:
      run_context.request_stop()


class CheckpointSaverHook(session_run_hook.SessionRunHook):
  """Saves checkpoints every N steps or seconds."""

  def __init__(self,
               checkpoint_dir,
               save_secs=None,
               save_steps=None,
               saver=None,
               checkpoint_basename="model.ckpt",
               scaffold=None):
    """Initialize CheckpointSaverHook monitor.

    Args:
      checkpoint_dir: `str`, base directory for the checkpoint files.
      save_secs: `int`, save every N secs.
      save_steps: `int`, save every N steps.
      saver: `Saver` object, used for saving.
      checkpoint_basename: `str`, base name for the checkpoint files.
      scaffold: `Scaffold`, use to get saver object.

    Raises:
      ValueError: One of `save_steps` or `save_secs` should be set.
    """
    logging.info("Create CheckpointSaverHook")
    self._saver = saver
    self._checkpoint_dir = checkpoint_dir
    self._summary_writer = SummaryWriterCache.get(checkpoint_dir)
    self._save_path = os.path.join(checkpoint_dir, checkpoint_basename)
    self._scaffold = scaffold
    self._save_secs = save_secs
    self._save_steps = save_steps
    self._last_saved_time = None
    self._last_saved_step = None

    if save_steps is None and save_secs is None:
      raise ValueError("Either save_steps or save_secs should be provided")
    if (save_steps is not None) and (save_secs is not None):
      raise ValueError("Can not provide both save_steps and save_secs.")

  def begin(self):
    self._last_saved_time = None
    self._last_saved_step = None
    self._global_step_tensor = contrib_variables.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError(
          "Global step should be created to use CheckpointSaverHook.")

  def before_run(self, run_context):  # pylint: disable=unused-argument
    if self._last_saved_time is None:
      # Write graph in the first call
      training_util.write_graph(
          ops.get_default_graph().as_graph_def(add_shapes=True),
          self._checkpoint_dir,
          "graph.pbtxt")
      self._summary_writer.add_graph(ops.get_default_graph())

    return SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context, run_values):
    global_step = run_values.results
    if self._last_saved_time is None:
      self._save(global_step, run_context.session)

    if self._save_steps is not None:
      if global_step >= self._last_saved_step + self._save_steps:
        self._save(global_step, run_context.session)

    if self._save_secs is not None:
      if time.time() >= self._last_saved_time + self._save_secs:
        self._save(global_step, run_context.session)

  def end(self, session):
    last_step = session.run(contrib_variables.get_global_step())
    self._save(last_step, session)

  def _save(self, step, session):
    """Saves the latest checkpoint."""
    if step == self._last_saved_step:
      return
    logging.info("Saving checkpoints for %d into %s.", step, self._save_path)
    self._last_saved_time = time.time()
    self._last_saved_step = step
    if self._saver is None:
      self._scaffold.saver.save(session, self._save_path, global_step=step)
    else:
      self._saver.save(session, self._save_path, global_step=step)
    self._summary_writer.add_session_log(
        SessionLog(
            status=SessionLog.CHECKPOINT, checkpoint_path=self._save_path),
        step)


class StepCounterHook(session_run_hook.SessionRunHook):
  """Steps per second monitor."""

  def __init__(self, every_n_steps=100, output_dir=None, summary_writer=None):
    self._summary_tag = "global_step/sec"
    self._every_n_steps = every_n_steps
    self._summary_writer = summary_writer
    if summary_writer is None and output_dir:
      self._summary_writer = SummaryWriterCache.get(output_dir)

  def begin(self):
    self._last_reported_time = None
    self._last_reported_step = None
    self._global_step_tensor = contrib_variables.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError(
          "Global step should be created to use StepCounterHook.")

  def before_run(self, run_context):  # pylint: disable=unused-argument
    return SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context, run_values):
    _ = run_context
    if not self._summary_writer:
      return

    global_step = run_values.results
    current_time = time.time()
    if self._last_reported_time is None:
      self._last_reported_step = global_step
      self._last_reported_time = current_time
    else:
      if global_step >= self._every_n_steps + self._last_reported_step:
        added_steps = global_step - self._last_reported_step
        elapsed_time = current_time - self._last_reported_time
        steps_per_sec = added_steps / elapsed_time
        summary = Summary(value=[Summary.Value(
            tag=self._summary_tag, simple_value=steps_per_sec)])
        self._summary_writer.add_summary(summary, global_step)
        self._last_reported_step = global_step
        self._last_reported_time = current_time


class NanLossDuringTrainingError(RuntimeError):

  def __str__(self):
    return "NaN loss during training."


class NanTensorHook(session_run_hook.SessionRunHook):
  """NaN Loss monitor.

  Monitors loss and stops training if loss is NaN.
  Can either fail with exception or just stop training.
  """

  def __init__(self, loss_tensor, fail_on_nan_loss=True):
    """Initializes NanLoss monitor.

    Args:
      loss_tensor: `Tensor`, the loss tensor.
      fail_on_nan_loss: `bool`, whether to raise exception when loss is NaN.
    """
    self._loss_tensor = loss_tensor
    self._fail_on_nan_loss = fail_on_nan_loss

  def before_run(self, run_context):  # pylint: disable=unused-argument
    return SessionRunArgs(self._loss_tensor)

  def after_run(self, run_context, run_values):
    if np.isnan(run_values.results):
      failure_message = "Model diverged with loss = NaN."
      if self._fail_on_nan_loss:
        logging.error(failure_message)
        raise NanLossDuringTrainingError
      else:
        logging.warning(failure_message)
        # We don't raise an error but we request stop without an exception.
        run_context.request_stop()


class SummarySaverHook(session_run_hook.SessionRunHook):
  """Saves summaries every N steps."""

  def __init__(self,
               save_steps=100,
               output_dir=None,
               summary_writer=None,
               scaffold=None,
               summary_op=None):
    """Initializes a `SummarySaver` monitor.

    Args:
      save_steps: `int`, save summaries every N steps. See `EveryN`.
      output_dir: `string`, the directory to save the summaries to. Only used
          if no `summary_writer` is supplied.
      summary_writer: `SummaryWriter`. If `None` and an `output_dir` was passed,
          one will be created accordingly.
      scaffold: `Scaffold` to get summary_op if it's not provided.
      summary_op: `Tensor` of type `string`. A serialized `Summary` protocol
          buffer, as output by TF summary methods like `scalar_summary` or
          `merge_all_summaries`.
    """
    # TODO(ipolosukhin): Implement every N seconds.
    self._summary_op = summary_op
    self._summary_writer = summary_writer
    if summary_writer is None and output_dir:
      self._summary_writer = SummaryWriterCache.get(output_dir)
    self._scaffold = scaffold
    self._save_steps = save_steps
    # TODO(mdan): Throw an error if output_dir and summary_writer are None.

  def begin(self):
    self._last_saved_step = None
    self._request_summary = True
    self._global_step_tensor = contrib_variables.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError(
          "Global step should be created to use SummarySaverHook.")

  def before_run(self, run_context):  # pylint: disable=unused-argument
    requests = {"global_step": self._global_step_tensor}
    if self._request_summary:
      if self._summary_op is not None:
        requests["summary"] = self._summary_op
      elif self._scaffold.summary_op is not None:
        requests["summary"] = self._scaffold.summary_op

    return SessionRunArgs(requests)

  def after_run(self, run_context, run_values):
    _ = run_context
    if not self._summary_writer:
      return

    global_step = run_values.results["global_step"]

    if self._last_saved_step is None:
      self._summary_writer.add_session_log(
          SessionLog(status=SessionLog.START), global_step)

    if self._request_summary:
      self._last_saved_step = global_step
      if "summary" in run_values.results:
        self._summary_writer.add_summary(run_values.results["summary"],
                                         global_step)

    self._request_summary = (
        global_step >= self._last_saved_step + self._save_steps - 1)

  def end(self, session=None):
    if self._summary_writer:
      self._summary_writer.flush()


def _as_graph_element(obj):
  """Retrieves Graph element."""
  graph = ops.get_default_graph()
  if not isinstance(obj, six.string_types):
    if not hasattr(obj, "graph") or obj.graph != graph:
      raise ValueError("Passed %s should have graph attribute that is equal "
                       "to current graph %s." % (obj, graph))
    return obj
  if ":" in obj:
    element = graph.as_graph_element(obj)
  else:
    element = graph.as_graph_element(obj + ":0")
    # Check that there is no :1 (e.g. it's single output).
    try:
      graph.as_graph_element(obj + ":1")
    except (KeyError, ValueError):
      pass
    else:
      raise ValueError("Name %s is ambiguous, "
                       "as this `Operation` has multiple outputs "
                       "(at least 2)." % obj)
  return element
