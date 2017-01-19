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

@@LoggingTensorHook
@@StopAtStepHook
@@CheckpointSaverHook
@@StepCounterHook
@@NanLossDuringTrainingError
@@NanTensorHook
@@SummarySaverHook
@@GlobalStepWaiterHook

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import six

from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.training.summary_io import SummaryWriterCache


class SecondOrStepTimer(object):
  """Timer that triggers at most once every N seconds or once every N steps.
  """

  def __init__(self, every_secs=None, every_steps=None):
    self._every_secs = every_secs
    self._every_steps = every_steps
    self._last_triggered_step = None
    self._last_triggered_time = None

    if self._every_secs is None and self._every_steps is None:
      raise ValueError("Either every_secs or every_steps should be provided.")
    if (self._every_secs is not None) and (self._every_steps is not None):
      raise ValueError("Can not provide both every_secs and every_steps.")

  def should_trigger_for_step(self, step):
    """Return true if the timer should trigger for the specified step.

    Args:
      step: Training step to trigger on.

    Returns:
      True if the difference between the current time and the time of the last
      trigger exceeds `every_secs`, or if the difference between the current
      step and the last triggered step exceeds `every_steps`. False otherwise.
    """
    if self._last_triggered_step == step:
      return False

    if self._last_triggered_step is None:
      return True

    if self._every_secs is not None:
      if time.time() >= self._last_triggered_time + self._every_secs:
        return True

    if self._every_steps is not None:
      if step >= self._last_triggered_step + self._every_steps:
        return True

    return False

  def update_last_triggered_step(self, step):
    """Update the last triggered time and step number.

    Args:
      step: The current step.

    Returns:
      A pair `(elapsed_time, elapsed_steps)`, where `elapsed_time` is the number
      of seconds between the current trigger and the last one (a float), and
      `elapsed_steps` is the number of steps between the current trigger and
      the last one. Both values will be set to `None` on the first trigger.
    """
    current_time = time.time()
    if self._last_triggered_time is None:
      elapsed_secs = None
      elapsed_steps = None
    else:
      elapsed_secs = current_time - self._last_triggered_time
      elapsed_steps = step - self._last_triggered_step

    self._last_triggered_time = current_time
    self._last_triggered_step = step
    return (elapsed_secs, elapsed_steps)

  def last_triggered_step(self):
    return self._last_triggered_step


class LoggingTensorHook(session_run_hook.SessionRunHook):
  """Prints the given tensors once every N local steps or once every N seconds.

  The tensors will be printed to the log, with `INFO` severity.
  """

  def __init__(self, tensors, every_n_iter=None, every_n_secs=None,
               formatter=None):
    """Initializes a LoggingHook monitor.

    Args:
      tensors: `dict` that maps string-valued tags to tensors/tensor names,
          or `iterable` of tensors/tensor names.
      every_n_iter: `int`, print the values of `tensors` once every N local
          steps taken on the current worker.
      every_n_secs: `int` or `float`, print the values of `tensors` once every N
          seconds. Exactly one of `every_n_iter` and `every_n_secs` should be
          provided.
      formatter: function, takes dict of `tag`->`Tensor` and returns a string.
          If `None` uses default printing all tensors.

    Raises:
      ValueError: if `every_n_iter` is non-positive.
    """
    if (every_n_iter is None) == (every_n_secs is None):
      raise ValueError(
          "exactly one of every_n_iter and every_n_secs must be provided.")
    if every_n_iter is not None and every_n_iter <= 0:
      raise ValueError("invalid every_n_iter=%s." % every_n_iter)
    if not isinstance(tensors, dict):
      self._tag_order = tensors
      tensors = {item: item for item in tensors}
    else:
      self._tag_order = tensors.keys()
    self._tensors = tensors
    self._formatter = formatter
    self._timer = SecondOrStepTimer(every_secs=every_n_secs,
                                    every_steps=every_n_iter)

  def begin(self):
    self._iter_count = 0
    # Convert names to tensors if given
    self._current_tensors = {tag: _as_graph_element(tensor)
                             for (tag, tensor) in self._tensors.items()}

  def before_run(self, run_context):  # pylint: disable=unused-argument
    self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
    if self._should_trigger:
      return SessionRunArgs(self._current_tensors)
    else:
      return None

  def after_run(self, run_context, run_values):
    _ = run_context
    if self._should_trigger:
      original = np.get_printoptions()
      np.set_printoptions(suppress=True)
      elapsed_secs, _ = self._timer.update_last_triggered_step(self._iter_count)
      if self._formatter:
        logging.info(self._formatter(run_values.results))
      else:
        stats = []
        for tag in self._tag_order:
          stats.append("%s = %s" % (tag, run_values.results[tag]))
        logging.info("%s (%.3f sec)", ", ".join(stats), elapsed_secs)
      np.set_printoptions(**original)
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
    self._global_step_tensor = training_util.get_global_step()
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


class CheckpointSaverListener(object):
  """An interface for event hooks that depend on a checkpoint.

  CheckpointSaverListeners are similar to SessionRunHooks, and can be useful to
  track training, report progress, and more.  The distinction is that
  CheckpointSaverListeners run only in steps when CheckpointSaverHook is
  triggered, and provide callbacks to run before or after the checkpoint is
  generated.  This is in contrast to SessionRunHooks, which may run in steps
  when no checkpoint is written, and which have no guaranteed execution order
  in any case.  CheckpointSaverListeners use the observer pattern and notify at
  the following points:
   - when a session starts being used
   - before each call to `Saver.save()`
   - after each call to `Saver.save()`
   - when the session closed

  Custom CheckpointSaverListeners look like this:
    class ExampleCheckpointSaverListerner(CheckpointSaverListener):
      def begin(self):
        # You can add ops to the graph here.
        print('Starting the session.')
        self.your_tensor = ...

      def before_save(self, session, global_step_value):
        print('About to write a checkpoint')

      def after_save(self, session, global_step_value):
        print('Done writing checkpoint.')

      def end(self, session, global_step_value):
        print('Done with the session.')

  A CheckpointSaverListener may simply take some action after every checkpoint.
  It is also possible for the listener to use its own schedule to act less
  frequently, based on wall clock time or on global_step_value.  In this case,
  implementors must be careful about what happens at end().  When end is called,
  The CheckpointSaverHook will have already triggered after_save() in the same
  global_step, but the listener may or may not have actually acted on it.
  The listener may want to be sure to act at end() if there is a fresh
  checkpoint available, but should not act twice if after_save() already handled
  it.  In this case, end() should have logic to detect the situation and do the
  right thing, similar to what CheckpointSaverHook.end() does using
  self._timer.last_triggered_step().

  To use such listeners, pass them in the checkpoint_listeners argument to
  graph_actions._monitored_train().  If using tf.Learn Estimators, create a
  custom Estimator and override _get_checkpoint_listeners().
  """

  def begin(self):
    pass

  def before_save(self, session, global_step_value):
    pass

  def after_save(self, session, global_step_value):
    pass

  def end(self, session, global_step_value):
    pass


class CheckpointSaverHook(session_run_hook.SessionRunHook):
  """Saves checkpoints every N steps or seconds."""

  def __init__(self,
               checkpoint_dir,
               save_secs=None,
               save_steps=None,
               saver=None,
               checkpoint_basename="model.ckpt",
               scaffold=None,
               listeners=None):
    """Initialize CheckpointSaverHook monitor.

    Args:
      checkpoint_dir: `str`, base directory for the checkpoint files.
      save_secs: `int`, save every N secs.
      save_steps: `int`, save every N steps.
      saver: `Saver` object, used for saving.
      checkpoint_basename: `str`, base name for the checkpoint files.
      scaffold: `Scaffold`, use to get saver object.
      listeners: List of `CheckpointSaverListener` subclass instances.
        Used for callbacks that run immediately after the corresponding
        CheckpointSaverHook callbacks, only in steps where the
        CheckpointSaverHook was triggered.

    Raises:
      ValueError: One of `save_steps` or `save_secs` should be set.
      ValueError: Exactly one of saver or scaffold should be set.
    """
    logging.info("Create CheckpointSaverHook.")
    if ((saver is None and scaffold is None) or
        (saver is not None and scaffold is not None)):
      raise ValueError("Exactly one of saver or scaffold must be provided.")
    self._saver = saver
    self._checkpoint_dir = checkpoint_dir
    self._save_path = os.path.join(checkpoint_dir, checkpoint_basename)
    self._scaffold = scaffold
    self._timer = SecondOrStepTimer(every_secs=save_secs,
                                    every_steps=save_steps)
    self._listeners = listeners or []

  def begin(self):
    self._summary_writer = SummaryWriterCache.get(self._checkpoint_dir)
    self._global_step_tensor = training_util.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError(
          "Global step should be created to use CheckpointSaverHook.")
    for l in self._listeners:
      l.begin()

  def before_run(self, run_context):  # pylint: disable=unused-argument
    if self._timer.last_triggered_step() is None:
      # We do write graph and saver_def at the first call of before_run.
      # We cannot do this in begin, since we let other hooks to change graph and
      # add variables in begin. Graph is finalized after all begin calls.
      training_util.write_graph(
          ops.get_default_graph().as_graph_def(add_shapes=True),
          self._checkpoint_dir,
          "graph.pbtxt")
      saver_def = self._get_saver().saver_def if self._get_saver() else None
      graph = ops.get_default_graph()
      meta_graph_def = meta_graph.create_meta_graph_def(
          graph_def=graph.as_graph_def(add_shapes=True),
          saver_def=saver_def)
      self._summary_writer.add_graph(graph)
      self._summary_writer.add_meta_graph(meta_graph_def)

    return SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context, run_values):
    global_step = run_values.results
    if self._timer.should_trigger_for_step(global_step):
      self._timer.update_last_triggered_step(global_step)
      self._save(global_step, run_context.session)

  def end(self, session):
    last_step = session.run(training_util.get_global_step())
    if last_step != self._timer.last_triggered_step():
      self._save(last_step, session)
    for l in self._listeners:
      l.end(session, last_step)

  def _save(self, step, session):
    """Saves the latest checkpoint."""
    logging.info("Saving checkpoints for %d into %s.", step, self._save_path)

    for l in self._listeners:
      l.before_save(session, step)

    self._get_saver().save(session, self._save_path, global_step=step)
    self._summary_writer.add_session_log(
        SessionLog(
            status=SessionLog.CHECKPOINT, checkpoint_path=self._save_path),
        step)

    for l in self._listeners:
      l.after_save(session, step)

  def _get_saver(self):
    if self._saver is not None:
      return self._saver
    elif self._scaffold is not None:
      return self._scaffold.saver
    return None


class StepCounterHook(session_run_hook.SessionRunHook):
  """Steps per second monitor."""

  def __init__(self,
               every_n_steps=100,
               every_n_secs=None,
               output_dir=None,
               summary_writer=None):

    if (every_n_steps is None) == (every_n_secs is None):
      raise ValueError(
          "exactly one of every_n_steps and every_n_secs should be provided.")
    self._timer = SecondOrStepTimer(every_steps=every_n_steps,
                                    every_secs=every_n_secs)

    self._summary_writer = summary_writer
    self._output_dir = output_dir

  def begin(self):
    if self._summary_writer is None and self._output_dir:
      self._summary_writer = SummaryWriterCache.get(self._output_dir)
    self._global_step_tensor = training_util.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError(
          "Global step should be created to use StepCounterHook.")
    self._summary_tag = self._global_step_tensor.op.name + "/sec"

  def before_run(self, run_context):  # pylint: disable=unused-argument
    return SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context, run_values):
    _ = run_context

    global_step = run_values.results
    if self._timer.should_trigger_for_step(global_step):
      elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(
          global_step)
      if elapsed_time is not None:
        steps_per_sec = elapsed_steps / elapsed_time
        if self._summary_writer is not None:
          summary = Summary(value=[Summary.Value(
              tag=self._summary_tag, simple_value=steps_per_sec)])
          self._summary_writer.add_summary(summary, global_step)
        logging.info("%s: %g", self._summary_tag, steps_per_sec)


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
               save_steps=None,
               save_secs=None,
               output_dir=None,
               summary_writer=None,
               scaffold=None,
               summary_op=None):
    """Initializes a `SummarySaver` monitor.

    Args:
      save_steps: `int`, save summaries every N steps. Exactly one of
          `save_secs` and `save_steps` should be set.
      save_secs: `int`, save summaries every N seconds.
      output_dir: `string`, the directory to save the summaries to. Only used
          if no `summary_writer` is supplied.
      summary_writer: `SummaryWriter`. If `None` and an `output_dir` was passed,
          one will be created accordingly.
      scaffold: `Scaffold` to get summary_op if it's not provided.
      summary_op: `Tensor` of type `string` containing the serialized `Summary`
          protocol buffer or a list of `Tensor`. They are most likely an output
          by TF summary methods like `tf.summary.scalar` or
          `tf.summary.merge_all`. It can be passed in as one tensor; if more
          than one, they must be passed in as a list.

    Raises:
      ValueError: Exactly one of scaffold or summary_op should be set.
    """
    if ((scaffold is None and summary_op is None) or
        (scaffold is not None and summary_op is not None)):
      raise ValueError(
          "Exactly one of scaffold or summary_op must be provided.")
    self._summary_op = summary_op
    self._summary_writer = summary_writer
    self._output_dir = output_dir
    self._scaffold = scaffold
    self._timer = SecondOrStepTimer(every_secs=save_secs,
                                    every_steps=save_steps)
    # TODO(mdan): Throw an error if output_dir and summary_writer are None.

  def begin(self):
    if self._summary_writer is None and self._output_dir:
      self._summary_writer = SummaryWriterCache.get(self._output_dir)
    self._next_step = None
    self._global_step_tensor = training_util.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError(
          "Global step should be created to use SummarySaverHook.")

  def before_run(self, run_context):  # pylint: disable=unused-argument
    self._request_summary = (
        self._next_step is None or
        self._timer.should_trigger_for_step(self._next_step))
    requests = {"global_step": self._global_step_tensor}
    if self._request_summary:
      if self._get_summary_op() is not None:
        requests["summary"] = self._get_summary_op()

    return SessionRunArgs(requests)

  def after_run(self, run_context, run_values):
    _ = run_context
    if not self._summary_writer:
      return

    global_step = run_values.results["global_step"]

    if self._next_step is None:
      self._summary_writer.add_session_log(
          SessionLog(status=SessionLog.START), global_step)

    if self._request_summary:
      self._timer.update_last_triggered_step(global_step)
      if "summary" in run_values.results:
        for summary in run_values.results["summary"]:
          self._summary_writer.add_summary(summary, global_step)

    self._next_step = global_step + 1

  def end(self, session=None):
    if self._summary_writer:
      self._summary_writer.flush()

  def _get_summary_op(self):
    """Fetches the summary op either from self._summary_op or self._scaffold.

    Returns:
      Returns a list of summary `Tensor`.
    """
    summary_op = None
    if self._summary_op is not None:
      summary_op = self._summary_op
    elif self._scaffold.summary_op is not None:
      summary_op = self._scaffold.summary_op

    if summary_op is None:
      return None

    if not isinstance(summary_op, list):
      return [summary_op]
    return summary_op


class GlobalStepWaiterHook(session_run_hook.SessionRunHook):
  """Delay execution until global step reaches to wait_until_step.

  This hook delays execution until global step reaches to `wait_until_step`. It
  is used to gradually start workers in distributed settings. One example usage
  would be setting `wait_until_step=int(K*log(task_id+1))` assuming that
  task_id=0 is the chief.
  """

  def __init__(self, wait_until_step):
    """Create a _GlobalStepWaiterHook.

    Args:
      wait_until_step: an `int` shows until which global step should we wait.
    """
    self._wait_until_step = wait_until_step

  def begin(self):
    self._worker_is_started = False
    self._global_step_tensor = training_util.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError(
          "Global step should be created to use _GlobalStepWaiterHook.")

  def before_run(self, run_context):
    if self._worker_is_started:
      return None

    if self._wait_until_step <= 0:
      self._worker_is_started = True
      return None

    logging.info("Waiting for global step %d before starting training.",
                 self._wait_until_step)
    last_logged_step = 0
    while True:
      current_step = run_context.session.run(self._global_step_tensor)
      if current_step >= self._wait_until_step:
        self._worker_is_started = True
        return None
      if current_step - last_logged_step > 1000:
        logging.info("Waiting for global step %d before starting training. "
                     "Current step is %d.", self._wait_until_step, current_step)
        last_logged_step = current_step
      time.sleep(0.5)


class FinalOpsHook(session_run_hook.SessionRunHook):
  """A run hook which evaluates `Tensors` at the end of a session."""

  def __init__(self, final_ops, final_ops_feed_dict=None):
    """Constructs the FinalOpHook with ops to run at the end of the session.

    Args:
      final_ops: A single `Tensor`, a list of `Tensors` or a dictionary of
        names to `Tensors`.
      final_ops_feed_dict: A feed dictionary to use when running
        `final_ops_dict`.
    """
    self._final_ops = final_ops
    self._final_ops_feed_dict = final_ops_feed_dict
    self._final_ops_values = None

  @property
  def final_ops_values(self):
    return self._final_ops_values

  def end(self, session):
    if self._final_ops is not None:
      self._final_ops_values = session.run(self._final_ops,
                                           feed_dict=self._final_ops_feed_dict)


class FeedFnHook(session_run_hook.SessionRunHook):
  """Runs `feed_fn` and sets the `feed_dict` accordingly."""

  def __init__(self, feed_fn):
    """Constructs the FeedFnHook with given `feed_fn`.

    Args:
      feed_fn: function, no arguments and returns `dict` to feed.
    """
    self.feed_fn = feed_fn

  def before_run(self, run_context):  # pylint: disable=unused-argument
    return session_run_hook.SessionRunArgs(
        fetches=None, feed_dict=self.feed_fn())


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
