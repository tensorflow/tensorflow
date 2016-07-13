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

"""Monitors allow user instrumentaiton of the training process.

Monitors are useful to track training, report progress, request early
stopping and more. Monitors use the observer pattern and notify at the following
points:
 - when training begins
 - before a training step
 - after a training step
 - when training ends

Monitors are not intended to be reusable.

There are a few pre-defined monitors:
 - CaptureVariable: saves a variable's values
 - GraphDump: intended for debug only - saves all tensor values
 - PrintTensor: outputs one or more tensor values to log
 - SummarySaver: saves summaries to a summary writer
 - ValidationMonitor: runs model validation, by periodically calculating eval
     metrics on a separate data set; supports optional early stopping

For more specific needs, you can create custom monitors by extending one of the
following classes:
 - BaseMonitor: the base class for all monitors
 - EveryN: triggers a callback every N training steps

Example:

  class ExampleMonitor(monitors.BaseMonitor):
    def __init__(self):
      print 'Init'

    def begin(self, max_steps):
      print 'Starting run. Will train until step %d.' % max_steps

    def end(self):
      print 'Completed run.'

    def step_begin(self, step):
      print 'About to run step %d...' % step
      return ['loss_1:0']

    def step_end(self, step, outputs):
      print 'Done running step %d. The value of "loss" tensor: %s' % (
        step, outputs['loss_1:0'])

  linear_regressor = LinearRegressor()
  example_monitor = ExampleMonitor()
  linear_regressor.fit(
    x, y, steps=2, batch_size=1, monitors=[example_monitor])

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six

from tensorflow.contrib.learn.python.learn.utils import export
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver
from tensorflow.python.training import summary_io


# TODO(ptucker): Split each monitor class into a separate file.
# TODO(ptucker): Fail if epoch or step does not monotonically increase?
class BaseMonitor(object):
  """Base class for Monitors.

  Defines basic interfaces of Monitors.
  """

  def __init__(self):
    self._begun = False
    self._current_epoch = None
    self._current_step = None
    self._max_steps = None
    self._estimator = None

  def set_estimator(self, estimator):
    """A setter called automatically by the target estimator.

    Args:
      estimator: the estimator that this monitor monitors.

    Raises:
      ValueError: if the estimator is None.
    """
    if estimator is None:
      raise ValueError("Missing estimator.")
    # TODO(mdan): This should fail if called twice with the same estimator.
    self._estimator = estimator

  def begin(self, max_steps=None):
    """Called at the beginning of training.

    Args:
      max_steps: `int`, the maximum global step this training will run until.

    Raises:
      ValueError: if we've already begun a run.
    """
    if self._begun:
      raise ValueError("begin called twice without end.")
    self._max_steps = max_steps
    self._begun = True

  def end(self):
    """Callback at the end of training/evaluation.

    Raises:
      ValueError: if we've not begun a run.
    """
    if not self._begun:
      raise ValueError("end called without begin.")
    self._max_steps = None
    self._begun = False

  def epoch_begin(self, epoch):
    """Begin epoch.

    Args:
      epoch: `int`, the epoch number.

    Raises:
      ValueError: if we've already begun an epoch, or `epoch` < 0.
    """
    if self._current_epoch is not None:
      raise ValueError("epoch_begin called twice without epoch_end.")
    if epoch < 0:
      raise ValueError("Invalid epoch %s." % epoch)
    self._current_epoch = epoch

  def epoch_end(self, epoch):
    """End epoch.

    Args:
      epoch: `int`, the epoch number.

    Raises:
      ValueError: if we've not begun an epoch, or `epoch` number does not match.
    """
    if self._current_epoch != epoch:
      raise ValueError(
          "epoch_end expected %s but got %s.", self._current_epoch, epoch)
    self._current_epoch = None

  def step_begin(self, step):
    """Callback before training step begins.

    You may use this callback to request evaluation of additional tensors
    in the graph.

    Args:
      step: `int`, the current value of the global step.

    Returns:
      List of `Tensor` objects or string tensor names to be run.

    Raises:
      ValueError: if we've already begun a step, or `step` < 0, or
          `step` > `max_steps`.
    """
    if self._current_step is not None:
      raise ValueError("step_begin called twice without step_end.")
    if (step < 0) or (
        (self._max_steps is not None) and (step > self._max_steps)):
      raise ValueError("Invalid step %s." % step)
    self._current_step = step
    return []

  def step_end(self, step, output):  # pylint: disable=unused-argument
    """Callback after training step finished.

    This callback provides access to the tensors/ops evaluated at this step,
    including the additional tensors for which evaluation was requested in
    `step_begin`.

    In addition, the callback has the opportunity to stop training by returning
    `True`. This is useful for early stopping, for example.

    Args:
      step: `int`, the current value of the global step.
      output: `dict` mapping `string` values representing tensor names to
        the value resulted from running these tensors. Values may be either
        scalars, for scalar tensors, or Numpy `array`, for non-scalar tensors.

    Returns:
      `bool`. True if training should stop.

    Raises:
      ValueError: if we've not begun a step, or `step` number does not match.
    """
    if self._current_step != step:
      raise ValueError(
          "step_end expected %s but got %s.", self._current_step, step)
    self._current_step = None
    return False


class EveryN(BaseMonitor):
  """Base class for monitors that execute callbacks every n steps.

  This class adds two new callbacks:
    - every_n_step_begin
    - every_n_step_end

  The callbacks are executed every n steps, or optionally every step for the
  first m steps, where m and n can both be user-specified.

  When extending this class, note that if you wish to use any of the
  `BaseMonitor` callbacks, you must call their respective super implementation:

    def step_begin(self, step):
      super(ExampleMonitor, self).step_begin(step)
      return []

  Failing to call the super implementation will cause unpredictible behavior.

  """
  # TODO(ipolosukhin): Add also every n seconds.

  def __init__(self, every_n_steps=100, first_n_steps=1):
    """Initialized an `EveryN` monitor.

    Args:
      every_n_steps: `int`, the number of steps to allow between callbacks.
      first_n_steps: `int`, specifying the number of initial steps during
        which the callbacks will always be executed, regardless of the value
        of `every_n_steps`. Note that this value is relative to the global step
    """
    super(EveryN, self).__init__()
    self._every_n_steps = every_n_steps
    self._first_n_steps = first_n_steps
    self._last_step = 0
    self._active_step = None

  def begin(self, max_steps=None):
    """Overrides `BaseMonitor.begin`.

    When overriding this method, you must call the super implementation.

    Args:
      max_steps: `int`, the maximum global step value.
    """
    super(EveryN, self).begin(max_steps)

  def every_n_step_begin(self, step):  # pylint: disable=unused-argument
    """Callback before every n'th step begins.

    Args:
      step: `int`, the current value of the global step.

    Returns:
      A `list` of tensors that will be evaluated at this step.
    """
    return []

  def every_n_step_end(self, step, outputs):  # pylint: disable=unused-argument
    """Callback after every n'th step finished.

    This callback provides access to the tensors/ops evaluated at this step,
    including the additional tensors for which evaluation was requested in
    `step_begin`.

    In addition, the callback has the opportunity to stop training by returning
    `True`. This is useful for early stopping, for example.

    Args:
      step: `int`, the current value of the global step.
      outputs: `dict` mapping `string` values representing tensor names to
        the value resulted from running these tensors. Values may be either
        scalars, for scalar tensors, or Numpy `array`, for non-scalar tensors.

    Returns:
      `bool`. True if training should stop.
    """
    return False

  def step_begin(self, step):
    """Overrides `BaseMonitor.step_begin`.

    When overriding this method, you must call the super implementation.

    Args:
      step: `int`, the current value of the global step.
    Returns:
      A `list`, the result of every_n_step_begin, if that was called this step,
      or an empty list otherwise.

    Raises:
      ValueError: if called more than once during a step.
    """
    super(EveryN, self).step_begin(step)
    if (step <= self._first_n_steps or
        step >= (self._every_n_steps + self._last_step) or
        step == self._max_steps):
      if self._active_step is not None:
        raise ValueError(
            "Starting step %s, %s still active." % (step, self._active_step))
      self._active_step = step
      return self.every_n_step_begin(step)
    return []

  def step_end(self, step, output):
    """Overrides `BaseMonitor.step_end`.

    When overriding this method, you must call the super implementation.

    Args:
      step: `int`, the current value of the global step.
      output: `dict` mapping `string` values representing tensor names to
        the value resulted from running these tensors. Values may be either
        scalars, for scalar tensors, or Numpy `array`, for non-scalar tensors.
    Returns:
      `bool`, the result of every_n_step_end, if that was called this step,
      or `False` otherwise.
    """
    super(EveryN, self).step_end(step, output)
    to_stop = False
    if (self._active_step is not None) and (self._active_step == step):
      self._last_step = step
      to_stop = self.every_n_step_end(step, output)
      self._active_step = None
    return to_stop


# TODO(ptucker): Rename to LoggingTensor since it's not writing to stdout.
class PrintTensor(EveryN):
  """Prints given tensors every N steps.

  This is an `EveryN` monitor and has consistent semantic for `every_n`
  and `first_n`.

  The tensors will be printed to the log, with `INFO` severity.
  """

  def __init__(self, tensor_names, every_n=100, first_n=1):
    """Initializes a PrintTensor monitor.

    Args:
      tensor_names: `dict` of tag to tensor names or
          `iterable` of tensor names (strings).
      every_n: `int`, print every N steps. See `PrintN.`
      first_n: `int`, also print the first N steps. See `PrintN.`
    """
    super(PrintTensor, self).__init__(every_n, first_n)
    if not isinstance(tensor_names, dict):
      tensor_names = {item: item for item in tensor_names}
    self._tensor_names = tensor_names

  def every_n_step_begin(self, step):
    super(PrintTensor, self).every_n_step_begin(step)
    return list(self._tensor_names.values())

  def every_n_step_end(self, step, outputs):
    super(PrintTensor, self).every_n_step_end(step, outputs)
    stats = []
    for tag, tensor_name in six.iteritems(self._tensor_names):
      if tensor_name in outputs:
        stats.append("%s = %s" % (tag, str(outputs[tensor_name])))
    logging.info("Step %d: %s", step, ", ".join(stats))


class SummarySaver(EveryN):
  """Saves summaries every N steps."""

  def __init__(self, summary_op, save_steps=100, output_dir=None,
               summary_writer=None):
    """Initializes a `SummarySaver` monitor.

    Args:
      summary_op: `Tensor` of type `string`. A serialized `Summary` protocol
          buffer, as output by TF summary methods like `scalar_summary` or
          `merge_all_summaries`.
      save_steps: `int`, save summaries every N steps. See `EveryN`.
      output_dir: `string`, the directory to save the summaries to. Only used
          if no `summary_writer` is supplied.
      summary_writer: `SummaryWriter`. If `None` and an `output_dir` was passed,
          one will be created accordingly.
    """
    # TODO(ipolosukhin): Implement every N seconds.
    super(SummarySaver, self).__init__(every_n_steps=save_steps)
    self._summary_op = summary_op
    self._summary_writer = summary_writer
    if summary_writer is None and output_dir:
      self._summary_writer = summary_io.SummaryWriter(output_dir)
    # TODO(mdan): Throw an error if output_dir and summary_writer are None.

  def set_estimator(self, estimator):
    super(SummarySaver, self).set_estimator(estimator)
    # TODO(mdan): This line looks redundant.
    if self._summary_writer is None:
      self._summary_writer = summary_io.SummaryWriter(estimator.model_dir)

  def every_n_step_begin(self, step):
    super(SummarySaver, self).every_n_step_begin(step)
    return [self._summary_op]

  def every_n_step_end(self, step, outputs):
    super(SummarySaver, self).every_n_step_end(step, outputs)
    summary_strs = outputs[self._summary_op.name]
    if self._summary_writer:
      self._summary_writer.add_summary(summary_strs, step)
    return False

  def end(self):
    super(SummarySaver, self).end()
    if self._summary_writer:
      self._summary_writer.flush()


class ValidationMonitor(EveryN):
  """Runs evaluation of a given estimator, at most every N steps.

  Note that the evaluation is done based on the saved checkpoint, which will
  usually be older than the current step.

  Can do early stopping on validation metrics if `early_stopping_rounds` is
  provided.
  """

  def __init__(self, x=None, y=None, input_fn=None, batch_size=None,
               eval_steps=None,
               every_n_steps=100, metrics=None, early_stopping_rounds=None,
               early_stopping_metric="loss",
               early_stopping_metric_minimize=True, name=None):
    """Initializes a ValidationMonitor.

    Args:
      x: See `BaseEstimator.evaluate`.
      y: See `BaseEstimator.evaluate`.
      input_fn: See `BaseEstimator.evaluate`.
      batch_size: See `BaseEstimator.evaluate`.
      eval_steps: See `BaseEstimator.evaluate`.
      every_n_steps: Check for new checkpoints to evaluate every N steps. If a
          new checkpoint is found, it is evaluated. See `EveryN`.
      metrics: See `BaseEstimator.evaluate`.
      early_stopping_rounds: `int`. If the metric indicated by
          `early_stopping_metric` does not change according to
          `early_stopping_metric_minimize` for this many steps, then training
          will be stopped.
      early_stopping_metric: `string`, name of the metric to check for early
          stopping.
      early_stopping_metric_minimize: `bool`, True if `early_stopping_metric` is
          expected to decrease (thus early stopping occurs when this metric
          stops decreasing), False if `early_stopping_metric` is expected to
          increase. Typically, `early_stopping_metric_minimize` is True for
          loss metrics like mean squared error, and False for performance
          metrics like accuracy.
      name: See `BaseEstimator.evaluate`.

    Raises:
      ValueError: If both x and input_fn are provided.
    """
    super(ValidationMonitor, self).__init__(every_n_steps=every_n_steps,
                                            first_n_steps=-1)
    # TODO(mdan): Checks like this are already done by evaluate.
    if x is None and input_fn is None:
      raise ValueError("Either x or input_fn should be provided.")
    self.x = x
    self.y = y
    self.input_fn = input_fn
    self.batch_size = batch_size
    self.eval_steps = eval_steps
    self.metrics = metrics
    self.early_stopping_rounds = early_stopping_rounds
    self.early_stopping_metric = early_stopping_metric
    self.early_stopping_metric_minimize = early_stopping_metric_minimize
    self.name = name
    self._best_value_step = None
    self._best_value = None
    self._early_stopped = False
    self._latest_path = None
    self._latest_path_step = None

  @property
  def early_stopped(self):
    """Returns True if this monitor caused an early stop."""
    return self._early_stopped

  @property
  def best_step(self):
    """Returns the step at which the best early stopping metric was found."""
    return self._best_value_step

  @property
  def best_value(self):
    """Returns the best early stopping metric value found so far."""
    return self._best_value

  def every_n_step_end(self, step, outputs):
    super(ValidationMonitor, self).every_n_step_end(step, outputs)
    # TODO(mdan): The use of step below is probably misleading.
    # The code should probably use the step from the checkpoint, because
    # that's what is being evaluated.
    if self._estimator is None:
      raise ValueError("Missing call to set_estimator.")
    # Check that we are not running evaluation on the same checkpoint.
    latest_path = saver.latest_checkpoint(self._estimator.model_dir)
    if latest_path == self._latest_path:
      logging.info("Skipping evaluation due to same checkpoint %s for step %d "
                   "as for step %d.", latest_path, step, self._latest_path_step)
      return False
    self._latest_path = latest_path
    self._latest_path_step = step

    # Run evaluation and log it.
    outputs = self._estimator.evaluate(
        x=self.x, y=self.y, input_fn=self.input_fn, batch_size=self.batch_size,
        steps=self.eval_steps, metrics=self.metrics, name=self.name)
    stats = []
    for name in outputs:
      stats.append("%s = %s" % (name, str(outputs[name])))
    logging.info("Validation (step %d): %s", step, ", ".join(stats))

    # Early stopping logic.
    if self.early_stopping_rounds is not None:
      if (self._best_value is None or
          (self.early_stopping_metric_minimize and
           outputs[self.early_stopping_metric] < self._best_value) or
          (not self.early_stopping_metric_minimize and
           outputs[self.early_stopping_metric] > self._best_value)):
        self._best_value = outputs[self.early_stopping_metric]
        self._best_value_step = step
      stop_now = (step - self._best_value_step >= self.early_stopping_rounds)
      if stop_now:
        logging.info("Stopping. Best step: {} with {} = {}."
                     .format(self._best_value_step,
                             self.early_stopping_metric, self._best_value))
        self._early_stopped = True
        return True
    return False


# TODO(ptucker): This really reads any tensor, not just vars, and requires the
# ':0' suffix on var_name.
class CaptureVariable(EveryN):
  """Captures a variable's values into a collection.

  This monitor is useful for unit testing. You should exercise caution when
  using this monitor in production, since it never discards values.

  This is an `EveryN` monitor and has consistent semantic for `every_n`
  and `first_n`.
  """

  def __init__(self, var_name, every_n=100, first_n=1):
    """Initializes a CaptureVariable monitor.

    Args:
      var_name: `string`. The variable name, including suffix (typically ":0").
      every_n: `int`, print every N steps. See `PrintN.`
      first_n: `int`, also print the first N steps. See `PrintN.`
    """
    super(CaptureVariable, self).__init__(every_n, first_n)
    self._var_name = var_name
    self._var_values = {}

  @property
  def values(self):
    """Returns the values captured so far.

    Returns:
      `dict` mapping `int` step numbers to that values of the variable at the
          respective step.
    """
    return self._var_values

  def every_n_step_begin(self, step):
    super(CaptureVariable, self).every_n_step_begin(step)
    return [self._var_name]

  def every_n_step_end(self, step, outputs):
    super(CaptureVariable, self).every_n_step_end(step, outputs)
    self._var_values[step] = outputs[self._var_name]


def get_default_monitors(loss_op=None, summary_op=None, save_summary_steps=100,
                         output_dir=None, summary_writer=None):
  """Returns a default set of typically-used monitors.

  Args:
    loss_op: `Tensor`, the loss tensor. This will be printed using `PrintTensor`
        at the default interval.
    summary_op: See `SummarySaver`.
    save_summary_steps: See `SummarySaver`.
    output_dir:  See `SummarySaver`.
    summary_writer:  See `SummarySaver`.
  Returns:
    `list` of monitors.
  """

  monitors = []
  if loss_op is not None:
    monitors.append(PrintTensor(tensor_names={"loss": loss_op.name}))
  if summary_op is not None:
    monitors.append(SummarySaver(summary_op, save_steps=save_summary_steps,
                                 output_dir=output_dir,
                                 summary_writer=summary_writer))
  return monitors


class GraphDump(BaseMonitor):
  """Dumps almost all tensors in the graph at every step.

  Note, this is very expensive, prefer `PrintTensor` in production.
  """

  IGNORE_OPS = ["Const", "Assign", "Identity", "Placeholder",
                "RandomUniform", "Cast", "RestoreSlice"]

  def __init__(self, ignore_ops=None):
    """Initializes GraphDump monitor.

    Args:
      ignore_ops: `list` of `string`. Names of ops to ignore.
          If None, `GraphDump.IGNORE_OPS` is used.
    """
    super(GraphDump, self).__init__()
    self._ignore_ops = ignore_ops or GraphDump.IGNORE_OPS
    self._data = {}

  def begin(self, max_steps):
    super(GraphDump, self).begin(max_steps)
    self._tensors = []
    graph = ops.get_default_graph()
    graph_def = graph.as_graph_def()
    for node in graph_def.node:
      if node.op in self._ignore_ops:
        continue
      logging.info("op=%s name=%s.", node.op, node.name)
      try:
        self._tensors.append(graph.get_tensor_by_name(node.name + ":0"))
      except KeyError:
        pass

  def step_begin(self, step):
    super(GraphDump, self).step_begin(step)
    return self._tensors

  def step_end(self, step, output):
    super(GraphDump, self).step_end(step, output)
    self._data[step] = output

  @property
  def data(self):
    return self._data

  # TODO(ptucker): Handle keys that are in one but not the other.
  def compare(self, other_dump, step, atol=1e-06):
    """Compares two `GraphDump` monitors and returns differences.

    Args:
      other_dump: Another `GraphDump` monitor.
      step: `int`, step to compare on.
      atol: `float`, absolute tolerance in comparison of floating arrays.

    Returns:
      Returns tuple:
        matched: `list` of keys that matched.
        non_matched: `dict` of keys to tuple of 2 mismatched values.

    Raises:
      ValueError: if a key in `data` is missing from `other_dump` at `step`.
    """
    non_matched = {}
    matched = []
    this_output = self.data[step] if step in self.data else {}
    other_output = other_dump.data[step] if step in other_dump.data else {}
    for key in this_output:
      if not isinstance(key, str) and not isinstance(key, unicode):
        continue
      if key not in other_output:
        raise ValueError("%s missing at step %s.", (key, step))
      value1 = this_output[key]
      value2 = other_output[key]
      if isinstance(value1, str):
        continue
      if isinstance(value1, np.ndarray):
        if not np.allclose(value1, value2, atol=atol):
          non_matched[key] = value1 - value2
        else:
          matched.append(key)
      else:
        if value1 != value2:
          non_matched[key] = (value1, value2)
        else:
          matched.append(key)
    return matched, non_matched


class ExportMonitor(EveryN):
  """Monitor that exports Estimator every N steps."""

  def __init__(self,
               every_n_steps,
               export_dir,
               exports_to_keep=5,
               signature_fn=None):
    """Initializes ExportMonitor.

    Args:
      every_n_steps: Run monitor every N steps.
      export_dir: str, folder to export.
      exports_to_keep: int, number of exports to keep.
      signature_fn: Function that given `Tensor` of `Example` strings,
        `dict` of `Tensor`s for features and `dict` of `Tensor`s for predictions
        and returns default and named exporting signautres.
    """
    super(ExportMonitor, self).__init__(every_n_steps=every_n_steps)
    self.export_dir = export_dir
    self.exports_to_keep = exports_to_keep
    self.signature_fn = signature_fn

  def every_n_step_end(self, step, outputs):
    super(ExportMonitor, self).every_n_step_end(step, outputs)
    try:
      export.export_estimator(self._estimator,
                              self.export_dir,
                              exports_to_keep=self.exports_to_keep,
                              signature_fn=self.signature_fn)
    except (RuntimeError, TypeError):
      # Currently we are not syncronized with saving checkpoints, which leads to
      # runtime errors when we are calling export on the same global step.
      logging.info("Skipping exporting for the same step. "
                   "Consider exporting less frequently.")

  def end(self):
    super(ExportMonitor, self).end()
    export.export_estimator(self._estimator,
                            self.export_dir,
                            exports_to_keep=self.exports_to_keep,
                            signature_fn=self.signature_fn)
