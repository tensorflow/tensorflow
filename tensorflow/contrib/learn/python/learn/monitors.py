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

"""Monitors to track training, report progress and request early stopping."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six

from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver
from tensorflow.python.training import summary_io


class BaseMonitor(object):
  """Base class for Monitors.

  Defines basic interfaces of Monitors.
  """

  def set_estimator(self, estimator):
    self._estimator = estimator

  def begin(self, max_steps=None):
    """Callback at the beginning of training/evaluation.

    Args:
      max_steps: Maximum steps this training will run until.
    """
    pass

  def end(self):
    """Callback at the end of training/evaluation."""
    pass

  def epoch_begin(self, epoch):
    pass

  def epoch_end(self, epoch):
    pass

  def step_begin(self, step):  # pylint: disable=unused-argument
    """Callback before training step begins.

    Use this callback to:
     - override which tensors to run.

    Args:
      step: int, global step of the model.

    Returns:
      List of `Tensors` that going to be ran.
    """
    return []

  def step_end(self, step, output):  # pylint: disable=unused-argument
    """Callback after training step finished.

    Use this callback to:
     - log results.
     - save checkpoints.
     - compute validation score.
     - perform early stopping.

    Args:
      step: `int`, global step of the model.
      output: `dict` of `np.array` results executed.

    Returns:
      `bool`, `True` if model should stop, `False` or `None` if continue.
    """
    return False


class EveryN(BaseMonitor):
  """Base class for monitors that execute callbacks every n steps.

  Parameters:
    every_n_steps: int, calls `every_n_step_{begin,end}` every this many steps.
    first_n_steps: int, calls `every_n_step_{begin,end}` for first n steps.

  TODO(ipolosukhin): Add also every n seconds.
  """

  def __init__(
      self, every_n_steps=100, first_n_steps=1):
    self._every_n_steps = every_n_steps
    self._first_n_steps = first_n_steps
    self._max_steps = None
    self._last_step = 0

  def begin(self, max_steps=None):
    self._max_steps = max_steps

  def every_n_step_begin(self, step):  # pylint: disable=unused-argument
    return []

  def every_n_step_end(self, step, outputs):  # pylint: disable=unused-argument
    return False

  def step_begin(self, step):
    if (step <= self._first_n_steps or
        step >= (self._every_n_steps + self._last_step) or
        step == self._max_steps):
      return self.every_n_step_begin(step)
    return []

  def step_end(self, step, output):
    to_stop = False
    if (step <= self._first_n_steps or
        step >= (self._every_n_steps + self._last_step) or
        step == self._max_steps):
      self._last_step = step
      to_stop = self.every_n_step_end(step, output)
    return to_stop


class PrintTensor(EveryN):
  """Prints given tensors every N steps.

  Print the tensors provided in `tensor_names` `every_n`
  steps, starting with the `first_n`th step.

  """

  def __init__(self, tensor_names, every_n=100, first_n=1):
    """Initializes PrintTensor monitor.

    Args:
      tensor_names: `dict` of tag to tensor names or
          `iterable` of tensor names (strings).
      every_n: Print every N steps.
      first_n: Print first N steps.
    """
    super(PrintTensor, self).__init__(every_n, first_n)
    if not isinstance(tensor_names, dict):
      tensor_names = {item: item for item in tensor_names}
    self._tensor_names = tensor_names

  def every_n_step_begin(self, unused_step):
    return list(self._tensor_names.values())

  def every_n_step_end(self, step, outputs):
    stats = []
    for tag, tensor_name in six.iteritems(self._tensor_names):
      if tensor_name in outputs:
        stats.append("%s = %s" % (tag, str(outputs[tensor_name])))
    logging.info("Step %d: %s" % (step, ", ".join(stats)))


class SummarySaver(EveryN):
  """Saves a summary every N steps."""

  def __init__(self, summary_op, save_steps=100, output_dir=None,
               summary_writer=None):
    # TODO(ipolosukhin): Implement every N seconds.
    super(SummarySaver, self).__init__(every_n_steps=save_steps)
    self._summary_op = summary_op
    self._summary_writer = summary_writer
    if summary_writer is None and output_dir:
      self._summary_writer = summary_io.SummaryWriter(output_dir)

  def set_estimator(self, estimator):
    super(SummarySaver, self).set_estimator(estimator)
    if self._summary_writer is None:
      self._summary_writer = summary_io.SummaryWriter(self._estimator.model_dir)

  def every_n_step_begin(self, unused_step):
    return [self._summary_op]

  def every_n_step_end(self, step, outputs):
    summary_strs = outputs[self._summary_op.name]
    if self._summary_writer:
      self._summary_writer.add_summary(summary_strs, step)
    return False

  def end(self):
    self._summary_writer.flush()


class ValidationMonitor(EveryN):
  """Runs evaluation of the Estimator every n steps.

  Can do early stopping on validation metrics if
  `early_stopping_rounds` provided.
  """

  def __init__(self, x=None, y=None, input_fn=None, batch_size=None,
               every_n_steps=100, metrics=None, early_stopping_rounds=None,
               early_stopping_metric="loss",
               early_stopping_metric_minimize=True, name=None):
    """Initializes ValidationMonitor.

    Args:
      x: matrix or tensor of shape [n_samples, n_features...]. Can be
         iterator that returns arrays of features. The training input
         samples for fitting the model. If set, `input_fn` must be `None`.
      y: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
         iterator that returns array of targets. The training target values
         (class labels in classification, real numbers in regression). If set,
         `input_fn` must be `None`.
      input_fn: Input function. If set, `x`, `y`, and `batch_size` must be
          `None`.
      batch_size: minibatch size to use on the input, defaults to first
          dimension of `x`. Must be `None` if `input_fn` is provided.
      every_n_steps: Runs this monitor every N steps.
      metrics: Dict of metric ops to run. If None, the default metric functions
        are used; if {}, no metrics are used.
      early_stopping_rounds: If validation metric didn't go down for this many
          steps, then stop training.
      early_stopping_metric: `str`, name of the metric to early stop.
      early_stopping_metric_minimize: `bool`, True if minimize, False
          if maximize. For example, minimize `loss` or `mean_squared_error` and
          maximize `accuracy` or `f1`.
      name: `str`, appended to output sub-folder. If None uses `eval`
          sub-folder, else, `eval-%name%` is used to save sum.

    Raises:
      ValueError: If both x and input_fn are provided.
    """
    super(ValidationMonitor, self).__init__(every_n_steps=every_n_steps,
                                            first_n_steps=-1)
    if x is None and input_fn is None:
      raise ValueError("Either x or input_fn should be provided.")
    self.x = x
    self.y = y
    self.input_fn = input_fn
    self.batch_size = batch_size
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
    return self._early_stopped

  @property
  def best_step(self):
    return self._best_value_step

  @property
  def best_value(self):
    return self._best_value

  def every_n_step_end(self, step, unused_outputs):
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
        metrics=self.metrics, name=self.name)
    stats = []
    for name in outputs:
      stats.append("%s = %s" % (name, str(outputs[name])))
    logging.info("Validation (step %d): %s" % (step, ", ".join(stats)))

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


class CaptureVariable(EveryN):
  """Capture a variable value into a `list`.

  This monitor is useful for unit testing.
  """

  def __init__(self, var_name, every_n=100, first_n=1):
    super(CaptureVariable, self).__init__(every_n, first_n)
    self.var_name = var_name
    self.var_values = []

  def every_n_step_begin(self, unused_step):
    return [self.var_name]

  def every_n_step_end(self, step, outputs):
    self.var_values.append(outputs[self.var_name])


def get_default_monitors(loss_op=None, summary_op=None, save_summary_steps=100,
                         output_dir=None, summary_writer=None):
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

  Note, this is very expensive, prefer `PrintTensor` or `CaptureVariable` if
  you are not debugging.
  """

  IGNORE_OPS = ["Const", "Assign", "Identity", "Placeholder",
                "RandomUniform", "Cast", "RestoreSlice"]

  def __init__(self, ignore_ops=None):
    """Initializes GraphDump monitor.

    Args:
      ignore_ops: `list` of string names of `Operation`s to ignore.
          If `None` GraphDump.IGNORE_OPS list is used.
    """
    self.ignore_ops = ignore_ops or GraphDump.IGNORE_OPS
    self._data = []

  def begin(self, max_steps):
    self.tensors = []
    graph = ops.get_default_graph()
    graph_def = graph.as_graph_def()
    for node in graph_def.node:
      if node.op in self.ignore_ops:
        continue
      try:
        self.tensors.append(graph.get_tensor_by_name(node.name + ":0"))
      except KeyError:
        pass

  def step_begin(self, step):
    return self.tensors

  def step_end(self, step, outputs):
    self._data.append(outputs)

  @property
  def data(self):
    return self._data

  def compare(self, other_dump, step, atol=1e-06):
    """Compares two `GraphDump` monitors and returns differences.

    Args:
      other_dump: Another `GraphDump` monitor.
      step: `int`, step to compare on.
      atol: `float`, absolute tolerance in comparison of floating arrays.

    Returns:
      Returns tuple:
        matched: `list` of keys that matched.
        non_matched: `dict` of keys to difference.
    """
    non_matched = {}
    matched = []
    for key in self.data[step]:
      if not isinstance(key, str) and not isinstance(key, unicode):
        continue
      value1, value2 = self.data[step][key], other_dump.data[step][key]
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
