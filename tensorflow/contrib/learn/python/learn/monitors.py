# pylint: disable=g-bad-file-header
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

from tensorflow.python.platform import tf_logging as logging
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

  def step_begin(self, step, tensors):  # pylint: disable=unused-argument
    """Callback before training step begins.

    Use this callback to:
     - override which tensors to run.

    Args:
      step: int, global step of the model.
      tensors: list of `Tensors` that going to be passed to session.run.

    Returns:
      Dict of `Tensors` that going to be ran.
    """
    return tensors

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
  """Base class for monitors that execute callbacks every n steps / seconds.

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

  def begin(self, max_steps=None):
    self._max_steps = max_steps

  def every_n_step_begin(self, step, tensors):  # pylint: disable=unused-argument
    return tensors

  def every_n_step_end(self, step, outputs):  # pylint: disable=unused-argument
    return False

  def step_begin(self, step, tensors):
    if (step <= self._first_n_steps or step % self._every_n_steps == 0 or
        step == self._max_steps):
      tensors = self.every_n_step_begin(step, tensors)
    return tensors

  def step_end(self, step, output):
    to_stop = False
    if (step <= self._first_n_steps or step % self._every_n_steps == 0 or
        step == self._max_steps):
      to_stop = self.every_n_step_end(step, output)
    return to_stop


class PrintTensor(EveryN):

  def __init__(self, tensor_names, every_n=100, first_n=1):
    super(PrintTensor, self).__init__(every_n, first_n)
    self._tensor_names = tensor_names

  def every_n_step_begin(self, unused_step, tensors):
    return tensors + self._tensor_names

  def every_n_step_end(self, step, outputs):
    stats = []
    for name in self._tensor_names:
      if name in outputs:
        stats.append("%s = %s" % (name, str(outputs[name])))
    logging.info("Step %d: %s" % (step, ", ".join(stats)))


class SummarySaver(EveryN):
  """Saves summary every N seconds."""

  def __init__(self, summary_op, save_steps=100, output_dir=None):
    # TODO(ipolosukhin): Implement every N seconds.
    super(SummarySaver, self).__init__(every_n_steps=save_steps)
    self._summary_op = summary_op
    self._summary_writer = None
    if output_dir:
      self._summary_writer = summary_io.SummaryWriter(output_dir)

  def set_estimator(self, estimator):
    super(SummarySaver, self).set_estimator(estimator)
    self._summary_writer = summary_io.SummaryWriter(self._estimator.model_dir)

  def every_n_step_begin(self, unused_step, tensors):
    return tensors + [self._summary_op]

  def every_n_step_end(self, step, outputs):
    summary_strs = outputs[self._summary_op.name]
    if self._summary_writer:
      self._summary_writer.add_summary(summary_strs, step)
    return False

  def end(self):
    self._summary_writer.flush()


class ValidationMonitor(EveryN):
  """Runs evaluation every n steps.

  Can do early stopping on validation loss if `early_stopping_rounds` provided.

  """

  def __init__(self, x=None, y=None, input_fn=None,
               every_n_steps=100, early_stopping_rounds=None):
    super(ValidationMonitor, self).__init__(every_n_steps=every_n_steps)
    if x is None and input_fn is None:
      raise ValueError("Either x or input_fn should be provided.")
    self.x = x
    self.y = y
    self.input_fn = input_fn
    self.min_loss_step = 0
    self.min_loss = None
    self.early_stopping_rounds = early_stopping_rounds

  def every_n_step_end(self, step, unused_outputs):
    outputs = self._estimator.evaluate(
        x=self.x, y=self.y, input_fn=self.input_fn)
    stats = []
    for name in outputs:
      stats.append("%s = %s" % (name, str(outputs[name])))
    logging.info("Validation (step %d): %s" % (step, ", ".join(stats)))
    if self.early_stopping_rounds is not None:
      if self.min_loss is None or outputs["loss"] < self.min_loss:
        self.min_loss = outputs["loss"]
        self.min_loss_step = step
      stop_now = (step - self.min_loss_step >= self.early_stopping_rounds)
      if stop_now:
        logging.info("Stopping. Best step: {} with loss {}."
                     .format(self.min_loss_step, self.min_loss))
        return True
    return False


def get_default_monitors(loss_op=None, summary_op=None, save_summary_steps=100,
                         output_dir=None):
  monitors = []
  if loss_op is not None:
    monitors.append(PrintTensor([loss_op.name]))
  if summary_op is not None:
    monitors.append(SummarySaver(summary_op, save_steps=save_summary_steps,
                                 output_dir=output_dir))
  return monitors
