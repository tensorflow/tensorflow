# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""TensorBoard Summary Writer for TensorFlow Eager Execution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid

from tensorflow.contrib.summary import gen_summary_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import summary_op_util
from tensorflow.python.ops import variable_scope


def _maybe_cpu(v):
  if isinstance(v, (ops.EagerTensor, ops.Tensor)):
    return v.cpu()
  else:
    return v


def _summary_writer_function(name, tensor, function, family=None):
  def record():
    with summary_op_util.summary_scope(
        name, family, values=[tensor]) as (tag, scope):
      function(tag, scope)
      return True
  return record


class SummaryWriter(object):
  """Writes summaries for TensorBoard, compatible with eager execution.

  This class is the supported way of writing TensorBoard summaries under
  eager execution.
  """

  _CPU_DEVICE = "cpu:0"

  def __init__(self,
               logdir,
               max_queue=10,
               flush_secs=120,
               filename_suffix=""):
    """Summary writer for TensorBoard, compatible with eager execution.

    If necessary, multiple instances of `SummaryWriter` can be created, with
    distinct `logdir`s and `name`s. Each `SummaryWriter` instance will retain
    its independent `global_step` counter and data writing destination.

    Example:
    ```python
    writer = tfe.SummaryWriter("my_model")

    # ... Code that sets up the model and data batches ...

    for _ in xrange(train_iters):
      loss = model.train_batch(batch)
      writer.scalar("loss", loss)
      writer.step()
    ```

    Args:
      logdir: Directory in which summary files will be written.
      max_queue: Number of summary items to buffer before flushing to
        filesystem. If 0, summaries will be flushed immediately.
      flush_secs: Number of secondsbetween forced commits to disk.
      filename_suffix: Suffix of the event protobuf files in which the summary
        data are stored.

    Raises:
      ValueError: If this constructor is called not under eager execution.
    """
    # TODO(apassos, ashankar): Make this class and the underlying
    # contrib.summary_ops compatible with graph model and remove this check.
    if not context.in_eager_mode():
      raise ValueError(
          "Use of SummaryWriter is currently supported only with eager "
          "execution enabled. File an issue at "
          "https://github.com/tensorflow/tensorflow/issues/new to express "
          "interest in fixing this.")

    # TODO(cais): Consider adding name keyword argument, which if None or empty,
    # will register the global global_step that training_util.get_global_step()
    # can find.
    with context.device(self._CPU_DEVICE):
      self._name = uuid.uuid4().hex
      self._global_step = 0
      self._global_step_tensor = variable_scope.get_variable(
          "global_step/summary_writer/" + self._name,
          shape=[], dtype=dtypes.int64,
          initializer=init_ops.zeros_initializer())
      self._global_step_dirty = False
      self._resource = gen_summary_ops.summary_writer(shared_name=self._name)
      gen_summary_ops.create_summary_file_writer(
          self._resource, logdir, max_queue, flush_secs, filename_suffix)
      # Delete the resource when this object is deleted
      self._resource_deleter = resource_variable_ops.EagerResourceDeleter(
          handle=self._resource, handle_device=self._CPU_DEVICE)

  def step(self):
    """Increment the global step counter of this SummaryWriter instance."""
    self._global_step += 1
    self._global_step_dirty = True

  @property
  def global_step(self):
    """Obtain the current global_step value of this SummaryWriter instance.

    Returns:
      An `int` representing the current value of the global_step of this
       `SummaryWriter` instance.
    """
    return self._global_step

  def _update_global_step_tensor(self):
    with context.device(self._CPU_DEVICE):
      if self._global_step_dirty:
        self._global_step_dirty = False
        return state_ops.assign(self._global_step_tensor, self._global_step)
      else:
        return self._global_step_tensor

  def generic(self, name, tensor, metadata, family=None):
    """Write a generic-type summary.

    Args:
      name: A name for the generated node. Will also serve as the series name in
        TensorBoard.
      tensor: A `Tensor` or compatible value type containing the value of the
        summary.
      metadata: Metadata about the summary.
      family: Optional; if provided, used as the prefix of the summary tag name,
        which controls the tab name used for display on Tensorboard.
    """
    with context.device(self._CPU_DEVICE):
      with summary_op_util.summary_scope(
          name, family, values=[tensor]) as (tag, scope):
        gen_summary_ops.write_summary(
            self._resource,
            self._update_global_step_tensor(),
            _maybe_cpu(tensor),
            tag,
            _maybe_cpu(metadata),
            name=scope)

  def scalar(self, name, tensor, family=None):
    """Write a scalar summary.

    Args:
      name: A name for the generated node. Will also serve as the series name in
        TensorBoard.
      tensor: A real numeric `Tensor` or compatible value type containing a
        single value.
      family: Optional; if provided, used as the prefix of the summary tag name,
        which controls the tab name used for display on Tensorboard.

    Returns:
      A summary writer function for scalars.
    """
    with context.device(self._CPU_DEVICE):
      with summary_op_util.summary_scope(
          name, family, values=[tensor]) as (tag, scope):
        gen_summary_ops.write_scalar_summary(
            self._resource, self._update_global_step_tensor(),
            tag, _maybe_cpu(tensor), name=scope)

  def histogram(self, name, tensor, family=None):
    """Write a histogram summary.

    Args:
      name: A name for the generated node. Will also serve as a series name in
        TensorBoard.
      tensor: A real numeric `Tensor` or compatible value type. Any shape.
        Values to use to build the histogram.
      family: Optional; if provided, used as the prefix of the summary tag name,
        which controls the tab name used for display on Tensorboard.
    """
    with context.device(self._CPU_DEVICE):
      with summary_op_util.summary_scope(
          name, family, values=[tensor]) as (tag, scope):
        gen_summary_ops.write_histogram_summary(
            self._resource, self._update_global_step_tensor(),
            tag, _maybe_cpu(tensor), name=scope)

  def image(self, name, tensor, bad_color=None, max_images=3, family=None):
    """Write an image summary."""
    with context.device(self._CPU_DEVICE):
      if bad_color is None:
        bad_color_ = constant_op.constant([255, 0, 0, 255], dtype=dtypes.uint8)
      with summary_op_util.summary_scope(
          name, family, values=[tensor]) as (tag, scope):
        gen_summary_ops.write_image_summary(
            self._resource, self._update_global_step_tensor(),
            tag, _maybe_cpu(tensor), bad_color_, max_images,
            name=scope)

  def audio(self, name, tensor, sample_rate, max_outputs, family=None):
    """Write an audio summary.

    Args:
      name: A name for the generated node. Will also serve as a series name in
        TensorBoard.
      tensor: A 3-D `float32` `Tensor` of shape `[batch_size, frames, channels]`
        or a 2-D `float32` `Tensor` of shape `[batch_size, frames]`, or
        compatible value type.
      sample_rate: A Scalar `float32` `Tensor` indicating the sample rate of the
        signal in hertz.
      max_outputs: Max number of batch elements to generate audio for.
      family: Optional; if provided, used as the prefix of the summary tag name,
        which controls the tab name used for display on Tensorboard.
    """
    with context.device(self._CPU_DEVICE):
      with summary_op_util.summary_scope(
          name, family, values=[tensor]) as (tag, scope):
        gen_summary_ops.write_audio_summary(
            self._resource, self._update_global_step_tensor(),
            tag,
            _maybe_cpu(tensor),
            sample_rate=_maybe_cpu(sample_rate),
            max_outputs=max_outputs,
            name=scope)
