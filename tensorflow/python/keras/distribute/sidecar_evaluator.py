# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Python module for evaluation loop."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training.tracking import util as tracking_util

_PRINT_EVAL_STEP_EVERY_SEC = 60.0
_ITERATIONS_UNINITIALIZED = -1


class SidecarEvaluator(object):
  """A class designed for a dedicated evaluator task.

  `SidecarEvaluator` is expected to be run on a process in
  a separate machine from the training cluster. It continuously loads
  checkpoints saved periodically by that training counterpart, and performs
  evaluation using the model (with compiled metrics) provided at `__init__`. It
  is expected to be used for the purpose of a dedicated evaluator, evaluating
  the metric results of a training cluster which has one or more workers
  performing the training and saving checkpoints. `SidecarEvaluator` uses
  `model.evaluate` for evaluation.

  Example:
  ```python
  model = tf.keras.models.Sequential(...)
  model.compile(metrics=tf.keras.metrics.SparseCategoricalAccuracy(
      name="eval_metrics"))
  data = tf.data.Dataset.from_tensor_slices(...)

  SidecarEvaluator(
      model=model,
      data=data,
      checkpoint_dir='/tmp/checkpoint_dir',  # dir for training-saved checkpoint
      steps=None,  # Eval until dataset is exhausted
      log_dir='/tmp/log_dir',
      max_evaluations=None  # The evaluation needs to be stopped manually
  ).start()
  ```

  `SidecarEvaluator.start` writes a series of summary
  files which can be visualized by tensorboard (which provides a webpage link):

  ```bash
  $ tensorboard --logdir=/tmp/log_dir
  ...
  TensorBoard 2.4.0a0 at http://host:port (Press CTRL+C to quit)
  ```

  The `checkpoint_dir` should contain checkpoints that track `model` and
  `optimizer` to fulfill `SidecarEvaluator`'s
  expectation:

  ```python
  checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
  checkpoint_manager = tf.train.CheckpointManager(
      checkpoint, checkpoint_dir=..., max_to_keep=...)
  checkpoint_manager.save()
  ```

  """

  def __init__(self,
               model,
               data,
               checkpoint_dir,
               log_dir=None,
               steps=None,
               max_evaluations=None):
    """Initializes an `SidecarEvaluator` object.

    Args:
      model: Model to use for evaluation. The model object used here should be a
        `tf.keras.Model`, and should be the same as the one that is used in
        training, where `tf.keras.Model`s are checkpointed. The model should
        have one or more metrics compiled before using `SidecarEvaluator`.
      data: The input data for evaluation. `SidecarEvaluator` supports all data
        types that Keras `model.evaluate` supports as the input data `x`, such
        as a `tf.data.Dataset`.
      checkpoint_dir: Directory where checkpoint files are saved.
      log_dir: Directory where summary files for TensorBoard are saved.
      steps: Number of steps to perform evaluation for, when evaluating a single
        checkpoint file. If `None`, evaluation continues until the dataset is
        exhausted. For repeated evaluation dataset, user must specify `steps` to
        avoid infinite evaluation loop.
      max_evaluations: Maximum number of the checkpoint file to be evaluated,
        for `SidecarEvaluator` to know when to stop. The evaluator will stop
        after it evaluates a checkpoint filepath ending with
        '<ckpt_name>-<max_evaluations>'. If using
        `tf.train.CheckpointManager.save` for saving checkpoints, the kth saved
        checkpoint has the filepath suffix '<ckpt_name>-<k>' (k=1 for the first
        saved), and if checkpoints are saved every epoch after training, the
        filepath saved at the kth epoch would end with '<ckpt_name>-<k>. Thus,
        if training runs for n epochs, and the evaluator should end after the
        training finishes, use n for this parameter. Note that this is not
        necessarily equal to the number of total evaluations, since some
        checkpoints may be skipped if evaluation is slower than checkpoint
        creation. If `None`, `SidecarEvaluator` will evaluate indefinitely, and
        the user must terminate evaluator program themselves.
    """
    self.model = model
    self.data = data
    self.checkpoint_dir = checkpoint_dir
    if log_dir:
      self._summary_writer = summary_ops_v2.create_file_writer_v2(
          logdir=log_dir)
    else:
      self._summary_writer = None
    self._iterations = variables.Variable(
        name='iterations',
        initial_value=_ITERATIONS_UNINITIALIZED,
        dtype=dtypes.int64)
    self.max_evaluations = max_evaluations
    self.steps = steps

  def start(self):
    """Starts the evaluation loop."""
    optimizer_checkpoint = tracking_util.Checkpoint(iter=self._iterations)
    checkpoint = tracking_util.Checkpoint(
        model=self.model, optimizer=optimizer_checkpoint)

    for latest_checkpoint in checkpoint_utils.checkpoints_iterator(
        self.checkpoint_dir):
      try:
        # `expect_partial` because the checkpoint can have other `Trackable`s
        # such as `optimizer`.
        checkpoint.restore(latest_checkpoint).expect_partial()
      except (errors_impl.OpError,) as e:
        # A couple errors can happen here with the coordinator racing to write
        # checkpoint:
        # 1) OpError: open failed for <file path>: No such file or directory
        # 2) NotFoundError (subclass of OpError): Unsuccessful
        # TensorSliceReader constructor.
        # TODO(rchao): Remove this except block once b/150954027 is resolved.
        logging.info(
            'SidecarEvaluator has an error loading '
            'checkpoint: %s. Retrying. Error: %s: %s', latest_checkpoint,
            e.__class__.__name__, e)
        continue

      if self._iterations.numpy() == _ITERATIONS_UNINITIALIZED:
        raise RuntimeError(
            '`iterations` cannot be loaded from the '
            'checkpoint file. Please ensure `iterations` is '
            'tracked in the `checkpoint` saved by the coordinator.')

      logging.info(
          'Evaluation starts: Model weights loaded from latest '
          'checkpoint file: %s.', latest_checkpoint)

      # TODO(rchao): Support arbitrary callback for extensibility.
      self.model.evaluate(self.data, steps=self.steps)

      logging.info('End of evaluation. Accuracy: %r', [
          metric.result().numpy()
          for metric in self.model.compiled_metrics.metrics
      ])

      if self._summary_writer:
        with summary_ops_v2.always_record_summaries(
        ), self._summary_writer.as_default():
          for metric in self.model.compiled_metrics.metrics:
            summary_ops_v2.scalar(
                metric.name,
                metric.result(),
                step=self._iterations.read_value())

      # TODO(rchao): Make the max evaluation robust in case users save the
      # checkpoints with epoch format {epoch:03d}.
      if (self.max_evaluations and
          latest_checkpoint.endswith('-{}'.format(self.max_evaluations))):
        # Exit the loop because we have evaluated the final checkpoint file.
        logging.info('Last checkpoint evaluated. SidecarEvaluator stops.')
        return
