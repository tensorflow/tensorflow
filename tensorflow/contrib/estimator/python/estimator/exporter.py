# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Implements StepsExporter to export the model in user specified steps."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.estimator import exporter
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging
from tensorflow.python.summary import summary_iterator

DEFAULT_GLOBAL_STEP_KEY = ops.GraphKeys.GLOBAL_STEP


class StepsExporter(exporter.Exporter):
  """This class exports the model in user specified steps.

  This class exports the model at the steps given by the `steps_to_keep`
  argument. Each number in the list is treated as a lower bound for model
  exports, to handle the case when evaluation is performed at different steps.

  Consider this example:

  ```
  steps_to_keep = [1, 2, 3, 6, 7, 10, 12, 25]
  ```

  The model is evaluated at step increments of 5: `[5, 10, 15, 20, 25, 30]`.
  The `StepsExporter` will export the model when it has reached steps
  `[5, 10, 15, 25]`.

  This example illustrates the two cases when the model is exported:

  1. Model is evaluated on a step defined in the list `steps_to_keep`.

     In the example, the model is exported on step `10` and `25`.

  2. Model is evaluated on a step not defined in the list `steps_to_keep`, but
     is still exported because a step in `steps_to_keep` was missed.

     In the example, when the model reaches step `5`, the model is exported even
     though  `steps_to_keep` does not contain `5`. Step `5` is exported to make
     up for step `3`, which was missed. Steps `1` and `2` in `steps_to_keep` are
     skipped completely (e.g. say the model is evaluated at step `6`. It will
     **not** be exported to make up for step `2`).

  Using the `steps_to_keep` list as a lower bound allows users to define
  approximate step boundaries for exporting their models, and avoid frustrating
  off-by-one calculation errors.

  Sample Use Cases:
    There are specific points during the training when having a saved version of
    the model would be useful. One example is at the end of each training phase
    when the set of freezed weights is changed.
    Another good use case is saving the model at the end of each epoch for
    visualization or retraining.
  """

  def __init__(self,
               steps_to_keep,
               name='steps_exporter',
               serving_input_receiver_fn=None,
               event_file_pattern='eval/*.tfevents.*',
               assets_extra=None,
               as_text=False):
    """Create an `StepsExporter` to use with `tf.estimator.EvalSpec`.

    Example of creating a StepsExporter for training and evaluation:

    ```python
    categorical_feature_a = categorical_column_with_hash_bucket(...)
    categorical_feature_b = categorical_column_with_hash_bucket(...)

    categorical_feature_a_emb = embedding_column(
        categorical_column=categorical_feature_a, ...)
    categorical_feature_b_emb = embedding_column(
        categorical_column=categorical_feature_b, ...)

    estimator = tf.estimator.DNNClassifier(
        feature_columns=[categorical_feature_a_emb, categorical_feature_b_emb],
        hidden_units=[1024, 512, 256])

    # Input pipeline for train and evaluate.
    def train_input_fn: # returns x, y
      # please shuffle the data.
      pass
    def eval_input_fn_eval: # returns x, y
      pass

    exporter = tf.contrib.estimator.exporter.StepsExporter(
        name="steps_exporter",
        serving_input_receiver_fn=serving_input_receiver_fn,
        event_file_pattern='eval/*.tfevents.*'
        steps_to_keep=[...])

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=1000)

    eval_spec = [tf.estimator.EvalSpec(
      input_fn=eval_input_fn,
      steps=1,
      exporters=exporter,
      start_delay_secs=0,
      throttle_secs=5)]

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # Models will be exported to estimator.model_dir in timestamped directories,
    # which can be used for serving, analysis with TFMA, or directly loaded in.
    # For example:
    export_dir = os.path.join(estimator.model_dir,
                              <timestamped directory name>)

    with ops.Graph().as_default() as graph:
      with session.Session(graph=graph) as sess:
        tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING], export_dir)

    ```

    Args:
      steps_to_keep: Non-empty list of positive integers containing
        the step numbers at which the model should be exported. All the exports
        will be kept, so there is no garbage collection.
      name: Unique name of this `Exporter` that is going to be used in the
        export path.
      serving_input_receiver_fn: A function that takes no arguments and returns
        a `ServingInputReceiver`.
      event_file_pattern: Event file name pattern relative to model_dir. If
        None, however, the exporter would not be preemption-safe. To be
        preemption-safe, event_file_pattern should be specified.
      assets_extra: An optional dict specifying how to populate the assets.extra
        directory within the exported SavedModel.  Each key should give the
        destination path (including the filename) relative to the assets.extra
        directory.  The corresponding value gives the full path of the source
        file to be copied.  For example, the simple case of copying a single
        file without renaming it is specified as `{'my_asset_file.txt':
        '/path/to/my_asset_file.txt'}`.
      as_text: Whether to write the SavedModel proto in text format. Defaults to
        `False`.

    Raises:
      ValueError: If any arguments is invalid.
    """
    # pylint: disable=protected-access
    self._saved_model_exporter = exporter._SavedModelExporter(
        name, serving_input_receiver_fn, assets_extra, as_text)
    # pylint: enable=protected-access

    self._event_file_pattern = event_file_pattern
    self._model_dir = None

    self._input_steps_to_keep = steps_to_keep
    steps_to_keep = [step for step in steps_to_keep if isinstance(step, int)]
    steps_to_keep = [step for step in steps_to_keep if step > 0]
    if not steps_to_keep:
      raise ValueError(
          '`steps_to_keep` list must have at least one positive integer')
    elif self._input_steps_to_keep != steps_to_keep:
      tf_logging.warn('Changed `steps_to_keep`, by omitting non-integer or'
                      ' less than 1 elements, to [%s]',
                      ', '.join(str(step) for step in steps_to_keep))
    self._steps_to_keep = sorted(steps_to_keep)
    self._steps_kept = []

  @property
  def name(self):
    return self._saved_model_exporter.name

  def export(self, estimator, export_path, checkpoint_path, eval_result,
             is_the_final_export):
    """Exports the given Estimator to a specific format.

    Args:
      estimator: A `tf.estimator.Estimator` instance to export.
      export_path: A string containing a directory where to write the export.
      checkpoint_path: The checkpoint path to export.
      eval_result: The output of Estimator.evaluate on this checkpoint.
      is_the_final_export: This boolean is True when this is an export in the
        end of training. It is False for the intermediate exports during the
        training. When passing Exporter to tf.estimator.train_and_evaluate
        is_the_final_export is always False if TrainSpec.max_steps is None.

    Returns:
      The string path to the exported directory or None if export is skipped.

    Raises:
      ValueError: If `eval_result` is None or doesn't have
        `ops.GraphKeys.GLOBAL_STEP` as a key.
    """
    export_result = None

    if not eval_result or DEFAULT_GLOBAL_STEP_KEY not in eval_result:
      raise ValueError(
          '`eval_result` is empty, or does not have global step. This'
          ' should never happen as Estimator always sets the global step in '
          '`eval_result`. Please file a bug report. Got eval_result: %s'
          % str(eval_result))

    if self._model_dir != estimator.model_dir and self._event_file_pattern:
      tf_logging.info('Loads the steps that the model was already evaluated at,'
                      'from event files')
      self._model_dir = estimator.model_dir
      full_event_file_pattern = os.path.join(self._model_dir,
                                             self._event_file_pattern)
      self._steps_kept = self._get_kept_steps(full_event_file_pattern)

      if self._steps_kept:
        self._steps_kept = sorted(self._steps_kept)
        self._steps_to_keep = [step for step in self._steps_to_keep if
                               step > self._steps_kept[-1]]
    # It is assumed that the model is exported at any evaluated step 'n' if
    # there is any `steps_missed` lower than 'n'. As a result, all the steps in
    # `_steps_to_keep` lower than the last evaluated step will be removed.
    steps_missed = [step for step in self._steps_to_keep
                    if step <= eval_result[DEFAULT_GLOBAL_STEP_KEY]]

    if steps_missed:
      # update the `_steps_to_keep` list by omitting all steps smaller than the
      # current global step which are missed to be exported
      export_result = self._saved_model_exporter.export(estimator, export_path,
                                                        checkpoint_path,
                                                        eval_result,
                                                        is_the_final_export)
      self._steps_to_keep = [step for step in self._steps_to_keep if step
                             not in steps_missed]
      # contains all the steps in which export has happened.
      self._steps_kept.append(eval_result[DEFAULT_GLOBAL_STEP_KEY])
      # Show warning for all the missed steps except the last one
      if steps_missed[:-1]:
        tf_logging.warn('Missed steps [%s] for exporting, as no evaluation'
                        ' took place at them.', ', '.join(str(step) for step in
                                                          steps_missed[:-1]))
      # Log model export if the last missed step is the same as the current step
      if steps_missed[-1] == eval_result[DEFAULT_GLOBAL_STEP_KEY]:
        tf_logging.info('Performing model export at step %d.',
                        eval_result[DEFAULT_GLOBAL_STEP_KEY])
      # Show warning for exporting model at another step instead of the user
      #   specified one
      else:
        tf_logging.warn('Performing model export at step %d instead of %d, as'
                        ' no evaluation took place at step %d.',
                        eval_result[DEFAULT_GLOBAL_STEP_KEY], steps_missed[-1],
                        steps_missed[-1])
    return export_result

  def _get_kept_steps(self, event_files):
    """Get the steps that the model was evaluated at, from event files.

    Args:
      event_files: Absolute pattern of event files.

    Returns:
      steps_kept: A list of steps in which the model was evaluated.
    """
    if not event_files:
      return None

    steps_kept = []
    for event_file in gfile.Glob(os.path.join(event_files)):
      for event in summary_iterator.summary_iterator(event_file):
        if event.step not in steps_kept:
          steps_kept.append(event.step)
    return steps_kept
