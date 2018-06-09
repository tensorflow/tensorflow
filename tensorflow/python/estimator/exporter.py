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
"""`Exporter` class represents different flavors of model export."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import os

from tensorflow.python.estimator import gc
from tensorflow.python.estimator import util
from tensorflow.python.estimator.canned import metric_keys
from tensorflow.python.framework import errors_impl
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging
from tensorflow.python.summary import summary_iterator
from tensorflow.python.util.tf_export import tf_export


@tf_export('estimator.Exporter')
class Exporter(object):
  """A class representing a type of model export."""

  @abc.abstractproperty
  def name(self):
    """Directory name.

    A directory name under the export base directory where exports of
    this type are written.  Should not be `None` nor empty.
    """
    pass

  @abc.abstractmethod
  def export(self, estimator, export_path, checkpoint_path, eval_result,
             is_the_final_export):
    """Exports the given `Estimator` to a specific format.

    Args:
      estimator: the `Estimator` to export.
      export_path: A string containing a directory where to write the export.
      checkpoint_path: The checkpoint path to export.
      eval_result: The output of `Estimator.evaluate` on this checkpoint.
      is_the_final_export: This boolean is True when this is an export in the
        end of training.  It is False for the intermediate exports during
        the training.
        When passing `Exporter` to `tf.estimator.train_and_evaluate`
        `is_the_final_export` is always False if `TrainSpec.max_steps` is
        `None`.

    Returns:
      The string path to the exported directory or `None` if export is skipped.
    """
    pass


class _SavedModelExporter(Exporter):
  """This class exports the serving graph and checkpoints.

     This class provides a basic exporting functionality and serves as a
     foundation for specialized `Exporter`s.
  """

  def __init__(self,
               name,
               serving_input_receiver_fn,
               assets_extra=None,
               as_text=False,
               strip_default_attrs=True):
    """Create an `Exporter` to use with `tf.estimator.EvalSpec`.

    Args:
      name: unique name of this `Exporter` that is going to be used in the
        export path.
      serving_input_receiver_fn: a function that takes no arguments and returns
        a `ServingInputReceiver`.
      assets_extra: An optional dict specifying how to populate the assets.extra
        directory within the exported SavedModel.  Each key should give the
        destination path (including the filename) relative to the assets.extra
        directory.  The corresponding value gives the full path of the source
        file to be copied.  For example, the simple case of copying a single
        file without renaming it is specified as
        `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.
      as_text: whether to write the SavedModel proto in text format. Defaults to
        `False`.
      strip_default_attrs: Boolean. If set, default attrs in the `GraphDef` will
        be stripped on write. This is the default behavior and recommended for
        better forward compatibility of the resulting `SavedModel`.

    Raises:
      ValueError: if any arguments is invalid.
    """
    self._name = name
    self._serving_input_receiver_fn = serving_input_receiver_fn
    self._assets_extra = assets_extra
    self._as_text = as_text
    self._strip_default_attrs = strip_default_attrs

  @property
  def name(self):
    return self._name

  def export(self, estimator, export_path, checkpoint_path, eval_result,
             is_the_final_export):
    del is_the_final_export

    export_result = estimator.export_savedmodel(
        export_path,
        self._serving_input_receiver_fn,
        assets_extra=self._assets_extra,
        as_text=self._as_text,
        checkpoint_path=checkpoint_path,
        strip_default_attrs=self._strip_default_attrs)

    return export_result


def _loss_smaller(best_eval_result, current_eval_result):
  """Compares two evaluation results and returns true if the 2nd one is smaller.

  Both evaluation results should have the values for MetricKeys.LOSS, which are
  used for comparison.

  Args:
    best_eval_result: best eval metrics.
    current_eval_result: current eval metrics.

  Returns:
    True if the loss of current_eval_result is smaller; otherwise, False.

  Raises:
    ValueError: If input eval result is None or no loss is available.
  """
  default_key = metric_keys.MetricKeys.LOSS
  if not best_eval_result or default_key not in best_eval_result:
    raise ValueError(
        'best_eval_result cannot be empty or no loss is found in it.')

  if not current_eval_result or default_key not in current_eval_result:
    raise ValueError(
        'current_eval_result cannot be empty or no loss is found in it.')

  return best_eval_result[default_key] > current_eval_result[default_key]


def _verify_compare_fn_args(compare_fn):
  """Verifies compare_fn arguments."""
  args = set(util.fn_args(compare_fn))
  if 'best_eval_result' not in args:
    raise ValueError(
        'compare_fn (%s) must include best_eval_result argument.' % compare_fn)
  if 'current_eval_result' not in args:
    raise ValueError(
        'compare_fn (%s) must include current_eval_result argument.' %
        compare_fn)
  non_valid_args = list(args - set(['best_eval_result', 'current_eval_result']))
  if non_valid_args:
    raise ValueError('compare_fn (%s) has following not expected args: %s' %
                     (compare_fn, non_valid_args))


@tf_export('estimator.BestExporter')
class BestExporter(Exporter):
  """This class exports the serving graph and checkpoints of the best models.

  This class performs a model export everytime when the new model is better
  than any exsiting model.
  """

  def __init__(self,
               name='best_exporter',
               serving_input_receiver_fn=None,
               event_file_pattern='eval/*.tfevents.*',
               compare_fn=_loss_smaller,
               assets_extra=None,
               as_text=False,
               exports_to_keep=5):
    """Create an `Exporter` to use with `tf.estimator.EvalSpec`.

    Example of creating a BestExporter for training and evluation:
    ```python
    def make_train_and_eval_fn():
      # Set up feature columns.
      categorial_feature_a = (
          tf.feature_column.categorical_column_with_hash_bucket(...))
      categorial_feature_a_emb = embedding_column(
          categorical_column=categorial_feature_a, ...)
      ...  # other feature columns

      estimator = tf.estimator.DNNClassifier(
          config=tf.estimator.RunConfig(
              model_dir='/my_model', save_summary_steps=100),
          feature_columns=[categorial_feature_a_emb, ...],
          hidden_units=[1024, 512, 256])

      serving_feature_spec = tf.feature_column.make_parse_example_spec(
          categorial_feature_a_emb)
      serving_input_receiver_fn = (
          tf.estimator.export.build_parsing_serving_input_receiver_fn(
          serving_feature_spec))

      exporter = tf.estimator.BestExporter(
          name="best_exporter",
          serving_input_receiver_fn=serving_input_receiver_fn,
          exports_to_keep=5)

      train_spec = tf.estimator.TrainSpec(...)

      eval_spec = [tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        steps=100,
        exporters=exporter,
        start_delay_secs=0,
        throttle_secs=5)]

      return tf.estimator.DistributedTrainingSpec(estimator, train_spec,
                                                  eval_spec)
    ```

    Args:
      name: unique name of this `Exporter` that is going to be used in the
        export path.
      serving_input_receiver_fn: a function that takes no arguments and returns
        a `ServingInputReceiver`.
      event_file_pattern: event file name pattern relative to model_dir. If
        None, however, the exporter would not be preemption-safe. To be
        preemption-safe, event_file_pattern should be specified.
      compare_fn: a function that compares two evaluation results and returns
        true if current evaluation result is better. Follows the signature:
        * Args:
          * `best_eval_result`: This is the evaluation result of the best model.
          * `current_eval_result`: This is the evaluation result of current
                 candidate model.
        * Returns:
          True if current evaluation result is better; otherwise, False.
      assets_extra: An optional dict specifying how to populate the assets.extra
        directory within the exported SavedModel.  Each key should give the
        destination path (including the filename) relative to the assets.extra
        directory.  The corresponding value gives the full path of the source
        file to be copied.  For example, the simple case of copying a single
        file without renaming it is specified as `{'my_asset_file.txt':
        '/path/to/my_asset_file.txt'}`.
      as_text: whether to write the SavedModel proto in text format. Defaults to
        `False`.
      exports_to_keep: Number of exports to keep.  Older exports will be
        garbage-collected.  Defaults to 5.  Set to `None` to disable garbage
        collection.

    Raises:
      ValueError: if any arguments is invalid.
    """
    self._compare_fn = compare_fn
    if self._compare_fn is None:
      raise ValueError('`compare_fn` must not be None.')
    _verify_compare_fn_args(self._compare_fn)

    self._saved_model_exporter = _SavedModelExporter(
        name, serving_input_receiver_fn, assets_extra, as_text)

    self._event_file_pattern = event_file_pattern
    self._model_dir = None
    self._best_eval_result = None

    self._exports_to_keep = exports_to_keep
    if exports_to_keep is not None and exports_to_keep <= 0:
      raise ValueError(
          '`exports_to_keep`, if provided, must be positive number')

  @property
  def name(self):
    return self._saved_model_exporter.name

  def export(self, estimator, export_path, checkpoint_path, eval_result,
             is_the_final_export):
    export_result = None

    if self._model_dir != estimator.model_dir and self._event_file_pattern:
      # Loads best metric from event files.
      tf_logging.info('Loading best metric from event files.')

      self._model_dir = estimator.model_dir
      full_event_file_pattern = os.path.join(self._model_dir,
                                             self._event_file_pattern)
      self._best_eval_result = self._get_best_eval_result(
          full_event_file_pattern)

    if self._best_eval_result is None or self._compare_fn(
        best_eval_result=self._best_eval_result,
        current_eval_result=eval_result):
      tf_logging.info('Performing best model export.')
      self._best_eval_result = eval_result
      export_result = self._saved_model_exporter.export(
          estimator, export_path, checkpoint_path, eval_result,
          is_the_final_export)
      self._garbage_collect_exports(export_path)

    return export_result

  def _garbage_collect_exports(self, export_dir_base):
    """Deletes older exports, retaining only a given number of the most recent.

    Export subdirectories are assumed to be named with monotonically increasing
    integers; the most recent are taken to be those with the largest values.

    Args:
      export_dir_base: the base directory under which each export is in a
        versioned subdirectory.
    """
    if self._exports_to_keep is None:
      return

    def _export_version_parser(path):
      # create a simple parser that pulls the export_version from the directory.
      filename = os.path.basename(path.path)
      if not (len(filename) == 10 and filename.isdigit()):
        return None
      return path._replace(export_version=int(filename))

    # pylint: disable=protected-access
    keep_filter = gc._largest_export_versions(self._exports_to_keep)
    delete_filter = gc._negation(keep_filter)
    for p in delete_filter(
        gc._get_paths(export_dir_base, parser=_export_version_parser)):
      try:
        gfile.DeleteRecursively(p.path)
      except errors_impl.NotFoundError as e:
        tf_logging.warn('Can not delete %s recursively: %s', p.path, e)
    # pylint: enable=protected-access

  def _get_best_eval_result(self, event_files):
    """Get the best eval result from event files.

    Args:
      event_files: Absolute pattern of event files.

    Returns:
      The best eval result.
    """
    if not event_files:
      return None

    best_eval_result = None
    for event_file in gfile.Glob(os.path.join(event_files)):
      for event in summary_iterator.summary_iterator(event_file):
        if event.HasField('summary'):
          event_eval_result = {}
          for value in event.summary.value:
            if value.HasField('simple_value'):
              event_eval_result[value.tag] = value.simple_value
          if event_eval_result:
            if best_eval_result is None or self._compare_fn(
                best_eval_result, event_eval_result):
              best_eval_result = event_eval_result
    return best_eval_result


@tf_export('estimator.FinalExporter')
class FinalExporter(Exporter):
  """This class exports the serving graph and checkpoints in the end.

  This class performs a single export in the end of training.
  """

  def __init__(self,
               name,
               serving_input_receiver_fn,
               assets_extra=None,
               as_text=False):
    """Create an `Exporter` to use with `tf.estimator.EvalSpec`.

    Args:
      name: unique name of this `Exporter` that is going to be used in the
        export path.
      serving_input_receiver_fn: a function that takes no arguments and returns
        a `ServingInputReceiver`.
      assets_extra: An optional dict specifying how to populate the assets.extra
        directory within the exported SavedModel.  Each key should give the
        destination path (including the filename) relative to the assets.extra
        directory.  The corresponding value gives the full path of the source
        file to be copied.  For example, the simple case of copying a single
        file without renaming it is specified as
        `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.
      as_text: whether to write the SavedModel proto in text format. Defaults to
        `False`.

    Raises:
      ValueError: if any arguments is invalid.
    """
    self._saved_model_exporter = _SavedModelExporter(
        name, serving_input_receiver_fn, assets_extra, as_text)

  @property
  def name(self):
    return self._saved_model_exporter.name

  def export(self, estimator, export_path, checkpoint_path, eval_result,
             is_the_final_export):
    if not is_the_final_export:
      return None

    tf_logging.info('Performing the final export in the end of training.')

    return self._saved_model_exporter.export(estimator, export_path,
                                             checkpoint_path, eval_result,
                                             is_the_final_export)


@tf_export('estimator.LatestExporter')
class LatestExporter(Exporter):
  """This class regularly exports the serving graph and checkpoints.

  In addition to exporting, this class also garbage collects stale exports.
  """

  def __init__(self,
               name,
               serving_input_receiver_fn,
               assets_extra=None,
               as_text=False,
               exports_to_keep=5):
    """Create an `Exporter` to use with `tf.estimator.EvalSpec`.

    Args:
      name: unique name of this `Exporter` that is going to be used in the
        export path.
      serving_input_receiver_fn: a function that takes no arguments and returns
        a `ServingInputReceiver`.
      assets_extra: An optional dict specifying how to populate the assets.extra
        directory within the exported SavedModel.  Each key should give the
        destination path (including the filename) relative to the assets.extra
        directory.  The corresponding value gives the full path of the source
        file to be copied.  For example, the simple case of copying a single
        file without renaming it is specified as
        `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.
      as_text: whether to write the SavedModel proto in text format. Defaults to
        `False`.
      exports_to_keep: Number of exports to keep.  Older exports will be
        garbage-collected.  Defaults to 5.  Set to `None` to disable garbage
        collection.

    Raises:
      ValueError: if any arguments is invalid.
    """
    self._saved_model_exporter = _SavedModelExporter(
        name, serving_input_receiver_fn, assets_extra, as_text)
    self._exports_to_keep = exports_to_keep
    if exports_to_keep is not None and exports_to_keep <= 0:
      raise ValueError(
          '`exports_to_keep`, if provided, must be positive number')

  @property
  def name(self):
    return self._saved_model_exporter.name

  def export(self, estimator, export_path, checkpoint_path, eval_result,
             is_the_final_export):
    export_result = self._saved_model_exporter.export(
        estimator, export_path, checkpoint_path, eval_result,
        is_the_final_export)

    self._garbage_collect_exports(export_path)
    return export_result

  def _garbage_collect_exports(self, export_dir_base):
    """Deletes older exports, retaining only a given number of the most recent.

    Export subdirectories are assumed to be named with monotonically increasing
    integers; the most recent are taken to be those with the largest values.

    Args:
      export_dir_base: the base directory under which each export is in a
        versioned subdirectory.
    """
    if self._exports_to_keep is None:
      return

    def _export_version_parser(path):
      # create a simple parser that pulls the export_version from the directory.
      filename = os.path.basename(path.path)
      if not (len(filename) == 10 and filename.isdigit()):
        return None
      return path._replace(export_version=int(filename))

    # pylint: disable=protected-access
    keep_filter = gc._largest_export_versions(self._exports_to_keep)
    delete_filter = gc._negation(keep_filter)
    for p in delete_filter(
        gc._get_paths(export_dir_base, parser=_export_version_parser)):
      try:
        gfile.DeleteRecursively(p.path)
      except errors_impl.NotFoundError as e:
        tf_logging.warn('Can not delete %s recursively: %s', p.path, e)
    # pylint: enable=protected-access
