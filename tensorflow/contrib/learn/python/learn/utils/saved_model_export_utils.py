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
"""Utilities supporting export to SavedModel.

Some contents of this file are moved to tensorflow/python/estimator/export.py:

get_input_alternatives() -> obsolete
get_output_alternatives() -> obsolete, but see _get_default_export_output()
build_all_signature_defs() -> build_all_signature_defs()
get_timestamped_export_directory() -> get_timestamped_export_directory()
_get_* -> obsolete
_is_* -> obsolete

Functionality of build_standardized_signature_def() is moved to
tensorflow/python/estimator/export_output.py as ExportOutput.as_signature_def().

Anything to do with ExportStrategies or garbage collection is not moved.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from tensorflow.contrib.layers.python.layers import feature_column
from tensorflow.contrib.learn.python.learn import export_strategy
from tensorflow.contrib.learn.python.learn.estimators import constants
from tensorflow.contrib.learn.python.learn.estimators import metric_key
from tensorflow.contrib.learn.python.learn.estimators import prediction_key
from tensorflow.contrib.learn.python.learn.utils import gc
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils
from tensorflow.python.estimator import estimator as core_estimator
from tensorflow.python.estimator.export import export as core_export
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.training import saver

from tensorflow.python.util import compat

# A key for use in the input_alternatives dict indicating the default input.
# This is the input that will be expected when a serving request does not
# specify a specific signature.
# The default input alternative specifies placeholders that the input_fn
# requires to be fed (in the typical case, a single placeholder for a
# serialized tf.Example).
DEFAULT_INPUT_ALTERNATIVE_KEY = 'default_input_alternative'

# A key for use in the input_alternatives dict indicating the features input.
# The features inputs alternative specifies the feature Tensors provided as
# input to the model_fn, i.e. the outputs of the input_fn.
FEATURES_INPUT_ALTERNATIVE_KEY = 'features_input_alternative'

# A key for use in the output_alternatives dict indicating the default output.
# This is the output that will be provided when a serving request does not
# specify a specific signature.
# In a single-headed model, the single output is automatically the default.
# In a multi-headed model, the name of the desired default head should be
# provided to get_output_alternatives.
_FALLBACK_DEFAULT_OUTPUT_ALTERNATIVE_KEY = 'default_output_alternative'


def build_standardized_signature_def(input_tensors, output_tensors,
                                     problem_type):
  """Build a SignatureDef using problem type and input and output Tensors.

  Note that this delegates the actual creation of the signatures to methods in
  //third_party/tensorflow/python/saved_model/signature_def_utils.py, which may
  assign names to the input and output tensors (depending on the problem type)
  that are standardized in the context of SavedModel.

  Args:
    input_tensors: a dict of string key to `Tensor`
    output_tensors: a dict of string key to `Tensor`
    problem_type: an instance of constants.ProblemType, specifying
      classification, regression, etc.

  Returns:
    A SignatureDef using SavedModel standard keys where possible.

  Raises:
    ValueError: if input_tensors or output_tensors is None or empty.
  """

  if not input_tensors:
    raise ValueError('input_tensors must be provided.')
  if not output_tensors:
    raise ValueError('output_tensors must be provided.')

  # Per-method signature_def functions will standardize the keys if possible
  if _is_classification_problem(problem_type, input_tensors, output_tensors):
    (_, examples), = input_tensors.items()
    classes = _get_classification_classes(output_tensors)
    scores = _get_classification_scores(output_tensors)
    if classes is None and scores is None:
      items = list(output_tensors.items())
      if items[0][1].dtype == dtypes.string:
        (_, classes), = items
      else:
        (_, scores), = items
    return signature_def_utils.classification_signature_def(
        examples, classes, scores)
  elif _is_regression_problem(problem_type, input_tensors, output_tensors):
    (_, examples), = input_tensors.items()
    (_, predictions), = output_tensors.items()
    return signature_def_utils.regression_signature_def(examples, predictions)
  else:
    return signature_def_utils.predict_signature_def(input_tensors,
                                                     output_tensors)


def _get_classification_scores(output_tensors):
  scores = output_tensors.get(prediction_key.PredictionKey.SCORES)
  if scores is None:
    scores = output_tensors.get(prediction_key.PredictionKey.PROBABILITIES)
  return scores


def _get_classification_classes(output_tensors):
  classes = output_tensors.get(prediction_key.PredictionKey.CLASSES)
  if classes is not None and classes.dtype != dtypes.string:
    # Servo classification can only serve string classes.
    return None
  return classes


def _is_classification_problem(problem_type, input_tensors, output_tensors):
  classes = _get_classification_classes(output_tensors)
  scores = _get_classification_scores(output_tensors)
  return ((problem_type == constants.ProblemType.CLASSIFICATION or
           problem_type == constants.ProblemType.LOGISTIC_REGRESSION) and
          len(input_tensors) == 1 and
          (classes is not None or scores is not None or
           len(output_tensors) == 1))


def _is_regression_problem(problem_type, input_tensors, output_tensors):
  return (problem_type == constants.ProblemType.LINEAR_REGRESSION and
          len(input_tensors) == 1 and len(output_tensors) == 1)


def get_input_alternatives(input_ops):
  """Obtain all input alternatives using the input_fn output and heuristics."""
  input_alternatives = {}
  if isinstance(input_ops, input_fn_utils.InputFnOps):
    features, unused_labels, default_inputs = input_ops
    input_alternatives[DEFAULT_INPUT_ALTERNATIVE_KEY] = default_inputs
  else:
    features, unused_labels = input_ops

  if not features:
    raise ValueError('Features must be defined.')

  # TODO(b/34253951): reinstate the "features" input_signature.
  # The "features" input_signature, as written, does not work with
  # SparseTensors.  It is simply commented out as a stopgap, pending discussion
  # on the bug as to the correct solution.

  # Add the "features" input_signature in any case.
  # Note defensive copy because model_fns alter the features dict.
  # input_alternatives[FEATURES_INPUT_ALTERNATIVE_KEY] = (
  #    copy.copy(features))

  return input_alternatives, features


def get_output_alternatives(model_fn_ops, default_output_alternative_key=None):
  """Obtain all output alternatives using the model_fn output and heuristics.

  Args:
    model_fn_ops: a `ModelFnOps` object produced by a `model_fn`.  This may or
      may not have output_alternatives populated.
    default_output_alternative_key: the name of the head to serve when an
      incoming serving request does not explicitly request a specific head.
      Not needed for single-headed models.

  Returns:
    A tuple of (output_alternatives, actual_default_output_alternative_key),
    where the latter names the head that will actually be served by default.
    This may differ from the requested default_output_alternative_key when
    a) no output_alternatives are provided at all, so one must be generated, or
    b) there is exactly one head, which is used regardless of the requested
    default.

  Raises:
    ValueError: if the requested default_output_alternative_key is not available
      in output_alternatives, or if there are multiple output_alternatives and
      no default is specified.
  """
  output_alternatives = model_fn_ops.output_alternatives

  if not output_alternatives:
    if default_output_alternative_key:
      raise ValueError('Requested default_output_alternative: {}, '
                       'but available output_alternatives are: []'.format(
                           default_output_alternative_key))

    # Lacking provided output alternatives, the best we can do is to
    # interpret the model as single-headed of unknown type.
    default_problem_type = constants.ProblemType.UNSPECIFIED
    default_outputs = model_fn_ops.predictions
    if not isinstance(default_outputs, dict):
      default_outputs = {prediction_key.PredictionKey.GENERIC: default_outputs}
    actual_default_output_alternative_key = (
        _FALLBACK_DEFAULT_OUTPUT_ALTERNATIVE_KEY)
    output_alternatives = {
        actual_default_output_alternative_key: (default_problem_type,
                                                default_outputs)
    }
    return output_alternatives, actual_default_output_alternative_key

  if default_output_alternative_key:
    # If a default head is provided, use it.
    if default_output_alternative_key in output_alternatives:
      return output_alternatives, default_output_alternative_key

    raise ValueError('Requested default_output_alternative: {}, '
                     'but available output_alternatives are: {}'.format(
                         default_output_alternative_key,
                         sorted(output_alternatives.keys())))

  if len(output_alternatives) == 1:
    # If there is only one head, use it as the default regardless of its name.
    (actual_default_output_alternative_key, _), = output_alternatives.items()
    return output_alternatives, actual_default_output_alternative_key

  raise ValueError('Please specify a default_output_alternative.  '
                   'Available output_alternatives are: {}'.format(
                       sorted(output_alternatives.keys())))


def build_all_signature_defs(input_alternatives, output_alternatives,
                             actual_default_output_alternative_key):
  """Build `SignatureDef`s from all pairs of input and output alternatives."""

  signature_def_map = {('%s:%s' % (input_key, output_key or 'None')):
                       build_standardized_signature_def(inputs, outputs,
                                                        problem_type)
                       for input_key, inputs in input_alternatives.items()
                       for output_key, (problem_type,
                                        outputs) in output_alternatives.items()}

  # Add the default SignatureDef
  default_inputs = input_alternatives.get(DEFAULT_INPUT_ALTERNATIVE_KEY)
  if not default_inputs:
    raise ValueError('A default input_alternative must be provided.')
    # default_inputs = input_alternatives[FEATURES_INPUT_ALTERNATIVE_KEY]
  # default outputs are guaranteed to exist above
  (default_problem_type, default_outputs) = (
      output_alternatives[actual_default_output_alternative_key])
  signature_def_map[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = (
      build_standardized_signature_def(default_inputs, default_outputs,
                                       default_problem_type))

  return signature_def_map


# When we create a timestamped directory, there is a small chance that the
# directory already exists because another worker is also writing exports.
# In this case we just wait one second to get a new timestamp and try again.
# If this fails several times in a row, then something is seriously wrong.
MAX_DIRECTORY_CREATION_ATTEMPTS = 10


def get_timestamped_export_dir(export_dir_base):
  """Builds a path to a new subdirectory within the base directory.

  Each export is written into a new subdirectory named using the
  current time.  This guarantees monotonically increasing version
  numbers even across multiple runs of the pipeline.
  The timestamp used is the number of seconds since epoch UTC.

  Args:
    export_dir_base: A string containing a directory to write the exported
        graph and checkpoints.
  Returns:
    The full path of the new subdirectory (which is not actually created yet).

  Raises:
    RuntimeError: if repeated attempts fail to obtain a unique timestamped
      directory name.
  """
  attempts = 0
  while attempts < MAX_DIRECTORY_CREATION_ATTEMPTS:
    export_timestamp = int(time.time())

    export_dir = os.path.join(
        compat.as_bytes(export_dir_base),
        compat.as_bytes(str(export_timestamp)))
    if not gfile.Exists(export_dir):
      # Collisions are still possible (though extremely unlikely): this
      # directory is not actually created yet, but it will be almost
      # instantly on return from this function.
      return export_dir
    time.sleep(1)
    attempts += 1
    logging.warn('Export directory {} already exists; retrying (attempt {}/{})'.
                 format(export_dir, attempts, MAX_DIRECTORY_CREATION_ATTEMPTS))
  raise RuntimeError('Failed to obtain a unique export directory name after '
                     '{} attempts.'.format(MAX_DIRECTORY_CREATION_ATTEMPTS))


def get_temp_export_dir(timestamped_export_dir):
  """Builds a directory name based on the argument but starting with 'temp-'.

  This relies on the fact that TensorFlow Serving ignores subdirectories of
  the base directory that can't be parsed as integers.

  Args:
    timestamped_export_dir: the name of the eventual export directory, e.g.
      /foo/bar/<timestamp>

  Returns:
    A sister directory prefixed with 'temp-', e.g. /foo/bar/temp-<timestamp>.
  """
  (dirname, basename) = os.path.split(timestamped_export_dir)
  temp_export_dir = os.path.join(
      compat.as_bytes(dirname), compat.as_bytes('temp-{}'.format(basename)))
  return temp_export_dir


# create a simple parser that pulls the export_version from the directory.
def _export_version_parser(path):
  filename = os.path.basename(path.path)
  if not (len(filename) == 10 and filename.isdigit()):
    return None
  return path._replace(export_version=int(filename))


def get_most_recent_export(export_dir_base):
  """Locate the most recent SavedModel export in a directory of many exports.

  This method assumes that SavedModel subdirectories are named as a timestamp
  (seconds from epoch), as produced by get_timestamped_export_dir().

  Args:
    export_dir_base: A base directory containing multiple timestamped
                     directories.

  Returns:
    A gc.Path, with is just a namedtuple of (path, export_version).
  """
  select_filter = gc.largest_export_versions(1)
  results = select_filter(
      gc.get_paths(export_dir_base, parser=_export_version_parser))
  return next(iter(results or []), None)


def garbage_collect_exports(export_dir_base, exports_to_keep):
  """Deletes older exports, retaining only a given number of the most recent.

  Export subdirectories are assumed to be named with monotonically increasing
  integers; the most recent are taken to be those with the largest values.

  Args:
    export_dir_base: the base directory under which each export is in a
      versioned subdirectory.
    exports_to_keep: the number of recent exports to retain.
  """
  if exports_to_keep is None:
    return

  keep_filter = gc.largest_export_versions(exports_to_keep)
  delete_filter = gc.negation(keep_filter)
  for p in delete_filter(
      gc.get_paths(export_dir_base, parser=_export_version_parser)):
    try:
      gfile.DeleteRecursively(p.path)
    except errors_impl.NotFoundError as e:
      logging.warn('Can not delete %s recursively: %s', p.path, e)


def make_export_strategy(serving_input_fn,
                         default_output_alternative_key=None,
                         assets_extra=None,
                         as_text=False,
                         exports_to_keep=5,
                         strip_default_attrs=False):
  # pylint: disable=line-too-long
  """Create an ExportStrategy for use with Experiment.

  Args:
    serving_input_fn: A function that takes no arguments and returns an
      `InputFnOps`.
    default_output_alternative_key: the name of the head to serve when an
      incoming serving request does not explicitly request a specific head.
      Must be `None` if the estimator inherits from ${tf.estimator.Estimator}
      or for single-headed models.
    assets_extra: A dict specifying how to populate the assets.extra directory
      within the exported SavedModel.  Each key should give the destination
      path (including the filename) relative to the assets.extra directory.
      The corresponding value gives the full path of the source file to be
      copied.  For example, the simple case of copying a single file without
      renaming it is specified as
      `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.
    as_text: whether to write the SavedModel proto in text format.
    exports_to_keep: Number of exports to keep.  Older exports will be
      garbage-collected.  Defaults to 5.  Set to None to disable garbage
      collection.
    strip_default_attrs: Boolean. If `True`, default-valued attributes will be
      removed from the NodeDefs. For a detailed guide, see
      [Stripping Default-Valued Attributes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes).

  Returns:
    An ExportStrategy that can be passed to the Experiment constructor.
  """
  # pylint: enable=line-too-long

  def export_fn(estimator, export_dir_base, checkpoint_path=None):
    """Exports the given Estimator as a SavedModel.

    Args:
      estimator: the Estimator to export.
      export_dir_base: A string containing a directory to write the exported
        graph and checkpoints.
      checkpoint_path: The checkpoint path to export.  If None (the default),
        the most recent checkpoint found within the model directory is chosen.

    Returns:
      The string path to the exported directory.

    Raises:
      ValueError: If `estimator` is a ${tf.estimator.Estimator} instance
        and `default_output_alternative_key` was specified.
    """
    if isinstance(estimator, core_estimator.Estimator):
      if default_output_alternative_key is not None:
        raise ValueError(
            'default_output_alternative_key is not supported in core '
            'Estimator. Given: {}'.format(default_output_alternative_key))
      export_result = estimator.export_savedmodel(
          export_dir_base,
          serving_input_fn,
          assets_extra=assets_extra,
          as_text=as_text,
          checkpoint_path=checkpoint_path,
          strip_default_attrs=strip_default_attrs)
    else:
      export_result = estimator.export_savedmodel(
          export_dir_base,
          serving_input_fn,
          default_output_alternative_key=default_output_alternative_key,
          assets_extra=assets_extra,
          as_text=as_text,
          checkpoint_path=checkpoint_path,
          strip_default_attrs=strip_default_attrs)

    garbage_collect_exports(export_dir_base, exports_to_keep)
    return export_result

  return export_strategy.ExportStrategy('Servo', export_fn)


def make_parsing_export_strategy(feature_columns,
                                 default_output_alternative_key=None,
                                 assets_extra=None,
                                 as_text=False,
                                 exports_to_keep=5,
                                 target_core=False,
                                 strip_default_attrs=False):
  # pylint: disable=line-too-long
  """Create an ExportStrategy for use with Experiment, using `FeatureColumn`s.

  Creates a SavedModel export that expects to be fed with a single string
  Tensor containing serialized tf.Examples.  At serving time, incoming
  tf.Examples will be parsed according to the provided `FeatureColumn`s.

  Args:
    feature_columns: An iterable of `FeatureColumn`s representing the features
      that must be provided at serving time (excluding labels!).
    default_output_alternative_key: the name of the head to serve when an
      incoming serving request does not explicitly request a specific head.
      Must be `None` if the estimator inherits from ${tf.estimator.Estimator}
      or for single-headed models.
    assets_extra: A dict specifying how to populate the assets.extra directory
      within the exported SavedModel.  Each key should give the destination
      path (including the filename) relative to the assets.extra directory.
      The corresponding value gives the full path of the source file to be
      copied.  For example, the simple case of copying a single file without
      renaming it is specified as
      `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.
    as_text: whether to write the SavedModel proto in text format.
    exports_to_keep: Number of exports to keep.  Older exports will be
      garbage-collected.  Defaults to 5.  Set to None to disable garbage
      collection.
    target_core: If True, prepare an ExportStrategy for use with
      tensorflow.python.estimator.*.  If False (default), prepare an
      ExportStrategy for use with tensorflow.contrib.learn.python.learn.*.
    strip_default_attrs: Boolean. If `True`, default-valued attributes will be
      removed from the NodeDefs. For a detailed guide, see
      [Stripping Default-Valued Attributes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes).

  Returns:
    An ExportStrategy that can be passed to the Experiment constructor.
  """
  # pylint: enable=line-too-long
  feature_spec = feature_column.create_feature_spec_for_parsing(feature_columns)
  if target_core:
    serving_input_fn = (
        core_export.build_parsing_serving_input_receiver_fn(feature_spec))
  else:
    serving_input_fn = (
        input_fn_utils.build_parsing_serving_input_fn(feature_spec))
  return make_export_strategy(
      serving_input_fn,
      default_output_alternative_key=default_output_alternative_key,
      assets_extra=assets_extra,
      as_text=as_text,
      exports_to_keep=exports_to_keep,
      strip_default_attrs=strip_default_attrs)


def _default_compare_fn(curr_best_eval_result, cand_eval_result):
  """Compares two evaluation results and returns true if the 2nd one is better.

  Both evaluation results should have the values for MetricKey.LOSS, which are
  used for comparison.

  Args:
    curr_best_eval_result: current best eval metrics.
    cand_eval_result: candidate eval metrics.

  Returns:
    True if cand_eval_result is better.

  Raises:
    ValueError: If input eval result is None or no loss is available.
  """
  default_key = metric_key.MetricKey.LOSS
  if not curr_best_eval_result or default_key not in curr_best_eval_result:
    raise ValueError(
        'curr_best_eval_result cannot be empty or no loss is found in it.')

  if not cand_eval_result or default_key not in cand_eval_result:
    raise ValueError(
        'cand_eval_result cannot be empty or no loss is found in it.')

  return curr_best_eval_result[default_key] > cand_eval_result[default_key]


class BestModelSelector(object):
  """A helper that keeps track of export selection candidates."""

  def __init__(self, compare_fn=None):
    """Constructor of this class.

    Args:
      compare_fn: a function that returns true if the candidate is better than
        the current best model.
    """
    self._best_eval_result = None
    self._compare_fn = compare_fn or _default_compare_fn

  def update(self, checkpoint_path, eval_result):
    """Records a given checkpoint and exports if this is the best model.

    Args:
      checkpoint_path: the checkpoint path to export.
      eval_result: a dictionary which is usually generated in evaluation runs.
        By default, eval_results contains 'loss' field.

    Returns:
      A string representing the path to the checkpoint to be exported.
      A dictionary of the same type of eval_result.

    Raises:
      ValueError: if checkpoint path is empty.
      ValueError: if eval_results is None object.
    """
    if not checkpoint_path:
      raise ValueError('Checkpoint path is empty.')
    if eval_result is None:
      raise ValueError('%s has empty evaluation results.', checkpoint_path)

    if (self._best_eval_result is None or
        self._compare_fn(self._best_eval_result, eval_result)):
      self._best_eval_result = eval_result
      return checkpoint_path, eval_result
    else:
      return '', None


def make_best_model_export_strategy(serving_input_fn,
                                    exports_to_keep=1,
                                    compare_fn=None,
                                    default_output_alternative_key=None,
                                    strip_default_attrs=False):
  # pylint: disable=line-too-long
  """Creates an custom ExportStrategy for use with tf.contrib.learn.Experiment.

  Args:
    serving_input_fn: a function that takes no arguments and returns an
      `InputFnOps`.
    exports_to_keep: an integer indicating how many historical best models need
      to be preserved.
    compare_fn: a function that select the 'best' candidate from a dictionary
        of evaluation result keyed by corresponding checkpoint path.
    default_output_alternative_key: the key for default serving signature for
        multi-headed inference graphs.
    strip_default_attrs: Boolean. If `True`, default-valued attributes will be
      removed from the NodeDefs. For a detailed guide, see
      [Stripping Default-Valued Attributes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes).

  Returns:
    An ExportStrategy that can be passed to the Experiment constructor.
  """
  # pylint: enable=line-too-long
  best_model_export_strategy = make_export_strategy(
      serving_input_fn,
      exports_to_keep=exports_to_keep,
      default_output_alternative_key=default_output_alternative_key,
      strip_default_attrs=strip_default_attrs)

  best_model_selector = BestModelSelector(compare_fn)

  def export_fn(estimator, export_dir_base, checkpoint_path, eval_result=None):
    """Exports the given Estimator as a SavedModel.

    Args:
      estimator: the Estimator to export.
      export_dir_base: A string containing a directory to write the exported
        graph and checkpoints.
      checkpoint_path: The checkpoint path to export.  If None (the default),
        the most recent checkpoint found within the model directory is chosen.
      eval_result: placehold args matching the call signature of ExportStrategy.

    Returns:
      The string path to the exported directory.
    """
    if not checkpoint_path:
      # TODO(b/67425018): switch to
      #    checkpoint_path = estimator.latest_checkpoint()
      #  as soon as contrib is cleaned up and we can thus be sure that
      #  estimator is a tf.estimator.Estimator and not a
      #  tf.contrib.learn.Estimator
      checkpoint_path = saver.latest_checkpoint(estimator.model_dir)
    export_checkpoint_path, export_eval_result = best_model_selector.update(
        checkpoint_path, eval_result)

    if export_checkpoint_path and export_eval_result is not None:
      checkpoint_base = os.path.basename(export_checkpoint_path)
      export_dir = os.path.join(export_dir_base, checkpoint_base)
      return best_model_export_strategy.export(
          estimator, export_dir, export_checkpoint_path, export_eval_result)
    else:
      return ''

  return export_strategy.ExportStrategy('best_model', export_fn)


# TODO(b/67013778): Revisit this approach when corresponding changes to
# TF Core are finalized.
def extend_export_strategy(base_export_strategy,
                           post_export_fn,
                           post_export_name=None):
  """Extend ExportStrategy, calling post_export_fn after export.

  Args:
    base_export_strategy: An ExportStrategy that can be passed to the Experiment
      constructor.
    post_export_fn: A user-specified function to call after exporting the
      SavedModel. Takes two arguments - the path to the SavedModel exported by
      base_export_strategy and the directory where to export the SavedModel
      modified by the post_export_fn. Returns the path to the exported
      SavedModel.
    post_export_name: The directory name under the export base directory where
      SavedModels generated by the post_export_fn will be written. If None, the
      directory name of base_export_strategy is used.

  Returns:
    An ExportStrategy that can be passed to the Experiment constructor.
  """
  def export_fn(estimator, export_dir_base, checkpoint_path=None):
    """Exports the given Estimator as a SavedModel and invokes post_export_fn.

    Args:
      estimator: the Estimator to export.
      export_dir_base: A string containing a directory to write the exported
        graphs and checkpoint.
      checkpoint_path: The checkpoint path to export. If None (the default),
        the most recent checkpoint found within the model directory is chosen.

    Returns:
      The string path to the SavedModel indicated by post_export_fn.

    Raises:
      ValueError: If `estimator` is a ${tf.estimator.Estimator} instance
        and `default_output_alternative_key` was specified or if post_export_fn
        does not return a valid directory.
      RuntimeError: If unable to create temporary or final export directory.
    """
    tmp_base_export_folder = 'temp-base-export-' + str(int(time.time()))
    tmp_base_export_dir = os.path.join(export_dir_base, tmp_base_export_folder)
    if gfile.Exists(tmp_base_export_dir):
      raise RuntimeError('Failed to obtain base export directory')
    gfile.MakeDirs(tmp_base_export_dir)
    tmp_base_export = base_export_strategy.export(
        estimator, tmp_base_export_dir, checkpoint_path)

    tmp_post_export_folder = 'temp-post-export-' + str(int(time.time()))
    tmp_post_export_dir = os.path.join(export_dir_base, tmp_post_export_folder)
    if gfile.Exists(tmp_post_export_dir):
      raise RuntimeError('Failed to obtain temp export directory')

    gfile.MakeDirs(tmp_post_export_dir)
    tmp_post_export = post_export_fn(tmp_base_export, tmp_post_export_dir)

    if not tmp_post_export.startswith(tmp_post_export_dir):
      raise ValueError('post_export_fn must return a sub-directory of {}'
                       .format(tmp_post_export_dir))
    post_export_relpath = os.path.relpath(tmp_post_export, tmp_post_export_dir)
    post_export = os.path.join(export_dir_base, post_export_relpath)
    if gfile.Exists(post_export):
      raise RuntimeError('Failed to obtain final export directory')
    gfile.Rename(tmp_post_export, post_export)

    gfile.DeleteRecursively(tmp_base_export_dir)
    gfile.DeleteRecursively(tmp_post_export_dir)
    return post_export

  name = post_export_name if post_export_name else base_export_strategy.name
  return export_strategy.ExportStrategy(name, export_fn)
