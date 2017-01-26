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

"""Utilities supporting export to SavedModel."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import time

from tensorflow.contrib.learn.python.learn import export_strategy
from tensorflow.contrib.learn.python.learn.estimators import constants
from tensorflow.contrib.learn.python.learn.estimators import prediction_key
from tensorflow.contrib.learn.python.learn.utils import gc
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils

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
DEFAULT_OUTPUT_ALTERNATIVE_KEY = 'default_output_alternative'


def build_standardized_signature_def(
    input_tensors, output_tensors, problem_type):
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
    classes = output_tensors.get(prediction_key.PredictionKey.CLASSES)
    scores = _get_classification_scores(output_tensors)
    if classes is None and scores is None:
      (_, classes), = output_tensors.items()
    return signature_def_utils.classification_signature_def(
        examples, classes, scores)
  elif _is_regression_problem(problem_type, input_tensors, output_tensors):
    (_, examples), = input_tensors.items()
    (_, predictions), = output_tensors.items()
    return signature_def_utils.regression_signature_def(examples, predictions)
  else:
    return signature_def_utils.predict_signature_def(
        input_tensors, output_tensors)


def _get_classification_scores(output_tensors):
  scores = output_tensors.get(prediction_key.PredictionKey.SCORES)
  if scores is None:
    scores = output_tensors.get(prediction_key.PredictionKey.PROBABILITIES)
  return scores


def _is_classification_problem(problem_type, input_tensors, output_tensors):
  classes = output_tensors.get(prediction_key.PredictionKey.CLASSES)
  scores = _get_classification_scores(output_tensors)
  return ((problem_type == constants.ProblemType.CLASSIFICATION or
           problem_type == constants.ProblemType.LOGISTIC_REGRESSION)
          and len(input_tensors) == 1
          and (classes is not None or
               scores is not None or
               len(output_tensors) == 1))


def _is_regression_problem(problem_type, input_tensors, output_tensors):
  return (problem_type == constants.ProblemType.LINEAR_REGRESSION
          and len(input_tensors) == 1
          and len(output_tensors) == 1)


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


def get_output_alternatives(
    model_fn_ops,
    default_output_alternative_key=DEFAULT_OUTPUT_ALTERNATIVE_KEY):
  """Obtain all output alternatives using the model_fn output and heuristics."""
  output_alternatives = model_fn_ops.output_alternatives

  # Identify the default outputs, creating them if needed.
  if (output_alternatives
      and default_output_alternative_key not in output_alternatives):
    raise ValueError('default_output_alternative_key not in '
                     'output_alternatives: %s' % default_output_alternative_key)

  if (output_alternatives
      and default_output_alternative_key in output_alternatives):
    # If a default head is provided, use it.
    actual_default_output_alternative_key = default_output_alternative_key
    return output_alternatives, actual_default_output_alternative_key

  if output_alternatives and len(output_alternatives) == 1:
    # If there is only one head, use it as the default.
    (actual_default_output_alternative_key, _), = output_alternatives.items()
    return output_alternatives, actual_default_output_alternative_key

  # Lacking provided output alternatives, the best we can do is to
  # interpret the model as single-headed of unknown type.
  default_problem_type = constants.ProblemType.UNSPECIFIED
  default_outputs = model_fn_ops.predictions
  if not isinstance(default_outputs, dict):
    default_outputs = {prediction_key.PredictionKey.GENERIC: default_outputs}
  actual_default_output_alternative_key = DEFAULT_OUTPUT_ALTERNATIVE_KEY
  output_alternatives = {actual_default_output_alternative_key:
                         (default_problem_type, default_outputs)}
  return output_alternatives, actual_default_output_alternative_key


def build_all_signature_defs(input_alternatives, output_alternatives,
                             actual_default_output_alternative_key):
  """Build `SignatureDef`s from all pairs of input and output alternatives."""

  signature_def_map = {
      ('%s:%s' % (input_key, output_key or 'None')):
      build_standardized_signature_def(
          inputs, outputs, problem_type)
      for input_key, inputs in input_alternatives.items()
      for output_key, (problem_type, outputs)
      in output_alternatives.items()}

  # Add the default SignatureDef
  default_inputs = input_alternatives.get(DEFAULT_INPUT_ALTERNATIVE_KEY)
  if not default_inputs:
    raise ValueError('A default input_alternative must be provided.')
    # default_inputs = input_alternatives[FEATURES_INPUT_ALTERNATIVE_KEY]
  # default outputs are guaranteed to exist above
  (default_problem_type, default_outputs) = (
      output_alternatives[actual_default_output_alternative_key])
  signature_def_map[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = (
      build_standardized_signature_def(
          default_inputs, default_outputs, default_problem_type))

  return signature_def_map


def get_timestamped_export_dir(export_dir_base):
  """Builds a path to a new subdirectory within the base directory.

  Each export is written into a new subdirectory named using the
  current time.  This guarantees monotonically increasing version
  numbers even across multiple runs of the pipeline.
  The timestamp used is the number of milliseconds since epoch UTC.

  Args:
    export_dir_base: A string containing a directory to write the exported
        graph and checkpoints.
  Returns:
    The full path of the new subdirectory (which is not actually created yet).
  """
  export_timestamp = int(time.time() * 1e3)

  export_dir = os.path.join(
      compat.as_bytes(export_dir_base),
      compat.as_bytes(str(export_timestamp)))
  return export_dir


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

  # Export dir must not end with / or it will break the re match below.
  if export_dir_base.endswith('/'):
    export_dir_base = export_dir_base[:-1]

  # create a simple parser that pulls the export_version from the directory.
  def parser(path):
    match = re.match('^' + export_dir_base + '/(\\d{13})$', path.path)
    if not match:
      return None
    return path._replace(export_version=int(match.group(1)))

  for p in delete_filter(gc.get_paths(export_dir_base, parser=parser)):
    gfile.DeleteRecursively(p.path)


def make_export_strategy(export_input_fn,
                         default_output_alternative_key='default',
                         assets_extra=None,
                         as_text=False,
                         exports_to_keep=None):
  """Create an ExportStrategy for use with Experiment."""

  def export_fn(estimator, export_dir_base):
    """Exports the given Estimator as a SavedModel."""
    export_result = estimator.export_savedmodel(
        export_dir_base,
        export_input_fn,
        default_output_alternative_key=default_output_alternative_key,
        assets_extra=assets_extra,
        as_text=as_text,
        exports_to_keep=exports_to_keep)

    garbage_collect_exports(export_dir_base, exports_to_keep)
    return export_result

  return export_strategy.ExportStrategy('Servo', export_fn)
