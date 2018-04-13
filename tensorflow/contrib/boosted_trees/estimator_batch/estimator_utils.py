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
"""Utilities for converting between core and contrib feature columns."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.learn.python.learn.estimators import constants
from tensorflow.contrib.learn.python.learn.estimators import model_fn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as contrib_model_fn_lib
from tensorflow.contrib.learn.python.learn.estimators import prediction_key
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator.export import export_output

_CORE_MODE_TO_CONTRIB_MODE_ = {
    model_fn_lib.ModeKeys.TRAIN: contrib_model_fn_lib.ModeKeys.TRAIN,
    model_fn_lib.ModeKeys.EVAL: contrib_model_fn_lib.ModeKeys.EVAL,
    model_fn_lib.ModeKeys.PREDICT: contrib_model_fn_lib.ModeKeys.INFER
}


def _core_mode_to_contrib_mode(mode):
  return _CORE_MODE_TO_CONTRIB_MODE_[mode]


def _export_outputs_to_output_alternatives(export_outputs):
  """Converts EstimatorSpec.export_outputs to output_alternatives.

  Args:
    export_outputs: export_outputs created by create_estimator_spec.
  Returns:
    converted output_alternatives.
  """
  output = dict()
  if export_outputs is not None:
    for key, value in export_outputs.items():
      if isinstance(value, export_output.ClassificationOutput):
        exported_predictions = {
            prediction_key.PredictionKey.SCORES: value.scores,
            prediction_key.PredictionKey.CLASSES: value.classes
        }
        output[key] = (constants.ProblemType.CLASSIFICATION,
                       exported_predictions)
    return output
  return None


def estimator_spec_to_model_fn_ops(estimator_spec):
  alternatives = _export_outputs_to_output_alternatives(
      estimator_spec.export_outputs)

  return model_fn.ModelFnOps(
      mode=_core_mode_to_contrib_mode(estimator_spec.mode),
      predictions=estimator_spec.predictions,
      loss=estimator_spec.loss,
      train_op=estimator_spec.train_op,
      eval_metric_ops=estimator_spec.eval_metric_ops,
      output_alternatives=alternatives)
