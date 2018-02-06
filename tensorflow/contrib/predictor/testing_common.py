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

"""Common code used for testing `Predictor`s."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.learn.python.learn.estimators import constants
from tensorflow.contrib.learn.python.learn.estimators import estimator as contrib_estimator
from tensorflow.contrib.learn.python.learn.estimators import model_fn as contrib_model_fn
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils
from tensorflow.python.estimator import estimator as core_estimator
from tensorflow.python.estimator import model_fn
from tensorflow.python.estimator.export import export_lib
from tensorflow.python.estimator.export import export_output
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.saved_model import signature_constants


def get_arithmetic_estimator(core=True, model_dir=None):
  """Returns an `Estimator` that performs basic arithmetic.

  Args:
    core: if `True`, returns a `tensorflow.python.estimator.Estimator`.
      Otherwise, returns a `tensorflow.contrib.learn.Estimator`.
    model_dir: directory in which to export checkpoints and saved models.
  Returns:
    An `Estimator` that performs arithmetic operations on its inputs.
  """
  def _model_fn(features, labels, mode):
    _ = labels
    x = features['x']
    y = features['y']
    with ops.name_scope('outputs'):
      predictions = {'sum': math_ops.add(x, y, name='sum'),
                     'product': math_ops.multiply(x, y, name='product'),
                     'difference': math_ops.subtract(x, y, name='difference')}
    if core:
      export_outputs = {k: export_output.PredictOutput({k: v})
                        for k, v in predictions.items()}
      export_outputs[signature_constants.
                     DEFAULT_SERVING_SIGNATURE_DEF_KEY] = export_outputs['sum']
      return model_fn.EstimatorSpec(mode=mode,
                                    predictions=predictions,
                                    export_outputs=export_outputs,
                                    loss=constant_op.constant(0),
                                    train_op=control_flow_ops.no_op())
    else:
      output_alternatives = {k: (constants.ProblemType.UNSPECIFIED, {k: v})
                             for k, v in predictions.items()}
      return contrib_model_fn.ModelFnOps(
          mode=mode,
          predictions=predictions,
          output_alternatives=output_alternatives,
          loss=constant_op.constant(0),
          train_op=control_flow_ops.no_op())
  if core:
    return core_estimator.Estimator(_model_fn)
  else:
    return contrib_estimator.Estimator(_model_fn, model_dir=model_dir)


def get_arithmetic_input_fn(core=True, train=False):
  """Returns a input functions or serving input receiver function."""
  def _input_fn():
    with ops.name_scope('inputs'):
      x = array_ops.placeholder_with_default(0.0, shape=[], name='x')
      y = array_ops.placeholder_with_default(0.0, shape=[], name='y')
    label = constant_op.constant(0.0)
    features = {'x': x, 'y': y}
    if core:
      if train:
        return features, label
      return export_lib.ServingInputReceiver(
          features=features,
          receiver_tensors=features)
    else:
      if train:
        return features, label
      return input_fn_utils.InputFnOps(
          features=features,
          labels={},
          default_inputs=features)
  return _input_fn
