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
"""SignatureDef utility functions implementation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import utils


def build_signature_def(inputs=None, outputs=None, method_name=None):
  """Utility function to build a SignatureDef protocol buffer.

  Args:
    inputs: Inputs of the SignatureDef defined as a proto map of string to
        tensor info.
    outputs: Outputs of the SignatureDef defined as a proto map of string to
        tensor info.
    method_name: Method name of the SignatureDef as a string.

  Returns:
    A SignatureDef protocol buffer constructed based on the supplied arguments.
  """
  signature_def = meta_graph_pb2.SignatureDef()
  if inputs is not None:
    for item in inputs:
      signature_def.inputs[item].CopyFrom(inputs[item])
  if outputs is not None:
    for item in outputs:
      signature_def.outputs[item].CopyFrom(outputs[item])
  if method_name is not None:
    signature_def.method_name = method_name
  return signature_def


def regression_signature_def(examples, predictions):
  """Creates regression signature from given examples and predictions.

  Args:
    examples: `Tensor`.
    predictions: `Tensor`.

  Returns:
    A regression-flavored signature_def.

  Raises:
    ValueError: If examples is `None`.
  """
  if examples is None:
    raise ValueError('examples cannot be None for regression.')
  if predictions is None:
    raise ValueError('predictions cannot be None for regression.')

  input_tensor_info = utils.build_tensor_info(examples)
  signature_inputs = {signature_constants.REGRESS_INPUTS: input_tensor_info}

  output_tensor_info = utils.build_tensor_info(predictions)
  signature_outputs = {signature_constants.REGRESS_OUTPUTS: output_tensor_info}
  signature_def = build_signature_def(
      signature_inputs, signature_outputs,
      signature_constants.REGRESS_METHOD_NAME)

  return signature_def


def classification_signature_def(examples, classes, scores):
  """Creates classification signature from given examples and predictions.

  Args:
    examples: `Tensor`.
    classes: `Tensor`.
    scores: `Tensor`.

  Returns:
    A classification-flavored signature_def.

  Raises:
    ValueError: If examples is `None`.
  """
  if examples is None:
    raise ValueError('examples cannot be None for classification.')
  if classes is None and scores is None:
    raise ValueError('classes and scores cannot both be None for '
                     'classification.')

  input_tensor_info = utils.build_tensor_info(examples)
  signature_inputs = {signature_constants.CLASSIFY_INPUTS: input_tensor_info}

  signature_outputs = {}
  if classes is not None:
    classes_tensor_info = utils.build_tensor_info(classes)
    signature_outputs[signature_constants.CLASSIFY_OUTPUT_CLASSES] = (
        classes_tensor_info)
  if scores is not None:
    scores_tensor_info = utils.build_tensor_info(scores)
    signature_outputs[signature_constants.CLASSIFY_OUTPUT_SCORES] = (
        scores_tensor_info)

  signature_def = build_signature_def(
      signature_inputs, signature_outputs,
      signature_constants.CLASSIFY_METHOD_NAME)

  return signature_def


def predict_signature_def(inputs, outputs):
  """Creates prediction signature from given inputs and outputs.

  Args:
    inputs: dict of string to `Tensor`.
    outputs: dict of string to `Tensor`.

  Returns:
    A prediction-flavored signature_def.

  Raises:
    ValueError: If inputs or outputs is `None`.
  """
  if inputs is None or not inputs:
    raise ValueError('inputs cannot be None or empty for prediction.')
  if outputs is None:
    raise ValueError('outputs cannot be None or empty for prediction.')

  signature_inputs = {key: utils.build_tensor_info(tensor)
                      for key, tensor in inputs.items()}
  signature_outputs = {key: utils.build_tensor_info(tensor)
                       for key, tensor in outputs.items()}

  signature_def = build_signature_def(
      signature_inputs, signature_outputs,
      signature_constants.PREDICT_METHOD_NAME)

  return signature_def
