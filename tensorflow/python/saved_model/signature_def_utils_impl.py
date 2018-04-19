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


from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util.tf_export import tf_export


@tf_export('saved_model.signature_def_utils.build_signature_def')
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


@tf_export('saved_model.signature_def_utils.regression_signature_def')
def regression_signature_def(examples, predictions):
  """Creates regression signature from given examples and predictions.

  This function produces signatures intended for use with the TensorFlow Serving
  Regress API (tensorflow_serving/apis/prediction_service.proto), and so
  constrains the input and output types to those allowed by TensorFlow Serving.

  Args:
    examples: A string `Tensor`, expected to accept serialized tf.Examples.
    predictions: A float `Tensor`.

  Returns:
    A regression-flavored signature_def.

  Raises:
    ValueError: If examples is `None`.
  """
  if examples is None:
    raise ValueError('Regression examples cannot be None.')
  if not isinstance(examples, ops.Tensor):
    raise ValueError('Regression examples must be a string Tensor.')
  if predictions is None:
    raise ValueError('Regression predictions cannot be None.')

  input_tensor_info = utils.build_tensor_info(examples)
  if input_tensor_info.dtype != types_pb2.DT_STRING:
    raise ValueError('Regression examples must be a string Tensor.')
  signature_inputs = {signature_constants.REGRESS_INPUTS: input_tensor_info}

  output_tensor_info = utils.build_tensor_info(predictions)
  if output_tensor_info.dtype != types_pb2.DT_FLOAT:
    raise ValueError('Regression output must be a float Tensor.')
  signature_outputs = {signature_constants.REGRESS_OUTPUTS: output_tensor_info}

  signature_def = build_signature_def(
      signature_inputs, signature_outputs,
      signature_constants.REGRESS_METHOD_NAME)

  return signature_def


@tf_export('saved_model.signature_def_utils.classification_signature_def')
def classification_signature_def(examples, classes, scores):
  """Creates classification signature from given examples and predictions.

  This function produces signatures intended for use with the TensorFlow Serving
  Classify API (tensorflow_serving/apis/prediction_service.proto), and so
  constrains the input and output types to those allowed by TensorFlow Serving.

  Args:
    examples: A string `Tensor`, expected to accept serialized tf.Examples.
    classes: A string `Tensor`.  Note that the ClassificationResponse message
      requires that class labels are strings, not integers or anything else.
    scores: a float `Tensor`.

  Returns:
    A classification-flavored signature_def.

  Raises:
    ValueError: If examples is `None`.
  """
  if examples is None:
    raise ValueError('Classification examples cannot be None.')
  if not isinstance(examples, ops.Tensor):
    raise ValueError('Classification examples must be a string Tensor.')
  if classes is None and scores is None:
    raise ValueError('Classification classes and scores cannot both be None.')

  input_tensor_info = utils.build_tensor_info(examples)
  if input_tensor_info.dtype != types_pb2.DT_STRING:
    raise ValueError('Classification examples must be a string Tensor.')
  signature_inputs = {signature_constants.CLASSIFY_INPUTS: input_tensor_info}

  signature_outputs = {}
  if classes is not None:
    classes_tensor_info = utils.build_tensor_info(classes)
    if classes_tensor_info.dtype != types_pb2.DT_STRING:
      raise ValueError('Classification classes must be a string Tensor.')
    signature_outputs[signature_constants.CLASSIFY_OUTPUT_CLASSES] = (
        classes_tensor_info)
  if scores is not None:
    scores_tensor_info = utils.build_tensor_info(scores)
    if scores_tensor_info.dtype != types_pb2.DT_FLOAT:
      raise ValueError('Classification scores must be a float Tensor.')
    signature_outputs[signature_constants.CLASSIFY_OUTPUT_SCORES] = (
        scores_tensor_info)

  signature_def = build_signature_def(
      signature_inputs, signature_outputs,
      signature_constants.CLASSIFY_METHOD_NAME)

  return signature_def


@tf_export('saved_model.signature_def_utils.predict_signature_def')
def predict_signature_def(inputs, outputs):
  """Creates prediction signature from given inputs and outputs.

  This function produces signatures intended for use with the TensorFlow Serving
  Predict API (tensorflow_serving/apis/prediction_service.proto). This API
  imposes no constraints on the input and output types.

  Args:
    inputs: dict of string to `Tensor`.
    outputs: dict of string to `Tensor`.

  Returns:
    A prediction-flavored signature_def.

  Raises:
    ValueError: If inputs or outputs is `None`.
  """
  if inputs is None or not inputs:
    raise ValueError('Prediction inputs cannot be None or empty.')
  if outputs is None or not outputs:
    raise ValueError('Prediction outputs cannot be None or empty.')

  signature_inputs = {key: utils.build_tensor_info(tensor)
                      for key, tensor in inputs.items()}
  signature_outputs = {key: utils.build_tensor_info(tensor)
                       for key, tensor in outputs.items()}

  signature_def = build_signature_def(
      signature_inputs, signature_outputs,
      signature_constants.PREDICT_METHOD_NAME)

  return signature_def


@tf_export('saved_model.signature_def_utils.is_valid_signature')
def is_valid_signature(signature_def):
  """Determine whether a SignatureDef can be served by TensorFlow Serving."""
  if signature_def is None:
    return False
  return (_is_valid_classification_signature(signature_def) or
          _is_valid_regression_signature(signature_def) or
          _is_valid_predict_signature(signature_def))


def _is_valid_predict_signature(signature_def):
  """Determine whether the argument is a servable 'predict' SignatureDef."""
  if signature_def.method_name != signature_constants.PREDICT_METHOD_NAME:
    return False
  if not signature_def.inputs.keys():
    return False
  if not signature_def.outputs.keys():
    return False
  return True


def _is_valid_regression_signature(signature_def):
  """Determine whether the argument is a servable 'regress' SignatureDef."""
  if signature_def.method_name != signature_constants.REGRESS_METHOD_NAME:
    return False

  if (set(signature_def.inputs.keys())
      != set([signature_constants.REGRESS_INPUTS])):
    return False
  if (signature_def.inputs[signature_constants.REGRESS_INPUTS].dtype !=
      types_pb2.DT_STRING):
    return False

  if (set(signature_def.outputs.keys())
      != set([signature_constants.REGRESS_OUTPUTS])):
    return False
  if (signature_def.outputs[signature_constants.REGRESS_OUTPUTS].dtype !=
      types_pb2.DT_FLOAT):
    return False

  return True


def _is_valid_classification_signature(signature_def):
  """Determine whether the argument is a servable 'classify' SignatureDef."""
  if signature_def.method_name != signature_constants.CLASSIFY_METHOD_NAME:
    return False

  if (set(signature_def.inputs.keys())
      != set([signature_constants.CLASSIFY_INPUTS])):
    return False
  if (signature_def.inputs[signature_constants.CLASSIFY_INPUTS].dtype !=
      types_pb2.DT_STRING):
    return False

  allowed_outputs = set([signature_constants.CLASSIFY_OUTPUT_CLASSES,
                         signature_constants.CLASSIFY_OUTPUT_SCORES])

  if not signature_def.outputs.keys():
    return False
  if set(signature_def.outputs.keys()) - allowed_outputs:
    return False
  if (signature_constants.CLASSIFY_OUTPUT_CLASSES in signature_def.outputs
      and
      signature_def.outputs[signature_constants.CLASSIFY_OUTPUT_CLASSES].dtype
      != types_pb2.DT_STRING):
    return False
  if (signature_constants.CLASSIFY_OUTPUT_SCORES in signature_def.outputs
      and
      signature_def.outputs[signature_constants.CLASSIFY_OUTPUT_SCORES].dtype !=
      types_pb2.DT_FLOAT):
    return False

  return True


def _get_shapes_from_tensor_info_dict(tensor_info_dict):
  """Returns a map of keys to TensorShape objects.

  Args:
    tensor_info_dict: map with TensorInfo proto as values.

  Returns:
    Map with corresponding TensorShape objects as values.
  """
  return {
      key: tensor_shape.TensorShape(tensor_info.tensor_shape)
      for key, tensor_info in tensor_info_dict.items()
  }


def _get_types_from_tensor_info_dict(tensor_info_dict):
  """Returns a map of keys to DType objects.

  Args:
    tensor_info_dict: map with TensorInfo proto as values.

  Returns:
    Map with corresponding DType objects as values.
  """
  return {
      key: dtypes.DType(tensor_info.dtype)
      for key, tensor_info in tensor_info_dict.items()
  }


def get_signature_def_input_shapes(signature):
  """Returns map of parameter names to their shapes.

  Args:
    signature: SignatureDef proto.

  Returns:
    Map from string to TensorShape objects.
  """
  return _get_shapes_from_tensor_info_dict(signature.inputs)


def get_signature_def_input_types(signature):
  """Returns map of output names to their types.

  Args:
    signature: SignatureDef proto.

  Returns:
    Map from string to DType objects.
  """
  return _get_types_from_tensor_info_dict(signature.inputs)


def get_signature_def_output_shapes(signature):
  """Returns map of output names to their shapes.

  Args:
    signature: SignatureDef proto.

  Returns:
    Map from string to TensorShape objects.
  """
  return _get_shapes_from_tensor_info_dict(signature.outputs)


def get_signature_def_output_types(signature):
  """Returns map of output names to their types.

  Args:
    signature: SignatureDef proto.

  Returns:
    Map from string to DType objects.
  """
  return _get_types_from_tensor_info_dict(signature.outputs)
