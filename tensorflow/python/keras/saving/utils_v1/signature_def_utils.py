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

from tensorflow.python.keras.saving.utils_v1 import unexported_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import utils_impl as utils


# LINT.IfChange
def supervised_train_signature_def(
    inputs, loss, predictions=None, metrics=None):
  return _supervised_signature_def(
      unexported_constants.SUPERVISED_TRAIN_METHOD_NAME, inputs, loss=loss,
      predictions=predictions, metrics=metrics)


def supervised_eval_signature_def(
    inputs, loss, predictions=None, metrics=None):
  return _supervised_signature_def(
      unexported_constants.SUPERVISED_EVAL_METHOD_NAME, inputs, loss=loss,
      predictions=predictions, metrics=metrics)


def _supervised_signature_def(
    method_name, inputs, loss=None, predictions=None,
    metrics=None):
  """Creates a signature for training and eval data.

  This function produces signatures that describe the inputs and outputs
  of a supervised process, such as training or evaluation, that
  results in loss, metrics, and the like. Note that this function only requires
  inputs to be not None.

  Args:
    method_name: Method name of the SignatureDef as a string.
    inputs: dict of string to `Tensor`.
    loss: dict of string to `Tensor` representing computed loss.
    predictions: dict of string to `Tensor` representing the output predictions.
    metrics: dict of string to `Tensor` representing metric ops.

  Returns:
    A train- or eval-flavored signature_def.

  Raises:
    ValueError: If inputs or outputs is `None`.
  """
  if inputs is None or not inputs:
    raise ValueError('{} inputs cannot be None or empty.'.format(method_name))

  signature_inputs = {key: utils.build_tensor_info(tensor)
                      for key, tensor in inputs.items()}

  signature_outputs = {}
  for output_set in (loss, predictions, metrics):
    if output_set is not None:
      sig_out = {key: utils.build_tensor_info(tensor)
                 for key, tensor in output_set.items()}
      signature_outputs.update(sig_out)

  signature_def = signature_def_utils.build_signature_def(
      signature_inputs, signature_outputs, method_name)

  return signature_def
# LINT.ThenChange(//tensorflow/python/keras/saving/utils_v1/signature_def_utils.py)
