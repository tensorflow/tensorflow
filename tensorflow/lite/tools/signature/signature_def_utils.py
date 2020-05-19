# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utility functions related to SignatureDefs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.lite.tools.signature import _pywrap_signature_def_util_wrapper as signature_def_util


def set_signature_defs(tflite_model, signature_def_map):
  """Sets SignatureDefs to the Metadata of a TfLite flatbuffer buffer.

  Args:
    tflite_model: Binary TFLite model (bytes or bytes-like object) to which to
      add signature_def.
    signature_def_map: dict containing SignatureDefs to store in metadata.
  Returns:
    buffer: A TFLite model binary identical to model buffer with
      metadata field containing SignatureDef.

  Raises:
    ValueError:
      tflite_model buffer does not contain a valid TFLite model.
      signature_def_map is empty or does not contain a SignatureDef.
  """
  model = tflite_model
  if not isinstance(tflite_model, bytearray):
    model = bytearray(tflite_model)
  serialized_signature_def_map = {
      k: v.SerializeToString() for k, v in signature_def_map.items()}
  model_buffer = signature_def_util.SetSignatureDefMap(
      model, serialized_signature_def_map)
  return model_buffer


def get_signature_defs(tflite_model):
  """Get SignatureDef dict from the Metadata of a TfLite flatbuffer buffer.

  Args:
    tflite_model: TFLite model buffer to get the signature_def.

  Returns:
    dict containing serving names to SignatureDefs if exists, otherwise, empty
      dict.

  Raises:
    ValueError:
      tflite_model buffer does not contain a valid TFLite model.
    DecodeError:
      SignatureDef cannot be parsed from TfLite SignatureDef metadata.
  """
  model = tflite_model
  if not isinstance(tflite_model, bytearray):
    model = bytearray(tflite_model)
  serialized_signature_def_map = signature_def_util.GetSignatureDefMap(model)
  def _deserialize(serialized):
    signature_def = meta_graph_pb2.SignatureDef()
    signature_def.ParseFromString(serialized)
    return signature_def
  return {k: _deserialize(v) for k, v in serialized_signature_def_map.items()}


def clear_signature_defs(tflite_model):
  """Clears SignatureDefs from the Metadata of a TfLite flatbuffer buffer.

  Args:
    tflite_model: TFLite model buffer to remove signature_defs.

  Returns:
    buffer: A TFLite model binary identical to model buffer with
      no SignatureDef metadata.

  Raises:
    ValueError:
      tflite_model buffer does not contain a valid TFLite model.
  """
  model = tflite_model
  if not isinstance(tflite_model, bytearray):
    model = bytearray(tflite_model)
  return signature_def_util.ClearSignatureDefs(model)
