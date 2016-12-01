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
"""SavedModel utility functions.

Utility functions to assist with setup and construction of the SavedModel proto.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import dtypes

# TensorInfo helpers.


def build_tensor_info(tensor):
  """Utility function to build TensorInfo proto.

  Args:
    tensor: Tensor whose name, dtype and shape are used to build the TensorInfo.

  Returns:
    A TensorInfo protocol buffer constructed based on the supplied argument.
  """
  dtype_enum = dtypes.as_dtype(tensor.dtype).as_datatype_enum
  return meta_graph_pb2.TensorInfo(
      name=tensor.name,
      dtype=dtype_enum,
      tensor_shape=tensor.get_shape().as_proto())

# SignatureDef helpers.


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
