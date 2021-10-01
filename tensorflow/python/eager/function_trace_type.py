# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Utitiles for Cache Key generation based on Function Trace Type."""

from tensorflow.python import pywrap_tfe


def make_input_signature(inputs, include_tensor_ranks_only,
                         encode_variables_by_resource_id):
  """Generates an input signature representation.

  Args:
    inputs: The function inputs that need to be formed into a signature
    include_tensor_ranks_only: If Tensors should be considered by rank
    encode_variables_by_resource_id: If Variables should be considered by
      resource id

  Returns:
    An object representing the input signature
  """
  return pywrap_tfe.TFE_Py_EncodeArg(
      inputs, include_tensor_ranks_only, encode_variables_by_resource_id)
