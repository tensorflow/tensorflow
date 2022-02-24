# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Custom Aggregator op is for collecting numeric metrics from the given input."""

from tensorflow.compiler.mlir.quantization.tensorflow.calibrator import custom_aggregator_op_wrapper
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

_custom_aggregator_op = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_custom_aggregator_op.so'))


def custom_aggregator(input_tensor, tensor_id: str):
  """Creates custom aggregator op that collects numeric metrics from the tensor.

  Args:
    input_tensor: Tensor to be scanned through this operator. This tensor will
      be bypassed to the output tensor of this operator.
    tensor_id: String, the identity of the tensor to be scanned.

  Returns:
    A `Tensor` of the same value as `input_tensor`.

  Raises:
    ValueError: If the given type of `input_tensor` is not float32.
  """
  if input_tensor.dtype != dtypes.float32:
    raise ValueError('Custom aggregator op only accept float32 values.')
  return custom_aggregator_op_wrapper.custom_aggregator(input_tensor, tensor_id)
