# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Defines a wrapper class for overridden python method definitions."""
import uuid

from tensorflow.compiler.mlir.quantization.tensorflow import exported_model_pb2
from tensorflow.compiler.mlir.quantization.tensorflow.python import pywrap_function_lib


class PyFunctionLibrary(pywrap_function_lib.PyFunctionLibrary):
  """Wrapper class for overridden python method definitions.

  This class contains python methods that overrides C++ virtual functions
  declared in `pywrap_function_lib.PyFunctionLibrary`.
  """

  def assign_ids_to_custom_aggregator_ops(
      self,
      exported_model_serialized: bytes,
  ) -> bytes:
    """Assigns UUIDs to each CustomAggregator op find in the graph def.

    Args:
      exported_model_serialized: Serialized `ExportedModel` instance.

    Returns:
      Serialized `ExportedModel` whose CustomAggregator ops are assigned UUIDs
      to their `id` attributes.
    """
    exported_model = exported_model_pb2.ExportedModel.FromString(
        exported_model_serialized
    )

    graph_def = exported_model.graph_def
    for function_def in graph_def.library.function:
      for node_def in function_def.node_def:
        if node_def.op == 'CustomAggregator':
          node_def.attr['id'].s = uuid.uuid4().hex.encode('ascii')

    return exported_model.SerializeToString()
