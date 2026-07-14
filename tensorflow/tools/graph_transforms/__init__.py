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
"""Exposes the Python wrapper for graph transforms."""
# pylint: disable=unused-import,wildcard-import, line-too-long
from tensorflow.core.framework import graph_pb2
from tensorflow.python.util import compat
from tensorflow.python.util._pywrap_transform_graph import TransformGraphWithStringInputs


def TransformGraph(input_graph_def, inputs, outputs, transforms):
  """Python wrapper for the Graph Transform Tool.

  Gives access to all graph transforms available through the command line tool.
  See documentation at https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md
  for full details of the options available.

  Args:
    input_graph_def: GraphDef object containing a model to be transformed.
    inputs: List of node names for the model inputs.
    outputs: List of node names for the model outputs.
    transforms: List of strings containing transform names and parameters.

  Returns:
    New GraphDef with transforms applied.
  """

  input_graph_def_string = input_graph_def.SerializeToString()
  inputs_string = compat.as_bytes(",".join(inputs))
  outputs_string = compat.as_bytes(",".join(outputs))
  transforms_string = compat.as_bytes(" ".join(transforms))
  output_graph_def_string = TransformGraphWithStringInputs(
      input_graph_def_string, inputs_string, outputs_string, transforms_string)
  output_graph_def = graph_pb2.GraphDef()
  output_graph_def.ParseFromString(output_graph_def_string)
  return output_graph_def
