# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""mlir is an experimental library that provides support APIs for MLIR."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import pywrap_tensorflow as import_graphdef
from tensorflow.python.util.tf_export import tf_export


@tf_export('mlir.experimental.convert_graph_def')
def convert_graph_def(graph_def, pass_pipeline='tf-standard-pipeline'):
  """Import a GraphDef and convert it to a textual MLIR module.

  Args:
    graph_def: An object of type graph_pb2.GraphDef or a textual proto
      representation of a valid GraphDef.
    pass_pipeline: A textual description of an MLIR Pass Pipeline to run on the
      module, see MLIR documentation for the
      [textual pass pipeline syntax](https://github.com/tensorflow/mlir/blob/master/g3doc/WritingAPass.md#textual-pass-pipeline-specification).

  Returns:
    A textual representation of the MLIR module corresponding to the graphdef.
    Raises a RuntimeError on error.

  """
  return import_graphdef.import_graphdef(graph_def, pass_pipeline)
