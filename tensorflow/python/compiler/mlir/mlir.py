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

from tensorflow.python import pywrap_mlir
from tensorflow.python.util.tf_export import tf_export


@tf_export('mlir.experimental.convert_graph_def')
def convert_graph_def(graph_def,
                      pass_pipeline='tf-standard-pipeline',
                      show_debug_info=False):
  """Import a GraphDef and convert it to a textual MLIR module.

  This API is only intended for inspecting the internals of TensorFlow and the
  string returned is at the moment intended for debugging purposes.

  Args:
    graph_def: An object of type graph_pb2.GraphDef or a textual proto
      representation of a valid GraphDef.
    pass_pipeline: A textual description of an MLIR Pass Pipeline to run on the
      module, see MLIR documentation for the
      [textual pass pipeline syntax](https://mlir.llvm.org/docs/PassManagement/#textual-pass-pipeline-specification).
    show_debug_info: Whether to include locations in the emitted textual form.

  Returns:
    A textual representation of the MLIR module corresponding to the graphdef.

  Raises:
    InvalidArgumentError: if graph_def is invalid or cannot be converted to
      MLIR.

  """
  return pywrap_mlir.import_graphdef(graph_def, pass_pipeline, show_debug_info)


@tf_export('mlir.experimental.convert_function')
def convert_function(concrete_function,
                     pass_pipeline='tf-standard-pipeline',
                     show_debug_info=False):
  """Import a ConcreteFunction and convert it to a textual MLIR module.

  This API is only intended for inspecting the internals of TensorFlow and the
  string returned is at the moment intended for debugging purposes.

  A [tf.function](https://www.tensorflow.org/api_docs/python/tf/function) can be
  imported and converted from TensorFlow to TensorFlow MLIR with this API by
  extracting its ConcreteFunction (eagerly-executing wrapper around a
  [tf.Graph](https://www.tensorflow.org/api_docs/python/tf/Graph)).

  For example:
  >>> @tf.function
  ... def add(a, b):
  ...   return a + b

  >>> concrete_function = add.get_concrete_function(
  ...     tf.TensorSpec(None, tf.dtypes.float32),
  ...     tf.TensorSpec(None, tf.dtypes.float32))
  >>> tf.mlir.experimental.convert_function(concrete_function)
  '...module attributes {...} {...}...'

  Args:
    concrete_function: An object of type ConcreteFunction.
    pass_pipeline: A textual description of an MLIR Pass Pipeline to run on the
      module, see MLIR documentation for the
      [textual pass pipeline syntax](https://mlir.llvm.org/docs/PassManagement/#textual-pass-pipeline-specification).
    show_debug_info: Whether to include locations in the emitted textual form.

  Returns:
    A textual representation of the MLIR module corresponding to the
    ConcreteFunction.

  Raises:
    InvalidArgumentError: if concrete_function is invalid or cannot be converted
      to MLIR.

  """
  return pywrap_mlir.import_function(concrete_function, pass_pipeline,
                                     show_debug_info)
