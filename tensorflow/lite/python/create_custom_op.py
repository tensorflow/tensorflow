# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
r"""Replaces a subgraph of a TensorFlow GraphDef with a single node.

In conjunction with TOCO's --allow_custom_op this script allows selected
portions of a TensorFlow GraphDef to be executed by custom code.

Example:

bazel run tensorflow/lite/python:create_custom_op  -- \
  --input_graph=/tmp/input.pb \
  --output_graph=/tmp/output.pb \
  --inputs=concat,concat_1 \
  --outputs=detection_classes \
  --op_definition='op:"PostProcessing" attr{key:"num" value:{i:10}}'

The above will identify a subgraph starting at nodes 'concat' and 'concat_1',
and ending at 'detection_classes'. All nodes in between will be removed and
replaced by a new op called 'PostProcessing'.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import uuid as _uuid
from absl import app
from absl import flags
from google.protobuf import text_format
from tensorflow.contrib.framework.python.framework.graph_util import fuse_op
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python.platform import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string("input_graph", "", "Binary graphdef to load.")
flags.DEFINE_string("output_graph", "", "Resulting binary graphdef.")

flags.DEFINE_string("inputs", "",
                    "Comma-separated list of inputs to the subgraph.")
flags.DEFINE_string("outputs", "",
                    "Comma-separated list of outputs of the subgraph.")
flags.DEFINE_string("op_definition", "",
                    "A text NodeDef defining the contents of the custom op.")


def _read_graph_def(filename):
  if not gfile.Exists(filename):
    raise ValueError("Input graph file '" + filename + "' does not exist!")

  graph_def = graph_pb2.GraphDef()
  with gfile.GFile(filename, "rb") as f:
    graph_def.ParseFromString(f.read())
  return graph_def


def _write_graph_def(graph_def, filename):
  if not filename:
    raise ValueError("Output graph file not specified")

  with gfile.Open(filename, "wb") as f:
    f.write(graph_def.SerializeToString())


def _collapse_subgraph(graph_def, inputs, outputs, op_definition):
  """Substitute a custom op for the subgraph delimited by inputs and outputs."""
  name = _uuid.uuid1().hex
  # We need a default type, but it can be changed using 'op_definition'.
  default_type = types_pb2.DT_FLOAT
  new_graph = fuse_op(
      graph_def=graph_def,
      input_nodes=inputs,
      output_nodes=outputs,
      output_dtypes=[default_type for _ in outputs],
      output_quantized=False,
      op_name=name,
      op_type="CustomTfLiteOp")
  node_def = node_def_pb2.NodeDef()
  text_format.Parse(op_definition, node_def)
  for node in new_graph.node:
    if node.name == name:
      node.MergeFrom(node_def)
  return new_graph


def main(argv):
  del argv  # unused
  graph = _read_graph_def(filename=flags.FLAGS.input_graph)
  graph = _collapse_subgraph(
      graph_def=graph,
      inputs=flags.FLAGS.inputs.split(","),
      outputs=flags.FLAGS.outputs.split(","),
      op_definition=flags.FLAGS.op_definition)
  _write_graph_def(graph_def=graph, filename=flags.FLAGS.output_graph)


if __name__ == "__main__":
  app.run(main)
