# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
r"""Removes unneeded nodes from a GraphDef file.

This script is designed to help streamline models, by taking the input and
output nodes that will be used by an application and figuring out the smallest
set of operations that are required to run for those arguments. The resulting
minimal graph is then saved out.

The advantages of running this script are:
 - You may be able to shrink the file size.
 - Operations that are unsupported on your platform but still present can be
   safely removed.
The resulting graph may not be as flexible as the original though, since any
input nodes that weren't explicitly mentioned may not be accessible any more.

An example of command-line usage is:
bazel build tensorflow/python/tools:strip_unused && \
bazel-bin/tensorflow/python/tools/strip_unused \
--input_graph=some_graph_def.pb \
--output_graph=/tmp/stripped_graph.pb \
--input_node_names=input0
--output_node_names=softmax

You can also look at strip_unused_test.py for an example of how to use it.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy

import tensorflow as tf

from google.protobuf import text_format
from tensorflow.python.framework import graph_util


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("input_graph", "",
                           """TensorFlow 'GraphDef' file to load.""")
tf.app.flags.DEFINE_boolean("input_binary", False,
                            """Whether the input files are in binary format.""")
tf.app.flags.DEFINE_string("output_graph", "",
                           """Output 'GraphDef' file name.""")
tf.app.flags.DEFINE_string("input_node_names", "",
                           """The name of the input nodes, comma separated.""")
tf.app.flags.DEFINE_string("output_node_names", "",
                           """The name of the output nodes, comma separated.""")
tf.app.flags.DEFINE_integer("placeholder_type_enum",
                            tf.float32.as_datatype_enum,
                            """The AttrValue enum to use for placeholders.""")


def strip_unused(input_graph, input_binary, output_graph, input_node_names,
                 output_node_names, placeholder_type_enum):
  """Removes unused nodes from a graph."""

  if not tf.gfile.Exists(input_graph):
    print("Input graph file '" + input_graph + "' does not exist!")
    return -1

  if not output_node_names:
    print("You need to supply the name of a node to --output_node_names.")
    return -1

  input_graph_def = tf.GraphDef()
  mode = "rb" if input_binary else "r"
  with tf.gfile.FastGFile(input_graph, mode) as f:
    if input_binary:
      input_graph_def.ParseFromString(f.read())
    else:
      text_format.Merge(f.read(), input_graph_def)

  # Here we replace the nodes we're going to override as inputs with
  # placeholders so that any unused nodes that are inputs to them are
  # automatically stripped out by extract_sub_graph().
  input_node_names_list = input_node_names.split(",")
  inputs_replaced_graph_def = tf.GraphDef()
  for node in input_graph_def.node:
    if node.name in input_node_names_list:
      placeholder_node = tf.NodeDef()
      placeholder_node.op = "Placeholder"
      placeholder_node.name = node.name
      placeholder_node.attr["dtype"].CopyFrom(tf.AttrValue(
          type=placeholder_type_enum))
      inputs_replaced_graph_def.node.extend([placeholder_node])
    else:
      inputs_replaced_graph_def.node.extend([copy.deepcopy(node)])

  output_graph_def = graph_util.extract_sub_graph(inputs_replaced_graph_def,
                                                  output_node_names.split(","))

  with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())
  print("%d ops in the final graph." % len(output_graph_def.node))


def main(unused_args):
  strip_unused(FLAGS.input_graph, FLAGS.input_binary, FLAGS.output_graph,
               FLAGS.input_node_names, FLAGS.output_node_names,
               FLAGS.placeholder_type_enum)

if __name__ == "__main__":
  tf.app.run()
