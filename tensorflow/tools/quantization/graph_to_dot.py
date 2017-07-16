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
"""Converts a GraphDef file into a DOT format suitable for visualization.

This script takes a GraphDef representing a network, and produces a DOT file
that can then be visualized by GraphViz tools like dot and xdot.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile

FLAGS = flags.FLAGS

flags.DEFINE_string("graph", "", """TensorFlow 'GraphDef' file to load.""")
flags.DEFINE_bool("input_binary", True,
                  """Whether the input files are in binary format.""")
flags.DEFINE_string("dot_output", "", """Where to write the DOT output.""")


def main(unused_args):
  if not gfile.Exists(FLAGS.graph):
    print("Input graph file '" + FLAGS.graph + "' does not exist!")
    return -1

  graph = graph_pb2.GraphDef()
  with open(FLAGS.graph, "r") as f:
    if FLAGS.input_binary:
      graph.ParseFromString(f.read())
    else:
      text_format.Merge(f.read(), graph)

  with open(FLAGS.dot_output, "wb") as f:
    print("digraph graphname {", file=f)
    for node in graph.node:
      output_name = node.name
      print("  \"" + output_name + "\" [label=\"" + node.op + "\"];", file=f)
      for input_full_name in node.input:
        parts = input_full_name.split(":")
        input_name = re.sub(r"^\^", "", parts[0])
        print("  \"" + input_name + "\" -> \"" + output_name + "\";", file=f)
    print("}", file=f)
  print("Created DOT file '" + FLAGS.dot_output + "'.")


if __name__ == "__main__":
  app.run()
