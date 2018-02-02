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
# =============================================================================
"""A tool for cost analysis."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.grappler import cost_analyzer
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.platform import app
from tensorflow.python.platform import gfile
from tensorflow.python.training import saver


def main(_):
  if FLAGS.metagraphdef:
    with gfile.GFile(FLAGS.metagraphdef) as meta_file:
      metagraph = meta_graph_pb2.MetaGraphDef()
      metagraph.ParseFromString(meta_file.read())
  else:
    with gfile.GFile(FLAGS.graphdef) as graph_file:
      graph_def = graph_pb2.GraphDef()
      graph_def.ParseFromString(graph_file.read())
      importer.import_graph_def(graph_def, name="")
      graph = ops.get_default_graph()
      fetch = graph.get_operation_by_name(FLAGS.fetch)
      graph.add_to_collection("train_op", fetch)
      metagraph = saver.export_meta_graph(
          graph_def=graph.as_graph_def(), graph=graph)

  if FLAGS.rewriter_config is not None:
    rewriter_config = rewriter_config_pb2.RewriterConfig()
    text_format.Merge(FLAGS.rewriter_config, rewriter_config)
    optimized_graph = tf_optimizer.OptimizeGraph(rewriter_config, metagraph)
    metagraph.graph_def.CopyFrom(optimized_graph)

  report = cost_analyzer.GenerateCostReport(metagraph, FLAGS.per_node_report)
  print(report)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--metagraphdef",
      type=str,
      default=None,
      help="Input .meta MetaGraphDef file path.")
  parser.add_argument(
      "--graphdef",
      type=str,
      default=None,
      help="Input .pb GraphDef file path.")
  # Consider making flag fetch work together with flag metagraphdef. As some
  # MetaGraphDef files don't have collection train_op.
  parser.add_argument(
      "--fetch",
      type=str,
      default=None,
      help=
      "The name of the fetch node. This flag is ignored if flag "
      "metagraphdef is used."
  )
  parser.add_argument(
      "--rewriter_config",
      type=str,
      default=None,
      help="Configuration for the grappler optimizers, described as a "
      "RewriterConfig protocol buffer. Usage example 1: "
      "--rewriter_config='optimize_tensor_layout: true "
      "disable_model_pruning: true'. Usage example 2: "
      "--rewriter_config='optimizers: \"constfold\" optimizers: \"layout\"'")
  parser.add_argument(
      "--per_node_report",
      action="store_true",
      help="Generate per-node report. By default the report contains stats "
      "aggregated on a per op type basis, per_node_report adds results "
      "for each individual node to the report.")
  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
