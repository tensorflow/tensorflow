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

from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.grappler import cost_analyzer
from tensorflow.python.platform import app


def main(_):
  with open(FLAGS.input) as input_file:
    metagraph = meta_graph_pb2.MetaGraphDef()
    metagraph.ParseFromString(input_file.read())

  report = cost_analyzer.GenerateCostReport(metagraph, FLAGS.per_node_report)
  print(report)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--input", type=str, default=None, help="Input .meta file path.")
  parser.add_argument(
      "--per_node_report",
      action="store_true",
      help="Generate per-node report. By default the report contains stats "
      "aggregated on a per op type basis, per_node_report adds results "
      "for each individual node to the report.")
  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
