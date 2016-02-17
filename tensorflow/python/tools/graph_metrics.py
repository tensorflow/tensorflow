# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Gives estimates of computation and parameter sizes for a GraphDef.

This script takes a GraphDef representing a network, and produces rough
estimates of the number of floating-point operations needed to implement it and
how many parameters are stored. You need to pass in the input size, and the
results are only approximate, since it only calculates them for a subset of
common operations.

If you have downloaded the Inception graph for the label_image example, an
example of using this script would be:

bazel-bin/third_party/tensorflow/python/tools/graph_metrics \
--graph tensorflow_inception_graph.pb                       \
--statistics=weight_parameters,flops

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import locale

import tensorflow as tf

from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import ops


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("graph", "", """TensorFlow 'GraphDef' file to load.""")
tf.flags.DEFINE_bool("input_binary", True,
                     """Whether the input files are in binary format.""")
tf.flags.DEFINE_string("input_layer", "Mul:0",
                       """The name of the input node.""")
tf.flags.DEFINE_integer("batch_size", 1,
                        """The batch size to use for the calculations.""")
tf.flags.DEFINE_string("statistics", "weight_parameters,flops",
                       """Which statistic types to examine.""")


def main(unused_args):
  if not tf.gfile.Exists(FLAGS.graph):
    print("Input graph file '" + FLAGS.graph + "' does not exist!")
    return -1

  graph_def = graph_pb2.GraphDef()
  with open(FLAGS.graph, "rb") as f:
    if FLAGS.input_binary:
      graph_def.ParseFromString(f.read())
    else:
      text_format.Merge(f.read(), graph_def)
  _ = tf.import_graph_def(graph_def, name="")

  statistic_types = FLAGS.statistics.split(",")
  total_stats = {}
  for statistic_type in statistic_types:
    total_stats[statistic_type] = ops.OpStats(statistic_type)
  with tf.Session() as sess:
    input_tensor = sess.graph.get_tensor_by_name(FLAGS.input_layer)
    input_shape = input_tensor.get_shape()
    input_shape = [FLAGS.batch_size, input_shape[1], input_shape[2],
                   input_shape[3]]
    input_tensor.set_shape(input_shape)
    for node in graph_def.node:
      for statistic_type in statistic_types:
        node_stats = ops.get_stats_for_node_def(sess.graph, node,
                                                statistic_type)
        total_stats[statistic_type] += node_stats
  # Make sure we get pretty-printed numbers with separators.
  locale.setlocale(locale.LC_ALL, "")
  for statistic_type in statistic_types:
    value = total_stats[statistic_type].value
    if value is None:
      friendly_value = "None"
    else:
      friendly_value = locale.format("%d", value, grouping=True)
    print("%s=%s" % (statistic_type, friendly_value))


if __name__ == "__main__":
  tf.app.run()
