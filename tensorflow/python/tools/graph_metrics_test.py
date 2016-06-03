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
"""Tests the graph metrics tool."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.tools import graph_metrics


class GraphMetricsTest(tf.test.TestCase):

  def testGraphMetrics(self):
    with tf.Graph().as_default():
      input_node = tf.placeholder(tf.float32, shape=[10, 20], name="input_node")
      weights_node = tf.constant(0.0,
                                 dtype=tf.float32,
                                 shape=[20, 5],
                                 name="weights_node")
      tf.matmul(input_node, weights_node, name="matmul_node")
      sess = tf.Session()
      graph_def = sess.graph.as_graph_def()
    statistic_types = ["weight_parameters", "flops"]
    total_stats, node_stats = graph_metrics.calculate_graph_metrics(
        graph_def, statistic_types, "input_node:0", None, 10)
    expected = {"weight_parameters": 100, "flops": 2000}
    for statistic_type in statistic_types:
      current_stats = node_stats[statistic_type]["matmul_node"]
      self.assertEqual(expected[statistic_type], current_stats.value)
    for statistic_type in statistic_types:
      current_stats = total_stats[statistic_type]
      self.assertEqual(expected[statistic_type], current_stats.value)


if __name__ == "__main__":
  tf.test.main()
