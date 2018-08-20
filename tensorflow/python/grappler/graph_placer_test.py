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
"""Tests the graph placer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.core.protobuf import device_properties_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.grappler import cluster
from tensorflow.python.grappler import graph_placer
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test


class GraphPlacerTest(test.TestCase):

  @staticmethod
  def _buildMnist(batch_size=128,
                  input_size=256,
                  num_classes=1024,
                  num_layers=10,
                  hidden_size=256,
                  name='mnist'):
    g = tf_ops.get_default_graph()
    with g.as_default():
      ops = {}
      x = random_ops.random_uniform(
          [batch_size, input_size], -0.1, 0.1, dtype=dtypes.float32)
      for layer_id in range(num_layers):
        with variable_scope.variable_scope('layer_{}'.format(layer_id)):
          a = input_size if layer_id == 0 else hidden_size
          b = hidden_size if layer_id < num_layers - 1 else num_classes
          w = variable_scope.get_variable('w', [a, b])
          x = math_ops.matmul(x, w)
          x = nn_ops.relu(x)
      ops['y_preds'] = math_ops.argmax(x, axis=1)

    train_op = g.get_collection_ref(tf_ops.GraphKeys.TRAIN_OP)
    train_op.append(ops['y_preds'])
    return g

  @staticmethod
  def _buildCluster(num_cpus=1, num_gpus=1):
    devices = []
    if num_gpus > 0:
      device_properties = device_properties_pb2.DeviceProperties(
          type='GPU',
          vendor='NVidia',
          model='GeForce GTX TITAN X',
          frequency=1076,
          num_cores=24,
          environment={'architecture': '5.2',
                       'cuda': '8000',
                       'cudnn': '6021'},
          num_registers=65536,
          l1_cache_size=24576,
          l2_cache_size=3145728,
          shared_memory_size_per_multiprocessor=98304,
          memory_size=12783648768,
          bandwidth=336480000)
      for i in range(num_gpus):
        devices.append(
            device_properties_pb2.NamedDevice(
                properties=device_properties, name='/GPU:' + str(i)))

    assert num_cpus > 0
    device_properties = device_properties_pb2.DeviceProperties(
        type='CPU',
        frequency=2000,
        num_cores=4,
        l1_cache_size=32768,
        l2_cache_size=262144,
        l3_cache_size=12582912)
    for i in range(num_cpus):
      devices.append(
          device_properties_pb2.NamedDevice(
              properties=device_properties, name='/CPU:' + str(i)))

    return cluster.Cluster(devices=devices)

  def testBasic(self):
    """Place a trivial graph."""
    a = constant_op.constant(10, name='a')
    b = constant_op.constant(20, name='b')
    c = math_ops.add_n([a, b], name='c')
    d = math_ops.add_n([b, c], name='d')
    train_op = tf_ops.get_collection_ref(tf_ops.GraphKeys.TRAIN_OP)
    train_op.append(d)
    mg = meta_graph.create_meta_graph_def(graph=tf_ops.get_default_graph())

    gcluster = cluster.Cluster()
    placed_mg = graph_placer.PlaceGraph(mg, allotted_time=15, cluster=gcluster)

    self.assertEqual(4, len(placed_mg.graph_def.node))
    self.assertItemsEqual([node.name for node in placed_mg.graph_def.node],
                          [node.name for node in mg.graph_def.node])

    available_devices = [device.name for device in gcluster.ListDevices()]
    for node in placed_mg.graph_def.node:
      # The constant nodes are optimized away before the placer is run, and
      # therefore won't be placed.
      self.assertTrue(not node.device or node.device in available_devices)

  def testMNIST(self):
    graph = GraphPlacerTest._buildMnist()
    mg = meta_graph.create_meta_graph_def(graph=graph)
    gcluster = GraphPlacerTest._buildCluster(num_gpus=1)
    # Spend 15 seconds trying to optimize the placement of the model. This
    # should give us enough time to exercise the code, but not enough to find
    # a good placement, so we'll just check for legality.
    placed_mg = graph_placer.PlaceGraph(mg, allotted_time=15, cluster=gcluster)
    self.assertEqual(len(placed_mg.graph_def.node), len(mg.graph_def.node))
    self.assertItemsEqual([node.name for node in placed_mg.graph_def.node],
                          [node.name for node in mg.graph_def.node])
    available_devices = [device.name for device in gcluster.ListDevices()]
    for node in placed_mg.graph_def.node:
      self.assertTrue(not node.device or node.device in available_devices)


if __name__ == '__main__':
  test.main()
