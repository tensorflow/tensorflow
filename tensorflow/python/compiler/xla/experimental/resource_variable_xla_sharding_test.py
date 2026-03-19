# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.python.compiler.xla.experimental import xla_sharding
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.tpu import device_assignment
from tensorflow.python.tpu import tpu
from tensorflow.python.training import adagrad


# Gets all the nodes of `op` in graph that have `input_node_name` as one of the
# inputs
def _get_op_nodes_with_input(input_node_name, op, graph):
  nodes_with_input = []
  for node in graph.node:
    nodes_with_input += [
        node
        for input in node.input
        if input == input_node_name and node.op == op
    ]
  return nodes_with_input


# Gets XlaSharding ops connected to ReadVariableOp for the given variable_name
def _get_xla_sharding_nodes_for_variable(variable_name, graph):
  read_variable_op_nodes = _get_op_nodes_with_input(
      variable_name, 'ReadVariableOp', graph
  )
  xla_sharding_op_nodes = []
  for read_variable_op_node in read_variable_op_nodes:
    xla_sharding_op_nodes += _get_op_nodes_with_input(
        read_variable_op_node.name, 'XlaSharding', graph
    )
  return xla_sharding_op_nodes


def _get_xla_sharding_proto_from_node(node):
  sharding_proto = xla_sharding.xla_data_pb2.OpSharding()
  sharding_proto.ParseFromString(node.attr['sharding'].s)
  return sharding_proto


class ResourceVariableXlaShardingTest(test.TestCase):

  def setUp(self) -> None:
    super().setUp()

    context.enable_xla_sharding_for_resource_variables()
    self.topology = tpu_cluster_resolver.initialize_tpu_system()
    if len(config.list_logical_devices('TPU')) != 8:
      self.skipTest('All tests require 8 TPUs.')

    self.da = device_assignment.DeviceAssignment.build(
        self.topology, computation_shape=[2, 2, 1, 2], num_replicas=1
    )

  def test_xla_sharding_ops_created_for_optimizer_slot_variables(self):
    w = variables.Variable(
        initial_value=math_ops.range(8, dtype=dtypes.float32),
        name='w',
    )
    self.assertIsInstance(w, resource_variable_ops.BaseResourceVariable)
    w = xla_sharding.split(
        w,
        split_dimension=0,
        num_devices=8,
    )
    sharding_proto = xla_sharding.xla_data_pb2.OpSharding()
    sharding_proto.ParseFromString(xla_sharding.get_tensor_sharding(w))
    opt = adagrad.AdagradOptimizer(1.0)

    @def_function.function
    def computation(x):
      def tpu_fn(x):
        y = math_ops.add(w, x)
        loss = math_ops.reduce_sum(y)
        opt.minimize(loss, None, [w])
        return loss

      output = tpu.replicate(tpu_fn, [[x]], device_assignment=self.da)
      return output

    inputs = array_ops.reshape(math_ops.range(16, dtype=dtypes.float32), (2, 8))
    result = computation(inputs)
    self.assertSequenceEqual([[176.0]], self.evaluate(result))
    graph = computation.get_concrete_function(inputs).graph.as_graph_def()

    update_op_nodes = [
        node for node in graph.node if node.op == 'ResourceApplyAdagrad'
    ]
    self.assertLen(update_op_nodes, 1)
    update_op_node = update_op_nodes[0]

    var_input_name = update_op_node.input[0]
    var_sharding_nodes = _get_xla_sharding_nodes_for_variable(
        var_input_name, graph
    )
    self.assertLen(var_sharding_nodes, 1)
    self.assertProtoEquals(
        _get_xla_sharding_proto_from_node(var_sharding_nodes[0]), sharding_proto
    )

    slot_var_input_name = update_op_node.input[1]
    slot_var_sharding_nodes = _get_xla_sharding_nodes_for_variable(
        slot_var_input_name, graph
    )
    self.assertLen(slot_var_sharding_nodes, 1)
    self.assertProtoEquals(
        _get_xla_sharding_proto_from_node(slot_var_sharding_nodes[0]),
        sharding_proto,
    )

  def test_disabling_xla_sharding_ops_temporarily(self):
    w = variables.Variable(
        initial_value=math_ops.range(8, dtype=dtypes.float32),
        name='w',
    )
    self.assertIsInstance(w, resource_variable_ops.BaseResourceVariable)

    context.enable_xla_sharding_for_resource_variables()
    with context.temporarily_disable_xla_sharding_for_resource_variables():
      with self.assertRaisesRegex(
          AttributeError,
          '.*Tensor.op is undefined when eager execution is enabled.*',
      ):
        xla_sharding.split(
            w,
            split_dimension=0,
            num_devices=8,
        )

    # xla_sharding_for_resource_variables is enabled again. Following line
    # doesn't throw an error.
    xla_sharding.split(
        w,
        split_dimension=0,
        num_devices=8,
    )

    context.disable_xla_sharding_for_resource_variables()
    with context.temporarily_disable_xla_sharding_for_resource_variables():
      with self.assertRaisesRegex(
          AttributeError,
          '.*Tensor.op is undefined when eager execution is enabled.*',
      ):
        xla_sharding.split(
            w,
            split_dimension=0,
            num_devices=8,
        )

    # xla_sharding_for_resource_variables stays disabled.
    with self.assertRaisesRegex(
        AttributeError,
        '.*Tensor.op is undefined when eager execution is enabled.*',
    ):
      xla_sharding.split(
          w,
          split_dimension=0,
          num_devices=8,
      )


if __name__ == '__main__':
  test.main()
