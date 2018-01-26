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
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import six

from tensorflow.contrib.eager.python import checkpointable
from tensorflow.contrib.eager.python import network as network_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.layers import core
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import adam
from tensorflow.python.training import training_util


class CheckpointableDenseLayer(core.Dense, checkpointable.Checkpointable):

  def __init__(self, *args, **kwargs):
    checkpointable.Checkpointable.__init__(self)
    core.Dense.__init__(self, *args, **kwargs)

  def add_variable(self, name, shape, **kwargs):
    # Calls both Checkpointable.add_variable and Layer.add_variable. Eventually
    # Layer.add_variable should inherit from Checkpointable and simply call
    # super and then do post-processing.
    return checkpointable.Checkpointable.add_variable(
        self,
        name=name,
        shape=shape,
        getter=functools.partial(core.Dense.add_variable, self),
        **kwargs)


# pylint: disable=not-callable
class CheckpointableNetwork(network_lib.Network, checkpointable.Checkpointable):

  def __init__(self):
    network_lib.Network.__init__(self)
    checkpointable.Checkpointable.__init__(self)

  def track_layer(self, layer, name=None):
    self.track_checkpointable(layer, name=name)
    return super(CheckpointableNetwork, self).track_layer(layer)


class CheckpointableAdam(adam.AdamOptimizer, checkpointable.Checkpointable):

  def __init__(self, *args, **kwargs):
    checkpointable.Checkpointable.__init__(self)
    adam.AdamOptimizer.__init__(self, *args, **kwargs)

  # NOTE: Copied from Optimizer with modifications to use add_variable
  # for non-slot variables. These contortions are necessary to maintain
  # checkpoint compatibility with variable.name based saving.
  # TODO(allenl): Make this cleaner.
  def _create_non_slot_variable(self, initial_value, name, colocate_with):
    """Add an extra variable, not associated with a slot."""
    if context.in_graph_mode():
      graph = colocate_with.graph
    else:
      graph = None

    key = (name, graph)
    v = self._non_slot_dict.get(key, None)
    if v is None:
      with ops.colocate_with(colocate_with):
        def _variable_getter(name, shape, dtype, initializer):
          del shape, dtype  # not used, but there for compatibility
          return variable_scope.variable(
              name=name, initial_value=initializer, trainable=False)

        initial_value = ops.convert_to_tensor(initial_value)
        v = self.add_variable(
            name=name,
            shape=initial_value.get_shape(),
            initializer=initial_value,
            getter=_variable_getter)

      self._non_slot_dict[key] = v

    return v

  # TODO(allenl): Override slot variable creation (_get_or_make_slot,
  # _get_or_make_slot_with_initializer, _zeros_slot) to allow deferred
  # loading. Likely no need to run this through add_variable, since gathering
  # slot variables is special cased anyway.


class MyNetwork(CheckpointableNetwork):
  """A concrete Network for testing."""

  def __init__(self):
    super(MyNetwork, self).__init__()
    self._named = self.track_layer(
        CheckpointableDenseLayer(1, use_bias=True), name="named_dense")
    self._unnamed = self.track_layer(
        CheckpointableDenseLayer(1, use_bias=False))

  def call(self, values):
    return self._unnamed(self._named(values))


class Root(checkpointable.Checkpointable):
  """A stand-in for a Trainer class."""

  def __init__(self, optimizer, network):
    super(Root, self).__init__()
    self.track_checkpointable(optimizer, name="optimizer")
    self.track_checkpointable(network, name="network")
    self._global_step = None

  @property
  def global_step(self):
    if self._global_step is None:
      # Get the default create_global_step utility to actually call
      # self.add_variable, by setting a custom getter.
      def _owned_variable_as_custom_getter(getter, *args, **kwargs):
        return self.add_variable(*args, getter=getter, **kwargs)

      with variable_scope.variable_scope(
          "", custom_getter=_owned_variable_as_custom_getter):
        self._global_step = training_util.create_global_step()
    return self._global_step


class CheckpointNamingTests(test.TestCase):

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testNamingWithOptimizer(self):
    input_value = constant_op.constant([[3.]])
    network = MyNetwork()
    # A nuisance Network using the same optimizer. Its slot variables should not
    # go in the checkpoint, since it is never depended on.
    other_network = MyNetwork()
    optimizer = CheckpointableAdam(0.001)
    root_checkpointable = Root(optimizer=optimizer, network=network)
    if context.in_eager_mode():
      optimizer.minimize(
          lambda: network(input_value),
          global_step=root_checkpointable.global_step)
      optimizer.minimize(
          lambda: other_network(input_value),
          global_step=root_checkpointable.global_step)
    else:
      train_op = optimizer.minimize(
          network(input_value), global_step=root_checkpointable.global_step)
      optimizer.minimize(
          other_network(input_value),
          global_step=root_checkpointable.global_step)
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(train_op)
    named_variables, serialized_graph = checkpointable._serialize_object_graph(
        root_checkpointable)
    expected_checkpoint_names = (
        # Created in the root node, so no prefix.
        "global_step",
        # No name provided to track_checkpointable(), so the position (1, after
        # the named track_checkpointable() which is 0) is used instead.
        "network/_1/kernel",
        # track_checkpointable() with a name provided, so that's used
        "network/named_dense/kernel",
        "network/named_dense/bias",
        # The optimizer creates two non-slot variables
        "optimizer/beta1_power",
        "optimizer/beta2_power",
        # Slot variables
        "network/_1/kernel/_OPTIMIZER_SLOT/optimizer/m",
        "network/_1/kernel/_OPTIMIZER_SLOT/optimizer/v",
        "network/named_dense/kernel/_OPTIMIZER_SLOT/optimizer/m",
        "network/named_dense/kernel/_OPTIMIZER_SLOT/optimizer/v",
        "network/named_dense/bias/_OPTIMIZER_SLOT/optimizer/m",
        "network/named_dense/bias/_OPTIMIZER_SLOT/optimizer/v",
    )
    six.assertCountEqual(self, expected_checkpoint_names,
                         named_variables.keys())
    # Check that we've mapped to the right variable objects (not exhaustive)
    self.assertEqual("global_step:0", named_variables["global_step"].name)
    self.assertEqual("my_network/checkpointable_dense_layer_1/kernel:0",
                     named_variables["network/_1/kernel"].name)
    self.assertEqual("my_network/checkpointable_dense_layer/kernel:0",
                     named_variables["network/named_dense/kernel"].name)
    self.assertEqual("beta1_power:0",
                     named_variables["optimizer/beta1_power"].name)
    self.assertEqual("beta2_power:0",
                     named_variables["optimizer/beta2_power"].name)
    # Spot check the generated protocol buffers.
    self.assertEqual(0, serialized_graph.nodes[0].children[0].local_uid)
    self.assertEqual("optimizer",
                     serialized_graph.nodes[0].children[0].local_name)
    optimizer_node = serialized_graph.nodes[serialized_graph.nodes[0].children[
        0].node_id]
    self.assertEqual("beta1_power", optimizer_node.variables[0].local_name)
    self.assertEqual("beta1_power", optimizer_node.variables[0].full_name)
    self.assertEqual(
        "kernel", optimizer_node.slot_variables[0].original_variable_local_name)
    original_variable_owner = serialized_graph.nodes[
        optimizer_node.slot_variables[0].original_variable_node_id]
    self.assertEqual("kernel", original_variable_owner.variables[0].local_name)
    self.assertEqual("m", optimizer_node.slot_variables[0].slot_name)
    # We strip off the :0 suffix, as variable.name-based saving does.
    self.assertEqual("my_network/checkpointable_dense_layer/kernel/Adam",
                     optimizer_node.slot_variables[0].full_name)
    self.assertEqual("my_network/checkpointable_dense_layer/kernel/Adam:0",
                     optimizer.get_slot(
                         var=named_variables["network/named_dense/kernel"],
                         name="m").name)

  def _get_checkpoint_name(self, name):
    root = checkpointable.Checkpointable()
    with variable_scope.variable_scope("get_checkpoint_name"):
      # Create the variable in a variable scope so that we get more relaxed
      # naming rules (variables outside a scope may not start with "_", "/" or
      # "-"). Since we don't use the scope part of the name, these cases are
      # somewhat annoying.
      root.add_variable(name=name, shape=[1, 2], dtype=dtypes.float64)
    named_variables, _ = checkpointable._serialize_object_graph(root)
    checkpoint_name, = named_variables.keys()
    with ops.name_scope("root/" + checkpoint_name):
      pass  # Make sure we can use this as an op name if we prefix it.
    return checkpoint_name

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testVariableNameEscaping(self):
    self.assertEqual(r"a_S__b_S__c", self._get_checkpoint_name(r"a/b/c"))
    self.assertEqual(r"", self._get_checkpoint_name(r""))
    self.assertEqual(r"_S__", self._get_checkpoint_name(r"/"))
    self.assertEqual(r"_S___S_._", self._get_checkpoint_name(r"/_S__"))

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testNumberedPath(self):
    root = checkpointable.Checkpointable()
    leaf = checkpointable.Checkpointable()
    root.track_checkpointable(leaf)
    leaf.add_variable(name="v", shape=[])
    named_variables, _ = checkpointable._serialize_object_graph(root)
    variable_name, = named_variables.keys()
    self.assertEqual(r"_0/v", variable_name)

  @test_util.run_in_graph_and_eager_modes()
  def testLocalNameValidation(self):
    root = checkpointable.Checkpointable()
    leaf = checkpointable.Checkpointable()
    with self.assertRaisesRegexp(ValueError, "invalid name"):
      # Leading underscores are reserved, which avoids conflicts with
      # un-named edges in paths and the optimizer slots identifier.
      root.track_checkpointable(leaf, name="_12")


if __name__ == "__main__":
  test.main()
