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
import os

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
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import adam
from tensorflow.python.training import saver as core_saver
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
        # No name provided to track_checkpointable(), so the position is used
        # instead (one-based).
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
    # Variable ordering is arbitrary but deterministic (alphabetized)
    self.assertEqual(
        "bias", optimizer_node.slot_variables[0].original_variable_local_name)
    original_variable_owner = serialized_graph.nodes[
        optimizer_node.slot_variables[0].original_variable_node_id]
    self.assertEqual("network/named_dense/bias",
                     original_variable_owner.variables[0].checkpoint_key)
    self.assertEqual("bias", original_variable_owner.variables[0].local_name)
    self.assertEqual("m", optimizer_node.slot_variables[0].slot_name)
    self.assertEqual("network/named_dense/bias/_OPTIMIZER_SLOT/optimizer/m",
                     optimizer_node.slot_variables[0].checkpoint_key)
    # We strip off the :0 suffix, as variable.name-based saving does.
    self.assertEqual("my_network/checkpointable_dense_layer/bias/Adam",
                     optimizer_node.slot_variables[0].full_name)
    self.assertEqual("my_network/checkpointable_dense_layer/bias/Adam:0",
                     optimizer.get_slot(
                         var=named_variables["network/named_dense/bias"],
                         name="m").name)

  @test_util.run_in_graph_and_eager_modes()
  def testSaveRestore(self):
    network = MyNetwork()
    optimizer = CheckpointableAdam(0.001)
    root_checkpointable = Root(optimizer=optimizer, network=network)
    input_value = constant_op.constant([[3.]])
    if context.in_eager_mode():
      optimizer.minimize(
          lambda: network(input_value),
          global_step=root_checkpointable.global_step)
    else:
      train_op = optimizer.minimize(
          network(input_value), global_step=root_checkpointable.global_step)
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(train_op)
    prefix = os.path.join(self.get_temp_dir(), "ckpt")
    self.evaluate(state_ops.assign(network._named.variables[1], [42.]))
    m_bias_slot = optimizer.get_slot(network._named.variables[1], "m")
    self.evaluate(state_ops.assign(m_bias_slot, [1.5]))
    serialized_graph, save_path = checkpointable.save(
        file_prefix=prefix,
        root_checkpointable=root_checkpointable,
        global_step=root_checkpointable.global_step)
    self.evaluate(state_ops.assign(network._named.variables[1], [43.]))
    self.evaluate(state_ops.assign(root_checkpointable.global_step, 3))
    optimizer_variables = self.evaluate(optimizer.variables())
    self.evaluate(state_ops.assign(m_bias_slot, [-2.]))
    # Immediate restoration
    checkpointable.restore(
        save_path=save_path,
        root_checkpointable=root_checkpointable,
        object_graph_proto=serialized_graph)
    self.assertAllEqual([42.], self.evaluate(network._named.variables[1]))
    self.assertAllEqual(1, self.evaluate(root_checkpointable.global_step))
    self.assertAllEqual([1.5], self.evaluate(m_bias_slot))
    with ops.Graph().as_default():
      on_create_network = MyNetwork()
      on_create_optimizer = CheckpointableAdam(0.001)
      on_create_root = Root(
          optimizer=on_create_optimizer, network=on_create_network)
      with self.test_session(graph=ops.get_default_graph()):
        # Deferred restoration
        checkpointable.restore(
            save_path=save_path,
            root_checkpointable=on_create_root,
            object_graph_proto=serialized_graph)
        on_create_network(constant_op.constant([[3.]]))  # create variables
        self.assertAllEqual(1, self.evaluate(on_create_root.global_step))
        self.assertAllEqual([42.],
                            self.evaluate(
                                on_create_network._named.variables[1]))
        on_create_m_bias_slot = on_create_optimizer.get_slot(
            on_create_network._named.variables[1], "m")
        # Optimizer slot variables are created when the original variable is
        # restored.
        self.assertAllEqual([1.5], self.evaluate(on_create_m_bias_slot))
        # beta1_power and beta2_power haven't been created yet, but everything
        # else matches.
        self.assertAllEqual(optimizer_variables[2:],
                            self.evaluate(on_create_optimizer.variables()))
        on_create_optimizer._create_slots(
            [resource_variable_ops.ResourceVariable([1.])])
        beta1_power, beta2_power = on_create_optimizer._get_beta_accumulators()
        self.assertAllEqual(optimizer_variables[0], self.evaluate(beta1_power))
        self.assertAllEqual(optimizer_variables[1], self.evaluate(beta2_power))

  def testDeferredRestorationUsageEager(self):
    """An idiomatic eager execution example."""
    num_training_steps = 10
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    latest_object_graph = None  # Will be saved with the checkpoint eventually.
    for training_continuation in range(3):
      with ops.Graph().as_default():
        network = MyNetwork()
        optimizer = CheckpointableAdam(0.001)
        root = Root(optimizer=optimizer, network=network)
        checkpointable.restore(
            save_path=core_saver.latest_checkpoint(checkpoint_directory),
            root_checkpointable=root,
            object_graph_proto=latest_object_graph)
        for _ in range(num_training_steps):
          # TODO(allenl): Use a Dataset and serialize/checkpoint it.
          input_value = constant_op.constant([[3.]])
          optimizer.minimize(
              lambda: network(input_value),  # pylint: disable=cell-var-from-loop
              global_step=root.global_step)
        latest_object_graph, _ = checkpointable.save(
            file_prefix=checkpoint_prefix,
            root_checkpointable=root)
        self.assertEqual((training_continuation + 1) * num_training_steps,
                         root.global_step.numpy())

  def testUsageGraph(self):
    """Expected usage when graph building."""
    with context.graph_mode():
      num_training_steps = 10
      checkpoint_directory = self.get_temp_dir()
      checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
      latest_object_graph = None
      for training_continuation in range(3):
        with ops.Graph().as_default():
          network = MyNetwork()
          optimizer = CheckpointableAdam(0.001)
          root = Root(optimizer=optimizer, network=network)
          input_value = constant_op.constant([[3.]])
          train_op = optimizer.minimize(
              network(input_value),
              global_step=root.global_step)
          init_op = variables.global_variables_initializer()
          checkpoint_path = core_saver.latest_checkpoint(checkpoint_directory)
          with self.test_session(graph=ops.get_default_graph()) as session:
            if checkpoint_path is None:
              self.assertEqual(0, training_continuation)
              session.run(init_op)
              # Another alternative would be to run initializers automatically
              # if no checkpoint is being loaded. This would make deferred
              # loading a bit more useful with graph execution.
            else:
              checkpointable.restore(
                  save_path=checkpoint_path,
                  root_checkpointable=root,
                  object_graph_proto=latest_object_graph,
                  session=session)
            for _ in range(num_training_steps):
              session.run(train_op)
            latest_object_graph, _ = checkpointable.save(
                file_prefix=checkpoint_prefix,
                root_checkpointable=root,
                session=session)
            self.assertEqual((training_continuation + 1) * num_training_steps,
                             session.run(root.global_step))

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
    self.assertEqual(r"_1/v", variable_name)

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
