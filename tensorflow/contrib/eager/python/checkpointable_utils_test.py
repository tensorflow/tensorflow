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
import weakref

import six

from tensorflow.contrib.eager.python import checkpointable_utils
from tensorflow.contrib.eager.python import network as network_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.layers import base
from tensorflow.python.layers import core
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import adam
from tensorflow.python.training import checkpointable
from tensorflow.python.training import saver as core_saver
from tensorflow.python.training import training_util


class CheckpointableDenseLayer(core.Dense, checkpointable.Checkpointable):

  def __init__(self, *args, **kwargs):
    checkpointable.Checkpointable.__init__(self)
    core.Dense.__init__(self, *args, **kwargs)

  def add_variable(self, name, shape, **kwargs):
    # Calls both Checkpointable._add_variable and Layer.add_variable. Eventually
    # Layer.add_variable should inherit from Checkpointable and simply call
    # super and then do post-processing.
    return checkpointable.Checkpointable._add_variable_with_custom_getter(
        self,
        name=name,
        shape=shape,
        getter=functools.partial(core.Dense.add_variable, self),
        **kwargs)


# pylint: disable=not-callable
class CheckpointableNetwork(network_lib.Network, checkpointable.Checkpointable):

  def __setattr__(self, name, value):
    if isinstance(value, base.Layer):
      self.track_layer(value, name=name)
    # Checkpointable is next in the method resolution order, so this will catch
    # Checkpointable objects which aren't Layers.
    super(CheckpointableNetwork, self).__setattr__(name, value)

  def track_layer(self, layer, name):
    self._track_checkpointable(layer, name=name)
    return super(CheckpointableNetwork, self).track_layer(layer)


class CheckpointableAdam(adam.AdamOptimizer, checkpointable.Checkpointable):

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
        v = self._add_variable_with_custom_getter(
            name=name,
            shape=initial_value.get_shape(),
            initializer=initial_value,
            getter=_variable_getter)

      self._non_slot_dict[key] = v

    return v


class NonLayerCheckpointable(checkpointable.Checkpointable):

  def __init__(self):
    super(NonLayerCheckpointable, self).__init__()
    self.a_variable = checkpointable_utils.add_variable(
        self, name="a_variable", shape=[])


class MyNetwork(CheckpointableNetwork):
  """A concrete Network for testing."""

  def __init__(self):
    super(MyNetwork, self).__init__()
    self._named_dense = CheckpointableDenseLayer(1, use_bias=True)
    self._via_track_layer = self.track_layer(
        CheckpointableDenseLayer(1, use_bias=False), name="via_track_layer")
    # We can still track Checkpointables which aren't Layers.
    self._non_layer = NonLayerCheckpointable()

  def call(self, values):
    return self._via_track_layer(self._named_dense(values))


class Checkpoint(checkpointable.Checkpointable):
  """A utility class which groups `Checkpointable` objects."""

  def __init__(self, **kwargs):
    super(Checkpoint, self).__init__()
    for k, v in sorted(kwargs.items(), key=lambda item: item[0]):
      setattr(self, k, v)
    self._save_counter = None
    self._saver = checkpointable_utils.Saver(weakref.ref(self))

  @property
  def save_counter(self):
    """An integer variable which starts at zero and is incremented on save.

    Used to number checkpoints.

    Returns:
      The save counter variable.
    """
    if self._save_counter is None:
      # Initialized to 0 and incremented before saving.
      self._save_counter = checkpointable_utils.add_variable(
          self, name="save_counter", initializer=0, dtype=dtypes.int64)
    return self._save_counter

  def save(self, file_prefix, session=None):
    assign_op = self.save_counter.assign_add(1)
    if context.in_graph_mode():
      if session is None:
        session = ops.get_default_session()
      session.run(assign_op)
    return self._saver.save(
        file_prefix=file_prefix,
        checkpoint_number=self.save_counter,
        session=session)

  def restore(self, save_path):
    return self._saver.restore(
        save_path=save_path)


class InterfaceTests(test.TestCase):

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testAddVariable(self):
    obj = NonLayerCheckpointable()
    with self.assertRaisesRegexp(ValueError, "do not specify shape"):
      checkpointable_utils.add_variable(
          obj, name="shape_specified_twice", shape=[], initializer=1)
    constant_initializer = checkpointable_utils.add_variable(
        obj, name="constant_initializer", initializer=1)
    with variable_scope.variable_scope("some_variable_scope"):
      ones_initializer = checkpointable_utils.add_variable(
          obj,
          name="ones_initializer",
          shape=[2],
          initializer=init_ops.ones_initializer(dtype=dtypes.float32))
    bare_initializer = checkpointable_utils.add_variable(
        obj,
        name="bare_initializer",
        shape=[2, 2],
        dtype=dtypes.float64,
        initializer=init_ops.zeros_initializer)

    # Even in graph mode, there are no naming conflicts between objects, only
    # naming conflicts within an object.
    other_duplicate = resource_variable_ops.ResourceVariable(
        name="duplicate", initial_value=1.)
    duplicate = checkpointable_utils.add_variable(
        obj, name="duplicate", shape=[])
    with self.assertRaisesRegexp(ValueError, "'duplicate' already exists"):
      checkpointable_utils.add_variable(obj, name="duplicate", shape=[])

    if context.in_graph_mode():
      self.evaluate(variables.global_variables_initializer())
    self.assertEqual("constant_initializer:0", constant_initializer.name)
    self.assertEqual(1, self.evaluate(constant_initializer))
    self.assertEqual("some_variable_scope/ones_initializer:0",
                     ones_initializer.name)
    self.assertAllEqual([1, 1], self.evaluate(ones_initializer))
    self.assertAllEqual([[0., 0.],
                         [0., 0.]], self.evaluate(bare_initializer))
    self.assertEqual("a_variable:0", obj.a_variable.name)
    self.assertEqual("duplicate:0", other_duplicate.name)
    if context.in_graph_mode():
      # The .name attribute may be globally influenced, but the checkpoint name
      # won't be (tested below).
      self.assertEqual("duplicate_1:0", duplicate.name)
    else:
      # When executing eagerly, there's no uniquification of variable names. The
      # checkpoint name will be the same.
      self.assertEqual("duplicate:0", duplicate.name)
    named_variables, _ = checkpointable_utils._serialize_object_graph(obj)
    expected_checkpoint_names = (
        "a_variable/.ATTRIBUTES/VARIABLE_VALUE",
        "bare_initializer/.ATTRIBUTES/VARIABLE_VALUE",
        "constant_initializer/.ATTRIBUTES/VARIABLE_VALUE",
        "duplicate/.ATTRIBUTES/VARIABLE_VALUE",
        "ones_initializer/.ATTRIBUTES/VARIABLE_VALUE",
    )
    six.assertCountEqual(
        self, expected_checkpoint_names, named_variables.keys())

  def testInitNotCalled(self):

    class NoInit(checkpointable.Checkpointable):

      def __init__(self):
        pass

    # __init__ for Checkpointable will be called implicitly.
    checkpointable_utils.add_variable(NoInit(), "var", shape=[])

  def testShapeDtype(self):
    root = checkpointable.Checkpointable()
    v1 = checkpointable_utils.add_variable(
        root, name="v1", initializer=3., dtype=dtypes.float64)
    self.assertEqual(dtypes.float64, v1.dtype)
    v2 = checkpointable_utils.add_variable(
        root,
        name="v2",
        shape=[3],
        initializer=init_ops.ones_initializer,
        dtype=dtypes.float64)
    self.assertEqual(dtypes.float64, v2.dtype)
    self.assertAllEqual([1., 1., 1.], self.evaluate(v2))


class CheckpointingTests(test.TestCase):

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testNamingWithOptimizer(self):
    input_value = constant_op.constant([[3.]])
    network = MyNetwork()
    # A nuisance Network using the same optimizer. Its slot variables should not
    # go in the checkpoint, since it is never depended on.
    other_network = MyNetwork()
    optimizer = CheckpointableAdam(0.001)
    optimizer_step = training_util.get_or_create_global_step()
    root_checkpointable = Checkpoint(
        optimizer=optimizer, network=network, optimizer_step=optimizer_step)
    if context.in_eager_mode():
      optimizer.minimize(
          lambda: network(input_value),
          global_step=optimizer_step)
      optimizer.minimize(
          lambda: other_network(input_value),
          global_step=optimizer_step)
    else:
      train_op = optimizer.minimize(
          network(input_value), global_step=optimizer_step)
      optimizer.minimize(
          other_network(input_value),
          global_step=optimizer_step)
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(train_op)
    named_variables, serialized_graph = (
        checkpointable_utils._serialize_object_graph(root_checkpointable))
    expected_checkpoint_names = (
        # Created in the root node, so no prefix.
        "optimizer_step",
        # No name provided to track_checkpointable(), so the position is used
        # instead (one-based).
        "network/via_track_layer/kernel",
        # track_checkpointable() with a name provided, so that's used
        "network/_named_dense/kernel",
        "network/_named_dense/bias",
        # non-Layer dependency of the network
        "network/_non_layer/a_variable",
        # The optimizer creates two non-slot variables
        "optimizer/beta1_power",
        "optimizer/beta2_power",
        # Slot variables
        "network/via_track_layer/kernel/.OPTIMIZER_SLOT/optimizer/m",
        "network/via_track_layer/kernel/.OPTIMIZER_SLOT/optimizer/v",
        "network/_named_dense/kernel/.OPTIMIZER_SLOT/optimizer/m",
        "network/_named_dense/kernel/.OPTIMIZER_SLOT/optimizer/v",
        "network/_named_dense/bias/.OPTIMIZER_SLOT/optimizer/m",
        "network/_named_dense/bias/.OPTIMIZER_SLOT/optimizer/v",
    )
    suffix = "/.ATTRIBUTES/VARIABLE_VALUE"
    expected_checkpoint_names = [
        name + suffix for name in expected_checkpoint_names]
    six.assertCountEqual(self, expected_checkpoint_names,
                         named_variables.keys())
    # Check that we've mapped to the right variable objects (not exhaustive)
    self.assertEqual(
        "global_step:0",
        named_variables["optimizer_step" + suffix].name)
    self.assertEqual(
        "my_network/checkpointable_dense_layer_1/kernel:0",
        named_variables["network/via_track_layer/kernel" + suffix].name)
    self.assertEqual(
        "my_network/checkpointable_dense_layer/kernel:0",
        named_variables["network/_named_dense/kernel" + suffix].name)
    self.assertEqual(
        "beta1_power:0",
        named_variables["optimizer/beta1_power" + suffix].name)
    self.assertEqual(
        "beta2_power:0",
        named_variables["optimizer/beta2_power" + suffix].name)
    # Spot check the generated protocol buffers.
    self.assertEqual("optimizer",
                     serialized_graph.nodes[0].children[1].local_name)
    optimizer_node = serialized_graph.nodes[serialized_graph.nodes[0].children[
        1].node_id]
    self.assertEqual("beta1_power",
                     optimizer_node.children[0].local_name)
    self.assertEqual("beta1_power",
                     serialized_graph.nodes[optimizer_node.children[0].node_id]
                     .attributes[0].full_name)
    self.assertEqual(
        "my_network/checkpointable_dense_layer/kernel",
        serialized_graph.nodes[optimizer_node.slot_variables[0]
                               .original_variable_node_id]
        .attributes[0].full_name)
    # We strip off the :0 suffix, as variable.name-based saving does.
    self.assertEqual(
        "my_network/checkpointable_dense_layer/kernel/Adam",
        serialized_graph.nodes[optimizer_node.slot_variables[0]
                               .slot_variable_node_id]
        .attributes[0].full_name)
    self.assertEqual(
        "my_network/checkpointable_dense_layer/kernel/Adam:0",
        optimizer.get_slot(
            var=named_variables["network/_named_dense/kernel" + suffix],
            name="m").name)
    self.assertEqual(
        "network/_named_dense/kernel" + suffix,
        serialized_graph.nodes[
            optimizer_node.slot_variables[0]
            .original_variable_node_id].attributes[0].checkpoint_key)
    self.assertEqual("m", optimizer_node.slot_variables[0].slot_name)
    self.assertEqual(
        "network/_named_dense/kernel/.OPTIMIZER_SLOT/optimizer/m" + suffix,
        serialized_graph.nodes[
            optimizer_node.slot_variables[0]
            .slot_variable_node_id].attributes[0].checkpoint_key)

  @test_util.run_in_graph_and_eager_modes()
  def testSaveRestore(self):
    network = MyNetwork()
    optimizer = CheckpointableAdam(0.001)
    root_checkpointable = Checkpoint(optimizer=optimizer, network=network)
    input_value = constant_op.constant([[3.]])
    if context.in_eager_mode():
      optimizer.minimize(
          lambda: network(input_value))
    else:
      train_op = optimizer.minimize(network(input_value))
      # TODO(allenl): Make initialization more pleasant when graph building.
      root_checkpointable.save_counter  # pylint: disable=pointless-statement
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(train_op)
    prefix = os.path.join(self.get_temp_dir(), "ckpt")
    self.evaluate(state_ops.assign(network._named_dense.variables[1], [42.]))
    m_bias_slot = optimizer.get_slot(network._named_dense.variables[1], "m")
    self.evaluate(state_ops.assign(m_bias_slot, [1.5]))
    save_path = root_checkpointable.save(file_prefix=prefix)
    self.evaluate(state_ops.assign(network._named_dense.variables[1], [43.]))
    self.evaluate(state_ops.assign(root_checkpointable.save_counter, 3))
    optimizer_variables = self.evaluate(optimizer.variables())
    self.evaluate(state_ops.assign(m_bias_slot, [-2.]))
    # Immediate restoration
    status = root_checkpointable.restore(save_path=save_path).assert_consumed()
    status.run_restore_ops()
    self.assertAllEqual([42.], self.evaluate(network._named_dense.variables[1]))
    self.assertAllEqual(1, self.evaluate(root_checkpointable.save_counter))
    self.assertAllEqual([1.5], self.evaluate(m_bias_slot))
    if context.in_graph_mode():
      return  # Restore-on-create is only supported when executing eagerly
    on_create_network = MyNetwork()
    on_create_optimizer = CheckpointableAdam(0.001)
    on_create_root = Checkpoint(
        optimizer=on_create_optimizer, network=on_create_network)
    # Deferred restoration
    status = on_create_root.restore(save_path=save_path)
    on_create_network(constant_op.constant([[3.]]))  # create variables
    self.assertAllEqual(1, self.evaluate(on_create_root.save_counter))
    self.assertAllEqual([42.],
                        self.evaluate(
                            on_create_network._named_dense.variables[1]))
    on_create_m_bias_slot = on_create_optimizer.get_slot(
        on_create_network._named_dense.variables[1], "m")
    # Optimizer slot variables are created when the original variable is
    # restored.
    self.assertAllEqual([1.5], self.evaluate(on_create_m_bias_slot))
    self.assertAllEqual(optimizer_variables[2:],
                        self.evaluate(on_create_optimizer.variables()))
    on_create_optimizer._create_slots(
        [resource_variable_ops.ResourceVariable([1.])])
    status.assert_consumed()
    beta1_power, beta2_power = on_create_optimizer._get_beta_accumulators()
    self.assertAllEqual(optimizer_variables[0], self.evaluate(beta1_power))
    self.assertAllEqual(optimizer_variables[1], self.evaluate(beta2_power))

  def testDeferredRestorationUsageEager(self):
    """An idiomatic eager execution example."""
    num_training_steps = 10
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    for training_continuation in range(3):
      network = MyNetwork()
      optimizer = CheckpointableAdam(0.001)
      root = Checkpoint(
          optimizer=optimizer, network=network,
          optimizer_step=training_util.get_or_create_global_step())
      root.restore(core_saver.latest_checkpoint(checkpoint_directory))
      for _ in range(num_training_steps):
        # TODO(allenl): Use a Dataset and serialize/checkpoint it.
        input_value = constant_op.constant([[3.]])
        optimizer.minimize(
            lambda: network(input_value),  # pylint: disable=cell-var-from-loop
            global_step=root.optimizer_step)
      root.save(file_prefix=checkpoint_prefix)
      self.assertEqual((training_continuation + 1) * num_training_steps,
                       root.optimizer_step.numpy())

  def testUsageGraph(self):
    """Expected usage when graph building."""
    with context.graph_mode():
      num_training_steps = 10
      checkpoint_directory = self.get_temp_dir()
      checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
      for training_continuation in range(3):
        with ops.Graph().as_default():
          network = MyNetwork()
          optimizer = CheckpointableAdam(0.001)
          root = Checkpoint(
              optimizer=optimizer, network=network,
              global_step=training_util.get_or_create_global_step())
          input_value = constant_op.constant([[3.]])
          train_op = optimizer.minimize(
              network(input_value),
              global_step=root.global_step)
          root.save_counter  # pylint: disable=pointless-statement
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
              status = root.restore(save_path=checkpoint_path).assert_consumed()
              status.run_restore_ops()
            for _ in range(num_training_steps):
              session.run(train_op)
            root.save(file_prefix=checkpoint_prefix,
                      session=session)
            self.assertEqual((training_continuation + 1) * num_training_steps,
                             session.run(root.global_step))
            self.assertEqual(training_continuation + 1,
                             session.run(root.save_counter))

  def _get_checkpoint_name(self, name):
    root = checkpointable.Checkpointable()
    checkpointable_utils.add_variable(
        root, name=name, shape=[1, 2], dtype=dtypes.float64)
    named_variables, _ = checkpointable_utils._serialize_object_graph(root)
    checkpoint_name, = named_variables.keys()
    with ops.name_scope("root/" + checkpoint_name):
      pass  # Make sure we can use this as an op name if we prefix it.
    return checkpoint_name

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testVariableNameEscaping(self):
    suffix = "/.ATTRIBUTES/VARIABLE_VALUE"
    self.assertEqual(r"a.Sb.Sc" + suffix, self._get_checkpoint_name(r"a/b/c"))
    self.assertEqual(r"b" + suffix, self._get_checkpoint_name(r"b"))
    self.assertEqual(r"c.S" + suffix, self._get_checkpoint_name(r"c/"))
    self.assertEqual(r"d.S..S" + suffix, self._get_checkpoint_name(r"d/.S"))
    self.assertEqual(r"d.S..ATTRIBUTES.Sf" + suffix,
                     self._get_checkpoint_name(r"d/.ATTRIBUTES/f"))

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testNumberedPath(self):
    root = checkpointable.Checkpointable()
    leaf = checkpointable.Checkpointable()
    root.leaf = leaf
    checkpointable_utils.add_variable(leaf, name="v", shape=[])
    named_variables, _ = checkpointable_utils._serialize_object_graph(root)
    variable_name, = named_variables.keys()
    self.assertEqual(r"leaf/v/.ATTRIBUTES/VARIABLE_VALUE", variable_name)

  @test_util.run_in_graph_and_eager_modes()
  def testLocalNameValidation(self):
    root = checkpointable.Checkpointable()
    leaf = checkpointable.Checkpointable()
    # Dots are escaped, which avoids conflicts with reserved names.
    root._track_checkpointable(leaf, name=".ATTRIBUTES")
    checkpointable_utils.add_variable(checkpointable=leaf, name="a", shape=[])
    named_variables, _ = checkpointable_utils._serialize_object_graph(root)
    name, = named_variables.keys()
    self.assertEqual(name, "..ATTRIBUTES/a/.ATTRIBUTES/VARIABLE_VALUE")

  @test_util.run_in_graph_and_eager_modes()
  def testLateDependencyTracking(self):

    class Dependency(checkpointable.Checkpointable):

      def build(self):
        self.var = checkpointable_utils.add_variable(
            self, "var", initializer=0.)

    class LateDependencies(checkpointable.Checkpointable):

      def add_dep(self):
        self.dep = Dependency()
        self.dep.build()

    original = LateDependencies()
    original.add_dep()
    self.evaluate(state_ops.assign(original.dep.var, 123.))
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    save_path = checkpointable_utils.Saver(original).save(checkpoint_prefix)
    load_into = LateDependencies()
    status = checkpointable_utils.Saver(load_into).restore(save_path)
    with self.assertRaises(AssertionError):
      status.assert_consumed()
    load_into.add_dep()
    status.assert_consumed()
    status.run_restore_ops()
    self.assertEqual(123., self.evaluate(load_into.dep.var))

  @test_util.run_in_graph_and_eager_modes()
  def testDepAfterVar(self):

    class Dependency(checkpointable.Checkpointable):

      def build(self):
        self.var = checkpointable_utils.add_variable(
            self, "var", initializer=0.)

    class DepAfterVar(checkpointable.Checkpointable):

      def add_dep(self):
        dep = Dependency()
        dep.build()
        self.dep = dep

    dep_after_var = DepAfterVar()
    dep_after_var.add_dep()
    self.evaluate(state_ops.assign(dep_after_var.dep.var, -14.))
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    save_path = checkpointable_utils.Saver(dep_after_var).save(
        checkpoint_prefix)

    loaded_dep_after_var = DepAfterVar()
    status = checkpointable_utils.Saver(loaded_dep_after_var).restore(save_path)
    loaded_dep_after_var.add_dep()
    status.assert_consumed()
    status.run_restore_ops()
    self.assertEqual(-14., self.evaluate(loaded_dep_after_var.dep.var))

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testDeferredSlotRestoration(self):
    checkpoint_directory = self.get_temp_dir()

    root = checkpointable.Checkpointable()
    root.var = checkpointable_utils.add_variable(
        root, name="var", initializer=0.)
    optimizer = CheckpointableAdam(0.1)
    if context.in_graph_mode():
      train_op = optimizer.minimize(root.var)
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(train_op)
    else:
      optimizer.minimize(root.var.read_value)
    self.evaluate(state_ops.assign(root.var, 12.))
    no_slots_path = checkpointable_utils.Saver(root).save(
        os.path.join(checkpoint_directory, "no_slots"))
    root.optimizer = optimizer
    self.evaluate(state_ops.assign(root.var, 13.))
    self.evaluate(state_ops.assign(optimizer.get_slot(name="m", var=root.var),
                                   14.))
    slots_path = checkpointable_utils.Saver(root).save(
        os.path.join(checkpoint_directory, "with_slots"))
    new_root = checkpointable.Checkpointable()
    # Load the slot-containing checkpoint (deferred), then immediately overwrite
    # the non-slot variable (also deferred).
    slot_status = checkpointable_utils.Saver(new_root).restore(slots_path)
    no_slot_status = checkpointable_utils.Saver(new_root).restore(no_slots_path)
    with self.assertRaises(AssertionError):
      no_slot_status.assert_consumed()
    new_root.var = checkpointable_utils.add_variable(
        new_root, name="var", shape=[])
    no_slot_status.assert_consumed()
    no_slot_status.run_restore_ops()
    self.assertEqual(12., self.evaluate(new_root.var))
    new_root.optimizer = CheckpointableAdam(0.1)
    with self.assertRaisesRegexp(AssertionError, "beta1_power"):
      slot_status.assert_consumed()
    self.assertEqual(12., self.evaluate(new_root.var))
    if context.in_eager_mode():
      # Slot variables are only created with restoring initializers when
      # executing eagerly.
      self.assertEqual(14., self.evaluate(
          new_root.optimizer.get_slot(name="m", var=new_root.var)))
    else:
      self.assertIs(new_root.optimizer.get_slot(name="m", var=new_root.var),
                    None)
    if context.in_graph_mode():
      train_op = new_root.optimizer.minimize(new_root.var)
      # The slot variable now exists; restore() didn't create it, but we should
      # now have a restore op for it.
      slot_status.run_restore_ops()
      self.assertEqual(14., self.evaluate(
          new_root.optimizer.get_slot(name="m", var=new_root.var)))
      self.evaluate(train_op)
    else:
      new_root.optimizer.minimize(new_root.var.read_value)
    slot_status.assert_consumed()

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testOverlappingRestores(self):
    checkpoint_directory = self.get_temp_dir()
    save_root = checkpointable.Checkpointable()
    save_root.dep = checkpointable.Checkpointable()
    save_root.dep.var = checkpointable_utils.add_variable(
        save_root.dep, name="var", initializer=0.)
    self.evaluate(state_ops.assign(save_root.dep.var, 12.))
    saver = checkpointable_utils.Saver(save_root)
    first_path = saver.save(os.path.join(checkpoint_directory, "first"))
    self.evaluate(state_ops.assign(save_root.dep.var, 13.))
    second_path = saver.save(os.path.join(checkpoint_directory, "second"))

    first_root = checkpointable.Checkpointable()
    second_root = checkpointable.Checkpointable()
    first_status = checkpointable_utils.Saver(first_root).restore(first_path)
    second_status = checkpointable_utils.Saver(second_root).restore(second_path)
    load_dep = checkpointable.Checkpointable()
    load_dep.var = checkpointable_utils.add_variable(
        load_dep, name="var", shape=[])
    first_root.dep = load_dep
    first_status.assert_consumed()
    first_status.run_restore_ops()
    self.assertEqual(12., self.evaluate(load_dep.var))
    second_root.dep = load_dep
    second_status.assert_consumed()
    second_status.run_restore_ops()
    self.assertEqual(13., self.evaluate(load_dep.var))

    # Try again with the order of the restore() reversed. The last restore
    # determines the final value.
    first_root = checkpointable.Checkpointable()
    second_root = checkpointable.Checkpointable()
    second_status = checkpointable_utils.Saver(second_root).restore(second_path)
    first_status = checkpointable_utils.Saver(first_root).restore(first_path)
    load_dep = checkpointable.Checkpointable()
    load_dep.var = checkpointable_utils.add_variable(
        load_dep, name="var", shape=[])
    first_root.dep = load_dep
    first_status.assert_consumed()
    first_status.run_restore_ops()
    self.assertEqual(12., self.evaluate(load_dep.var))
    second_root.dep = load_dep
    second_status.assert_consumed()
    second_status.run_restore_ops()
    self.assertEqual(12., self.evaluate(load_dep.var))

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testAmbiguousLoad(self):
    # Not OK to split one checkpoint object into two
    checkpoint_directory = self.get_temp_dir()
    save_root = checkpointable.Checkpointable()
    save_root.dep_one = checkpointable.Checkpointable()
    save_root.dep_two = checkpointable.Checkpointable()
    dep_three = checkpointable.Checkpointable()
    save_root.dep_one.dep_three = dep_three
    save_root.dep_two.dep_three = dep_three
    checkpointable_utils.add_variable(dep_three, name="var", initializer=0.)
    self.evaluate(variables.global_variables_initializer())
    save_path = checkpointable_utils.Saver(save_root).save(
        os.path.join(checkpoint_directory, "ckpt"))
    load_root = checkpointable.Checkpointable()
    checkpointable_utils.Saver(load_root).restore(save_path)
    load_root.dep_one = checkpointable.Checkpointable()
    load_root.dep_two = checkpointable.Checkpointable()
    load_root.dep_one.dep_three = checkpointable.Checkpointable()
    with self.assertRaisesRegexp(AssertionError,
                                 "resolved to different objects"):
      load_root.dep_two.dep_three = checkpointable.Checkpointable()

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testObjectsCombined(self):
    # Currently fine to load two checkpoint objects into one Python object
    checkpoint_directory = self.get_temp_dir()
    save_root = checkpointable.Checkpointable()
    save_root.dep_one = checkpointable.Checkpointable()
    save_root.dep_two = checkpointable.Checkpointable()
    checkpointable_utils.add_variable(
        save_root.dep_one, name="var1", initializer=32., dtype=dtypes.float64)
    checkpointable_utils.add_variable(
        save_root.dep_two, name="var2", initializer=64., dtype=dtypes.float64)
    self.evaluate(variables.global_variables_initializer())
    save_path = checkpointable_utils.Saver(save_root).save(
        os.path.join(checkpoint_directory, "ckpt"))
    load_root = checkpointable.Checkpointable()
    load_root.dep_one = checkpointable.Checkpointable()
    load_root.dep_two = load_root.dep_one
    v1 = checkpointable_utils.add_variable(
        load_root.dep_one, name="var1", shape=[], dtype=dtypes.float64)
    v2 = checkpointable_utils.add_variable(
        load_root.dep_one, name="var2", shape=[], dtype=dtypes.float64)
    status = checkpointable_utils.Saver(load_root).restore(
        save_path).assert_consumed()
    status.run_restore_ops()
    self.assertEqual(32., self.evaluate(v1))
    self.assertEqual(64., self.evaluate(v2))

  @test_util.run_in_graph_and_eager_modes()
  def testDependencyLoop(self):
    # Note: this test creates garbage during eager execution because it
    # purposefully creates a reference cycle.
    first = checkpointable.Checkpointable()
    second = checkpointable.Checkpointable()
    first.second = second
    second.first = first
    first.v = checkpointable_utils.add_variable(
        first, "v1", initializer=[3., 1., 4.])
    second.v = checkpointable_utils.add_variable(
        second, "v2", initializer=[1., 1., 2., 3.])
    self.evaluate(variables.global_variables_initializer())
    checkpoint_directory = self.get_temp_dir()
    save_path = checkpointable_utils.Saver(first).save(
        os.path.join(checkpoint_directory, "ckpt"))

    # Test deferred loading
    first_load = checkpointable.Checkpointable()
    status = checkpointable_utils.Saver(first_load).restore(save_path)
    second_load = checkpointable.Checkpointable()
    first_load.second = second_load
    second_load.first = first_load
    with self.assertRaises(AssertionError):
      status.assert_consumed()
    first_load.v = checkpointable_utils.add_variable(
        first_load, "v1", shape=[3])
    second_load.v = checkpointable_utils.add_variable(
        second_load, "v2", shape=[4])
    status.assert_consumed()
    status.run_restore_ops()
    self.assertAllEqual([3., 1., 4.], self.evaluate(first_load.v))
    self.assertAllEqual([1., 1., 2., 3.], self.evaluate(second_load.v))

    # Test loading when variables have already been created
    self.evaluate(first_load.v.assign([2., 7., 1.]))
    self.assertAllEqual([2., 7., 1.], self.evaluate(first_load.v))
    self.evaluate(second_load.v.assign([2., 7., 1., 8.]))
    self.assertAllEqual([2., 7., 1., 8.], self.evaluate(second_load.v))
    status = checkpointable_utils.Saver(first_load).restore(
        save_path).assert_consumed()
    status.run_restore_ops()
    self.assertAllEqual([3., 1., 4.], self.evaluate(first_load.v))
    self.assertAllEqual([1., 1., 2., 3.], self.evaluate(second_load.v))

  @test_util.run_in_graph_and_eager_modes()
  def testRestoreOnAssign(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    save_graph = ops.Graph()
    with save_graph.as_default(), self.test_session(save_graph):
      first = checkpointable.Checkpointable()
      first.var1 = variable_scope.get_variable(
          name="outside_var", initializer=0.)
      first.var2 = variable_scope.get_variable(
          name="blah", initializer=0.)
      self.evaluate(first.var1.assign(4.))
      self.evaluate(first.var2.assign(8.))
      save_path = checkpointable_utils.Saver(first).save(
          checkpoint_prefix)
    restore_graph = ops.Graph()
    with restore_graph.as_default(), self.test_session(restore_graph):
      second = checkpointable.Checkpointable()
      second.var2 = variable_scope.get_variable(
          name="blah", initializer=0.)
      status = checkpointable_utils.Saver(second).restore(save_path)
      recreated_var1 = variable_scope.get_variable(
          name="outside_var", initializer=0.)
      status.run_restore_ops()
      self.assertEqual(8., self.evaluate(second.var2))
      self.evaluate(recreated_var1.assign(-2.))
      self.assertEqual(-2., self.evaluate(recreated_var1))
      second.var1 = recreated_var1
      status.run_restore_ops()
      self.assertEqual(4., self.evaluate(recreated_var1))

  def testManySavesGraph(self):
    """Saves after the first should not modify the graph."""
    with context.graph_mode():
      graph = ops.Graph()
      with graph.as_default(), self.test_session(graph):
        checkpoint_directory = self.get_temp_dir()
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        obj = checkpointable.Checkpointable()
        obj.var = variable_scope.get_variable(name="v", initializer=0.)
        obj.opt = CheckpointableAdam(0.1)
        obj.opt.minimize(obj.var.read_value())
        self.evaluate(variables.global_variables_initializer())
        saver = checkpointable_utils.Saver(obj)
        saver.save(checkpoint_prefix)
        before_ops = graph.get_operations()
        saver.save(checkpoint_prefix)
        self.assertEqual(before_ops, graph.get_operations())

  def testManyRestoresGraph(self):
    """Restores after the first should not modify the graph."""
    with context.graph_mode():
      graph = ops.Graph()
      with graph.as_default(), self.test_session(graph):
        checkpoint_directory = self.get_temp_dir()
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        obj = checkpointable.Checkpointable()
        obj.var = variable_scope.get_variable(name="v", initializer=0.)
        obj.opt = CheckpointableAdam(0.1)
        obj.opt.minimize(obj.var.read_value())
        self.evaluate(variables.global_variables_initializer())
        saver = checkpointable_utils.Saver(obj)
        save_path = saver.save(checkpoint_prefix)
        saver.restore(save_path)
        before_ops = graph.get_operations()
        saver.restore(save_path)
        self.assertEqual(before_ops, graph.get_operations())

if __name__ == "__main__":
  test.main()
