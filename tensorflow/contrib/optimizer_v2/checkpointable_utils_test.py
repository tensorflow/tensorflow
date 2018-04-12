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

# TODO(josh11b): Forked from contrib/eager/python to test OptimizerV2 the same way
# OptimizerV1 is tested. This file should be removed once the fork is resolved.

import functools
import os

import six

from tensorflow.contrib.eager.python import checkpointable_utils
from tensorflow.contrib.optimizer_v2 import adam
from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras._impl.keras.engine import training
from tensorflow.python.layers import core
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import checkpointable
from tensorflow.python.training import saver as core_saver
from tensorflow.python.training import training_util


class NonLayerCheckpointable(checkpointable.Checkpointable):

  def __init__(self):
    super(NonLayerCheckpointable, self).__init__()
    self.a_variable = checkpointable_utils.add_variable(
        self, name="a_variable", shape=[])


# pylint: disable=not-callable
class MyModel(training.Model):
  """A concrete Model for testing."""

  def __init__(self):
    super(MyModel, self).__init__()
    self._named_dense = core.Dense(1, use_bias=True)
    self._second = core.Dense(1, use_bias=False)
    # We can still track Checkpointables which aren't Layers.
    self._non_layer = NonLayerCheckpointable()

  def call(self, values):
    ret = self._second(self._named_dense(values))
    return ret


class _MirroringSaveable(
    core_saver.BaseSaverBuilder.ResourceVariableSaveable):

  def __init__(self, primary_variable, mirrored_variable, name):
    self._primary_variable = primary_variable
    self._mirrored_variable = mirrored_variable
    super(_MirroringSaveable, self).__init__(
        self._primary_variable, "", name)

  def restore(self, restored_tensors, restored_shapes):
    """Restore the same value into both variables."""
    tensor, = restored_tensors
    return control_flow_ops.group(
        self._primary_variable.assign(tensor),
        self._mirrored_variable.assign(tensor))


class _OwnsMirroredVariables(checkpointable.CheckpointableBase):
  """A Checkpointable object which returns a more complex SaveableObject."""

  def __init__(self):
    self.non_dep_variable = variable_scope.get_variable(
        name="non_dep_variable", initializer=6., use_resource=True)
    self.mirrored = variable_scope.get_variable(
        name="mirrored", initializer=15., use_resource=True)

  def _gather_saveables_for_checkpoint(self):
    def _saveable_factory(name=self.non_dep_variable.name):
      return _MirroringSaveable(
          primary_variable=self.non_dep_variable,
          mirrored_variable=self.mirrored,
          name=name)
    return {checkpointable.VARIABLE_VALUE_KEY: _saveable_factory}

  # The Saver sorts by name before parsing, so we need a name property.
  @property
  def name(self):
    return self.non_dep_variable.name


class CheckpointingTests(test.TestCase):

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testNamingWithOptimizer(self):
    input_value = constant_op.constant([[3.]])
    model = MyModel()
    # A nuisance Model using the same optimizer. Its slot variables should not
    # go in the checkpoint, since it is never depended on.
    other_model = MyModel()
    optimizer = adam.AdamOptimizer(0.001)
    optimizer_step = training_util.get_or_create_global_step()
    root_checkpointable = checkpointable_utils.Checkpoint(
        optimizer=optimizer, model=model, optimizer_step=optimizer_step)
    if context.executing_eagerly():
      optimizer.minimize(
          lambda: model(input_value),
          global_step=optimizer_step)
      optimizer.minimize(
          lambda: other_model(input_value),
          global_step=optimizer_step)
    else:
      train_op = optimizer.minimize(
          model(input_value), global_step=optimizer_step)
      optimizer.minimize(
          other_model(input_value),
          global_step=optimizer_step)
      self.evaluate(checkpointable_utils.gather_initializers(
          root_checkpointable))
      self.evaluate(train_op)
    named_variables, serialized_graph = (
        checkpointable_utils._serialize_object_graph(root_checkpointable))
    expected_checkpoint_names = (
        # Created in the root node, so no prefix.
        "optimizer_step",
        "model/_second/kernel",
        "model/_named_dense/kernel",
        "model/_named_dense/bias",
        # non-Layer dependency of the model
        "model/_non_layer/a_variable",
        # The optimizer creates two non-slot variables
        "optimizer/beta1_power",
        "optimizer/beta2_power",
        # Slot variables
        "model/_second/kernel/.OPTIMIZER_SLOT/optimizer/m",
        "model/_second/kernel/.OPTIMIZER_SLOT/optimizer/v",
        "model/_named_dense/kernel/.OPTIMIZER_SLOT/optimizer/m",
        "model/_named_dense/kernel/.OPTIMIZER_SLOT/optimizer/v",
        "model/_named_dense/bias/.OPTIMIZER_SLOT/optimizer/m",
        "model/_named_dense/bias/.OPTIMIZER_SLOT/optimizer/v",
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
        "my_model/dense_1/kernel:0",
        named_variables["model/_second/kernel" + suffix].name)
    self.assertEqual(
        "my_model/dense/kernel:0",
        named_variables["model/_named_dense/kernel" + suffix].name)
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
        "my_model/dense/kernel",
        serialized_graph.nodes[optimizer_node.slot_variables[0]
                               .original_variable_node_id]
        .attributes[0].full_name)
    # We strip off the :0 suffix, as variable.name-based saving does.
    self.assertEqual(
        "my_model/dense/kernel/Adam",
        serialized_graph.nodes[optimizer_node.slot_variables[0]
                               .slot_variable_node_id]
        .attributes[0].full_name)
    self.assertEqual(
        "my_model/dense/kernel/Adam:0",
        optimizer.get_slot(
            var=named_variables["model/_named_dense/kernel" + suffix],
            name="m").name)
    self.assertEqual(
        "model/_named_dense/kernel" + suffix,
        serialized_graph.nodes[
            optimizer_node.slot_variables[0]
            .original_variable_node_id].attributes[0].checkpoint_key)
    self.assertEqual("m", optimizer_node.slot_variables[0].slot_name)
    self.assertEqual(
        "model/_named_dense/kernel/.OPTIMIZER_SLOT/optimizer/m" + suffix,
        serialized_graph.nodes[
            optimizer_node.slot_variables[0]
            .slot_variable_node_id].attributes[0].checkpoint_key)

  @test_util.run_in_graph_and_eager_modes()
  def testSaveRestore(self):
    model = MyModel()
    optimizer = adam.AdamOptimizer(0.001)
    root_checkpointable = checkpointable_utils.Checkpoint(
        optimizer=optimizer, model=model)
    input_value = constant_op.constant([[3.]])
    if context.executing_eagerly():
      optimizer.minimize(
          lambda: model(input_value))
    else:
      train_op = optimizer.minimize(model(input_value))
      # TODO(allenl): Make initialization more pleasant when graph building.
      root_checkpointable.save_counter  # pylint: disable=pointless-statement
      self.evaluate(checkpointable_utils.gather_initializers(
          root_checkpointable))
      self.evaluate(train_op)
    prefix = os.path.join(self.get_temp_dir(), "ckpt")
    self.evaluate(state_ops.assign(model._named_dense.variables[1], [42.]))
    m_bias_slot = optimizer.get_slot(model._named_dense.variables[1], "m")
    self.evaluate(state_ops.assign(m_bias_slot, [1.5]))
    save_path = root_checkpointable.save(file_prefix=prefix)
    self.evaluate(state_ops.assign(model._named_dense.variables[1], [43.]))
    self.evaluate(state_ops.assign(root_checkpointable.save_counter, 3))
    optimizer_variables = self.evaluate(optimizer.variables())
    self.evaluate(state_ops.assign(m_bias_slot, [-2.]))
    # Immediate restoration
    status = root_checkpointable.restore(save_path=save_path).assert_consumed()
    status.run_restore_ops()
    self.assertAllEqual([42.], self.evaluate(model._named_dense.variables[1]))
    self.assertAllEqual(1, self.evaluate(root_checkpointable.save_counter))
    self.assertAllEqual([1.5], self.evaluate(m_bias_slot))
    if not context.executing_eagerly():
      return  # Restore-on-create is only supported when executing eagerly
    on_create_model = MyModel()
    on_create_optimizer = adam.AdamOptimizer(
        0.001,
        # Preserve beta1_power and beta2_power when appying gradients so we can
        # test that they've been restored correctly.
        beta1=1.0, beta2=1.0)
    on_create_root = checkpointable_utils.Checkpoint(
        optimizer=on_create_optimizer, model=on_create_model)
    # Deferred restoration
    status = on_create_root.restore(save_path=save_path)
    on_create_model(constant_op.constant([[3.]]))  # create variables
    self.assertAllEqual(1, self.evaluate(on_create_root.save_counter))
    self.assertAllEqual([42.],
                        self.evaluate(
                            on_create_model._named_dense.variables[1]))
    on_create_m_bias_slot = on_create_optimizer.get_slot(
        on_create_model._named_dense.variables[1], "m")
    # Optimizer slot variables are created when the original variable is
    # restored.
    self.assertAllEqual([1.5], self.evaluate(on_create_m_bias_slot))
    self.assertAllEqual(optimizer_variables[2:],
                        self.evaluate(on_create_optimizer.variables()))
    dummy_var = resource_variable_ops.ResourceVariable([1.])
    on_create_optimizer.minimize(loss=dummy_var.read_value)
    status.assert_consumed()
    beta1_power, beta2_power = on_create_optimizer._get_beta_accumulators()
    self.assertAllEqual(optimizer_variables[0], self.evaluate(beta1_power))
    self.assertAllEqual(optimizer_variables[1], self.evaluate(beta2_power))

  # TODO(allenl): Debug garbage created by this test in python3.
  def testDeferredRestorationUsageEager(self):
    """An idiomatic eager execution example."""
    num_training_steps = 10
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    for training_continuation in range(3):
      model = MyModel()
      optimizer = adam.AdamOptimizer(0.001)
      root = checkpointable_utils.Checkpoint(
          optimizer=optimizer, model=model,
          optimizer_step=training_util.get_or_create_global_step())
      root.restore(core_saver.latest_checkpoint(checkpoint_directory))
      for _ in range(num_training_steps):
        # TODO(allenl): Use a Dataset and serialize/checkpoint it.
        input_value = constant_op.constant([[3.]])
        optimizer.minimize(
            lambda: model(input_value),  # pylint: disable=cell-var-from-loop
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
          model = MyModel()
          optimizer = adam.AdamOptimizer(0.001)
          root = checkpointable_utils.Checkpoint(
              optimizer=optimizer, model=model,
              global_step=training_util.get_or_create_global_step())
          input_value = constant_op.constant([[3.]])
          train_op = optimizer.minimize(
              model(input_value),
              global_step=root.global_step)
          checkpoint_path = core_saver.latest_checkpoint(checkpoint_directory)
          with self.test_session(graph=ops.get_default_graph()) as session:
            status = root.restore(save_path=checkpoint_path)
            status.initialize_or_restore(session=session)
            if checkpoint_path is None:
              self.assertEqual(0, training_continuation)
              with self.assertRaises(AssertionError):
                status.assert_consumed()
            else:
              status.assert_consumed()
            for _ in range(num_training_steps):
              session.run(train_op)
            root.save(file_prefix=checkpoint_prefix, session=session)
            self.assertEqual((training_continuation + 1) * num_training_steps,
                             session.run(root.global_step))
            self.assertEqual(training_continuation + 1,
                             session.run(root.save_counter))

  @test_util.run_in_graph_and_eager_modes()
  def testAgnosticUsage(self):
    """Graph/eager agnostic usage."""
    # Does create garbage when executing eagerly due to ops.Graph() creation.
    num_training_steps = 10
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    for training_continuation in range(3):
      with ops.Graph().as_default(), self.test_session(
          graph=ops.get_default_graph()), test_util.device(use_gpu=True):
        model = MyModel()
        optimizer = adam.AdamOptimizer(0.001)
        root = checkpointable_utils.Checkpoint(
            optimizer=optimizer, model=model,
            global_step=training_util.get_or_create_global_step())
        checkpoint_path = core_saver.latest_checkpoint(checkpoint_directory)
        status = root.restore(save_path=checkpoint_path)
        input_value = constant_op.constant([[3.]])
        train_fn = functools.partial(
            optimizer.minimize,
            functools.partial(model, input_value),
            global_step=root.global_step)
        if not context.executing_eagerly():
          train_fn = functools.partial(self.evaluate, train_fn())
        status.initialize_or_restore()
        for _ in range(num_training_steps):
          train_fn()
        root.save(file_prefix=checkpoint_prefix)
        self.assertEqual((training_continuation + 1) * num_training_steps,
                         self.evaluate(root.global_step))
        self.assertEqual(training_continuation + 1,
                         self.evaluate(root.save_counter))

  def _get_checkpoint_name(self, name):
    root = checkpointable.Checkpointable()
    checkpointable_utils.add_variable(
        root, name=name, shape=[1, 2], dtype=dtypes.float64)
    named_variables, _ = checkpointable_utils._serialize_object_graph(root)
    checkpoint_name, = named_variables.keys()
    with ops.name_scope("root/" + checkpoint_name):
      pass  # Make sure we can use this as an op name if we prefix it.
    return checkpoint_name

  def testAnonymousVarsInInit(self):

    class Model(training.Model):

      def __init__(self):
        super(Model, self).__init__()
        self.w = resource_variable_ops.ResourceVariable(0.0)
        self.b = resource_variable_ops.ResourceVariable(0.0)
        self.vars = [self.w, self.b]

      def call(self, x):
        return x * self.w + self.b

    with context.eager_mode():
      model = Model()
      optimizer = adam.AdamOptimizer(learning_rate=0.05)
      checkpoint_directory = self.get_temp_dir()
      checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
      checkpoint = checkpointable_utils.Checkpoint(
          model=model, optimizer=optimizer)
      for _ in range(2):
        checkpoint.save(checkpoint_prefix)
        with backprop.GradientTape() as tape:
          loss = (constant_op.constant(1.)
                  - model(constant_op.constant(1.))) ** 2
        grad = tape.gradient(loss, model.vars)
        optimizer.apply_gradients(
            [(g, v) for g, v in zip(grad, model.vars)])

  @test_util.run_in_graph_and_eager_modes()
  def testDeferredSlotRestoration(self):
    checkpoint_directory = self.get_temp_dir()

    root = checkpointable.Checkpointable()
    root.var = checkpointable_utils.add_variable(
        root, name="var", initializer=0.)
    optimizer = adam.AdamOptimizer(0.1)
    if context.executing_eagerly():
      optimizer.minimize(root.var.read_value)
    else:
      train_op = optimizer.minimize(root.var)
      # Note that `optimizer` has not been added as a dependency of
      # `root`. Create a one-off grouping so that slot variables for `root.var`
      # get initialized too.
      self.evaluate(checkpointable_utils.gather_initializers(
          checkpointable_utils.Checkpoint(root=root, optimizer=optimizer)))
      self.evaluate(train_op)
    self.evaluate(state_ops.assign(root.var, 12.))
    no_slots_path = checkpointable_utils.CheckpointableSaver(root).save(
        os.path.join(checkpoint_directory, "no_slots"))
    root.optimizer = optimizer
    self.evaluate(state_ops.assign(root.var, 13.))
    self.evaluate(state_ops.assign(optimizer.get_slot(name="m", var=root.var),
                                   14.))
    slots_path = checkpointable_utils.CheckpointableSaver(root).save(
        os.path.join(checkpoint_directory, "with_slots"))
    new_root = checkpointable.Checkpointable()
    # Load the slot-containing checkpoint (deferred), then immediately overwrite
    # the non-slot variable (also deferred).
    slot_status = checkpointable_utils.CheckpointableSaver(
        new_root).restore(slots_path)
    no_slot_status = checkpointable_utils.CheckpointableSaver(
        new_root).restore(no_slots_path)
    with self.assertRaises(AssertionError):
      no_slot_status.assert_consumed()
    new_root.var = checkpointable_utils.add_variable(
        new_root, name="var", shape=[])
    no_slot_status.assert_consumed()
    no_slot_status.run_restore_ops()
    self.assertEqual(12., self.evaluate(new_root.var))
    new_root.optimizer = adam.AdamOptimizer(0.1)
    with self.assertRaisesRegexp(AssertionError, "beta1_power"):
      slot_status.assert_consumed()
    self.assertEqual(12., self.evaluate(new_root.var))
    if context.executing_eagerly():
      # Slot variables are only created with restoring initializers when
      # executing eagerly.
      self.assertEqual(14., self.evaluate(
          new_root.optimizer.get_slot(name="m", var=new_root.var)))
    else:
      self.assertIs(new_root.optimizer.get_slot(name="m", var=new_root.var),
                    None)
    if context.executing_eagerly():
      new_root.optimizer.minimize(new_root.var.read_value)
    else:
      train_op = new_root.optimizer.minimize(new_root.var)
      # The slot variable now exists; restore() didn't create it, but we should
      # now have a restore op for it.
      slot_status.run_restore_ops()
      self.assertEqual(14., self.evaluate(
          new_root.optimizer.get_slot(name="m", var=new_root.var)))
      self.evaluate(train_op)
    slot_status.assert_consumed()

  def testManySavesGraph(self):
    """Saves after the first should not modify the graph."""
    with context.graph_mode():
      graph = ops.Graph()
      with graph.as_default(), self.test_session(graph):
        checkpoint_directory = self.get_temp_dir()
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        obj = checkpointable.Checkpointable()
        obj.var = variable_scope.get_variable(name="v", initializer=0.)
        obj.opt = adam.AdamOptimizer(0.1)
        obj.opt.minimize(obj.var.read_value())
        self.evaluate(checkpointable_utils.gather_initializers(obj))
        saver = checkpointable_utils.CheckpointableSaver(obj)
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
        obj.opt = adam.AdamOptimizer(0.1)
        obj.opt.minimize(obj.var.read_value())
        self.evaluate(checkpointable_utils.gather_initializers(obj))
        saver = checkpointable_utils.CheckpointableSaver(obj)
        save_path = saver.save(checkpoint_prefix)
        saver.restore(save_path)
        before_ops = graph.get_operations()
        saver.restore(save_path)
        self.assertEqual(before_ops, graph.get_operations())

  def testMultipleGraphsNonSlotVariables(self):
    with context.graph_mode():
      checkpoint_directory = self.get_temp_dir()
      checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
      optimizer = adam.AdamOptimizer(0.001)
      # Construct a model in one graph
      first_graph = ops.Graph()
      first_session = session_lib.Session(graph=first_graph)
      with first_graph.as_default(), first_session.as_default():
        first_variable = resource_variable_ops.ResourceVariable([1.])
        first_root_checkpointable = checkpointable_utils.Checkpoint(
            optimizer=optimizer, variable=first_variable)
        train_op = optimizer.minimize(first_variable.read_value)
        self.evaluate(checkpointable_utils.gather_initializers(
            first_root_checkpointable))
        self.evaluate(train_op)
        self.evaluate(first_variable.assign([1.]))
        self.evaluate(optimizer.get_slot(
            var=first_variable, name="m").assign([2.]))
        beta1_power, _ = optimizer._get_beta_accumulators()
        self.evaluate(beta1_power.assign(3.))

      # Save and load in a second graph
      second_graph = ops.Graph()
      with second_graph.as_default(), session_lib.Session(graph=second_graph):
        second_variable = resource_variable_ops.ResourceVariable([1.])
        second_root_checkpointable = checkpointable_utils.Checkpoint(
            optimizer=optimizer, variable=second_variable)
        train_op = optimizer.minimize(second_variable.read_value)
        second_root_checkpointable.restore(None).initialize_or_restore()
        self.evaluate(train_op)
        self.evaluate(second_variable.assign([4.]))
        self.evaluate(optimizer.get_slot(
            var=second_variable, name="m").assign([5.]))
        beta1_power, _ = optimizer._get_beta_accumulators()
        self.evaluate(beta1_power.assign(6.))
        save_path = second_root_checkpointable.save(checkpoint_prefix)
        self.evaluate(second_variable.assign([7.]))
        self.evaluate(optimizer.get_slot(
            var=second_variable, name="m").assign([8.]))
        beta1_power, _ = optimizer._get_beta_accumulators()
        self.assertAllEqual(6., self.evaluate(beta1_power))
        status = second_root_checkpointable.restore(save_path)
        status.assert_consumed().run_restore_ops()
        self.assertAllEqual([4.], self.evaluate(second_variable))
        self.assertAllEqual([5.], self.evaluate(optimizer.get_slot(
            var=second_variable, name="m")))
        beta1_power, _ = optimizer._get_beta_accumulators()
        self.assertAllEqual(6., self.evaluate(beta1_power))

      # Check that the first graph is unmolested
      with first_graph.as_default(), first_session.as_default():
        self.assertAllEqual([1.], self.evaluate(first_variable))
        self.assertAllEqual([2.], self.evaluate(optimizer.get_slot(
            var=first_variable, name="m")))
        beta1_power, _ = optimizer._get_beta_accumulators()
        self.assertAllEqual(3., self.evaluate(beta1_power))


class CheckpointCompatibilityTests(test.TestCase):

  def _initialized_model(self):
    input_value = constant_op.constant([[3.]])
    model = MyModel()
    optimizer = adam.AdamOptimizer(0.001)
    optimizer_step = training_util.get_or_create_global_step()
    root_checkpointable = checkpointable_utils.Checkpoint(
        optimizer=optimizer, model=model, optimizer_step=optimizer_step)
    train_op = optimizer.minimize(
        functools.partial(model, input_value),
        global_step=optimizer_step)
    self.evaluate(checkpointable_utils.gather_initializers(
        root_checkpointable))
    self.evaluate(train_op)
    # A regular variable, a slot variable, and a non-slot Optimizer variable
    # with known values to check when loading.
    self.evaluate(model._named_dense.bias.assign([1.]))
    self.evaluate(optimizer.get_slot(
        var=model._named_dense.bias, name="m").assign([2.]))
    beta1_power, _ = optimizer._get_beta_accumulators()
    self.evaluate(beta1_power.assign(3.))
    return root_checkpointable

  def _set_sentinels(self, root_checkpointable):
    self.evaluate(root_checkpointable.model._named_dense.bias.assign([101.]))
    self.evaluate(
        root_checkpointable.optimizer.get_slot(
            var=root_checkpointable.model._named_dense.bias, name="m")
        .assign([102.]))
    beta1_power, _ = root_checkpointable.optimizer._get_beta_accumulators()
    self.evaluate(beta1_power.assign(103.))

  def _check_sentinels(self, root_checkpointable):
    self.assertAllEqual(
        [1.], self.evaluate(root_checkpointable.model._named_dense.bias))
    self.assertAllEqual([2.], self.evaluate(
        root_checkpointable.optimizer.get_slot(
            var=root_checkpointable.model._named_dense.bias, name="m")))
    beta1_power, _ = root_checkpointable.optimizer._get_beta_accumulators()
    self.assertAllEqual(3., self.evaluate(beta1_power))

  def _write_name_based_checkpoint(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    with context.graph_mode():
      save_graph = ops.Graph()
      with save_graph.as_default(), self.test_session(
          graph=save_graph) as session:
        root = self._initialized_model()
        name_saver = core_saver.Saver()
        return name_saver.save(
            sess=session, save_path=checkpoint_prefix,
            global_step=root.optimizer_step)

  @test_util.run_in_graph_and_eager_modes()
  def testLoadFromNameBasedSaver(self):
    """Save a name-based checkpoint, load it using the object-based API."""
    with test_util.device(use_gpu=True):
      save_path = self._write_name_based_checkpoint()
      root = self._initialized_model()
      self._set_sentinels(root)
      with self.assertRaises(AssertionError):
        self._check_sentinels(root)
      object_saver = checkpointable_utils.CheckpointableSaver(root)
      status = object_saver.restore(save_path)
      with self.assertRaises(AssertionError):
        status.assert_consumed()
      status.run_restore_ops()
      self._check_sentinels(root)
      self._set_sentinels(root)
      status.initialize_or_restore()
      self._check_sentinels(root)

  # TODO(allenl): Test for the core name-based saver loading object-based
  # checkpoints once object-based checkpointing is in core.

  def testSaveGraphLoadEager(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    with context.graph_mode():
      save_graph = ops.Graph()
      with save_graph.as_default(), self.test_session(
          graph=save_graph) as session:
        root = self._initialized_model()
        object_saver = checkpointable_utils.CheckpointableSaver(root)
        save_path = object_saver.save(
            session=session, file_prefix=checkpoint_prefix)
    with context.eager_mode():
      root = self._initialized_model()
      self._set_sentinels(root)
      root.restore(save_path).assert_consumed()
      self._check_sentinels(root)

  def testSaveEagerLoadGraph(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    with context.eager_mode():
      root = self._initialized_model()
      object_saver = checkpointable_utils.CheckpointableSaver(root)
      save_path = object_saver.save(file_prefix=checkpoint_prefix)
    with context.graph_mode():
      save_graph = ops.Graph()
      with save_graph.as_default(), self.test_session(
          graph=save_graph):
        root = self._initialized_model()
        self._set_sentinels(root)
        root.restore(save_path).assert_consumed().run_restore_ops()
        self._check_sentinels(root)

if __name__ == "__main__":
  test.main()
