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

from absl.testing import parameterized
import six

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import template
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import training_util
from tensorflow.python.training.tracking import graph_view
from tensorflow.python.training.tracking import tracking
from tensorflow.python.training.tracking import util as trackable_utils


# pylint: disable=not-callable
class MyModel(training.Model):
  """A concrete Model for testing."""

  def __init__(self):
    super(MyModel, self).__init__()
    self._named_dense = core.Dense(1, use_bias=True)
    self._second = core.Dense(1, use_bias=False)
    # We can still track Trackables which aren't Layers.
    self._non_layer = NonLayerTrackable()

  def call(self, values):
    ret = self._second(self._named_dense(values))
    return ret


class NonLayerTrackable(tracking.AutoTrackable):

  def __init__(self):
    super(NonLayerTrackable, self).__init__()
    self.a_variable = trackable_utils.add_variable(
        self, name="a_variable", shape=[])


class InterfaceTests(test.TestCase):

  def testLayerDeduplication(self):
    model = training.Model()
    layer_one = core.Dense(1)
    layer_two = core.Dense(1)
    model.other_path = [layer_one, layer_two]
    model.l2 = layer_two
    model.l1 = layer_one
    self.assertEqual([layer_one, layer_two], model.layers)

  def testSaveWithOnlyKerasSession(self):

    with ops.Graph().as_default():
      inp = input_layer.Input([1])
      dense = core.Dense(1)(inp)
      model = training.Model(inp, dense)
      model.compile(optimizer="sgd", loss="mse")
      model.fit([1.], [2.])
      checkpoint = trackable_utils.Checkpoint(model=model)
      checkpoint.save(os.path.join(self.get_temp_dir(), "ckpt"))

  def testObjectMetadata(self):
    with context.eager_mode():
      checkpoint_directory = self.get_temp_dir()
      checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
      dense = core.Dense(1)
      checkpoint = trackable_utils.Checkpoint(dense=dense)
      dense(constant_op.constant([[1.]]))
      save_path = checkpoint.save(checkpoint_prefix)

    objects = trackable_utils.object_metadata(save_path)
    all_variable_names = []
    for obj in objects.nodes:
      for attribute in obj.attributes:
        all_variable_names.append(attribute.full_name)
    self.assertIn("dense/kernel", all_variable_names)


class CheckpointingTests(parameterized.TestCase, test.TestCase):

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testNamingWithOptimizer(self):
    input_value = constant_op.constant([[3.]])
    model = MyModel()
    # A nuisance Model using the same optimizer. Its slot variables should not
    # go in the checkpoint, since it is never depended on.
    other_model = MyModel()
    optimizer = adam.Adam(0.001)
    step = training_util.get_or_create_global_step()
    root_trackable = trackable_utils.Checkpoint(
        optimizer=optimizer, model=model, step=step)

    with backprop.GradientTape() as tape:
      loss = model(input_value)
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    train_op = control_flow_ops.group(
        optimizer.apply_gradients(zip(gradients, variables)),
        step.assign_add(1))

    with backprop.GradientTape() as tape:
      loss = other_model(input_value)
    variables = other_model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    self.evaluate(trackable_utils.gather_initializers(
        root_trackable))
    self.evaluate(train_op)
    named_variables, serialized_graph, _ = graph_view.ObjectGraphView(
        root_trackable).serialize_object_graph()
    expected_slot_keys = (
        "model/_second/kernel/.OPTIMIZER_SLOT/optimizer/m",
        "model/_second/kernel/.OPTIMIZER_SLOT/optimizer/v",
        "model/_named_dense/kernel/.OPTIMIZER_SLOT/optimizer/m",
        "model/_named_dense/kernel/.OPTIMIZER_SLOT/optimizer/v",
        "model/_named_dense/bias/.OPTIMIZER_SLOT/optimizer/m",
        "model/_named_dense/bias/.OPTIMIZER_SLOT/optimizer/v",
    )
    expected_checkpoint_names = (
        # Created in the root node, so no prefix.
        "step",
        "model/_second/kernel",
        "model/_named_dense/kernel",
        "model/_named_dense/bias",
        # non-Layer dependency of the model
        "model/_non_layer/a_variable",
        "optimizer/learning_rate",
        "optimizer/beta_1",
        "optimizer/beta_2",
        "optimizer/iter",
        "optimizer/decay",
    ) + expected_slot_keys
    suffix = "/.ATTRIBUTES/VARIABLE_VALUE"
    expected_checkpoint_names = [
        name + suffix for name in expected_checkpoint_names]
    named_variables = {v.name: v for v in named_variables}
    six.assertCountEqual(self, expected_checkpoint_names,
                         named_variables.keys())
    # Check that we've mapped to the right variable objects (not exhaustive)
    self.assertEqual(
        "global_step",
        named_variables["step" + suffix].full_name)
    self.assertEqual(
        "my_model/dense_1/kernel",
        named_variables["model/_second/kernel" + suffix].full_name)
    self.assertEqual(
        "my_model/dense/kernel",
        named_variables["model/_named_dense/kernel" + suffix].full_name)
    self.assertEqual("Adam/beta_1",
                     named_variables["optimizer/beta_1" + suffix].full_name)
    self.assertEqual("Adam/beta_2",
                     named_variables["optimizer/beta_2" + suffix].full_name)
    # Spot check the generated protocol buffers.
    self.assertEqual("optimizer",
                     serialized_graph.nodes[0].children[1].local_name)
    optimizer_node = serialized_graph.nodes[
        serialized_graph.nodes[0].children[1].node_id]
    children = [node.local_name for node in optimizer_node.children]
    six.assertCountEqual(
        self,
        # hyper variable dependencies
        ["beta_1", "beta_2", "iter", "decay", "learning_rate"],
        children)
    serialized_slot_keys = []
    for slot in optimizer_node.slot_variables:
      for attribute in (
          serialized_graph.nodes[slot.slot_variable_node_id].attributes):
        serialized_slot_keys.append(attribute.checkpoint_key)
    six.assertCountEqual(
        self,
        [key + suffix for key in expected_slot_keys],
        serialized_slot_keys)

  @test_util.run_in_graph_and_eager_modes
  def testSaveRestore(self):
    model = MyModel()
    optimizer = adam.Adam(0.001)
    root_trackable = trackable_utils.Checkpoint(
        optimizer=optimizer, model=model)
    input_value = constant_op.constant([[3.]])
    with backprop.GradientTape() as tape:
      loss = model(input_value)
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    train_op = optimizer.apply_gradients(zip(gradients, variables))
    self.assertFalse(root_trackable.save_counter.trainable)
    self.evaluate(trackable_utils.gather_initializers(
        root_trackable))
    self.evaluate(train_op)
    prefix = os.path.join(self.get_temp_dir(), "ckpt")
    self.evaluate(state_ops.assign(model._named_dense.variables[1], [42.]))
    m_bias_slot = optimizer.get_slot(model._named_dense.variables[1], "m")
    self.evaluate(state_ops.assign(m_bias_slot, [1.5]))
    save_path = root_trackable.save(file_prefix=prefix)
    self.evaluate(state_ops.assign(model._named_dense.variables[1], [43.]))
    self.evaluate(state_ops.assign(root_trackable.save_counter, 3))
    optimizer_variables = self.evaluate(
        sorted(optimizer.variables(), key=lambda v: v.name))
    self.evaluate(state_ops.assign(m_bias_slot, [-2.]))
    # Immediate restoration
    status = root_trackable.restore(save_path=save_path).assert_consumed()
    status.run_restore_ops()
    self.assertAllEqual([42.], self.evaluate(model._named_dense.variables[1]))
    self.assertAllEqual(1, self.evaluate(root_trackable.save_counter))
    self.assertAllEqual([1.5], self.evaluate(m_bias_slot))
    if not context.executing_eagerly():
      return  # Restore-on-create is only supported when executing eagerly
    on_create_model = MyModel()
    on_create_optimizer = adam.Adam(0.001)
    on_create_root = trackable_utils.Checkpoint(
        optimizer=on_create_optimizer, model=on_create_model)
    # Deferred restoration
    status = on_create_root.restore(save_path=save_path)
    status.assert_nontrivial_match()
    status.assert_existing_objects_matched()
    with self.assertRaises(AssertionError):
      status.assert_consumed()
    on_create_model(constant_op.constant([[3.]]))  # create variables
    self.assertAllEqual(1, self.evaluate(on_create_root.save_counter))
    self.assertAllEqual([42.],
                        self.evaluate(
                            on_create_model._named_dense.variables[1]))
    on_create_m_bias_slot = on_create_optimizer.get_slot(
        on_create_model._named_dense.variables[1], "m")
    status.assert_existing_objects_matched()
    if not context.executing_eagerly():
      with self.assertRaises(AssertionError):
        status.assert_consumed()
    # Optimizer slot variables are created when the original variable is
    # restored.
    self.assertAllEqual([1.5], self.evaluate(on_create_m_bias_slot))
    dummy_var = resource_variable_ops.ResourceVariable([1.])
    on_create_optimizer.minimize(loss=dummy_var.read_value,
                                 var_list=[dummy_var])
    status.assert_existing_objects_matched()
    status.assert_consumed()
    self.assertAllEqual(
        optimizer_variables,
        # Creation order is different, so .variables() needs to be re-sorted.
        self.evaluate(sorted(optimizer.variables(), key=lambda v: v.name)))

  # TODO(allenl): Debug garbage created by this test in python3.
  def testDeferredRestorationUsageEager(self):
    """An idiomatic eager execution example."""
    num_training_steps = 10
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    for training_continuation in range(3):
      model = MyModel()
      optimizer = adam.Adam(0.001)
      root = trackable_utils.Checkpoint(
          optimizer=optimizer, model=model)
      root.restore(checkpoint_management.latest_checkpoint(
          checkpoint_directory))
      for _ in range(num_training_steps):
        # TODO(allenl): Use a Dataset and serialize/checkpoint it.
        input_value = constant_op.constant([[3.]])
        with backprop.GradientTape() as tape:
          loss = model(input_value)
        variables = model.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
      root.save(file_prefix=checkpoint_prefix)
      self.assertEqual((training_continuation + 1) * num_training_steps,
                       root.optimizer.iterations.numpy())

  def testUsageGraph(self):
    """Expected usage when graph building."""
    with context.graph_mode():
      num_training_steps = 10
      checkpoint_directory = self.get_temp_dir()
      checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
      for training_continuation in range(3):
        with ops.Graph().as_default():
          model = MyModel()
          optimizer = adam.Adam(0.001)
          root = trackable_utils.CheckpointV1(
              optimizer=optimizer, model=model)
          input_value = constant_op.constant([[3.]])
          with backprop.GradientTape() as tape:
            loss = model(input_value)
          variables = model.trainable_variables
          gradients = tape.gradient(loss, variables)
          train_op = optimizer.apply_gradients(zip(gradients, variables))

          checkpoint_path = checkpoint_management.latest_checkpoint(
              checkpoint_directory)
          with self.session(graph=ops.get_default_graph()) as session:
            status = root.restore(save_path=checkpoint_path)
            status.initialize_or_restore(session=session)
            if checkpoint_path is None:
              self.assertEqual(0, training_continuation)
              with self.assertRaises(AssertionError):
                status.assert_consumed()
              with self.assertRaises(AssertionError):
                status.assert_existing_objects_matched()
            else:
              status.assert_consumed()
              status.assert_existing_objects_matched()
            for _ in range(num_training_steps):
              session.run(train_op)
            root.save(file_prefix=checkpoint_prefix, session=session)
            self.assertEqual((training_continuation + 1) * num_training_steps,
                             session.run(root.optimizer.iterations))
            self.assertEqual(training_continuation + 1,
                             session.run(root.save_counter))

  @test_util.run_in_graph_and_eager_modes
  def testAgnosticUsage(self):
    """Graph/eager agnostic usage."""
    # Does create garbage when executing eagerly due to ops.Graph() creation.
    num_training_steps = 10
    checkpoint_directory = self.get_temp_dir()
    def _train_fn(model, input_value):
      with backprop.GradientTape() as tape:
        loss = model(input_value)
      variables = model.trainable_variables
      gradients = tape.gradient(loss, variables)
      return optimizer.apply_gradients(zip(gradients, variables))
    for training_continuation in range(3):
      with test_util.device(use_gpu=True):
        model = MyModel()
        optimizer = adam.Adam(0.001)
        root = trackable_utils.Checkpoint(
            optimizer=optimizer, model=model)
        manager = checkpoint_management.CheckpointManager(
            root, checkpoint_directory, max_to_keep=1)
        status = root.restore(save_path=manager.latest_checkpoint)
        input_value = constant_op.constant([[3.]])
        train_fn = functools.partial(_train_fn, model, input_value)
        if not context.executing_eagerly():
          train_fn = functools.partial(self.evaluate, train_fn())
        status.initialize_or_restore()
        for _ in range(num_training_steps):
          train_fn()
        manager.save()
        self.assertEqual((training_continuation + 1) * num_training_steps,
                         self.evaluate(root.optimizer.iterations))
        self.assertEqual(training_continuation + 1,
                         self.evaluate(root.save_counter))

  def testPartialRestoreWarningObject(self):
    with context.eager_mode():
      optimizer = adam.Adam(0.0)
      original_root = trackable_utils.Checkpoint(v1=variables_lib.Variable(2.),
                                                 v2=variables_lib.Variable(3.),
                                                 optimizer=optimizer)
      # Create a slot variable to save
      optimizer.minimize(original_root.v1.read_value, [original_root.v1])
      prefix = os.path.join(self.get_temp_dir(), "ckpt")
      save_path = original_root.save(prefix)
      partial_root = trackable_utils.Checkpoint(v1=variables_lib.Variable(0.))
      weak_partial_root = weakref.ref(partial_root)
      weak_v1 = weakref.ref(partial_root.v1)
      partial_root.restore(save_path)
      self.assertEqual(2., partial_root.v1.numpy())
      with test.mock.patch.object(logging, "warning") as mock_log:
        del partial_root
        self.assertIsNone(weak_partial_root())
        self.assertIsNone(weak_v1())
        messages = str(mock_log.call_args_list)
      self.assertIn("(root).v2'", messages)
      self.assertIn("(root).optimizer's state 'm' for (root).v1", messages)
      self.assertNotIn("(root).v1'", messages)
      self.assertIn("expect_partial()", messages)

  # pylint: disable=cell-var-from-loop
  @test_util.run_in_graph_and_eager_modes
  @test_util.run_v1_only("b/120545219")
  def testWithDefun(self):
    num_training_steps = 2
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    for training_continuation in range(3):
      with test_util.device(use_gpu=True):
        model = MyModel()
        # Don't actually train so we can test variable values
        optimizer = adam.Adam(0.)
        root = trackable_utils.Checkpoint(
            optimizer=optimizer, model=model)
        checkpoint_path = checkpoint_management.latest_checkpoint(
            checkpoint_directory)
        status = root.restore(save_path=checkpoint_path)
        def train_fn():
          @def_function.function
          def _call_model(x):
            return model(x)
          with backprop.GradientTape() as tape:
            loss = _call_model(constant_op.constant([[3.]]))
          gradients = tape.gradient(loss, model.variables)
          return optimizer.apply_gradients(zip(gradients, model.variables))
        if not context.executing_eagerly():
          train_fn = functools.partial(
              self.evaluate, train_fn())
        status.initialize_or_restore()
        for _ in range(num_training_steps):
          train_fn()
        if training_continuation > 0:
          status.assert_consumed()
          self.assertAllClose([[42.]], self.evaluate(model.variables[0]))
        else:
          self.evaluate(model.variables[0].assign([[42.]]))
        root.save(file_prefix=checkpoint_prefix)
        self.assertEqual((training_continuation + 1) * num_training_steps,
                         self.evaluate(optimizer.iterations))
        self.assertEqual(training_continuation + 1,
                         self.evaluate(root.save_counter))
  # pylint: enable=cell-var-from-loop

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
      optimizer = adam.Adam(learning_rate=0.05)
      checkpoint_directory = self.get_temp_dir()
      checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
      checkpoint = trackable_utils.Checkpoint(
          model=model, optimizer=optimizer)
      for _ in range(2):
        checkpoint.save(checkpoint_prefix)
        with backprop.GradientTape() as tape:
          loss = (constant_op.constant(1.)
                  - model(constant_op.constant(1.))) ** 2
        grad = tape.gradient(loss, model.vars)
        optimizer.apply_gradients(
            [(g, v) for g, v in zip(grad, model.vars)])

  @test_util.run_in_graph_and_eager_modes
  def testDeferredSlotRestoration(self):
    checkpoint_directory = self.get_temp_dir()

    root = trackable_utils.Checkpoint()
    root.var = trackable_utils.add_variable(
        root, name="var", initializer=0.)
    optimizer = adam.Adam(0.1)
    variables = [root.var]
    gradients = [1.]
    train_op = optimizer.apply_gradients(zip(gradients, variables))
    # Note that `optimizer` has not been added as a dependency of
    # `root`. Create a one-off grouping so that slot variables for `root.var`
    # get initialized too.
    self.evaluate(trackable_utils.gather_initializers(
        trackable_utils.Checkpoint(root=root, optimizer=optimizer)))
    self.evaluate(train_op)
    self.evaluate(state_ops.assign(root.var, 12.))
    no_slots_path = root.save(os.path.join(checkpoint_directory, "no_slots"))
    root.optimizer = optimizer
    self.evaluate(state_ops.assign(root.var, 13.))
    self.evaluate(state_ops.assign(
        optimizer.get_slot(slot_name="m", var=root.var),
        14.))
    slots_path = root.save(os.path.join(checkpoint_directory, "with_slots"))
    new_root = trackable_utils.Checkpoint()
    # Load the slot-containing checkpoint (deferred), then immediately overwrite
    # the non-slot variable (also deferred).
    slot_status = new_root.restore(slots_path)
    no_slot_status = new_root.restore(no_slots_path)
    with self.assertRaises(AssertionError):
      no_slot_status.assert_consumed()
    new_root.var = trackable_utils.add_variable(
        new_root, name="var", shape=[])
    no_slot_status.assert_consumed()
    no_slot_status.run_restore_ops()
    self.assertEqual(12., self.evaluate(new_root.var))
    new_root.optimizer = adam.Adam(0.1)
    slot_status.assert_existing_objects_matched()
    if not context.executing_eagerly():
      with self.assertRaisesRegex(AssertionError, "Unresolved object"):
        slot_status.assert_consumed()
    self.assertEqual(12., self.evaluate(new_root.var))
    if context.executing_eagerly():
      # Slot variables are only created with restoring initializers when
      # executing eagerly.
      self.assertEqual(14., self.evaluate(
          new_root.optimizer.get_slot(slot_name="m", var=new_root.var)))
    else:
      # Slot variables are not created eagerly when graph building.
      with self.assertRaises(KeyError):
        new_root.optimizer.get_slot(slot_name="m", var=new_root.var)
    variables = [new_root.var]
    gradients = [1.]
    train_op = new_root.optimizer.apply_gradients(zip(gradients, variables))
    # The slot variable now exists; restore() didn't create it, but we should
    # now have a restore op for it.
    slot_status.run_restore_ops()
    if not context.executing_eagerly():
      # The train op hasn't run when graph building, so the slot variable has
      # its restored value. It has run in eager, so the value will be different.
      self.assertEqual(14., self.evaluate(
          new_root.optimizer.get_slot(slot_name="m", var=new_root.var)))
    self.evaluate(train_op)
    slot_status.assert_consumed()

  def testManySavesGraph(self):
    """Saves after the first should not modify the graph."""
    with context.graph_mode():
      graph = ops.Graph()
      with graph.as_default(), self.session(graph):
        checkpoint_directory = self.get_temp_dir()
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        obj = trackable_utils.Checkpoint()
        obj.var = variables_lib.Variable(0., name="v")
        obj.opt = adam.Adam(0.1)
        variables = [obj.var]
        gradients = [1.]
        obj.opt.apply_gradients(zip(gradients, variables))
        self.evaluate(trackable_utils.gather_initializers(obj))
        obj.save(checkpoint_prefix)
        graph.finalize()
        obj.save(checkpoint_prefix)

  def testManyRestoresGraph(self):
    """Restores after the first should not modify the graph."""
    with context.graph_mode():
      graph = ops.Graph()
      with graph.as_default(), self.session(graph):
        checkpoint_directory = self.get_temp_dir()
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        obj = trackable_utils.Checkpoint()
        obj.var = variables_lib.Variable(0., name="v")
        obj.opt = adam.Adam(0.1)
        variables = [obj.var]
        gradients = [1.]
        obj.opt.apply_gradients(zip(gradients, variables))
        self.evaluate(trackable_utils.gather_initializers(obj))
        save_path = obj.save(checkpoint_prefix)
        obj.restore(save_path)
        graph.finalize()
        obj.restore(save_path)

  @test_util.run_in_graph_and_eager_modes
  def test_sequential(self):
    model = sequential.Sequential()
    checkpoint = trackable_utils.Checkpoint(model=model)
    model.add(core.Dense(4))
    second_dense = core.Dense(5)
    model.add(second_dense)
    model(constant_op.constant([[1.]]))
    checkpoint.restore(None).initialize_or_restore()
    self.evaluate(second_dense.bias.assign(
        constant_op.constant([1., 2., 3., 4., 5.])))
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    save_path = checkpoint.save(checkpoint_prefix)
    self.evaluate(second_dense.bias.assign(
        constant_op.constant([5., 6., 7., 8., 9.])))
    checkpoint.restore(save_path).assert_consumed().run_restore_ops()
    self.assertAllEqual([1., 2., 3., 4., 5.], self.evaluate(second_dense.bias))

    deferred_sequential = sequential.Sequential()
    deferred_sequential_checkpoint = trackable_utils.Checkpoint(
        model=deferred_sequential)
    status = deferred_sequential_checkpoint.restore(save_path)
    deferred_sequential.add(core.Dense(4))
    deferred_second_dense = core.Dense(5)
    deferred_sequential.add(deferred_second_dense)
    deferred_sequential(constant_op.constant([[1.]]))
    status.run_restore_ops()
    self.assertAllEqual([1., 2., 3., 4., 5.],
                        self.evaluate(deferred_second_dense.bias))

  @test_util.run_in_graph_and_eager_modes
  def test_initialize_if_not_restoring(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    optimizer_only_prefix = os.path.join(checkpoint_directory, "opt")
    with test_util.device(use_gpu=True):
      model = MyModel()
      optimizer = adam.Adam(0.001)
      root = trackable_utils.Checkpoint(
          model=model)  # Do not save the optimizer with the checkpoint.
      optimizer_checkpoint = trackable_utils.Checkpoint(
          optimizer=optimizer)

      checkpoint_path = checkpoint_management.latest_checkpoint(
          checkpoint_directory)
      status = root.restore(save_path=checkpoint_path)
      input_value = constant_op.constant([[3.]])
      def train_fn():
        with backprop.GradientTape() as tape:
          loss = model(input_value)
        variables = model.trainable_variables
        gradients = tape.gradient(loss, variables)
        return optimizer.apply_gradients(zip(gradients, variables))
      if not context.executing_eagerly():
        train_fn = functools.partial(self.evaluate, train_fn())
      status.initialize_or_restore()
      # TODO(tanzheny): Add hyper variables to .variables(), and set them with
      # set_weights etc.
      variables_not_in_the_variables_property = [
          obj for obj in optimizer._hyper.values()
          if isinstance(obj, variables_lib.Variable)]
      self.evaluate([v.initializer for v
                     in optimizer.variables()
                     + variables_not_in_the_variables_property])
      train_fn()
      model_save_path = root.save(file_prefix=checkpoint_prefix)
      self.evaluate(optimizer.beta_1.assign(42.))
      optimizer_save_path = optimizer_checkpoint.save(optimizer_only_prefix)
    del train_fn

    # Restore into a graph with the optimizer
    with test_util.device(use_gpu=True):
      model = MyModel()
      optimizer = adam.Adam(0.001)
      root = trackable_utils.Checkpoint(
          optimizer=optimizer, model=model)
      status = root.restore(save_path=model_save_path)
      input_value = constant_op.constant([[3.]])
      def train_fn1():
        with backprop.GradientTape() as tape:
          loss = model(input_value)
        variables = model.trainable_variables
        gradients = tape.gradient(loss, variables)
        return optimizer.apply_gradients(zip(gradients, variables))
      if not context.executing_eagerly():
        train_fn1 = functools.partial(self.evaluate, train_fn1())
      status.initialize_or_restore()
      train_fn1()
      with self.assertRaises(AssertionError):
        status.assert_existing_objects_matched()
      with self.assertRaises(AssertionError):
        status.assert_consumed()
    del train_fn1

    # Make sure initialization doesn't clobber later restores
    with test_util.device(use_gpu=True):
      model = MyModel()
      optimizer = adam.Adam(0.001, beta_1=1.0)
      root = trackable_utils.Checkpoint(
          optimizer=optimizer, model=model)
      opt_root = trackable_utils.Checkpoint(
          optimizer=optimizer)
      status = root.restore(save_path=model_save_path)
      init_only_optimizer_status = opt_root.restore(save_path=None)
      optimizer_status = opt_root.restore(save_path=optimizer_save_path)
      input_value = constant_op.constant([[3.]])
      def train_fn2():
        with backprop.GradientTape() as tape:
          loss = model(input_value)
        variables = model.trainable_variables
        gradients = tape.gradient(loss, variables)
        return optimizer.apply_gradients(zip(gradients, variables))
      if not context.executing_eagerly():
        train_fn2 = functools.partial(self.evaluate, train_fn2())
      optimizer_status.run_restore_ops()
      status.initialize_or_restore()
      init_only_optimizer_status.initialize_or_restore()
      train_fn2()
      self.assertEqual(42., self.evaluate(optimizer.beta_1))


class _ManualScope(tracking.AutoTrackable):

  def __call__(self):
    with variable_scope.variable_scope("ManualScope") as vs:
      self.variable_scope = vs
      with trackable_utils.capture_dependencies(template=self):
        return self._build()

  def _build(self):
    return variable_scope.get_variable(name="in_manual_scope", shape=[])


class TemplateTests(parameterized.TestCase, test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_trackable_save_restore(self):

    def _templated():
      v = variable_scope.get_variable(
          "v", shape=[1], initializer=init_ops.zeros_initializer(),
          use_resource=True)
      v2 = variable_scope.get_variable(
          "v2", shape=[1], initializer=init_ops.zeros_initializer(),
          use_resource=True)
      manual = _ManualScope()
      return v, v + 1., v2, manual, manual()

    save_template = template.make_template("s1", _templated)
    v1_save, _, v2_save, manual_scope, manual_scope_v = save_template()
    six.assertCountEqual(
        self,
        [id(v1_save), id(v2_save), id(manual_scope),
         id(manual_scope_v), id(save_template)],
        map(id, trackable_utils.list_objects(save_template)))
    manual_dep, = manual_scope._checkpoint_dependencies
    self.assertEqual("in_manual_scope", manual_dep.name)
    self.assertIs(manual_scope_v, manual_dep.ref)
    optimizer = adam.Adam(0.0)
    save_root = trackable_utils.Checkpoint(
        my_template=save_template, optimizer=optimizer)
    optimizer.minimize(v1_save.read_value,
                       var_list=[v1_save])
    self.evaluate([v.initializer for v in save_template.variables])
    optimizer_variables = optimizer.variables() + list(
        optimizer._hyper.values())
    self.evaluate([v.initializer for v in optimizer_variables])
    self.evaluate(v1_save.assign([12.]))
    self.evaluate(v2_save.assign([14.]))
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    save_path = save_root.save(checkpoint_prefix)

    load_template = template.make_template("s2", _templated)
    load_optimizer = adam.Adam(0.0)
    load_root = trackable_utils.Checkpoint(
        my_template=load_template, optimizer=load_optimizer)
    status = load_root.restore(save_path)
    var, var_plus_one, var2, _, _ = load_template()
    load_optimizer.minimize(var.read_value, var_list=[var])
    self.assertLen(load_template._checkpoint_dependencies, 3)
    self.assertEqual("v", load_template._checkpoint_dependencies[0].name)
    self.assertEqual("v2", load_template._checkpoint_dependencies[1].name)
    self.assertEqual("ManualScope",
                     load_template._checkpoint_dependencies[2].name)
    status.assert_consumed().run_restore_ops()
    self.assertAllEqual([12.], self.evaluate(var))
    self.assertAllEqual([13.], self.evaluate(var_plus_one))
    self.assertAllEqual([14.], self.evaluate(var2))


class CheckpointCompatibilityTests(test.TestCase):

  def _initialized_model(self):
    input_value = constant_op.constant([[3.]])
    model = MyModel()
    optimizer = adam.Adam(0.001)
    root_trackable = trackable_utils.Checkpoint(
        optimizer=optimizer, model=model)
    with backprop.GradientTape() as tape:
      loss = model(input_value)
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    train_op = optimizer.apply_gradients(zip(gradients, variables))
    self.evaluate(trackable_utils.gather_initializers(
        root_trackable))
    self.evaluate(train_op)
    # A regular variable, a slot variable, and a non-slot Optimizer variable
    # with known values to check when loading.
    self.evaluate(model._named_dense.bias.assign([1.]))
    self.evaluate(optimizer.get_slot(
        var=model._named_dense.bias, slot_name="m").assign([2.]))
    self.evaluate(optimizer.beta_1.assign(3.))
    return root_trackable

  def _set_sentinels(self, root_trackable):
    self.evaluate(root_trackable.model._named_dense.bias.assign([101.]))
    self.evaluate(
        root_trackable.optimizer.get_slot(
            var=root_trackable.model._named_dense.bias, slot_name="m")
        .assign([102.]))
    self.evaluate(root_trackable.optimizer.beta_1.assign(103.))

  def _check_sentinels(self, root_trackable):
    self.assertAllEqual(
        [1.], self.evaluate(root_trackable.model._named_dense.bias))
    self.assertAllEqual([2.], self.evaluate(
        root_trackable.optimizer.get_slot(
            var=root_trackable.model._named_dense.bias, slot_name="m")))
    self.assertAllEqual(3.,
                        self.evaluate(root_trackable.optimizer.beta_1))

  def _write_name_based_checkpoint(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    with context.graph_mode():
      save_graph = ops.Graph()
      with save_graph.as_default(), self.session(
          graph=save_graph) as session:
        root = self._initialized_model()
        name_saver = saver_lib.Saver()
        return name_saver.save(
            sess=session,
            save_path=checkpoint_prefix,
            global_step=root.optimizer.iterations)

  @test_util.run_in_graph_and_eager_modes
  def testLoadFromNameBasedSaver(self):
    """Save a name-based checkpoint, load it using the object-based API."""
    with test_util.device(use_gpu=True):
      save_path = self._write_name_based_checkpoint()
      root = self._initialized_model()
      self._set_sentinels(root)
      with self.assertRaises(AssertionError):
        self._check_sentinels(root)
      object_saver = trackable_utils.TrackableSaver(
          graph_view.ObjectGraphView(root))
      self._set_sentinels(root)
      status = object_saver.restore(save_path)
      if context.executing_eagerly():
        self._check_sentinels(root)
      if context.executing_eagerly():
        status.assert_consumed()
        status.assert_existing_objects_matched()
        status.assert_nontrivial_match()
      else:
        # When graph building, we haven't read any keys, so we don't know
        # whether the restore will be complete.
        with self.assertRaisesRegex(AssertionError, "not restored"):
          status.assert_consumed()
        with self.assertRaisesRegex(AssertionError, "not restored"):
          status.assert_existing_objects_matched()
        with self.assertRaisesRegex(AssertionError, "not restored"):
          status.assert_nontrivial_match()
      status.run_restore_ops()
      self._check_sentinels(root)
      self._set_sentinels(root)
      status = object_saver.restore(save_path)
      status.initialize_or_restore()
      status.assert_nontrivial_match()
      self._check_sentinels(root)
      # Check that there is no error when keys are missing from the name-based
      # checkpoint.
      root.not_in_name_checkpoint = resource_variable_ops.ResourceVariable([1.])
      status = object_saver.restore(save_path)
      with self.assertRaises(AssertionError):
        status.assert_existing_objects_matched()

  def testSaveGraphLoadEager(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    with context.graph_mode():
      save_graph = ops.Graph()
      with save_graph.as_default(), self.session(
          graph=save_graph):
        root = self._initialized_model()
        save_path = root.save(file_prefix=checkpoint_prefix)
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
      save_path = root.save(file_prefix=checkpoint_prefix)
    with context.graph_mode():
      save_graph = ops.Graph()
      with save_graph.as_default(), self.session(
          graph=save_graph):
        root = self._initialized_model()
        self._set_sentinels(root)
        root.restore(save_path).assert_consumed().run_restore_ops()
        self._check_sentinels(root)

  def testIgnoreSaveCounter(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    with self.cached_session() as session:
      # Create and save a model using Saver() before using a Checkpoint. This
      # generates a snapshot without the Checkpoint's `save_counter`.
      model = sequential.Sequential()
      model.add(core.Flatten(input_shape=(1,)))
      model.add(core.Dense(1))
      name_saver = saver_lib.Saver(model.trainable_variables)
      save_path = name_saver.save(
          sess=session, save_path=checkpoint_prefix, global_step=1)
      # Checkpoint.restore must successfully load that checkpoint.
      ckpt = trackable_utils.Checkpoint(model=model)
      status = ckpt.restore(save_path)
      status.assert_existing_objects_matched()
      # It should, however, refuse to load a checkpoint where an unrelated
      # `save_counter` variable is missing.
      model.layers[1].var = variables_lib.Variable(0., name="save_counter")
      status = ckpt.restore(save_path)
      with self.assertRaises(AssertionError):
        status.assert_existing_objects_matched()


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
