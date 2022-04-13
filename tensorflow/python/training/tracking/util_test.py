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
import copy
import os
import pathlib
import sys
import weakref

from absl.testing import parameterized
import six

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import template
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import save as saved_model_save
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training.saving import checkpoint_options
from tensorflow.python.training.tracking import base
from tensorflow.python.training.tracking import graph_view
from tensorflow.python.training.tracking import tracking
from tensorflow.python.training.tracking import util as trackable_utils


class NonLayerTrackable(tracking.AutoTrackable):

  def __init__(self):
    super(NonLayerTrackable, self).__init__()
    self.a_variable = trackable_utils.add_variable(
        self, name="a_variable", shape=[])


class InterfaceTests(test.TestCase):

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testAddVariable(self):
    obj = NonLayerTrackable()
    with self.assertRaisesRegex(ValueError, "do not specify shape"):
      trackable_utils.add_variable(
          obj, name="shape_specified_twice", shape=[], initializer=1)
    constant_initializer = trackable_utils.add_variable(
        obj, name="constant_initializer", initializer=1)
    with variable_scope.variable_scope("some_variable_scope"):
      ones_initializer = trackable_utils.add_variable(
          obj,
          name="ones_initializer",
          shape=[2],
          initializer=init_ops.ones_initializer(dtype=dtypes.float32))
    bare_initializer = trackable_utils.add_variable(
        obj,
        name="bare_initializer",
        shape=[2, 2],
        dtype=dtypes.float64,
        initializer=init_ops.zeros_initializer)

    # Even in graph mode, there are no naming conflicts between objects, only
    # naming conflicts within an object.
    other_duplicate = resource_variable_ops.ResourceVariable(
        name="duplicate", initial_value=1.)
    duplicate = trackable_utils.add_variable(
        obj, name="duplicate", shape=[])
    with self.assertRaisesRegex(ValueError, "'duplicate'.*already declared"):
      trackable_utils.add_variable(obj, name="duplicate", shape=[])

    self.evaluate(trackable_utils.gather_initializers(obj))
    self.assertEqual("constant_initializer:0", constant_initializer.name)
    self.assertEqual(1, self.evaluate(constant_initializer))
    self.assertEqual("some_variable_scope/ones_initializer:0",
                     ones_initializer.name)
    self.assertAllEqual([1, 1], self.evaluate(ones_initializer))
    self.assertAllEqual([[0., 0.],
                         [0., 0.]], self.evaluate(bare_initializer))
    self.assertEqual("a_variable:0", obj.a_variable.name)
    self.assertEqual("duplicate:0", other_duplicate.name)
    if context.executing_eagerly():
      # When executing eagerly, there's no uniquification of variable names. The
      # checkpoint name will be the same.
      self.assertEqual("duplicate:0", duplicate.name)
    else:
      # The .name attribute may be globally influenced, but the checkpoint name
      # won't be (tested below).
      self.assertEqual("duplicate_1:0", duplicate.name)
    named_variables, _, _ = (
        graph_view.ObjectGraphView(obj).serialize_object_graph())
    expected_checkpoint_names = (
        "a_variable/.ATTRIBUTES/VARIABLE_VALUE",
        "bare_initializer/.ATTRIBUTES/VARIABLE_VALUE",
        "constant_initializer/.ATTRIBUTES/VARIABLE_VALUE",
        "duplicate/.ATTRIBUTES/VARIABLE_VALUE",
        "ones_initializer/.ATTRIBUTES/VARIABLE_VALUE",
    )
    six.assertCountEqual(
        self, expected_checkpoint_names, [v.name for v in named_variables])

  def testInitNotCalled(self):

    class NoInit(tracking.AutoTrackable):

      def __init__(self):
        pass

    # __init__ for Trackable will be called implicitly.
    trackable_utils.add_variable(NoInit(), "var", shape=[])

  def testShapeDtype(self):
    root = tracking.AutoTrackable()
    v1 = trackable_utils.add_variable(
        root, name="v1", initializer=3., dtype=dtypes.float64)
    self.assertEqual(dtypes.float64, v1.dtype)
    v2 = trackable_utils.add_variable(
        root,
        name="v2",
        shape=[3],
        initializer=init_ops.ones_initializer,
        dtype=dtypes.float64)
    self.assertEqual(dtypes.float64, v2.dtype)
    self.assertAllEqual([1., 1., 1.], self.evaluate(v2))


class _MirroringSaveable(saver_lib.BaseSaverBuilder.SaveableObject):

  def __init__(self, primary_variable, mirrored_variable, name):
    self._primary_variable = primary_variable
    self._mirrored_variable = mirrored_variable
    tensor = self._primary_variable.read_value()
    spec = saver_lib.BaseSaverBuilder.SaveSpec(
        tensor=tensor,
        slice_spec="",
        name=name)
    super(_MirroringSaveable, self).__init__(
        tensor, [spec], name)

  def restore(self, restored_tensors, restored_shapes):
    """Restore the same value into both variables."""
    tensor, = restored_tensors
    return control_flow_ops.group(
        self._primary_variable.assign(tensor),
        self._mirrored_variable.assign(tensor))


class _OwnsMirroredVariables(base.Trackable):
  """A Trackable object which returns a more complex SaveableObject."""

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
    return {base.VARIABLE_VALUE_KEY: _saveable_factory}

  # The Saver sorts by name before parsing, so we need a name property.
  @property
  def name(self):
    return self.non_dep_variable.name


class CheckpointingTests(parameterized.TestCase, test.TestCase):

  @parameterized.named_parameters(
      ("_enable_async_ckpt", True),
      ("_disable_async_ckpt", False)
    )
  @test_util.run_in_graph_and_eager_modes
  def testMoreComplexSaveableReturned(self, enable_async_ckpt):
    if enable_async_ckpt and not context.executing_eagerly():
      self.skipTest(
          "Skipping this test as async checkpoint does not support graph mode.")
    v = _OwnsMirroredVariables()
    checkpoint = trackable_utils.Checkpoint(v=v)
    test_dir = self.get_temp_dir()
    prefix = os.path.join(test_dir, "ckpt")
    self.evaluate(v.non_dep_variable.assign(42.))
    ckpt_options = checkpoint_options.CheckpointOptions(
        experimental_enable_async_checkpoint=enable_async_ckpt)
    save_path = checkpoint.save(file_prefix=prefix, options=ckpt_options)
    self.evaluate(v.non_dep_variable.assign(43.))
    self.evaluate(v.mirrored.assign(44.))
    checkpoint.restore(save_path).assert_consumed().initialize_or_restore()
    self.assertEqual(42., self.evaluate(v.non_dep_variable))
    self.assertEqual(42., self.evaluate(v.mirrored))
    self.evaluate(v.non_dep_variable.assign(44.))
    save_path = checkpoint.save(file_prefix=prefix, options=ckpt_options)
    self.evaluate(v.non_dep_variable.assign(45.))
    checkpoint.restore(save_path).assert_consumed().initialize_or_restore()
    self.assertEqual(44., self.evaluate(v.non_dep_variable))
    self.assertEqual(44., self.evaluate(v.mirrored))

  @test_util.run_in_graph_and_eager_modes
  def testMoreComplexSaveableReturnedWithGlobalName(self):
    # The same object can also be saved using the name-based saver.
    v = _OwnsMirroredVariables()
    saver = saver_lib.Saver(var_list=[v])
    test_dir = self.get_temp_dir()
    prefix = os.path.join(test_dir, "ckpt")
    with self.cached_session() as sess:
      self.evaluate(v.non_dep_variable.assign(42.))
      save_path = saver.save(sess, prefix)
      self.evaluate(v.non_dep_variable.assign(43.))
      self.evaluate(v.mirrored.assign(44.))
      saver.restore(sess, save_path)
      self.assertEqual(42., self.evaluate(v.non_dep_variable))
      self.assertEqual(42., self.evaluate(v.mirrored))

  @parameterized.named_parameters(
      ("_enable_async_ckpt", True),
      ("_disable_async_ckpt", False)
    )
  @test_util.run_in_graph_and_eager_modes
  def testAssertConsumedNoCheckpoint(self, enable_async_ckpt):
    if enable_async_ckpt and not context.executing_eagerly():
      self.skipTest(
          "Skipping this test as async checkpoint does not support graph mode.")
    prefix = os.path.join(self.get_temp_dir(), "ckpt")
    v = variable_scope.get_variable(name="v", initializer=0.)
    self.evaluate(v.initializer)
    ckpt = trackable_utils.Checkpoint(v=v)
    self.evaluate(trackable_utils.gather_initializers(ckpt))
    ckpt_options = checkpoint_options.CheckpointOptions(
        experimental_enable_async_checkpoint=enable_async_ckpt)
    save_path = ckpt.save(file_prefix=prefix, options=ckpt_options)
    status = ckpt.restore(save_path=save_path)
    del ckpt
    status.assert_consumed()

  def testDeepCopyCheckpoint(self):
    prefix = os.path.join(self.get_temp_dir(), "ckpt")
    v = variables_lib.Variable(1.)
    original_ckpt = trackable_utils.Checkpoint(v=v)
    copied_ckpt = copy.deepcopy(original_ckpt)
    copied_ckpt.v.assign(2.)
    self.assertAllClose(1., v)
    save_path = copied_ckpt.save(file_prefix=prefix)
    original_ckpt.restore(save_path=save_path).assert_consumed()
    self.assertAllClose(2., v)

  @test_util.run_in_graph_and_eager_modes
  def testPassingCheckpointOptions(self):
    localhost = "/job:localhost/device:CPU:0"
    options = checkpoint_options.CheckpointOptions(
        experimental_io_device=localhost)
    prefix = os.path.join(self.get_temp_dir(), "ckpt")
    v = variable_scope.get_variable(name="v", initializer=0.)
    self.evaluate(v.initializer)
    ckpt = trackable_utils.Checkpoint(v=v)
    self.evaluate(trackable_utils.gather_initializers(ckpt))
    save_path = ckpt.save(file_prefix=prefix, options=options)
    status = ckpt.restore(save_path=save_path, options=options)
    del ckpt
    status.assert_consumed()

    # In graph mode, verify that the save and restore ops were set to run on
    # localhost.
    if not context.executing_eagerly():
      for op in ops.get_default_graph().get_operations():
        if op.type in ("SaveV2", "RestoreV2"):
          self.assertEqual(localhost, op.device)

  @test_util.run_in_graph_and_eager_modes
  def testFreezing(self):
    with test_util.use_gpu():
      # Save an object-based checkpoint using a frozen saver
      directory = self.get_temp_dir()
      prefix = os.path.join(directory, "ckpt")
      v = resource_variable_ops.ResourceVariable(0, dtype=dtypes.int64)
      checkpoint = trackable_utils.Checkpoint(v=v)
      self.evaluate(v.assign(3))
      # Create the save counter so assert_consumed doesn't complain about it not
      # existing in the checkpoint on restore.
      self.evaluate(checkpoint.save_counter.assign(12))
      saver = trackable_utils.frozen_saver(checkpoint)
      with ops.device("cpu:0"):
        prefix_tensor = constant_op.constant(prefix)
      self.evaluate(saver.save(prefix_tensor))
      self.evaluate(v.assign(10))
      # Use the frozen saver to restore the same object graph
      self.evaluate(saver.restore(prefix_tensor))
      self.assertEqual(3, self.evaluate(v))

      # Restore using another frozen saver on an identical object graph
      del v, checkpoint, saver
      v = resource_variable_ops.ResourceVariable(0, dtype=dtypes.int64)
      checkpoint = trackable_utils.Checkpoint(v=v)
      saver = trackable_utils.frozen_saver(checkpoint)
      self.evaluate(saver.restore(prefix_tensor))
      self.assertEqual(3, self.evaluate(v))

      # Restore as an object-based checkpoint
      del v, checkpoint, saver
      checkpoint = trackable_utils.Checkpoint()
      status = checkpoint.restore(prefix)
      v = resource_variable_ops.ResourceVariable(0, dtype=dtypes.int64)
      if context.executing_eagerly():
        self.assertEqual(12, self.evaluate(checkpoint.save_counter))
        self.assertEqual(0, self.evaluate(v))
      checkpoint.v = v
      status.assert_consumed().run_restore_ops()
      self.assertEqual(3, self.evaluate(v))
      self.assertEqual(12, self.evaluate(checkpoint.save_counter))

  @parameterized.named_parameters(
      ("_enable_async_ckpt", True),
      ("_disable_async_ckpt", False)
    )
  @test_util.run_in_graph_and_eager_modes
  def testCustomNumbering(self, enable_async_ckpt):
    if enable_async_ckpt and not context.executing_eagerly():
      self.skipTest(
          "Skipping this test as async checkpoint does not support graph mode.")
    directory = self.get_temp_dir()
    prefix = os.path.join(directory, "ckpt")
    step = resource_variable_ops.ResourceVariable(0, dtype=dtypes.int64)
    checkpoint = trackable_utils.Checkpoint(step=step)
    ckpt_options = checkpoint_options.CheckpointOptions(
        experimental_enable_async_checkpoint=enable_async_ckpt)
    self.evaluate(step.initializer)
    for i in range(5):
      path = checkpoint.write("%s-%d" % (prefix, self.evaluate(step)),
                              options=ckpt_options)
      expected_suffix = "-%d" % (2 * i,)
      if not path.endswith(expected_suffix):
        self.fail("%s should have suffix %s" % (path, expected_suffix))
      self.evaluate(step.assign_add(2))

  def testPartialRestoreWarningAttribute(self):
    with context.eager_mode():
      original_root = trackable_utils.Checkpoint(v1=variables_lib.Variable(2.),
                                                 v2=variables_lib.Variable(3.))
      prefix = os.path.join(self.get_temp_dir(), "ckpt")
      save_path = original_root.save(prefix)
      partial_root = trackable_utils.Checkpoint(v1=base.Trackable(),
                                                v2=variables_lib.Variable(0.))
      weak_partial_root = weakref.ref(partial_root)
      with test.mock.patch.object(logging, "warning") as mock_log:
        # Note: Unlike in testPartialRestoreWarningObject, the warning actually
        # prints immediately here, since all of the objects have been created
        # and there's no deferred restoration sitting around.
        partial_root.restore(save_path)
        self.assertEqual(3., partial_root.v2.numpy())
        del partial_root
        self.assertIsNone(weak_partial_root())
        messages = str(mock_log.call_args_list)
      self.assertIn("(root).v1", messages)
      self.assertNotIn("(root).v2", messages)
      self.assertIn("expect_partial()", messages)

  def testAttributeException(self):
    with context.eager_mode():
      original_root = trackable_utils.Checkpoint(v1=variables_lib.Variable(2.),
                                                 v2=variables_lib.Variable(3.))
      prefix = os.path.join(self.get_temp_dir(), "ckpt")
      save_path = original_root.save(prefix)
      partial_root = trackable_utils.Checkpoint(v1=base.Trackable(),
                                                v2=variables_lib.Variable(0.))
      status = partial_root.restore(save_path)
      with self.assertRaisesRegex(AssertionError,
                                  r"Unused attributes(.|\n)*\(root\).v1"):
        status.assert_consumed()

  def testSilencePartialWarning(self):
    with context.eager_mode():
      original_root = trackable_utils.Checkpoint(v1=variables_lib.Variable(2.),
                                                 v2=variables_lib.Variable(3.))
      prefix = os.path.join(self.get_temp_dir(), "ckpt")
      save_path = original_root.save(prefix)
      partial_root = trackable_utils.Checkpoint(v1=variables_lib.Variable(0.))
      weak_partial_root = weakref.ref(partial_root)
      weak_v1 = weakref.ref(partial_root.v1)
      partial_root.restore(save_path).expect_partial()
      self.assertEqual(2., partial_root.v1.numpy())
      with test.mock.patch.object(logging, "warning") as mock_log:
        del partial_root
        self.assertIsNone(weak_partial_root())
        self.assertIsNone(weak_v1())
        self.assertEmpty(mock_log.call_args_list)

  def _get_checkpoint_name(self, name):
    root = tracking.AutoTrackable()
    trackable_utils.add_variable(
        root, name=name, shape=[1, 2], dtype=dtypes.float64)
    (named_variable,), _, _ = graph_view.ObjectGraphView(
        root).serialize_object_graph()
    with ops.name_scope("root/" + named_variable.name):
      pass  # Make sure we can use this as an op name if we prefix it.
    return named_variable.name

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
    root = tracking.AutoTrackable()
    leaf = tracking.AutoTrackable()
    root.leaf = leaf
    trackable_utils.add_variable(leaf, name="v", shape=[])
    (named_variable,), _, _ = graph_view.ObjectGraphView(
        root).serialize_object_graph()
    self.assertEqual(r"leaf/v/.ATTRIBUTES/VARIABLE_VALUE", named_variable.name)

  @test_util.run_in_graph_and_eager_modes
  def testLocalNameValidation(self):
    root = tracking.AutoTrackable()
    leaf = tracking.AutoTrackable()
    # Dots are escaped, which avoids conflicts with reserved names.
    root._track_trackable(leaf, name=".ATTRIBUTES")
    trackable_utils.add_variable(trackable=leaf, name="a", shape=[])
    (named_variable,), _, _ = graph_view.ObjectGraphView(
        root).serialize_object_graph()
    self.assertEqual("..ATTRIBUTES/a/.ATTRIBUTES/VARIABLE_VALUE",
                     named_variable.name)

  @test_util.run_in_graph_and_eager_modes
  def testLateDependencyTracking(self):

    class Dependency(tracking.AutoTrackable):

      def build(self):
        self.var = trackable_utils.add_variable(
            self, "var", initializer=0.)

    class LateDependencies(trackable_utils.Checkpoint):

      def add_dep(self):
        self.dep = Dependency()
        self.dep.build()

    original = LateDependencies()
    original.add_dep()
    self.evaluate(state_ops.assign(original.dep.var, 123.))
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    save_path = original.save(checkpoint_prefix)
    load_into = LateDependencies()
    status = load_into.restore(save_path)
    status.assert_existing_objects_matched()
    with self.assertRaises(AssertionError):
      status.assert_consumed()
    load_into.add_dep()
    status.assert_consumed()
    status.assert_existing_objects_matched().run_restore_ops()
    self.assertEqual(123., self.evaluate(load_into.dep.var))

  @test_util.run_in_graph_and_eager_modes
  def testDepAfterVar(self):

    class Dependency(tracking.AutoTrackable):

      def build(self):
        self.var = trackable_utils.add_variable(
            self, "var", initializer=0.)

    class DepAfterVar(trackable_utils.Checkpoint):

      def add_dep(self):
        dep = Dependency()
        dep.build()
        self.dep = dep

    dep_after_var = DepAfterVar()
    dep_after_var.add_dep()
    self.evaluate(state_ops.assign(dep_after_var.dep.var, -14.))
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    save_path = dep_after_var.save(checkpoint_prefix)

    loaded_dep_after_var = DepAfterVar()
    status = loaded_dep_after_var.restore(save_path)
    loaded_dep_after_var.add_dep()
    status.assert_consumed()
    status.run_restore_ops()
    self.assertEqual(-14., self.evaluate(loaded_dep_after_var.dep.var))

  @test_util.run_in_graph_and_eager_modes
  def testOverlappingRestores(self):
    checkpoint_directory = self.get_temp_dir()
    save_root = trackable_utils.Checkpoint()
    save_root.dep = tracking.AutoTrackable()
    save_root.dep.var = trackable_utils.add_variable(
        save_root.dep, name="var", initializer=0.)
    self.evaluate(state_ops.assign(save_root.dep.var, 12.))
    first_path = save_root.save(os.path.join(checkpoint_directory, "first"))
    self.evaluate(state_ops.assign(save_root.dep.var, 13.))
    second_path = save_root.save(os.path.join(checkpoint_directory, "second"))

    first_root = trackable_utils.Checkpoint()
    second_root = trackable_utils.Checkpoint()
    first_status = first_root.restore(first_path)
    second_status = second_root.restore(second_path)
    load_dep = tracking.AutoTrackable()
    load_dep.var = trackable_utils.add_variable(
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
    first_root = trackable_utils.Checkpoint()
    second_root = trackable_utils.Checkpoint()
    second_status = second_root.restore(second_path)
    first_status = first_root.restore(first_path)
    load_dep = tracking.AutoTrackable()
    load_dep.var = trackable_utils.add_variable(
        load_dep, name="var", shape=[])
    first_root.dep = load_dep
    first_status.assert_consumed()
    first_status.run_restore_ops()
    self.assertEqual(12., self.evaluate(load_dep.var))
    second_root.dep = load_dep
    second_status.assert_consumed()
    second_status.run_restore_ops()
    self.assertEqual(12., self.evaluate(load_dep.var))

  @test_util.run_in_graph_and_eager_modes
  def testAmbiguousLoad(self):
    # Not OK to split one checkpoint object into two
    checkpoint_directory = self.get_temp_dir()
    save_root = trackable_utils.Checkpoint()
    save_root.dep_one = tracking.AutoTrackable()
    save_root.dep_two = tracking.AutoTrackable()
    dep_three = tracking.AutoTrackable()
    save_root.dep_one.dep_three = dep_three
    save_root.dep_two.dep_three = dep_three
    trackable_utils.add_variable(dep_three, name="var", initializer=0.)
    self.evaluate(trackable_utils.gather_initializers(save_root))
    save_path = save_root.save(os.path.join(checkpoint_directory, "ckpt"))
    load_root = trackable_utils.Checkpoint()
    status = load_root.restore(save_path)
    load_root.dep_one = tracking.AutoTrackable()
    load_root.dep_two = tracking.AutoTrackable()
    load_root.dep_one.dep_three = tracking.AutoTrackable()
    load_root.dep_two.dep_three = tracking.AutoTrackable()
    trackable_utils.add_variable(
        load_root.dep_one.dep_three, name="var", initializer=0.)
    trackable_utils.add_variable(
        load_root.dep_two.dep_three, name="var", initializer=0.)
    with self.assertRaises(AssertionError):
      status.assert_consumed()
    with self.assertRaises(AssertionError):
      status.assert_existing_objects_matched()

  @test_util.run_in_graph_and_eager_modes
  def testObjectsCombined(self):
    # Currently fine to load two checkpoint objects into one Python object
    checkpoint_directory = self.get_temp_dir()
    save_root = trackable_utils.Checkpoint()
    save_root.dep_one = tracking.AutoTrackable()
    save_root.dep_two = tracking.AutoTrackable()
    trackable_utils.add_variable(
        save_root.dep_one, name="var1", initializer=32., dtype=dtypes.float64)
    trackable_utils.add_variable(
        save_root.dep_two, name="var2", initializer=64., dtype=dtypes.float64)
    self.evaluate(trackable_utils.gather_initializers(save_root))
    save_path = save_root.save(os.path.join(checkpoint_directory, "ckpt"))
    load_root = trackable_utils.Checkpoint()
    load_root.dep_one = tracking.AutoTrackable()
    load_root.dep_two = load_root.dep_one
    v1 = trackable_utils.add_variable(
        load_root.dep_one, name="var1", shape=[], dtype=dtypes.float64)
    v2 = trackable_utils.add_variable(
        load_root.dep_one, name="var2", shape=[], dtype=dtypes.float64)
    status = load_root.restore(
        save_path).assert_consumed().assert_existing_objects_matched()
    status.run_restore_ops()
    self.assertEqual(32., self.evaluate(v1))
    self.assertEqual(64., self.evaluate(v2))

  @test_util.run_in_graph_and_eager_modes
  def testEmptyContainersIgnored(self):
    checkpoint_directory = self.get_temp_dir()
    save_root = trackable_utils.Checkpoint(a=[])
    path = save_root.save(checkpoint_directory)
    load_root = trackable_utils.Checkpoint(b=[])
    load_root.dep = []
    load_root.dep.append([])
    status = load_root.restore(path)
    status.assert_consumed()
    status.assert_existing_objects_matched()
    status.assert_nontrivial_match()

  @test_util.run_in_graph_and_eager_modes
  def testDependencyLoop(self):
    # Note: this test creates garbage during eager execution because it
    # purposefully creates a reference cycle.
    first = trackable_utils.Checkpoint()
    second = trackable_utils.Checkpoint()
    first.second = second
    second.first = first
    first.v = trackable_utils.add_variable(
        first, "v1", initializer=[3., 1., 4.])
    second.v = trackable_utils.add_variable(
        second, "v2", initializer=[1., 1., 2., 3.])
    self.evaluate(trackable_utils.gather_initializers(first))
    checkpoint_directory = self.get_temp_dir()
    save_path = first.save(os.path.join(checkpoint_directory, "ckpt"))

    # Test deferred loading
    first_load = trackable_utils.Checkpoint()
    status = first_load.restore(save_path)
    second_load = tracking.AutoTrackable()
    first_load.second = second_load
    second_load.first = first_load
    with self.assertRaises(AssertionError):
      status.assert_consumed()
    first_load.v = trackable_utils.add_variable(
        first_load, "v1", shape=[3])
    second_load.v = trackable_utils.add_variable(
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
    status = first_load.restore(save_path).assert_consumed()
    status.run_restore_ops()
    self.assertAllEqual([3., 1., 4.], self.evaluate(first_load.v))
    self.assertAllEqual([1., 1., 2., 3.], self.evaluate(second_load.v))

  @test_util.run_in_graph_and_eager_modes
  def testRestoreOnAssign(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    first = trackable_utils.Checkpoint()
    first.var1 = variables_lib.Variable(0., name="outside_var")
    first.var2 = variables_lib.Variable(0., name="blah")
    self.evaluate(first.var1.assign(4.))
    self.evaluate(first.var2.assign(8.))
    save_path = first.save(checkpoint_prefix)

    second = trackable_utils.Checkpoint()
    second.var2 = variables_lib.Variable(0., name="blah")
    status = second.restore(save_path)
    recreated_var1 = variables_lib.Variable(0., name="outside_var")
    status.run_restore_ops()
    self.assertEqual(8., self.evaluate(second.var2))
    self.evaluate(recreated_var1.assign(-2.))
    self.assertEqual(-2., self.evaluate(recreated_var1))
    second.var1 = recreated_var1
    status.run_restore_ops()
    self.assertEqual(4., self.evaluate(recreated_var1))

  @test_util.run_in_graph_and_eager_modes
  def testCheckpointState(self):
    # No checkpoints are deleted by default
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    obj = tracking.AutoTrackable()
    obj.var = variable_scope.get_variable(name="v", initializer=0.)
    self.evaluate(trackable_utils.gather_initializers(obj))
    saver = trackable_utils.Checkpoint(obj=obj)
    for _ in range(10):
      saver.save(checkpoint_prefix)
    expected_filenames = ["checkpoint"]
    for checkpoint_number in range(1, 11):
      expected_filenames.append("ckpt-%d.index" % (checkpoint_number,))
    self.assertEmpty(
        set(expected_filenames)
        - set(os.listdir(checkpoint_directory)))

  @test_util.run_in_graph_and_eager_modes
  def testCheckpointStateChangingVarList(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    obj = tracking.AutoTrackable()
    obj.var = variable_scope.get_variable(name="v", initializer=0.)
    self.evaluate(trackable_utils.gather_initializers(obj))
    checkpoint = trackable_utils.Checkpoint(obj=obj)
    looped_variables = []
    for iteration in range(10):
      new_variable = resource_variable_ops.ResourceVariable(iteration)
      self.evaluate(new_variable.initializer)
      setattr(checkpoint, "var_%d" % iteration, new_variable)
      checkpoint.save(checkpoint_prefix)
      looped_variables.append(new_variable)
    expected_filenames = ["checkpoint"]
    # We've copied the saver each time, but checkpoint management should still
    # be consistent. Nothing gets deleted.
    for checkpoint_number in range(1, 11):
      expected_filenames.append("ckpt-%d.index" % (checkpoint_number,))
    self.assertEmpty(
        set(expected_filenames)
        - set(os.listdir(checkpoint_directory)))
    self.assertEqual(
        checkpoint_prefix + "-10",
        checkpoint_management.latest_checkpoint(checkpoint_directory))
    # The checkpoint list only contains the most recent checkpoint, but they're
    # all on disk. This means we won't eventually run into proto size limits.
    self.assertEqual(
        [checkpoint_prefix + "-10"],
        (checkpoint_management.get_checkpoint_state(checkpoint_directory)
         .all_model_checkpoint_paths))
    for v in looped_variables:
      self.evaluate(v.assign(314))
    checkpoint.restore(checkpoint_prefix + "-6").run_restore_ops()
    self.assertEqual(314, self.evaluate(checkpoint.var_9))
    self.assertEqual(314, self.evaluate(checkpoint.var_8))
    self.assertEqual(314, self.evaluate(checkpoint.var_6))
    self.assertEqual(5, self.evaluate(checkpoint.var_5))
    self.assertEqual(1, self.evaluate(checkpoint.var_1))
    self.assertEqual(0, self.evaluate(checkpoint.var_0))
    checkpoint.restore(checkpoint_prefix + "-10").run_restore_ops()
    self.assertEqual(9, self.evaluate(checkpoint.var_9))
    self.assertEqual(8, self.evaluate(checkpoint.var_8))
    self.assertEqual(1, self.evaluate(checkpoint.var_1))
    self.assertEqual(0, self.evaluate(checkpoint.var_0))

  @test_util.run_in_graph_and_eager_modes
  def test_restore_after_adding_empty_trackable_data_structure(self):
    model = NonLayerTrackable()
    checkpoint = trackable_utils.Checkpoint(model=model)
    checkpoint.restore(None).initialize_or_restore()
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    save_path = checkpoint.save(checkpoint_prefix)

    del model, checkpoint

    model = NonLayerTrackable()
    model.dict = {"a": 1}
    model.list = {"b": 1}
    checkpoint = trackable_utils.Checkpoint(model=model)
    load_status = checkpoint.restore(save_path)
    load_status.assert_existing_objects_matched().run_restore_ops()

  @test_util.run_in_graph_and_eager_modes
  def test_write_checkpoint_path_str_from_function(self):

    checkpoint_prefix = os.path.join(self.get_temp_dir(), "ckpt")
    save_checkpoint = trackable_utils.Checkpoint(v=variables_lib.Variable(1.))

    @def_function.function
    def _write_checkpoint():
      save_path = save_checkpoint.write(checkpoint_prefix)
      return save_path

    self.evaluate([save_checkpoint.v.initializer])
    self.evaluate(_write_checkpoint())
    load_checkpoint = trackable_utils.Checkpoint(v=variables_lib.Variable(0.))
    # Use read() instead of restore() which allows us to check that all
    # existing objects were loaded.
    status = load_checkpoint.read(checkpoint_prefix)
    status.assert_existing_objects_matched()
    status.assert_consumed()
    status.run_restore_ops()
    self.assertEqual(1., self.evaluate(load_checkpoint.v))
    self.evaluate(save_checkpoint.v.assign(3.))
    self.evaluate(_write_checkpoint())
    self.evaluate(save_checkpoint.v.assign(0.))
    status = load_checkpoint.read(checkpoint_prefix)
    status.assert_existing_objects_matched()
    status.assert_consumed()
    status.run_restore_ops()
    self.assertEqual(3., self.evaluate(load_checkpoint.v))

  @test_util.run_in_graph_and_eager_modes
  def test_write_checkpoint_path_tensor_from_function(self):
    # Same as the previous test, but the path is a tensor not a python string.
    checkpoint_prefix = os.path.join(self.get_temp_dir(), "ckpt")

    checkpoint_prefix_tensor = constant_op.constant(checkpoint_prefix)

    save_checkpoint = trackable_utils.Checkpoint(v=variables_lib.Variable(1.))

    @def_function.function
    def _write_checkpoint(prefix):
      save_path = save_checkpoint.write(prefix)
      return save_path

    self.evaluate([save_checkpoint.v.initializer])
    self.evaluate(_write_checkpoint(checkpoint_prefix_tensor))
    load_checkpoint = trackable_utils.Checkpoint(v=variables_lib.Variable(0.))
    # Use read() instead of restore() which allows us to check that all
    # existing objects were loaded.
    status = load_checkpoint.read(checkpoint_prefix)
    status.assert_existing_objects_matched()
    status.assert_consumed()
    status.run_restore_ops()
    self.assertEqual(1., self.evaluate(load_checkpoint.v))
    self.evaluate(save_checkpoint.v.assign(3.))
    self.evaluate(_write_checkpoint(checkpoint_prefix_tensor))
    self.evaluate(save_checkpoint.v.assign(0.))
    status = load_checkpoint.read(checkpoint_prefix)
    status.assert_existing_objects_matched()
    status.assert_consumed()
    status.run_restore_ops()
    self.assertEqual(3., self.evaluate(load_checkpoint.v))

  @test_util.run_in_graph_and_eager_modes
  def test_write_checkpoint_path_tensor_does_not_exist_from_function(self):
    # Same as the previous test, but the path is a tensor not a python string.
    checkpoint_prefix = os.path.join(
        self.get_temp_dir(), "DOES_NOT_EXIST", "ckpt")

    checkpoint_prefix_tensor = constant_op.constant(checkpoint_prefix)

    save_checkpoint = trackable_utils.Checkpoint(v=variables_lib.Variable(1.))

    @def_function.function
    def _write_checkpoint(prefix):
      save_path = save_checkpoint.write(prefix)
      return save_path

    self.evaluate([save_checkpoint.v.initializer])
    with self.assertRaises(errors_impl.NotFoundError):
      self.evaluate(_write_checkpoint(checkpoint_prefix_tensor))

  @parameterized.named_parameters(
      ("_enable_async_ckpt", True),
      ("_disable_async_ckpt", False))
  def test_inititialize_with_data_structures(self, enable_async_ckpt):
    if enable_async_ckpt and not context.executing_eagerly():
      self.skipTest(
          "Skipping this test as async checkpoint does not support graph mode.")
    checkpoint = trackable_utils.Checkpoint(
        a=[variables_lib.Variable(0.), variables_lib.Variable(1.)],
        b={"a": variables_lib.Variable(2.), "b": variables_lib.Variable(3.)})
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    ckpt_options = checkpoint_options.CheckpointOptions(
        experimental_enable_async_checkpoint=enable_async_ckpt)
    save_path = checkpoint.save(file_prefix=checkpoint_prefix,
                                options=ckpt_options)
    load_checkpoint = trackable_utils.Checkpoint(
        a=[variables_lib.Variable(4.), variables_lib.Variable(5.)],
        b={"a": variables_lib.Variable(6.), "b": variables_lib.Variable(7.)})
    # When async checkpoint is enabled, we need to first make sure that the
    # checkpoint saving is fully complete before the checkpoint file can be
    # loaded by another checkpoint instance. Calling checkpoint.restore() is a
    # trick to make sure its async thread is joined.
    if enable_async_ckpt:
      checkpoint.restore(save_path)
    load_checkpoint.restore(save_path)
    self.assertAllClose(self.evaluate(load_checkpoint.a), [0, 1])
    self.assertAllClose(self.evaluate(load_checkpoint.b), {"a": 2, "b": 3})

  def _create_trackable(self):
    class Model(tracking.AutoTrackable):

      def __init__(self):
        self.v = variables_lib.Variable(2.)

      def __call__(self, x):
        return self.v * x
    return Model()

  def test_initialize_with_root_object(self):
    model = self._create_trackable()
    input_value = constant_op.constant([[3.]])
    expected_output = self.evaluate(model(input_value))
    model.deferred_variable = variables_lib.Variable(5.)

    checkpoint = trackable_utils.Checkpoint(model)
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    save_path = checkpoint.save(checkpoint_prefix)

    new_model = self._create_trackable()
    load_checkpoint = trackable_utils.Checkpoint(new_model)
    load_checkpoint.restore(save_path)
    self.assertAllClose(expected_output, new_model(input_value))

    new_model.deferred_variable = variables_lib.Variable(1.)
    self.assertEqual(self.evaluate(new_model.deferred_variable), 5)

  def test_initialize_with_root_object_and_kwargs(self):
    model = self._create_trackable()
    model.v.assign(3.)
    separate_variable = variables_lib.Variable(5.)

    with self.assertRaisesRegex(ValueError, "root.v already exists"):
      trackable_utils.Checkpoint(model, v=separate_variable)

    checkpoint = trackable_utils.Checkpoint(
        model, separate_variable=separate_variable)
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    save_path = checkpoint.save(checkpoint_prefix)

    # Case 1: Loading checkpoint with same configuration.
    new_model = self._create_trackable()
    separate_variable = variables_lib.Variable(1.)
    load_checkpoint = trackable_utils.Checkpoint(
        new_model, separate_variable=separate_variable)
    load_checkpoint.restore(save_path).assert_consumed()
    self.assertEqual(self.evaluate(new_model.v), 3)
    self.assertEqual(self.evaluate(separate_variable), 5)
    self.assertEqual(self.evaluate(load_checkpoint.save_counter), 1)

    # Case 2: Loading checkpoint where v and separate_variable are swapped:
    # v is not attached to the root, while separate variable is attached to root
    new_model = tracking.AutoTrackable()
    new_model.separate_variable = variables_lib.Variable(200.)
    v = variables_lib.Variable(100.)
    load_checkpoint = trackable_utils.Checkpoint(new_model, v=v)
    load_checkpoint.restore(save_path).assert_consumed()
    self.assertEqual(self.evaluate(v), 3)
    self.assertEqual(self.evaluate(new_model.separate_variable), 5)
    self.assertEqual(self.evaluate(load_checkpoint.save_counter), 1)

    # Case 3: Loading checkpoint where no root object is specified
    separate_variable = variables_lib.Variable(200.)
    v = variables_lib.Variable(100.)
    load_checkpoint = trackable_utils.Checkpoint(
        v=v, separate_variable=separate_variable)
    load_checkpoint.restore(save_path).assert_consumed()
    self.assertEqual(self.evaluate(v), 3)
    self.assertEqual(self.evaluate(new_model.separate_variable), 5)
    self.assertEqual(self.evaluate(load_checkpoint.save_counter), 1)

  def test_checkpoint_saved_model_compatibility(self):
    model = self._create_trackable()
    input_value = constant_op.constant([[3.]])
    expected_output = self.evaluate(model(input_value))
    model.deferred_variable = variables_lib.Variable(5.)
    saved_model_dir = os.path.join(self.get_temp_dir(), "saved_model")
    saved_model_save.save(model, saved_model_dir)

    new_model = self._create_trackable()
    load_checkpoint = trackable_utils.Checkpoint(new_model)

    with self.assertRaisesRegex(
        errors_impl.NotFoundError,
        "Error when restoring from checkpoint or SavedModel"):
      load_checkpoint.restore(saved_model_dir + "no").expect_partial()

    load_checkpoint.restore(saved_model_dir).expect_partial()
    self.assertAllClose(expected_output, new_model(input_value))

    new_model.deferred_variable = variables_lib.Variable(1.)
    self.assertEqual(self.evaluate(new_model.deferred_variable), 5)

  def test_deferred_dependency_avoids_reference_cycles(self):
    # Tests that there are no reference cycles when running garbage collection.
    # Python uses reference counts as the primary garbage collector, which will
    # not delete and finalize (__del__) objects in a cycle. The deletion is
    # eventually triggered by gc, which only runs when the garbage has reached
    # a certain threshold.

    delete_counter = 0

    class TrackableWithDel(tracking.AutoTrackable):

      def __del__(self):
        nonlocal delete_counter
        delete_counter += 1

    x = tracking.AutoTrackable()
    x.v = variables_lib.Variable(100.)
    x.has_del = TrackableWithDel()

    checkpoint = trackable_utils.Checkpoint(x)
    checkpoint_prefix = os.path.join(self.get_temp_dir(), "ckpt")
    save_path = checkpoint.save(checkpoint_prefix)

    self.assertEqual(delete_counter, 0)
    del checkpoint
    del x
    self.assertEqual(delete_counter, 1)

    no_v = tracking.AutoTrackable()
    no_v.has_del = TrackableWithDel()
    checkpoint = trackable_utils.Checkpoint(no_v)
    checkpoint.restore(save_path).expect_partial()
    del checkpoint
    del no_v
    self.assertEqual(delete_counter, 2)

  def test_defer_objects_with_values_only(self):
    # Tests that deferred dependencies are only added if the node in the
    # object graph has children or checkpointed values.
    root = tracking.AutoTrackable()
    root.branch_with_value = tracking.AutoTrackable()
    root.branch_with_value.v = variables_lib.Variable(5.0)
    root.branch_no_value = tracking.AutoTrackable()
    root.branch_no_value.child = tracking.AutoTrackable()
    root.v = variables_lib.Variable(1.0)

    checkpoint = trackable_utils.Checkpoint(model=root)
    checkpoint_prefix = os.path.join(self.get_temp_dir(), "ckpt")
    save_path = checkpoint.save(checkpoint_prefix)

    new_root = tracking.AutoTrackable()
    checkpoint = trackable_utils.Checkpoint(model=new_root)
    checkpoint.restore(save_path)

    # root should have two nodes with values/children (`branch-with_value`/`v`).
    self.assertLen(new_root._deferred_dependencies, 2)

    new_root.branch_no_value = tracking.AutoTrackable()
    self.assertLen(new_root._deferred_dependencies, 2)

    new_root.branch_with_value = tracking.AutoTrackable()
    self.assertLen(new_root._deferred_dependencies, 1)

    new_root.v = variables_lib.Variable(1.0)
    self.assertEmpty(new_root._deferred_dependencies, 1)

  def test_root_arg(self):
    root = tracking.AutoTrackable()
    root.v = variables_lib.Variable(1)
    w = variables_lib.Variable(2)
    y = variables_lib.Variable(3)
    root_ckpt = trackable_utils.Checkpoint(root=root, w=w, y=y)

    root2 = tracking.AutoTrackable()
    root2.w = variables_lib.Variable(4)
    v2 = variables_lib.Variable(5)
    z = variables_lib.Variable(6)
    root2_ckpt = trackable_utils.Checkpoint(root=root2,
                                            v=v2,
                                            z=z)

    root_save_path = root_ckpt.save(os.path.join(self.get_temp_dir(),
                                                 "root_ckpt"))
    root2_save_path = root2_ckpt.save(os.path.join(self.get_temp_dir(),
                                                   "root2_ckpt"))

    root_ckpt.restore(root2_save_path)
    root2_ckpt.restore(root_save_path)

    self.assertEqual(root.v.numpy(), 5)
    self.assertEqual(w.numpy(), 4)
    self.assertEqual(y.numpy(), 3)

    self.assertEqual(root2.w.numpy(), 2)
    self.assertEqual(v2.numpy(), 1)
    self.assertEqual(z.numpy(), 6)

  def test_weakref_root(self):
    root = tracking.AutoTrackable()
    root.v = variables_lib.Variable(1)
    ref = root.v.ref()

    ckpt = trackable_utils.Checkpoint(root=weakref.ref(root))
    save_path = ckpt.save(os.path.join(self.get_temp_dir(), "ckpt"))
    root.v.assign(2)
    ckpt.restore(save_path)
    self.assertEqual(root.v.numpy(), 1)

    del root

    # Verifying if the variable is only referenced from `ref`.
    # We expect the reference counter to be 1, but `sys.getrefcount` reports
    # one higher reference counter because a temporary is created when we call
    # sys.getrefcount().  Hence check if the number returned is 2.
    # https://docs.python.org/3/library/sys.html#sys.getrefcount
    self.assertEqual(sys.getrefcount(ref.deref()), 2)

  def test_restore_incompatible_shape(self):
    v = variables_lib.Variable([1.0, 1.0])
    w = variables_lib.Variable([1.0])
    ckpt = trackable_utils.Checkpoint(v=v)
    save_path = ckpt.save(os.path.join(self.get_temp_dir(), "ckpt"))

    with self.assertRaisesRegex(ValueError, "incompatible tensor with shape"):
      trackable_utils.Checkpoint(v=w).restore(save_path)

  def test_save_restore_fspath(self):
    v = variables_lib.Variable(1.0)
    w = variables_lib.Variable(0.0)
    ckpt = trackable_utils.Checkpoint(v=v)
    prefix = pathlib.Path(self.get_temp_dir()) / "ckpt"
    save_path = ckpt.save(prefix)
    save_path = pathlib.Path(save_path)
    ckpt2 = trackable_utils.Checkpoint(v=w)
    ckpt2.restore(save_path)
    self.assertEqual(ckpt.v.numpy(), 1.0)

  def test_read_write_fspath(self):
    v = variables_lib.Variable(1.0)
    w = variables_lib.Variable(0.0)
    ckpt = trackable_utils.Checkpoint(v=v)
    prefix = pathlib.Path(self.get_temp_dir()) / "ckpt"
    save_path = ckpt.write(prefix)
    save_path = pathlib.Path(save_path)
    ckpt2 = trackable_utils.Checkpoint(v=w)
    ckpt2.read(save_path)
    self.assertEqual(ckpt.v.numpy(), 1.0)


class TemplateTests(parameterized.TestCase, test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_trackable_save_restore_nested(self):

    def _inner_template():
      v = variable_scope.get_variable(
          "v", shape=[1], initializer=init_ops.zeros_initializer())
      return v

    def _outer_template():
      first_inner = template.make_template("i1", _inner_template)
      second_inner = template.make_template("i2", _inner_template)
      v1 = first_inner()
      v2 = second_inner()
      v3 = second_inner()
      return (first_inner, second_inner), (v1, v2, v3)

    with variable_scope.variable_scope("ignored"):
      save_template = template.make_template("s1", _outer_template)
      save_root = trackable_utils.Checkpoint(my_template=save_template)
      (inner_template_one, inner_template_two), _ = save_template()
    self.evaluate(inner_template_one.variables[0].assign([20.]))
    self.evaluate(inner_template_two.variables[0].assign([25.]))
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    save_path = save_root.save(checkpoint_prefix)

    load_template = template.make_template("s2", _outer_template)
    load_root = trackable_utils.Checkpoint(my_template=load_template)
    status = load_root.restore(save_path)
    (inner_template_one, inner_template_two), (v1, v2, v3) = load_template()
    outer_template_dependencies = load_root.my_template._trackable_children()
    self.assertLen(outer_template_dependencies, 2)
    self.assertDictEqual({"i1": inner_template_one, "i2": inner_template_two},
                         outer_template_dependencies)
    self.assertLen(inner_template_one._trackable_children(), 1)
    self.assertIn("v", inner_template_one._trackable_children())
    self.assertLen(inner_template_two._trackable_children(), 1)
    self.assertIn("v", inner_template_two._trackable_children())
    status.assert_consumed().run_restore_ops()
    self.assertAllEqual([20.], self.evaluate(v1))
    self.assertAllEqual([25.], self.evaluate(v2))
    self.assertAllEqual([25.], self.evaluate(v3))


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
