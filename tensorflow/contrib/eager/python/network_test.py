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

import gc

from tensorflow.contrib.eager.python import network
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.layers import core
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import training_util


# pylint: disable=not-callable
class MyNetwork(network.Network):

  def __init__(self, name=None):
    super(MyNetwork, self).__init__(name=name)
    self.l1 = self.track_layer(core.Dense(1, use_bias=False))

  def call(self, x):
    return self.l1(x)


class RegularizedNetwork(network.Network):

  def __init__(self):
    super(RegularizedNetwork, self).__init__()
    self.l1 = self.track_layer(core.Dense(
        1,
        bias_regularizer=regularizers.l1_regularizer(2.0),
        kernel_regularizer=regularizers.l1_regularizer(2.0)))
    self.l2 = self.track_layer(core.Dense(
        1,
        bias_regularizer=regularizers.l1_regularizer(2.0)))

  def call(self, values):
    return self.l2(self.l1(values))


class NetworkTest(test.TestCase):

  def _save_modify_load_network_built(self, net, global_step=None):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_path = network.save_network_checkpoint(
        network=net, save_path=checkpoint_directory, global_step=global_step)
    input_value = constant_op.constant([[42.0]])
    original_output = self.evaluate(net(input_value))
    for var in net.variables:
      self.evaluate(var.assign(var + 1.))
    self.assertGreater(
        self.evaluate(net(input_value)),
        original_output)
    # Either the returned explicit checkpoint path or the directory should work.
    network.restore_network_checkpoint(net, save_path=checkpoint_directory)
    self.assertAllEqual(
        original_output,
        self.evaluate(net(input_value)))
    for var in net.variables:
      self.evaluate(var.assign(var + 2.))
    network.restore_network_checkpoint(net, save_path=checkpoint_path)
    self.assertAllEqual(
        original_output,
        self.evaluate(net(input_value)))

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testTrainableAttribute(self):
    net = network.Network()
    self.assertTrue(net.trainable)
    with self.assertRaises(AttributeError):
      net.trainable = False
    self.assertTrue(net.trainable)

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testNetworkCall(self):
    net = MyNetwork(name="abcd")
    net(constant_op.constant([[2.0]]))  # Force variables to be created.
    self.assertEqual(1, len(net.trainable_variables))
    self.evaluate(net.trainable_variables[0].assign([[17.0]]))
    # TODO(josh11b): Support passing Python values to networks.
    result = net(constant_op.constant([[2.0]]))
    self.assertEqual(34.0, self.evaluate(result))

  def testReplacingNetworkCallWithDefun(self):
    net = MyNetwork(name="abcd")
    net.call = function.defun(net.call)
    x = constant_op.constant([[2.0]])
    net(x)  # Force variables to be created.
    self.evaluate(net.trainable_variables[0].assign([[17.0]]))

    result = net(x)  # Build and execute the TensorFlow function
    self.assertEqual(34.0, self.evaluate(result))

    # Force the creation of another TensorFlow function by changing input shape
    y = constant_op.constant([[1.0], [2.0]])
    result = net(y)
    self.assertAllEqual([[17.0], [34.0]], self.evaluate(result))

  # TODO(allenl): This test creates garbage in some Python versions
  @test_util.run_in_graph_and_eager_modes()
  def testNetworkSaveRestoreAlreadyBuilt(self):
    net = MyNetwork(name="abcd")
    with self.assertRaisesRegexp(
        ValueError, "Attempt to save the Network before it was first called"):
      network.save_network_checkpoint(net, self.get_temp_dir())
    net(constant_op.constant([[2.0]]))
    self.evaluate(net.trainable_variables[0].assign([[17.0]]))
    self._save_modify_load_network_built(net, global_step=None)
    self._save_modify_load_network_built(net, global_step=10)

  # TODO(allenl): This test creates garbage in some Python versions
  @test_util.run_in_graph_and_eager_modes()
  def testSaveRestoreDefaultGlobalStep(self):
    net = MyNetwork(name="abcd")
    net(constant_op.constant([[2.0]]))
    self.evaluate(net.variables[0].assign([[3.]]))
    default_global_step = training_util.get_or_create_global_step()
    self.evaluate(default_global_step.assign(4242))
    save_path = network.save_network_checkpoint(net, self.get_temp_dir())
    self.assertIn("abcd-4242", save_path)

  # TODO(allenl): This test creates garbage in some Python versions
  @test_util.run_in_graph_and_eager_modes()
  def testNetworkSaveAndRestoreIntoUnbuilt(self):
    save_dir = self.get_temp_dir()
    net1 = MyNetwork()
    test_input = constant_op.constant([[2.0]])
    net1(test_input)
    self.evaluate(net1.trainable_variables[0].assign([[17.0]]))
    save_path = network.save_network_checkpoint(net1, save_dir)
    # With a pre-build restore we should have the same value.
    net2 = MyNetwork()
    network.restore_network_checkpoint(net2, save_path)
    self.assertAllEqual(self.evaluate(net1(test_input)),
                        self.evaluate(net2(test_input)))
    self.assertIsNot(net1.variables[0], net2.variables[0])
    self.assertAllEqual(self.evaluate(net1.variables[0]),
                        self.evaluate(net2.variables[0]))

  @test_util.run_in_graph_and_eager_modes()
  def testNetworkMatchesLayerVariableNames(self):
    zero = constant_op.constant([[0.]])
    layer_one = core.Dense(1, use_bias=False)
    layer_one(zero)
    layer_two = core.Dense(1, use_bias=False)
    layer_two(zero)

    class TwoLayerNet(network.Network):

      def __init__(self, name=None):
        super(TwoLayerNet, self).__init__(name=name)
        self.first = self.track_layer(core.Dense(
            1, use_bias=False))
        self.second = self.track_layer(core.Dense(
            1, use_bias=False))

      def call(self, x):
        return self.second(self.first(x))

    net = TwoLayerNet()
    net(zero)
    self.assertEqual("two_layer_net/" + layer_one.variables[0].name,
                     net.first.variables[0].name)
    self.assertEqual("two_layer_net/" + layer_two.variables[0].name,
                     net.second.variables[0].name)

  @test_util.run_in_graph_and_eager_modes()
  def testLoadIntoUnbuiltSharedLayer(self):

    class Owner(network.Network):

      def __init__(self, name=None):
        super(Owner, self).__init__(name=name)
        self.first = self.track_layer(core.Dense(
            1, name="first_layer", use_bias=False))

      def call(self, x):
        return self.first(x)

    first_owner = Owner()

    class User(network.Network):

      def __init__(self, use_layer, name=None):
        super(User, self).__init__(name=name)
        self.first = self.track_layer(use_layer)
        self.second = self.track_layer(core.Dense(
            1, name="second_layer", use_bias=False))

      def call(self, x):
        return self.second(self.first(x))

    class LikeUserButNotSharing(network.Network):

      def __init__(self, name=None):
        super(LikeUserButNotSharing, self).__init__(name=name)
        self.first = self.track_layer(core.Dense(
            1, name="first_layer", use_bias=False))
        self.second = self.track_layer(core.Dense(
            1, name="second_layer", use_bias=False))

      def call(self, x):
        return self.second(self.first(x))

    checkpoint_creator = LikeUserButNotSharing(name="checkpoint_creator")
    one = constant_op.constant([[1.0]])
    checkpoint_creator(one)
    self.assertEqual(2, len(checkpoint_creator.variables))
    self.evaluate(checkpoint_creator.variables[0].assign([[5.]]))
    self.evaluate(checkpoint_creator.variables[1].assign([[6.]]))
    # Re-map the variable names so that with default restore mapping we'll
    # attempt to restore into the unbuilt Layer.
    name_mapping = {
        "checkpoint_creator/first_layer/kernel": "owner/first_layer/kernel",
        "checkpoint_creator/second_layer/kernel": "second_layer/kernel",
    }
    save_path = network.save_network_checkpoint(
        checkpoint_creator,
        self.get_temp_dir(),
        map_func=lambda full_name: name_mapping[full_name])
    load_into = User(use_layer=first_owner.first)
    network.restore_network_checkpoint(load_into, save_path)
    self.assertEqual(0, len(first_owner.variables))
    self.assertAllEqual(self.evaluate(checkpoint_creator(one)),
                        self.evaluate(load_into(one)))
    self.assertEqual(1, len(first_owner.variables))
    self.assertAllEqual([[5.]], self.evaluate(load_into.variables[0]))
    self.assertAllEqual([[6.]], self.evaluate(load_into.variables[1]))
    first_owner(one)
    self.assertAllEqual([[5.]], self.evaluate(first_owner.variables[0]))

    # Try again with a garbage collected parent.
    first_owner = Owner()
    load_into = User(use_layer=first_owner.first)
    del first_owner
    gc.collect()
    def _restore_map_func(original_name):
      if original_name.startswith("owner/"):
        return original_name.replace("owner/", "owner_1/")
      else:
        return "user_1/" + original_name
    with self.assertRaisesRegexp(ValueError, "garbage collected"):
      network.restore_network_checkpoint(
          load_into, save_path, map_func=_restore_map_func)

  @test_util.run_in_graph_and_eager_modes()
  def testRestoreIntoSubNetwork(self):

    class Parent(network.Network):

      def __init__(self, name=None):
        super(Parent, self).__init__(name=name)
        self.first = self.track_layer(MyNetwork())
        self.second = self.track_layer(MyNetwork())

      def call(self, x):
        return self.first(self.second(x))

    one = constant_op.constant([[3.]])
    whole_model_saver = Parent()
    whole_model_saver(one)
    self.evaluate(whole_model_saver.variables[0].assign([[15.]]))
    self.evaluate(whole_model_saver.variables[1].assign([[16.]]))
    whole_model_checkpoint = network.save_network_checkpoint(
        whole_model_saver, self.get_temp_dir())

    save_from = MyNetwork()
    save_from(one)
    self.evaluate(save_from.variables[0].assign([[5.]]))
    checkpoint = network.save_network_checkpoint(save_from, self.get_temp_dir())
    save_into_parent = Parent()
    network.restore_network_checkpoint(save_into_parent, whole_model_checkpoint)
    network.restore_network_checkpoint(save_into_parent.first, checkpoint)
    # deferred loading multiple times is fine
    network.restore_network_checkpoint(save_into_parent.first, checkpoint)
    save_into_parent(one)  # deferred loading
    self.assertAllEqual([[5.]], self.evaluate(save_into_parent.variables[0]))
    self.assertAllEqual([[16.]], self.evaluate(save_into_parent.variables[1]))

    # Try again with the opposite ordering, and we should get different results
    # (deferred restoration should happen the same way non-deferred happens,
    # with later restorations overwriting older ones).
    save_into_parent = Parent()
    # deferred loading multiple times is fine
    network.restore_network_checkpoint(save_into_parent.first, checkpoint)
    network.restore_network_checkpoint(save_into_parent, whole_model_checkpoint)
    save_into_parent(one)  # deferred loading
    # We've overwritten the sub-Network restore.
    self.assertAllEqual([[15.]], self.evaluate(save_into_parent.variables[0]))
    self.assertAllEqual([[16.]], self.evaluate(save_into_parent.variables[1]))

    self.evaluate(save_into_parent.variables[0].assign([[3.]]))
    self.evaluate(save_into_parent.variables[1].assign([[4.]]))
    network.restore_network_checkpoint(save_into_parent.second, checkpoint)
    self.assertAllEqual([[5.]], self.evaluate(save_into_parent.variables[1]))
    with self.assertRaisesRegexp(errors_impl.NotFoundError,
                                 "not found in checkpoint"):
      # The checkpoint is incompatible.
      network.restore_network_checkpoint(save_into_parent, checkpoint)

  @test_util.run_in_graph_and_eager_modes()
  def testCustomMapCollisionErrors(self):

    class Parent(network.Network):

      def __init__(self, name=None):
        super(Parent, self).__init__(name=name)
        self.first = self.track_layer(MyNetwork())
        self.second = self.track_layer(MyNetwork())

      def call(self, x):
        return self.first(self.second(x))

    make_checkpoint = Parent()
    one = constant_op.constant([[1.]])
    make_checkpoint(one)
    self.evaluate(make_checkpoint.variables[0].assign([[2.]]))
    self.evaluate(make_checkpoint.variables[1].assign([[3.]]))
    with self.assertRaisesRegexp(
        ValueError,
        "The map_func passed to save_network_checkpoint for the Network "
        "'parent' resulted in two variables named 'foo'"):
      network.save_network_checkpoint(
          make_checkpoint, self.get_temp_dir(), map_func=lambda n: "foo")
    checkpoint = network.save_network_checkpoint(
        network=make_checkpoint.first,
        save_path=self.get_temp_dir(),
        map_func=lambda n: "foo")
    loader = Parent()
    network.restore_network_checkpoint(
        loader, checkpoint, map_func=lambda n: "foo")
    with self.assertRaisesRegexp(
        ValueError,
        ("The map_func passed to restore_network_checkpoint for the Network"
         " 'parent_1' resulted in two variables named 'foo'")):
      loader(one)
    loader = Parent()
    loader(one)
    with self.assertRaisesRegexp(
        ValueError,
        ("The map_func passed to restore_network_checkpoint for the Network"
         " 'parent_2' resulted in two variables named 'foo'")):
      network.restore_network_checkpoint(
          loader, checkpoint, map_func=lambda n: "foo")

  @test_util.run_in_graph_and_eager_modes()
  def testDefaultMapCollisionErrors(self):

    one = constant_op.constant([[1.]])
    first = core.Dense(1, name="dense", use_bias=False)
    first(one)

    class Parent(network.Network):

      def __init__(self, name=None):
        super(Parent, self).__init__(name=name)
        self.first = self.track_layer(first)
        self.second = self.track_layer(core.Dense(1, use_bias=False))

      def call(self, x):
        return self.first(self.second(x))

    make_checkpoint = Parent()
    one = constant_op.constant([[1.]])
    make_checkpoint(one)
    self.evaluate(make_checkpoint.variables[0].assign([[2.]]))
    self.evaluate(make_checkpoint.variables[1].assign([[3.]]))
    with self.assertRaisesRegexp(
        ValueError,
        ("The default checkpoint variable name mapping strategy for Network "
         "'parent' resulted in a naming conflict.")):
      network.save_network_checkpoint(make_checkpoint, self.get_temp_dir())

    class Compatible(network.Network):

      def __init__(self, name=None):
        super(Compatible, self).__init__(name=name)
        self.first = self.track_layer(core.Dense(1, use_bias=False))

      def call(self, x):
        return self.first(x)

    successful_checkpoint = Compatible()
    successful_checkpoint(one)
    self.evaluate(successful_checkpoint.variables[0].assign([[-1.]]))
    checkpoint_path = network.save_network_checkpoint(
        successful_checkpoint, self.get_temp_dir())
    load_checkpoint = Parent()
    load_checkpoint(one)
    with self.assertRaisesRegexp(
        ValueError,
        ("The default checkpoint variable name mapping strategy for Network "
         "'parent_1' resulted in a naming conflict.")):
      network.restore_network_checkpoint(load_checkpoint, checkpoint_path)

  def testNoReferenceCyclesAfterCall(self):

    class ChildNetwork(network.Network):

      def __init__(self, name=None):
        super(ChildNetwork, self).__init__(name=name)

      def call(self, x):
        return x * 2.

    class ParentNetwork(network.Network):

      def __init__(self, name=None):
        super(ParentNetwork, self).__init__(name=name)
        self.l1 = self.track_layer(ChildNetwork())

      def call(self, x):
        return self.l1(x)

    one = constant_op.constant([[1.0]])
    gc.disable()
    gc.collect()
    previous_gc_debug_flags = gc.get_debug()
    gc.set_debug(gc.DEBUG_SAVEALL)
    preexisting = len(gc.garbage)
    net = ParentNetwork()
    net(one)
    del net
    gc.collect()
    # There should be no additional garbage requiring collection.
    self.assertEqual(preexisting, len(gc.garbage))
    gc.set_debug(previous_gc_debug_flags)
    gc.enable()

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testAnonymousNoNameInitially(self):
    net = MyNetwork()
    with self.assertRaisesRegexp(ValueError, "does not yet have a final name"):
      net.name  # pylint: disable=pointless-statement

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testExplicitHasNameInitially(self):
    net = MyNetwork(name="abcd")
    self.assertEqual("abcd", net.name)

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testUsingResourceVariables(self):
    net = MyNetwork()
    net(constant_op.constant([[0.]]))
    self.assertIsInstance(net.trainable_weights[0],
                          resource_variable_ops.ResourceVariable)

  def testGraphOpNames(self):
    """Network operation names should match variable naming."""

    def _check_op_prefixes(expected_prefix, checked_ops):
      for operation in ops.get_default_graph().get_operations():
        if operation.name == "ignore":
          continue
        if operation.name in checked_ops:
          continue
        checked_ops.add(operation.name)
        self.assertStartsWith(expected_start=expected_prefix,
                              actual=operation.name)
        self.assertNotIn("my_network", operation.name[len(expected_prefix):])
        self.assertNotIn("dense", operation.name[len(expected_prefix):])

    with context.graph_mode():
      net = MyNetwork()
      zero = constant_op.constant([[0.]], name="ignore")
      net(zero)
      checked_ops = set()
      _check_op_prefixes(expected_prefix="my_network/dense/",
                         checked_ops=checked_ops)
      net.net2 = net.track_layer(MyNetwork())
      net.net2(zero)
      _check_op_prefixes(expected_prefix="my_network/my_network/dense/",
                         checked_ops=checked_ops)
      MyNetwork()(zero)
      _check_op_prefixes(expected_prefix="my_network_1/dense/",
                         checked_ops=checked_ops)

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testVariableRegularizers(self):
    net = RegularizedNetwork()
    net(constant_op.constant([[1.]]))
    self.evaluate(net.variables[0].assign([[2.]]))
    self.evaluate(net.variables[1].assign([3.]))
    self.evaluate(net.variables[2].assign([[-2.]]))
    self.evaluate(net.variables[3].assign([4.]))
    self.assertAllEqual([4., 6., 8.], self.evaluate(net.losses))
    self.evaluate(net.variables[3].assign([5.]))
    self.assertAllEqual([4., 6., 10.], self.evaluate(net.losses))

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testDuplicateNameError(self):
    one = constant_op.constant([[1.]])
    net = MyNetwork(name="foo")
    net(one)
    with self.assertRaisesRegexp(
        ValueError, "named 'foo' already exists"):
      net1 = MyNetwork(name="foo")
      net1(one)

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testWrappingInVariableScope(self):
    one = constant_op.constant([[1.]])
    # Naming happens in the order of first build rather than the order of
    # construction, but for clarity they're the same here and construction is
    # annotated.
    outside_net_before = MyNetwork()  # name=my_network
    outside_net_before(one)
    captured_scope = variable_scope.get_variable_scope()
    with variable_scope.variable_scope("outside_scope"):
      net1 = MyNetwork()  # name=outside_scope/my_network
      net1(one)
      name_conflict1 = MyNetwork(name="name_conflict")  # fine, unique so far
      name_conflict2 = MyNetwork(name="name_conflict")  # error on build
      with variable_scope.variable_scope("inside_scope"):
        # No issue here since the name is unique within its scope.
        name_conflict3 = MyNetwork(name="name_conflict")
      net2 = MyNetwork()  # name=outside_scope/my_network_2 to avoid the
                          # variable_scope my_network_1 below.
      vs_name_conflict = MyNetwork(name="vs_name_conflict")  # conflict below
    with variable_scope.variable_scope("intervening_scope"):
      with variable_scope.variable_scope(captured_scope):
        with variable_scope.variable_scope("outside_scope"):
          name_conflict4 = MyNetwork(name="name_conflict")  # error on build
          with variable_scope.variable_scope("my_network_1"):
            pass
          with variable_scope.variable_scope("vs_name_conflict"):
            pass
          net3 = MyNetwork()  # name=outside_scope/my_network_4
    name_conflict1(one)
    with self.assertRaisesRegexp(
        ValueError, "named 'name_conflict' already exists"):
      name_conflict2(one)
    name_conflict3(one)
    net2(one)
    with self.assertRaisesRegexp(
        ValueError, "or a variable_scope was created with this name"):
      vs_name_conflict(one)
    with self.assertRaisesRegexp(
        ValueError, "named 'name_conflict' already exists"):
      name_conflict4(one)
    self.assertEqual("outside_scope/name_conflict",
                     name_conflict1.name)
    self.assertStartsWith(
        expected_start="outside_scope/name_conflict/dense/",
        actual=name_conflict1.variables[0].name)
    self.assertEqual("outside_scope/inside_scope/name_conflict",
                     name_conflict3.name)
    self.assertStartsWith(
        expected_start="outside_scope/inside_scope/name_conflict/dense/",
        actual=name_conflict3.variables[0].name)
    self.assertEqual("outside_scope/my_network", net1.name)
    self.assertStartsWith(
        expected_start="outside_scope/my_network/dense/",
        actual=net1.trainable_weights[0].name)
    self.assertEqual("outside_scope/my_network_2", net2.name)
    self.assertStartsWith(
        expected_start="outside_scope/my_network_2/dense/",
        actual=net2.trainable_weights[0].name)
    net3(one)
    self.assertEqual("outside_scope/my_network_3", net3.name)
    self.assertStartsWith(
        expected_start="outside_scope/my_network_3/dense/",
        actual=net3.trainable_weights[0].name)
    outside_net_after = MyNetwork()
    outside_net_after(one)
    self.assertEqual("my_network", outside_net_before.name)
    self.assertStartsWith(
        expected_start="my_network/dense/",
        actual=outside_net_before.trainable_weights[0].name)
    self.assertEqual("my_network_1", outside_net_after.name)
    self.assertStartsWith(
        expected_start="my_network_1/dense/",
        actual=outside_net_after.trainable_weights[0].name)

  @test_util.run_in_graph_and_eager_modes()
  def testVariableScopeStripping(self):
    with variable_scope.variable_scope("scope1"):
      with variable_scope.variable_scope("scope2"):
        net = MyNetwork()
    net(constant_op.constant([[2.0]]))
    self.evaluate(net.variables[0].assign([[42.]]))
    self.assertEqual(net.name, "scope1/scope2/my_network")
    self.assertStartsWith(
        expected_start="scope1/scope2/my_network/dense/",
        actual=net.trainable_weights[0].name)
    save_path = network.save_network_checkpoint(net, self.get_temp_dir())
    self.assertIn("scope1_scope2_my_network", save_path)
    restore_net = MyNetwork()
    # Delayed restoration
    network.restore_network_checkpoint(restore_net, save_path)
    restore_net(constant_op.constant([[1.0]]))
    self.assertAllEqual([[42.]],
                        self.evaluate(restore_net.variables[0]))
    self.evaluate(restore_net.variables[0].assign([[-1.]]))
    # Immediate restoration
    network.restore_network_checkpoint(restore_net, save_path)
    self.assertAllEqual([[42.]],
                        self.evaluate(restore_net.variables[0]))

  @test_util.run_in_graph_and_eager_modes()
  def testLayerNamesRespected(self):
    class ParentNetwork(network.Network):

      def __init__(self):
        super(ParentNetwork, self).__init__()
        self.first = self.track_layer(
            core.Dense(1, use_bias=False, name="explicit_name"))

      def call(self, x):
        return self.first(x)

    one = constant_op.constant([[1.]])
    net = ParentNetwork()
    net(one)
    self.assertStartsWith(expected_start="parent_network/explicit_name/",
                          actual=net.trainable_weights[0].name)
    self.assertEqual("explicit_name", net.first.name)

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testWrappingInAnonymousVariableScope(self):
    # Named outside variable_scopes are not supported at the moment. However,
    # blank-named top level variable scopes do not change variable names, and so
    # can be used to set the properties of Network variables.
    was_called = [False]
    def _custom_getter(getter, *args, **kwargs):
      was_called[0] = True
      return getter(*args, **kwargs)
    with variable_scope.variable_scope("", custom_getter=_custom_getter):
      net = MyNetwork()
      one = constant_op.constant([[1.]])
      net(one)
    self.assertTrue(was_called[0])

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testReasonableSlashError(self):
    with self.assertRaisesRegexp(
        ValueError, "not allowed in Network names"):
      MyNetwork(name="slash/slash")

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testNoVariableScopeNames(self):
    with self.assertRaisesRegexp(
        ValueError, "VariableScopes are not valid Network names"):
      with variable_scope.variable_scope("some_scope") as vs:
        MyNetwork(name=vs)

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testVariableScopeNameCollision(self):
    with variable_scope.variable_scope("abcd"):
      pass
    with self.assertRaisesRegexp(
        ValueError, "or a variable_scope was created with this name"):
      net = MyNetwork(name="abcd")
      one = constant_op.constant([[1.]])
      net(one)

  @test_util.run_in_graph_and_eager_modes(assert_no_eager_garbage=True)
  def testNetworkVariablesDoNotInterfere(self):
    core.Dense(1, use_bias=True)  # Should not interfere with naming.
    net1 = MyNetwork()
    net2 = MyNetwork()
    one = constant_op.constant([[1.]])
    net1(one)
    net2(one)
    # Layer names typically are globally unique rather than being unique within
    # the scope of their first use. However, within a Network they must be named
    # locally so that previous Layer consutrciton does not interfere with
    # variable naming (e.g. add a Layer construction before the Network,
    # suddenly your previously saved checkpoint is incompatible).
    self.assertEqual("dense", net1.l1.name)
    self.assertEqual("dense", net2.l1.name)
    self.evaluate(net1.trainable_weights[0].assign([[1.]]))
    self.evaluate(net2.trainable_weights[0].assign([[2.]]))
    self.assertEqual(2., self.evaluate(net2.trainable_weights[0]))
    self.assertEqual(1., self.evaluate(net1.trainable_weights[0]))
    self.assertStartsWith(expected_start="my_network/dense/",
                          actual=net1.trainable_weights[0].name)
    self.assertStartsWith(expected_start="my_network_1/dense/",
                          actual=net2.trainable_weights[0].name)

  @test_util.run_in_graph_and_eager_modes()
  def testNestableAnonymous(self):

    # The case where no explicit names are specified. We make up unique names,
    # and these should match the variable names.
    class ParentNetwork(network.Network):

      def __init__(self):
        super(ParentNetwork, self).__init__()
        self.first = self.track_layer(MyNetwork())
        self.second = self.track_layer(MyNetwork())

      def call(self, x):
        return self.second(self.first(x))

    one = constant_op.constant([[1.]])
    net = ParentNetwork()
    net(one)
    self.assertStartsWith(expected_start="parent_network/my_network/dense",
                          actual=net.trainable_weights[0].name)
    self.assertStartsWith(expected_start="parent_network/my_network/dense",
                          actual=net.first.trainable_weights[0].name)
    self.assertStartsWith(expected_start="parent_network/my_network_1/dense",
                          actual=net.trainable_weights[1].name)
    self.assertStartsWith(expected_start="parent_network/my_network_1/dense",
                          actual=net.second.trainable_weights[0].name)
    self.assertEqual("parent_network", net.name)
    self.assertEqual("my_network", net.first.name)
    self.assertEqual("my_network_1", net.second.name)

    net2 = ParentNetwork()
    net2(one)
    self.assertStartsWith(expected_start="parent_network_1/my_network/dense",
                          actual=net2.trainable_weights[0].name)
    self.assertStartsWith(expected_start="parent_network_1/my_network/dense",
                          actual=net2.first.trainable_weights[0].name)
    self.assertStartsWith(expected_start="parent_network_1/my_network_1/dense",
                          actual=net2.trainable_weights[1].name)
    self.assertStartsWith(expected_start="parent_network_1/my_network_1/dense",
                          actual=net2.second.trainable_weights[0].name)
    self.assertEqual("parent_network_1", net2.name)
    self.assertEqual("my_network", net2.first.name)
    self.assertEqual("my_network_1", net2.second.name)

  @test_util.run_in_graph_and_eager_modes()
  def testNestableExplicit(self):

    # We have explicit network names and everything is globally unique.
    class ParentNetwork(network.Network):

      def __init__(self):
        super(ParentNetwork, self).__init__(name="unique_parent_name")
        self.first = self.track_layer(
            MyNetwork(name="first_unique_child_name"))
        self.second = self.track_layer(
            MyNetwork(name="second_unique_child_name"))

      def call(self, x):
        return self.second(self.first(x))

    one = constant_op.constant([[1.]])
    net = ParentNetwork()
    net(one)
    self.assertStartsWith(
        expected_start="unique_parent_name/first_unique_child_name/dense",
        actual=net.trainable_weights[0].name)
    self.assertStartsWith(
        expected_start="unique_parent_name/second_unique_child_name/dense",
        actual=net.trainable_weights[1].name)
    self.assertEqual("unique_parent_name", net.name)
    self.assertEqual("first_unique_child_name", net.first.name)
    self.assertEqual("second_unique_child_name", net.second.name)

  @test_util.run_in_graph_and_eager_modes()
  def testLayerNetworkNameInteractions(self):

    # Same base name as core.Dense; Networks and non-Network Layers with the
    # same base name should use the same numbering system.
    class Dense(network.Network):

      def __init__(self):
        super(Dense, self).__init__()
        self.first = self.track_layer(core.Dense(1, use_bias=False))

      def call(self, x):
        return self.first(x)

    class MixedLayerNetwork(network.Network):

      def __init__(self):
        super(MixedLayerNetwork, self).__init__()
        self.first = self.track_layer(core.Dense(1, use_bias=False))
        self.second = self.track_layer(core.Dense(1, use_bias=False))
        self.third = self.track_layer(Dense())
        self.fourth = self.track_layer(core.Dense(1, use_bias=False))
        self.fifth = self.track_layer(core.Dense(1, use_bias=False))

      def call(self, x):
        return self.fifth(self.fourth(self.third(self.second(self.first(x)))))

    one = constant_op.constant([[1.]])
    net = MixedLayerNetwork()
    net(one)
    self.assertEqual("dense", net.first.name)
    self.assertEqual("dense_1", net.second.name)
    self.assertEqual("dense_2", net.third.name)
    self.assertEqual("dense_3", net.fourth.name)
    self.assertEqual("dense_4", net.fifth.name)
    # Note that this is _not_ the default naming behavior for Layers. Layers
    # which are added to Networks follow Network variable naming conventions
    # (i.e. variable names = network name unless variable sharing). Nested
    # Layers revert to Layer behavior.
    self.assertStartsWith(expected_start="mixed_layer_network/dense/",
                          actual=net.trainable_weights[0].name)
    self.assertStartsWith(expected_start="mixed_layer_network/dense_1/",
                          actual=net.trainable_weights[1].name)
    self.assertStartsWith(expected_start="mixed_layer_network/dense_2/",
                          actual=net.trainable_weights[2].name)
    self.assertStartsWith(expected_start="mixed_layer_network/dense_3/",
                          actual=net.trainable_weights[3].name)
    self.assertStartsWith(expected_start="mixed_layer_network/dense_4/",
                          actual=net.trainable_weights[4].name)
    self.assertEqual("mixed_layer_network", net.name)

  @test_util.run_in_graph_and_eager_modes()
  def testNestableExplicitCollisions(self):

    # We have explicit network names and they are unique within the layer
    # they're added to.
    class ParentNetwork(network.Network):

      def __init__(self):
        super(ParentNetwork, self).__init__(name="nonunique_name")
        self.first = self.track_layer(
            MyNetwork(name="nonunique_name"))
        self.second = self.track_layer(
            MyNetwork(name="second_unique_child_name"))

      def call(self, x):
        return self.second(self.first(x))

    one = constant_op.constant([[1.]])
    net = ParentNetwork()
    net(one)
    self.assertStartsWith(
        expected_start="nonunique_name/nonunique_name/dense",
        actual=net.trainable_weights[0].name)
    self.assertStartsWith(
        expected_start="nonunique_name/second_unique_child_name/dense",
        actual=net.trainable_weights[1].name)
    self.assertEqual("nonunique_name", net.name)
    self.assertEqual("nonunique_name", net.first.name)
    self.assertEqual("second_unique_child_name", net.second.name)

  @test_util.run_in_graph_and_eager_modes()
  def testNestableExplicitWithAnonymousParent(self):

    # A parent network is instantiated multiple times with explicitly named
    # children. We shouldn't throw any name errors.
    class ParentNetwork(network.Network):

      def __init__(self):
        super(ParentNetwork, self).__init__()
        self.first = self.track_layer(
            MyNetwork(name="first_unique_child_name"))
        self.second = self.track_layer(
            MyNetwork(name="second_unique_child_name"))

      def call(self, x):
        return self.second(self.first(x))

    one = constant_op.constant([[1.]])
    net = ParentNetwork()
    net(one)
    self.assertStartsWith(
        expected_start="parent_network/first_unique_child_name/dense/",
        actual=net.trainable_weights[0].name)
    self.assertStartsWith(
        expected_start="parent_network/second_unique_child_name/dense/",
        actual=net.trainable_weights[1].name)
    self.assertEqual("parent_network", net.name)
    self.assertEqual("first_unique_child_name", net.first.name)
    self.assertEqual("second_unique_child_name", net.second.name)

    net2 = ParentNetwork()
    net2(one)
    self.assertStartsWith(
        expected_start="parent_network_1/first_unique_child_name/dense",
        actual=net2.trainable_weights[0].name)
    self.assertStartsWith(
        expected_start="parent_network_1/second_unique_child_name/dense",
        actual=net2.trainable_weights[1].name)
    self.assertEqual("parent_network_1", net2.name)
    self.assertEqual("first_unique_child_name", net2.first.name)
    self.assertEqual("second_unique_child_name", net2.second.name)

  @test_util.run_in_graph_and_eager_modes()
  def testNestableExplicitSameLayerCollisions(self):

    # We have explicit network names and they are _not_ unique within the layer
    # they're added to. Error.
    class ParentNetwork(network.Network):

      def __init__(self):
        super(ParentNetwork, self).__init__(name="unique_parent_name")
        self.first = self.track_layer(MyNetwork(name="nonunique_name"))
        self.second = self.track_layer(MyNetwork(name="nonunique_name"))

      def call(self, x):
        return self.second(self.first(x))

    with self.assertRaisesRegexp(ValueError, "nonunique_name"):
      ParentNetwork()

  @test_util.run_in_graph_and_eager_modes()
  def testAnonymousVariableSharing(self):

    # Two "owned" Networks
    class FirstParentNetwork(network.Network):

      def __init__(self):
        super(FirstParentNetwork, self).__init__()
        self.first = self.track_layer(MyNetwork())
        self.second = self.track_layer(MyNetwork())

      def call(self, x):
        return self.second(self.first(x))

    one = constant_op.constant([[1.]])
    net = FirstParentNetwork()
    net(one)

    # One Network shared with FirstParentNetwork, one owned Network. Same name,
    # but this is OK because only one is owned. This name collision is
    # avoidable; we could have looked at the base_name of the non-owned Network
    # and incremented our naming based on that.
    class SecondParentNetwork(network.Network):

      def __init__(self):
        super(SecondParentNetwork, self).__init__()
        self.first = self.track_layer(net.first)
        self.second = self.track_layer(MyNetwork())

      def call(self, x):
        return self.second(self.first(x))

    net2 = SecondParentNetwork()
    net2(one)

    self.assertStartsWith(
        expected_start="first_parent_network/my_network/dense/",
        actual=net2.trainable_weights[0].name)
    self.assertStartsWith(
        expected_start="second_parent_network/my_network/dense/",
        actual=net2.trainable_weights[1].name)
    self.assertEqual("second_parent_network", net2.name)
    self.assertTrue(net2.first is net.first)
    self.assertEqual("my_network", net2.first.name)
    self.assertEqual("my_network", net2.second.name)

    # No name collision; the owned Network is added first and has a different
    # name than the shared Network.
    class ThirdParentNetwork(network.Network):

      def __init__(self):
        super(ThirdParentNetwork, self).__init__()
        self.first = self.track_layer(MyNetwork())
        self.second = self.track_layer(net.second)

      def call(self, x):
        return self.second(self.first(x))

    net3 = ThirdParentNetwork()
    net3(one)

    self.assertStartsWith(
        expected_start="third_parent_network/my_network/dense",
        actual=net3.trainable_weights[0].name)
    self.assertStartsWith(
        expected_start="first_parent_network/my_network_1/dense",
        actual=net3.trainable_weights[1].name)
    self.assertEqual("third_parent_network", net3.name)
    self.assertTrue(net3.second is net.second)
    self.assertEqual("my_network", net3.first.name)
    self.assertEqual("my_network_1", net3.second.name)

    # "Unavoidable" same-name Layer. The owned name is added first (fixed), then
    # a shared Network is added with the same name.
    class FourthParentNetwork(network.Network):

      def __init__(self):
        super(FourthParentNetwork, self).__init__()
        self.first = self.track_layer(MyNetwork())
        self.second = self.track_layer(net.first)

      def call(self, x):
        return self.second(self.first(x))

    net4 = FourthParentNetwork()
    net4(one)

    self.assertStartsWith(
        expected_start="fourth_parent_network/my_network/dense/",
        actual=net4.trainable_weights[0].name)
    self.assertStartsWith(
        expected_start="first_parent_network/my_network/dense/",
        actual=net4.trainable_weights[1].name)
    self.assertEqual("fourth_parent_network", net4.name)
    self.assertTrue(net4.second is net.first)
    self.assertEqual("my_network", net4.first.name)
    self.assertEqual("my_network", net4.second.name)

  @test_util.run_in_graph_and_eager_modes()
  def testRecursiveLayerRenaming(self):
    core.Dense(1)  # Under default Layer naming, would change subsequent names.

    class NetworkWithLayerChildren(network.Network):

      def __init__(self):
        super(NetworkWithLayerChildren, self).__init__()
        self.first = self.track_layer(core.Dense(1, use_bias=False))
        self.second = self.track_layer(core.Dense(1, use_bias=False))

      def call(self, x):
        return self.second(self.first(x))

    class ParentNetwork(network.Network):

      def __init__(self):
        super(ParentNetwork, self).__init__()
        self.first = self.track_layer(NetworkWithLayerChildren())
        self.second = self.track_layer(NetworkWithLayerChildren())

      def call(self, x):
        return self.second(self.first(x))

    net = ParentNetwork()
    one = constant_op.constant([[1.]])
    net(one)

    self.assertStartsWith(
        expected_start=("parent_network/network_with_layer_children/"
                        "dense/"),
        actual=net.trainable_weights[0].name)
    self.assertStartsWith(
        expected_start=("parent_network/network_with_layer_children/"
                        "dense_1/"),
        actual=net.trainable_weights[1].name)
    self.assertStartsWith(
        expected_start=("parent_network/network_with_layer_children_1/"
                        "dense/"),
        actual=net.trainable_weights[2].name)
    self.assertStartsWith(
        expected_start=("parent_network/network_with_layer_children_1/"
                        "dense_1/"),
        actual=net.trainable_weights[3].name)
    self.assertEqual("parent_network", net.name)
    self.assertEqual("network_with_layer_children", net.first.name)
    self.assertEqual("network_with_layer_children_1", net.second.name)
    self.assertEqual("dense", net.first.first.name)
    self.assertEqual("dense_1", net.first.second.name)
    self.assertEqual("dense", net.second.first.name)
    self.assertEqual("dense_1", net.second.second.name)

  @test_util.run_in_graph_and_eager_modes()
  def testCallInDifferentOrderThanConstruct(self):
    shared_network = MyNetwork()

    class FirstNetwork(network.Network):

      def __init__(self):
        super(FirstNetwork, self).__init__()
        self.first = self.track_layer(shared_network)
        self.second = self.track_layer(MyNetwork())

      def call(self, x):
        return self.second(self.first(x))

    class SecondNetwork(network.Network):

      def __init__(self):
        super(SecondNetwork, self).__init__()
        self.first = self.track_layer(shared_network)
        self.second = self.track_layer(MyNetwork())

      def call(self, x):
        return self.second(self.first(x))

    net1 = FirstNetwork()
    net2 = SecondNetwork()

    one = constant_op.constant([[1.]])
    net2(one)
    net1(one)

    self.assertStartsWith(
        expected_start="first_network/my_network/dense/",
        actual=net1.trainable_weights[0].name)
    self.assertStartsWith(
        expected_start="first_network/my_network_1/dense/",
        actual=net1.trainable_weights[1].name)
    self.assertStartsWith(
        expected_start="first_network/my_network/dense/",
        actual=net2.trainable_weights[0].name)
    self.assertStartsWith(
        expected_start="second_network/my_network/dense/",
        actual=net2.trainable_weights[1].name)
    self.assertTrue(net1.trainable_weights[0] is net2.trainable_weights[0])
    self.assertEqual("first_network", net1.name)
    self.assertEqual("my_network", net1.first.name)
    self.assertEqual("my_network_1", net1.second.name)
    self.assertTrue(net2.first is net1.first)
    self.assertEqual("my_network", net2.second.name)

  @test_util.run_in_graph_and_eager_modes()
  def testLayerCallInDifferentOrderThanConstruct(self):
    # Same idea as testCallInDifferentOrderThanConstruct, but this time with a
    # non-Network Layer shared between two Networks rather than a
    # Network. Naming should follow the same rules.
    shared_layer = core.Dense(1, use_bias=False)

    class FirstNetwork(network.Network):

      def __init__(self):
        super(FirstNetwork, self).__init__()
        self.first = self.track_layer(shared_layer)
        self.second = self.track_layer(core.Dense(1, use_bias=False))

      def call(self, x):
        return self.second(self.first(x))

    class SecondNetwork(network.Network):

      def __init__(self):
        super(SecondNetwork, self).__init__()
        self.first = self.track_layer(shared_layer)
        self.second = self.track_layer(core.Dense(1, use_bias=False))

      def call(self, x):
        return self.second(self.first(x))

    net1 = FirstNetwork()
    net2 = SecondNetwork()

    one = constant_op.constant([[1.]])
    net2(one)
    net1(one)

    self.assertStartsWith(
        expected_start="first_network/dense/",
        actual=net1.trainable_weights[0].name)
    self.assertStartsWith(
        expected_start="first_network/dense_1/",
        actual=net1.trainable_weights[1].name)
    self.assertStartsWith(
        expected_start="first_network/dense/",
        actual=net2.trainable_weights[0].name)
    self.assertStartsWith(
        expected_start="second_network/dense/",
        actual=net2.trainable_weights[1].name)
    self.assertTrue(net1.trainable_weights[0] is net2.trainable_weights[0])
    self.assertEqual("first_network", net1.name)
    self.assertEqual("dense", net1.first.name)
    self.assertEqual("dense_1", net1.second.name)
    self.assertTrue(net2.first is net1.first)
    self.assertEqual("dense", net2.second.name)

  @test_util.run_in_graph_and_eager_modes()
  def testLayerAlreadyBuilt(self):
    one = constant_op.constant([[1.]])
    core.Dense(1, use_bias=False)  # pre-built layers use global naming
    one = constant_op.constant([[1.]])
    core.Dense(1, use_bias=False)(one)
    shared_layer = core.Dense(1, use_bias=False)
    shared_layer(one)

    class FirstNetwork(network.Network):

      def __init__(self):
        super(FirstNetwork, self).__init__()
        self.first = self.track_layer(shared_layer)
        self.second = self.track_layer(core.Dense(1, use_bias=False))

      def call(self, x):
        return self.second(self.first(x))

    net = FirstNetwork()
    net(one)

    self.assertStartsWith(
        expected_start="dense_1/",  # Pre-built layers have variable names which
                                    # do not match their layer names.
        actual=net.trainable_weights[0].name)
    self.assertStartsWith(
        expected_start="first_network/dense/",
        actual=net.trainable_weights[1].name)
    self.assertTrue(
        net.trainable_weights[0] is shared_layer.trainable_weights[0])
    self.assertEqual("first_network", net.name)
    self.assertEqual("dense_3", net.first.name)
    self.assertEqual("dense", net.second.name)


class SequentialTest(test.TestCase):

  @test_util.assert_no_garbage_created
  def testTwoLayers(self):
    # Create a sequential network with one layer.
    net = network.Sequential([core.Dense(1, use_bias=False)])

    # Set that layer's weights so it multiplies by 3
    l1 = net.get_layer(index=0)
    net(constant_op.constant([[2.0]]))  # Create l1's variables
    self.assertEqual(1, len(l1.trainable_variables))
    l1.trainable_variables[0].assign([[3.0]])
    self.assertEqual(21.0, net(constant_op.constant([[7.0]])).numpy())

    # Add a second layer to the network.
    l2 = core.Dense(1, use_bias=False)
    net.add(l2)

    # Set the second layer's weights so it multiplies by 11
    net(constant_op.constant([[2.0]]))  # Create l2's variables
    self.assertEqual(1, len(l2.trainable_variables))
    l2.trainable_variables[0].assign([[11.0]])
    self.assertEqual(231.0, net(constant_op.constant([[7.0]])).numpy())

  @test_util.assert_no_garbage_created
  def testFunctions(self):
    # Create a sequential network with one function.
    net = network.Sequential([nn_ops.relu])
    two = constant_op.constant(2.0)
    self.assertEqual(2.0, net(two).numpy())
    self.assertEqual(0.0, net(-two).numpy())
    # Add a second function.
    net.add(math_ops.negative)
    self.assertEqual(-2.0, net(two).numpy())

  @test_util.assert_no_garbage_created
  def testTrainingLayer(self):
    net = network.Sequential([core.Dropout(0.99999)])
    two = constant_op.constant(2.0)
    self.assertEqual(2.0, net(two).numpy())
    self.assertEqual(2.0, net(two, training=False).numpy())
    for _ in range(20):
      with_dropout = net(two, training=True).numpy()
      self.assertIn(with_dropout, [0.0, 2.0])
      if with_dropout == 0.0:
        return
    # Should only fail spuriously 1 in 10^100 runs.
    self.fail("Didn't see dropout happen after 20 tries.")

  @test_util.assert_no_garbage_created
  def testTrainingFunction(self):
    # Output depends on value of "training".
    def add_training(input_value, training=None):
      if training is None:
        return input_value
      elif training:
        return input_value + 1
      return input_value - 1

    # Passing a "training" argument to double would cause an error.
    def double(input_value):
      return 2 * input_value

    net = network.Sequential([add_training, double])
    two = constant_op.constant(2)
    self.assertEqual(4, net(two).numpy())
    self.assertEqual(2, net(two, training=False).numpy())
    self.assertEqual(6, net(two, training=True).numpy())


if __name__ == "__main__":
  test.main()
