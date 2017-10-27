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
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.layers import core
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope


# pylint: disable=not-callable
class MyNetwork(network.Network):

  def __init__(self, name=None):
    super(MyNetwork, self).__init__(name=name)
    self.l1 = self.track_layer(core.Dense(1, use_bias=False))

  def call(self, x):
    return self.l1(x)


class NetworkTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes()
  def testTrainableAttribute(self):
    net = network.Network()
    self.assertTrue(net.trainable)
    with self.assertRaises(AttributeError):
      net.trainable = False
    self.assertTrue(net.trainable)

  @test_util.run_in_graph_and_eager_modes()
  def testNetworkCall(self):
    net = MyNetwork(name="abcd")
    net(constant_op.constant([[2.0]]))  # Force variables to be created.
    self.assertEqual(1, len(net.trainable_variables))
    self.evaluate(net.trainable_variables[0].assign([[17.0]]))
    # TODO(josh11b): Support passing Python values to networks.
    result = net(constant_op.constant([[2.0]]))
    self.assertEqual(34.0, self.evaluate(result))

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

  @test_util.run_in_graph_and_eager_modes()
  def testAnonymousNoNameInitially(self):
    net = MyNetwork()
    with self.assertRaisesRegexp(ValueError, "does not yet have a final name"):
      net.name  # pylint: disable=pointless-statement

  @test_util.run_in_graph_and_eager_modes()
  def testExplicitHasNameInitially(self):
    net = MyNetwork(name="abcd")
    self.assertEqual("abcd", net.name)

  @test_util.run_in_graph_and_eager_modes()
  def testUsingResourceVariables(self):
    net = MyNetwork()
    net(constant_op.constant([[0.]]))
    self.assertIsInstance(net.trainable_weights[0],
                          resource_variable_ops.ResourceVariable)

  @test_util.run_in_graph_and_eager_modes()
  def testDuplicateNameError(self):
    one = constant_op.constant([[1.]])
    net = MyNetwork(name="foo")
    net(one)
    with self.assertRaisesRegexp(
        ValueError, "named 'foo' already exists"):
      net1 = MyNetwork(name="foo")
      net1(one)

  @test_util.run_in_graph_and_eager_modes()
  def testWrappingInVariableScope(self):
    with variable_scope.variable_scope("outside_scope"):
      net = MyNetwork()
      one = constant_op.constant([[1.]])
      with self.assertRaisesRegexp(
          ValueError,
          ("Creating Networks inside named variable_scopes is currently not "
           "supported")):
        net(one)
      # Alternatively, we could re-name the Network to match the variable_scope:
      # self.assertEqual("outside_scope/my_network_1", net.name)
      # self.assertStartsWith(
      #     expected_start="outside_scope/my_network_1/dense/",
      #     actual=net.trainable_weights[0].name)

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
    self.assertStartsWith(expected_start="parent_network_1/explicit_name/",
                          actual=net.trainable_weights[0].name)
    self.assertEqual("explicit_name", net.first.name)

  @test_util.run_in_graph_and_eager_modes()
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

  @test_util.run_in_graph_and_eager_modes()
  def testReasonableSlashError(self):
    with self.assertRaisesRegexp(
        ValueError, "not allowed in Network names"):
      MyNetwork(name="slash/slash")

  @test_util.run_in_graph_and_eager_modes()
  def testNoVariableScopeNames(self):
    with self.assertRaisesRegexp(
        ValueError, "VariableScopes are not valid Network names"):
      with variable_scope.variable_scope("some_scope") as vs:
        MyNetwork(name=vs)

  @test_util.run_in_graph_and_eager_modes()
  def testVariableScopeNameCollision(self):
    with variable_scope.variable_scope("abcd"):
      pass
    with self.assertRaisesRegexp(
        ValueError, "or a variable_scope was created with this name"):
      net = MyNetwork(name="abcd")
      one = constant_op.constant([[1.]])
      net(one)

  @test_util.run_in_graph_and_eager_modes()
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
    self.assertEqual("dense_1", net1.l1.name)
    self.assertEqual("dense_1", net2.l1.name)
    self.evaluate(net1.trainable_weights[0].assign([[1.]]))
    self.evaluate(net2.trainable_weights[0].assign([[2.]]))
    self.assertEqual(2., self.evaluate(net2.trainable_weights[0]))
    self.assertEqual(1., self.evaluate(net1.trainable_weights[0]))
    self.assertStartsWith(expected_start="my_network_1/dense_1/",
                          actual=net1.trainable_weights[0].name)
    self.assertStartsWith(expected_start="my_network_2/dense_1/",
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
    self.assertStartsWith(expected_start="parent_network_1/my_network_1/dense",
                          actual=net.trainable_weights[0].name)
    self.assertStartsWith(expected_start="parent_network_1/my_network_1/dense",
                          actual=net.first.trainable_weights[0].name)
    self.assertStartsWith(expected_start="parent_network_1/my_network_2/dense",
                          actual=net.trainable_weights[1].name)
    self.assertStartsWith(expected_start="parent_network_1/my_network_2/dense",
                          actual=net.second.trainable_weights[0].name)
    self.assertEqual("parent_network_1", net.name)
    self.assertEqual("my_network_1", net.first.name)
    self.assertEqual("my_network_2", net.second.name)

    net2 = ParentNetwork()
    net2(one)
    self.assertStartsWith(expected_start="parent_network_2/my_network_1/dense",
                          actual=net2.trainable_weights[0].name)
    self.assertStartsWith(expected_start="parent_network_2/my_network_1/dense",
                          actual=net2.first.trainable_weights[0].name)
    self.assertStartsWith(expected_start="parent_network_2/my_network_2/dense",
                          actual=net2.trainable_weights[1].name)
    self.assertStartsWith(expected_start="parent_network_2/my_network_2/dense",
                          actual=net2.second.trainable_weights[0].name)
    self.assertEqual("parent_network_2", net2.name)
    self.assertEqual("my_network_1", net2.first.name)
    self.assertEqual("my_network_2", net2.second.name)

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
    self.assertEqual("dense_1", net.first.name)
    self.assertEqual("dense_2", net.second.name)
    self.assertEqual("dense_3", net.third.name)
    self.assertEqual("dense_4", net.fourth.name)
    self.assertEqual("dense_5", net.fifth.name)
    # Note that this is _not_ the default naming behavior for Layers. Layers
    # which are added to Networks follow Network variable naming conventions
    # (i.e. variable names = network name unless variable sharing). Nested
    # Layers revert to Layer behavior.
    self.assertStartsWith(expected_start="mixed_layer_network_1/dense_1/",
                          actual=net.trainable_weights[0].name)
    self.assertStartsWith(expected_start="mixed_layer_network_1/dense_2/",
                          actual=net.trainable_weights[1].name)
    self.assertStartsWith(expected_start="mixed_layer_network_1/dense_3/",
                          actual=net.trainable_weights[2].name)
    self.assertStartsWith(expected_start="mixed_layer_network_1/dense_4/",
                          actual=net.trainable_weights[3].name)
    self.assertStartsWith(expected_start="mixed_layer_network_1/dense_5/",
                          actual=net.trainable_weights[4].name)
    self.assertEqual("mixed_layer_network_1", net.name)

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
        expected_start="parent_network_1/first_unique_child_name/dense_1/",
        actual=net.trainable_weights[0].name)
    self.assertStartsWith(
        expected_start="parent_network_1/second_unique_child_name/dense_1/",
        actual=net.trainable_weights[1].name)
    self.assertEqual("parent_network_1", net.name)
    self.assertEqual("first_unique_child_name", net.first.name)
    self.assertEqual("second_unique_child_name", net.second.name)

    net2 = ParentNetwork()
    net2(one)
    self.assertStartsWith(
        expected_start="parent_network_2/first_unique_child_name/dense",
        actual=net2.trainable_weights[0].name)
    self.assertStartsWith(
        expected_start="parent_network_2/second_unique_child_name/dense",
        actual=net2.trainable_weights[1].name)
    self.assertEqual("parent_network_2", net2.name)
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
        expected_start="first_parent_network_1/my_network_1/dense_1/",
        actual=net2.trainable_weights[0].name)
    self.assertStartsWith(
        expected_start="second_parent_network_1/my_network_1/dense_1/",
        actual=net2.trainable_weights[1].name)
    self.assertEqual("second_parent_network_1", net2.name)
    self.assertTrue(net2.first is net.first)
    self.assertEqual("my_network_1", net2.first.name)
    self.assertEqual("my_network_1", net2.second.name)

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
        expected_start="third_parent_network_1/my_network_1/dense",
        actual=net3.trainable_weights[0].name)
    self.assertStartsWith(
        expected_start="first_parent_network_1/my_network_2/dense",
        actual=net3.trainable_weights[1].name)
    self.assertEqual("third_parent_network_1", net3.name)
    self.assertTrue(net3.second is net.second)
    self.assertEqual("my_network_1", net3.first.name)
    self.assertEqual("my_network_2", net3.second.name)

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
        expected_start="fourth_parent_network_1/my_network_1/dense_1/",
        actual=net4.trainable_weights[0].name)
    self.assertStartsWith(
        expected_start="first_parent_network_1/my_network_1/dense_1/",
        actual=net4.trainable_weights[1].name)
    self.assertEqual("fourth_parent_network_1", net4.name)
    self.assertTrue(net4.second is net.first)
    self.assertEqual("my_network_1", net4.first.name)
    self.assertEqual("my_network_1", net4.second.name)

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
        expected_start=("parent_network_1/network_with_layer_children_1/"
                        "dense_1/"),
        actual=net.trainable_weights[0].name)
    self.assertStartsWith(
        expected_start=("parent_network_1/network_with_layer_children_1/"
                        "dense_2/"),
        actual=net.trainable_weights[1].name)
    self.assertStartsWith(
        expected_start=("parent_network_1/network_with_layer_children_2/"
                        "dense_1/"),
        actual=net.trainable_weights[2].name)
    self.assertStartsWith(
        expected_start=("parent_network_1/network_with_layer_children_2/"
                        "dense_2/"),
        actual=net.trainable_weights[3].name)
    self.assertEqual("parent_network_1", net.name)
    self.assertEqual("network_with_layer_children_1", net.first.name)
    self.assertEqual("network_with_layer_children_2", net.second.name)
    self.assertEqual("dense_1", net.first.first.name)
    self.assertEqual("dense_2", net.first.second.name)
    self.assertEqual("dense_1", net.second.first.name)
    self.assertEqual("dense_2", net.second.second.name)

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
        expected_start="first_network_1/my_network_1/dense_1/",
        actual=net1.trainable_weights[0].name)
    self.assertStartsWith(
        expected_start="first_network_1/my_network_2/dense_1/",
        actual=net1.trainable_weights[1].name)
    self.assertStartsWith(
        expected_start="first_network_1/my_network_1/dense_1/",
        actual=net2.trainable_weights[0].name)
    self.assertStartsWith(
        expected_start="second_network_1/my_network_1/dense_1/",
        actual=net2.trainable_weights[1].name)
    self.assertTrue(net1.trainable_weights[0] is net2.trainable_weights[0])
    self.assertEqual("first_network_1", net1.name)
    self.assertEqual("my_network_1", net1.first.name)
    self.assertEqual("my_network_2", net1.second.name)
    self.assertTrue(net2.first is net1.first)
    self.assertEqual("my_network_1", net2.second.name)

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
        expected_start="first_network_1/dense_1/",
        actual=net1.trainable_weights[0].name)
    self.assertStartsWith(
        expected_start="first_network_1/dense_2/",
        actual=net1.trainable_weights[1].name)
    self.assertStartsWith(
        expected_start="first_network_1/dense_1/",
        actual=net2.trainable_weights[0].name)
    self.assertStartsWith(
        expected_start="second_network_1/dense_1/",
        actual=net2.trainable_weights[1].name)
    self.assertTrue(net1.trainable_weights[0] is net2.trainable_weights[0])
    self.assertEqual("first_network_1", net1.name)
    self.assertEqual("dense_1", net1.first.name)
    self.assertEqual("dense_2", net1.second.name)
    self.assertTrue(net2.first is net1.first)
    self.assertEqual("dense_1", net2.second.name)

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
        expected_start="first_network_1/dense_1/",
        actual=net.trainable_weights[1].name)
    self.assertTrue(
        net.trainable_weights[0] is shared_layer.trainable_weights[0])
    self.assertEqual("first_network_1", net.name)
    self.assertEqual("dense_3", net.first.name)
    self.assertEqual("dense_1", net.second.name)


class SequentialTest(test.TestCase):

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

  def testFunctions(self):
    # Create a sequential network with one function.
    net = network.Sequential([nn_ops.relu])
    two = constant_op.constant(2.0)
    self.assertEqual(2.0, net(two).numpy())
    self.assertEqual(0.0, net(-two).numpy())
    # Add a second function.
    net.add(math_ops.negative)
    self.assertEqual(-2.0, net(two).numpy())

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
