# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `tf.Module`."""

import abc
import collections
import itertools

from absl.testing import parameterized
import six

from tensorflow.python import tf2
from tensorflow.python.distribute import ps_values
from tensorflow.python.distribute import tpu_values
from tensorflow.python.distribute import values as distributed_values
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import type_spec
from tensorflow.python.module import module
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class TestModuleNaming(test_util.TensorFlowTestCase):

  def test_single_name(self):
    mod = module.Module(name="simple")
    self.assertEqual(mod.name, "simple")
    self.assertEqual(mod.name_scope.name, "simple/")

  def test_construct_in_scope(self):
    with ops.name_scope("foo", skip_on_eager=False):
      mod = module.Module(name="bar")
    self.assertEqual(mod.name, "bar")
    self.assertEqual(mod.name_scope.name, "foo/bar/")

  def test_enters_name_scope_in_call(self):
    mod = ReturnsNameScopeModule()
    for _ in range(3):
      self.assertEqual(mod(), mod.name_scope.name)

  def test_enters_name_scope_in_other_method(self):
    mod = ReturnsNameScopeModule()
    for _ in range(3):
      self.assertEqual(mod.alternative_forward(), mod.name_scope.name)

  def test_subclassed_module(self):
    mod = SubclassedReturnsNameScopeModule()
    for _ in range(3):
      self.assertEqual(mod.alternative_forward(), mod.name_scope.name)
      self.assertEqual(mod.alternative_alternative_forward(),
                       mod.name_scope.name)

  def test_submodule_created_late(self):
    m = TreeModule()
    self.assertEqual(m.name, "tree_module")
    self.assertEqual(m.name_scope.name, "tree_module/")
    leaf1 = m.new_leaf()
    self.assertEqual(leaf1.name, "tree_module")
    self.assertEqual(leaf1.name_scope.name, "tree_module/tree_module/")

  def test_does_not_evaluate_property_methods(self):
    mod = PropertyThrowsWhenCalledModule()
    with self.assertRaises(AssertionError):
      mod.raise_assertion_error  # pylint: disable=pointless-statement

  def test_overridden_name_scope(self):
    mod = ModuleOverridingNameScope()
    self.assertEqual(mod(), mod.name_scope.name)
    self.assertEqual(mod.alternative_forward(), mod.name_scope.name)

  def test_patched_callable(self):
    with ops.name_scope("foo", skip_on_eager=False):
      mod = module.Module(name="bar")
    mod.foo = get_name_scope
    # `foo` is not a method so we do not re-enter the name scope.
    self.assertEqual(mod.foo(), "")

  def test_property(self):
    mod = PropertyModule()
    mod.some_property = None, None  # None, None for the linter.
    getter_scope_name, setter_scope_name = mod.some_property
    self.assertEqual(getter_scope_name, "property_module/")
    self.assertEqual(setter_scope_name, "property_module/")

  def test_property_no_name_scope(self):
    mod = PropertyModule()
    mod.no_name_scope_property = None, None  # None, None for the linter.
    getter_scope_name, setter_scope_name = mod.no_name_scope_property
    self.assertEqual(getter_scope_name, "")
    self.assertEqual(setter_scope_name, "")

  def test_invalid_name(self):
    msg = ".* is not a valid module name"
    with self.assertRaisesRegex(ValueError, msg):
      module.Module(name="$Foo")

  @test_util.run_in_graph_and_eager_modes
  def test_modules_not_numbered_in_eager(self):
    if not context.executing_eagerly():
      self.skipTest("Eager specific")

    mod = RecursiveModule(2)
    self.assertEqual(mod.name_scope.name, "badger/")
    self.assertEqual(mod.child.name_scope.name, "badger/badger/")

    mod = RecursiveModule(2)
    self.assertEqual(mod.name_scope.name, "badger/")
    self.assertEqual(mod.child.name_scope.name, "badger/badger/")

  @test_util.run_in_graph_and_eager_modes
  def test_module_numbering_in_graph(self):
    if context.executing_eagerly():
      self.skipTest("Graph specific")

    mod = RecursiveModule(2)
    self.assertEqual(mod.name_scope.name, "badger/")
    self.assertEqual(mod.child.name_scope.name, "badger/badger/")

    mod = RecursiveModule(2)
    self.assertEqual(mod.name_scope.name, "badger_1/")
    self.assertEqual(mod.child.name_scope.name, "badger_1/badger/")

  def test_ctor_error_closes_name_scope(self):
    with self.assertRaises(ErrorModuleError):
      # If super constructor is called then a name scope is opened then an error
      # is thrown. The metaclass should handle this and close the namescope
      # before re-throwing the exception.
      ErrorModule(call_super=True)

    self.assertEqual("", get_name_scope())

  def test_ctor_error_handles_ctor_not_opening_name_scope(self):
    with self.assertRaises(ErrorModuleError):
      # If super ctor is not called then the name scope isn't opened. We need to
      # ensure that this doesn't trigger an exception (e.g. the metaclass trying
      # to __exit__ a non-existent name scope).
      ErrorModule(call_super=False)

    self.assertEqual("", get_name_scope())

  def test_forward_method_closes_name_scope(self):
    mod = ErrorModule(call_super=True, raise_in_constructor=False)
    with self.assertRaises(ErrorModuleError):
      mod()

    self.assertEqual("", get_name_scope())

  def test_get_attr_doesnt_enter_name_scope(self):
    scope_names = []

    class GetAttrModule(module.Module):

      def __getattr__(self, name):
        scope_names.append((name, get_name_scope()))
        return super(GetAttrModule, self).__getattr__(name)

    mod = GetAttrModule()
    with self.assertRaises(AttributeError):
      mod.does_not_exist  # pylint: disable=pointless-statement
    self.assertIn(("does_not_exist", ""), scope_names)

  def test_get_attribute_doesnt_enter_name_scope(self):
    scope_names = []

    class GetAttributeModule(module.Module):

      def __getattribute__(self, name):
        scope_names.append((name, get_name_scope()))
        return super(GetAttributeModule, self).__getattribute__(name)

    mod = GetAttributeModule()
    with self.assertRaises(AttributeError):
      mod.does_not_exist  # pylint: disable=pointless-statement
    self.assertIn(("does_not_exist", ""), scope_names)


class VariableNamingTest(test_util.TensorFlowTestCase):

  def test_variable_names(self):
    mod = RecursiveModule(3)
    self.assertEqual(mod.w.name, "badger/mushroom:0")
    self.assertEqual(mod.child.w.name, "badger/badger/mushroom:0")
    self.assertEqual(mod.child.child.w.name, "badger/badger/badger/mushroom:0")


class NameScopeTest(test_util.TensorFlowTestCase):

  @test_util.run_deprecated_v1
  def test_not_memoized_in_tf1(self):
    if tf2.enabled():
      self.skipTest("Requires TF1")

    mod = module.Module(name="name")
    name_scope_1 = mod.name_scope
    name_scope_2 = mod.name_scope
    self.assertIsNot(name_scope_1, name_scope_2)
    self.assertEqual(name_scope_1.name, name_scope_2.name)

  def test_memoized_in_tf2(self):
    if not tf2.enabled():
      self.skipTest("Requires TF2")

    mod = module.Module(name="name")
    name_scope_1 = mod.name_scope
    name_scope_2 = mod.name_scope
    self.assertIs(name_scope_1, name_scope_2)


class VariableTrackingTest(test_util.TensorFlowTestCase):

  def test_variables(self):
    m = RecursiveModule(3)
    self.assertEqual(m.variables, (m.w, m.child.w, m.child.child.w))
    self.assertEqual(m.child.variables, (m.child.w, m.child.child.w))
    self.assertEqual(m.child.child.variables, (m.child.child.w,))

  def test_trainable_variables(self):
    m = RecursiveModule(3)
    self.assertEqual(m.trainable_variables,
                     (m.w, m.child.w, m.child.child.w))
    self.assertEqual(m.child.trainable_variables,
                     (m.child.w, m.child.child.w))
    self.assertEqual(m.child.child.trainable_variables, (m.child.child.w,))

  def test_trainable_variables_ignores_non_trainable(self):
    m = RecursiveModule(3, trainable=False)
    self.assertEqual(len(m.trainable_variables), 0)
    self.assertEqual(len(m.child.trainable_variables), 0)
    self.assertEqual(len(m.child.child.trainable_variables), 0)

  def test_supports_distributed_variables(self):
    mirrored = distributed_values.MirroredVariable(
        None, [variables.Variable(1.)], variables.VariableAggregation.SUM)
    tpu = tpu_values.TPUMirroredVariable(
        strategy=None, values=[variables.Variable(42.)], aggregation=None)
    aggregating = ps_values.AggregatingVariable(
        strategy=None, v=variables.Variable(1.), aggregation=None)

    m = module.Module()
    m.a = mirrored
    m.b = tpu
    m.c = aggregating
    self.assertEqual(m.variables, (mirrored, tpu, aggregating))

  def test_composite_variable(self):

    class Spec(type_spec.TypeSpec):

      value_type = property(lambda self: CompositeVariable)

      def _component_specs(self):
        pass

      def _serialize(self):
        pass

      def _to_components(self, value):
        return value._variables

      def _from_components(self, variable_list):
        return CompositeVariable(variable_list)

    class CompositeVariable(composite_tensor.CompositeTensor):

      def __init__(self, variable_list):
        self._variables = variable_list

      @property
      def _type_spec(self):
        return Spec()

    m = module.Module()
    m.a = CompositeVariable([variables.Variable(1.), variables.Variable(2.)])
    self.assertAllEqual(m.variables, m.a._variables)


class ModuleTrackingTest(test_util.TensorFlowTestCase):

  def test_submodules(self):
    m = RecursiveModule(3)
    self.assertEqual(list(m.submodules), [m.child, m.child.child])
    self.assertEqual(list(m.child.submodules), [m.child.child])
    self.assertEqual(list(m.child.child.submodules), [])

  def test_non_ctor_submodule(self):
    m = TreeModule()
    leaf1 = m.new_leaf()
    self.assertEqual(set(m.submodules), {leaf1})
    leaf2 = m.new_leaf()
    self.assertEqual(set(m.submodules), {leaf1, leaf2})


class ForwardMethodsTest(test_util.TensorFlowTestCase):

  def testFunctionType(self):
    mod = ModuleWithFunctionAnnotatedCall()
    self.assertIsInstance(mod.forward, def_function.Function)
    self.assertIsInstance(mod.forward_ag, def_function.Function)

  def testEntersNameScope_call(self):
    mod = ModuleWithFunctionAnnotatedCall()
    self.assertEqual(self.evaluate(mod.forward()),
                     b"module_with_function_annotated_call/")
    self.assertEqual(self.evaluate(mod.forward_ag()),
                     b"module_with_function_annotated_call/")

  def testEntersNameScope_concreteFunction(self):
    mod = ModuleWithFunctionAnnotatedCall()
    self.assertEqual(self.evaluate(mod.forward.get_concrete_function()()),
                     b"module_with_function_annotated_call/")
    self.assertEqual(self.evaluate(mod.forward_ag.get_concrete_function()()),
                     b"module_with_function_annotated_call/")


class AbcTest(test_util.TensorFlowTestCase):

  def testAbstract(self):
    msg = "Can't instantiate.*abstract"
    with self.assertRaisesRegex(TypeError, msg):
      AbstractModule()  # pylint: disable=abstract-class-instantiated

  def testConcrete(self):
    mod = ConcreteModule()
    x, scope_name = mod(2.)
    self.assertEqual(x, 4.)
    self.assertEqual(scope_name, "concrete_module/")
    self.assertEqual(get_name_scope(), "")


def get_name_scope():
  with ops.name_scope("x", skip_on_eager=False) as ns:
    ns = "/".join(ns.split("/")[:-2])
    return ns + "/" if ns else ""


class ErrorModuleError(Exception):
  pass


class ErrorModule(module.Module):

  def __init__(self, call_super, raise_in_constructor=True):
    if call_super:
      super(ErrorModule, self).__init__()
    if raise_in_constructor:
      raise ErrorModuleError("Deliberate error!")

  def __call__(self):
    raise ErrorModuleError("Deliberate error!")


class RecursiveModule(module.Module):

  def __init__(self, depth, trainable=True):
    super(RecursiveModule, self).__init__(name="badger")
    with self.name_scope:
      self.child = None
      if depth > 1:
        self.child = RecursiveModule(depth - 1, trainable=trainable)
      self.w = variables.Variable(1.0, trainable=trainable, name="mushroom")


@six.add_metaclass(abc.ABCMeta)
class AbstractModule(module.Module):

  @abc.abstractmethod
  def __call__(self, x):
    pass


class ConcreteModule(AbstractModule):

  @module.Module.with_name_scope
  def __call__(self, x):
    return x ** 2, get_name_scope()


class TreeModule(module.Module):

  def __init__(self, name=None):
    super(TreeModule, self).__init__(name=name)
    self._leaves = []

  @module.Module.with_name_scope
  def new_leaf(self, name=None):
    leaf = TreeModule(name=name)
    self._leaves.append(leaf)
    return leaf


class ReturnsNameScopeModule(module.Module):

  @module.Module.with_name_scope
  def alternative_forward(self):
    return get_name_scope()

  @module.Module.with_name_scope
  def __call__(self):
    return get_name_scope()


class SubclassedReturnsNameScopeModule(ReturnsNameScopeModule):

  @module.Module.with_name_scope
  def alternative_alternative_forward(self):
    return get_name_scope()


class PropertyThrowsWhenCalledModule(module.Module):

  @property
  def raise_assertion_error(self):
    raise AssertionError


class ModuleOverridingNameScope(ReturnsNameScopeModule):

  @property
  def name_scope(self):
    return ops.name_scope("yolo/", skip_on_eager=False)


class ModuleWithFunctionAnnotatedCall(module.Module):

  @def_function.function(autograph=False)
  @module.Module.with_name_scope
  def forward(self):
    return get_name_scope()

  @def_function.function(autograph=True)
  @module.Module.with_name_scope
  def forward_ag(self):
    return get_name_scope()


class PropertyModule(module.Module):

  def __init__(self):
    super(PropertyModule, self).__init__()
    self._setter_scope_name = None

  @property
  @module.Module.with_name_scope
  def some_property(self):
    getter_scope_name = get_name_scope()
    return getter_scope_name, self._setter_scope_name

  @some_property.setter
  @module.Module.with_name_scope
  def some_property(self, my_property):
    self._setter_scope_name = get_name_scope()

  @property
  def no_name_scope_property(self):
    getter_scope_name = get_name_scope()
    return getter_scope_name, self._setter_scope_name

  @no_name_scope_property.setter
  def no_name_scope_property(self, my_property):
    self._setter_scope_name = get_name_scope()

NamedPair = collections.namedtuple("NamedPair", ("first", "second"))
mk_index_dict = lambda v: dict(enumerate(v))


class FlattenTest(parameterized.TestCase, test_util.TensorFlowTestCase):

  @parameterized.parameters(lambda v: NamedPair(*v), list, tuple, mk_index_dict)
  def test_flatten(self, container_type):
    parent = SimpleModule(container_type=container_type)
    child = parent.c

    self.assertEqual(
        list(parent._flatten(recursive=False, predicate=is_member)),
        [parent.a[0], parent.a[1], parent.z])

    self.assertEqual(
        list(parent._flatten(predicate=is_member)),
        [parent.a[0], parent.a[1], parent.z, child.a[0], child.a[1], child.z])

  def test_attribute_traversal_key(self):
    mod = LayerModule()
    self.assertEqual(
        mod.variables,
        mod._trainable_variables + mod._non_trainable_variables + [mod._bonus])

  def test_attributes_to_ignore(self):
    class DangerousModule(module.Module):
      _TF_MODULE_IGNORED_PROPERTIES = frozenset(itertools.chain(
          ("dangerous_submodule", "dangerous_variable"),
          module.Module._TF_MODULE_IGNORED_PROPERTIES
      ))

    mod = DangerousModule()
    mod.dangerous_submodule = module.Module()
    mod.dangerous_variable = variables.Variable(1.)
    mod.normal_variable = variables.Variable(2.)

    self.assertEmpty(mod.submodules)
    self.assertLen(mod.variables, 1)
    self.assertEqual(mod.variables[0], mod.normal_variable)

  def test_with_path(self):
    mod = module.Module()
    mod.w = variables.Variable(1.)
    mod.encoder = module.Module()
    mod.encoder.w = [({"k": mod.w}, {"k": mod.w})]
    mod.decoder = mod.encoder

    state_dict = dict(
        mod._flatten(with_path=True, predicate=module._is_variable))

    self.assertEqual(state_dict,
                     {("w",): mod.w,
                      ("encoder", "w", 0, 0, "k"): mod.encoder.w[0][0]["k"],
                      ("encoder", "w", 0, 1, "k"): mod.encoder.w[0][1]["k"],
                      ("decoder", "w", 0, 0, "k"): mod.decoder.w[0][0]["k"],
                      ("decoder", "w", 0, 1, "k"): mod.decoder.w[0][1]["k"]},)

  def test_cycles_with_path(self):
    mod = module.Module()
    mod.w = variables.Variable(1.)
    mod.encoder = module.Module()
    mod.encoder.w = [({"k": mod.w}, {"k": mod.w})]
    mod.decoder = mod.encoder

    # This introduces two cycles: on mod.encoder.mod and mod.decoder.mod.
    mod.decoder.mod = mod

    state_dict = dict(
        mod._flatten(with_path=True, predicate=module._is_variable))

    self.assertEqual(state_dict,
                     {("w",): mod.w,
                      ("encoder", "mod", "w"): mod.encoder.mod.w,
                      ("decoder", "mod", "w"): mod.decoder.mod.w,
                      ("encoder", "w", 0, 0, "k"): mod.encoder.w[0][0]["k"],
                      ("encoder", "w", 0, 1, "k"): mod.encoder.w[0][1]["k"],
                      ("decoder", "w", 0, 0, "k"): mod.decoder.w[0][0]["k"],
                      ("decoder", "w", 0, 1, "k"): mod.decoder.w[0][1]["k"]},)

  def test_raises_error_with_path(self):
    non_orderable = object

    m = module.Module()
    m.layers = {non_orderable(): None, non_orderable(): None}
    with self.assertRaisesRegex(ValueError,
                                "Error processing property 'layers'"):
      m.variables  # pylint: disable=pointless-statement


class LayerModule(module.Module):

  def __init__(self):
    super(LayerModule, self).__init__()
    self._trainable_variables = [
        variables.Variable(1., name="a"),
        variables.Variable(2., name="b"),
    ]
    self._non_trainable_variables = [
        variables.Variable(3., name="c"),
        variables.Variable(4., name="d"),
    ]
    self._bonus = variables.Variable(5., name="e")

  @property
  def variables(self):
    def key_function(name):
      indexes = {"_trainable_variables": 0, "_non_trainable_variables": 1}
      return indexes.get(name, 2), name

    return list(
        self._flatten(
            predicate=module._is_variable,
            attribute_traversal_key=key_function))


class MemberType(object):
  """A simple type to search for."""
  pass


class SimpleModule(module.Module):

  def __init__(self, create_child=True, container_type=list):
    super(SimpleModule, self).__init__()
    self.z = MemberType()
    self.a = container_type([MemberType(), MemberType()])
    if create_child:
      self.c = SimpleModule(create_child=False)

is_member = lambda v: isinstance(v, MemberType)

if __name__ == "__main__":
  test.main()
