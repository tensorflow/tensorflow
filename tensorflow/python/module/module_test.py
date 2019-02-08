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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl.testing import parameterized

from tensorflow.python.compat import v2_compat
from tensorflow.python.framework import ops
from tensorflow.python.module import module
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class TestModuleNaming(test.TestCase):

  def test_single_name(self):
    mod = module.Module(name="simple")
    self.assertEqual(mod.name, "simple")
    self.assertEqual(mod.name_scope.name, "simple/")

  def test_construct_in_scope(self):
    with ops.name_scope("foo"):
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
    with ops.name_scope("foo"):
      mod = module.Module(name="bar")
    mod.foo = get_name_scope
    # `foo` is not a method so we do not re-enter the name scope.
    self.assertEqual(mod.foo(), "")

  def test_invalid_name(self):
    msg = ".* is not a valid module name"
    with self.assertRaisesRegexp(ValueError, msg):
      module.Module(name="$Foo")

  def test_modules_not_numbered_in_eager(self):
    mod = RecursiveModule(2)
    self.assertEqual(mod.name_scope.name, "badger/")
    self.assertEqual(mod.child.name_scope.name, "badger/badger/")

    mod = RecursiveModule(2)
    self.assertEqual(mod.name_scope.name, "badger/")
    self.assertEqual(mod.child.name_scope.name, "badger/badger/")

  def test_module_numbering_in_graph(self):
    with ops.Graph().as_default():
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
      # to __exit__ a non-existant name scope).
      ErrorModule(call_super=False)

    self.assertEqual("", get_name_scope())

  def test_forward_method_closes_name_scope(self):
    mod = ErrorModule(call_super=True, raise_in_constructor=False)
    with self.assertRaises(ErrorModuleError):
      mod()

    self.assertEqual("", get_name_scope())


class VariableNamingTest(test.TestCase):

  def test_variable_names(self):
    mod = RecursiveModule(3)
    self.assertEqual(mod.w.name, "badger/mushroom:0")
    self.assertEqual(mod.child.w.name, "badger/badger/mushroom:0")
    self.assertEqual(mod.child.child.w.name, "badger/badger/badger/mushroom:0")


class VariableTrackingTest(test.TestCase):

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


class ModuleTrackingTest(test.TestCase):

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


class CommonErrorsTest(test.TestCase):

  def test_not_calling_super_constructor(self):
    msg = ("Constructing a tf.Module without calling the super constructor is "
           "not supported")
    with self.assertRaisesRegexp(ValueError, msg):
      DoesNotCallSuperConstructorModule()

  def test_calls_method_before_super(self):
    msg = "super constructor must be called before any other methods"
    with self.assertRaisesRegexp(AttributeError, msg):
      CallsMethodBeforeSuperConstructorModule(allowed_method=False)

  def test_annotated_method_is_allowed(self):
    self.assertIsNotNone(
        CallsMethodBeforeSuperConstructorModule(allowed_method=True))


def get_name_scope():
  with ops.name_scope("x") as ns:
    return ns[:-2]


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
    self.child = None
    if depth > 1:
      self.child = RecursiveModule(depth - 1, trainable=trainable)
    self.w = variables.Variable(1.0, trainable=trainable, name="mushroom")


class TreeModule(module.Module):

  def __init__(self, name=None):
    super(TreeModule, self).__init__(name=name)
    self._leaves = []

  def new_leaf(self, name=None):
    leaf = TreeModule(name=name)
    self._leaves.append(leaf)
    return leaf


class ReturnsNameScopeModule(module.Module):

  def alternative_forward(self):
    return get_name_scope()

  def __call__(self):
    return get_name_scope()


class SubclassedReturnsNameScopeModule(ReturnsNameScopeModule):

  def alternative_alternative_forward(self):
    return get_name_scope()


class PropertyThrowsWhenCalledModule(module.Module):

  @property
  def raise_assertion_error(self):
    raise AssertionError


class ModuleOverridingNameScope(ReturnsNameScopeModule):

  @property
  def name_scope(self):
    return ops.name_scope("yolo/")


class DoesNotCallSuperConstructorModule(module.Module):

  def __init__(self):
    # NOTE: Intentionally does not call super constructor.
    pass


class CallsMethodBeforeSuperConstructorModule(module.Module):

  def __init__(self, allowed_method):
    if allowed_method:
      self.no_name_scope()
    else:
      self.with_name_scope()
    super(CallsMethodBeforeSuperConstructorModule, self).__init__()

  @module.Module.no_name_scope
  def no_name_scope(self):
    pass

  def with_name_scope(self):
    pass

NamedPair = collections.namedtuple("NamedPair", ("first", "second"))
mk_index_dict = lambda v: dict(enumerate(v))


class FlattenTest(parameterized.TestCase, test.TestCase):

  @parameterized.parameters(lambda v: NamedPair(*v), list, tuple, mk_index_dict)
  def test_flatten(self, container_type):
    parent = SimpleModule(container_type=container_type)
    child = parent.c

    self.assertEqual(
        list(parent._flatten(recursive=False, predicate=IS_MEMBER)),
        [parent.a[0], parent.a[1], parent.z])

    self.assertEqual(
        list(parent._flatten(predicate=IS_MEMBER)),
        [parent.a[0], parent.a[1], parent.z, child.a[0], child.a[1], child.z])

  def test_attribute_traversal_key(self):
    mod = LayerModule()
    self.assertEqual(
        mod.variables,
        mod._trainable_variables + mod._non_trainable_variables + [mod._bonus])

  def test_with_path(self):
    mod = module.Module()
    mod.w = variables.Variable(1.)
    mod.encoder = module.Module()
    mod.encoder.w = [({"k": mod.w}, {"k": mod.w})]
    mod.decoder = mod.encoder

    state_dict = dict(
        mod._flatten(with_path=True, predicate=module._IS_VARIABLE))

    self.assertEqual(state_dict,
                     {("w",): mod.w,
                      ("encoder", "w", 0, 0, "k"): mod.encoder.w[0][0]["k"],
                      ("encoder", "w", 0, 1, "k"): mod.encoder.w[0][1]["k"],
                      ("decoder", "w", 0, 0, "k"): mod.decoder.w[0][0]["k"],
                      ("decoder", "w", 0, 1, "k"): mod.decoder.w[0][1]["k"]},)


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

    return list(self._flatten(predicate=module._IS_VARIABLE,
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


IS_MEMBER = lambda v: isinstance(v, MemberType)
IS_MODULE = lambda v: isinstance(v, module.Module)

if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  test.main()
