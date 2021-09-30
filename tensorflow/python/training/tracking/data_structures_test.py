# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
import collections
import copy
import json
import os
import pickle

from absl.testing import parameterized

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import core as non_keras_core
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.training.tracking import tracking
from tensorflow.python.training.tracking import util
from tensorflow.python.util import nest
from tensorflow.python.util import serialization


class ListTests(test.TestCase):

  def testJSONSerialization(self):
    obj = tracking.AutoTrackable()
    obj.l = [1]
    json.dumps(obj.l, default=serialization.get_json_type)

  def testNotTrackable(self):
    class NotTrackable(object):
      pass

    with self.assertRaises(ValueError):
      data_structures.List([NotTrackable()])

  def testCallNotImplemented(self):
    with self.assertRaisesRegex(TypeError, "not callable"):
      data_structures.List()(1.)  # pylint: disable=not-callable

  def testNoPop(self):
    with self.assertRaises(AttributeError):
      data_structures.List().pop()

  def testNesting(self):
    with context.graph_mode():
      inner = data_structures.List()
      outer = data_structures.List([inner])
      inner.append(non_keras_core.Dense(1))
      inner[0](array_ops.ones([2, 3]))
      self.assertEqual(2, len(outer.variables))
      self.assertIsInstance(
          outer.variables[0],
          resource_variable_ops.ResourceVariable)

  def testNonLayerVariables(self):
    v = resource_variable_ops.ResourceVariable([1.])
    l = data_structures.List([v])
    self.assertTrue(l.trainable)
    self.assertEqual([], l.layers)
    self.assertEqual([v], l.variables)
    self.assertEqual([v], l.trainable_weights)
    self.assertEqual([], l.non_trainable_variables)
    l.trainable = False
    self.assertEqual([v], l.variables)
    self.assertEqual([], l.trainable_variables)
    self.assertEqual([v], l.non_trainable_variables)
    l.trainable = True
    v2 = resource_variable_ops.ResourceVariable(1., trainable=False)
    l.append(v2)
    self.assertEqual([v, v2], l.weights)
    self.assertEqual([v], l.trainable_weights)
    self.assertEqual([v2], l.non_trainable_weights)

  def testCopy(self):
    v1 = resource_variable_ops.ResourceVariable(1.)
    v2 = resource_variable_ops.ResourceVariable(1.)
    v3 = resource_variable_ops.ResourceVariable(1.)

    l1 = data_structures.List([v1, v2])
    l2 = l1.copy()
    l2.append(v3)
    self.assertEqual(list(l1), [v1, v2])
    self.assertEqual(list(l2), [v1, v2, v3])

  def testSlicing(self):
    v1 = resource_variable_ops.ResourceVariable(1.)
    v2 = resource_variable_ops.ResourceVariable(1.)
    v3 = resource_variable_ops.ResourceVariable(1.)
    v4 = resource_variable_ops.ResourceVariable(1.)

    l = data_structures.List([v1, v2, v3, v4])
    self.assertEqual(l[1:], [v2, v3, v4])
    self.assertEqual(l[1:-1], [v2, v3])
    self.assertEqual(l[:-1], [v1, v2, v3])

  def testHash(self):
    has_sequences = {data_structures.List(), data_structures.List()}
    self.assertEqual(2, len(has_sequences))
    self.assertNotIn(data_structures.List(), has_sequences)

  def testIMul_zero(self):
    l = data_structures.List([])
    with self.assertRaisesRegex(ValueError, "List only supports append"):
      l *= 0

  def testIMul(self):
    v = resource_variable_ops.ResourceVariable(1.)
    l = data_structures.List([v])
    l *= 2
    self.assertEqual(list(l), [v] * 2)

  def testMul(self):
    v = resource_variable_ops.ResourceVariable(1.)
    l = data_structures.List([v, v, v])
    self.assertEqual(list(l * 2), [v, v, v] * 2)

  def testRMul(self):
    v = resource_variable_ops.ResourceVariable(1.)
    l = data_structures.List([v, v, v])
    self.assertEqual(list(2 * l), [v, v, v] * 2)


class ListWrapperTest(test.TestCase):

  IGNORED = ("__new__", "__init__", "__subclasshook__", "__getattribute__")

  def test_overrides_all_list_methods(self):
    not_overridden = []

    for name in dir(list):
      if name in ListWrapperTest.IGNORED:
        continue

      list_method = getattr(list, name)

      if not callable(list_method):
        continue

      object_method = getattr(object, name, None)
      if object_method is not None and object_method == list_method:
        # Skip methods that aren't overridden from object.
        continue

      if list_method == getattr(data_structures.ListWrapper, name):
        not_overridden.append(name)

    if not_overridden:
      self.fail("ListWrapper does not override %s" % (not_overridden))

  def testPickle(self):
    original = data_structures.ListWrapper([1, 2])
    serialized = pickle.dumps(original)
    del original
    deserialized = pickle.loads(serialized)
    self.assertEqual([1, 2], deserialized)

  def testSameStructure(self):
    l = [1]
    nest.assert_same_structure(l, data_structures.ListWrapper(copy.copy(l)))

  def testMutateWithoutTrackableComponents(self):
    m = module.Module()
    m.l = [1, 2]
    m.l.insert(0, 0)
    self.assertEqual(m.l, [0, 1, 2])
    self.assertEqual(m.l._checkpoint_dependencies, [])

  def testFunctionCaching(self):
    @def_function.function
    def f(list_input):
      return list_input[0] + constant_op.constant(1.)

    first_trace = f.get_concrete_function([constant_op.constant(2.)])
    second_trace = f.get_concrete_function(
        data_structures.ListWrapper([constant_op.constant(3.)]))
    self.assertIs(first_trace, second_trace)

  def testListWrapperBasic(self):
    # ListWrapper, unlike List, compares like the built-in list type (since it
    # is used to automatically replace lists).
    a = tracking.AutoTrackable()
    b = tracking.AutoTrackable()
    self.assertEqual([a, a],
                     [a, a])
    self.assertEqual(data_structures.ListWrapper([a, a]),
                     data_structures.ListWrapper([a, a]))
    self.assertEqual([a, a],
                     data_structures.ListWrapper([a, a]))
    self.assertEqual(data_structures.ListWrapper([a, a]),
                     [a, a])
    self.assertNotEqual([a, a],
                        [b, a])
    self.assertNotEqual(data_structures.ListWrapper([a, a]),
                        data_structures.ListWrapper([b, a]))
    self.assertNotEqual([a, a],
                        data_structures.ListWrapper([b, a]))
    self.assertLess([a], [a, b])
    self.assertLess(data_structures.ListWrapper([a]),
                    data_structures.ListWrapper([a, b]))
    self.assertLessEqual([a], [a, b])
    self.assertLessEqual(data_structures.ListWrapper([a]),
                         data_structures.ListWrapper([a, b]))
    self.assertGreater([a, b], [a])
    self.assertGreater(data_structures.ListWrapper([a, b]),
                       data_structures.ListWrapper([a]))
    self.assertGreaterEqual([a, b], [a])
    self.assertGreaterEqual(data_structures.ListWrapper([a, b]),
                            data_structures.ListWrapper([a]))
    self.assertEqual([a], data_structures.ListWrapper([a]))
    self.assertEqual([a], list(data_structures.List([a])))
    self.assertEqual([a, a], data_structures.ListWrapper([a]) + [a])
    self.assertEqual([a, a], [a] + data_structures.ListWrapper([a]))
    self.assertIsInstance(data_structures.ListWrapper([a]), list)
    self.assertEqual(
        tensor_shape.TensorShape([None, 2]).as_list(),
        (data_structures.ListWrapper([None])
         + tensor_shape.TensorShape([2])).as_list())

  def testAcceptsNonTrackableContent(self):
    l = data_structures.ListWrapper([1, 2, 3])
    self.assertEqual(l, [1, 2, 3])

  def testWrapperChangesList(self):
    l = []
    l_wrapper = data_structures.ListWrapper(l)
    l_wrapper.append(1)
    self.assertEqual([1], l)

  def testListChangesWrapper(self):
    l = []
    l_wrapper = data_structures.ListWrapper(l)
    l.append(1)
    self.assertEqual([1], l_wrapper)

  def testNotHashable(self):
    with self.assertRaises(TypeError):
      hash(data_structures.ListWrapper())  # pylint: disable=no-value-for-parameter

  def testDelItem(self):
    l = data_structures.ListWrapper([1, 2, 3, [4]])
    del l[0]
    self.assertEqual(l, [2, 3, [4]])
    self.assertUnableToSave(l, "Unable to save .*__delitem__")

  def testDelSlice(self):
    l = data_structures.ListWrapper([1, 2, 3, [4]])
    del l[2:3]
    self.assertEqual(l, [1, 2, [4]])
    self.assertUnableToSave(l, "Unable to save .*__delslice__")

  def testSetSlice_canSaveForNonTrackableItems(self):
    l = data_structures.ListWrapper([1, 2, 3, 4])
    l[:] = 2, 8, 9, 0
    self.assertEqual(l, [2, 8, 9, 0])
    l._maybe_initialize_trackable()  # pylint: disable=protected-access
    self.assertEqual(len(l._checkpoint_dependencies), 0)  # pylint: disable=protected-access

  def testSetSlice_cannotSaveIfTrackableModified(self):
    v1 = resource_variable_ops.ResourceVariable(1.)
    v2 = resource_variable_ops.ResourceVariable(1.)
    l = data_structures.ListWrapper([1, 2, v1, v2])
    l[:] = 2, 8, 9, v2
    self.assertEqual(l, [2, 8, 9, v2])
    self.assertUnableToSave(l, "Unable to save .*__setslice__")

  def testSetSlice_truncate(self):
    l = data_structures.ListWrapper([1, 2, 3, 4])
    l[:] = []
    self.assertEqual(l, [])

  def testSetSlice_extend(self):
    l = data_structures.ListWrapper([1, 2, 3, 4])
    l[2:] = 1, 2, 3, 4
    self.assertEqual(l, [1, 2, 1, 2, 3, 4])

  def testIMulNegative(self):
    l = data_structures.ListWrapper([1, 2, 3, [4]])
    l *= -1
    self.assertEqual(l, [1, 2, 3, [4]] * -1)
    self.assertUnableToSave(l, "Unable to save")

  def testIMulPositive(self):
    v = variables.Variable(1.)
    l = data_structures.ListWrapper([1, 2, 3, 4, v])
    self.assertEqual([("4", v)], l._checkpoint_dependencies)
    root = util.Checkpoint(l=l)
    prefix = os.path.join(self.get_temp_dir(), "ckpt")
    path = root.save(prefix)
    v.assign(5.)
    l *= 2
    self.assertEqual(l, [1, 2, 3, 4, v, 1, 2, 3, 4, v])
    self.assertEqual([("4", v), ("9", v)], l._checkpoint_dependencies)
    root.restore(path)
    self.assertAllClose(1., v.numpy())

  def testSort(self):
    l = data_structures.ListWrapper([[1], [2], [3], [4]])
    l.sort()
    self.assertAllEqual(l, [[1], [2], [3], [4]])
    # Regardless of being a no-op for the input list, we still refuse to save.
    # This is intentional since otherwise we would end up with a hard to debug
    # case for users (e.g. sometimes sort on a ListWrapper is trackable and
    # other times it is not).
    self.assertUnableToSave(l, "Unable to save .*sort")

  def assertUnableToSave(self, l, msg):
    l._maybe_initialize_trackable()  # pylint: disable=protected-access
    with self.assertRaisesRegex(ValueError, msg):
      return l._checkpoint_dependencies  # pylint: disable=protected-access


class MappingTests(test.TestCase):

  def testJSONSerialization(self):
    obj = tracking.AutoTrackable()
    obj.d = {"a": 2}
    json.dumps(obj.d, default=serialization.get_json_type)

  def testNoOverwrite(self):
    mapping = data_structures.Mapping()
    original = data_structures.List()
    mapping["a"] = original
    with self.assertRaises(ValueError):
      mapping["a"] = data_structures.List()
    self.assertIs(original, mapping["a"])
    with self.assertRaises(AttributeError):
      del mapping["a"]  # pylint: disable=unsupported-delete-operation
    mapping.update(b=data_structures.Mapping())
    with self.assertRaises(ValueError):
      mapping.update({"b": data_structures.Mapping()})

  def testNonStringKeys(self):
    mapping = data_structures.Mapping()
    with self.assertRaises(TypeError):
      mapping[1] = data_structures.List()

  def testHashing(self):
    has_mappings = set([data_structures.Mapping(),
                        data_structures.Mapping()])
    self.assertEqual(2, len(has_mappings))
    self.assertNotIn(data_structures.Mapping(), has_mappings)
    # In contrast to Mapping, dict wrappers are not hashable
    a = tracking.AutoTrackable()
    a.d = {}
    self.assertEqual({}, a.d)
    self.assertFalse({} != a.d)  # pylint: disable=g-explicit-bool-comparison
    self.assertNotEqual({1: 2}, a.d)
    with self.assertRaisesRegex(TypeError, "unhashable"):
      set([a.d])

  def testListShallowCopy(self):
    root = tracking.AutoTrackable()
    orig_list = [[1.]]
    root.a = orig_list
    copied = copy.copy(root.a)
    self.assertAllEqual([[1.]], copied)
    self.assertIsNot(root.a, copied)
    self.assertIs(root.a[0], copied[0])

    # Dirtiness should be inherited
    util.list_objects(root.a)
    orig_list.append(1.)
    with self.assertRaises(ValueError):
      util.list_objects(root.a)
    with self.assertRaises(ValueError):
      util.list_objects(copy.copy(root.a))

  def testListDeepCopy(self):
    root = tracking.AutoTrackable()
    orig_list = [[1.]]
    root.a = orig_list
    copied = copy.deepcopy(root.a)
    self.assertAllEqual([[1.]], copied)
    self.assertIsNot(root.a, copied)
    self.assertIsNot(root.a[0], copied[0])

    # Dirtiness should be inherited
    util.list_objects(root.a)
    orig_list.append(1.)
    with self.assertRaises(ValueError):
      util.list_objects(root.a)
    with self.assertRaises(ValueError):
      util.list_objects(copy.deepcopy(root.a))

  def testDictShallowCopy(self):
    root = tracking.AutoTrackable()
    orig_dict = {"a": [1.]}
    root.a = orig_dict
    copied = copy.copy(root.a)
    self.assertAllEqual([1.], copied["a"])
    self.assertIsNot(root.a, copied)
    self.assertIs(root.a["a"], copied["a"])

    copied = root.a.copy()
    self.assertAllEqual([1.], copied["a"])
    self.assertIsNot(root.a, copied)
    self.assertIs(root.a["a"], copied["a"])

    # Dirtiness should be inherited
    util.list_objects(root.a)
    orig_dict["b"] = []
    with self.assertRaises(ValueError):
      util.list_objects(root.a)
    with self.assertRaises(ValueError):
      util.list_objects(copy.copy(root.a))

  def testDictDeepCopy(self):
    root = tracking.AutoTrackable()
    orig_dict = {"a": [1.]}
    root.a = orig_dict
    copied = copy.deepcopy(root.a)
    self.assertAllEqual([1.], copied["a"])
    self.assertIsNot(root.a, copied)
    self.assertIsNot(root.a["a"], copied["a"])

    # Dirtiness should be inherited
    util.list_objects(root.a)
    orig_dict["b"] = []
    with self.assertRaises(ValueError):
      util.list_objects(root.a)
    with self.assertRaises(ValueError):
      util.list_objects(copy.deepcopy(root.a))

  def testShallowCopyTrackable(self):
    original = tracking.AutoTrackable()
    original_sub = tracking.AutoTrackable()
    original.a = [[1.]]
    original.b = {"a": original_sub}
    shallow_copied = copy.copy(original)
    self.assertIs(original_sub, shallow_copied.b["a"])
    self.assertIsNot(original, shallow_copied)
    self.assertEqual([[1.]], shallow_copied.a)
    shallow_deps = util.list_objects(shallow_copied)
    self.assertIn(shallow_copied.a, shallow_deps)
    self.assertIn(shallow_copied.b, shallow_deps)
    self.assertIn(shallow_copied.b["a"], shallow_deps)

  def testDeepCopyTrackable(self):
    original = tracking.AutoTrackable()
    original_sub = tracking.AutoTrackable()
    original.a = [[1.]]
    original.b = {"a": original_sub}
    self.assertIsInstance(original.b, dict)
    deep_copied = copy.deepcopy(original)
    self.assertIsInstance(deep_copied.b, dict)
    self.assertIsNot(original, deep_copied)
    self.assertIsNot(original_sub, deep_copied.b["a"])
    self.assertEqual([[1.]], deep_copied.a)
    self.assertIsInstance(deep_copied.b["a"], tracking.AutoTrackable)
    deps = util.list_objects(deep_copied)
    self.assertIn(deep_copied.a, deps)
    self.assertIn(deep_copied.b, deps)
    self.assertIn(deep_copied.b["a"], deps)
    self.assertNotIn(original_sub, deps)

  def testConstructableFromSequence(self):
    result = data_structures._DictWrapper([(1, 2), (3, 4)])
    self.assertIsInstance(result, dict)
    self.assertEqual({1: 2, 3: 4}, result)

  def testPickle(self):
    original = data_structures._DictWrapper(dict(a=1, b=2))
    serialized = pickle.dumps(original)
    del original
    deserialized = pickle.loads(serialized)
    self.assertEqual(dict(a=1, b=2), deserialized)

  def testListAddOrder(self):
    self.assertEqual([1., 2.],
                     data_structures.ListWrapper([1.])
                     + data_structures.ListWrapper([2.]))
    self.assertEqual([1., 2.],
                     data_structures.ListWrapper([1.])
                     + [2.])
    self.assertEqual([1., 2.],
                     [1.]
                     + data_structures.ListWrapper([2.]))

  def testSameStructure(self):
    d = {1: "a"}
    nest.assert_same_structure(d, data_structures._DictWrapper(d.copy()))

  def testFunctionCaching(self):
    @def_function.function
    def f(dict_input):
      return dict_input["x"] + constant_op.constant(1.)

    first_trace = f.get_concrete_function({"x": constant_op.constant(2.)})
    second_trace = f.get_concrete_function(
        data_structures._DictWrapper({"x": constant_op.constant(3.)}))
    self.assertIs(first_trace, second_trace)


class TupleTests(test.TestCase, parameterized.TestCase):

  def testJSONSerialization(self):
    obj = tracking.AutoTrackable()
    obj.l = (1,)
    json.dumps(obj.l, default=serialization.get_json_type)

  def testNonLayerVariables(self):
    v = resource_variable_ops.ResourceVariable([1.])
    l = data_structures._TupleWrapper((v,))
    self.assertEqual([], l.layers)
    self.assertEqual([v], l.variables)
    self.assertEqual([v], l.trainable_weights)
    self.assertEqual([], l.non_trainable_variables)

  def testCopy(self):
    v1 = resource_variable_ops.ResourceVariable(1.)
    v2 = resource_variable_ops.ResourceVariable(1.)

    l1 = data_structures._TupleWrapper((v1, v2))
    l2 = copy.copy(l1)
    self.assertEqual(l1, (v1, v2))
    self.assertEqual(l2, (v1, v2))
    self.assertIs(l1[0], l2[0])
    l2_deep = copy.deepcopy(l1)
    self.assertIsNot(l1[0], l2_deep[0])
    with self.assertRaises(AttributeError):
      l2.append(v1)

  def testSlicing(self):
    v1 = resource_variable_ops.ResourceVariable(1.)
    v2 = resource_variable_ops.ResourceVariable(1.)
    v3 = resource_variable_ops.ResourceVariable(1.)
    v4 = resource_variable_ops.ResourceVariable(1.)

    l = data_structures._TupleWrapper((v1, v2, v3, v4))
    self.assertEqual(l[1:], (v2, v3, v4))
    self.assertEqual(l[1:-1], (v2, v3))
    self.assertEqual(l[:-1], (v1, v2, v3))

  def testHash(self):
    has_sequences = set([data_structures._TupleWrapper(),
                         data_structures._TupleWrapper()])
    self.assertLen(has_sequences, 1)
    self.assertIn(data_structures._TupleWrapper(), has_sequences)

  def testIMul_zero(self):
    l = data_structures._TupleWrapper((1,))
    l *= 0
    self.assertEqual((), l)

  def testIMul(self):
    # Note: tuple behavior differs from list behavior. Lists are mutated by
    # imul/iadd, tuples assign a new object to the left hand side of the
    # expression.
    v = resource_variable_ops.ResourceVariable(1.)
    l = data_structures._TupleWrapper((v,))
    original = l
    l *= 2
    self.assertEqual(l, (v,) * 2)
    self.assertNotEqual(original, (v,) * 2)

  def testIAdd(self):
    v = resource_variable_ops.ResourceVariable(1.)
    l = data_structures._TupleWrapper((v,))
    original = l
    l += (1,)
    self.assertEqual(l, (v, 1))
    self.assertNotEqual(original, (v, 1))
    self.assertEqual(original, (v,))

  def testMul(self):
    v = resource_variable_ops.ResourceVariable(1.)
    l = data_structures._TupleWrapper((v, v, v))
    self.assertEqual(l * 2, (v, v, v) * 2)

  def testRMul(self):
    v = resource_variable_ops.ResourceVariable(1.)
    l = data_structures._TupleWrapper((v, v, v))
    self.assertEqual(2 * l, (v, v, v) * 2)

  def testPickle(self):
    original = data_structures._TupleWrapper((1, 2))
    serialized = pickle.dumps(original)
    del original
    deserialized = pickle.loads(serialized)
    self.assertEqual((1, 2), deserialized)

  def testNamedTuple(self):
    named = collections.namedtuple("Named", ("x", "y"))
    v = variables.Variable(2)
    nt = named(x=v, y=2)
    m = module.Module()
    m.nt = nt
    self.assertIs(v, m.nt.x)
    self.assertIs(v, m.nt[0])
    self.assertIs(
        v, m._checkpoint_dependencies[0].ref._checkpoint_dependencies[0].ref)
    self.assertEqual(2, m.nt.y)

  def testNamedTupleConflictingAttributes(self):
    named = collections.namedtuple("Named", ("x", "weights"))
    v = variables.Variable(2)
    nt = named(x=v, weights=3)
    m = module.Module()
    m.nt = nt
    self.assertEqual(3, m.nt.weights)

  def testNamedSubclassing(self):
    named = collections.namedtuple("Named", ("x", "y"))
    v = variables.Variable(2)

    class NamedSubclass(named):

      def __new__(cls, x, y):
        del y  # unused
        return super(NamedSubclass, cls).__new__(cls, x, 3)

      @property
      def summed(self):
        return self.x + self.y

    nt = NamedSubclass(x=v, y=2)
    m = module.Module()
    m.nt = nt
    self.assertEqual(3, m.nt.y)
    self.assertIs(v, m.nt.x)
    self.assertIs(
        v, m._checkpoint_dependencies[0].ref._checkpoint_dependencies[0].ref)
    self.assertEqual("x", m.nt._checkpoint_dependencies[0].name)
    self.assertEqual("0", m.nt._checkpoint_dependencies[1].name)
    self.assertEqual(5, self.evaluate(m.nt.summed))

  def testUnnamedSubclassing(self):
    v = variables.Variable(2)

    class UnnamedSubclass(tuple):

      @property
      def summed(self):
        return self[0] + self[1]

    unt = UnnamedSubclass([v, 2])
    m = module.Module()
    m.unt = unt
    self.assertEqual("0", m.unt._checkpoint_dependencies[0].name)
    self.assertLen(m.unt._checkpoint_dependencies, 1)
    self.assertEqual(4, self.evaluate(m.unt.summed))
    nest.assert_same_structure(
        [m.unt], nest.map_structure(lambda x: x, [m.unt]))

  def testNamedtupleSubclassWithCustomNew(self):
    class SubclassWithDifferentArgs(collections.namedtuple("A", ["x"])):

      def __new__(cls):
        return super(SubclassWithDifferentArgs, cls).__new__(cls, [])

    nt = SubclassWithDifferentArgs()
    m = module.Module()
    m.nt = nt
    m.nt.x.append(variables.Variable(1.))
    prefix = os.path.join(self.get_temp_dir(), "ckpt")
    ckpt = util.Checkpoint(m=m)
    with self.assertRaises(ValueError):
      ckpt.save(prefix)

  def testSameStructure(self):
    t = (variables.Variable(1.),)
    m = module.Module()
    m.t = t
    nest.assert_same_structure(t, m.t)
    nest.assert_same_structure(m.t, t)

    nt_type = collections.namedtuple("nt", ["x", "y"])
    nt = nt_type(x=1, y=2)
    m.nt = nt
    nest.assert_same_structure(m.nt, nt)
    with self.assertRaises(TypeError):  # pylint: disable=g-error-prone-assert-raises
      nest.assert_same_structure(m.nt, m.t)

  def testFlatten(self):
    t = data_structures._TupleWrapper((1, data_structures._TupleWrapper((2,))))
    self.assertEqual([1, 2], nest.flatten(t))
    self.assertEqual(
        nest.flatten_with_tuple_paths((1, (2,))),
        nest.flatten_with_tuple_paths(t))
    self.assertEqual((3, (4,)),
                     nest.pack_sequence_as(t, [3, 4]))
    nt_type = collections.namedtuple("nt", ["x", "y"])
    nt = nt_type(1., 2.)
    wrapped_nt = data_structures._TupleWrapper(nt)
    self.assertEqual(
        nest.flatten_with_tuple_paths(nt),
        nest.flatten_with_tuple_paths(wrapped_nt))
    self.assertEqual((3, 4,),
                     nest.pack_sequence_as(wrapped_nt, [3, 4]))
    self.assertEqual(3, nest.pack_sequence_as(wrapped_nt, [3, 4]).x)

  def testFunctionCaching(self):
    @def_function.function
    def f(tuple_input):
      return tuple_input[0] + constant_op.constant(1.)

    first_trace = f.get_concrete_function((constant_op.constant(2.),))
    second_trace = f.get_concrete_function(
        data_structures._TupleWrapper((constant_op.constant(3.),)))
    self.assertIs(first_trace, second_trace)

  def testPythonMapImpl(self):
    t = data_structures._TupleWrapper((1, data_structures._TupleWrapper((2,))))
    self.assertEqual(
        (4, (5,)),
        nest.map_structure_up_to((None, (None,)), lambda x: x + 3, t,
                                 check_types=True))
    nest.assert_shallow_structure((None, None), t)

  def testDatasetMap(self):
    dataset = dataset_ops.Dataset.from_tensor_slices(
        constant_op.constant([1, 2, 3]))
    dataset = dataset.map(lambda x: data_structures._TupleWrapper((x,)))
    for index, element in enumerate(dataset):
      self.assertEqual((index + 1,), self.evaluate(element))

  def testDatasetMapNamed(self):
    nt_type = collections.namedtuple("A", ["x"])
    dataset = dataset_ops.Dataset.from_tensor_slices(
        constant_op.constant([1, 2, 3]))
    dataset = dataset.map(lambda x: data_structures._TupleWrapper(nt_type(x)))
    for index, element in enumerate(dataset):
      self.assertEqual((index + 1,), self.evaluate(element))

  def testLoopAssignedModule(self):
    m = module.Module()
    m.s = (m,)
    self.assertLen(m._checkpoint_dependencies, 1)
    self.assertIs(m.s, m._checkpoint_dependencies[0].ref)
    self.assertIs("s", m._checkpoint_dependencies[0].name)
    self.assertEqual((), m.trainable_variables)


if __name__ == "__main__":
  test.main()
