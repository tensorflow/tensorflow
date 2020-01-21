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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import os
import pickle

from absl.testing import parameterized
import numpy
import six

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.layers import normalization
from tensorflow.python.layers import core as non_keras_core
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.training.tracking import tracking
from tensorflow.python.training.tracking import util
from tensorflow.python.util import nest
from tensorflow.python.util import serialization


class HasList(training.Model):

  def __init__(self):
    super(HasList, self).__init__()
    self.layer_list = data_structures.List([core.Dense(3)])
    self.layer_list.append(core.Dense(4))
    self.layer_list.extend(
        [core.Dense(5),
         core.Dense(6, kernel_regularizer=math_ops.reduce_sum)])
    self.layer_list += [
        core.Dense(7, bias_regularizer=math_ops.reduce_sum),
        core.Dense(8)
    ]
    self.layer_list += (
        data_structures.List([core.Dense(9)]) + data_structures.List(
            [core.Dense(10)]))
    self.layer_list.extend(
        data_structures.List(
            list([core.Dense(11)]) + [core.Dense(12)]))
    self.layers_with_updates = data_structures.List(
        (normalization.BatchNormalization(),))

  def call(self, x):
    aggregation = 0.
    for l in self.layer_list:
      x = l(x)
      aggregation += math_ops.reduce_sum(x)
    bn, = self.layers_with_updates
    return bn(x) / aggregation


class ListTests(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_v1_only("b/120545219")
  def testTracking(self):
    model = HasList()
    output = model(array_ops.ones([32, 2]))
    self.assertAllEqual([32, 12], output.shape)
    self.assertEqual(11, len(model.layers))
    self.assertEqual(10, len(model.layer_list.layers))
    six.assertCountEqual(
        self,
        model.layers,
        model.layer_list.layers + model.layers_with_updates)
    for index in range(10):
      self.assertEqual(3 + index, model.layer_list.layers[index].units)
    self.assertEqual(2, len(model._checkpoint_dependencies))
    self.assertIs(model.layer_list, model._checkpoint_dependencies[0].ref)
    self.assertIs(model.layers_with_updates,
                  model._checkpoint_dependencies[1].ref)
    self.assertEqual(
        10, len(model._checkpoint_dependencies[0].ref._checkpoint_dependencies))
    self.evaluate([v.initializer for v in model.variables])
    self.evaluate(model.variables[0].assign([[1., 2., 3.], [4., 5., 6.]]))
    save_path = os.path.join(self.get_temp_dir(), "ckpt")
    model.save_weights(save_path)
    self.evaluate(model.variables[0].assign(array_ops.zeros([2, 3])))
    model.load_weights(save_path)
    self.assertAllEqual([[1., 2., 3.], [4., 5., 6.]],
                        self.evaluate(model.variables[0]))
    v = variables.Variable(1.)
    model.var_list = [v]
    self.assertIn(v, model.variables)
    self.assertIn(v, model.trainable_variables)
    self.assertNotIn(v, model.non_trainable_variables)
    self.assertIn(model.layer_list[0].trainable_weights[0],
                  model.trainable_weights)

  def testSubModelTracking(self):
    model = training.Model()
    model.v = variables.Variable(1.)
    self.assertIn(model.v, model.trainable_weights)
    model2 = training.Model()
    model2.m = [model]
    self.assertIn(model.v, model2.trainable_weights)

  def testSubSequentialTracking(self):

    class _Subclassed(training.Model):

      def __init__(self, wrapped):
        super(_Subclassed, self).__init__()
        self._wrapped = wrapped

      def call(self, x):
        return self._wrapped(x)

    model = sequential.Sequential()
    layer = core.Dense(1)
    model.add(layer)
    model2 = _Subclassed(model)
    model2(array_ops.ones([1, 2]))
    model2.m = [model]
    self.assertIn(layer.kernel, model2.trainable_weights)

  def testLayerTrackedThroughSequential(self):
    class AttrDict(dict):

      def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def ffnet(layer_sizes, name):
      ff = sequential.Sequential(name=name)
      for i, width in enumerate(layer_sizes):
        ff.add(core.Dense(
            width,
            activation=("relu" if i < len(layer_sizes)-1 else None)))
      return ff

    class MyModel2(training.Model):

      def __init__(self, config, name="my_model_2"):
        super(MyModel2, self).__init__(name=name)
        self._num_tokens = config.num_tokens

        # list of sub-models
        self._ffnet = [ffnet(config.module_layers + (self._num_tokens,), "ff")]

      def null_input(self):
        return array_ops.zeros([1, self._num_tokens], dtype=dtypes.float32)

      def call(self, input_, module_index=None):
        return self._ffnet[0](input_)

    m2 = MyModel2(AttrDict(
        num_tokens=5,
        module_layers=(50, 30)))

    # Construct
    m2(m2.null_input())
    self.assertLen(m2.trainable_variables, 6)

  def testJSONSerialization(self):
    obj = tracking.AutoTrackable()
    obj.l = [1]
    json.dumps(obj.l, default=serialization.get_json_type)

  @test_util.run_v1_only("b/120545219")
  def testUpdatesForwarded(self):
    with context.graph_mode():
      model = HasList()
      model_input = array_ops.ones([32, 2])
      model(model_input)
      self.assertGreater(len(model.layers_with_updates[0].updates), 0)
      self.assertEqual(set(model.layers_with_updates[0].updates),
                       set(model.updates))

    with context.eager_mode():
      model = HasList()
      model_input = array_ops.ones([32, 2])
      model(model_input)
      self.assertEqual(0, len(model.updates))

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_v1_only("b/120545219")
  def testLossesForwarded(self):
    model = HasList()
    model_input = array_ops.ones([32, 2])
    model(model_input)
    self.assertEqual(2, len(model.losses))

  def testModelContainersCompareEqual(self):
    class HasEqualContainers(training.Model):

      def __init__(self):
        super(HasEqualContainers, self).__init__()
        self.l1 = []
        self.l2 = []

    model = HasEqualContainers()
    first_layer = HasEqualContainers()
    model.l1.append(first_layer)
    second_layer = HasEqualContainers()
    model.l2.append(second_layer)
    self.assertEqual([first_layer, second_layer], model.layers)

  def testNotTrackable(self):
    class NotTrackable(object):
      pass

    with self.assertRaises(ValueError):
      data_structures.List([NotTrackable()])

  def testCallNotImplemented(self):
    with self.assertRaisesRegexp(TypeError, "not callable"):
      data_structures.List()(1.)

  def testNoPop(self):
    with self.assertRaises(AttributeError):
      data_structures.List().pop()

  @test_util.run_in_graph_and_eager_modes
  def testTensorConversion(self):

    class ListToTensor(training.Model):

      def __init__(self):
        super(ListToTensor, self).__init__()
        self.l = [1., 2., 3.]

    self.assertAllEqual(
        [1., 2., 3.],
        self.evaluate(constant_op.constant(ListToTensor().l)))

    self.assertAllEqual(
        [1., 2., 3.],
        self.evaluate(array_ops.pack(ListToTensor().l)))

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
    has_sequences = set([data_structures.List(),
                         data_structures.List()])
    self.assertEqual(2, len(has_sequences))
    self.assertNotIn(data_structures.List(), has_sequences)

  def testIMul_zero(self):
    l = data_structures.List([])
    with self.assertRaisesRegexp(ValueError, "List only supports append"):
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

  def testLayerCollectionWithExternalMutation(self):
    l = []
    l_wrapper = data_structures.ListWrapper(l)
    layer = core.Dense(1)
    l.append(layer)
    self.assertEqual([layer], l_wrapper.layers)

  def testNotHashable(self):
    with self.assertRaises(TypeError):
      hash(data_structures.ListWrapper())

  def testDelItem(self):
    l = data_structures.ListWrapper([1, 2, 3, 4])
    del l[0]
    self.assertEqual(l, [2, 3, 4])
    self.assertUnableToSave(l, "Unable to save .*__delitem__")

  def testDelSlice(self):
    l = data_structures.ListWrapper([1, 2, 3, 4])
    del l[2:3]
    self.assertEqual(l, [1, 2, 4])
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
    l = data_structures.ListWrapper([1, 2, 3, 4])
    l *= -1
    self.assertEqual(l, [1, 2, 3, 4] * -1)
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
    l = data_structures.ListWrapper([1, 2, 3, 4])
    l.sort()
    self.assertEqual(l, [1, 2, 3, 4])
    # Regardless of being a no-op for the input list, we still refuse to save.
    # This is intentional since otherwise we would end up with a hard to debug
    # case for users (e.g. sometimes sort on a ListWrapper is trackable and
    # other times it is not).
    self.assertUnableToSave(l, "Unable to save .*sort")

  def assertUnableToSave(self, l, msg):
    l._maybe_initialize_trackable()  # pylint: disable=protected-access
    with self.assertRaisesRegexp(ValueError, msg):
      return l._checkpoint_dependencies  # pylint: disable=protected-access


class HasMapping(training.Model):

  def __init__(self):
    super(HasMapping, self).__init__()
    self.layer_dict = data_structures.Mapping(output=core.Dense(7))
    self.layer_dict["norm"] = data_structures.List()
    self.layer_dict["dense"] = data_structures.List()
    self.layer_dict["dense"].extend(
        [core.Dense(5),
         core.Dense(6, kernel_regularizer=math_ops.reduce_sum)])
    self.layer_dict["norm"].append(
        normalization.BatchNormalization())
    self.layer_dict["norm"].append(
        normalization.BatchNormalization())

  def call(self, x):
    aggregation = 0.
    for norm, dense in zip(self.layer_dict["norm"], self.layer_dict["dense"]):
      x = norm(dense(x))
      aggregation += math_ops.reduce_sum(x)
    return self.layer_dict["output"](x) / aggregation


class MappingTests(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testTracking(self):
    model = HasMapping()
    output = model(array_ops.ones([32, 2]))
    self.assertAllEqual([32, 7], output.shape.as_list())
    self.assertEqual(5, len(model.layers))
    six.assertCountEqual(self, model.layers, model.layer_dict.layers)
    self.assertEqual(1, len(model._checkpoint_dependencies))
    self.assertIs(model.layer_dict, model._checkpoint_dependencies[0].ref)
    self.evaluate([v.initializer for v in model.variables])
    test_var = model.layer_dict["output"].kernel
    self.evaluate(test_var.assign(array_ops.ones([6, 7])))
    save_path = os.path.join(self.get_temp_dir(), "ckpt")
    model.save_weights(save_path)
    self.evaluate(test_var.assign(array_ops.zeros([6, 7])))
    model.load_weights(save_path)
    self.assertAllEqual(numpy.ones([6, 7]),
                        self.evaluate(test_var))

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
      del mapping["a"]
    mapping.update(b=data_structures.Mapping())
    with self.assertRaises(ValueError):
      mapping.update({"b": data_structures.Mapping()})

  def testNonStringKeys(self):
    mapping = data_structures.Mapping()
    with self.assertRaises(TypeError):
      mapping[1] = data_structures.List()

  def testLayerCollectionWithExternalMutation(self):
    d = {}
    root = tracking.AutoTrackable()
    root.wrapper = d
    self.assertEqual([], root.wrapper.layers)
    self.assertEqual([], root.wrapper.trainable_weights)
    layer1 = core.Dense(1)
    layer2 = core.Dense(1)
    d["a"] = layer1
    d["b"] = layer2
    self.assertEqual([layer1, layer2], root.wrapper.layers)
    # The layers have still not created variables
    self.assertEqual([], root.wrapper.trainable_weights)

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
    with self.assertRaisesRegexp(TypeError, "unhashable"):
      set([a.d])

  def testDictWrapperBadKeys(self):
    a = tracking.AutoTrackable()
    a.d = {}
    a.d[1] = data_structures.List()
    model = training.Model()
    model.sub = a
    save_path = os.path.join(self.get_temp_dir(), "ckpt")
    with self.assertRaisesRegexp(ValueError, "non-string key"):
      model.save_weights(save_path)

  def testDictWrapperNoDependency(self):
    a = tracking.AutoTrackable()
    a.d = data_structures.NoDependency({})
    a.d[1] = [3]
    self.assertEqual([a], util.list_objects(a))
    model = training.Model()
    model.sub = a
    save_path = os.path.join(self.get_temp_dir(), "ckpt")
    model.save_weights(save_path)
    model.load_weights(save_path)

  def testNonStringKeyNotTrackableValue(self):
    a = tracking.AutoTrackable()
    a.d = {}
    a.d["a"] = [3]
    a.d[1] = data_structures.NoDependency([3])
    self.assertEqual([a, a.d, a.d["a"]], util.list_objects(a))
    model = training.Model()
    model.sub = a
    save_path = os.path.join(self.get_temp_dir(), "ckpt")
    model.save_weights(save_path)
    model.load_weights(save_path)

  def testNonAppendNotTrackable(self):
    # Non-append mutations (deleting or overwriting values) are OK when the
    # values aren't tracked.
    a = tracking.AutoTrackable()
    a.d = {}
    a.d["a"] = [3]
    a.d[1] = 3
    a.d[1] = 2
    self.assertEqual(2, a.d[1])
    del a.d[1]
    a.d[2] = data_structures.NoDependency(tracking.AutoTrackable())
    second = tracking.AutoTrackable()
    a.d[2] = data_structures.NoDependency(second)
    self.assertIs(second, a.d[2])
    self.assertEqual([a, a.d, a.d["a"]], util.list_objects(a))
    model = training.Model()
    model.sub = a
    save_path = os.path.join(self.get_temp_dir(), "ckpt")
    model.save_weights(save_path)
    model.load_weights(save_path)

  def testPopNoSave(self):
    model = training.Model()
    model.d = {}
    model.d["a"] = []
    model.d.pop("a")
    save_path = os.path.join(self.get_temp_dir(), "ckpt")
    with self.assertRaisesRegexp(ValueError, "Unable to save"):
      model.save_weights(save_path)

  def testExternalModificationNoSave(self):
    model = training.Model()
    external_reference = {}
    model.d = external_reference
    external_reference["a"] = []
    save_path = os.path.join(self.get_temp_dir(), "ckpt")
    with self.assertRaisesRegexp(ValueError, "modified outside the wrapper"):
      model.save_weights(save_path)

  def testOverwriteCanStillSave(self):
    model = training.Model()
    model.d = {}
    model.d["a"] = {}
    model.d["a"] = {}
    save_path = os.path.join(self.get_temp_dir(), "ckpt")
    model.save_weights(save_path)

  def testIter(self):
    model = training.Model()
    model.d = {1: 3}
    model.d[1] = 3
    self.assertEqual([1], list(model.d))
    new_dict = {}
    # This update() is super tricky. If the dict wrapper subclasses dict,
    # CPython will access its storage directly instead of calling any
    # methods/properties on the object. So the options are either not to
    # subclass dict (in which case update will call normal iter methods, but the
    # object won't pass isinstance checks) or to subclass dict and keep that
    # storage updated (no shadowing all its methods like ListWrapper).
    new_dict.update(model.d)
    self.assertEqual({1: 3}, new_dict)

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


class HasTuple(training.Model):

  def __init__(self):
    super(HasTuple, self).__init__()
    self.layer_list = (
        core.Dense(3), core.Dense(4),
        core.Dense(5, kernel_regularizer=math_ops.reduce_sum))
    self.layers_with_updates = (normalization.BatchNormalization(),)

  def call(self, x):
    aggregation = 0.
    for l in self.layer_list:
      x = l(x)
      aggregation += math_ops.reduce_sum(x)
    bn, = self.layers_with_updates
    return bn(x) / aggregation


class TupleTests(test.TestCase, parameterized.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testTracking(self):
    model = HasTuple()
    output = model(array_ops.ones([32, 2]))
    self.assertAllEqual([32, 5], output.shape.as_list())
    self.assertLen(model.layers, 4)
    self.assertLen(model.layer_list.layers, 3)
    six.assertCountEqual(
        self,
        model.layers,
        tuple(model.layer_list.layers) + model.layers_with_updates)
    self.assertEqual(3, model.layer_list.layers[0].units)
    self.assertEqual(4, model.layer_list.layers[1].units)
    self.assertEqual(5, model.layer_list.layers[2].units)
    self.assertLen(model._checkpoint_dependencies, 2)
    self.assertIs(model.layer_list, model._checkpoint_dependencies[0].ref)
    self.assertIs(model.layers_with_updates,
                  model._checkpoint_dependencies[1].ref)
    self.assertLen(
        model._checkpoint_dependencies[0].ref._checkpoint_dependencies, 3)
    self.evaluate([v.initializer for v in model.variables])
    self.evaluate(model.variables[0].assign([[1., 2., 3.], [4., 5., 6.]]))
    save_path = os.path.join(self.get_temp_dir(), "ckpt")
    model.save_weights(save_path)
    self.evaluate(model.variables[0].assign(array_ops.zeros([2, 3])))
    model.load_weights(save_path)
    self.assertAllEqual([[1., 2., 3.], [4., 5., 6.]],
                        self.evaluate(model.variables[0]))
    v = variables.Variable(1.)
    model.var_list = (v,)
    self.assertIn(id(v), [id(obj) for obj in model.variables])
    self.assertIn(id(v), [id(obj) for obj in model.trainable_variables])
    self.assertNotIn(id(v), [id(obj) for obj in model.non_trainable_variables])
    self.assertIn(id(model.layer_list[0].trainable_weights[0]),
                  [id(obj) for obj in model.trainable_weights])

  @parameterized.named_parameters(
      ("Module", module.Module),
      ("Model", training.Model),
  )
  def testSubModelTracking(self, module_subclass):
    model = module_subclass()
    model.v = variables.Variable(1.)
    self.assertIn(model.v, model.trainable_variables)
    model2 = module_subclass()
    model2.m = (model,)
    self.assertIn(model.v, model2.trainable_variables)

  def testSubSequentialTracking(self):

    class _Subclassed(training.Model):

      def __init__(self, wrapped):
        super(_Subclassed, self).__init__()
        self._wrapped = wrapped

      def call(self, x):
        return self._wrapped(x)

    model = sequential.Sequential()
    layer = core.Dense(1)
    model.add(layer)
    model2 = _Subclassed(model)
    model2(array_ops.ones([1, 2]))
    model2.m = (model,)
    self.assertIn(layer.kernel, model2.trainable_weights)

  def testJSONSerialization(self):
    obj = tracking.AutoTrackable()
    obj.l = (1,)
    json.dumps(obj.l, default=serialization.get_json_type)

  def testUpdatesForwarded(self):
    with ops.Graph().as_default():
      model = HasTuple()
      model_input = array_ops.ones([32, 2])
      model(model_input)
      self.assertNotEmpty(model.layers_with_updates[0].updates)
      self.assertEqual(set(model.layers_with_updates[0].updates),
                       set(model.updates))

    model = HasTuple()
    model_input = array_ops.ones([32, 2])
    model(model_input)
    self.assertEmpty(model.updates)

  @test_util.run_in_graph_and_eager_modes
  def testLossesForwarded(self):
    model = HasTuple()
    model_input = array_ops.ones([32, 2])
    model(model_input)
    self.assertLen(model.losses, 1)

  def testModelContainersCompareEqual(self):
    class HasEqualContainers(training.Model):

      def __init__(self):
        super(HasEqualContainers, self).__init__()
        self.l1 = ()
        self.l2 = ()

    model = HasEqualContainers()
    first_layer = HasEqualContainers()
    model.l1 = (first_layer,)
    second_layer = HasEqualContainers()
    model.l2 = (second_layer,)
    self.assertEqual((first_layer,), model.l1)
    d = {model.l1: 1, model.l2: 2}
    self.assertEqual(1, d[model.l1])
    self.assertEqual(1, d[(first_layer,)])
    self.assertEqual(2, d[model.l2])
    self.assertEqual(2, d[(second_layer,)])
    self.assertEqual([first_layer, second_layer], model.layers)

  @test_util.run_in_graph_and_eager_modes
  def testTensorConversion(self):

    class TupleToTensor(training.Model):

      def __init__(self):
        super(TupleToTensor, self).__init__()
        self.l = (1., 2., 3.)

    self.assertAllEqual(
        (1., 2., 3.),
        self.evaluate(constant_op.constant(TupleToTensor().l)))

    self.assertAllEqual(
        (1., 2., 3.),
        self.evaluate(array_ops.pack(TupleToTensor().l)))

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
