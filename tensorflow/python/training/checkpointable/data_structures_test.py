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

import copy
import os

import numpy
import six

from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.layers import normalization
from tensorflow.python.layers import core as non_keras_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.training.checkpointable import data_structures
from tensorflow.python.training.checkpointable import tracking
from tensorflow.python.training.checkpointable import util


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
            list(sequence=[core.Dense(11)]) + [core.Dense(12)]))
    self.layers_with_updates = data_structures.List(
        sequence=(normalization.BatchNormalization(),))

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

  def testNotCheckpointable(self):
    class NotCheckpointable(object):
      pass

    with self.assertRaises(ValueError):
      data_structures.List([NotCheckpointable()])

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

  def testListWrapperBasic(self):
    # _ListWrapper, unlike List, compares like the built-in list type (since it
    # is used to automatically replace lists).
    a = tracking.Checkpointable()
    b = tracking.Checkpointable()
    self.assertEqual([a, a],
                     [a, a])
    self.assertEqual(data_structures._ListWrapper([a, a]),
                     data_structures._ListWrapper([a, a]))
    self.assertEqual([a, a],
                     data_structures._ListWrapper([a, a]))
    self.assertEqual(data_structures._ListWrapper([a, a]),
                     [a, a])
    self.assertNotEqual([a, a],
                        [b, a])
    self.assertNotEqual(data_structures._ListWrapper([a, a]),
                        data_structures._ListWrapper([b, a]))
    self.assertNotEqual([a, a],
                        data_structures._ListWrapper([b, a]))
    self.assertLess([a], [a, b])
    self.assertLess(data_structures._ListWrapper([a]),
                    data_structures._ListWrapper([a, b]))
    self.assertLessEqual([a], [a, b])
    self.assertLessEqual(data_structures._ListWrapper([a]),
                         data_structures._ListWrapper([a, b]))
    self.assertGreater([a, b], [a])
    self.assertGreater(data_structures._ListWrapper([a, b]),
                       data_structures._ListWrapper([a]))
    self.assertGreaterEqual([a, b], [a])
    self.assertGreaterEqual(data_structures._ListWrapper([a, b]),
                            data_structures._ListWrapper([a]))
    self.assertEqual([a], data_structures._ListWrapper([a]))
    self.assertEqual([a], list(data_structures.List([a])))
    self.assertEqual([a, a], data_structures._ListWrapper([a]) + [a])
    self.assertEqual([a, a], [a] + data_structures._ListWrapper([a]))
    self.assertIsInstance(data_structures._ListWrapper([a]), list)

  def testWrapperChangesList(self):
    l = []
    l_wrapper = data_structures._ListWrapper(l)
    l_wrapper.append(1)
    self.assertEqual([1], l)

  def testListChangesWrapper(self):
    l = []
    l_wrapper = data_structures._ListWrapper(l)
    l.append(1)
    self.assertEqual([1], l_wrapper)

  def testLayerCollectionWithExternalMutation(self):
    l = []
    l_wrapper = data_structures._ListWrapper(l)
    layer = core.Dense(1)
    l.append(layer)
    self.assertEqual([layer], l_wrapper.layers)

  def testHashing(self):
    has_sequences = set([data_structures.List(),
                         data_structures.List()])
    self.assertEqual(2, len(has_sequences))
    self.assertNotIn(data_structures.List(), has_sequences)
    with self.assertRaises(TypeError):
      has_sequences.add(data_structures._ListWrapper([]))


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
  @test_util.run_v1_only("b/120545219")
  def testTracking(self):
    model = HasMapping()
    output = model(array_ops.ones([32, 2]))
    self.assertAllEqual([32, 7], output.shape)
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
    root = tracking.Checkpointable()
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
    a = tracking.Checkpointable()
    a.d = {}
    self.assertEqual({}, a.d)
    self.assertFalse({} != a.d)  # pylint: disable=g-explicit-bool-comparison
    self.assertNotEqual({1: 2}, a.d)
    with self.assertRaisesRegexp(TypeError, "unhashable"):
      set([a.d])

  def testDictWrapperBadKeys(self):
    a = tracking.Checkpointable()
    a.d = {}
    a.d[1] = data_structures.List()
    model = training.Model()
    model.sub = a
    save_path = os.path.join(self.get_temp_dir(), "ckpt")
    with self.assertRaisesRegexp(ValueError, "non-string key"):
      model.save_weights(save_path)

  def testDictWrapperNoDependency(self):
    a = tracking.Checkpointable()
    a.d = data_structures.NoDependency({})
    a.d[1] = [3]
    self.assertEqual([a], util.list_objects(a))
    model = training.Model()
    model.sub = a
    save_path = os.path.join(self.get_temp_dir(), "ckpt")
    model.save_weights(save_path)
    model.load_weights(save_path)

  def testNonStringKeyNotCheckpointableValue(self):
    a = tracking.Checkpointable()
    a.d = {}
    a.d["a"] = [3]
    a.d[1] = data_structures.NoDependency([3])
    self.assertEqual([a, a.d, a.d["a"]], util.list_objects(a))
    model = training.Model()
    model.sub = a
    save_path = os.path.join(self.get_temp_dir(), "ckpt")
    model.save_weights(save_path)
    model.load_weights(save_path)

  def testNonAppendNotCheckpointable(self):
    # Non-append mutations (deleting or overwriting values) are OK when the
    # values aren't tracked.
    a = tracking.Checkpointable()
    a.d = {}
    a.d["a"] = [3]
    a.d[1] = 3
    a.d[1] = 2
    self.assertEqual(2, a.d[1])
    del a.d[1]
    a.d[2] = data_structures.NoDependency(tracking.Checkpointable())
    second = tracking.Checkpointable()
    a.d[2] = data_structures.NoDependency(second)
    self.assertIs(second, a.d[2])
    self.assertEqual([a, a.d, a.d["a"]], util.list_objects(a))
    model = training.Model()
    model.sub = a
    save_path = os.path.join(self.get_temp_dir(), "ckpt")
    model.save_weights(save_path)
    model.load_weights(save_path)

  def testDelNoSave(self):
    model = training.Model()
    model.d = {}
    model.d["a"] = []
    del model.d["a"]
    save_path = os.path.join(self.get_temp_dir(), "ckpt")
    with self.assertRaisesRegexp(ValueError, "overwritten or deleted"):
      model.save_weights(save_path)

  def testPopNoSave(self):
    model = training.Model()
    model.d = {}
    model.d["a"] = []
    model.d.pop("a")
    save_path = os.path.join(self.get_temp_dir(), "ckpt")
    with self.assertRaisesRegexp(ValueError, "overwritten or deleted"):
      model.save_weights(save_path)

  def testExternalModificationNoSave(self):
    model = training.Model()
    external_reference = {}
    model.d = external_reference
    external_reference["a"] = []
    save_path = os.path.join(self.get_temp_dir(), "ckpt")
    with self.assertRaisesRegexp(ValueError, "modified outside the wrapper"):
      model.save_weights(save_path)

  def testOverwriteNoSave(self):
    model = training.Model()
    model.d = {}
    model.d["a"] = {}
    model.d["a"] = {}
    save_path = os.path.join(self.get_temp_dir(), "ckpt")
    with self.assertRaisesRegexp(ValueError, "overwritten or deleted"):
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
    # storage updated (no shadowing all its methods like _ListWrapper).
    new_dict.update(model.d)
    self.assertEqual({1: 3}, new_dict)

  def testListShallowCopy(self):
    root = tracking.Checkpointable()
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
    root = tracking.Checkpointable()
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
    root = tracking.Checkpointable()
    orig_dict = {"a": [1.]}
    root.a = orig_dict
    copied = copy.copy(root.a)
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
    root = tracking.Checkpointable()
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

  def testShallowCopyCheckpointable(self):
    original = tracking.Checkpointable()
    original_sub = tracking.Checkpointable()
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

  def testDeepCopyCheckpointable(self):
    original = tracking.Checkpointable()
    original_sub = tracking.Checkpointable()
    original.a = [[1.]]
    original.b = {"a": original_sub}
    deep_copied = copy.deepcopy(original)
    self.assertIsNot(original, deep_copied)
    self.assertIsNot(original_sub, deep_copied.b["a"])
    self.assertEqual([[1.]], deep_copied.a)
    self.assertIsInstance(deep_copied.b["a"], tracking.Checkpointable)
    deps = util.list_objects(deep_copied)
    self.assertIn(deep_copied.a, deps)
    self.assertIn(deep_copied.b, deps)
    self.assertIn(deep_copied.b["a"], deps)
    self.assertNotIn(original_sub, deps)

  def testConstructableFromSequence(self):
    result = data_structures._DictWrapper([(1, 2), (3, 4)])
    self.assertIsInstance(result, dict)
    self.assertEqual({1: 2, 3: 4}, result)

if __name__ == "__main__":
  test.main()
