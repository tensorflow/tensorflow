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

import os

from absl.testing import parameterized
import numpy
import six

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import combinations
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.layers import normalization
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training.tracking import base
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.training.tracking import util


class HasList(training.Model):

  def __init__(self):
    super(HasList, self).__init__()
    self.layer_list = data_structures.wrap_or_unwrap([core.Dense(3)])
    self.layer_list.append(core.Dense(4))
    self.layer_list.extend(
        [core.Dense(5),
         core.Dense(6, kernel_regularizer=math_ops.reduce_sum)])
    self.layer_list += [
        core.Dense(7, bias_regularizer=math_ops.reduce_sum),
        core.Dense(8)
    ]
    self.layer_list += (
        data_structures.wrap_or_unwrap([core.Dense(9)]) +
        data_structures.wrap_or_unwrap([core.Dense(10)]))
    self.layer_list.extend(
        data_structures.wrap_or_unwrap(
            list([core.Dense(11)]) + [core.Dense(12)]))
    self.layers_with_updates = data_structures.wrap_or_unwrap(
        [normalization.BatchNormalization()])

  def call(self, x):
    aggregation = 0.
    for l in self.layer_list:
      x = l(x)
      aggregation += math_ops.reduce_sum(x)
    bn, = self.layers_with_updates
    return bn(x) / aggregation


class ListTests(keras_parameterized.TestCase):

  @combinations.generate(combinations.combine(mode=["graph", "eager"]))
  def testTracking(self):
    with self.test_session():
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
          10,
          len(model._checkpoint_dependencies[0].ref._checkpoint_dependencies))
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
    self.assertTrue(any(v is t for t in model.variables))
    self.assertTrue(any(v is t for t in model.trainable_variables))
    self.assertFalse(any(v is t for t in model.non_trainable_variables))
    self.assertTrue(any(model.layer_list[0].trainable_weights[0]
                        is t for t in model.trainable_weights))

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

  @combinations.generate(combinations.combine(mode=["graph", "eager"]))
  def testUpdatesForwarded(self):
    model = HasList()
    model_input = array_ops.ones([32, 2])
    model(model_input)
    if context.executing_eagerly():
      self.assertEqual(0, len(model.updates))
    else:
      self.assertGreater(len(model.layers_with_updates[0].updates), 0)
      self.assertEqual(set(model.layers_with_updates[0].updates),
                       set(model.updates))

  @combinations.generate(combinations.combine(mode=["graph", "eager"]))
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

  @combinations.generate(combinations.combine(mode=["graph", "eager"]))
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
        self.evaluate(gen_array_ops.Pack(values=ListToTensor().l)))


class ListWrapperTest(test.TestCase):

  def testLayerCollectionWithExternalMutation(self):
    l = []
    l_wrapper = data_structures.wrap_or_unwrap(l)
    layer = core.Dense(1)
    l.append(layer)
    self.assertEqual([layer], l_wrapper.layers)


class HasMapping(training.Model):

  def __init__(self):
    super(HasMapping, self).__init__()
    self.layer_dict = data_structures.wrap_or_unwrap(dict(output=core.Dense(7)))
    self.layer_dict["norm"] = data_structures.wrap_or_unwrap([])
    self.layer_dict["dense"] = data_structures.wrap_or_unwrap([])
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


class MappingTests(keras_parameterized.TestCase):

  @combinations.generate(combinations.combine(mode=["graph", "eager"]))
  def testTracking(self):
    with self.test_session():
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

  def testLayerCollectionWithExternalMutation(self):
    d = {}
    root = module.Module()
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

  def testDictWrapperBadKeys(self):
    a = module.Module()
    a.d = {}
    a.d[1] = data_structures.wrap_or_unwrap([])
    model = training.Model()
    model.sub = a
    save_path = os.path.join(self.get_temp_dir(), "ckpt")
    with self.assertRaisesRegex(ValueError, "non-string key"):
      model.save_weights(save_path)

  def testDictWrapperNoDependency(self):
    a = module.Module()
    a.d = data_structures.NoDependency({})
    a.d[1] = [3]
    self.assertEqual([a], util.list_objects(a))
    model = training.Model()
    model.sub = a
    save_path = os.path.join(self.get_temp_dir(), "ckpt")
    model.save_weights(save_path)
    model.load_weights(save_path)

  def testNonStringKeyNotTrackableValue(self):
    a = module.Module()
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
    a = module.Module()
    a.d = {}
    a.d["a"] = [3]
    a.d[1] = 3
    a.d[1] = 2
    self.assertEqual(2, a.d[1])
    del a.d[1]
    a.d[2] = data_structures.NoDependency(module.Module())
    second = module.Module()
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
    with self.assertRaisesRegex(ValueError, "Unable to save"):
      model.save_weights(save_path)

  def testExternalModificationNoSave(self):
    model = training.Model()
    external_reference = {}
    model.d = external_reference
    external_reference["a"] = []
    save_path = os.path.join(self.get_temp_dir(), "ckpt")
    with self.assertRaisesRegex(ValueError, "modified outside the wrapper"):
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


class TupleTests(keras_parameterized.TestCase):

  @combinations.generate(combinations.combine(mode=["graph", "eager"]))
  def testTracking(self):
    with self.test_session():
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
      self.assertNotIn(id(v),
                       [id(obj) for obj in model.non_trainable_variables])
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

  @combinations.generate(combinations.combine(mode=["graph", "eager"]))
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

  @combinations.generate(combinations.combine(mode=["graph", "eager"]))
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
        self.evaluate(gen_array_ops.Pack(values=TupleToTensor().l)))


class InterfaceTests(keras_parameterized.TestCase):

  def testNoDependency(self):
    root = module.Module()
    hasdep = module.Module()
    root.hasdep = hasdep
    nodep = module.Module()
    root.nodep = data_structures.NoDependency(nodep)
    self.assertEqual(1, len(root._checkpoint_dependencies))
    self.assertIs(root._checkpoint_dependencies[0].ref, root.hasdep)
    self.assertIs(root.hasdep, hasdep)
    self.assertIs(root.nodep, nodep)

    class NoDependencyModel(training.Model):

      @base.no_automatic_dependency_tracking
      def __init__(self):
        super(NoDependencyModel, self).__init__()
        self.a = []
        self.b = module.Module()

    nodeps = NoDependencyModel()
    self.assertEqual([nodeps], util.list_objects(nodeps))

  @combinations.generate(combinations.combine(mode=["graph", "eager"]))
  def testDictionariesBasic(self):
    a = training.Model()
    b = training.Model()
    a.attribute = {"b": b}
    c = training.Model()
    a.attribute["c"] = []
    a.attribute["c"].append(c)
    a_deps = util.list_objects(a)
    self.assertIn(b, a_deps)
    self.assertIn(c, a_deps)
    self.assertIs(b, a.attribute["b"])
    six.assertCountEqual(
        self,
        ["b", "c"],
        [dep.name for dep in a.attribute._checkpoint_dependencies])
    self.assertEqual([b, c], a.layers)
    self.assertEqual([b, c], a.attribute.layers)
    self.assertEqual([c], a.attribute["c"].layers)
    checkpoint = util.Checkpoint(a=a)
    save_path = checkpoint.save(os.path.join(self.get_temp_dir(), "ckpt"))
    with self.cached_session():
      checkpoint.restore(save_path).assert_consumed().initialize_or_restore()

  @combinations.generate(combinations.combine(mode=["graph", "eager"]))
  def testNoDepList(self):
    a = training.Model()
    a.l1 = data_structures.NoDependency([])
    a.l1.insert(1, 0)
    self.assertIsInstance(a.l1, list)
    checkpoint = util.Checkpoint(a=a)
    checkpoint.save(os.path.join(self.get_temp_dir(), "ckpt"))
    a.l2 = []
    a.l2.insert(1, module.Module())
    with self.assertRaisesRegex(ValueError, "A list element was replaced"):
      checkpoint.save(os.path.join(self.get_temp_dir(), "ckpt"))


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
