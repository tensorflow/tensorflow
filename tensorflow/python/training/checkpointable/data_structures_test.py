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

import numpy

from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import test_util
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.layers import normalization
from tensorflow.python.layers import core as non_keras_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training.checkpointable import data_structures


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

  @test_util.run_in_graph_and_eager_modes()
  def testTracking(self):
    model = HasList()
    output = model(array_ops.ones([32, 2]))
    self.assertAllEqual([32, 12], output.shape)
    self.assertEqual(2, len(model.layers))
    self.assertIs(model.layer_list, model.layers[0])
    self.assertEqual(10, len(model.layers[0].layers))
    for index in range(10):
      self.assertEqual(3 + index, model.layers[0].layers[index].units)
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

  @test_util.run_in_graph_and_eager_modes()
  def testLossesForwarded(self):
    model = HasList()
    model_input = array_ops.ones([32, 2])
    model(model_input)
    self.assertEqual(2, len(model.losses))

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

  def testHashing(self):
    has_sequences = set([data_structures.List(),
                         data_structures.List()])
    self.assertEqual(2, len(has_sequences))
    self.assertNotIn(data_structures.List(), has_sequences)


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

  @test_util.run_in_graph_and_eager_modes()
  def testTracking(self):
    model = HasMapping()
    output = model(array_ops.ones([32, 2]))
    self.assertAllEqual([32, 7], output.shape)
    self.assertEqual(1, len(model.layers))
    self.assertIs(model.layer_dict, model.layers[0])
    self.assertEqual(3, len(model.layers[0].layers))
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

  def testHashing(self):
    has_mappings = set([data_structures.Mapping(),
                        data_structures.Mapping()])
    self.assertEqual(2, len(has_mappings))
    self.assertNotIn(data_structures.Mapping(), has_mappings)

if __name__ == "__main__":
  test.main()
