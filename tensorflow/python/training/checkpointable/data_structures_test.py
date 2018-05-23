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


if __name__ == "__main__":
  test.main()
