# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for Keras utilities to split v1 and v2 classes."""

import abc

import numpy as np

from tensorflow.python import keras
from tensorflow.python.framework import ops
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import base_layer_v1
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.engine import training_v1
from tensorflow.python.platform import test


@keras_parameterized.run_all_keras_modes
class SplitUtilsTest(keras_parameterized.TestCase):

  def _check_model_class(self, model_class):
    if ops.executing_eagerly_outside_functions():
      self.assertEqual(model_class, training.Model)
    else:
      self.assertEqual(model_class, training_v1.Model)

  def _check_layer_class(self, layer):
    if ops.executing_eagerly_outside_functions():
      self.assertIsInstance(layer, base_layer.Layer)
      self.assertNotIsInstance(layer, base_layer_v1.Layer)
    else:
      self.assertIsInstance(layer, base_layer_v1.Layer)

  def test_functional_model(self):
    inputs = keras.Input(10)
    outputs = keras.layers.Dense(1)(inputs)
    model = keras.Model(inputs, outputs)
    self._check_model_class(model.__class__.__bases__[0])
    self._check_layer_class(model)

  def test_subclass_model_with_functional_init(self):
    inputs = keras.Input(10)
    outputs = keras.layers.Dense(1)(inputs)

    class MyModel(keras.Model):
      pass

    model = MyModel(inputs, outputs)
    model_class = model.__class__.__bases__[0].__bases__[0]
    self._check_model_class(model_class)
    self._check_layer_class(model)

  def test_subclass_model_with_functional_init_interleaved_v1_functional(self):
    with ops.Graph().as_default():
      inputs = keras.Input(10)
      outputs = keras.layers.Dense(1)(inputs)
      _ = keras.Model(inputs, outputs)

    inputs = keras.Input(10)
    outputs = keras.layers.Dense(1)(inputs)

    class MyModel(keras.Model):
      pass

    model = MyModel(inputs, outputs)
    model_class = model.__class__.__bases__[0].__bases__[0]
    self._check_model_class(model_class)
    self._check_layer_class(model)

  def test_sequential_model(self):
    model = keras.Sequential([keras.layers.Dense(1)])
    model_class = model.__class__.__bases__[0].__bases__[0]
    self._check_model_class(model_class)
    self._check_layer_class(model)

  def test_subclass_model(self):

    class MyModel(keras.Model):

      def call(self, x):
        return 2 * x

    model = MyModel()
    model_class = model.__class__.__bases__[0]
    self._check_model_class(model_class)
    self._check_layer_class(model)

  def test_layer(self):
    class IdentityLayer(base_layer.Layer):
      """A layer that returns it's input.

      Useful for testing a layer without a variable.
      """

      def call(self, inputs):
        return inputs

    layer = IdentityLayer()
    self._check_layer_class(layer)

  def test_multiple_subclass_model(self):

    class Model1(keras.Model):
      pass

    class Model2(Model1):

      def call(self, x):
        return 2 * x

    model = Model2()
    model_class = model.__class__.__bases__[0].__bases__[0]
    self._check_model_class(model_class)
    self._check_layer_class(model)

  def test_user_provided_metaclass(self):

    class AbstractModel(keras.Model, metaclass=abc.ABCMeta):

      @abc.abstractmethod
      def call(self, inputs):
        """Calls the model."""

    class MyModel(AbstractModel):

      def call(self, inputs):
        return 2 * inputs

    with self.assertRaisesRegex(TypeError, 'instantiate abstract class'):
      AbstractModel()  # pylint: disable=abstract-class-instantiated

    model = MyModel()
    model_class = model.__class__.__bases__[0].__bases__[0]
    self._check_model_class(model_class)
    self._check_layer_class(model)

  def test_multiple_inheritance(self):

    class Return2(object):

      def return_2(self):
        return 2

    class MyModel(keras.Model, Return2):

      def call(self, x):
        return self.return_2() * x

    model = MyModel()
    bases = model.__class__.__bases__
    self._check_model_class(bases[0])
    self.assertEqual(bases[1], Return2)
    self.assertEqual(model.return_2(), 2)
    self._check_layer_class(model)

  def test_fit_error(self):
    if not ops.executing_eagerly_outside_functions():
      # Error only appears on the v2 class.
      return

    model = keras.Sequential([keras.layers.Dense(1)])
    model.compile('sgd', 'mse')
    x, y = np.ones((10, 10)), np.ones((10, 1))
    with ops.get_default_graph().as_default():
      with self.assertRaisesRegex(
          ValueError, 'instance was constructed with eager mode enabled'):
        model.fit(x, y, batch_size=2)


if __name__ == '__main__':
  test.main()
