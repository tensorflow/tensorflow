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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import numpy as np
import six

from tensorflow.python import keras
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras import keras_parameterized
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

  def test_functional_model(self):
    inputs = keras.Input(10)
    outputs = keras.layers.Dense(1)(inputs)
    model = keras.Model(inputs, outputs)
    self._check_model_class(model.__class__)

  def test_sequential_model(self):
    model = keras.Sequential([keras.layers.Dense(1)])
    model_class = model.__class__.__bases__[0]
    self._check_model_class(model_class)

  def test_subclass_model(self):

    class MyModel(keras.Model):

      def call(self, x):
        return 2 * x

    model = MyModel()
    model_class = model.__class__.__bases__[0]
    self._check_model_class(model_class)

  def test_multiple_subclass_model(self):

    class Model1(keras.Model):
      pass

    class Model2(Model1):

      def call(self, x):
        return 2 * x

    model = Model2()
    model_class = model.__class__.__bases__[0].__bases__[0]
    self._check_model_class(model_class)

  def test_user_provided_metaclass(self):

    @six.add_metaclass(abc.ABCMeta)
    class AbstractModel(keras.Model):

      @abc.abstractmethod
      def call(self, inputs):
        """Calls the model."""

    class MyModel(AbstractModel):

      def call(self, inputs):
        return 2 * inputs

    with self.assertRaisesRegexp(TypeError, 'instantiate abstract class'):
      AbstractModel()

    model = MyModel()
    model_class = model.__class__.__bases__[0].__bases__[0]
    self._check_model_class(model_class)

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

  def test_fit_error(self):
    if not ops.executing_eagerly_outside_functions():
      # Error only appears on the v2 class.
      return

    model = keras.Sequential([keras.layers.Dense(1)])
    model.compile('sgd', 'mse')
    x, y = np.ones((10, 10)), np.ones((10, 1))
    with context.graph_mode():
      with self.assertRaisesRegexp(
          ValueError, 'instance was constructed with eager mode enabled'):
        model.fit(x, y, batch_size=2)


if __name__ == '__main__':
  test.main()
