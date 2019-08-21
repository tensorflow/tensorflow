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
"""Keras models for use in Model subclassing tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import keras


# pylint: disable=missing-docstring,not-callable
class SimpleTestModel(keras.Model):

  def __init__(self, use_bn=False, use_dp=False, num_classes=10):
    super(SimpleTestModel, self).__init__(name='test_model')
    self.use_bn = use_bn
    self.use_dp = use_dp
    self.num_classes = num_classes

    self.dense1 = keras.layers.Dense(32, activation='relu')
    self.dense2 = keras.layers.Dense(num_classes, activation='softmax')
    if self.use_dp:
      self.dp = keras.layers.Dropout(0.5)
    if self.use_bn:
      self.bn = keras.layers.BatchNormalization(axis=-1)

  def call(self, x):
    x = self.dense1(x)
    if self.use_dp:
      x = self.dp(x)
    if self.use_bn:
      x = self.bn(x)
    return self.dense2(x)


class SimpleConvTestModel(keras.Model):

  def __init__(self, num_classes=10):
    super(SimpleConvTestModel, self).__init__(name='test_model')
    self.num_classes = num_classes

    self.conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu')
    self.flatten = keras.layers.Flatten()
    self.dense1 = keras.layers.Dense(num_classes, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    return self.dense1(x)


class MultiIOTestModel(keras.Model):

  def __init__(self, use_bn=False, use_dp=False, num_classes=(2, 3)):
    super(MultiIOTestModel, self).__init__(name='test_model')
    self.use_bn = use_bn
    self.use_dp = use_dp
    self.num_classes = num_classes

    self.dense1 = keras.layers.Dense(32, activation='relu')
    self.dense2 = keras.layers.Dense(num_classes[0], activation='softmax')
    self.dense3 = keras.layers.Dense(num_classes[1], activation='softmax')
    if use_dp:
      self.dp = keras.layers.Dropout(0.5)
    if use_bn:
      self.bn = keras.layers.BatchNormalization()

  def call(self, inputs):
    x1, x2 = inputs
    x1 = self.dense1(x1)
    x2 = self.dense1(x2)
    if self.use_dp:
      x1 = self.dp(x1)
    if self.use_bn:
      x2 = self.bn(x2)
    return [self.dense2(x1), self.dense3(x2)]


class NestedTestModel1(keras.Model):
  """A model subclass nested inside a model subclass.
  """

  def __init__(self, num_classes=2):
    super(NestedTestModel1, self).__init__(name='nested_model_1')
    self.num_classes = num_classes
    self.dense1 = keras.layers.Dense(32, activation='relu')
    self.dense2 = keras.layers.Dense(num_classes, activation='relu')
    self.bn = keras.layers.BatchNormalization()
    self.test_net = SimpleTestModel(num_classes=4,
                                    use_bn=True,
                                    use_dp=True)

  def call(self, inputs):
    x = self.dense1(inputs)
    x = self.bn(x)
    x = self.test_net(x)
    return self.dense2(x)


class NestedTestModel2(keras.Model):
  """A model subclass with a functional-API graph network inside.
  """

  def __init__(self, num_classes=2):
    super(NestedTestModel2, self).__init__(name='nested_model_2')
    self.num_classes = num_classes
    self.dense1 = keras.layers.Dense(32, activation='relu')
    self.dense2 = keras.layers.Dense(num_classes, activation='relu')
    self.bn = self.bn = keras.layers.BatchNormalization()
    self.test_net = self.get_functional_graph_model(32, 4)

  @staticmethod
  def get_functional_graph_model(input_dim, num_classes):
    # A simple functional-API model (a.k.a. graph network)
    inputs = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(32, activation='relu')(inputs)
    x = keras.layers.BatchNormalization()(x)
    outputs = keras.layers.Dense(num_classes)(x)
    return keras.Model(inputs, outputs)

  def call(self, inputs):
    x = self.dense1(inputs)
    x = self.bn(x)
    x = self.test_net(x)
    return self.dense2(x)


def get_nested_model_3(input_dim, num_classes):
  # A functional-API model with a subclassed model inside.
  # NOTE: this requires the inner subclass to implement `compute_output_shape`.

  inputs = keras.Input(shape=(input_dim,))
  x = keras.layers.Dense(32, activation='relu')(inputs)
  x = keras.layers.BatchNormalization()(x)

  class Inner(keras.Model):

    def __init__(self):
      super(Inner, self).__init__()
      self.dense1 = keras.layers.Dense(32, activation='relu')
      self.dense2 = keras.layers.Dense(5, activation='relu')
      self.bn = keras.layers.BatchNormalization()

    def call(self, inputs):
      x = self.dense1(inputs)
      x = self.dense2(x)
      return self.bn(x)

  test_model = Inner()
  x = test_model(x)
  outputs = keras.layers.Dense(num_classes)(x)
  return keras.Model(inputs, outputs, name='nested_model_3')


class CustomCallModel(keras.Model):

  def __init__(self):
    super(CustomCallModel, self).__init__()
    self.dense1 = keras.layers.Dense(1, activation='relu')
    self.dense2 = keras.layers.Dense(1, activation='softmax')

  def call(self, first, second, fiddle_with_output='no', training=True):
    combined = self.dense1(first) + self.dense2(second)
    if fiddle_with_output == 'yes':
      return 10. * combined
    else:
      return combined


class TrainingNoDefaultModel(keras.Model):

  def __init__(self):
    super(TrainingNoDefaultModel, self).__init__()
    self.dense1 = keras.layers.Dense(1)

  def call(self, x, training):
    return self.dense1(x)


class TrainingMaskingModel(keras.Model):

  def __init__(self):
    super(TrainingMaskingModel, self).__init__()
    self.dense1 = keras.layers.Dense(1)

  def call(self, x, training=False, mask=None):
    return self.dense1(x)
