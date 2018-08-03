# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Keras integration tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib import autograph


class MinimalKeras(tf.keras.Model):

  def call(self, x):
    return x * 3


class ModelWithStaticConditional(object):

  def __init__(self, initial):
    self.initial = initial
    if self.initial:
      self.h = 15

  @autograph.convert()
  def call(self):
    x = 10
    if self.initial:
      x += self.h
    return x


class BasicBlock(tf.keras.Model):

  def __init__(self):
    super(BasicBlock, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(8, 3)
    self.pool = tf.keras.layers.GlobalAveragePooling2D()
    self.dense = tf.keras.layers.Dense(3)

  def call(self, x):
    x = self.conv1(x)
    x = self.pool(x)
    x = self.dense(x)
    return x


class CompoundModel(tf.keras.Model):

  def __init__(self):
    super(CompoundModel, self).__init__()
    self.block = BasicBlock()

  @autograph.convert(recursive=True)
  def call(self, x):
    x = self.block(x)  # pylint: disable=not-callable
    return x


class KerasTest(tf.test.TestCase):

  def test_basic(self):
    MinimalKeras()

  def test_conditional_attributes_False(self):
    model = ModelWithStaticConditional(False)
    self.assertEqual(model.call(), 10)

  def test_conditional_attributes_True(self):
    model = ModelWithStaticConditional(True)
    self.assertEqual(model.call(), 25)

  def test_recursive_true(self):
    with self.assertRaisesRegexp(NotImplementedError,
                                 'Object conversion is not yet supported.'):
      with tf.Graph().as_default():
        model = CompoundModel()
        model.build(tf.TensorShape((None, 10, 10, 1)))
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
          sess.run(init)
          sample_input = tf.random_uniform((1, 10, 10, 1))
          output = model(sample_input)  # pylint: disable=not-callable
          self.assertEqual(sess.run(output).shape, (1, 3))


if __name__ == '__main__':
  tf.test.main()
