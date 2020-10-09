# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

import tensorflow as tf


class ModuleTest(tf.test.TestCase):

  def test_module_discover_layer_variable(self):
    m = tf.Module()
    m.a = tf.keras.layers.Dense(1)
    m.b = tf.keras.layers.Dense(2)

    # The weights of the layer has not been created yet.
    self.assertEmpty(m.variables)
    self.assertLen(m.submodules, 2)

    inputs = tf.keras.layers.Input((1,))
    m.a(inputs)
    m.b(inputs)

    variable_list = m.variables
    self.assertLen(variable_list, 4)
    self.assertIs(variable_list[0], m.a.kernel)
    self.assertIs(variable_list[1], m.a.bias)
    self.assertIs(variable_list[2], m.b.kernel)
    self.assertIs(variable_list[3], m.b.bias)

  def test_model_discover_submodule(self):
    m = tf.keras.models.Sequential(
        layers=[tf.keras.layers.Dense(1), tf.keras.layers.Dense(2)])

    self.assertEqual(m.submodules, (m.layers[0], m.layers[1]))
    m(tf.keras.layers.Input((1,)))
    self.assertLen(m.variables, 4)

  def test_model_wrapped_in_module_discovers_submodules(self):
    linear = tf.keras.models.Sequential(
        [tf.keras.layers.Dense(units=1, input_shape=[1])])
    linear.compile(optimizer="sgd", loss="mean_squared_error")
    m = tf.Module()
    m.l = linear
    self.assertNotEmpty(m.submodules)
    self.assertLen(m.variables, 2)


if __name__ == "__main__":
  tf.test.main()
