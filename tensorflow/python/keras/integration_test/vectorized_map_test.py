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


class VectorizedMapTest(tf.test.TestCase):

  def test_vectorized_map(self):
    batch_size = 10
    num_features = 32
    layer = tf.keras.layers.Dense(1)

    def model_fn(arg):
      with tf.GradientTape() as g:
        inp, label = arg
        inp = tf.expand_dims(inp, 0)
        label = tf.expand_dims(label, 0)
        prediction = layer(inp)
        loss = tf.nn.l2_loss(label - prediction)
      return g.gradient(loss, (layer.kernel, layer.bias))

    inputs = tf.random.uniform([batch_size, num_features])
    labels = tf.random.uniform([batch_size, 1])
    per_example_gradients = tf.vectorized_map(model_fn, (inputs, labels))
    self.assertEqual(per_example_gradients[0].shape,
                     (batch_size, num_features, 1))
    self.assertEqual(per_example_gradients[1].shape, (batch_size, 1))


if __name__ == "__main__":
  tf.test.main()
