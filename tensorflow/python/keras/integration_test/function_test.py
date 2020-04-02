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


class FunctionTest(tf.test.TestCase):

  def testStandardTrainingLoopInFunction(self):
    layer = tf.keras.layers.Dense(2)
    dataset = (
        tf.data.Dataset.from_tensors((tf.ones([784]), tf.ones([], tf.int32)))
        .map(lambda x, y: (x, y))
        .repeat(10)
        .batch(32))
    optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train():
      for x, y in dataset:
        with tf.GradientTape() as tape:
          out = layer(x)
          loss = tf.reduce_mean(
              tf.nn.sparse_softmax_cross_entropy_with_logits(
                  logits=out, labels=y))
        layer_variables = layer.trainable_variables
        gradients = tape.gradient(loss, layer_variables)
        optimizer.apply_gradients(zip(gradients, layer_variables))

    train()


if __name__ == '__main__':
  tf.test.main()
