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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.eager.python import tfe
from tensorflow.contrib.eager.python.examples.rnn_colorbot import rnn_colorbot


LABEL_DIMENSION = 5


def device():
  return "/device:GPU:0" if tfe.num_gpus() else "/device:CPU:0"


def random_dataset():
  batch_size = 64
  time_steps = 10
  alphabet = 50
  chars = tf.one_hot(
      tf.random_uniform(
          [batch_size, time_steps], minval=0, maxval=alphabet, dtype=tf.int32),
      alphabet)
  sequence_length = tf.constant(
      [time_steps for _ in range(batch_size)], dtype=tf.int64)
  labels = tf.random_normal([batch_size, LABEL_DIMENSION])
  return tf.data.Dataset.from_tensors((labels, chars, sequence_length))


class RNNColorbotTest(tf.test.TestCase):

  def testTrainOneEpoch(self):
    model = rnn_colorbot.RNNColorbot(
        rnn_cell_sizes=[256, 128, 64],
        label_dimension=LABEL_DIMENSION,
        keep_prob=1.0)
    optimizer = tf.train.AdamOptimizer(learning_rate=.01)
    dataset = random_dataset()
    with tf.device(device()):
      rnn_colorbot.train_one_epoch(model, optimizer, dataset)

  def testTest(self):
    model = rnn_colorbot.RNNColorbot(
        rnn_cell_sizes=[256],
        label_dimension=LABEL_DIMENSION,
        keep_prob=1.0)
    dataset = random_dataset()
    with tf.device(device()):
      rnn_colorbot.test(model, dataset)


if __name__ == "__main__":
  tf.enable_eager_execution()
  tf.test.main()
