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

import tensorflow.contrib.eager as tfe
from tensorflow.contrib.eager.python.examples.mnist import mnist


def device():
  return "/device:GPU:0" if tfe.num_gpus() else "/device:CPU:0"


def data_format():
  return "channels_first" if tfe.num_gpus() else "channels_last"


def random_dataset():
  batch_size = 64
  images = tf.random_normal([batch_size, 784])
  digits = tf.random_uniform([batch_size], minval=0, maxval=10, dtype=tf.int32)
  labels = tf.one_hot(digits, 10)
  return tf.data.Dataset.from_tensors((images, labels))


class MNISTTest(tf.test.TestCase):

  def testTrainOneEpoch(self):
    model = mnist.MNISTModel(data_format())
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    dataset = random_dataset()
    with tf.device(device()):
      tf.train.get_or_create_global_step()
      mnist.train_one_epoch(model, optimizer, dataset)

  def testTest(self):
    model = mnist.MNISTModel(data_format())
    dataset = random_dataset()
    with tf.device(device()):
      tf.train.get_or_create_global_step()
      mnist.test(model, dataset)


if __name__ == "__main__":
  tfe.enable_eager_execution()
  tf.test.main()
