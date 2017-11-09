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

import numpy as np
import tensorflow as tf
from tensorflow.contrib.eager.python.examples.mnist import mnist


def data_format():
  return "channels_first" if tf.test.is_gpu_available() else "channels_last"


class MNISTGraphTest(tf.test.TestCase):

  def testTrainGraph(self):
    # The MNISTModel class can be executed eagerly (as in mnist.py and
    # mnist_test.py) and also be used to construct a TensorFlow graph, which is
    # then trained in a session.
    with tf.Graph().as_default():
      # Generate some random data.
      batch_size = 64
      images = np.random.randn(batch_size, 784).astype(np.float32)
      digits = np.random.randint(low=0, high=10, size=batch_size)
      labels = np.zeros((batch_size, 10))
      labels[np.arange(batch_size), digits] = 1.

      # Create a model, optimizer, and dataset as would be done
      # for eager execution as well.
      model = mnist.MNISTModel(data_format())
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
      dataset = tf.data.Dataset.from_tensors((images, labels))

      # Define the loss tensor (as opposed to a loss function when
      # using eager execution).
      (images, labels) = dataset.make_one_shot_iterator().get_next()
      predictions = model(images, training=True)
      loss = mnist.loss(predictions, labels)

      train_op = optimizer.minimize(loss)
      init = tf.global_variables_initializer()
      with tf.Session() as sess:
        # Variables have to be initialized in the session.
        sess.run(init)
        # Train using the optimizer.
        sess.run(train_op)


if __name__ == "__main__":
  tf.test.main()
