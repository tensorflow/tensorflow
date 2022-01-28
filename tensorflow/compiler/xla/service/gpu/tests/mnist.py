# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""MNIST end-to-end training and inference example.

The source code here is from
https://www.tensorflow.org/xla/tutorials/jit_compile, where there is also a
walkthrough.

To execute in TFRT BEF, run with `--config=cuda
--//third_party/tensorflow/compiler/xla/service/gpu:enable_bef_executable=true`.

To dump debug output (e.g., LMHLO MLIR, TFRT MLIR, TFRT BEF), run with
`XLA_FLAGS="--xla_dump_to=/tmp/mnist"`.
"""

from absl import app

import tensorflow as tf
tf.compat.v1.enable_eager_execution()


# Size of each input image, 28 x 28 pixels
IMAGE_SIZE = 28 * 28
# Number of distinct number labels, [0..9]
NUM_CLASSES = 10
# Number of examples in each training batch (step)
TRAIN_BATCH_SIZE = 100
# Number of training steps to run
TRAIN_STEPS = 1000


def main(_):
  # Loads MNIST dataset.
  train, test = tf.keras.datasets.mnist.load_data()
  train_ds = tf.data.Dataset.from_tensor_slices(train).batch(
      TRAIN_BATCH_SIZE).repeat()

  # Casting from raw data to the required datatypes.
  def cast(images, labels):
    images = tf.cast(
        tf.reshape(images, [-1, IMAGE_SIZE]), tf.float32)
    labels = tf.cast(labels, tf.int64)
    return (images, labels)

  layer = tf.keras.layers.Dense(NUM_CLASSES)
  optimizer = tf.keras.optimizers.Adam()

  @tf.function(jit_compile=True)
  def train_mnist(images, labels):
    images, labels = cast(images, labels)

    with tf.GradientTape() as tape:
      predicted_labels = layer(images)
      loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=predicted_labels, labels=labels
      ))
    layer_variables = layer.trainable_variables
    grads = tape.gradient(loss, layer_variables)
    optimizer.apply_gradients(zip(grads, layer_variables))

  for images, labels in train_ds:
    if optimizer.iterations > TRAIN_STEPS:
      break
    train_mnist(images, labels)

  images, labels = cast(test[0], test[1])
  predicted_labels = layer(images)
  correct_prediction = tf.equal(tf.argmax(predicted_labels, 1), labels)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print("Prediction accuracy after training: %s" % accuracy)


if __name__ == "__main__":
  app.run(main)
