# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Run MNIST on multiple GPUs on using MirroredStrategy with eager execution.

By default, runs on all available GPUs, or CPU if no GPUs are available.

NOTE: Currently, this takes more time than when running MNIST in eager without
MirroredStrategy because of a number overheads. Therefore, this is just a
proof of concept right now and cannot be used to actually scale up training.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf


flags.DEFINE_string("tpu", None, "Name of the TPU to use.")
flags.DEFINE_integer("batch_size", 64,
                     "What should be the size of each batch?")
flags.DEFINE_integer("num_epochs", 10, "How many epochs to run?")
flags.DEFINE_float("learning_rate", 0.01, "Learning Rate")
flags.DEFINE_float("momentum", 0.5, "SGD momentum")

FLAGS = flags.FLAGS
NUM_TRAIN_IMAGES = 60000


def create_model():
  max_pool = tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding="same")
  # The model consists of a sequential chain of layers, so tf.keras.Sequential
  # (a subclass of tf.keras.Model) makes for a compact description.
  return tf.keras.Sequential([
      tf.keras.layers.Reshape(
          target_shape=[28, 28, 1],
          input_shape=(28, 28,)),
      tf.keras.layers.Conv2D(2, 5, padding="same", activation=tf.nn.relu),
      max_pool,
      tf.keras.layers.Conv2D(4, 5, padding="same", activation=tf.nn.relu),
      max_pool,
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(32, activation=tf.nn.relu),
      tf.keras.layers.Dropout(0.4),
      tf.keras.layers.Dense(10)])


def compute_loss(logits, labels):
  loss = tf.reduce_sum(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=labels))
  # Scale loss by global batch size.
  return loss * (1. / FLAGS.batch_size)


def mnist_datasets():
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  # Numpy defaults to dtype=float64; TF defaults to float32. Stick with float32.
  x_train, x_test = x_train / np.float32(255), x_test / np.float32(255)
  y_train, y_test = y_train.astype(np.int64), y_test.astype(np.int64)
  # TODO(priyag): `strategy.make_numpy_iterator` can be used directly instead of
  # converting to datasets.
  train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
  return train_dataset, test_dataset


def main(argv):
  """Run a CNN model on MNIST data to demonstrate DistributedStrategies."""
  del argv  # Unused.
  tf.disable_v2_behavior()

  cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
      tpu=FLAGS.tpu)
  strategy = tf.contrib.distribute.TPUStrategy(cluster_resolver)

  with strategy.scope():
    train_ds, test_ds = mnist_datasets()
    train_ds = train_ds.shuffle(NUM_TRAIN_IMAGES).batch(FLAGS.batch_size)
    test_ds = test_ds.batch(FLAGS.batch_size)

    model = create_model()
    optimizer = tf.keras.optimizers.SGD(FLAGS.learning_rate, FLAGS.momentum)
    training_loss = tf.keras.metrics.Mean("training_loss", dtype=tf.float32)
    training_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        "training_accuracy", dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean("test_loss", dtype=tf.float32)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        "test_accuracy", dtype=tf.float32)

    def train_step(inputs):  # pylint: disable=missing-docstring
      images, labels = inputs
      with tf.GradientTape() as tape:
        logits = model(images, training=True)
        loss = compute_loss(logits, labels)
      grads = tape.gradient(loss, model.variables)
      update_vars = optimizer.apply_gradients(zip(grads, model.variables))
      update_loss = training_loss.update_state(loss)
      update_accuracy = training_accuracy.update_state(labels, logits)

      with tf.control_dependencies([update_vars, update_loss, update_accuracy]):
        return tf.identity(loss)

    def test_step(inputs):
      images, labels = inputs
      logits = model(images, training=False)
      loss = compute_loss(logits, labels)
      update_loss = test_loss.update_state(loss)
      update_accuracy = test_accuracy.update_state(labels, logits)

      with tf.control_dependencies([update_loss, update_accuracy]):
        return tf.identity(loss)

    train_iterator = strategy.make_dataset_iterator(train_ds)
    test_iterator = strategy.make_dataset_iterator(test_ds)

    dist_train = strategy.unwrap(
        strategy.experimental_run(train_step, train_iterator))
    dist_test = strategy.unwrap(
        strategy.experimental_run(test_step, test_iterator))

    training_loss_result = training_loss.result()
    training_accuracy_result = training_accuracy.result()
    test_loss_result = test_loss.result()
    test_accuracy_result = test_accuracy.result()

    tf.contrib.distribute.initialize_tpu_system(cluster_resolver)

    train_iterator_init = train_iterator.initialize()
    test_iterator_init = test_iterator.initialize()

    all_variables = (
        tf.global_variables() +
        training_loss.variables +
        training_accuracy.variables +
        test_loss.variables +
        test_accuracy.variables)

    with tf.Session(cluster_resolver.master()) as session:
      session.run([v.initializer for v in all_variables])

      for epoch in range(0, FLAGS.num_epochs):
        # Train
        print("Starting epoch {}".format(epoch))
        session.run(train_iterator_init)
        while True:
          try:
            session.run(dist_train)
          except tf.errors.OutOfRangeError:
            break
        print("Training loss: {:0.4f}, accuracy: {:0.2f}%".format(
            session.run(training_loss_result),
            session.run(training_accuracy_result) * 100))
        training_loss.reset_states()
        training_accuracy.reset_states()

        # Test
        session.run(test_iterator_init)
        while True:
          try:
            session.run(dist_test)
          except tf.errors.OutOfRangeError:
            break
        print("Test loss: {:0.4f}, accuracy: {:0.2f}%".format(
            session.run(test_loss_result),
            session.run(test_accuracy_result) * 100))
        test_loss.reset_states()
        test_accuracy.reset_states()


if __name__ == "__main__":
  flags.mark_flag_as_required("tpu")
  app.run(main)
