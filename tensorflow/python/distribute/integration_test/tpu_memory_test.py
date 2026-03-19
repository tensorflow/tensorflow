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
"""This file contains integration test for TPUStrategy in regards to memory."""

import gc

import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS
NUM_CLASS = 10


def get_dataset():

  def generate_data(_):
    image = tf.ones([500, 500, 3], dtype=tf.float32)
    label = tf.zeros([1], dtype=tf.int32)
    return image, label

  def preprocess(image, label):
    label = tf.cast(label, tf.int32)
    label = tf.one_hot(label, NUM_CLASS)
    label = tf.reshape(label, [NUM_CLASS])
    return image, label

  dataset = tf.data.Dataset.range(1)
  dataset = dataset.repeat()
  dataset = dataset.map(
      generate_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.map(
      preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.repeat()
  dataset = dataset.batch(128, drop_remainder=True)
  return dataset


class TpuMemoryTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    # Clear all cached tensors
    context._reset_context()
    # Run garbage collection to free any tensors from previous
    # runs.
    gc.collect()

    # Run a small program and copy the result to CPU.
    # This causes deferred deallocations to be flushed and new memory to be
    # allocated in a less fragmented way.
    # Turning deferred deallocations off no longer seems to work.
    assert tf.reduce_sum(tf.random.uniform(
        (1024, 128), dtype=tf.float32)).numpy() > 1.0

    self.resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu="", project=None, zone=None)

    tf.config.experimental_connect_to_cluster(self.resolver)
    tf.tpu.experimental.initialize_tpu_system(self.resolver)

  def testAutoDefragInProgramLoading(self):
    # This test covers the case when training a large model on TPU. TPU HBM
    # is not big enough to hold all TPU buffers and preserve stack for the
    # TPU program. Runtime will automatically unload unused TPU program to
    # free up space for TPU buffers. Having lots of TPU buffer may also
    # introduce fragmentation in HBM to prevent us loading a TPU program
    # properly. Runtime will automatically defrag in order to load a large
    # TPU program.

    strategy = tf.distribute.TPUStrategy(self.resolver)
    dataset = get_dataset()
    iterator = iter(
        strategy.experimental_distribute_dataset(dataset,
                                                 tf.distribute.InputOptions()))

    # Create a dummy big model that is close to HBM limit (15G):
    # Temp HBM: 11G
    # Sharded variable size: 2G
    # Unsharded variables size: 4G
    with strategy.scope():
      x = tf.keras.layers.Input(shape=(500, 500, 3), name="input")
      y = tf.keras.layers.Conv2D(
          384, (15, 15),
          strides=(2, 2),
          padding="valid",
          use_bias=False,
          kernel_initializer="he_normal",
          name="conv1")(
              x)
      y = tf.keras.layers.BatchNormalization(
          momentum=0.997, center=True, scale=True)(
              y)
      y = tf.keras.layers.Dense(
          10,
          activation="softmax",
          kernel_initializer=tf.random_normal_initializer(stddev=0.01))(
              y)
      y = tf.keras.layers.Conv2D(
          64, (9, 9),
          strides=(2, 2),
          padding="valid",
          use_bias=False,
          kernel_initializer="he_normal",
          name="conv2")(
              y)
      y = tf.keras.layers.Flatten()(y)
      y = tf.keras.layers.Dense(
          1024,
          activation="softmax",
          kernel_initializer=tf.random_normal_initializer(stddev=0.01))(
              y)
      y = tf.keras.layers.Dense(
          1024,
          activation="softmax",
          kernel_initializer=tf.random_normal_initializer(stddev=0.01))(
              y)
      y = tf.keras.layers.Dense(
          NUM_CLASS,
          activation="softmax",
          kernel_initializer=tf.random_normal_initializer(stddev=0.01))(
              y)
      model = tf.keras.Model(x, y)
      optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.1)
      loss_obj = tf.keras.losses.CategoricalCrossentropy(
          label_smoothing=0.0, reduction=tf.keras.losses.Reduction.NONE)
      model.compile(optimizer=optimizer, loss=loss_obj)

    @tf.function
    def train_step(iterator):

      def step_fn(inputs):
        images, targets = inputs
        with tf.GradientTape() as tape:
          outputs = model(images, training=True)
          loss = model.loss(targets, outputs)

        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

      # Using host training loop here to trigger weight-update-sharding. It will
      # introduce shard variable and unshard variable ops into the graph.
      # When running unshard variable op, HBM won't have enough space for
      # unsharded variables: 11G + 2G + 4G > 15G. So Runtime will have to
      # automatically unload step function to free up space for unshard
      # variable op.
      for _ in tf.range(tf.constant(20)):
        strategy.run(step_fn, args=(next(iterator),))

      # We want to load the step function again after unshard variable op.
      # However, we won't have enough space due to fragamentation:
      # 15G - 2G - 4G < 11G. So Runtime will have to automatically defrag
      # in order to load the program successfully.
      strategy.run(step_fn, args=(next(iterator),))

      # A dummy result to indicate this @tf.function has finished.
      return 1.0

    if FLAGS.tpu_use_tfrt:
      result = train_step(iterator)

      self.assertAllClose(1.0, result, atol=1e-07)
    else:
      # TPU StreamExecutor does not support auto-defrag in program loading. So
      # it will return a ResourceExhaustedError.
      with self.assertRaises(tf.errors.ResourceExhaustedError):
        _ = train_step(iterator)

  def testAutoDefragInBufferAllocation(self):
    if not FLAGS.tpu_use_tfrt:
      self.skipTest(
          "TPU StreamExecutor does not support auto-defrag in allocation.")
    with tf.device("TPU:0"):
      # DF has ~15G HBM. Following 7 buffers will consume most HBM.
      # pylint: disable=unused-variable
      buffer_2g_1 = tf.random.uniform((2, 256, 1024, 1024), dtype=tf.float32)
      buffer_2g_2 = tf.random.uniform((2, 256, 1024, 1024), dtype=tf.float32)
      buffer_2g_3 = tf.random.uniform((2, 256, 1024, 1024), dtype=tf.float32)
      buffer_2g_4 = tf.random.uniform((2, 256, 1024, 1024), dtype=tf.float32)
      buffer_2g_5 = tf.random.uniform((2, 256, 1024, 1024), dtype=tf.float32)
      buffer_2g_6 = tf.random.uniform((2, 256, 1024, 1024), dtype=tf.float32)
      buffer_2g_7 = tf.random.uniform((2, 256, 1024, 1024), dtype=tf.float32)
      #  pylint: enable=unused-variable

      # Deallocate two buffers.
      del buffer_2g_1, buffer_2g_3
      gc.collect()

      # The buffer we just deallocated doesn't provide enough contiguous region
      # for allocating 4G. This allocation will trigger auto-defrag.
      buffer_4g = tf.random.uniform((4, 256, 1024, 1024), dtype=tf.float32)

    self.assertEndsWith(buffer_4g.device, "device:TPU:0")


if __name__ == "__main__":
  tf.test.main()
