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
"""Benchmarks using custom training loop on MNIST dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import timeit
import numpy as np

import tensorflow as tf

class CustomMnistBenchmark(tf.test.Benchmark):
  """Benchmarks for custom training loop using `tf.test.Benchmark`."""

  def __init__(self):
    super(CustomMnistBenchmark, self).__init__()
    self.num_classes = 10
    self.input_shape = (28, 28, 1)
    self.epochs = 15
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_train = np.expand_dims(x_train, -1)
    y_train = tf.keras.utils.to_categorical(y_train, self.num_classes)
    self.num_examples = x_train.shape[0]
    #  Use `tf.data.Dataset` for custom training loop.
    self.train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

  def _build_model(self):
    """Model from https://keras.io/examples/vision/mnist_convnet/."""
    model = tf.keras.Sequential([
        tf.keras.Input(shape=self.input_shape),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(self.num_classes, activation='softmax'),
    ])

    return model

  def train_step(self,
                 model,
                 train_dataset,
                 loss_fn,
                 optimizer,
                 epochs=2):
    """Train model in custom training loop and return average
    train_step_time.

    Arguments:
      model: Model function to be benchmarked.
      train_dataset: `tf.data` dataset. Should return a tuple
        of either (inputs, targets) or (inputs, targets, sample_weights).
      loss_fn: `tf.keras.losses.Loss` instance.
      optimizer: `tf.keras.optimizers` instance.
      epochs: Integer. Number of epochs to train the model.
        If unspecified, `epochs` will default to 2.

    Returns:
      Average train_step_time.
    """
    train_step_time_list = []
    timer = timeit.default_timer

    for epoch in range(epochs):
      print("\nStart of epoch %d" % (epoch,))

      # Iterate over the batches of the dataset.
      for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

          # Open a GradientTape to record the operations run
          # during the forward pass, which enables autodifferentiation.
        with tf.GradientTape() as tape:
          start_time = timer()

          logits = model(x_batch_train, training=True)
          loss_value = loss_fn(y_batch_train, logits)

          grads = tape.gradient(loss_value, model.trainable_weights)

          # Run one step of gradient descent by updating
          # the value of the variables to minimize the loss.
          optimizer.apply_gradients(zip(grads, model.trainable_weights))

          end_time = timer()
          train_step_time_list.append(end_time - start_time)
        # Log every 200 batches.
        if step % 200 == 0:
          print(
              "Training loss (for one batch) at step %d: %.4f"
              % (step, float(loss_value)))

    return np.mean(train_step_time_list)

  def measure_performance(self,
                          model_fn,
                          dataset,
                          loss_fn,
                          optimizer,
                          run_iters=4,
                          epochs=2):
    """Run models and measure the performance.

    Arguments:
      model_fn: Model function to be benchmarked.
      dataset: `tf.data` dataset. Should return a tuple
        of either (inputs, targets) or (inputs, targets, sample_weights).
      loss_fn: `tf.keras.losses.Loss` instance.
      optimizer: `tf.keras.optimizers` instance.
      run_iters: Integer. Number of iterations to run the performance
        measurement. If unspecified, `run_iters` will default to 4.
      epochs: Integer. Number of epochs to train the model.
        If unspecified, `epochs` will default to 2.

    Returns:
      Performance summary, which contains build_time, avg_epoch_time,
      wall_time, exp_per_sec, epochs, warmup_time, train_step_time.

    Raise:
      ValueError: if `dataset` is None or if `optimizer` instance is
      not provided or if `loss_fn` instance is not provided.
    """
    if not isinstance(dataset, tf.data.Dataset):
      raise ValueError('`tf.data` is required.')

    if not isinstance(loss_fn, tf.keras.losses.Loss):
      raise ValueError('`tf.keras.losses.Loss` instance '
                       'for loss_fn is required.')

    if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
      raise ValueError('`tf.keras.optimizers` instance '
                       'for optimizer is required.')

    build_time_list, avg_epoch_time_list, train_step_time_list = [], [], []
    wall_time_list, exp_per_sec_list, warmup_time_list = [], [], []

    total_num_examples = epochs * self.num_examples

    for _ in range(run_iters):
      timer = timeit.default_timer
      start_time = timer()
      t0 = timer()
      model = model_fn()

      build_time = timer() - t0
      # Run one warm up epoch.
      t1 = timer()
      self.train_step(model, dataset, loss_fn, optimizer, epochs=1)
      warmup_time = timer() - t1

      t2 = timer()
      train_step_time = self.train_step(
          model, dataset, loss_fn, optimizer, epochs=epochs)
      end_time = timer()

      train_step_time_list.append(train_step_time)
      warmup_time_list.append(warmup_time)
      build_time_list.append(build_time)
      wall_time_list.append(end_time - start_time)
      exp_per_sec_list.append(total_num_examples / (end_time - t2))
      avg_epoch_time_list.append((end_time - t2) / epochs)

    metrics = []
    metrics.append({'name': 'build_time', 'value': np.mean(build_time_list)})
    metrics.append({'name': 'avg_epoch_time',
                    'value': np.mean(avg_epoch_time_list)})
    metrics.append({'name': 'exp_per_sec', 'value': np.mean(exp_per_sec_list)})
    metrics.append({'name': 'warmup_time', 'value': np.mean(warmup_time_list)})
    metrics.append({'name': 'train_step_time',
                    'value': np.mean(train_step_time_list)})
    metrics.append({'name': 'epochs', 'value': epochs})

    wall_time = np.mean(wall_time_list)

    return metrics, wall_time

  def benchmark_custom_training_mnist_bs_128(self):
    """Measure performance with batch_size=128 and run_iters=2"""
    batch_size = 128
    run_iters = 2
    train_dataset = self.train_dataset.shuffle(
        buffer_size=1024).batch(batch_size)

    # Instantiate a loss function.
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    # Instantiate an optimizer to train the model.
    optimizer = tf.keras.optimizers.Adam()

    metrics, wall_time = self.measure_performance(self._build_model,
                                                  dataset=train_dataset,
                                                  loss_fn=loss_fn,
                                                  optimizer=optimizer,
                                                  run_iters=run_iters,
                                                  epochs=self.epochs)
    self.report_benchmark(
        iters=run_iters, wall_time=wall_time, metrics=metrics)


if __name__ == '__main__':
  tf.test.main()
