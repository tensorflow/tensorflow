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

from tensorflow.python.keras.benchmarks import benchmark_util
from tensorflow.python.keras.benchmarks import distribution_util


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

  def compute_loss(self, targets, predictions, loss_fn, batch_size):
    """Compute average loss."""
    per_example_loss = loss_fn(targets, predictions)
    return tf.nn.compute_average_loss(
        per_example_loss, global_batch_size=batch_size)

  @tf.function(experimental_relax_shapes=True)
  def train_step(self, inputs, model, loss_fn, optimizer, batch_size):
    """Compute loss and optimize model by optimizer.

    Args:
      inputs: `tf.data`.
      model: See `model` in `train_function()` method.
      loss_fn: See `loss_fn` in `train_function()` method.
      optimizer: See `optimizer` in `train_function()` method.
      batch_size: See `batch_size` in `train_function()` method.

    Returns:
      Loss value.
    """
    train_x, train_y = inputs
    with tf.GradientTape() as tape:
      predictions = model(train_x, training=True)
      loss = self.compute_loss(train_y, predictions, loss_fn, batch_size)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss

  @tf.function(experimental_relax_shapes=True)
  def distributed_train_step(self, batch_dataset, model, loss_fn, optimizer,
                             batch_size, distribution_strategy):
    """Train step in distribution strategy setting.

    Args:
      batch_dataset: `tf.data`.
      model: See `model` in `train_function()` method.
      loss_fn: See `loss_fn` in `train_function()` method.
      optimizer: See `optimizer` in `train_function()` method.
      batch_size: See `batch_size` in `train_function()` method.
      distribution_strategy: See `distribution_strategy` in `train_function()`
        method.

    Returns:
      Sum of per_replica_losses.
    """
    per_replica_losses = distribution_strategy.run(
        self.train_step,
        args=(
            batch_dataset,
            model,
            loss_fn,
            optimizer,
            batch_size,
        ))
    return distribution_strategy.reduce(
        tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

  def train_function(self,
                     model,
                     train_dataset,
                     loss_fn,
                     optimizer,
                     epochs=2,
                     distribution_strategy=None,
                     batch_size=256):
    """Train model in custom training loop and return average

    train_step_time.

    Args:
      model: Model function to be benchmarked.
      train_dataset: `tf.data` dataset. Should return a tuple of either (inputs,
        targets) or (inputs, targets, sample_weights).
      loss_fn: `tf.keras.losses.Loss` instance.
      optimizer: `tf.keras.optimizers` instance.
      epochs: Integer. Number of epochs to train the model. If unspecified,
        `epochs` will default to 2.
      distribution_strategy: Distribution strategies. It could be
        `multi_worker_mirrored`, `one_device`, `mirrored`. If unspecified,
        `distribution_strategy` will default to 'off'. Note that, `TPU` and
        `parameter_server` are not supported yet.
      batch_size: Integer. Number of samples per gradient update. If
        unspecified, `batch_size` will default to 32.

    Returns:
      Average train_step_time.
    """
    train_step_time_list = []
    timer = timeit.default_timer

    total_loss = 0.0
    num_batches = 0
    for _ in range(epochs):
      # Iterate over the batches of the dataset.
      for batch_dataset in train_dataset:

        start_time = timer()

        if distribution_strategy is not None:
          total_loss += self.distributed_train_step(batch_dataset, model,
                                                    loss_fn, optimizer,
                                                    batch_size,
                                                    distribution_strategy)
        else:
          total_loss += self.train_step(batch_dataset, model, loss_fn,
                                        optimizer, batch_size)
        num_batches += 1

        end_time = timer()
        train_step_time_list.append(end_time - start_time)

    return np.mean(train_step_time_list)

  def measure_performance(self,
                          model,
                          dataset,
                          loss_fn,
                          optimizer,
                          batch_size=32,
                          run_iters=4,
                          epochs=10,
                          distribution_strategy=None):
    """Run models and measure the performance.

    Args:
      model_fn: Model function to be benchmarked.
      dataset: `tf.data` dataset. Should return a tuple of either (inputs,
        targets) or (inputs, targets, sample_weights).
      loss_fn: `tf.keras.losses.Loss` instance.
      optimizer: `tf.keras.optimizers` instance.
      batch_size: Integer. Number of samples per gradient update. If
        unspecified, `batch_size` will default to 32.
      run_iters: Integer. Number of iterations to run the performance
        measurement. If unspecified, `run_iters` will default to 4.
      epochs: Integer. Number of epochs to train the model. If unspecified,
        `epochs` will default to 10.
      distribution_strategy: Distribution strategies. It could be
        `multi_worker_mirrored`, `one_device`, `mirrored`. If unspecified,
        `distribution_strategy` will default to 'off'. Note that, `TPU` and
        `parameter_server` are not supported yet.

    Returns:
      Performance summary, which contains build_time, avg_epoch_time,
      wall_time, exp_per_sec, epochs, warmup_time, train_step_time.

    Raise:
      ValueError: if `dataset` is None or if `optimizer` instance is
      not provided or if `loss_fn` instance is not provided.
    """
    if distribution_strategy is not None and \
      not isinstance(dataset, tf.distribute.DistributedDataset):
      raise ValueError('tf.distribute.DistributedDataset'
                       ' required in distribution strategy.')

    if distribution_strategy is None and \
      not isinstance(dataset, tf.data.Dataset):
      raise ValueError('`tf.data` is required.')

    if not isinstance(loss_fn, tf.keras.losses.Loss):
      raise ValueError('`tf.keras.losses.Loss` instance '
                       'for loss_fn is required.')

    if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
      raise ValueError('`tf.keras.optimizers` instance '
                       'for optimizer is required.')

    avg_epoch_time_list, train_step_time_list = [], []
    wall_time_list, exp_per_sec_list, warmup_time_list = [], [], []

    total_num_examples = epochs * self.num_examples

    for _ in range(run_iters):
      timer = timeit.default_timer
      start_time = timer()
      t1 = timer()
      self.train_function(model, dataset, loss_fn, optimizer, 1,
                          distribution_strategy, batch_size)
      warmup_time = timer() - t1

      t2 = timer()
      train_step_time = self.train_function(model, dataset, loss_fn, optimizer,
                                            epochs, distribution_strategy,
                                            batch_size)
      end_time = timer()

      train_step_time_list.append(train_step_time)
      warmup_time_list.append(warmup_time)
      wall_time_list.append(end_time - start_time)
      exp_per_sec_list.append(total_num_examples / (end_time - t2))
      avg_epoch_time_list.append((end_time - t2) / epochs)

    metrics = []
    metrics.append({
        'name': 'avg_epoch_time',
        'value': np.mean(avg_epoch_time_list)
    })
    metrics.append({'name': 'exp_per_sec', 'value': np.mean(exp_per_sec_list)})
    metrics.append({'name': 'warmup_time', 'value': np.mean(warmup_time_list)})
    metrics.append({
        'name': 'train_step_time',
        'value': np.mean(train_step_time_list)
    })
    metrics.append({'name': 'epochs', 'value': epochs})

    wall_time = np.mean(wall_time_list)

    return metrics, wall_time

  def benchmark_custom_training_mnist_bs_128(self):
    """Measure performance with batch_size=128 and run_iters=5."""
    batch_size = 128
    run_iters = 5
    train_dataset = self.train_dataset.shuffle(
        buffer_size=1024).batch(batch_size)

    # Instantiate a loss function.
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE)
    # Instantiate an optimizer to train the model.
    optimizer = tf.keras.optimizers.Adam()
    model = self._build_model()

    metrics, wall_time = self.measure_performance(model, train_dataset, loss_fn,
                                                  optimizer, batch_size,
                                                  run_iters, self.epochs)
    extras = benchmark_util.get_keras_examples_metadata('conv', batch_size,
                                                        '.keras.ctl_graph')
    self.report_benchmark(
        iters=run_iters, wall_time=wall_time, metrics=metrics, extras=extras)

  def benchmark_custom_training_mnist_bs_256(self):
    """Measure performance with batch_size=256 and run_iters=5."""
    batch_size = 256
    run_iters = 5
    train_dataset = self.train_dataset.shuffle(
        buffer_size=1024).batch(batch_size)

    # Instantiate a loss function.
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE)
    # Instantiate an optimizer to train the model.
    optimizer = tf.keras.optimizers.Adam()
    model = self._build_model()

    metrics, wall_time = self.measure_performance(model, train_dataset, loss_fn,
                                                  optimizer, batch_size,
                                                  run_iters, self.epochs)
    extras = benchmark_util.get_keras_examples_metadata('conv', batch_size,
                                                        '.keras.ctl_graph')
    self.report_benchmark(
        iters=run_iters, wall_time=wall_time, metrics=metrics, extras=extras)

  def benchmark_custom_training_mnist_bs_512(self):
    """Measure performance with batch_size=512 and run_iters=10."""
    batch_size = 512
    run_iters = 5
    train_dataset = self.train_dataset.shuffle(
        buffer_size=1024).batch(batch_size)

    # Instantiate a loss function.
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE)
    # Instantiate an optimizer to train the model.
    optimizer = tf.keras.optimizers.Adam()
    model = self._build_model()

    metrics, wall_time = self.measure_performance(model, train_dataset, loss_fn,
                                                  optimizer, batch_size,
                                                  run_iters, self.epochs)
    extras = benchmark_util.get_keras_examples_metadata('conv', batch_size,
                                                        '.keras.ctl_graph')
    self.report_benchmark(
        iters=run_iters, wall_time=wall_time, metrics=metrics, extras=extras)

  def benchmark_custom_training_mnist_bs_512_gpu_2(self):
    """Measure performance with batch_size=512, run_iters=10, gpu=2 and

    distribution_strategy='mirrored'.
    """
    batch_size = 512
    run_iters = 10
    train_dataset = self.train_dataset.shuffle(
        buffer_size=1024).batch(batch_size)

    distribution_strategy = 'mirrored'

    strategy = distribution_util.get_distribution_strategy(
        distribution_strategy=distribution_strategy, num_gpus=2)

    if distribution_strategy != 'off':
      train_dataset = strategy.experimental_distribute_dataset(train_dataset)

    strategy_scope = distribution_util.get_strategy_scope(strategy)

    with strategy_scope:
      # Instantiate a loss function.
      loss_fn = tf.keras.losses.CategoricalCrossentropy(
          reduction=tf.keras.losses.Reduction.NONE)
      # Instantiate an optimizer to train the model.
      optimizer = tf.keras.optimizers.Adam()
      model = self._build_model()

    metrics, wall_time = self.measure_performance(model, train_dataset, loss_fn,
                                                  optimizer, batch_size,
                                                  run_iters, self.epochs,
                                                  strategy)
    extras = benchmark_util.get_keras_examples_metadata('conv', batch_size,
                                                        '.keras.ctl_graph')
    self.report_benchmark(
        iters=run_iters, wall_time=wall_time, metrics=metrics, extras=extras)


if __name__ == '__main__':
  tf.test.main()
