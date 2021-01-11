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
"""Common utils for benchmark."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import timeit
import numpy as np

import tensorflow as tf

from tensorflow.python.keras.benchmarks import distribution_util


def get_benchmark_name(name):
  """Split the suffix of the benchmark name.

  For example, for the name = 'benchmark_layer_call__Conv2D_small_shape',
  the return value is ['Conv2D', 'small', 'shape'].

  This is to generate the metadata of the benchmark test.

  Args:
    name: A string, the benchmark name.

  Returns:
    A list of strings of the suffix in the benchmark name.
  """
  if '__' not in name or '_' not in name:
    raise ValueError('The format of the benchmark name is wrong.')
  return name.split('__')[-1].split('_')


def generate_benchmark_params_cpu_gpu(*params_list):
  """Extend the benchmark names with CPU and GPU suffix.

  Args:
    *params_list: A list of tuples represents the benchmark parameters.

  Returns:
    A list of strings with the benchmark name extended with CPU and GPU suffix.
  """
  benchmark_params = []
  for params in params_list:
    benchmark_params.extend([
        ((param[0] + '_CPU',) + param[1:]) for param in params
    ])
    benchmark_params.extend([
        ((param[0] + '_GPU',) + param[1:]) for param in params
    ])
  return benchmark_params


def get_keras_examples_metadata(keras_model,
                                batch_size,
                                impl='.keras.cfit_graph'):
  return {
      'model_name': 'keras_examples',
      'implementation': keras_model + impl,
      'parameters': 'bs_' + str(batch_size),
  }


class TimerCallBack(tf.keras.callbacks.Callback):
  """Callback for logging time in each epoch or batch."""

  def __init__(self):
    self.times = []
    self.timer = timeit.default_timer
    self.startup_time = timeit.default_timer()
    self.recorded_startup = False

  def on_epoch_begin(self, e, logs):
    self.epoch_start_time = self.timer()

  def on_epoch_end(self, e, logs):
    self.times.append(self.timer() - self.epoch_start_time)

  def on_batch_end(self, e, logs):
    if not self.recorded_startup:
      self.startup_time = self.timer() - self.startup_time
      self.recorded_startup = True


def measure_performance(model_fn,
                        x=None,
                        y=None,
                        epochs=2,
                        batch_size=32,
                        run_iters=4,
                        optimizer=None,
                        loss=None,
                        metrics=None,
                        verbose=0,
                        num_gpus=0,
                        distribution_strategy='off'):
  """Run models and measure the performance.

  Args:
    model_fn: Model function to be benchmarked.
    x: Input data. See `x` in the `fit()` method of `keras.Model`.
    y: Target data. See `y` in the `fit()` method of `keras.Model`.
    epochs: Integer. Number of epochs to train the model.
      If unspecified, `epochs` will default to 2.
    batch_size: Integer. Number of samples per gradient update. If unspecified,
      `batch_size` will default to 32.
    run_iters: Integer. Number of iterations to run the performance measurement.
      If unspecified, `run_iters` will default to 4.
    optimizer: String (name of optimizer) or optimizer instance. See
      `tf.keras.optimizers`.
    loss: String (name of objective function), objective function or
      `tf.keras.losses.Loss` instance. See `tf.keras.losses`.
    metrics: Lists of metrics to be evaluated by the model during training. See
      `metrics` in the `compile()` method of  `keras.Model`.
    verbose: 0, 1, 2. Verbosity mode. See `verbose` in the `fit()` method of
      `keras.Model`. If unspecified, `verbose` will default to 0.
    num_gpus: Number of GPUs to run the model.
    distribution_strategy: Distribution strategies. It could be
      `multi_worker_mirrored`, `one_device`, `mirrored`. If unspecified,
      `distribution_strategy` will default to 'off'. Note that, `TPU`
      and `parameter_server` are not supported yet.

  Returns:
    Performance summary, which contains build_time, compile_time,
    startup_time, avg_epoch_time, wall_time, exp_per_sec, epochs,
    distribution_strategy.

  Raise:
    ValueError: If `x` is none or if `optimizer` is not provided or
    if `loss` is not provided or if `num_gpus` is negative.
  """
  if 'x' is None:
    raise ValueError('Input data is required.')
  if 'optimizer' is None:
    raise ValueError('Optimizer is required.')
  if 'loss' is None:
    raise ValueError('Loss function is required.')
  if num_gpus < 0:
    raise ValueError('`num_gpus` cannot be negative')

  # TODO(xingyulong): we will add tfds support later and
  #  get the `num_examples` from info.
  num_examples = x.shape[0]

  build_time_list, compile_time_list, startup_time_list = [], [], []
  avg_epoch_time_list, wall_time_list, exp_per_sec_list = [], [], []
  total_num_examples = epochs * num_examples

  strategy = distribution_util.get_distribution_strategy(
      distribution_strategy=distribution_strategy, num_gpus=num_gpus)

  for _ in range(run_iters):
    timer = timeit.default_timer
    start_time = timer()
    # Init the distribution strategy scope for each iteration.
    strategy_scope = distribution_util.get_strategy_scope(strategy)
    with strategy_scope:
      t0 = timer()
      model = model_fn()
      build_time = timer() - t0

      t1 = timer()
      model.compile(
          optimizer=optimizer,
          loss=loss,
          metrics=metrics,
      )
      compile_time = timer() - t1
    # Run one warm up epoch.
    model.fit(x=x, y=y, batch_size=batch_size, epochs=1)
    cbk = TimerCallBack()
    t2 = timer()
    model.fit(
        x=x,
        y=y,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[cbk],
        verbose=verbose)
    end_time = timer()

    build_time_list.append(build_time)
    compile_time_list.append(compile_time)
    startup_time_list.append(cbk.startup_time)
    avg_epoch_time_list.append(np.mean(cbk.times))
    wall_time_list.append(end_time - start_time)
    exp_per_sec_list.append(total_num_examples / (end_time - t2))

  metrics = []
  metrics.append({'name': 'build_time', 'value': np.mean(build_time_list)})
  metrics.append({'name': 'compile_time', 'value': np.mean(compile_time_list)})
  metrics.append({'name': 'startup_time', 'value': np.mean(startup_time_list)})
  metrics.append({
      'name': 'avg_epoch_time',
      'value': np.mean(avg_epoch_time_list)
  })
  metrics.append({'name': 'exp_per_sec', 'value': np.mean(exp_per_sec_list)})
  metrics.append({'name': 'epochs', 'value': epochs})

  wall_time = np.mean(wall_time_list)
  extras = {
      'distribution_strategy': distribution_strategy,
      'num_gpus': num_gpus
  }

  return metrics, wall_time, extras
