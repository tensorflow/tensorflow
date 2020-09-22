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
# Lint as: python3
r"""Micro benchmark.

bazel run -c opt --config=cuda \
  //third_party/tensorflow/python/ops/numpy_ops/benchmarks:micro_benchmarks -- \
  --number=100 --repeat=100 \
  --benchmarks=.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import time

from absl import flags
from absl import logging

import numpy as np  # pylint: disable=unused-import
import tensorflow.compat.v2 as tf

from tensorflow.python.ops import numpy_ops as tfnp   # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops.numpy_ops.integration_test.benchmarks import numpy_mlp
from tensorflow.python.ops.numpy_ops.integration_test.benchmarks import tf_numpy_mlp

FLAGS = flags.FLAGS

flags.DEFINE_integer('repeat', 100, '#Measurements per benchmark.')
flags.DEFINE_integer('number', 100, '#Runs per a measure.')


class MicroBenchmarks(tf.test.Benchmark):
  """Main micro benchmark class."""

  def _benchmark_and_report(
      self,
      name,
      fn,
      repeat=None,
      number=None):
    """Run fn repeat * number times, report time, and return fastest time."""
    # Can't make these default above since the flags may not have been parsed
    # at module import time.
    repeat = repeat or int(FLAGS.repeat)
    number = number or int(FLAGS.number)

    # Warmup
    fn()

    times = []
    for _ in range(repeat):
      gc.disable()
      start = time.time()
      for _ in range(number):
        fn()
      times.append(time.time() - start)
      gc.enable()
      gc.collect()

    # Regular benchmark to report numbers.
    fastest_time_us = min(times) * 1e6 / number
    total_time = sum(times)
    self.report_benchmark(name=name,
                          wall_time=total_time,
                          extras={'fastest_time_us': fastest_time_us})

    return fastest_time_us

  def benchmark_tf_np_mlp_inference_batch_1_cpu(self):
    with tf.device('/CPU:0'):
      model = tf_numpy_mlp.MLP()
      x = tfnp.ones(shape=(1, 10)).astype(np.float32)
      self._benchmark_and_report(self._get_name(), lambda: model.inference(x))

  def benchmark_tf_np_tf_function_mlp_inference_batch_1_cpu(self):
    with tf.device('/CPU:0'):
      model = tf_numpy_mlp.MLP()
      x = tfnp.ones(shape=(1, 10)).astype(np.float32)
      self._benchmark_and_report(
          self._get_name(), tf.function(lambda: model.inference(x)))

  def benchmark_numpy_mlp_inference_batch_1_cpu(self):
    model = numpy_mlp.MLP()
    x = np.random.uniform(size=(1, 10)).astype(np.float32, copy=False)
    self._benchmark_and_report(self._get_name(), lambda: model.inference(x))

  def _benchmark_np_and_tf_np(self, name, op, args, repeat=None):  # pylint: disable=redefined-builtin
    fn = getattr(np, op)
    assert fn is not None

    np_time = self._benchmark_and_report(
        '{}_numpy'.format(name), lambda: fn(*args), repeat=repeat)

    fn = getattr(tfnp, op)
    assert fn is not None

    with tf.device('CPU:0'):
      tf_time = self._benchmark_and_report(
          '{}_tfnp_cpu'.format(name), lambda: fn(*args), repeat=repeat)

    return np_time, tf_time

  def _print_times(self, op, sizes, times):
    # For easy reporting.
    print('For np.{}:'.format(op))
    print('{:<15}  {:>11}  {:>11}'.format('Size', 'NP time', 'TF NP Time'))
    for size, (np_time, tf_time) in zip(sizes, times):
      print('{:<15} {:>10.5}us {:>10.5}us'.format(
          str(size), np_time, tf_time))
    print()

  def _benchmark_np_and_tf_np_unary(self, op):
    sizes = [(100,), (10000,), (1000000,)]
    repeats = [FLAGS.repeat] * 2 + [10]
    times = []
    for size, repeat in zip(sizes, repeats):
      x = np.random.uniform(size=size).astype(np.float32, copy=False)
      name = '{}_{}'.format(self._get_name(), size)
      times.append(self._benchmark_np_and_tf_np(name, op, (x,), repeat))
    self._print_times(op, sizes, times)

  def benchmark_count_nonzero(self):
    self._benchmark_np_and_tf_np_unary('count_nonzero')

  def benchmark_log(self):
    self._benchmark_np_and_tf_np_unary('log')

  def benchmark_exp(self):
    self._benchmark_np_and_tf_np_unary('exp')

  def benchmark_tanh(self):
    self._benchmark_np_and_tf_np_unary('tanh')

  def benchmark_matmul(self):
    sizes = [(2, 2), (10, 10), (100, 100), (200, 200), (1000, 1000)]
    # Override repeat flag since this can be very slow.
    repeats = [FLAGS.repeat] * 3 + [50, 10]
    times = []
    for size, repeat in zip(sizes, repeats):
      x = np.random.uniform(size=size).astype(np.float32, copy=False)
      name = '{}_{}'.format(self._get_name(), size)
      times.append(
          self._benchmark_np_and_tf_np(name, 'matmul', (x, x), repeat=repeat))

    self._print_times('matmul', sizes, times)


if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.enable_v2_behavior()
  tf.test.main()
