# coding=utf-8
# Copyright 2025 TF.Text Authors.
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

"""Benchmarking utils for TF.Text ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

import tensorflow_datasets as tfds

from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow_text.python import ops as text_ops
# [internal] import xprof_session
from tensorflow.python.util import tf_inspect


class OpsBaseBenchmark(tf.test.Benchmark):
  """Base class for op benchmarks."""

  def __init__(self):
    super(OpsBaseBenchmark, self).__init__()
    self.input_data = None
    self.batch_size = None
    self.use_tf_function = False

  def _get_method_name(self):
    """Returns the calling method name."""

    # Find the caller method (outermost Benchmark class)
    stack = tf_inspect.stack()
    name = None
    for frame in stack[::-1]:
      f_locals = frame[0].f_locals
      f_self = f_locals.get('self', None)
      if isinstance(f_self, tf.test.Benchmark):
        name = frame[3]
        break
    if name is None:
      raise ValueError('Unable to determine the method name.')

    return name

  def load_input_data(self, batch_size):
    """Loads the IMDB dataset and sets up the input data to run the ops on."""

    self.batch_size = batch_size
    data = tfds.load(
        'imdb_reviews/plain_text', split=tfds.Split.TRAIN).batch(batch_size)
    # The input data has shape [batch_size, data] and the op is run multiple
    # iterations over the first batch
    self.batch_number = 1

    if context.executing_eagerly():
      self.iterator = data.as_numpy_iterator()
      self.input_data = [x['text'] for x in self.iterator][0]
    else:
      self.iterator = dataset_ops.make_initializable_iterator(data)
      self.input_data = self.iterator.get_next()['text']

  def run_and_report(self,
                     fn,
                     iters,
                     burn_iters,
                     xprof_enabled=False,
                     benchmark_name=None,
                     **kwargs):
    """Runs the benchmark and reports results.

    Args:
      fn: Function to be benchmarked.
      iters: Number of iterations to run the benchmark.
      burn_iters: Number of warm-up iterations to run to reach a stable state.
      xprof_enabled: Enables xprof traces.
      benchmark_name: Overwrites the default name.
      **kwargs: Kwargs to the benchmarked function.

    Returns:
      Dict which contains the wall time report for the runned op.
    """
    name = benchmark_name or self._get_method_name()

    if context.executing_eagerly():
      self._run_and_report_eagerly(fn, iters, burn_iters, name, xprof_enabled,
                                   **kwargs)
    else:
      self._run_and_report_graphmode(fn, iters, burn_iters, name, xprof_enabled,
                                     **kwargs)

  def _convert_to_ragged_inputs(self, inputs):
    """Transforms the text batch inputs to a ragged shape."""
    if isinstance(self.input_data, ragged_tensor.RaggedTensor):
      return inputs

    inputs = text_ops.WhitespaceTokenizer().tokenize(inputs)
    return inputs

  def run_and_report_ragged_vs_dense(self,
                                     fn,
                                     iters,
                                     burn_iters,
                                     xprof_enabled=False,
                                     **kwargs):
    """Runs the op on ragged inputs and on its dense counterpart for comparison."""
    ragged_data = self._convert_to_ragged_inputs(self.input_data)

    self.input_data = ragged_data
    self.run_and_report(
        fn,
        iters,
        burn_iters,
        xprof_enabled,
        benchmark_name=self._get_method_name() + '_ragged',
        **kwargs)

    self.input_data = ragged_data.to_tensor()
    self.run_and_report(
        fn,
        iters,
        burn_iters,
        xprof_enabled,
        benchmark_name=self._get_method_name() + '_dense',
        **kwargs)

    self.load_input_data(self.batch_size)

  def _run_and_report_eagerly(self,
                              fn,
                              iters,
                              burn_iters,
                              benchmark_name,
                              xprof_enabled=False,
                              **kwargs):
    """Runs and reports benchmarks eagerly."""
    if self.input_data is None:
      raise ValueError(
          'Input data is missing for {} benchmark'.format(benchmark_name))

    @def_function.function
    def tf_func():
      fn(self.input_data, **kwargs)

    def func():
      fn(self.input_data, **kwargs)

    op = tf_func if self.use_tf_function else func

    for _ in range(burn_iters):
      op()

    def run_benchmark():
      total_time = 0
      for _ in range(iters):
        start = time.time()
        op()
        total_time += time.time() - start

      return total_time

    total_time = run_benchmark()
    mean_time = total_time / iters
    benchmark_name = benchmark_name + ('_function'
                                       if self.use_tf_function else '_eager')
    metrics = []
    extras = {'sec_per_batch': total_time / iters}
    if hasattr(self, 'batch_number'):
      extras.update({'batches_per_sec': self.batch_number / mean_time})
      metrics.append({
          'name': 'batches_per_sec',
          'value': self.batch_number / mean_time
      })

    if xprof_enabled:
      extras.update(self._run_with_xprof(run_benchmark))

    self.report_benchmark(
        wall_time=mean_time,
        name=benchmark_name,
        extras=extras,
        metrics=metrics)

  def _run_with_xprof(self, benchmark_fn):
    output = {}
    xprof = xprof_session.XprofSession()
    xprof.start_session(enable_python_tracer=True)
    _ = benchmark_fn()
    output['xprof_link'] = xprof.end_session_and_get_url()

    return output

  def _run_and_report_graphmode(self, fn, iters, burn_iters, benchmark_name,
                                xprof_enabled, **kwargs):
    """Runs and reports benchmarks in graph mode."""
    if self.input_data is None:
      raise ValueError(
          'Input data is missing for {} benchmark'.format(benchmark_name))

    # Uses the benchmark config to disable the static graph optimizations
    with session.Session(config=tf.test.benchmark_config()) as sess:
      if hasattr(self, 'iterator'):
        sess.run(self.iterator.initializer)

      sess.run(lookup_ops.tables_initializer())
      sess.run(variables_lib.global_variables_initializer())

      inputs = sess.run(self.input_data)
      placeholder = array_ops.placeholder(dtypes.string,
                                          tensor_shape.TensorShape({None}))
      op_feed_dict = {placeholder: inputs}
      benchmark_op = fn(placeholder, **kwargs)

      def run_benchmark():
        for _ in range(burn_iters):
          sess.run(benchmark_op, op_feed_dict)
        total_time = 0
        for _ in range(iters):
          start_time = time.time()
          sess.run(benchmark_op, op_feed_dict)
          total_time += time.time() - start_time

        return total_time

      total_time = run_benchmark()
      mean_time = total_time / iters
      extras = {'sec_per_batch': mean_time}

      metrics = []
      if hasattr(self, 'batch_number'):
        extras.update({'batches_per_sec': self.batch_number / mean_time})
        metrics.append({
            'name': 'batches_per_sec',
            'value': self.batch_number / mean_time
        })

      if xprof_enabled:
        extras.update(self._run_with_xprof(run_benchmark))

      self.report_benchmark(
          wall_time=mean_time,
          name=benchmark_name + '_graph',
          extras=extras,
          metrics=metrics)
