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
"""Microbenchmarks for Keras components in eager mode."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.layers import convolutional as conv_layers
from tensorflow.python.keras.layers import core as core_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.util import tf_inspect


def _run_benchmark(func, num_iters, execution_mode=None):
  ctx = context.context()
  with context.execution_mode(execution_mode):
    # call func to warm up
    func()
    if execution_mode == context.ASYNC:
      ctx.executor.wait()
    start = time.time()
    for _ in range(num_iters):
      func()
    if execution_mode == context.ASYNC:
      ctx.executor.wait()
    end = time.time()

    return end - start


class MicroBenchmarksBase(test.Benchmark):
  """Run and report benchmark results."""

  def run_report(self, run_benchmark, func, num_iters, execution_mode=None):
    """Run and report benchmark results."""
    total_time = run_benchmark(func, num_iters, execution_mode)
    mean_us = total_time * 1e6 / num_iters
    extras = {
        "examples_per_sec": float("{0:.3f}".format(num_iters / total_time)),
        "us_per_example": float("{0:.3f}".format(total_time * 1e6 / num_iters))
    }
    benchmark_name = self._get_benchmark_name()
    self.report_benchmark(
        iters=num_iters, wall_time=mean_us, extras=extras, name=benchmark_name)

  def _get_benchmark_name(self):
    """Mostly copied from benchmark.py _get_name()."""
    stack = tf_inspect.stack()
    name = None
    for frame in stack[::-1]:
      f_locals = frame[0].f_locals
      f_self = f_locals.get("self", None)
      if isinstance(f_self, test.Benchmark):
        name = frame[3]  # Get the method name
        # This is a hack to get around the fact that some methods might have a
        # disable_tfrt decorator around them. In that case a function called
        # 'decorated' wraps the real called function underneath and so we
        # peek one deeper into the stack to get the real name.
        if name == "decorated":
          continue
        else:
          break
    if name is None:
      raise ValueError("Unable to determine calling Benchmark function.")
    if context.is_tfrt_enabled():
      name = name + "_tfrt"
    return name

  def _run(self, func, num_iters, execution_mode=None):
    self.run_report(_run_benchmark, func, num_iters, execution_mode)

  def benchmark_tf_keras_layer_call_overhead(self):

    class OnlyOverheadLayer(base_layer.Layer):

      def call(self, x):
        return x

    layer = OnlyOverheadLayer()
    x = ops.convert_to_tensor([[1.]])

    def fn():
      layer(x)

    self._run(fn, 10000)

  def benchmark_tf_keras_dense_overhead(self):

    layer = core_layers.Dense(1)
    x = ops.convert_to_tensor([[1.]])

    def fn():
      layer(x)

    self._run(fn, 10000)

  def benchmark_tf_keras_flatten_overhead(self):

    layer = core_layers.Flatten()
    x = ops.convert_to_tensor([[[1.]]])

    def fn():
      layer(x)

    self._run(fn, 10000)

  def benchmark_tf_keras_conv1d_overhead(self):

    layer = conv_layers.Conv1D(1, (1,))
    x = array_ops.ones((1, 1, 1))

    def fn():
      layer(x)

    self._run(fn, 10000)

  def benchmark_tf_keras_conv2d_overhead(self):

    layer = conv_layers.Conv2D(1, (1, 1))
    x = array_ops.ones((1, 1, 1, 1))

    def fn():
      layer(x)

    self._run(fn, 10000)

  def benchmark_tf_keras_conv3d_overhead(self):

    layer = conv_layers.Conv3D(1, (1, 1, 1))
    x = array_ops.ones((1, 1, 1, 1, 1))

    def fn():
      layer(x)

    self._run(fn, 10000)


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
