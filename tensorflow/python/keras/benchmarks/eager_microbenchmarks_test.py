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

import tensorflow as tf

from tensorflow.python.eager import context
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

  def benchmark_layers_call_overhead(self):

    class OnlyOverheadLayer(tf.keras.layers.Layer):

      def call(self, x):
        return x

    layer = OnlyOverheadLayer()
    x = tf.convert_to_tensor([[1.]])

    def fn():
      layer(x)  # pylint: disable=not-callable

    self._run(fn, 10000)

  def benchmark_model_predict_tensorlike_overhead(self):

    class OnlyOverheadLayer(tf.keras.layers.Layer):

      def call(self, x):
        return x

    model = tf.keras.Sequential([OnlyOverheadLayer()])
    x = tf.convert_to_tensor([[1.]])

    def fn():
      model.predict(x)

    self._run(fn, 20)

  # Naming convention: benchmark_layers_{module_name}_{class}_overhead.
  def benchmark_layers_advanced_activations_leaky_relu_overhead(self):

    layer = tf.keras.layers.LeakyReLU()
    x = tf.ones((1, 1))

    def fn():
      layer(x)

    self._run(fn, 10000)

  def benchmark_layers_advanced_activations_prelu_overhead(self):

    layer = tf.keras.layers.PReLU()
    x = tf.ones((1, 1))

    def fn():
      layer(x)

    self._run(fn, 10000)

  def benchmark_layers_advanced_activations_elu_overhead(self):

    layer = tf.keras.layers.ELU()
    x = tf.ones((1, 1))

    def fn():
      layer(x)

    self._run(fn, 10000)

  def benchmark_layers_advanced_activations_thresholded_relu_overhead(self):

    layer = tf.keras.layers.ThresholdedReLU()
    x = tf.ones((1, 1))

    def fn():
      layer(x)

    self._run(fn, 10000)

  def benchmark_layers_advanced_activations_softmax_overhead(self):

    layer = tf.keras.layers.Softmax()
    x = tf.ones((1, 1))

    def fn():
      layer(x)

    self._run(fn, 10000)

  def benchmark_layers_advanced_activations_relu_overhead(self):

    layer = tf.keras.layers.ReLU()
    x = tf.ones((1, 1))

    def fn():
      layer(x)

    self._run(fn, 10000)

  def benchmark_layers_core_masking_overhead(self):

    layer = tf.keras.layers.Masking()
    x = tf.ones((1, 1))

    def fn():
      layer(x)

    self._run(fn, 10000)

  def benchmark_layers_core_dropout_overhead(self):

    layer = tf.keras.layers.Dropout(0.5)
    x = tf.ones((1, 1))

    def fn():
      layer(x, training=True)

    self._run(fn, 10000)

  def benchmark_layers_core_flatten_overhead(self):

    layer = tf.keras.layers.Flatten()
    x = tf.convert_to_tensor([[[1.]]])

    def fn():
      layer(x)

    self._run(fn, 10000)

  def benchmark_layers_core_dense_overhead(self):

    layer = tf.keras.layers.Dense(1)
    x = tf.convert_to_tensor([[1.]])

    def fn():
      layer(x)

    self._run(fn, 10000)

  def benchmark_layers_convolutional_conv1d_overhead(self):

    layer = tf.keras.layers.Conv1D(1, (1,))
    x = tf.ones((1, 1, 1))

    def fn():
      layer(x)

    self._run(fn, 10000)

  def benchmark_layers_convolutional_conv2d_overhead(self):

    layer = tf.keras.layers.Conv2D(1, (1, 1))
    x = tf.ones((1, 1, 1, 1))

    def fn():
      layer(x)

    self._run(fn, 10000)

  def benchmark_layers_convolutional_conv3d_overhead(self):

    layer = tf.keras.layers.Conv3D(1, (1, 1, 1))
    x = tf.ones((1, 1, 1, 1, 1))

    def fn():
      layer(x)

    self._run(fn, 10000)

  def benchmark_layers_embeddings_embedding_overhead(self):

    layer = tf.keras.layers.Embedding(1, 1)
    x = tf.zeros((1, 1), dtype="int32")

    def fn():
      layer(x)

    self._run(fn, 10000)

  def benchmark_layers_batch_norm_fused_inf(self):

    layer = tf.keras.layers.BatchNormalization(fused=True)
    x = tf.ones((1, 1, 1, 1))

    def fn():
      layer(x)

    self._run(fn, 10000)

  def benchmark_layers_batch_norm_fused_train(self):

    layer = tf.keras.layers.BatchNormalization(fused=True)
    x = tf.ones((1, 1, 1, 1))

    def fn():
      layer(x, training=True)

    self._run(fn, 10000)

  def benchmark_layers_batch_norm_nonfused_inf(self):

    layer = tf.keras.layers.BatchNormalization(fused=False)
    x = tf.ones((1, 1, 1, 1))

    def fn():
      layer(x)

    self._run(fn, 10000)

  def benchmark_layers_batch_norm_nonfused_train(self):

    layer = tf.keras.layers.BatchNormalization(fused=False)
    x = tf.ones((1, 1, 1, 1))

    def fn():
      layer(x, training=True)

    self._run(fn, 10000)

  def benchmark_layers_normalization_layer_normalization_overhead(self):

    layer = tf.keras.layers.LayerNormalization()
    x = tf.ones((1, 1))

    def fn():
      layer(x, training=True)

    self._run(fn, 10000)


if __name__ == "__main__":
  assert tf.executing_eagerly()
  test.main()
