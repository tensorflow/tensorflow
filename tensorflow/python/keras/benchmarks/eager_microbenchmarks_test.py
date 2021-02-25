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
import six

import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.eager.context import get_executor
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.platform import benchmark


def _run_benchmark(func, num_iters, execution_mode=None):
  with context.execution_mode(execution_mode):
    # call func to warm up
    func()
    if execution_mode == context.ASYNC:
      get_executor().wait()
    start = time.time()
    for _ in range(num_iters):
      func()
    if execution_mode == context.ASYNC:
      get_executor().wait()
    end = time.time()

    return end - start


class MicroBenchmarksBase(tf.test.Benchmark):
  """Run and report benchmark results."""

  def run_report(self, run_benchmark, func, num_iters, execution_mode=None):
    """Run and report benchmark results."""
    total_time = run_benchmark(func, num_iters, execution_mode)
    mean_us = total_time * 1e6 / num_iters
    metrics = [{
        "name": "exp_per_sec",
        "value": float("{0:.3f}".format(num_iters / total_time))
    }, {
        "name": "us_per_exp",
        "value": float("{0:.3f}".format(total_time * 1e6 / num_iters))
    }]
    benchmark_name = self._get_benchmark_name()
    self.report_benchmark(
        iters=num_iters,
        wall_time=mean_us,
        metrics=metrics,
        name=benchmark_name)

  def _get_benchmark_name(self):
    """Mostly copied from benchmark.py _get_name()."""
    stack = tf_inspect.stack()
    name = None
    for frame in stack[::-1]:
      f_locals = frame[0].f_locals
      f_self = f_locals.get("self", None)
      if isinstance(f_self, tf.test.Benchmark):
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

  def benchmark_op_layer_call_overhead(self):
    model_input = tf.keras.Input(shape=(1,))
    model_output = model_input
    x = tf.convert_to_tensor([[1.1]])

    for _ in range(20):
      model_output = tf.multiply(model_output, x)
    model = tf.keras.Model(inputs=model_input, outputs=model_output)

    def fn():
      model(x)  # pylint: disable=not-callable

    fn()
    self._run(fn, 100)

  def benchmark_model_predict_tensorlike_overhead(self):

    class OnlyOverheadLayer(tf.keras.layers.Layer):

      def call(self, x):
        return x

    model = tf.keras.Sequential([OnlyOverheadLayer()])
    x = tf.convert_to_tensor([[1.]])

    def fn():
      model.predict(x)

    self._run(fn, 20)

  def benchmark_layers_embeddings_embedding_overhead(self):

    layer = tf.keras.layers.Embedding(1, 1)
    x = tf.zeros((1, 1), dtype="int32")

    def fn():
      layer(x)

    self._run(fn, 10000)


class KerasLayerCallOverheadBenchmarks(
    six.with_metaclass(benchmark.ParameterizedBenchmark, MicroBenchmarksBase)):

  # The set of layers for benchmarking. To add benchmarks for new layers,
  # please add the parameter configs to "_benchmark_paramters".

  # The parameter of each layer benchmark is a tuple contains:
  # 1) The benchmark name with convention "{module_name}_{layer_name}";
  # 2) The layer instance;
  # 3) The shape of the input to the layer;
  # 4) The kwargs used in the benchmark. It can include the number of
  #    iterations to run the benchmarks, and kwargs used in the layer call.
  #    By default, # of iteratons is 10000.
  _benchmark_parameters = [
      ("advanced_activations_leaky_relu", tf.keras.layers.LeakyReLU(),
       (1, 1)),
      ("advanced_activations_prelu", tf.keras.layers.PReLU(), (1, 1)),
      ("advanced_activations_elu", tf.keras.layers.ELU(), (1, 1)),
      ("advanced_activations_thresholded_relu",
       tf.keras.layers.ThresholdedReLU(), (1, 1)),
      ("advanced_activations_softmax", tf.keras.layers.Softmax(), (1, 1)),
      ("advanced_activations_relu", tf.keras.layers.ReLU(), (1, 1)),
      ("core_masking", tf.keras.layers.Masking(), (1, 1)),
      ("core_dropout", tf.keras.layers.Dropout(0.5), (1, 1), {
          "training": True
      }),
      ("core_flatten", tf.keras.layers.Flatten(), (1, 1, 1)),
      ("core_dense", tf.keras.layers.Dense(1), (1, 1)),
      ("convolutional_conv1d", tf.keras.layers.Conv1D(1, (1,)), (1, 1, 1)),
      ("convolutional_conv2d", tf.keras.layers.Conv2D(1, (1, 1)), (1, 1, 1, 1)),
      ("convolutional_conv3d", tf.keras.layers.Conv3D(
          1, (1, 1, 1)), (1, 1, 1, 1, 1)),
      ("batch_norm_fused_inf", tf.keras.layers.BatchNormalization(fused=True),
       (1, 1, 1, 1)),
      ("batch_norm_fused_train", tf.keras.layers.BatchNormalization(fused=True),
       (1, 1, 1, 1), {"training": True}),
      ("batch_norm_nonfused_inf",
       tf.keras.layers.BatchNormalization(fused=False), (1, 1, 1, 1)),
      ("batch_norm_nonfused_train",
       tf.keras.layers.BatchNormalization(fused=False), (1, 1, 1, 1),
       {"training": True}),
      ("normalization_layer_normalization",
       tf.keras.layers.LayerNormalization(), (1, 1),
       {"iters": 100, "training": True}),
  ]

  def benchmark_layer(self, layer, input_shape, kwargs=None):

    x = tf.ones(input_shape)

    def fn():
      layer(x, **(kwargs or {}))

    default_iters = 10000
    iters = kwargs.pop("iters", default_iters) if kwargs else default_iters
    self._run(fn, iters)


if __name__ == "__main__":
  assert tf.executing_eagerly()
  tf.test.main()
