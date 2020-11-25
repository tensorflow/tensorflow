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
"""Benchmarks on Keras layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import six

import tensorflow as tf
from tensorflow.python.keras.benchmarks.layer_benchmarks import layer_benchmarks_test_base
from tensorflow.python.platform import benchmark


def _get_benchmark_name(name):
  return name.split("__")[-1].split("_")


def _get_metadata(name):
  return {
      "model_name": "ideal_layers",
      "parameters": name[1] + "_shape",
  }


def _generate_benchmark_params(*params_list):
  benchmark_params = []
  for params in params_list:
    benchmark_params.extend(
        [((param[0] + "_CPU",) + param[1:]) for param in params])
    benchmark_params.extend(
        [((param[0] + "_GPU",) + param[1:]) for param in params])
  return benchmark_params


def _layer_call_backward(layer, x):
  with tf.GradientTape() as tape:
    y = layer(x)
    loss = tf.reduce_mean(y**2)

  _ = tape.gradient(loss, layer.trainable_variables)


class KerasLayerBenchmarks(six.with_metaclass(
    benchmark.ParameterizedBenchmark,
    layer_benchmarks_test_base.LayerBenchmarksBase)):

  # The parameter of each layer benchmark is a tuple, and the first one is
  # the benchmark name. It must follow the convention of
  # "{layer_name}_{small|normal|large}_shape" to make it compatible with
  # `self.report_benchmark()` method.
  _benchmark_parameters = _generate_benchmark_params([
      ("Conv2D_small_shape", tf.keras.layers.Conv2D,
       {"filters": 1, "kernel_size": 1, "activation": "relu"},
       (1, 1, 1, 1), 10000),
      ("Conv2D_normal_shape", tf.keras.layers.Conv2D,
       {"filters": 1, "kernel_size": 1, "activation": "relu"},
       (64, 28, 28, 3), 10000),
      ("LSTM_small_shape", tf.keras.layers.LSTM,
       {"units": 1}, (1, 1, 1), 10000),
      ("LSTM_normal_shape", tf.keras.layers.LSTM,
       {"units": 4}, (32, 10, 8), 10000),
  ])

  def benchmark_layer_call(self, layer_cls, layer_args, input_shape, num_iters):
    layer = layer_cls(**layer_args)
    x = tf.ones(input_shape)

    fn = functools.partial(layer, x)
    name = _get_benchmark_name(self._get_name())
    metadata = {"implementation": name[0] + ".layer.call"}
    metadata.update(_get_metadata(name))
    self.run_report(fn, num_iters, metadata)

  def benchmark_layer_call_with_function(
      self, layer_cls, layer_args, input_shape, num_iters):
    layer = layer_cls(**layer_args)
    x = tf.ones(input_shape)
    layer.call = tf.function(layer.call)

    fn = functools.partial(layer, x)
    name = _get_benchmark_name(self._get_name())
    metadata = {"implementation": name[0] + ".layer.call.function"}
    metadata.update(_get_metadata(name))
    self.run_report(fn, num_iters, metadata)

  def benchmark_layer_call_with_xla(
      self, layer_cls, layer_args, input_shape, num_iters):
    layer = layer_cls(**layer_args)
    x = tf.ones(input_shape)
    layer.call = tf.function(
        layer.call, jit_compile=True)

    fn = functools.partial(layer, x)
    name = _get_benchmark_name(self._get_name())
    metadata = {"implementation": name[0] + ".layer.call.xla"}
    metadata.update(_get_metadata(name))
    self.run_report(fn, num_iters, metadata)

  def benchmark_layer_call_backward(
      self, layer_cls, layer_args, input_shape, num_iters):
    layer = layer_cls(**layer_args)
    x = tf.ones(input_shape)

    fn = functools.partial(_layer_call_backward, layer, x)
    name = _get_benchmark_name(self._get_name())
    metadata = {"implementation": name[0] + ".layer.call.backward"}
    metadata.update(_get_metadata(name))
    self.run_report(fn, num_iters, metadata)

  def benchmark_layer_call_backward_with_function(
      self, layer_cls, layer_args, input_shape, num_iters):
    layer = layer_cls(**layer_args)
    x = tf.ones(input_shape)
    layer.call = tf.function(layer.call)

    fn = functools.partial(_layer_call_backward, layer, x)
    name = _get_benchmark_name(self._get_name())
    metadata = {"implementation": name[0] + ".layer.call.backward.function"}
    metadata.update(_get_metadata(name))
    self.run_report(fn, num_iters, metadata)


class KerasLayerBenchmarksBackwardXLA(six.with_metaclass(
    benchmark.ParameterizedBenchmark,
    layer_benchmarks_test_base.LayerBenchmarksBase)):

  _benchmark_parameters = _generate_benchmark_params([
      ("Conv2D_small_shape", tf.keras.layers.Conv2D,
       {"filters": 1, "kernel_size": 1, "activation": "relu"},
       (1, 1, 1, 1), 10000),
      ("Conv2D_normal_shape", tf.keras.layers.Conv2D,
       {"filters": 1, "kernel_size": 1, "activation": "relu"},
       (64, 28, 28, 3), 10000),
      # TODO(b/153480400)
      # ("LSTM_small_shape", tf.keras.layers.LSTM,
      #  {"units": 1}, (1, 1, 1), 10000),
      # ("LSTM_normal_shape", tf.keras.layers.LSTM,
      #  {"units": 4}, (32, 10, 8), 10000),
  ])

  def benchmark_layer_call_backward_with_xla(
      self, layer_cls, layer_args, input_shape, num_iters):
    layer = layer_cls(**layer_args)
    x = tf.ones(input_shape)
    layer.call = tf.function(
        layer.call, jit_compile=True)

    fn = functools.partial(_layer_call_backward, layer, x)
    name = _get_benchmark_name(self._get_name())
    metadata = {"implementation": name[0] + ".layer.call.backward.xla"}
    metadata.update(_get_metadata(name))
    self.run_report(fn, num_iters, metadata)


if __name__ == "__main__":
  tf.test.main()
