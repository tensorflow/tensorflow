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
import numpy as np

import tensorflow as tf
from tensorflow.python.keras.benchmarks import benchmark_util
from tensorflow.python.keras.benchmarks.layer_benchmarks import layer_benchmarks_test_base
from tensorflow.python.platform import benchmark  # pylint: disable=unused-import


def _get_metadata(name):
  return {
      "model_name": "ideal_layers",
      "parameters": name[1] + "_shape",
  }


def _get_layer_args(layer_cls, layer_args):
  # To make benchmark parameters compatible with GPU platform.
  if layer_cls is tf.keras.layers.Bidirectional:
    return {"layer": tf.keras.layers.LSTM(1)}
  return layer_args


def _get_input_data(inputs):
  if "input_shape" in inputs:
    return tf.ones(inputs["input_shape"])
  elif "input" in inputs:
    return inputs["input"]
  else:
    raise ValueError("Please specificy either `input_shape` or `input`"
                     "for the benchmark test")


def _layer_call_backward(layer, x):
  with tf.GradientTape() as tape:
    y = layer(x)
    loss = tf.reduce_mean(y**2)

  _ = tape.gradient(loss, layer.trainable_variables)

CORE_LAYERS = [
    ("Dense_small_shape", tf.keras.layers.Dense,
     {"units": 32, "activation": "relu"},
     {"input_shape": (1, 16)}, 100),
    ("Activation_small_shape", tf.keras.layers.Activation,
     {"activation": "relu"},
     {"input_shape": (1, 4)}, 100),
    ("Embedding_small_shape", tf.keras.layers.Embedding,
     {"input_dim": 1, "output_dim": 1, "input_length": 1},
     {"input": np.random.randint(1, size=(1, 1))}, 100),
    ("Embedding_normal_shape", tf.keras.layers.Embedding,
     {"input_dim": 1000, "output_dim": 64, "input_length": 10},
     {"input": np.random.randint(1000, size=(32, 10))}, 100),
    ("Masking_small_shape", tf.keras.layers.Masking,
     {"mask_value": 1}, {"input_shape": (1, 1)}, 100),
    ("Lambda_small_shape", tf.keras.layers.Lambda,
     {"function": lambda x: x ** 2}, {"input_shape": (1, 1)}, 100),
    ("Flatten_small_shape", tf.keras.layers.Flatten,
     {}, {"input_shape": (1, 1)}, 100),
]

CONV_LAYERS = [
    ("Conv1D_small_shape", tf.keras.layers.Conv1D,
     {"filters": 1, "kernel_size": 1, "activation": "relu"},
     {"input_shape": (1, 1, 1)}, 100),
    ("Conv2D_small_shape", tf.keras.layers.Conv2D,
     {"filters": 1, "kernel_size": 1, "activation": "relu"},
     {"input_shape": (1, 1, 1, 1)}, 100),
    ("Conv2D_normal_shape", tf.keras.layers.Conv2D,
     {"filters": 1, "kernel_size": 1, "activation": "relu"},
     {"input_shape": (64, 28, 28, 3)}, 100),
    ("Conv3D_small_shape", tf.keras.layers.Conv3D,
     {"filters": 1, "kernel_size": 1, "activation": "relu"},
     {"input_shape": (1, 1, 1, 1, 1)}, 100),
    ("Conv1DTranspose_small_shape", tf.keras.layers.Conv1DTranspose,
     {"filters": 1, "kernel_size": 1, "activation": "relu"},
     {"input_shape": (1, 1, 1)}, 100),
    ("Conv2DTranspose_small_shape", tf.keras.layers.Conv2DTranspose,
     {"filters": 1, "kernel_size": 1, "activation": "relu"},
     {"input_shape": (1, 1, 1, 1)}, 100),
    ("Conv3DTranspose_small_shape", tf.keras.layers.Conv3DTranspose,
     {"filters": 1, "kernel_size": 1, "activation": "relu"},
     {"input_shape": (1, 1, 1, 1, 1)}, 100),
    ("SeparableConv1D_small_shape", tf.keras.layers.SeparableConv1D,
     {"filters": 1, "kernel_size": 1, "activation": "relu"},
     {"input_shape": (1, 1, 1)}, 100),
    ("SeparableConv2D_small_shape", tf.keras.layers.SeparableConv2D,
     {"filters": 1, "kernel_size": 1, "activation": "relu"},
     {"input_shape": (1, 1, 1, 1)}, 100),
    ("DepthwiseConv2D_small_shape", tf.keras.layers.DepthwiseConv2D,
     {"kernel_size": 1, "activation": "relu"},
     {"input_shape": (1, 1, 1, 1)}, 100),
]

RECURRENT_LAYERS = [
    ("LSTM_small_shape", tf.keras.layers.LSTM,
     {"units": 1}, {"input_shape": (1, 1, 1)}, 100),
    ("LSTM_normal_shape", tf.keras.layers.LSTM,
     {"units": 4}, {"input_shape": (32, 10, 8)}, 100),
    ("GRU_small_shape", tf.keras.layers.GRU,
     {"units": 1}, {"input_shape": (1, 1, 1)}, 100),
    ("SimpleRNN_small_shape", tf.keras.layers.SimpleRNN,
     {"units": 1}, {"input_shape": (1, 1, 1)}, 100),
    ("TimeDistributed_small_shape", tf.keras.layers.TimeDistributed,
     {"layer": tf.keras.layers.Conv2D(1, 1)},
     {"input_shape": (1, 1, 1, 1, 1)}, 100),
    ("Bidirectional_small_shape", tf.keras.layers.Bidirectional,
     {}, {"input_shape": (1, 1, 1)}, 100),
    ("ConvLSTM2D_small_shape", tf.keras.layers.ConvLSTM2D,
     {"filters": 1, "kernel_size": 1, "activation": "relu"},
     {"input_shape": (1, 1, 1, 1, 1)}, 100),
    ("RNN_small_shape", tf.keras.layers.RNN,
     {"cell": tf.keras.layers.LSTMCell(1)}, {"input_shape": (1, 1, 1)}, 100),
]

NORMALIZATION_LAYERS = [
    ("BatchNormalization_small_shape", tf.keras.layers.BatchNormalization,
     {"axis": -1}, {"input_shape": (1, 1, 1)}, 100),
    ("LayerNormalization_small_shape", tf.keras.layers.LayerNormalization,
     {"axis": -1}, {"input_shape": (1, 1, 1)}, 100),
]

REGULARIZATION_LAYERS = [
    ("Dropout_small_shape", tf.keras.layers.Dropout,
     {"rate": 0.2}, {"input_shape": (1, 1, 1)}, 100),
    ("SpatialDropout1D_small_shape", tf.keras.layers.SpatialDropout1D,
     {"rate": 0.2}, {"input_shape": (1, 1, 1)}, 100),
    ("SpatialDropout2D_small_shape", tf.keras.layers.SpatialDropout2D,
     {"rate": 0.2}, {"input_shape": (1, 1, 1, 1)}, 100),
    ("SpatialDropout3D_small_shape", tf.keras.layers.SpatialDropout3D,
     {"rate": 0.2}, {"input_shape": (1, 1, 1, 1, 1)}, 100),
    ("GaussianDropout_small_shape", tf.keras.layers.GaussianDropout,
     {"rate": 0.2}, {"input_shape": (1, 1, 1)}, 100),
    ("GaussianNoise_small_shape", tf.keras.layers.GaussianNoise,
     {"stddev": 0.1}, {"input_shape": (1, 1, 1)}, 100),
    ("ActivityRegularization_small_shape",
     tf.keras.layers.ActivityRegularization,
     {"l1": 0.3}, {"input_shape": (1, 1, 1)}, 100),
    ("AlphaDropout_small_shape", tf.keras.layers.AlphaDropout,
     {"rate": 0.2}, {"input_shape": (1, 1, 1)}, 100),
]


ATTENSION_LAYERS = [
    ("Attention_small_shape", tf.keras.layers.Attention,
     {"use_scale": False}, {"input": [np.ones((1, 1, 1)), np.ones((1, 1, 1))]},
     100),
    ("AdditiveAttention_small_shape", tf.keras.layers.AdditiveAttention,
     {"use_scale": True}, {"input": [np.ones((1, 1, 1)), np.ones((1, 1, 1))]},
     100),
]

POOLING_LAYERS = [
    ("MaxPooling1D_small_shape", tf.keras.layers.MaxPooling1D,
     {"pool_size": 1, "strides": 1}, {"input_shape": (1, 1, 1)}, 100),
    ("MaxPooling2D_small_shape", tf.keras.layers.MaxPooling2D,
     {"pool_size": 1, "strides": 1}, {"input_shape": (1, 1, 1, 1)}, 100),
    ("MaxPooling3D_small_shape", tf.keras.layers.MaxPooling3D,
     {"pool_size": 1, "strides": 1}, {"input_shape": (1, 1, 1, 1, 1)}, 100),
    ("AveragePooling1D_small_shape", tf.keras.layers.AveragePooling1D,
     {"pool_size": 1, "strides": 1}, {"input_shape": (1, 1, 1)}, 100),
    ("AveragePooling2D_small_shape", tf.keras.layers.AveragePooling2D,
     {"pool_size": 1, "strides": 1}, {"input_shape": (1, 1, 1, 1)}, 100),
    ("AveragePooling3D_small_shape", tf.keras.layers.AveragePooling3D,
     {"pool_size": 1, "strides": 1}, {"input_shape": (1, 1, 1, 1, 1)}, 100),
    ("GlobalMaxPooling1D_small_shape", tf.keras.layers.GlobalMaxPooling1D,
     {}, {"input_shape": (1, 1, 1)}, 100),
    ("GlobalMaxPooling2D_small_shape", tf.keras.layers.GlobalMaxPooling2D,
     {}, {"input_shape": (1, 1, 1, 1)}, 100),
    ("GlobalMaxPooling3D_small_shape", tf.keras.layers.GlobalMaxPooling3D,
     {}, {"input_shape": (1, 1, 1, 1, 1)}, 100),
    ("GlobalAveragePooling1D_small_shape",
     tf.keras.layers.GlobalAveragePooling1D,
     {}, {"input_shape": (1, 1, 1)}, 100),
    ("GlobalAveragePooling2D_small_shape",
     tf.keras.layers.GlobalAveragePooling2D,
     {}, {"input_shape": (1, 1, 1, 1)}, 100),
    ("GlobalAveragePooling3D_small_shape",
     tf.keras.layers.GlobalAveragePooling3D,
     {}, {"input_shape": (1, 1, 1, 1, 1)}, 100),
]


class KerasLayerBenchmarks(  # pylint: disable=undefined-variable
    layer_benchmarks_test_base.LayerBenchmarksBase,
    metaclass=benchmark.ParameterizedBenchmark):

  # The parameter of each layer benchmark is a tuple, and the first one is
  # the benchmark name. It must follow the convention of
  # "{layer_name}_{small|normal|large}_shape" to make it compatible with
  # `self.report_benchmark()` method.
  _benchmark_parameters = benchmark_util.generate_benchmark_params_cpu_gpu(
      CORE_LAYERS + CONV_LAYERS + RECURRENT_LAYERS + NORMALIZATION_LAYERS +
      REGULARIZATION_LAYERS + ATTENSION_LAYERS + POOLING_LAYERS)

  def benchmark_layer_call(self, layer_cls, layer_args, inputs, num_iters):
    layer = layer_cls(**_get_layer_args(layer_cls, layer_args))
    x = _get_input_data(inputs)

    fn = functools.partial(layer, x)
    name = benchmark_util.get_benchmark_name(self._get_name())
    metadata = {"implementation": name[0] + ".layer.call"}
    metadata.update(_get_metadata(name))
    self.run_report(fn, num_iters, metadata)

  def benchmark_layer_call_with_function(
      self, layer_cls, layer_args, inputs, num_iters):
    layer = layer_cls(**_get_layer_args(layer_cls, layer_args))
    x = _get_input_data(inputs)
    layer.call = tf.function(layer.call)

    fn = functools.partial(layer, x)
    name = benchmark_util.get_benchmark_name(self._get_name())
    metadata = {"implementation": name[0] + ".layer.call.function"}
    metadata.update(_get_metadata(name))
    self.run_report(fn, num_iters, metadata)

  def benchmark_layer_call_with_xla(
      self, layer_cls, layer_args, inputs, num_iters):
    name = benchmark_util.get_benchmark_name(self._get_name())
    # TODO(b/173461426)
    if layer_cls is tf.keras.layers.Embedding and name[-1] == "GPU":
      return
    layer = layer_cls(**_get_layer_args(layer_cls, layer_args))
    x = _get_input_data(inputs)
    layer.call = tf.function(
        layer.call, jit_compile=True)

    fn = functools.partial(layer, x)
    metadata = {"implementation": name[0] + ".layer.call.xla"}
    metadata.update(_get_metadata(name))
    self.run_report(fn, num_iters, metadata)

  def benchmark_layer_call_backward(
      self, layer_cls, layer_args, inputs, num_iters):
    layer = layer_cls(**_get_layer_args(layer_cls, layer_args))
    x = _get_input_data(inputs)

    fn = functools.partial(_layer_call_backward, layer, x)
    name = benchmark_util.get_benchmark_name(self._get_name())
    metadata = {"implementation": name[0] + ".layer.call.backward"}
    metadata.update(_get_metadata(name))
    self.run_report(fn, num_iters, metadata)

  def benchmark_layer_call_backward_with_function(
      self, layer_cls, layer_args, inputs, num_iters):
    layer = layer_cls(**_get_layer_args(layer_cls, layer_args))
    x = _get_input_data(inputs)
    layer.call = tf.function(layer.call)

    fn = functools.partial(_layer_call_backward, layer, x)
    name = benchmark_util.get_benchmark_name(self._get_name())
    metadata = {"implementation": name[0] + ".layer.call.backward.function"}
    metadata.update(_get_metadata(name))
    self.run_report(fn, num_iters, metadata)

  def benchmark_layer_call_backward_with_xla(
      self, layer_cls, layer_args, inputs, num_iters):
    name = benchmark_util.get_benchmark_name(self._get_name())
    # TODO(b/153480400)
    if layer_cls in [
        tf.keras.layers.LSTM, tf.keras.layers.Bidirectional,
        tf.keras.layers.ConvLSTM2D, tf.keras.layers.GRU, tf.keras.layers.RNN,
        tf.keras.layers.SimpleRNN
    ]:
      return
    # TODO(b/173461426)
    if layer_cls is tf.keras.layers.Embedding and name[-1] == "GPU":
      return
    layer = layer_cls(**_get_layer_args(layer_cls, layer_args))
    x = _get_input_data(inputs)
    layer.call = tf.function(
        layer.call, jit_compile=True)

    fn = functools.partial(_layer_call_backward, layer, x)
    metadata = {"implementation": name[0] + ".layer.call.backward.xla"}
    metadata.update(_get_metadata(name))
    self.run_report(fn, num_iters, metadata)


if __name__ == "__main__":
  tf.test.main()
