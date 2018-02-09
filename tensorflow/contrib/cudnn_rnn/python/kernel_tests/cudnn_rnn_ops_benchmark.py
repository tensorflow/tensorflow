# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Benchmarks for Cudnn RNN models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from six.moves import xrange
from tensorflow.contrib import rnn as contrib_rnn
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
from tensorflow.contrib.rnn.python.ops import lstm_ops
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import rnn
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class CudnnRNNBenchmark(test.Benchmark):
  """Benchmarks Cudnn LSTM and other related models.
  """

  def _GetTestConfig(self):
    return {
        "large": {
            "num_layers": 4,
            "num_units": 1024,
            "seq_length": 50,
            "batch_size": 64,
        },
        "medium": {
            "num_layers": 4,
            "num_units": 512,
            "seq_length": 50,
            "batch_size": 64,
        },
        "small": {
            "num_layers": 4,
            "num_units": 128,
            "seq_length": 50,
            "batch_size": 64,
        },
    }

  def _GetConfigDesc(self, config):
    num_layers = config["num_layers"]
    num_units = config["num_units"]
    batch_size = config["batch_size"]
    seq_length = config["seq_length"]

    return "y%d_u%d_b%d_q%d" % (num_layers, num_units, batch_size, seq_length)

  def _BenchmarkOp(self, op, desc):
    burn_in_steps = 10
    benchmark_steps = 20
    with session.Session() as sess:
      sess.run(variables.global_variables_initializer())
      for i in xrange(burn_in_steps + benchmark_steps):
        if i == burn_in_steps:
          start_time = time.time()
        sess.run(op)
      total_time = time.time() - start_time
      step_time = total_time / benchmark_steps
      print("%s takes %.4f sec/step" % (desc, step_time))
      self.report_benchmark(
          name=desc, iters=benchmark_steps, wall_time=total_time)

  def benchmarkCudnnLSTMTraining(self):
    test_configs = self._GetTestConfig()
    for config_name, config in test_configs.items():
      config = test_configs[config_name]
      num_layers = config["num_layers"]
      num_units = config["num_units"]
      batch_size = config["batch_size"]
      seq_length = config["seq_length"]

      with ops.Graph().as_default(), ops.device("/device:GPU:0"):
        model = cudnn_rnn_ops.CudnnLSTM(num_layers, num_units, num_units)
        params_size_t = model.params_size()
        input_data = variables.Variable(
            array_ops.ones([seq_length, batch_size, num_units]))
        input_h = variables.Variable(
            array_ops.ones([num_layers, batch_size, num_units]))
        input_c = variables.Variable(
            array_ops.ones([num_layers, batch_size, num_units]))
        params = variables.Variable(
            array_ops.ones([params_size_t]), validate_shape=False)
        output, output_h, output_c = model(
            is_training=True,
            input_data=input_data,
            input_h=input_h,
            input_c=input_c,
            params=params)
        all_grads = gradients_impl.gradients(
            [output, output_h, output_c],
            [params, input_data, input_h, input_c])
        training_op = control_flow_ops.group(*all_grads)
        self._BenchmarkOp(training_op, "cudnn_lstm %s %s" %
                          (config_name, self._GetConfigDesc(config)))

  def benchmarkTfRNNLSTMTraining(self):
    test_configs = self._GetTestConfig()
    for config_name, config in test_configs.items():
      num_layers = config["num_layers"]
      num_units = config["num_units"]
      batch_size = config["batch_size"]
      seq_length = config["seq_length"]

      with ops.Graph().as_default(), ops.device("/device:GPU:0"):
        inputs = array_ops.zeros([batch_size, seq_length, num_units],
                                 dtypes.float32)

        multi_cell = contrib_rnn.MultiRNNCell(
            [contrib_rnn.BasicLSTMCell(num_units) for _ in range(num_layers)])
        outputs, final_state = rnn.dynamic_rnn(
            multi_cell, inputs, dtype=dtypes.float32)
        trainable_variables = ops.get_collection(
            ops.GraphKeys.TRAINABLE_VARIABLES)
        gradients = gradients_impl.gradients([outputs, final_state],
                                             trainable_variables)
        training_op = control_flow_ops.group(*gradients)
        self._BenchmarkOp(training_op, "tf_rnn_lstm %s %s" %
                          (config_name, self._GetConfigDesc(config)))

  def benchmarkTfRNNLSTMBlockCellTraining(self):
    test_configs = self._GetTestConfig()
    for config_name, config in test_configs.items():
      num_layers = config["num_layers"]
      num_units = config["num_units"]
      batch_size = config["batch_size"]
      seq_length = config["seq_length"]

      with ops.Graph().as_default(), ops.device("/device:GPU:0"):
        inputs = array_ops.zeros([batch_size, seq_length, num_units],
                                 dtypes.float32)

        multi_cell = contrib_rnn.MultiRNNCell(
            [lstm_ops.LSTMBlockCell(num_units) for _ in range(num_layers)])
        outputs, final_state = rnn.dynamic_rnn(
            multi_cell, inputs, dtype=dtypes.float32)
        trainable_variables = ops.get_collection(
            ops.GraphKeys.TRAINABLE_VARIABLES)
        gradients = gradients_impl.gradients([outputs, final_state],
                                             trainable_variables)
        training_op = control_flow_ops.group(*gradients)
        self._BenchmarkOp(training_op, "tf_rnn_lstm_block_cell %s %s" %
                          (config_name, self._GetConfigDesc(config)))


if __name__ == "__main__":
  test.main()
