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
"""LSTM Block Cell ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.rnn.python.kernel_tests import benchmarking
from tensorflow.contrib.rnn.python.ops import lstm_ops
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

block_lstm = lstm_ops._block_lstm  # pylint: disable=protected-access


def blocks_match(sess, use_peephole):
  batch_size = 2
  input_size = 3
  cell_size = 4
  sequence_length = 4

  inputs = []
  for _ in range(sequence_length):
    inp = ops.convert_to_tensor(
        np.random.randn(batch_size, input_size), dtype=dtypes.float32)
    inputs.append(inp)
  stacked_inputs = array_ops.stack(inputs)

  initializer = init_ops.random_uniform_initializer(-0.01, 0.01, seed=19890212)

  with variable_scope.variable_scope("test", initializer=initializer):
    # magic naming so that the cells pick up these variables and resuse them
    if use_peephole:
      wci = variable_scope.get_variable(
          "rnn/lstm_cell/w_i_diag", shape=[cell_size], dtype=dtypes.float32)
      wcf = variable_scope.get_variable(
          "rnn/lstm_cell/w_f_diag", shape=[cell_size], dtype=dtypes.float32)
      wco = variable_scope.get_variable(
          "rnn/lstm_cell/w_o_diag", shape=[cell_size], dtype=dtypes.float32)

    w = variable_scope.get_variable(
        "rnn/lstm_cell/kernel",
        shape=[input_size + cell_size, cell_size * 4],
        dtype=dtypes.float32)
    b = variable_scope.get_variable(
        "rnn/lstm_cell/bias",
        shape=[cell_size * 4],
        dtype=dtypes.float32,
        initializer=init_ops.zeros_initializer())

    basic_cell = rnn_cell.LSTMCell(
        cell_size, use_peepholes=use_peephole, state_is_tuple=True, reuse=True)
    basic_outputs_op, basic_state_op = rnn.static_rnn(
        basic_cell, inputs, dtype=dtypes.float32)

    if use_peephole:
      _, _, _, _, _, _, block_outputs_op = block_lstm(
          ops.convert_to_tensor(sequence_length, dtype=dtypes.int64),
          inputs,
          w,
          b,
          wci=wci,
          wcf=wcf,
          wco=wco,
          cell_clip=0,
          use_peephole=True)
    else:
      _, _, _, _, _, _, block_outputs_op = block_lstm(
          ops.convert_to_tensor(sequence_length, dtype=dtypes.int64),
          inputs,
          w,
          b,
          cell_clip=0)

    fused_cell = lstm_ops.LSTMBlockFusedCell(
        cell_size, cell_clip=0, use_peephole=use_peephole, reuse=True,
        name="rnn/lstm_cell")
    fused_outputs_op, fused_state_op = fused_cell(
        stacked_inputs, dtype=dtypes.float32)

    sess.run([variables.global_variables_initializer()])
    basic_outputs, basic_state = sess.run([basic_outputs_op, basic_state_op[0]])
    basic_grads = sess.run(gradients_impl.gradients(basic_outputs_op, inputs))
    xs = [w, b]
    if use_peephole:
      xs += [wci, wcf, wco]
    basic_wgrads = sess.run(gradients_impl.gradients(basic_outputs_op, xs))

    block_outputs = sess.run(block_outputs_op)
    block_grads = sess.run(gradients_impl.gradients(block_outputs_op, inputs))
    block_wgrads = sess.run(gradients_impl.gradients(block_outputs_op, xs))

    xs = [w, b]
    if use_peephole:
      xs += [wci, wcf, wco]
    fused_outputs, fused_state = sess.run([fused_outputs_op, fused_state_op[0]])
    fused_grads = sess.run(gradients_impl.gradients(fused_outputs_op, inputs))
    fused_wgrads = sess.run(gradients_impl.gradients(fused_outputs_op, xs))

    return (basic_state, fused_state, basic_outputs, block_outputs,
            fused_outputs, basic_grads, block_grads, fused_grads, basic_wgrads,
            block_wgrads, fused_wgrads)


class LSTMBlockCellTest(test.TestCase):

  def testNoneDimsWithDynamicRNN(self):
    with self.test_session(use_gpu=True, graph=ops.Graph()) as sess:
      batch_size = 4
      num_steps = 5
      input_dim = 6
      cell_size = 7

      cell = lstm_ops.LSTMBlockCell(cell_size)
      x = array_ops.placeholder(dtypes.float32, shape=(None, None, input_dim))

      output, _ = rnn.dynamic_rnn(
          cell, x, time_major=True, dtype=dtypes.float32)
      sess.run(variables.global_variables_initializer())
      feed = {}
      feed[x] = np.random.randn(num_steps, batch_size, input_dim)
      sess.run(output, feed)

  def testLSTMBlockCell(self):
    with self.test_session(use_gpu=True, graph=ops.Graph()) as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 2])
        m0 = array_ops.zeros([1, 2])
        m1 = array_ops.zeros([1, 2])
        m2 = array_ops.zeros([1, 2])
        m3 = array_ops.zeros([1, 2])
        g, ((out_m0, out_m1), (out_m2, out_m3)) = rnn_cell.MultiRNNCell(
            [lstm_ops.LSTMBlockCell(2)
             for _ in range(2)], state_is_tuple=True)(x, ((m0, m1), (m2, m3)))
        sess.run([variables.global_variables_initializer()])
        res = sess.run([g, out_m0, out_m1, out_m2, out_m3], {
            x.name: np.array([[1., 1.]]),
            m0.name: 0.1 * np.ones([1, 2]),
            m1.name: 0.1 * np.ones([1, 2]),
            m2.name: 0.1 * np.ones([1, 2]),
            m3.name: 0.1 * np.ones([1, 2])
        })
        self.assertEqual(len(res), 5)
        self.assertAllClose(res[0], [[0.24024698, 0.24024698]])
        # These numbers are from testBasicLSTMCell and only test c/h.
        self.assertAllClose(res[1], [[0.68967271, 0.68967271]])
        self.assertAllClose(res[2], [[0.44848421, 0.44848421]])
        self.assertAllClose(res[3], [[0.39897051, 0.39897051]])
        self.assertAllClose(res[4], [[0.24024698, 0.24024698]])

  def testCompatibleNames(self):
    with self.test_session(use_gpu=True, graph=ops.Graph()):
      cell = rnn_cell.LSTMCell(10)
      pcell = rnn_cell.LSTMCell(10, use_peepholes=True)
      inputs = [array_ops.zeros([4, 5])] * 6
      rnn.static_rnn(cell, inputs, dtype=dtypes.float32, scope="basic")
      rnn.static_rnn(pcell, inputs, dtype=dtypes.float32, scope="peephole")
      basic_names = {
          v.name: v.get_shape()
          for v in variables.trainable_variables()
      }

    with self.test_session(use_gpu=True, graph=ops.Graph()):
      cell = lstm_ops.LSTMBlockCell(10)
      pcell = lstm_ops.LSTMBlockCell(10, use_peephole=True)
      inputs = [array_ops.zeros([4, 5])] * 6
      rnn.static_rnn(cell, inputs, dtype=dtypes.float32, scope="basic")
      rnn.static_rnn(pcell, inputs, dtype=dtypes.float32, scope="peephole")
      block_names = {
          v.name: v.get_shape()
          for v in variables.trainable_variables()
      }

    with self.test_session(use_gpu=True, graph=ops.Graph()):
      cell = lstm_ops.LSTMBlockFusedCell(10)
      pcell = lstm_ops.LSTMBlockFusedCell(10, use_peephole=True)
      inputs = array_ops.stack([array_ops.zeros([4, 5])] * 6)
      cell(inputs, dtype=dtypes.float32, scope="basic/lstm_cell")
      pcell(inputs, dtype=dtypes.float32, scope="peephole/lstm_cell")
      fused_names = {
          v.name: v.get_shape()
          for v in variables.trainable_variables()
      }

    self.assertEqual(basic_names, block_names)
    self.assertEqual(basic_names, fused_names)

  def testLSTMBasicToBlockCell(self):
    with self.test_session(use_gpu=True) as sess:
      x = array_ops.zeros([1, 2])
      x_values = np.random.randn(1, 2)

      m0_val = 0.1 * np.ones([1, 2])
      m1_val = -0.1 * np.ones([1, 2])
      m2_val = -0.2 * np.ones([1, 2])
      m3_val = 0.2 * np.ones([1, 2])

      initializer = init_ops.random_uniform_initializer(
          -0.01, 0.01, seed=19890212)
      with variable_scope.variable_scope("basic", initializer=initializer):
        m0 = array_ops.zeros([1, 2])
        m1 = array_ops.zeros([1, 2])
        m2 = array_ops.zeros([1, 2])
        m3 = array_ops.zeros([1, 2])
        g, ((out_m0, out_m1), (out_m2, out_m3)) = rnn_cell.MultiRNNCell(
            [rnn_cell.BasicLSTMCell(2, state_is_tuple=True) for _ in range(2)],
            state_is_tuple=True)(x, ((m0, m1), (m2, m3)))
        sess.run([variables.global_variables_initializer()])
        basic_res = sess.run([g, out_m0, out_m1, out_m2, out_m3], {
            x.name: x_values,
            m0.name: m0_val,
            m1.name: m1_val,
            m2.name: m2_val,
            m3.name: m3_val
        })

      with variable_scope.variable_scope("block", initializer=initializer):
        m0 = array_ops.zeros([1, 2])
        m1 = array_ops.zeros([1, 2])
        m2 = array_ops.zeros([1, 2])
        m3 = array_ops.zeros([1, 2])
        g, ((out_m0, out_m1), (out_m2, out_m3)) = rnn_cell.MultiRNNCell(
            [lstm_ops.LSTMBlockCell(2)
             for _ in range(2)], state_is_tuple=True)(x, ((m0, m1), (m2, m3)))
        sess.run([variables.global_variables_initializer()])
        block_res = sess.run([g, out_m0, out_m1, out_m2, out_m3], {
            x.name: x_values,
            m0.name: m0_val,
            m1.name: m1_val,
            m2.name: m2_val,
            m3.name: m3_val
        })

      self.assertEqual(len(basic_res), len(block_res))
      for basic, block in zip(basic_res, block_res):
        self.assertAllClose(basic, block)

  def testLSTMBasicToBlockCellPeeping(self):
    with self.test_session(use_gpu=True) as sess:
      x = array_ops.zeros([1, 2])
      x_values = np.random.randn(1, 2)

      m0_val = 0.1 * np.ones([1, 2])
      m1_val = -0.1 * np.ones([1, 2])
      m2_val = -0.2 * np.ones([1, 2])
      m3_val = 0.2 * np.ones([1, 2])

      initializer = init_ops.random_uniform_initializer(
          -0.01, 0.01, seed=19890212)
      with variable_scope.variable_scope("basic", initializer=initializer):
        m0 = array_ops.zeros([1, 2])
        m1 = array_ops.zeros([1, 2])
        m2 = array_ops.zeros([1, 2])
        m3 = array_ops.zeros([1, 2])
        g, ((out_m0, out_m1), (out_m2, out_m3)) = rnn_cell.MultiRNNCell(
            [
                rnn_cell.LSTMCell(2, use_peepholes=True, state_is_tuple=True)
                for _ in range(2)
            ],
            state_is_tuple=True)(x, ((m0, m1), (m2, m3)))
        sess.run([variables.global_variables_initializer()])
        basic_res = sess.run([g, out_m0, out_m1, out_m2, out_m3], {
            x.name: x_values,
            m0.name: m0_val,
            m1.name: m1_val,
            m2.name: m2_val,
            m3.name: m3_val
        })

      with variable_scope.variable_scope("block", initializer=initializer):
        m0 = array_ops.zeros([1, 2])
        m1 = array_ops.zeros([1, 2])
        m2 = array_ops.zeros([1, 2])
        m3 = array_ops.zeros([1, 2])
        g, ((out_m0, out_m1), (out_m2, out_m3)) = rnn_cell.MultiRNNCell(
            [lstm_ops.LSTMBlockCell(2, use_peephole=True) for _ in range(2)],
            state_is_tuple=True)(x, ((m0, m1), (m2, m3)))
        sess.run([variables.global_variables_initializer()])
        block_res = sess.run([g, out_m0, out_m1, out_m2, out_m3], {
            x.name: x_values,
            m0.name: m0_val,
            m1.name: m1_val,
            m2.name: m2_val,
            m3.name: m3_val
        })

      self.assertEqual(len(basic_res), len(block_res))
      for basic, block in zip(basic_res, block_res):
        self.assertAllClose(basic, block)

  def testLSTMBasicToBlock(self):
    with self.test_session(use_gpu=True) as sess:
      (basic_state, fused_state, basic_outputs, block_outputs, fused_outputs,
       basic_grads, block_grads, fused_grads, basic_wgrads, block_wgrads,
       fused_wgrads) = blocks_match(
           sess, use_peephole=False)

      self.assertAllClose(basic_outputs, block_outputs)
      self.assertAllClose(basic_grads, block_grads)
      for basic, block in zip(basic_wgrads, block_wgrads):
        self.assertAllClose(basic, block, rtol=1e-6, atol=1e-6)

      self.assertAllClose(basic_outputs, fused_outputs)
      self.assertAllClose(basic_state, fused_state)
      self.assertAllClose(basic_grads, fused_grads)
      for basic, fused in zip(block_wgrads, fused_wgrads):
        self.assertAllClose(basic, fused, rtol=1e-6, atol=1e-6)

  def testLSTMBasicToBlockPeeping(self):
    with self.test_session(use_gpu=True) as sess:
      (basic_state, fused_state, basic_outputs, block_outputs, fused_outputs,
       basic_grads, block_grads, fused_grads, basic_wgrads, block_wgrads,
       fused_wgrads) = blocks_match(
           sess, use_peephole=True)

      self.assertAllClose(basic_outputs, block_outputs)
      self.assertAllClose(basic_grads, block_grads)
      for basic, block in zip(basic_wgrads, block_wgrads):
        self.assertAllClose(basic, block, rtol=1e-6, atol=1e-6)

      self.assertAllClose(basic_outputs, fused_outputs)
      self.assertAllClose(basic_state, fused_state)
      self.assertAllClose(basic_grads, fused_grads)
      for basic, fused in zip(block_wgrads, fused_wgrads):
        self.assertAllClose(basic, fused, rtol=1e-6, atol=1e-6)

  def testLSTMFusedSequenceLengths(self):
    """Verify proper support for sequence lengths in LSTMBlockFusedCell."""
    with self.test_session(use_gpu=True) as sess:
      batch_size = 3
      input_size = 4
      cell_size = 5
      max_sequence_length = 6

      inputs = []
      for _ in range(max_sequence_length):
        inp = ops.convert_to_tensor(
            np.random.randn(batch_size, input_size), dtype=dtypes.float32)
        inputs.append(inp)
      seq_lengths = constant_op.constant([3, 4, 5])
      cell_inputs = array_ops.stack(inputs)

      initializer = init_ops.random_uniform_initializer(
          -0.01, 0.01, seed=19890213)

      with variable_scope.variable_scope("lstm_cell", initializer=initializer):
        # magic naming so that the cells pick up these variables and reuse them
        variable_scope.get_variable(
            "kernel",
            shape=[input_size + cell_size, cell_size * 4],
            dtype=dtypes.float32)

        variable_scope.get_variable(
            "bias",
            shape=[cell_size * 4],
            dtype=dtypes.float32,
            initializer=init_ops.zeros_initializer())

      cell = lstm_ops.LSTMBlockFusedCell(
          cell_size, cell_clip=0, use_peephole=False, reuse=True,
          name="lstm_cell")

      fused_outputs_op, fused_state_op = cell(
          cell_inputs, dtype=dtypes.float32, sequence_length=seq_lengths)

      cell_vars = [
          v for v in variables.trainable_variables()
          if v.name.endswith("kernel") or v.name.endswith("bias")
      ]

      # Verify that state propagation works if we turn our sequence into
      # tiny (single-time) subsequences, i.e. unfuse the cell
      unfused_outputs_op = []
      state = None
      with variable_scope.variable_scope(
          variable_scope.get_variable_scope(), reuse=True):
        for i, inp in enumerate(inputs):
          lengths = [int(i < l) for l in seq_lengths.eval()]
          output, state = cell(
              array_ops.expand_dims(inp, 0),
              initial_state=state,
              dtype=dtypes.float32,
              sequence_length=lengths)
          unfused_outputs_op.append(output[0])
      unfused_outputs_op = array_ops.stack(unfused_outputs_op)

      sess.run([variables.global_variables_initializer()])
      unfused_outputs, unfused_state = sess.run([unfused_outputs_op, state[0]])
      unfused_grads = sess.run(
          gradients_impl.gradients(unfused_outputs_op, inputs))
      unfused_wgrads = sess.run(
          gradients_impl.gradients(unfused_outputs_op, cell_vars))

      fused_outputs, fused_state = sess.run(
          [fused_outputs_op, fused_state_op[0]])
      fused_grads = sess.run(gradients_impl.gradients(fused_outputs_op, inputs))
      fused_wgrads = sess.run(
          gradients_impl.gradients(fused_outputs_op, cell_vars))

      self.assertAllClose(fused_outputs, unfused_outputs)
      self.assertAllClose(fused_state, unfused_state)
      self.assertAllClose(fused_grads, unfused_grads)
      for fused, unfused in zip(fused_wgrads, unfused_wgrads):
        self.assertAllClose(fused, unfused, rtol=1e-6, atol=1e-6)

#### Benchmarking.


class BenchmarkLSTMBlock(test.Benchmark):

  def benchmarkLSTMBlockCellFpropWithDynamicRNN(self):
    print("BlockLSTMCell forward propagation via dynamic_rnn().")
    print("--------------------------------------------------------------")
    print("LSTMBlockCell Seconds per inference.")
    print("batch_size,cell_size,input_size,time_steps,use_gpu,wall_time")
    iters = 10
    for config in benchmarking.dict_product({
        "batch_size": [1, 8, 13, 32, 67, 128],
        "cell_size": [128, 250, 512, 650, 1024, 1350],
        "time_steps": [40],
        "use_gpu": [True, False]
    }):
      with ops.Graph().as_default():
        with benchmarking.device(use_gpu=config["use_gpu"]):
          inputs = variable_scope.get_variable(
              "x",
              [config["time_steps"], config["batch_size"], config["cell_size"]])
          cell = lstm_ops.LSTMBlockCell(config["cell_size"])
          outputs = rnn.dynamic_rnn(
              cell, inputs, time_major=True, dtype=dtypes.float32)
          init_op = variables.global_variables_initializer()

        with session.Session() as sess:
          sess.run(init_op)
          wall_time = benchmarking.seconds_per_run(outputs, sess, iters)

        # Print to stdout. If the TEST_REPORT_FILE_PREFIX environment variable
        # is set, this will produce a copy-paste-able CSV file.
        print(",".join(
            map(str, [
                config["batch_size"], config["cell_size"], config["cell_size"],
                config["time_steps"], config["use_gpu"], wall_time
            ])))
        benchmark_name_template = "_".join([
            "LSTMBlockCell_fprop", "BS%(batch_size)i", "CS%(cell_size)i",
            "IS%(cell_size)i", "TS%(time_steps)i", "gpu_%(use_gpu)s"
        ])

        self.report_benchmark(
            name=benchmark_name_template % config,
            iters=iters,
            wall_time=wall_time,
            extras=config)

  def benchmarkLSTMBlockCellBpropWithDynamicRNN(self):
    print("BlockLSTMCell backward propagation via dynamic_rnn().")
    print("--------------------------------------------------------------")
    print("LSTMBlockCell Seconds per inference.")
    print("batch_size,cell_size,input_size,time_steps,use_gpu,wall_time")
    iters = 10
    for config in benchmarking.dict_product({
        "batch_size": [1, 8, 13, 32, 67, 128],
        "cell_size": [128, 250, 512, 650, 1024, 1350],
        "time_steps": [40],
        "use_gpu": [True, False]
    }):
      with ops.Graph().as_default():
        with benchmarking.device(use_gpu=config["use_gpu"]):
          time_steps = config["time_steps"]
          batch_size = config["batch_size"]
          cell_size = input_size = config["cell_size"]
          inputs = variable_scope.get_variable(
              "x", [time_steps, batch_size, cell_size],
              trainable=False,
              dtype=dtypes.float32)
          with variable_scope.variable_scope(
              "rnn", reuse=variable_scope.AUTO_REUSE):
            w = variable_scope.get_variable(
                "rnn/lstm_cell/kernel",
                shape=[input_size + cell_size, cell_size * 4],
                dtype=dtypes.float32)
            b = variable_scope.get_variable(
                "rnn/lstm_cell/bias",
                shape=[cell_size * 4],
                dtype=dtypes.float32,
                initializer=init_ops.zeros_initializer())
            cell = lstm_ops.LSTMBlockCell(cell_size)
            outputs = rnn.dynamic_rnn(
                cell, inputs, time_major=True, dtype=dtypes.float32)
          grads = gradients_impl.gradients(outputs, [inputs, w, b])
          init_op = variables.global_variables_initializer()

        with session.Session() as sess:
          sess.run(init_op)
          wall_time = benchmarking.seconds_per_run(grads, sess, iters)

        # Print to stdout. If the TEST_REPORT_FILE_PREFIX environment variable
        # is set, this will produce a copy-paste-able CSV file.
        print(",".join(
            map(str, [
                batch_size, cell_size, cell_size, time_steps, config["use_gpu"],
                wall_time
            ])))
        benchmark_name_template = "_".join([
            "LSTMBlockCell_bprop", "BS%(batch_size)i", "CS%(cell_size)i",
            "IS%(cell_size)i", "TS%(time_steps)i", "gpu_%(use_gpu)s"
        ])

        self.report_benchmark(
            name=benchmark_name_template % config,
            iters=iters,
            wall_time=wall_time,
            extras=config)


if __name__ == "__main__":
  test.main()
