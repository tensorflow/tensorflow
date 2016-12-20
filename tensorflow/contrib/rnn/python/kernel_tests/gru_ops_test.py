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
"""Tests for Block GRU module."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import gru_ops
from tensorflow.python.ops import variable_scope as vs


class GRUBlockCellTest(tf.test.TestCase):
  _use_gpu = False

  def testNoneDimsWithDynamicRNN(self):
    with self.test_session(use_gpu=self._use_gpu, graph=tf.Graph()) as sess:
      batch_size = 4
      cell_size = 5
      input_size = 6
      num_steps = 7

      cell = gru_ops.GRUBlockCell(cell_size)

      x = tf.placeholder(tf.float32, shape=(None, None, input_size))
      _, output = tf.nn.dynamic_rnn(cell, x, time_major=True, dtype=tf.float32)
      sess.run(tf.global_variables_initializer())
      feed = {}
      feed[x] = np.random.randn(num_steps, batch_size, input_size)
      sess.run(output, feed)

  def testBlockGRUToGRUCellSingleStep(self):
    with self.test_session(use_gpu=self._use_gpu, graph=tf.Graph()) as sess:
      batch_size = 4
      cell_size = 5
      input_size = 6

      seed = 1994
      initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=seed)

      # Inputs
      x = tf.zeros([batch_size, input_size])
      h = tf.zeros([batch_size, cell_size])

      # Values for the inputs.
      x_value = np.random.rand(batch_size, input_size)
      h_value = np.random.rand(batch_size, cell_size)

      # Output from the basic GRU cell implementation.
      with tf.variable_scope("basic", initializer=initializer):
        output = tf.contrib.rnn.GRUCell(cell_size)(x, h)
        sess.run([tf.global_variables_initializer()])
        basic_res = sess.run([output], {x: x_value, h: h_value})

      # Output from the block GRU cell implementation.
      with tf.variable_scope("block", initializer=initializer):
        output = gru_ops.GRUBlockCell(cell_size)(x, h)
        sess.run([tf.global_variables_initializer()])
        block_res = sess.run([output], {x: x_value, h: h_value})

      self.assertEqual(len(block_res), len(basic_res))
      for block, basic in zip(block_res, basic_res):
        self.assertAllClose(block, basic)

  def testBlockGRUToGRUCellMultiStep(self):
    with self.test_session(use_gpu=self._use_gpu, graph=tf.Graph()) as sess:
      batch_size = 2
      cell_size = 3
      input_size = 3
      time_steps = 4

      # Random initializers.
      seed = 1994
      initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=seed)
      np.random.seed(seed)

      # Inputs
      concat_x = tf.placeholder(
          tf.float32, shape=(time_steps, batch_size, input_size))
      h = tf.zeros([batch_size, cell_size])

      # Values for the inputs.
      x_values = np.random.rand(time_steps, batch_size, input_size)
      h_value = np.random.rand(batch_size, cell_size)

      # Output from the block GRU cell implementation.
      with tf.variable_scope("block", initializer=initializer):
        cell = gru_ops.GRUBlockCell(cell_size)
        outputs_dynamic, state_dynamic = tf.nn.dynamic_rnn(
            cell,
            inputs=concat_x,
            initial_state=h,
            time_major=True,
            dtype=tf.float32)
        feeds = {concat_x: x_values, h: h_value}
        sess.run([tf.global_variables_initializer()])
        block_res = sess.run([outputs_dynamic, state_dynamic], feeds)

      # Output from the basic GRU cell implementation.
      with tf.variable_scope("basic", initializer=initializer):
        cell = tf.contrib.rnn.GRUCell(cell_size)
        outputs_dynamic, state_dynamic = tf.nn.dynamic_rnn(
            cell,
            inputs=concat_x,
            initial_state=h,
            time_major=True,
            dtype=tf.float32)
        feeds = {concat_x: x_values, h: h_value}
        sess.run([tf.global_variables_initializer()])
        basic_res = sess.run([outputs_dynamic, state_dynamic], feeds)

      # Check the lengths of the outputs_dynamic, and states.
      self.assertEqual(len(block_res), len(basic_res))
      self.assertEqual(len(block_res[0]), len(basic_res[0]))
      self.assertEqual(len(block_res[1]), len(basic_res[1]))

      # Check the outputs_dynamic values.
      for block_output, basic_output in zip(block_res[0], basic_res[0]):
        self.assertAllClose(block_output, basic_output)

      # Check the state_dynamic value.
      self.assertAllClose(block_res[1], block_res[1])

  def testDerivativeOfBlockGRUToGRUCellSingleStep(self):
    with self.test_session(use_gpu=self._use_gpu, graph=tf.Graph()) as sess:
      batch_size = 2
      cell_size = 3
      input_size = 4

      seed = 1994
      initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=seed)
      np.random.seed(seed)

      # Inputs
      x = tf.zeros([batch_size, input_size])
      h = tf.zeros([batch_size, cell_size])

      # Values for the inputs.
      x_value = np.random.rand(batch_size, input_size)
      h_value = np.random.rand(batch_size, cell_size)

      # Gradients from the block GRU cell implementation.
      with tf.variable_scope("block", initializer=initializer):
        output = gru_ops.GRUBlockCell(cell_size)(x, h)
        sess.run([tf.global_variables_initializer()])

        all_variables = tf.global_variables()[0:4]
        [w_ru, b_ru, w_c, b_c] = all_variables

        d_new_h_wrt_x = tf.gradients([output], x)
        d_new_h_wrt_h = tf.gradients([output], h)
        d_new_h_wrt_w_ru = tf.gradients([output], w_ru)
        d_new_h_wrt_w_c = tf.gradients([output], w_c)
        d_new_h_wrt_b_ru = tf.gradients([output], b_ru)
        d_new_h_wrt_b_c = tf.gradients([output], b_c)

        d_block_res = sess.run([d_new_h_wrt_x, d_new_h_wrt_h, d_new_h_wrt_w_ru,
                                d_new_h_wrt_w_c, d_new_h_wrt_b_ru,
                                d_new_h_wrt_b_c], {x: x_value,
                                                   h: h_value})

      # Gradients from the basic GRU cell implementation.
      with tf.variable_scope("basic", initializer=initializer):
        output = tf.contrib.rnn.GRUCell(cell_size)(x, h)
        sess.run([tf.global_variables_initializer()])

        all_variables = tf.global_variables()[4:8]
        [w_ru, b_ru, w_c, b_c] = all_variables

        d_new_h_wrt_x = tf.gradients([output], x)
        d_new_h_wrt_h = tf.gradients([output], h)
        d_new_h_wrt_w_ru = tf.gradients([output], w_ru)
        d_new_h_wrt_w_c = tf.gradients([output], w_c)
        d_new_h_wrt_b_ru = tf.gradients([output], b_ru)
        d_new_h_wrt_b_c = tf.gradients([output], b_c)

        d_basic_res = sess.run([d_new_h_wrt_x, d_new_h_wrt_h, d_new_h_wrt_w_ru,
                                d_new_h_wrt_w_c, d_new_h_wrt_b_ru,
                                d_new_h_wrt_b_c], {x: x_value,
                                                   h: h_value})

      # Check lengths of derivative results.
      self.assertEqual(len(d_block_res), len(d_basic_res))
      # Check the value of every derivative result.
      for block, basic in zip(d_block_res, d_basic_res):
        self.assertAllClose(block, basic)

  def testDerivativeOfBlockGRUToGRUCellMultiSteps(self):
    batch_size = 2
    cell_size = 3
    input_size = 4
    time_steps = 2
    with self.test_session(use_gpu=self._use_gpu, graph=tf.Graph()) as sess:
      # Random initializers.
      seed = 1994
      initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=seed)
      np.random.seed(seed)

      # Inputs
      concat_x = tf.placeholder(
          tf.float32, shape=(time_steps, batch_size, input_size))
      h = tf.zeros([batch_size, cell_size])

      # Values for the inputs.
      x_values = np.random.rand(time_steps, batch_size, input_size)
      h_value = np.random.rand(batch_size, cell_size)
      feeds = {concat_x: x_values, h: h_value}

      # Gradients from the block GRU cell implementation.
      with tf.variable_scope("block", initializer=initializer):
        cell = gru_ops.GRUBlockCell(cell_size)

        outputs_dynamic, _ = tf.nn.dynamic_rnn(
            cell,
            inputs=concat_x,
            initial_state=h,
            time_major=True,
            dtype=tf.float32)
        grad_output_wrt_x = tf.gradients([outputs_dynamic[0]], concat_x)
        grad_output_wrt_h = tf.gradients([outputs_dynamic[0]], h)

        sess.run([tf.global_variables_initializer()])
        block_grad_res_x, block_grad_res_h = sess.run(
            [grad_output_wrt_x, grad_output_wrt_h], feeds)

      # Gradients from the basic GRU cell implementation.
      with tf.variable_scope("basic", initializer=initializer):
        cell = tf.contrib.rnn.GRUCell(cell_size)

        outputs_dynamic, _ = tf.nn.dynamic_rnn(
            cell,
            inputs=concat_x,
            initial_state=h,
            time_major=True,
            dtype=tf.float32)
        grad_output_wrt_x = tf.gradients([outputs_dynamic[0]], concat_x)
        grad_output_wrt_h = tf.gradients([outputs_dynamic[0]], h)

        sess.run([tf.global_variables_initializer()])
        basic_grad_res_x, basic_grad_res_h = sess.run(
            [grad_output_wrt_x, grad_output_wrt_h], feeds)

    # Check derivatives values of the outputs wrt to x.
    self.assertEqual(len(block_grad_res_x), len(basic_grad_res_x))

    # Check derivatives values of the outputs wrt to h.
    for block, basic in zip(block_grad_res_x, basic_grad_res_x):
      self.assertAllClose(block, basic)

    # Check derivatives values of the outputs wrt to x.
    self.assertEqual(len(block_grad_res_h), len(basic_grad_res_h))

    # Check derivatives values of the outputs wrt to h.
    for block, basic in zip(block_grad_res_h, basic_grad_res_h):
      self.assertAllClose(block, basic)

  def testGradient(self):
    with self.test_session(use_gpu=self._use_gpu, graph=tf.Graph()) as sess:
      batch_size = 1
      cell_size = 3
      input_size = 2

      # Inputs
      x = tf.zeros([batch_size, input_size])
      h = tf.zeros([batch_size, cell_size])
      output = gru_ops.GRUBlockCell(cell_size)(x, h)

      sess.run([tf.global_variables_initializer()])

      all_variables = tf.global_variables()

      [w_ru, b_ru, w_c, b_c] = all_variables[:4]

      error_x = tf.test.compute_gradient_error(x, (batch_size, input_size),
                                               output[0],
                                               (batch_size, cell_size))
      error_h = tf.test.compute_gradient_error(h, (batch_size, cell_size),
                                               output[0],
                                               (batch_size, cell_size))
      error_w_ru = tf.test.compute_gradient_error(w_ru, (input_size + cell_size,
                                                         2 * cell_size),
                                                  output[0],
                                                  (batch_size, cell_size))
      error_w_c = tf.test.compute_gradient_error(w_c, (input_size + cell_size,
                                                       cell_size), output[0],
                                                 (batch_size, cell_size))
      error_b_ru = tf.test.compute_gradient_error(b_ru, (2 * cell_size,),
                                                  output[0],
                                                  (batch_size, cell_size))
      error_b_c = tf.test.compute_gradient_error(b_c, (cell_size,), output[0],
                                                 (batch_size, cell_size))

    eps = 1e-4
    self.assertLess(error_x, eps)
    self.assertLess(error_h, eps)
    self.assertLess(error_w_ru, eps)
    self.assertLess(error_w_c, eps)
    self.assertLess(error_b_ru, eps)
    self.assertLess(error_b_c, eps)


class GRUBlockCellGpuTest(GRUBlockCellTest):
  _use_gpu = True

#### Benchmarking GRUBlockCell vs GRUCell.


def time_taken_by_op(op, sess, num_runs=50):
  """Time taken by the Op."""
  for _ in range(2):
    sess.run([op])

  start_time = time.time()
  for _ in range(num_runs):
    sess.run([op])

  end_time = time.time()
  time_taken = end_time - start_time
  return time_taken


def training_gru_block_vs_gru_cell(batch_size,
                                   cell_size,
                                   input_size,
                                   time_steps,
                                   use_gpu=False,
                                   iters=30):
  """Benchmark training speed between GRUBlockCell vs GRUCell."""
  tf.reset_default_graph()
  with tf.Session(graph=tf.Graph()) as sess:
    # Specify the device which is been used.
    with tf.device("/cpu:0" if not use_gpu else "/gpu:0"):

      # Random initializers.
      seed = 1994
      initializer = tf.random_uniform_initializer(-1, 1, seed=seed)
      np.random.seed(seed)

      # Inputs
      concat_x = vs.get_variable("concat_x",
                                 [time_steps, batch_size, input_size])
      h = vs.get_variable("h", [batch_size, cell_size])
      y = vs.get_variable("y", [time_steps, batch_size, cell_size])

      # Output from the basic GRU cell implementation.
      with tf.variable_scope("basic", initializer=initializer):
        cell = tf.contrib.rnn.GRUCell(cell_size)

        outputs_dynamic, _ = tf.nn.dynamic_rnn(
            cell,
            inputs=concat_x,
            initial_state=h,
            time_major=True,
            dtype=tf.float32)
        sess.run([tf.global_variables_initializer()])
        cost = tf.reduce_mean(tf.square(outputs_dynamic - y))
        learning_rate = 0.01
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
            cost)

        # time for a training step.
        basic_time_training = time_taken_by_op(optimizer, sess, iters)

      # Output from the basic GRU cell implementation.
      with tf.variable_scope("block", initializer=initializer):
        cell = gru_ops.GRUBlockCell(cell_size)

        outputs_dynamic, _ = tf.nn.dynamic_rnn(
            cell,
            inputs=concat_x,
            initial_state=h,
            time_major=True,
            dtype=tf.float32)
        sess.run([tf.global_variables_initializer()])
        cost = tf.reduce_mean(tf.square(outputs_dynamic - y))
        learning_rate = 0.01
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
            cost)

        # time for a training step.
        block_time_training = time_taken_by_op(optimizer, sess, iters)

    performance_training = (
        basic_time_training - block_time_training) * 100 / basic_time_training

    print(",".join([str(batch_size), str(cell_size), str(input_size), str(
        time_steps), str(use_gpu), str(basic_time_training), str(
            block_time_training), str(performance_training)]))

    return basic_time_training, block_time_training


def inference_gru_block_vs_gru_cell(batch_size,
                                    cell_size,
                                    input_size,
                                    time_steps,
                                    use_gpu=False,
                                    iters=30):
  """Benchmark inference speed between GRUBlockCell vs GRUCell."""
  tf.reset_default_graph()
  with tf.Session(graph=tf.Graph()) as sess:
    with tf.device("/cpu:0" if not use_gpu else "/gpu:0"):

      # Random initializers.
      seed = 1994
      initializer = tf.random_uniform_initializer(-1, 1, seed=seed)
      np.random.seed(seed)

      # Inputs
      concat_x = vs.get_variable("concat_x",
                                 [time_steps, batch_size, input_size])
      h = vs.get_variable("h", [batch_size, cell_size])

      # Output from the basic GRU cell implementation.
      with tf.variable_scope("basic", initializer=initializer):
        cell = tf.contrib.rnn.GRUCell(cell_size)
        outputs_dynamic, _ = tf.nn.dynamic_rnn(
            cell,
            inputs=concat_x,
            initial_state=h,
            time_major=True,
            dtype=tf.float32)
        sess.run([tf.global_variables_initializer()])
        basic_time_inference = time_taken_by_op(outputs_dynamic, sess, iters)

      # Output from the block GRU cell implementation.
      with tf.variable_scope("block", initializer=initializer):
        cell = gru_ops.GRUBlockCell(cell_size)
        outputs_dynamic, _ = tf.nn.dynamic_rnn(
            cell,
            inputs=concat_x,
            initial_state=h,
            time_major=True,
            dtype=tf.float32)
        sess.run([tf.global_variables_initializer()])
        block_time_inference = time_taken_by_op(outputs_dynamic, sess, iters)

    performance_inference = (basic_time_inference - block_time_inference
                            ) * 100 / basic_time_inference
    print(",".join([str(batch_size), str(cell_size), str(input_size), str(
        time_steps), str(use_gpu), str(basic_time_inference), str(
            block_time_inference), str(performance_inference)]))

    return basic_time_inference, block_time_inference


def single_bprop_step_gru_block_vs_gru_cell(batch_size,
                                            cell_size,
                                            input_size,
                                            use_gpu=False,
                                            iters=30):
  """Benchmark single bprop step speed between GRUBlockCell vs GRUCell."""
  tf.reset_default_graph()
  with tf.Session(graph=tf.Graph()) as sess:
    with tf.device("/cpu:0" if not use_gpu else "/gpu:0"):
      initializer = tf.random_uniform_initializer(-1, 1, seed=1989)
      # Inputs
      x = vs.get_variable("x", [batch_size, input_size])
      h = vs.get_variable("h", [batch_size, cell_size])

      # Output from the basic GRU cell implementation.
      with tf.variable_scope("basic", initializer=initializer):
        output = tf.contrib.rnn.GRUCell(cell_size)(tf.identity(x),
                                                   tf.identity(h))
        sess.run([tf.global_variables_initializer()])
        grad_output_wrt_input = tf.gradients([output], h)
        basic_time_bprop = time_taken_by_op(grad_output_wrt_input, sess, iters)

      # Output from the block GRU cell implementation.
      with tf.variable_scope("block", initializer=initializer):
        output = gru_ops.GRUBlockCell(cell_size)(tf.identity(x), tf.identity(h))
        sess.run([tf.global_variables_initializer()])
        grad_output_wrt_input = tf.gradients([output], h)
        block_time_bprop = time_taken_by_op(grad_output_wrt_input, sess, iters)

  performance_inference = (
      basic_time_bprop - block_time_bprop) * 100 / basic_time_bprop

  print(",".join([str(batch_size), str(cell_size), str(input_size), str(
      use_gpu), str(basic_time_bprop), str(block_time_bprop), str(
          performance_inference)]))

  return basic_time_bprop, block_time_bprop


class BenchmarkGRUBlock(tf.test.Benchmark):

  def benchmarkTrainingBlockGRUVsGRUCell(self):
    print("Comparison GRUBlockCell vs GRUCell")
    print("--------------------------------------------------------------")
    print("Training speed GRUBlockCell vs GRUCell")
    print("batch_size, cell_size, input_size, time_steps, GPU, "
          "basic_time_training, block_time_training, performance_training[%]")
    iters = 10
    for use_gpu in [True, False]:
      for batch_size in [1, 32, 128]:
        for cell_size in [128, 512]:
          for input_size in [128, 512]:
            for time_steps in [50]:
              basic_time, block_time = training_gru_block_vs_gru_cell(
                  batch_size, cell_size, input_size, time_steps, use_gpu, iters)
              self.report_benchmark(
                  name="GRUCell_training_time_BS%i_CS%i_IS%i_TS%i_gpu_%s" %
                  (batch_size, cell_size, input_size, time_steps, use_gpu),
                  iters=iters,
                  wall_time=basic_time)
              self.report_benchmark(
                  name="GRUBlockCell_training_time_BS%i_CS%i_IS%i_TS%i_gpu_%s" %
                  (batch_size, cell_size, input_size, time_steps, use_gpu),
                  iters=iters,
                  wall_time=block_time)

  def benchmarkInferenceBlockGRUVsGRUCell(self):
    print("--------------------------------------------------------------")
    print("Inference speed GRUBlockCell vs GRUCell")
    print(
        "batch_size, cell_size, input_size, time_steps, GPU, "
        "basic_time_inference, block_time_inference, performance_inference[%]")
    iters = 10
    for use_gpu in [True, False]:
      for batch_size in [1, 32, 128]:
        for cell_size in [128, 512]:
          for input_size in [128, 512]:
            for time_steps in [50]:
              basic_time, block_time = inference_gru_block_vs_gru_cell(
                  batch_size, cell_size, input_size, time_steps, use_gpu, iters)
              self.report_benchmark(
                  name="GRUCell_inference_time_BS%i_CS%i_IS%i_TS%i_gpu_%s" %
                  (batch_size, cell_size, input_size, time_steps, use_gpu),
                  iters=iters,
                  wall_time=basic_time)
              self.report_benchmark(
                  name="GRUBlockCell_inference_time_BS%i_CS%i_IS%i_TS%i_gpu_%s"
                  % (batch_size, cell_size, input_size, time_steps, use_gpu),
                  iters=iters,
                  wall_time=block_time)

  def benchmarkSingleBpropStepBlockGRUVsGRUCell(self):
    print("--------------------------------------------------------------")
    print("Single bprop step speed GRUBlockCell vs GRUCell")
    print("batch_size, cell_size, input_size, GPU, basic_time, "
          "block_time, performance_inference[%]")
    iters = 10
    for use_gpu in [True, False]:
      for batch_size in [1, 32, 128]:
        for cell_size in [128, 512]:
          for input_size in [128, 512]:
            basic_time, block_time = single_bprop_step_gru_block_vs_gru_cell(
                batch_size, cell_size, input_size, use_gpu, iters)
            self.report_benchmark(
                name="GRUCell_Bprop_single_step_time_BS%i_CS%i_IS%i_gpu_%s" %
                (batch_size, cell_size, input_size, use_gpu),
                iters=iters,
                wall_time=basic_time)
            self.report_benchmark(
                name="GRUBlockCell_Bprop_single_step_time_BS%i_CS%i_IS%i_gpu_%s"
                % (batch_size, cell_size, input_size, use_gpu),
                iters=iters,
                wall_time=block_time)

    print("--------------------------------------------------------------")


if __name__ == "__main__":
  tf.test.main()
