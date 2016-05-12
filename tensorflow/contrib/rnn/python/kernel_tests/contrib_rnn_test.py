"""Tests for rnn module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import timeit

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
from tensorflow.python.kernel_tests import rnn_test

tf.contrib.rnn.Load()


def _flatten(list_of_lists):
  return [x for y in list_of_lists for x in y]


class LSTMTest(tf.test.TestCase):

  def setUp(self):
    self._seed = 23489
    np.random.seed(self._seed)

  def _testLSTMBasicToCellBlockRNN(self, use_gpu, use_sequence_length):
    time_steps = 8
    num_units = 3
    num_proj = 4
    input_size = 5
    batch_size = 2

    input_values = np.random.randn(time_steps, batch_size, input_size)

    if use_sequence_length:
      sequence_length = np.random.randint(0, time_steps, size=batch_size)
    else:
      sequence_length = None

    ########### Step 1: Run BasicLSTMCell
    initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=self._seed)
    basic_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

    (values_basic, state_value_basic, basic_grad_values,
     basic_individual_grad_values,
     basic_individual_var_grad_values) = rnn_test.run_static_rnn(
         self, time_steps, batch_size, input_size, basic_lstm_cell,
         sequence_length, input_values, use_gpu, initializer=initializer)

    ########### Step 2: Run LSTMCellBlock
    lstm_cell_block_cell = tf.contrib.rnn.LSTMCellBlock(num_units)

    (values_block, state_value_block, block_grad_values,
     block_individual_grad_values,
     block_individual_var_grad_values) = rnn_test.run_static_rnn(
         self, time_steps, batch_size, input_size, lstm_cell_block_cell,
         sequence_length, input_values, use_gpu, initializer=initializer)

    ######### Step 3: Comparisons
    self.assertEqual(len(values_basic), len(values_block))
    for (value_basic, value_block) in zip(values_basic, values_block):
      self.assertAllClose(value_basic, value_block)

    self.assertAllClose(basic_grad_values, block_grad_values)

    self.assertEqual(len(basic_individual_grad_values),
                     len(block_individual_grad_values))
    self.assertEqual(len(basic_individual_var_grad_values),
                     len(block_individual_var_grad_values))

    for i, (a, b) in enumerate(zip(basic_individual_grad_values,
                                   block_individual_grad_values)):
      tf.logging.info("Comparing individual gradients iteration %d" % i)
      self.assertAllClose(a, b)

    for i, (a, b) in enumerate(reversed(zip(basic_individual_var_grad_values,
                                        block_individual_var_grad_values))):
      tf.logging.info(
          "Comparing individual variable gradients iteraiton %d" % i)
      self.assertAllClose(a, b)

  def testLSTMBasicToCellBlockRNN(self):
    for use_gpu in (True, False):
      for use_sequence_length in (True, False):
        self._testLSTMBasicToCellBlockRNN(
            use_gpu=use_gpu, use_sequence_length=use_sequence_length)

class BenchmarkLSTM(tf.test.Benchmark):

  def benchmarkGraphCreationBasicVsBlockLSTM(self):
    print("Graph Creation: Static Basic vs. Block Unroll LSTM")
    print("max_t \t dt(basic) \t dt(block) \t dt(basic)/dt(block)")
    for max_time in (1, 25, 50):
      s_dt, d_dt = graph_creation_basic_vs_block_rnn_benchmark(max_time)
      self.report_benchmark(name="graph_creation_time_basic_T%02d" % max_time,
                            iters=5, wall_time=s_dt)
      self.report_benchmark(name="graph_creation_time_block_T%02d" % max_time,
                            iters=5, wall_time=d_dt)

  def benchmarkBasicVsBlockLSTM(self):
    print("Calculation: Static Unroll with Basic LSTM vs. Block LSTM")
    print("batch \t max_t \t units \t gpu \t dt(basic) \t dt(block) "
          "\t dt(basic)/dt(block)")
    for batch_size in (512, 256, 128, 64, 32, 16):
      for max_time in (50,):
        for num_units in (512, 256, 128):
          for use_gpu in (False, True):
            s_dt, d_dt = basic_vs_block_rnn_benchmark(
                batch_size, max_time, num_units, use_gpu)
            self.report_benchmark(
                name="basic_unroll_time_T%02d_B%03d_N%03d_gpu_%s"
                % (max_time, batch_size, num_units, use_gpu),
                iters=20, wall_time=s_dt)
            self.report_benchmark(
                name="block_unroll_time_T%02d_B%03d_N%03d_gpu_%s"
                % (max_time, batch_size, num_units, use_gpu),
                iters=20, wall_time=d_dt)


def graph_creation_basic_vs_block_rnn_benchmark(max_time):
  config = tf.ConfigProto()
  config.allow_soft_placement = True

  # These parameters don't matter
  batch_size = 512
  num_units = 512

  # Set up sequence lengths
  np.random.seed([127])
  sequence_length = np.random.randint(0, max_time, size=batch_size)
  sequence_length = None
  inputs_list = [
      np.random.randn(batch_size, num_units).astype(np.float32)
      for _ in range(max_time)]
  inputs = np.dstack(inputs_list).transpose([0, 2, 1])  # batch x time x depth

  basic_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units)
  block_cell = tf.contrib.rnn.LSTMCellBlock(num_units=num_units)

  def _create_static_rnn(cell):
    with tf.Session(config=config, graph=tf.Graph()) as sess:
      inputs_list_t = [
          tf.Variable(x, trainable=False).value() for x in inputs_list]
      ops = rnn_test._rnn_benchmark_static(
          inputs_list_t, sequence_length, cell)

  delta_basic = timeit.timeit(lambda: _create_static_rnn(basic_cell), number=5)
  delta_block = timeit.timeit(lambda: _create_static_rnn(block_cell), number=5)

  print("%d \t %f \t %f \t %f" %
        (max_time, delta_basic, delta_block, delta_block / delta_basic))
  return delta_basic, delta_block


def basic_vs_block_rnn_benchmark(batch_size, max_time, num_units, use_gpu):
  config = tf.ConfigProto()
  config.allow_soft_placement = True

  # Set up sequence lengths
  np.random.seed([127])
  sequence_length = np.random.randint(0, max_time, size=batch_size)
  sequence_length = None
  inputs_list = [
      np.random.randn(batch_size, num_units).astype(np.float32)
      for _ in range(max_time)]
  inputs = np.dstack(inputs_list).transpose([0, 2, 1])  # batch x time x depth

  # Using basic.
  with tf.Session(config=config, graph=tf.Graph()) as sess:
    with tf.device("/cpu:0" if not use_gpu else None):
      cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units)
      inputs_list_t = [
          tf.Variable(x, trainable=False).value() for x in inputs_list]
      ops = rnn_test._rnn_benchmark_static(
          inputs_list_t, sequence_length, cell)
    tf.initialize_all_variables().run()
    delta_basic = _timer(sess, ops)

  # Using block.
  with tf.Session(config=config, graph=tf.Graph()) as sess:
    with tf.device("/cpu:0" if not use_gpu else None):
      cell = tf.contrib.rnn.LSTMCellBlock(num_units=num_units)
      inputs_list_t = [
          tf.Variable(x, trainable=False).value() for x in inputs_list]
      ops = rnn_test._rnn_benchmark_static(
          inputs_list_t, sequence_length, cell)
    tf.initialize_all_variables().run()
    delta_block = _timer(sess, ops)

  print("%d \t %d \t %d \t %s \t %s \t %f \t %f" %
        (batch_size, max_time, num_units, use_gpu, delta_basic, delta_block,
         delta_block/delta_basic))

  return delta_basic, delta_block


def _timer(sess, ops):
  # Warm in
  for _ in range(2):
    sess.run(ops)

  # Timing run
  runs = 20
  start = time.time()
  for _ in range(runs):
    sess.run(ops)
  end = time.time()
  return (end - start)/float(runs)


if __name__ == "__main__":
  tf.test.main()
