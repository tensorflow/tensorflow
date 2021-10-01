# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the LSTM cell and layer."""

import argparse
import os
import sys

import numpy as np

from tensorflow.compiler.tests import lstm
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def _DumpGraph(graph, basename):
  if FLAGS.dump_graph_dir:
    name = os.path.join(FLAGS.dump_graph_dir, basename + '.pbtxt')
    with open(name, 'w') as f:
      f.write(str(graph.as_graph_def()))


def _Sigmoid(x):
  return 1. / (1. + np.exp(-x))


def _Clip(x):
  return np.maximum(np.minimum(x, 1.), -1.)


class LSTMTest(test.TestCase):

  def setUp(self):
    # The tests for a single LSTM cell and LSTM layer use these values as
    # inputs.  We always set the dimensionality of num_inputs=1; thus batch_size
    # actually represents the different input cases.
    self._inputs = np.array([[-1.], [-.5], [0.], [.5], [1.]], np.float32)
    self._batch_size = len(self._inputs)

  def _NextC(self, inputs, weight, m_prev, c_prev):
    """Returns the next c states of an LSTM cell."""
    x = (inputs + m_prev) * weight
    return _Clip(_Clip(_Sigmoid(x) * c_prev) + _Clip(_Sigmoid(x) * np.tanh(x)))

  def _NextM(self, inputs, weight, m_prev, c_prev):
    """Returns the next m states of an LSTM cell."""
    x = (inputs + m_prev) * weight
    return _Clip(_Sigmoid(x) * self._NextC(inputs, weight, m_prev, c_prev))

  def _RunLSTMCell(self, basename, init_weights, m_prev_scalar, c_prev_scalar,
                   pad_scalar):
    with self.session() as sess:
      num_inputs = 1
      num_nodes = 1

      weights = init_weights(lstm.LSTMCellWeightsShape(num_inputs, num_nodes))
      m_prev = constant_op.constant([[m_prev_scalar]] * self._batch_size)
      c_prev = constant_op.constant([[c_prev_scalar]] * self._batch_size)
      x = constant_op.constant(self._inputs)
      pad = constant_op.constant([[pad_scalar]] * self._batch_size)

      m, c = lstm.LSTMCell(weights, m_prev, c_prev, x, pad)
      _DumpGraph(sess.graph, 'lstm_cell_%s_%d_%d_%d' %
                 (basename, m_prev_scalar, c_prev_scalar, pad_scalar))

      # Initialize variables and run the unrolled LSTM step.
      self.evaluate(variables.global_variables_initializer())
      return self.evaluate([m, c])

  def testLSTMCell(self):
    # Run with all-0 weights, no padding.
    m, c = self._RunLSTMCell('zeros', init_ops.zeros_initializer(), 0., 0., 0.)
    self.assertAllClose(m, [[0.]] * self._batch_size)
    self.assertAllClose(c, [[0.]] * self._batch_size)
    m, c = self._RunLSTMCell('zeros', init_ops.zeros_initializer(), 0., 1., 0.)
    self.assertAllClose(m, [[.25]] * self._batch_size)
    self.assertAllClose(c, [[.5]] * self._batch_size)
    m, c = self._RunLSTMCell('zeros', init_ops.zeros_initializer(), 1., 0., 0.)
    self.assertAllClose(m, [[.0]] * self._batch_size)
    self.assertAllClose(c, [[.0]] * self._batch_size)
    m, c = self._RunLSTMCell('zeros', init_ops.zeros_initializer(), 1., 1., 0.)
    self.assertAllClose(m, [[.25]] * self._batch_size)
    self.assertAllClose(c, [[.5]] * self._batch_size)

    # Run with all-1 weights, no padding.
    for m_prev in [0., 1.]:
      for c_prev in [0., 1.]:
        m, c = self._RunLSTMCell('ones',
                                 init_ops.ones_initializer(), m_prev, c_prev,
                                 0.)
        self.assertAllClose(m, self._NextM(self._inputs, 1., m_prev, c_prev))
        self.assertAllClose(c, self._NextC(self._inputs, 1., m_prev, c_prev))

    # Run with random weights.
    for weight in np.random.rand(3):
      weight_tf = constant_op.constant(weight, dtypes.float32)
      random_weight = lambda shape, w=weight_tf: array_ops.fill(shape, w)

      # No padding.
      for m_prev in [0., 1.]:
        for c_prev in [0., 1.]:
          m, c = self._RunLSTMCell('random', random_weight, m_prev, c_prev, 0.)
          self.assertAllClose(m,
                              self._NextM(self._inputs, weight, m_prev, c_prev))
          self.assertAllClose(c,
                              self._NextC(self._inputs, weight, m_prev, c_prev))

      # Set padding.
      for m_prev in [0., 1.]:
        for c_prev in [0., 1.]:
          m, c = self._RunLSTMCell('random', random_weight, m_prev, c_prev, 1.)
          self.assertAllClose(m, [[m_prev]] * self._batch_size)
          self.assertAllClose(c, [[c_prev]] * self._batch_size)

  def testLSTMLayerErrors(self):
    num_inputs = 1
    num_nodes = 1
    seq_length = 3

    weights = array_ops.zeros(lstm.LSTMCellWeightsShape(num_inputs, num_nodes))
    m = constant_op.constant([[0.]] * self._batch_size)
    c = constant_op.constant([[0.]] * self._batch_size)
    x_seq = [constant_op.constant(self._inputs)] * seq_length
    pad = constant_op.constant([[0.]] * self._batch_size)

    with self.assertRaisesWithPredicateMatch(ValueError, 'length of x_seq'):
      lstm.LSTMLayer('lstm', weights, m, c, x_seq, [pad])
    with self.assertRaisesWithPredicateMatch(ValueError, 'length of x_seq'):
      lstm.LSTMLayer('lstm', weights, m, c, x_seq, [pad] * 2)
    with self.assertRaisesWithPredicateMatch(ValueError, 'length of x_seq'):
      lstm.LSTMLayer('lstm', weights, m, c, x_seq, [pad] * 4)

  def _RunLSTMLayer(self, basename, init_weights, m_init_scalar, c_init_scalar,
                    pad_scalar):
    with self.session() as sess:
      num_inputs = 1
      num_nodes = 1
      seq_length = 3

      weights = init_weights(lstm.LSTMCellWeightsShape(num_inputs, num_nodes))
      m_init = constant_op.constant([[m_init_scalar]] * self._batch_size)
      c_init = constant_op.constant([[c_init_scalar]] * self._batch_size)
      x_seq = [constant_op.constant(self._inputs)] * seq_length
      pad_seq = [constant_op.constant([[pad_scalar]] * self._batch_size)
                ] * seq_length

      out_seq = lstm.LSTMLayer('lstm', weights, m_init, c_init, x_seq, pad_seq)
      _DumpGraph(sess.graph, 'lstm_layer_%s_%d_%d_%d' %
                 (basename, m_init_scalar, c_init_scalar, pad_scalar))

      # Initialize variables and run the unrolled LSTM layer.
      self.evaluate(variables.global_variables_initializer())
      return self.evaluate(out_seq)

  def testLSTMLayer(self):
    # Run with all-0 weights, no padding.
    o = self._RunLSTMLayer('zeros', init_ops.zeros_initializer(), 0., 0., 0.)
    self.assertAllClose(o, [[[0.]] * self._batch_size] * 3)
    o = self._RunLSTMLayer('zeros', init_ops.zeros_initializer(), 0., 1., 0.)
    self.assertAllClose(o, [[[.25]] * self._batch_size,
                            [[.125]] * self._batch_size,
                            [[.0625]] * self._batch_size])
    o = self._RunLSTMLayer('zeros', init_ops.zeros_initializer(), 1., 0., 0.)
    self.assertAllClose(o, [[[0.]] * self._batch_size] * 3)
    o = self._RunLSTMLayer('zeros', init_ops.zeros_initializer(), 1., 1., 0.)
    self.assertAllClose(o, [[[.25]] * self._batch_size,
                            [[.125]] * self._batch_size,
                            [[.0625]] * self._batch_size])

    # Run with all-1 weights, no padding.
    weight1 = 1.
    for m_init in [0., 1.]:
      for c_init in [0., 1.]:
        o = self._RunLSTMLayer('ones',
                               init_ops.ones_initializer(), m_init, c_init, 0.)
        m0 = self._NextM(self._inputs, weight1, m_init, c_init)
        c0 = self._NextC(self._inputs, weight1, m_init, c_init)
        self.assertAllClose(o[0], m0)
        m1 = self._NextM(self._inputs, weight1, m0, c0)
        c1 = self._NextC(self._inputs, weight1, m0, c0)
        self.assertAllClose(o[1], m1)
        m2 = self._NextM(self._inputs, weight1, m1, c1)
        self.assertAllClose(o[2], m2)

    # Run with random weights.
    for weight in np.random.rand(3):
      weight_tf = constant_op.constant(weight, dtypes.float32)
      random_weight = lambda shape, w=weight_tf: array_ops.fill(shape, w)

      # No padding.
      for m_init in [0., 1.]:
        for c_init in [0., 1.]:
          o = self._RunLSTMLayer('random', random_weight, m_init, c_init, 0.)
          m0 = self._NextM(self._inputs, weight, m_init, c_init)
          c0 = self._NextC(self._inputs, weight, m_init, c_init)
          self.assertAllClose(o[0], m0)
          m1 = self._NextM(self._inputs, weight, m0, c0)
          c1 = self._NextC(self._inputs, weight, m0, c0)
          self.assertAllClose(o[1], m1)
          m2 = self._NextM(self._inputs, weight, m1, c1)
          self.assertAllClose(o[2], m2)

      # Set padding.
      o = self._RunLSTMLayer('random', random_weight, 0., 0., 1.)
      self.assertAllClose(o, [[[0.]] * self._batch_size] * 3)
      o = self._RunLSTMLayer('random', random_weight, 0., 1., 1.)
      self.assertAllClose(o, [[[0.]] * self._batch_size] * 3)
      o = self._RunLSTMLayer('random', random_weight, 1., 0., 1.)
      self.assertAllClose(o, [[[1.]] * self._batch_size] * 3)
      o = self._RunLSTMLayer('random', random_weight, 1., 1., 1.)
      self.assertAllClose(o, [[[1.]] * self._batch_size] * 3)


class LSTMBenchmark(test.Benchmark):
  """Mcro-benchmarks for a single layer of LSTM cells."""

  def _LayerBuilder(self, do_training):
    out_seq, weights = lstm.BuildLSTMLayer(FLAGS.batch_size, FLAGS.seq_length,
                                           FLAGS.num_inputs, FLAGS.num_nodes)
    name, fetches = ('lstm_layer_inference', out_seq)
    if do_training:
      # Not a real loss function, but good enough for benchmarking backprop.
      loss = math_ops.reduce_sum(math_ops.add_n(out_seq))
      dw = gradients_impl.gradients(loss, weights)
      name, fetches = ('lstm_layer_training', dw)

    _DumpGraph(ops.get_default_graph(),
               '%s_%d_%d_%d_%d' % (name, FLAGS.batch_size, FLAGS.seq_length,
                                   FLAGS.num_inputs, FLAGS.num_nodes))
    return name, fetches

  def benchmarkLayerInference(self):
    xla_test.Benchmark(self, lambda: self._LayerBuilder(False), False,
                       FLAGS.device)

  def benchmarkLayerInferenceXLA(self):
    xla_test.Benchmark(self, lambda: self._LayerBuilder(False), True,
                       FLAGS.device)

  def benchmarkLayerTraining(self):
    xla_test.Benchmark(self, lambda: self._LayerBuilder(True), False,
                       FLAGS.device)

  def benchmarkLayerTrainingXLA(self):
    xla_test.Benchmark(self, lambda: self._LayerBuilder(True), True,
                       FLAGS.device)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--batch_size',
      type=int,
      default=128,
      help="""\
      Inputs are fed in batches of this size, for both inference and training.
      Larger values cause the matmul in each LSTM cell to have higher
      dimensionality.\
      """
  )
  parser.add_argument(
      '--seq_length',
      type=int,
      default=60,
      help="""\
      Length of the unrolled sequence of LSTM cells in a layer.Larger values
      cause more LSTM matmuls to be run.\
      """
  )
  parser.add_argument(
      '--num_inputs',
      type=int,
      default=1024,
      help='Dimension of inputs that are fed into each LSTM cell.'
  )
  parser.add_argument(
      '--num_nodes',
      type=int,
      default=1024,
      help='Number of nodes in each LSTM cell.'
  )
  parser.add_argument(
      '--device',
      type=str,
      default='gpu',
      help="""\
      TensorFlow device to assign ops to, e.g. "gpu", "cpu". For details see
      documentation for tf.Graph.device.\
      """
  )
  parser.add_argument(
      '--dump_graph_dir',
      type=str,
      default='',
      help='If non-empty, dump graphs in *.pbtxt format to this directory.'
  )
  global FLAGS  # pylint:disable=global-at-module-level
  FLAGS, unparsed = parser.parse_known_args()
  # This test is using Tensorflow sessions which are not compatible with eager
  # mode.
  ops.disable_eager_execution()
  test.main(argv=[sys.argv[0]] + unparsed)
