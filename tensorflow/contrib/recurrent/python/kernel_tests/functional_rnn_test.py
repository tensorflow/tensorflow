# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Functional RNN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


from tensorflow.contrib.recurrent.python.ops import functional_rnn
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import rnn as rnn_lib
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variables
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
import tensorflow.python.ops.tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test as test_lib
from tensorflow.python.platform import tf_logging as logging


def _CreateStackedLstmCell(*cell_sizes):
  subcells = [rnn_cell_impl.LSTMCell(cell_size) for cell_size in cell_sizes]
  return rnn_cell_impl.MultiRNNCell(subcells)


class FunctionalRnnTest(test_util.TensorFlowTestCase):

  _BATCH_SIZE = 3
  _TOTAL_TIME = 5
  _INPUT_SIZE = 11
  _NUM_UNITS = 7

  # Set this to some output if you want to use it.
  _LSTM_GRAPH_DEF_FILEPATH = None

  _CELLDEFS = {
      'gru': (rnn_cell_impl.GRUCell, [_NUM_UNITS]),
      'lstm': (rnn_cell_impl.LSTMCell, [_NUM_UNITS]),
      'stacked_lstm': (_CreateStackedLstmCell, [_NUM_UNITS] * 3)
  }

  def _CreateCell(self, celldef_name):
    func, args = self._CELLDEFS[celldef_name]
    return func(*args)

  def _CreateInputs(self, time_major=False):
    if time_major:
      inputs = np.random.random([
          FunctionalRnnTest._TOTAL_TIME, FunctionalRnnTest._BATCH_SIZE,
          FunctionalRnnTest._INPUT_SIZE
      ])
    else:
      inputs = np.random.random([
          FunctionalRnnTest._BATCH_SIZE, FunctionalRnnTest._TOTAL_TIME,
          FunctionalRnnTest._INPUT_SIZE
      ])
    # Always leave one time slot empty, to check max_length behavior.
    sequence_length = np.random.randint(
        0, high=FunctionalRnnTest._TOTAL_TIME - 1,
        size=FunctionalRnnTest._BATCH_SIZE,
        dtype=np.int)
    return (inputs, sequence_length)

  def _CreateSymmetricInputs(self):
    # total time = batch size
    inputs = np.zeros(
        (FunctionalRnnTest._BATCH_SIZE, FunctionalRnnTest._BATCH_SIZE,
         FunctionalRnnTest._INPUT_SIZE))
    for i in range(FunctionalRnnTest._BATCH_SIZE):
      for j in range(i, FunctionalRnnTest._BATCH_SIZE):
        inputs[i][j] = np.random.random([FunctionalRnnTest._INPUT_SIZE])
        inputs[j][i] = inputs[i][j]

    # Always leave one time slot empty, to check max_length behavior.
    sequence_length = np.random.randint(
        0,
        high=FunctionalRnnTest._BATCH_SIZE - 1,
        size=FunctionalRnnTest._BATCH_SIZE,
        dtype=np.int)
    return (inputs, sequence_length)

  def _CreateRnnGraph(self,
                      create_rnn_computation_func,
                      cell,
                      tf_inputs,
                      tf_sequence_length,
                      is_bidirectional,
                      initial_state=None,
                      time_major=None,
                      scope=None):
    if is_bidirectional:
      tf_result = create_rnn_computation_func(
          cell_fw=cell,
          cell_bw=cell,
          inputs=tf_inputs,
          sequence_length=tf_sequence_length,
          dtype=dtypes.float32,
          time_major=time_major,
          scope=scope)
    else:
      tf_result = create_rnn_computation_func(
          cell=cell,
          inputs=tf_inputs,
          sequence_length=tf_sequence_length,
          initial_state=initial_state,
          dtype=dtypes.float32,
          time_major=time_major,
          scope=scope)
    grad = gradients_impl.gradients(tf_result, variables.trainable_variables())
    return {'inference': tf_result, 'grad': grad}

  def _MaybeResetVariables(self, variable_cache, sess, var_list):
    """Possibly resets the variables to a previously seen value."""
    reset_ops = []
    fetches = []
    for var in var_list:
      if var.name in variable_cache:
        reset_ops += [var.assign(variable_cache[var.name])]
      else:
        fetches += [(var.name, var)]
    if reset_ops:
      sess.run(reset_ops)
    if fetches:
      val = sess.run(dict(fetches))
      for n, v in val.items():
        assert n not in variable_cache
        variable_cache[n] = v

  def _RunRnn(self, numpy_inputs, numpy_slen, cell_name, variable_cache,
              is_dynamic, time_major=None, is_bidirectional=False):
    with ops.Graph().as_default() as graph:
      tf_inputs = array_ops.placeholder(
          dtypes.float32, shape=numpy_inputs.shape)
      tf_slen = array_ops.placeholder(dtypes.int32)
      feeds = {tf_inputs: numpy_inputs, tf_slen: numpy_slen}
      cell = self._CreateCell(cell_name)
      if is_dynamic:
        if is_bidirectional:
          fn = rnn_lib.bidirectional_dynamic_rnn
        else:
          fn = rnn_lib.dynamic_rnn
      else:
        if is_bidirectional:
          fn = functional_rnn.bidirectional_functional_rnn
        else:
          fn = functional_rnn.functional_rnn

      fetches = self._CreateRnnGraph(
          fn, cell, tf_inputs, tf_slen, is_bidirectional, time_major=time_major)
      with self.session(graph=graph) as sess:
        sess.run(variables.global_variables_initializer())
        # Note that cell.trainable_variables it not always set.
        self._MaybeResetVariables(variable_cache, sess,
                                  variables.trainable_variables())
        val = sess.run(fetches, feed_dict=feeds)
      graph_def = graph.as_graph_def()
      return graph_def, val

  def testRunLstm(self):
    """Runs a simple LSTM. Does not check output."""
    np_inputs, np_slen = self._CreateInputs()
    var_cache = {}
    graphdef, _ = self._RunRnn(np_inputs, np_slen, 'lstm', var_cache, False)
    logging.info('graphdef: %s', graphdef)
    if self._LSTM_GRAPH_DEF_FILEPATH:
      with open(self._LSTM_GRAPH_DEF_FILEPATH, 'w') as f:
        f.write(str(graphdef))

  def testLstm(self):
    """Checks an LSTM against the reference implementation."""
    np_inputs, np_slen = self._CreateInputs()
    var_cache = {}
    _, func_rnn = self._RunRnn(np_inputs, np_slen, 'lstm', var_cache, False)
    _, dyn_rnn = self._RunRnn(np_inputs, np_slen, 'lstm', var_cache, True)
    self.assertAllClose(dyn_rnn['inference'], func_rnn['inference'])
    self.assertAllClose(dyn_rnn['grad'], func_rnn['grad'])

  def testGru(self):
    """Checks a GRU cell against the reference implementation."""
    np_inputs, np_slen = self._CreateInputs()
    var_cache = {}
    _, func_rnn = self._RunRnn(np_inputs, np_slen, 'gru', var_cache, False)
    _, dyn_rnn = self._RunRnn(np_inputs, np_slen, 'gru', var_cache, True)
    self.assertAllClose(dyn_rnn['inference'], func_rnn['inference'])
    self.assertAllClose(dyn_rnn['grad'], func_rnn['grad'])

  def testStackedLstm(self):
    """Checks a stacked LSTM cell against the reference implementation."""
    np_inputs, np_slen = self._CreateInputs()
    var_cache = {}
    args = [np_inputs, np_slen, 'stacked_lstm', var_cache]
    _, func_rnn = self._RunRnn(*(args + [False]))
    _, dyn_rnn = self._RunRnn(*(args + [True]))
    self.assertAllClose(dyn_rnn['inference'], func_rnn['inference'])
    self.assertAllClose(dyn_rnn['grad'], func_rnn['grad'])

  def testLstmWithTimeMajorInputs(self):
    """Checks an LSTM against the reference implementation, with time_major."""
    time_major = True
    np_inputs, np_slen = self._CreateInputs(time_major=True)
    var_cache = {}
    args = [np_inputs, np_slen, 'lstm', var_cache]
    _, func_rnn = self._RunRnn(*(args + [False]), time_major=time_major)
    _, dyn_rnn = self._RunRnn(*(args + [True]), time_major=time_major)
    self.assertAllClose(dyn_rnn['inference'], func_rnn['inference'])
    self.assertAllClose(dyn_rnn['grad'], func_rnn['grad'])

  def testBidirectionalLstmWithTimeMajorInputs(self):
    """Checks a bi-directional LSTM with time-major inputs."""
    time_major = True
    np_inputs, np_slen = self._CreateInputs(time_major)
    var_cache = {}
    args = [np_inputs, np_slen, 'lstm', var_cache]
    _, func_rnn = self._RunRnn(
        *(args + [False]), time_major=time_major, is_bidirectional=True)
    _, dyn_rnn = self._RunRnn(
        *(args + [True]), time_major=time_major, is_bidirectional=True)
    self.assertAllClose(dyn_rnn['inference'], func_rnn['inference'])
    # TODO(b/112170761): comment out this line after the bug is fixed.
    # self.assertAllClose(dyn_rnn['grad'], func_rnn['grad'])

  def testBidirectionalLstm(self):
    """Checks time-major and batch-major rnn produce consistent results."""
    time_major_inputs, np_slen = self._CreateInputs(True)
    batch_major_inputs = np.transpose(time_major_inputs, [1, 0, 2])
    var_cache = {}
    args = [np_slen, 'lstm', var_cache, False]
    _, time_major_rnn = self._RunRnn(
        *([time_major_inputs] + args), time_major=True, is_bidirectional=True)
    _, batch_major_rnn = self._RunRnn(
        *([batch_major_inputs]+ args), time_major=False, is_bidirectional=True)
    # Convert the batch-major outputs to be time-major before the comparasion.
    outputs, state = batch_major_rnn['inference']
    outputs = [np.transpose(x, [1, 0, 2]) for x in outputs]
    batch_major_rnn['inference'] = [outputs, state]
    self.assertAllClose(time_major_rnn['inference'],
                        batch_major_rnn['inference'])
    self.assertAllClose(time_major_rnn['grad'], batch_major_rnn['grad'])

  def testBidirectionalLstmWithSymmetricInputs(self):
    """Checks a bi-directional LSTM with symmetric inputs.

    time-major and batch-major rnn produce the same result with symmetric
    inputs.
    """
    np_inputs, np_slen = self._CreateSymmetricInputs()
    var_cache = {}
    args = [np_inputs, np_slen, 'lstm', var_cache]
    _, time_major_func_rnn = self._RunRnn(
        *(args + [False]), time_major=True, is_bidirectional=True)
    _, batch_major_func_rnn = self._RunRnn(
        *(args + [False]), time_major=False, is_bidirectional=True)
    _, time_major_dyn_rnn = self._RunRnn(
        *(args + [True]), time_major=True, is_bidirectional=True)
    _, batch_major_dyn_rnn = self._RunRnn(
        *(args + [True]), time_major=False, is_bidirectional=True)
    self.assertAllClose(time_major_func_rnn['inference'],
                        batch_major_func_rnn['inference'])
    self.assertAllClose(time_major_func_rnn['grad'],
                        batch_major_func_rnn['grad'])
    self.assertAllClose(time_major_dyn_rnn['inference'],
                        batch_major_dyn_rnn['inference'])
    self.assertAllClose(time_major_dyn_rnn['grad'], batch_major_dyn_rnn['grad'])
    self.assertAllClose(time_major_func_rnn['inference'],
                        batch_major_dyn_rnn['inference'])
    self.assertAllClose(time_major_func_rnn['grad'],
                        batch_major_dyn_rnn['grad'])


if __name__ == '__main__':
  test_lib.main()
