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

  def _CreateInputs(self):
    inputs = np.random.random([FunctionalRnnTest._BATCH_SIZE,
                               FunctionalRnnTest._TOTAL_TIME,
                               FunctionalRnnTest._INPUT_SIZE])
    # Always leave one time slot empty, to check max_length behavior.
    sequence_length = np.random.randint(
        0, high=FunctionalRnnTest._TOTAL_TIME - 1,
        size=FunctionalRnnTest._BATCH_SIZE,
        dtype=np.int)
    return (inputs, sequence_length)

  def _CreateRnnGraph(self, create_rnn_computation_func, cell, tf_inputs,
                      tf_sequence_length, initial_state=None,
                      time_major=None, scope=None):
    tf_result = create_rnn_computation_func(cell=cell, inputs=tf_inputs,
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
              is_dynamic):
    with ops.Graph().as_default() as graph:
      tf_inputs = array_ops.placeholder(
          dtypes.float32, shape=numpy_inputs.shape)
      tf_slen = array_ops.placeholder(dtypes.int32)
      feeds = {tf_inputs: numpy_inputs, tf_slen: numpy_slen}
      cell = self._CreateCell(cell_name)
      fn = rnn_lib.dynamic_rnn if is_dynamic else functional_rnn.functional_rnn
      fetches = self._CreateRnnGraph(fn, cell, tf_inputs, tf_slen)
      with self.test_session(graph=graph) as sess:
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


if __name__ == '__main__':
  test_lib.main()
