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
"""Tests for contrib.seq2seq.python.seq2seq.decoder."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.seq2seq.python.ops import basic_decoder
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.contrib.seq2seq.python.ops import helper as helper_py
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import test


@test_util.run_v1_only("contrib code not supported in TF2.0")
class DynamicDecodeRNNTest(test.TestCase):

  def _testDynamicDecodeRNN(self, time_major, maximum_iterations=None):

    sequence_length = [3, 4, 3, 1, 0]
    batch_size = 5
    max_time = 8
    input_depth = 7
    cell_depth = 10
    max_out = max(sequence_length)

    with self.session(use_gpu=True) as sess:
      if time_major:
        inputs = np.random.randn(max_time, batch_size,
                                 input_depth).astype(np.float32)
      else:
        inputs = np.random.randn(batch_size, max_time,
                                 input_depth).astype(np.float32)
      cell = rnn_cell.LSTMCell(cell_depth)
      helper = helper_py.TrainingHelper(
          inputs, sequence_length, time_major=time_major)
      my_decoder = basic_decoder.BasicDecoder(
          cell=cell,
          helper=helper,
          initial_state=cell.zero_state(
              dtype=dtypes.float32, batch_size=batch_size))

      final_outputs, final_state, final_sequence_length = (
          decoder.dynamic_decode(my_decoder, output_time_major=time_major,
                                 maximum_iterations=maximum_iterations))

      def _t(shape):
        if time_major:
          return (shape[1], shape[0]) + shape[2:]
        return shape

      self.assertTrue(
          isinstance(final_outputs, basic_decoder.BasicDecoderOutput))
      self.assertTrue(isinstance(final_state, rnn_cell.LSTMStateTuple))

      self.assertEqual(
          (batch_size,),
          tuple(final_sequence_length.get_shape().as_list()))
      self.assertEqual(
          _t((batch_size, None, cell_depth)),
          tuple(final_outputs.rnn_output.get_shape().as_list()))
      self.assertEqual(
          _t((batch_size, None)),
          tuple(final_outputs.sample_id.get_shape().as_list()))

      sess.run(variables.global_variables_initializer())
      sess_results = sess.run({
          "final_outputs": final_outputs,
          "final_state": final_state,
          "final_sequence_length": final_sequence_length,
      })

      # Mostly a smoke test
      time_steps = max_out
      expected_length = sequence_length
      if maximum_iterations is not None:
        time_steps = min(max_out, maximum_iterations)
        expected_length = [min(x, maximum_iterations) for x in expected_length]
      self.assertEqual(
          _t((batch_size, time_steps, cell_depth)),
          sess_results["final_outputs"].rnn_output.shape)
      self.assertEqual(
          _t((batch_size, time_steps)),
          sess_results["final_outputs"].sample_id.shape)
      self.assertItemsEqual(expected_length,
                            sess_results["final_sequence_length"])

  def testDynamicDecodeRNNBatchMajor(self):
    self._testDynamicDecodeRNN(time_major=False)

  def testDynamicDecodeRNNTimeMajor(self):
    self._testDynamicDecodeRNN(time_major=True)

  def testDynamicDecodeRNNZeroMaxIters(self):
    self._testDynamicDecodeRNN(time_major=True, maximum_iterations=0)

  def testDynamicDecodeRNNOneMaxIter(self):
    self._testDynamicDecodeRNN(time_major=True, maximum_iterations=1)

  def _testDynamicDecodeRNNWithTrainingHelperMatchesDynamicRNN(
      self, use_sequence_length):
    sequence_length = [3, 4, 3, 1, 0]
    batch_size = 5
    max_time = 8
    input_depth = 7
    cell_depth = 10
    max_out = max(sequence_length)

    with self.session(use_gpu=True) as sess:
      inputs = np.random.randn(batch_size, max_time,
                               input_depth).astype(np.float32)

      cell = rnn_cell.LSTMCell(cell_depth)
      zero_state = cell.zero_state(dtype=dtypes.float32, batch_size=batch_size)
      helper = helper_py.TrainingHelper(inputs, sequence_length)
      my_decoder = basic_decoder.BasicDecoder(
          cell=cell, helper=helper, initial_state=zero_state)

      # Match the variable scope of dynamic_rnn below so we end up
      # using the same variables
      with vs.variable_scope("root") as scope:
        final_decoder_outputs, final_decoder_state, _ = decoder.dynamic_decode(
            my_decoder,
            # impute_finished=True ensures outputs and final state
            # match those of dynamic_rnn called with sequence_length not None
            impute_finished=use_sequence_length,
            scope=scope)

      with vs.variable_scope(scope, reuse=True) as scope:
        final_rnn_outputs, final_rnn_state = rnn.dynamic_rnn(
            cell,
            inputs,
            sequence_length=sequence_length if use_sequence_length else None,
            initial_state=zero_state,
            scope=scope)

      sess.run(variables.global_variables_initializer())
      sess_results = sess.run({
          "final_decoder_outputs": final_decoder_outputs,
          "final_decoder_state": final_decoder_state,
          "final_rnn_outputs": final_rnn_outputs,
          "final_rnn_state": final_rnn_state
      })

      # Decoder only runs out to max_out; ensure values are identical
      # to dynamic_rnn, which also zeros out outputs and passes along state.
      self.assertAllClose(sess_results["final_decoder_outputs"].rnn_output,
                          sess_results["final_rnn_outputs"][:, 0:max_out, :])
      if use_sequence_length:
        self.assertAllClose(sess_results["final_decoder_state"],
                            sess_results["final_rnn_state"])

  def testDynamicDecodeRNNWithTrainingHelperMatchesDynamicRNNWithSeqLen(self):
    self._testDynamicDecodeRNNWithTrainingHelperMatchesDynamicRNN(
        use_sequence_length=True)

  def testDynamicDecodeRNNWithTrainingHelperMatchesDynamicRNNNoSeqLen(self):
    self._testDynamicDecodeRNNWithTrainingHelperMatchesDynamicRNN(
        use_sequence_length=False)

if __name__ == "__main__":
  test.main()
