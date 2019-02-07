# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
from tensorflow.contrib.seq2seq.python.ops import sampler as sampler_py
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


@keras_parameterized.run_all_keras_modes
class DecodeV2RNNTest(keras_parameterized.TestCase, test.TestCase):
  """Tests for DecoderV2."""

  def _testDecodeRNN(self, time_major, maximum_iterations=None):

    sequence_length = [3, 4, 3, 1, 0]
    batch_size = 5
    max_time = 8
    input_depth = 7
    cell_depth = 10
    max_out = max(sequence_length)

    with self.cached_session(use_gpu=True):
      if time_major:
        inputs = np.random.randn(max_time, batch_size,
                                 input_depth).astype(np.float32)
      else:
        inputs = np.random.randn(batch_size, max_time,
                                 input_depth).astype(np.float32)
      input_t = constant_op.constant(inputs)
      cell = rnn_cell.LSTMCell(cell_depth)
      sampler = sampler_py.TrainingSampler(time_major=time_major)
      my_decoder = basic_decoder.BasicDecoderV2(
          cell=cell,
          sampler=sampler,
          output_time_major=time_major,
          maximum_iterations=maximum_iterations)

      initial_state = cell.zero_state(
          dtype=dtypes.float32, batch_size=batch_size)
      (final_outputs, unused_final_state, final_sequence_length) = my_decoder(
          input_t, initial_state=initial_state, sequence_length=sequence_length)

      def _t(shape):
        if time_major:
          return (shape[1], shape[0]) + shape[2:]
        return shape

      if not context.executing_eagerly():
        self.assertEqual((batch_size,),
                         tuple(final_sequence_length.get_shape().as_list()))
        self.assertEqual(
            _t((batch_size, None, cell_depth)),
            tuple(final_outputs.rnn_output.get_shape().as_list()))
        self.assertEqual(
            _t((batch_size, None)),
            tuple(final_outputs.sample_id.get_shape().as_list()))

      self.evaluate(variables.global_variables_initializer())
      final_outputs = self.evaluate(final_outputs)
      final_sequence_length = self.evaluate(final_sequence_length)

      # Mostly a smoke test
      time_steps = max_out
      expected_length = sequence_length
      if maximum_iterations is not None:
        time_steps = min(max_out, maximum_iterations)
        expected_length = [min(x, maximum_iterations) for x in expected_length]
      if context.executing_eagerly() and maximum_iterations != 0:
        # Only check the shape of output when maximum_iterations > 0, see
        # b/123431432 for more details.
        self.assertEqual(
            _t((batch_size, time_steps, cell_depth)),
            final_outputs.rnn_output.shape)
        self.assertEqual(
            _t((batch_size, time_steps)), final_outputs.sample_id.shape)
      self.assertItemsEqual(expected_length, final_sequence_length)

  def testDynamicDecodeRNNBatchMajor(self):
    self._testDecodeRNN(time_major=False)

  def testDynamicDecodeRNNTimeMajor(self):
    self._testDecodeRNN(time_major=True)

  def testDynamicDecodeRNNZeroMaxIters(self):
    self._testDecodeRNN(time_major=True, maximum_iterations=0)

  def testDynamicDecodeRNNOneMaxIter(self):
    self._testDecodeRNN(time_major=True, maximum_iterations=1)

  def _testDynamicDecodeRNNWithTrainingHelperMatchesDynamicRNN(
      self, use_sequence_length):
    sequence_length = [3, 4, 3, 1, 0]
    batch_size = 5
    max_time = 8
    input_depth = 7
    cell_depth = 10
    max_out = max(sequence_length)

    with self.cached_session(use_gpu=True):
      inputs = np.random.randn(batch_size, max_time,
                               input_depth).astype(np.float32)
      inputs = constant_op.constant(inputs)

      cell = rnn_cell.LSTMCell(cell_depth)
      zero_state = cell.zero_state(dtype=dtypes.float32, batch_size=batch_size)
      sampler = sampler_py.TrainingSampler()
      my_decoder = basic_decoder.BasicDecoderV2(
          cell=cell, sampler=sampler, impute_finished=use_sequence_length)

      final_decoder_outputs, final_decoder_state, _ = my_decoder(
          inputs, initial_state=zero_state, sequence_length=sequence_length)

      final_rnn_outputs, final_rnn_state = rnn.dynamic_rnn(
          cell,
          inputs,
          sequence_length=sequence_length if use_sequence_length else None,
          initial_state=zero_state)

      self.evaluate(variables.global_variables_initializer())
      eval_result = self.evaluate({
          "final_decoder_outputs": final_decoder_outputs,
          "final_decoder_state": final_decoder_state,
          "final_rnn_outputs": final_rnn_outputs,
          "final_rnn_state": final_rnn_state
      })

      # Decoder only runs out to max_out; ensure values are identical
      # to dynamic_rnn, which also zeros out outputs and passes along state.
      self.assertAllClose(eval_result["final_decoder_outputs"].rnn_output,
                          eval_result["final_rnn_outputs"][:, 0:max_out, :])
      if use_sequence_length:
        self.assertAllClose(eval_result["final_decoder_state"],
                            eval_result["final_rnn_state"])

  def testDynamicDecodeRNNWithTrainingHelperMatchesDynamicRNNWithSeqLen(self):
    self._testDynamicDecodeRNNWithTrainingHelperMatchesDynamicRNN(
        use_sequence_length=True)

  def testDynamicDecodeRNNWithTrainingHelperMatchesDynamicRNNNoSeqLen(self):
    self._testDynamicDecodeRNNWithTrainingHelperMatchesDynamicRNN(
        use_sequence_length=False)

if __name__ == "__main__":
  test.main()
