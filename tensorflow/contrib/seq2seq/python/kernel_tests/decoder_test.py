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
# pylint: disable=unused-import,g-bad-import-order
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# pylint: enable=unused-import

import sys

# TODO(jart): #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, "getdlopenflags") and hasattr(sys, "setdlopenflags"):
  import ctypes  # pylint: disable=g-import-not-at-top
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

# pylint: disable=g-import-not-at-top
import numpy as np

from tensorflow.contrib.rnn import core_rnn_cell
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.contrib.seq2seq.python.ops import sampling_decoder
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import rnn
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import test
# pylint: enable=g-import-not-at-top


class DynamicDecodeRNNTest(test.TestCase):

  def _testDynamicDecodeRNN(self, time_major):

    sequence_length = [3, 4, 3, 1, 0]
    batch_size = 5
    max_time = 8
    input_depth = 7
    cell_depth = 10
    max_out = max(sequence_length)

    with self.test_session() as sess:
      if time_major:
        inputs = np.random.randn(max_time, batch_size,
                                 input_depth).astype(np.float32)
      else:
        inputs = np.random.randn(batch_size, max_time,
                                 input_depth).astype(np.float32)
      cell = core_rnn_cell.LSTMCell(cell_depth)
      sampler = sampling_decoder.BasicTrainingSampler(
          inputs, sequence_length, time_major=time_major)
      my_decoder = sampling_decoder.BasicSamplingDecoder(
          cell=cell,
          sampler=sampler,
          initial_state=cell.zero_state(
              dtype=dtypes.float32, batch_size=batch_size))

      final_outputs, final_state = decoder.dynamic_decode_rnn(
          my_decoder, output_time_major=time_major)

      def _t(shape):
        if time_major:
          return (shape[1], shape[0]) + shape[2:]
        return shape

      self.assertTrue(
          isinstance(final_outputs, sampling_decoder.SamplingDecoderOutput))
      self.assertTrue(isinstance(final_state, core_rnn_cell.LSTMStateTuple))

      self.assertEqual(
          _t((batch_size, None, cell_depth)),
          tuple(final_outputs.rnn_output.get_shape().as_list()))
      self.assertEqual(
          _t((batch_size, None)),
          tuple(final_outputs.sample_id.get_shape().as_list()))

      sess.run(variables.global_variables_initializer())
      sess_results = sess.run({
          "final_outputs": final_outputs,
          "final_state": final_state
      })

      self.assertEqual(
          _t((batch_size, max_out, cell_depth)),
          sess_results["final_outputs"].rnn_output.shape)
      self.assertEqual(
          _t((batch_size, max_out)),
          sess_results["final_outputs"].sample_id.shape)

  def testDynamicDecodeRNNBatchMajor(self):
    self._testDynamicDecodeRNN(time_major=False)

  def testDynamicDecodeRNNTimeMajor(self):
    self._testDynamicDecodeRNN(time_major=True)

  def testDynamicDecodeRNNWithBasicTrainingSamplerMatchesDynamicRNN(self):
    sequence_length = [3, 4, 3, 1, 0]
    batch_size = 5
    max_time = 8
    input_depth = 7
    cell_depth = 10
    max_out = max(sequence_length)

    with self.test_session() as sess:
      inputs = np.random.randn(batch_size, max_time,
                               input_depth).astype(np.float32)

      cell = core_rnn_cell.LSTMCell(cell_depth)
      zero_state = cell.zero_state(dtype=dtypes.float32, batch_size=batch_size)
      sampler = sampling_decoder.BasicTrainingSampler(inputs, sequence_length)
      my_decoder = sampling_decoder.BasicSamplingDecoder(
          cell=cell, sampler=sampler, initial_state=zero_state)

      # Match the variable scope of dynamic_rnn below so we end up
      # using the same variables
      with vs.variable_scope("rnn"):
        final_decoder_outputs, final_decoder_state = decoder.dynamic_decode_rnn(
            my_decoder)

      with vs.variable_scope(vs.get_variable_scope(), reuse=True):
        final_rnn_outputs, final_rnn_state = rnn.dynamic_rnn(
            cell,
            inputs,
            sequence_length=sequence_length,
            initial_state=zero_state)

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
      self.assertAllClose(sess_results["final_decoder_state"],
                          sess_results["final_rnn_state"])


if __name__ == "__main__":
  test.main()
