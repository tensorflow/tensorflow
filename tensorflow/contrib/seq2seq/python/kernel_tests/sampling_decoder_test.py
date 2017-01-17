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
"""Tests for contrib.seq2seq.python.seq2seq.sampling_decoder."""
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
from tensorflow.contrib.seq2seq.python.ops import sampling_decoder
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
# pylint: enable=g-import-not-at-top


class BasicSamplingDecoderTest(test.TestCase):

  def testStepWithBasicTrainingSampler(self):
    sequence_length = [3, 4, 3, 1, 0]
    batch_size = 5
    max_time = 8
    input_depth = 7
    cell_depth = 10

    with self.test_session() as sess:
      inputs = np.random.randn(batch_size, max_time,
                               input_depth).astype(np.float32)
      cell = core_rnn_cell.LSTMCell(cell_depth)
      sampler = sampling_decoder.BasicTrainingSampler(
          inputs, sequence_length, time_major=False)
      my_decoder = sampling_decoder.BasicSamplingDecoder(
          cell=cell,
          sampler=sampler,
          initial_state=cell.zero_state(
              dtype=dtypes.float32, batch_size=batch_size))
      output_size = my_decoder.output_size
      output_dtype = my_decoder.output_dtype
      batch_size_t = my_decoder.batch_size
      self.assertEqual(
          sampling_decoder.SamplingDecoderOutput(cell_depth,
                                                 tensor_shape.TensorShape([])),
          output_size)
      self.assertEqual(
          sampling_decoder.SamplingDecoderOutput(dtypes.float32, dtypes.int32),
          output_dtype)

      (first_finished, first_inputs, first_state) = my_decoder.initialize()
      (step_outputs, step_state, step_next_inputs,
       step_finished) = my_decoder.step(
           constant_op.constant(0), first_inputs, first_state)

      self.assertTrue(isinstance(first_state, core_rnn_cell.LSTMStateTuple))
      self.assertTrue(isinstance(step_state, core_rnn_cell.LSTMStateTuple))
      self.assertTrue(
          isinstance(step_outputs, sampling_decoder.SamplingDecoderOutput))
      self.assertEqual((batch_size, cell_depth), step_outputs[0].get_shape())
      self.assertEqual((batch_size,), step_outputs[1].get_shape())
      self.assertEqual((batch_size, cell_depth), first_state[0].get_shape())
      self.assertEqual((batch_size, cell_depth), first_state[1].get_shape())
      self.assertEqual((batch_size, cell_depth), step_state[0].get_shape())
      self.assertEqual((batch_size, cell_depth), step_state[1].get_shape())

      sess.run(variables.global_variables_initializer())
      sess_results = sess.run({
          "batch_size": batch_size_t,
          "first_finished": first_finished,
          "first_inputs": first_inputs,
          "first_state": first_state,
          "step_outputs": step_outputs,
          "step_state": step_state,
          "step_next_inputs": step_next_inputs,
          "step_finished": step_finished
      })

      self.assertAllEqual([False, False, False, False, True],
                          sess_results["first_finished"])
      self.assertAllEqual([False, False, False, True, True],
                          sess_results["step_finished"])
      self.assertAllEqual([-1] * 5, sess_results["step_outputs"].sample_id)


if __name__ == "__main__":
  test.main()
