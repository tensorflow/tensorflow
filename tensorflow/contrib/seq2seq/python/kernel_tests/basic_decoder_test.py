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
"""Tests for contrib.seq2seq.python.seq2seq.basic_decoder."""
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
from tensorflow.contrib.seq2seq.python.ops import helper as helper_py
from tensorflow.contrib.seq2seq.python.ops import basic_decoder
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
# pylint: enable=g-import-not-at-top


class BasicDecoderTest(test.TestCase):

  def _testStepWithTrainingHelper(self, use_output_layer):
    sequence_length = [3, 4, 3, 1, 0]
    batch_size = 5
    max_time = 8
    input_depth = 7
    cell_depth = 10
    output_layer_depth = 3

    with self.test_session() as sess:
      inputs = np.random.randn(batch_size, max_time,
                               input_depth).astype(np.float32)
      cell = core_rnn_cell.LSTMCell(cell_depth)
      helper = helper_py.TrainingHelper(
          inputs, sequence_length, time_major=False)
      if use_output_layer:
        output_layer = layers_core.Dense(output_layer_depth, use_bias=False)
        expected_output_depth = output_layer_depth
      else:
        output_layer = None
        expected_output_depth = cell_depth
      my_decoder = basic_decoder.BasicDecoder(
          cell=cell,
          helper=helper,
          initial_state=cell.zero_state(
              dtype=dtypes.float32, batch_size=batch_size),
          output_layer=output_layer)
      output_size = my_decoder.output_size
      output_dtype = my_decoder.output_dtype
      self.assertEqual(
          basic_decoder.BasicDecoderOutput(expected_output_depth,
                                           tensor_shape.TensorShape([])),
          output_size)
      self.assertEqual(
          basic_decoder.BasicDecoderOutput(dtypes.float32, dtypes.int32),
          output_dtype)

      (first_finished, first_inputs, first_state) = my_decoder.initialize()
      (step_outputs, step_state, step_next_inputs,
       step_finished) = my_decoder.step(
           constant_op.constant(0), first_inputs, first_state)
      batch_size_t = my_decoder.batch_size

      self.assertTrue(isinstance(first_state, core_rnn_cell.LSTMStateTuple))
      self.assertTrue(isinstance(step_state, core_rnn_cell.LSTMStateTuple))
      self.assertTrue(
          isinstance(step_outputs, basic_decoder.BasicDecoderOutput))
      self.assertEqual((batch_size, expected_output_depth),
                       step_outputs[0].get_shape())
      self.assertEqual((batch_size,), step_outputs[1].get_shape())
      self.assertEqual((batch_size, cell_depth), first_state[0].get_shape())
      self.assertEqual((batch_size, cell_depth), first_state[1].get_shape())
      self.assertEqual((batch_size, cell_depth), step_state[0].get_shape())
      self.assertEqual((batch_size, cell_depth), step_state[1].get_shape())

      if use_output_layer:
        # The output layer was accessed
        self.assertEqual(len(output_layer.variables), 1)

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
      self.assertAllEqual(
          np.argmax(sess_results["step_outputs"].rnn_output, -1),
          sess_results["step_outputs"].sample_id)

  def testStepWithTrainingHelperNoOutputLayer(self):
    self._testStepWithTrainingHelper(use_output_layer=False)

  def testStepWithTrainingHelperWithOutputLayer(self):
    self._testStepWithTrainingHelper(use_output_layer=True)

  def testStepWithGreedyEmbeddingHelper(self):
    batch_size = 5
    vocabulary_size = 7
    cell_depth = vocabulary_size  # cell's logits must match vocabulary size
    input_depth = 10
    start_tokens = [0] * batch_size
    end_token = 1

    with self.test_session() as sess:
      embeddings = np.random.randn(vocabulary_size,
                                   input_depth).astype(np.float32)
      cell = core_rnn_cell.LSTMCell(vocabulary_size)
      helper = helper_py.GreedyEmbeddingHelper(embeddings, start_tokens,
                                               end_token)
      my_decoder = basic_decoder.BasicDecoder(
          cell=cell,
          helper=helper,
          initial_state=cell.zero_state(
              dtype=dtypes.float32, batch_size=batch_size))
      output_size = my_decoder.output_size
      output_dtype = my_decoder.output_dtype
      self.assertEqual(
          basic_decoder.BasicDecoderOutput(cell_depth,
                                           tensor_shape.TensorShape([])),
          output_size)
      self.assertEqual(
          basic_decoder.BasicDecoderOutput(dtypes.float32, dtypes.int32),
          output_dtype)

      (first_finished, first_inputs, first_state) = my_decoder.initialize()
      (step_outputs, step_state, step_next_inputs,
       step_finished) = my_decoder.step(
           constant_op.constant(0), first_inputs, first_state)
      batch_size_t = my_decoder.batch_size

      self.assertTrue(isinstance(first_state, core_rnn_cell.LSTMStateTuple))
      self.assertTrue(isinstance(step_state, core_rnn_cell.LSTMStateTuple))
      self.assertTrue(
          isinstance(step_outputs, basic_decoder.BasicDecoderOutput))
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

      expected_sample_ids = np.argmax(
          sess_results["step_outputs"].rnn_output, -1)
      expected_step_finished = (expected_sample_ids == end_token)
      expected_step_next_inputs = embeddings[expected_sample_ids]
      self.assertAllEqual([False, False, False, False, False],
                          sess_results["first_finished"])
      self.assertAllEqual(expected_step_finished, sess_results["step_finished"])
      self.assertAllEqual(expected_sample_ids,
                          sess_results["step_outputs"].sample_id)
      self.assertAllEqual(expected_step_next_inputs,
                          sess_results["step_next_inputs"])

  def testStepWithScheduledEmbeddingTrainingHelper(self):
    sequence_length = [3, 4, 3, 1, 0]
    batch_size = 5
    max_time = 8
    input_depth = 7
    vocabulary_size = 10

    with self.test_session() as sess:
      inputs = np.random.randn(
          batch_size, max_time, input_depth).astype(np.float32)
      embeddings = np.random.randn(
          vocabulary_size, input_depth).astype(np.float32)
      half = constant_op.constant(0.5)
      cell = core_rnn_cell.LSTMCell(vocabulary_size)
      helper = helper_py.ScheduledEmbeddingTrainingHelper(
          inputs=inputs,
          sequence_length=sequence_length,
          embedding=embeddings,
          sampling_probability=half,
          time_major=False)
      my_decoder = basic_decoder.BasicDecoder(
          cell=cell,
          helper=helper,
          initial_state=cell.zero_state(
              dtype=dtypes.float32, batch_size=batch_size))
      output_size = my_decoder.output_size
      output_dtype = my_decoder.output_dtype
      self.assertEqual(
          basic_decoder.BasicDecoderOutput(vocabulary_size,
                                           tensor_shape.TensorShape([])),
          output_size)
      self.assertEqual(
          basic_decoder.BasicDecoderOutput(dtypes.float32, dtypes.int32),
          output_dtype)

      (first_finished, first_inputs, first_state) = my_decoder.initialize()
      (step_outputs, step_state, step_next_inputs,
       step_finished) = my_decoder.step(
           constant_op.constant(0), first_inputs, first_state)
      batch_size_t = my_decoder.batch_size

      self.assertTrue(isinstance(first_state, core_rnn_cell.LSTMStateTuple))
      self.assertTrue(isinstance(step_state, core_rnn_cell.LSTMStateTuple))
      self.assertTrue(
          isinstance(step_outputs, basic_decoder.BasicDecoderOutput))
      self.assertEqual((batch_size, vocabulary_size),
                       step_outputs[0].get_shape())
      self.assertEqual((batch_size,), step_outputs[1].get_shape())
      self.assertEqual((batch_size, vocabulary_size),
                       first_state[0].get_shape())
      self.assertEqual((batch_size, vocabulary_size),
                       first_state[1].get_shape())
      self.assertEqual((batch_size, vocabulary_size),
                       step_state[0].get_shape())
      self.assertEqual((batch_size, vocabulary_size),
                       step_state[1].get_shape())
      self.assertEqual((batch_size, input_depth),
                       step_next_inputs.get_shape())

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
      sample_ids = sess_results["step_outputs"].sample_id
      batch_where_not_sampling = np.where(sample_ids == -1)
      batch_where_sampling = np.where(sample_ids > -1)
      self.assertAllClose(
          sess_results["step_next_inputs"][batch_where_sampling],
          embeddings[sample_ids[batch_where_sampling]])
      self.assertAllClose(
          sess_results["step_next_inputs"][batch_where_not_sampling],
          np.squeeze(inputs[batch_where_not_sampling, 1]))


if __name__ == "__main__":
  test.main()
