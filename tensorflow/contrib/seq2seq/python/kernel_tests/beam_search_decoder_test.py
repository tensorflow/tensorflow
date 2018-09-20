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
"""Tests for contrib.seq2seq.python.seq2seq.beam_search_decoder."""
# pylint: disable=unused-import,g-bad-import-order
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# pylint: enable=unused-import

import numpy as np

from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops
from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

# pylint: enable=g-import-not-at-top


class TestGatherTree(test.TestCase):
  """Tests the gather_tree function."""

  def test_gather_tree(self):
    # (max_time = 3, batch_size = 2, beam_width = 3)

    # create (batch_size, max_time, beam_width) matrix and transpose it
    predicted_ids = np.array(
        [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[2, 3, 4], [5, 6, 7], [8, 9, 10]]],
        dtype=np.int32).transpose([1, 0, 2])
    parent_ids = np.array(
        [[[0, 0, 0], [0, 1, 1], [2, 1, 2]], [[0, 0, 0], [1, 2, 0], [2, 1, 1]]],
        dtype=np.int32).transpose([1, 0, 2])

    # sequence_lengths is shaped (batch_size = 3)
    max_sequence_lengths = [3, 3]

    expected_result = np.array([[[2, 2, 2], [6, 5, 6], [7, 8, 9]],
                                [[2, 4, 4], [7, 6, 6],
                                 [8, 9, 10]]]).transpose([1, 0, 2])

    res = beam_search_ops.gather_tree(
        predicted_ids,
        parent_ids,
        max_sequence_lengths=max_sequence_lengths,
        end_token=11)

    with self.cached_session() as sess:
      res_ = sess.run(res)

    self.assertAllEqual(expected_result, res_)

  def _test_gather_tree_from_array(self,
                                   depth_ndims=0,
                                   merged_batch_beam=False):
    array = np.array(
        [[[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0]],
         [[2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 0]]]).transpose([1, 0, 2])
    parent_ids = np.array(
        [[[0, 0, 0], [0, 1, 1], [2, 1, 2], [-1, -1, -1]],
         [[0, 0, 0], [1, 1, 0], [2, 0, 1], [0, 1, 0]]]).transpose([1, 0, 2])
    expected_array = np.array(
        [[[2, 2, 2], [6, 5, 6], [7, 8, 9], [0, 0, 0]],
         [[2, 3, 2], [7, 5, 7], [8, 9, 8], [11, 12, 0]]]).transpose([1, 0, 2])
    sequence_length = [[3, 3, 3], [4, 4, 3]]

    array = ops.convert_to_tensor(
        array, dtype=dtypes.float32)
    parent_ids = ops.convert_to_tensor(
        parent_ids, dtype=dtypes.int32)
    expected_array = ops.convert_to_tensor(
        expected_array, dtype=dtypes.float32)

    max_time = array_ops.shape(array)[0]
    batch_size = array_ops.shape(array)[1]
    beam_width = array_ops.shape(array)[2]

    def _tile_in_depth(tensor):
      # Generate higher rank tensors by concatenating tensor and tensor + 1.
      for _ in range(depth_ndims):
        tensor = array_ops.stack([tensor, tensor + 1], -1)
      return tensor

    if merged_batch_beam:
      array = array_ops.reshape(
          array, [max_time, batch_size * beam_width])
      expected_array = array_ops.reshape(
          expected_array, [max_time, batch_size * beam_width])

    if depth_ndims > 0:
      array = _tile_in_depth(array)
      expected_array = _tile_in_depth(expected_array)

    sorted_array = beam_search_decoder.gather_tree_from_array(
        array, parent_ids, sequence_length)

    with self.cached_session() as sess:
      sorted_array = sess.run(sorted_array)
      expected_array = sess.run(expected_array)
      self.assertAllEqual(expected_array, sorted_array)

  def test_gather_tree_from_array_scalar(self):
    self._test_gather_tree_from_array()

  def test_gather_tree_from_array_1d(self):
    self._test_gather_tree_from_array(depth_ndims=1)

  def test_gather_tree_from_array_1d_with_merged_batch_beam(self):
    self._test_gather_tree_from_array(depth_ndims=1, merged_batch_beam=True)

  def test_gather_tree_from_array_2d(self):
    self._test_gather_tree_from_array(depth_ndims=2)

  def test_gather_tree_from_array_complex_trajectory(self):
    # Max. time = 7, batch = 1, beam = 5.
    array = np.expand_dims(np.array(
        [[[25, 12, 114, 89, 97]],
         [[9, 91, 64, 11, 162]],
         [[34, 34, 34, 34, 34]],
         [[2, 4, 2, 2, 4]],
         [[2, 3, 6, 2, 2]],
         [[2, 2, 2, 3, 2]],
         [[2, 2, 2, 2, 2]]]), -1)
    parent_ids = np.array(
        [[[0, 0, 0, 0, 0]],
         [[0, 0, 0, 0, 0]],
         [[0, 1, 2, 3, 4]],
         [[0, 0, 1, 2, 1]],
         [[0, 1, 1, 2, 3]],
         [[0, 1, 3, 1, 2]],
         [[0, 1, 2, 3, 4]]])
    expected_array = np.expand_dims(np.array(
        [[[25, 25, 25, 25, 25]],
         [[9, 9, 91, 9, 9]],
         [[34, 34, 34, 34, 34]],
         [[2, 4, 2, 4, 4]],
         [[2, 3, 6, 3, 6]],
         [[2, 2, 2, 3, 2]],
         [[2, 2, 2, 2, 2]]]), -1)
    sequence_length = [[4, 6, 4, 7, 6]]

    array = ops.convert_to_tensor(
        array, dtype=dtypes.float32)
    parent_ids = ops.convert_to_tensor(
        parent_ids, dtype=dtypes.int32)
    expected_array = ops.convert_to_tensor(
        expected_array, dtype=dtypes.float32)

    sorted_array = beam_search_decoder.gather_tree_from_array(
        array, parent_ids, sequence_length)

    with self.cached_session() as sess:
      sorted_array, expected_array = sess.run([sorted_array, expected_array])
      self.assertAllEqual(expected_array, sorted_array)


class TestArrayShapeChecks(test.TestCase):

  def _test_array_shape_dynamic_checks(self, static_shape, dynamic_shape,
                                       batch_size, beam_width, is_valid=True):
    t = array_ops.placeholder_with_default(
        np.random.randn(*static_shape).astype(np.float32),
        shape=dynamic_shape)

    batch_size = array_ops.constant(batch_size)
    check_op = beam_search_decoder._check_batch_beam(t, batch_size, beam_width)  # pylint: disable=protected-access

    with self.cached_session() as sess:
      if is_valid:
        sess.run(check_op)
      else:
        with self.assertRaises(errors.InvalidArgumentError):
          sess.run(check_op)

  def test_array_shape_dynamic_checks(self):
    self._test_array_shape_dynamic_checks(
        (8, 4, 5, 10), (None, None, 5, 10), 4, 5, is_valid=True)
    self._test_array_shape_dynamic_checks(
        (8, 20, 10), (None, None, 10), 4, 5, is_valid=True)
    self._test_array_shape_dynamic_checks(
        (8, 21, 10), (None, None, 10), 4, 5, is_valid=False)
    self._test_array_shape_dynamic_checks(
        (8, 4, 6, 10), (None, None, None, 10), 4, 5, is_valid=False)
    self._test_array_shape_dynamic_checks(
        (8, 4), (None, None), 4, 5, is_valid=False)


class TestEosMasking(test.TestCase):
  """Tests EOS masking used in beam search."""

  def test_eos_masking(self):
    probs = constant_op.constant([
        [[-.2, -.2, -.2, -.2, -.2], [-.3, -.3, -.3, 3, 0], [5, 6, 0, 0, 0]],
        [[-.2, -.2, -.2, -.2, 0], [-.3, -.3, -.1, 3, 0], [5, 6, 3, 0, 0]],
    ])

    eos_token = 0
    previously_finished = np.array([[0, 1, 0], [0, 1, 1]], dtype=bool)
    masked = beam_search_decoder._mask_probs(probs, eos_token,
                                             previously_finished)

    with self.cached_session() as sess:
      probs = sess.run(probs)
      masked = sess.run(masked)

      self.assertAllEqual(probs[0][0], masked[0][0])
      self.assertAllEqual(probs[0][2], masked[0][2])
      self.assertAllEqual(probs[1][0], masked[1][0])

      self.assertEqual(masked[0][1][0], 0)
      self.assertEqual(masked[1][1][0], 0)
      self.assertEqual(masked[1][2][0], 0)

      for i in range(1, 5):
        self.assertAllClose(masked[0][1][i], np.finfo('float32').min)
        self.assertAllClose(masked[1][1][i], np.finfo('float32').min)
        self.assertAllClose(masked[1][2][i], np.finfo('float32').min)


class TestBeamStep(test.TestCase):
  """Tests a single step of beam search."""

  def setUp(self):
    super(TestBeamStep, self).setUp()
    self.batch_size = 2
    self.beam_width = 3
    self.vocab_size = 5
    self.end_token = 0
    self.length_penalty_weight = 0.6
    self.coverage_penalty_weight = 0.0

  def test_step(self):
    dummy_cell_state = array_ops.zeros([self.batch_size, self.beam_width])
    beam_state = beam_search_decoder.BeamSearchDecoderState(
        cell_state=dummy_cell_state,
        log_probs=nn_ops.log_softmax(
            array_ops.ones([self.batch_size, self.beam_width])),
        lengths=constant_op.constant(
            2, shape=[self.batch_size, self.beam_width], dtype=dtypes.int64),
        finished=array_ops.zeros(
            [self.batch_size, self.beam_width], dtype=dtypes.bool),
        accumulated_attention_probs=())

    logits_ = np.full([self.batch_size, self.beam_width, self.vocab_size],
                      0.0001)
    logits_[0, 0, 2] = 1.9
    logits_[0, 0, 3] = 2.1
    logits_[0, 1, 3] = 3.1
    logits_[0, 1, 4] = 0.9
    logits_[1, 0, 1] = 0.5
    logits_[1, 1, 2] = 2.7
    logits_[1, 2, 2] = 10.0
    logits_[1, 2, 3] = 0.2
    logits = ops.convert_to_tensor(logits_, dtype=dtypes.float32)
    log_probs = nn_ops.log_softmax(logits)

    outputs, next_beam_state = beam_search_decoder._beam_search_step(
        time=2,
        logits=logits,
        next_cell_state=dummy_cell_state,
        beam_state=beam_state,
        batch_size=ops.convert_to_tensor(self.batch_size),
        beam_width=self.beam_width,
        end_token=self.end_token,
        length_penalty_weight=self.length_penalty_weight,
        coverage_penalty_weight=self.coverage_penalty_weight)

    with self.cached_session() as sess:
      outputs_, next_state_, state_, log_probs_ = sess.run(
          [outputs, next_beam_state, beam_state, log_probs])

    self.assertAllEqual(outputs_.predicted_ids, [[3, 3, 2], [2, 2, 1]])
    self.assertAllEqual(outputs_.parent_ids, [[1, 0, 0], [2, 1, 0]])
    self.assertAllEqual(next_state_.lengths, [[3, 3, 3], [3, 3, 3]])
    self.assertAllEqual(next_state_.finished,
                        [[False, False, False], [False, False, False]])

    expected_log_probs = []
    expected_log_probs.append(state_.log_probs[0][[1, 0, 0]])
    expected_log_probs.append(state_.log_probs[1][[2, 1, 0]])  # 0 --> 1
    expected_log_probs[0][0] += log_probs_[0, 1, 3]
    expected_log_probs[0][1] += log_probs_[0, 0, 3]
    expected_log_probs[0][2] += log_probs_[0, 0, 2]
    expected_log_probs[1][0] += log_probs_[1, 2, 2]
    expected_log_probs[1][1] += log_probs_[1, 1, 2]
    expected_log_probs[1][2] += log_probs_[1, 0, 1]
    self.assertAllEqual(next_state_.log_probs, expected_log_probs)

  def test_step_with_eos(self):
    dummy_cell_state = array_ops.zeros([self.batch_size, self.beam_width])
    beam_state = beam_search_decoder.BeamSearchDecoderState(
        cell_state=dummy_cell_state,
        log_probs=nn_ops.log_softmax(
            array_ops.ones([self.batch_size, self.beam_width])),
        lengths=ops.convert_to_tensor(
            [[2, 1, 2], [2, 2, 1]], dtype=dtypes.int64),
        finished=ops.convert_to_tensor(
            [[False, True, False], [False, False, True]], dtype=dtypes.bool),
        accumulated_attention_probs=())

    logits_ = np.full([self.batch_size, self.beam_width, self.vocab_size],
                      0.0001)
    logits_[0, 0, 2] = 1.9
    logits_[0, 0, 3] = 2.1
    logits_[0, 1, 3] = 3.1
    logits_[0, 1, 4] = 0.9
    logits_[1, 0, 1] = 0.5
    logits_[1, 1, 2] = 5.7  # why does this not work when it's 2.7?
    logits_[1, 2, 2] = 1.0
    logits_[1, 2, 3] = 0.2
    logits = ops.convert_to_tensor(logits_, dtype=dtypes.float32)
    log_probs = nn_ops.log_softmax(logits)

    outputs, next_beam_state = beam_search_decoder._beam_search_step(
        time=2,
        logits=logits,
        next_cell_state=dummy_cell_state,
        beam_state=beam_state,
        batch_size=ops.convert_to_tensor(self.batch_size),
        beam_width=self.beam_width,
        end_token=self.end_token,
        length_penalty_weight=self.length_penalty_weight,
        coverage_penalty_weight=self.coverage_penalty_weight)

    with self.cached_session() as sess:
      outputs_, next_state_, state_, log_probs_ = sess.run(
          [outputs, next_beam_state, beam_state, log_probs])

    self.assertAllEqual(outputs_.parent_ids, [[1, 0, 0], [1, 2, 0]])
    self.assertAllEqual(outputs_.predicted_ids, [[0, 3, 2], [2, 0, 1]])
    self.assertAllEqual(next_state_.lengths, [[1, 3, 3], [3, 1, 3]])
    self.assertAllEqual(next_state_.finished,
                        [[True, False, False], [False, True, False]])

    expected_log_probs = []
    expected_log_probs.append(state_.log_probs[0][[1, 0, 0]])
    expected_log_probs.append(state_.log_probs[1][[1, 2, 0]])
    expected_log_probs[0][1] += log_probs_[0, 0, 3]
    expected_log_probs[0][2] += log_probs_[0, 0, 2]
    expected_log_probs[1][0] += log_probs_[1, 1, 2]
    expected_log_probs[1][2] += log_probs_[1, 0, 1]
    self.assertAllEqual(next_state_.log_probs, expected_log_probs)


class TestLargeBeamStep(test.TestCase):
  """Tests large beam step.

  Tests a single step of beam search in such case that beam size is larger than
  vocabulary size.
  """

  def setUp(self):
    super(TestLargeBeamStep, self).setUp()
    self.batch_size = 2
    self.beam_width = 8
    self.vocab_size = 5
    self.end_token = 0
    self.length_penalty_weight = 0.6
    self.coverage_penalty_weight = 0.0

  def test_step(self):

    def get_probs():
      """this simulates the initialize method in BeamSearchDecoder."""
      log_prob_mask = array_ops.one_hot(
          array_ops.zeros([self.batch_size], dtype=dtypes.int32),
          depth=self.beam_width,
          on_value=True,
          off_value=False,
          dtype=dtypes.bool)

      log_prob_zeros = array_ops.zeros(
          [self.batch_size, self.beam_width], dtype=dtypes.float32)
      log_prob_neg_inf = array_ops.ones(
          [self.batch_size, self.beam_width], dtype=dtypes.float32) * -np.Inf

      log_probs = array_ops.where(log_prob_mask, log_prob_zeros,
                                  log_prob_neg_inf)
      return log_probs

    log_probs = get_probs()
    dummy_cell_state = array_ops.zeros([self.batch_size, self.beam_width])

    # pylint: disable=invalid-name
    _finished = array_ops.one_hot(
        array_ops.zeros([self.batch_size], dtype=dtypes.int32),
        depth=self.beam_width,
        on_value=False,
        off_value=True,
        dtype=dtypes.bool)
    _lengths = np.zeros([self.batch_size, self.beam_width], dtype=np.int64)
    _lengths[:, 0] = 2
    _lengths = constant_op.constant(_lengths, dtype=dtypes.int64)

    beam_state = beam_search_decoder.BeamSearchDecoderState(
        cell_state=dummy_cell_state,
        log_probs=log_probs,
        lengths=_lengths,
        finished=_finished,
        accumulated_attention_probs=())

    logits_ = np.full([self.batch_size, self.beam_width, self.vocab_size],
                      0.0001)
    logits_[0, 0, 2] = 1.9
    logits_[0, 0, 3] = 2.1
    logits_[0, 1, 3] = 3.1
    logits_[0, 1, 4] = 0.9
    logits_[1, 0, 1] = 0.5
    logits_[1, 1, 2] = 2.7
    logits_[1, 2, 2] = 10.0
    logits_[1, 2, 3] = 0.2
    logits = constant_op.constant(logits_, dtype=dtypes.float32)
    log_probs = nn_ops.log_softmax(logits)

    outputs, next_beam_state = beam_search_decoder._beam_search_step(
        time=2,
        logits=logits,
        next_cell_state=dummy_cell_state,
        beam_state=beam_state,
        batch_size=ops.convert_to_tensor(self.batch_size),
        beam_width=self.beam_width,
        end_token=self.end_token,
        length_penalty_weight=self.length_penalty_weight,
        coverage_penalty_weight=self.coverage_penalty_weight)

    with self.cached_session() as sess:
      outputs_, next_state_, _, _ = sess.run(
          [outputs, next_beam_state, beam_state, log_probs])

    self.assertEqual(outputs_.predicted_ids[0, 0], 3)
    self.assertEqual(outputs_.predicted_ids[0, 1], 2)
    self.assertEqual(outputs_.predicted_ids[1, 0], 1)
    neg_inf = -np.Inf
    self.assertAllEqual(
        next_state_.log_probs[:, -3:],
        [[neg_inf, neg_inf, neg_inf], [neg_inf, neg_inf, neg_inf]])
    self.assertEqual((next_state_.log_probs[:, :-3] > neg_inf).all(), True)
    self.assertEqual((next_state_.lengths[:, :-3] > 0).all(), True)
    self.assertAllEqual(next_state_.lengths[:, -3:], [[0, 0, 0], [0, 0, 0]])


class BeamSearchDecoderTest(test.TestCase):

  def _testDynamicDecodeRNN(self, time_major, has_attention,
                            with_alignment_history=False):
    encoder_sequence_length = np.array([3, 2, 3, 1, 1])
    decoder_sequence_length = np.array([2, 0, 1, 2, 3])
    batch_size = 5
    decoder_max_time = 4
    input_depth = 7
    cell_depth = 9
    attention_depth = 6
    vocab_size = 20
    end_token = vocab_size - 1
    start_token = 0
    embedding_dim = 50
    max_out = max(decoder_sequence_length)
    output_layer = layers_core.Dense(vocab_size, use_bias=True, activation=None)
    beam_width = 3

    with self.cached_session() as sess:
      batch_size_tensor = constant_op.constant(batch_size)
      embedding = np.random.randn(vocab_size, embedding_dim).astype(np.float32)
      cell = rnn_cell.LSTMCell(cell_depth)
      initial_state = cell.zero_state(batch_size, dtypes.float32)
      coverage_penalty_weight = 0.0
      if has_attention:
        coverage_penalty_weight = 0.2
        inputs = array_ops.placeholder_with_default(
            np.random.randn(batch_size, decoder_max_time, input_depth).astype(
                np.float32),
            shape=(None, None, input_depth))
        tiled_inputs = beam_search_decoder.tile_batch(
            inputs, multiplier=beam_width)
        tiled_sequence_length = beam_search_decoder.tile_batch(
            encoder_sequence_length, multiplier=beam_width)
        attention_mechanism = attention_wrapper.BahdanauAttention(
            num_units=attention_depth,
            memory=tiled_inputs,
            memory_sequence_length=tiled_sequence_length)
        initial_state = beam_search_decoder.tile_batch(
            initial_state, multiplier=beam_width)
        cell = attention_wrapper.AttentionWrapper(
            cell=cell,
            attention_mechanism=attention_mechanism,
            attention_layer_size=attention_depth,
            alignment_history=with_alignment_history)
      cell_state = cell.zero_state(
          dtype=dtypes.float32, batch_size=batch_size_tensor * beam_width)
      if has_attention:
        cell_state = cell_state.clone(cell_state=initial_state)
      bsd = beam_search_decoder.BeamSearchDecoder(
          cell=cell,
          embedding=embedding,
          start_tokens=array_ops.fill([batch_size_tensor], start_token),
          end_token=end_token,
          initial_state=cell_state,
          beam_width=beam_width,
          output_layer=output_layer,
          length_penalty_weight=0.0,
          coverage_penalty_weight=coverage_penalty_weight)

      final_outputs, final_state, final_sequence_lengths = (
          decoder.dynamic_decode(
              bsd, output_time_major=time_major, maximum_iterations=max_out))

      def _t(shape):
        if time_major:
          return (shape[1], shape[0]) + shape[2:]
        return shape

      self.assertTrue(
          isinstance(final_outputs,
                     beam_search_decoder.FinalBeamSearchDecoderOutput))
      self.assertTrue(
          isinstance(final_state, beam_search_decoder.BeamSearchDecoderState))

      beam_search_decoder_output = final_outputs.beam_search_decoder_output
      self.assertEqual(
          _t((batch_size, None, beam_width)),
          tuple(beam_search_decoder_output.scores.get_shape().as_list()))
      self.assertEqual(
          _t((batch_size, None, beam_width)),
          tuple(final_outputs.predicted_ids.get_shape().as_list()))

      sess.run(variables.global_variables_initializer())
      sess_results = sess.run({
          'final_outputs': final_outputs,
          'final_state': final_state,
          'final_sequence_lengths': final_sequence_lengths
      })

      max_sequence_length = np.max(sess_results['final_sequence_lengths'])

      # A smoke test
      self.assertEqual(
          _t((batch_size, max_sequence_length, beam_width)),
          sess_results['final_outputs'].beam_search_decoder_output.scores.shape)
      self.assertEqual(
          _t((batch_size, max_sequence_length, beam_width)), sess_results[
              'final_outputs'].beam_search_decoder_output.predicted_ids.shape)

  def testDynamicDecodeRNNBatchMajorNoAttention(self):
    self._testDynamicDecodeRNN(time_major=False, has_attention=False)

  def testDynamicDecodeRNNBatchMajorYesAttention(self):
    self._testDynamicDecodeRNN(time_major=False, has_attention=True)

  def testDynamicDecodeRNNBatchMajorYesAttentionWithAlignmentHistory(self):
    self._testDynamicDecodeRNN(
        time_major=False,
        has_attention=True,
        with_alignment_history=True)


if __name__ == '__main__':
  test.main()
