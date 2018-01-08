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
"""Tests for contrib.seq2seq.python.ops.attention_wrapper."""
# pylint: disable=unused-import,g-bad-import-order
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# pylint: enable=unused-import

import collections
import functools

import numpy as np

from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper as wrapper
from tensorflow.contrib.seq2seq.python.ops import helper as helper_py
from tensorflow.contrib.seq2seq.python.ops import basic_decoder
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import test
from tensorflow.python.util import nest

# pylint: enable=g-import-not-at-top


# for testing
AttentionWrapperState = wrapper.AttentionWrapperState  # pylint: disable=invalid-name
LSTMStateTuple = rnn_cell.LSTMStateTuple  # pylint: disable=invalid-name
BasicDecoderOutput = basic_decoder.BasicDecoderOutput  # pylint: disable=invalid-name
float32 = np.float32
int32 = np.int32
array = np.array
dtype = np.dtype


class ResultSummary(
    collections.namedtuple('ResultSummary', ('shape', 'dtype', 'mean'))):
  pass


def get_result_summary(x):
  if isinstance(x, np.ndarray):
    return ResultSummary(x.shape, x.dtype, x.mean())
  return x


class AttentionWrapperTest(test.TestCase):

  def assertAllCloseOrEqual(self, x, y, **kwargs):
    if isinstance(x, np.ndarray) or isinstance(x, float):
      return super(AttentionWrapperTest, self).assertAllClose(
          x, y, atol=1e-3, **kwargs)
    else:
      self.assertAllEqual(x, y, **kwargs)

  def testAttentionWrapperState(self):
    num_fields = len(wrapper.AttentionWrapperState._fields)  # pylint: disable=protected-access
    state = wrapper.AttentionWrapperState(*([None] * num_fields))
    new_state = state.clone(time=1)
    self.assertEqual(state.time, None)
    self.assertEqual(new_state.time, 1)

  def _testWithAttention(self,
                         create_attention_mechanism,
                         expected_final_output,
                         expected_final_state,
                         attention_mechanism_depth=3,
                         alignment_history=False,
                         expected_final_alignment_history=None,
                         attention_layer_size=6,
                         name=''):
    self._testWithMaybeMultiAttention(
        is_multi=False,
        create_attention_mechanisms=[create_attention_mechanism],
        expected_final_output=expected_final_output,
        expected_final_state=expected_final_state,
        attention_mechanism_depths=[attention_mechanism_depth],
        alignment_history=alignment_history,
        expected_final_alignment_history=expected_final_alignment_history,
        attention_layer_sizes=[attention_layer_size],
        name=name)

  def _testWithMaybeMultiAttention(self,
                                   is_multi,
                                   create_attention_mechanisms,
                                   expected_final_output,
                                   expected_final_state,
                                   attention_mechanism_depths,
                                   alignment_history=False,
                                   expected_final_alignment_history=None,
                                   attention_layer_sizes=None,
                                   name=''):
    # Allow is_multi to be True with a single mechanism to enable test for
    # passing in a single mechanism in a list.
    assert len(create_attention_mechanisms) == 1 or is_multi
    encoder_sequence_length = [3, 2, 3, 1, 1]
    decoder_sequence_length = [2, 0, 1, 2, 3]
    batch_size = 5
    encoder_max_time = 8
    decoder_max_time = 4
    input_depth = 7
    encoder_output_depth = 10
    cell_depth = 9

    if attention_layer_sizes is None:
      attention_depth = encoder_output_depth * len(create_attention_mechanisms)
    else:
      # Compute sum of attention_layer_sizes. Use encoder_output_depth if None.
      attention_depth = sum([attention_layer_size or encoder_output_depth
                             for attention_layer_size in attention_layer_sizes])

    decoder_inputs = array_ops.placeholder_with_default(
        np.random.randn(batch_size, decoder_max_time,
                        input_depth).astype(np.float32),
        shape=(None, None, input_depth))
    encoder_outputs = array_ops.placeholder_with_default(
        np.random.randn(batch_size, encoder_max_time,
                        encoder_output_depth).astype(np.float32),
        shape=(None, None, encoder_output_depth))

    attention_mechanisms = [
        creator(num_units=depth,
                memory=encoder_outputs,
                memory_sequence_length=encoder_sequence_length)
        for creator, depth in zip(create_attention_mechanisms,
                                  attention_mechanism_depths)]

    with self.test_session(use_gpu=True) as sess:
      with vs.variable_scope(
          'root',
          initializer=init_ops.random_normal_initializer(stddev=0.01, seed=3)):
        cell = rnn_cell.LSTMCell(cell_depth)
        cell = wrapper.AttentionWrapper(
            cell,
            attention_mechanisms if is_multi else attention_mechanisms[0],
            attention_layer_size=(attention_layer_sizes if is_multi
                                  else attention_layer_sizes[0]),
            alignment_history=alignment_history)
        helper = helper_py.TrainingHelper(decoder_inputs,
                                          decoder_sequence_length)
        my_decoder = basic_decoder.BasicDecoder(
            cell=cell,
            helper=helper,
            initial_state=cell.zero_state(
                dtype=dtypes.float32, batch_size=batch_size))

        final_outputs, final_state, _ = decoder.dynamic_decode(my_decoder)

      self.assertTrue(
          isinstance(final_outputs, basic_decoder.BasicDecoderOutput))
      self.assertTrue(
          isinstance(final_state, wrapper.AttentionWrapperState))
      self.assertTrue(
          isinstance(final_state.cell_state, rnn_cell.LSTMStateTuple))

      self.assertEqual((batch_size, None, attention_depth),
                       tuple(final_outputs.rnn_output.get_shape().as_list()))
      self.assertEqual((batch_size, None),
                       tuple(final_outputs.sample_id.get_shape().as_list()))

      self.assertEqual((batch_size, attention_depth),
                       tuple(final_state.attention.get_shape().as_list()))
      self.assertEqual((batch_size, cell_depth),
                       tuple(final_state.cell_state.c.get_shape().as_list()))
      self.assertEqual((batch_size, cell_depth),
                       tuple(final_state.cell_state.h.get_shape().as_list()))

      if alignment_history:
        if is_multi:
          state_alignment_history = []
          for history_array in final_state.alignment_history:
            history = history_array.stack()
            self.assertEqual(
                (None, batch_size, None),
                tuple(history.get_shape().as_list()))
            state_alignment_history.append(history)
          state_alignment_history = tuple(state_alignment_history)
        else:
          state_alignment_history = final_state.alignment_history.stack()
          self.assertEqual(
              (None, batch_size, None),
              tuple(state_alignment_history.get_shape().as_list()))
        # Remove the history from final_state for purposes of the
        # remainder of the tests.
        final_state = final_state._replace(alignment_history=())  # pylint: disable=protected-access
      else:
        state_alignment_history = ()

      sess.run(variables.global_variables_initializer())
      sess_results = sess.run({
          'final_outputs': final_outputs,
          'final_state': final_state,
          'state_alignment_history': state_alignment_history,
      })

      final_output_info = nest.map_structure(get_result_summary,
                                             sess_results['final_outputs'])
      final_state_info = nest.map_structure(get_result_summary,
                                            sess_results['final_state'])
      print(name)
      print('Copy/paste:\nexpected_final_output = %s' % str(final_output_info))
      print('expected_final_state = %s' % str(final_state_info))
      nest.map_structure(self.assertAllCloseOrEqual, expected_final_output,
                         final_output_info)
      nest.map_structure(self.assertAllCloseOrEqual, expected_final_state,
                         final_state_info)
      if alignment_history:  # by default, the wrapper emits attention as output
        final_alignment_history_info = nest.map_structure(
            get_result_summary, sess_results['state_alignment_history'])
        print('expected_final_alignment_history = %s' %
              str(final_alignment_history_info))
        nest.map_structure(
            self.assertAllCloseOrEqual,
            # outputs are batch major but the stacked TensorArray is time major
            expected_final_alignment_history,
            final_alignment_history_info)

  def testBahdanauNotNormalized(self):
    create_attention_mechanism = wrapper.BahdanauAttention

    expected_final_output = BasicDecoderOutput(
        rnn_output=ResultSummary(
            shape=(5, 3, 6), dtype=dtype('float32'), mean=-0.0052250605),
        sample_id=ResultSummary(
            shape=(5, 3), dtype=dtype('int32'), mean=1.4))
    expected_final_state = AttentionWrapperState(
        cell_state=LSTMStateTuple(
            c=ResultSummary(
                shape=(5, 9), dtype=dtype('float32'), mean=-0.0040092287),
            h=ResultSummary(
                shape=(5, 9), dtype=dtype('float32'), mean=-0.0020015112)),
        attention=ResultSummary(
            shape=(5, 6), dtype=dtype('float32'), mean=-0.0052052638),
        time=3,
        alignments=ResultSummary(
            shape=(5, 8), dtype=dtype('float32'), mean=0.125),
        attention_state=ResultSummary(
            shape=(5, 8), dtype=dtype('float32'), mean=0.125),
        alignment_history=())
    expected_final_alignment_history = ResultSummary(
        shape=(3, 5, 8), dtype=dtype('float32'), mean=0.12500001)

    self._testWithAttention(
        create_attention_mechanism,
        expected_final_output,
        expected_final_state,
        alignment_history=True,
        expected_final_alignment_history=expected_final_alignment_history,
        name='testBahdanauNotNormalized')

  def testBahdanauNormalized(self):
    create_attention_mechanism = functools.partial(
        wrapper.BahdanauAttention, normalize=True)

    expected_final_output = BasicDecoderOutput(
        rnn_output=ResultSummary(
            shape=(5, 3, 6), dtype=dtype('float32'), mean=-0.00597103),
        sample_id=ResultSummary(
            shape=(5, 3), dtype=dtype('int32'), mean=1.6))
    expected_final_state = AttentionWrapperState(
        cell_state=LSTMStateTuple(
            c=ResultSummary(
                shape=(5, 9), dtype=dtype('float32'), mean=-0.0040052128),
            h=ResultSummary(
                shape=(5, 9), dtype=dtype('float32'), mean=-0.0019996136)),
        attention=ResultSummary(
            shape=(5, 6), dtype=dtype('float32'), mean=-0.00595117),
        time=3,
        alignments=ResultSummary(
            shape=(5, 8), dtype=dtype('float32'), mean=0.125),
        attention_state=ResultSummary(
            shape=(5, 8), dtype=dtype('float32'), mean=0.125),
        alignment_history=())

    self._testWithAttention(
        create_attention_mechanism,
        expected_final_output,
        expected_final_state,
        name='testBahdanauNormalized')

  def testLuongNotNormalized(self):
    create_attention_mechanism = wrapper.LuongAttention

    expected_final_output = BasicDecoderOutput(
        rnn_output=ResultSummary(
            shape=(5, 3, 6), dtype=dtype('float32'), mean=-0.0052615386),
        sample_id=ResultSummary(
            shape=(5, 3), dtype=dtype('int32'), mean=1.3333333333))
    expected_final_state = AttentionWrapperState(
        cell_state=LSTMStateTuple(
            c=ResultSummary(
                shape=(5, 9), dtype=dtype('float32'), mean=-0.004009536),
            h=ResultSummary(
                shape=(5, 9), dtype=dtype('float32'), mean=-0.0020016613)),
        attention=ResultSummary(
            shape=(5, 6), dtype=dtype('float32'), mean=-0.0051812846),
        time=3,
        alignments=ResultSummary(
            shape=(5, 8), dtype=dtype('float32'), mean=0.125),
        attention_state=ResultSummary(
            shape=(5, 8), dtype=dtype('float32'), mean=0.125),
        alignment_history=())

    self._testWithAttention(
        create_attention_mechanism,
        expected_final_output,
        expected_final_state,
        attention_mechanism_depth=9,
        name='testLuongNotNormalized')

  def testLuongScaled(self):
    create_attention_mechanism = functools.partial(
        wrapper.LuongAttention, scale=True)

    expected_final_output = BasicDecoderOutput(
        rnn_output=ResultSummary(
            shape=(5, 3, 6), dtype=dtype('float32'), mean=-0.0052615386),
        sample_id=ResultSummary(
            shape=(5, 3), dtype=dtype('int32'), mean=1.3333333333333333))
    expected_final_state = AttentionWrapperState(
        cell_state=LSTMStateTuple(
            c=ResultSummary(
                shape=(5, 9), dtype=dtype('float32'), mean=-0.004009536),
            h=ResultSummary(
                shape=(5, 9), dtype=dtype('float32'), mean=-0.0020016613)),
        attention=ResultSummary(
            shape=(5, 6), dtype=dtype('float32'), mean=-0.0051812846),
        time=3,
        alignments=ResultSummary(
            shape=(5, 8), dtype=dtype('float32'), mean=0.125),
        attention_state=ResultSummary(
            shape=(5, 8), dtype=dtype('float32'), mean=0.125),
        alignment_history=())

    self._testWithAttention(
        create_attention_mechanism,
        expected_final_output,
        expected_final_state,
        attention_mechanism_depth=9,
        name='testLuongScaled')

  def testNotUseAttentionLayer(self):
    create_attention_mechanism = wrapper.BahdanauAttention

    expected_final_output = BasicDecoderOutput(
        rnn_output=ResultSummary(
            shape=(5, 3, 10), dtype=dtype('float32'), mean=0.117389656),
        sample_id=ResultSummary(
            shape=(5, 3), dtype=dtype('int32'), mean=4.5999999999999996))
    expected_final_state = AttentionWrapperState(
        cell_state=LSTMStateTuple(
            c=ResultSummary(
                shape=(5, 9), dtype=dtype('float32'), mean=-0.0063607907),
            h=ResultSummary(
                shape=(5, 9), dtype=dtype('float32'), mean=-0.00323448)),
        attention=ResultSummary(
            shape=(5, 10), dtype=dtype('float32'), mean=0.117389656,),
        time=3,
        alignments=ResultSummary(
            shape=(5, 8), dtype=dtype('float32'), mean=0.125),
        attention_state=ResultSummary(
            shape=(5, 8), dtype=dtype('float32'), mean=0.125),
        alignment_history=())

    self._testWithAttention(
        create_attention_mechanism,
        expected_final_output,
        expected_final_state,
        attention_layer_size=None,
        name='testNotUseAttentionLayer')

  def test_safe_cumprod(self):
    # Create some random test input
    test_input = np.random.uniform(size=(10, 20))

    for axis in [0, 1]:
      for exclusive in [True, False]:
        with self.test_session():
          # Compute cumprod with regular tf.cumprod
          cumprod_output = math_ops.cumprod(
              test_input, axis=axis, exclusive=exclusive).eval()
          # Compute cumprod with safe_cumprod
          safe_cumprod_output = wrapper.safe_cumprod(
              test_input, axis=axis, exclusive=exclusive).eval()
        for x, y in zip(cumprod_output.shape, safe_cumprod_output.shape):
          self.assertEqual(x, y)
        for x, y in zip(cumprod_output.flatten(),
                        safe_cumprod_output.flatten()):
          # Use assertAlmostEqual for the actual values due to floating point
          self.assertAlmostEqual(x, y, places=5)

  def test_monotonic_attention(self):
    def monotonic_attention_explicit(p_choose_i, previous_attention):
      """Explicitly compute monotonic attention distribution using numpy."""
      # Base case for recurrence relation
      out = [previous_attention[0]]
      # Explicitly follow the recurrence relation
      for j in range(1, p_choose_i.shape[0]):
        out.append((1 - p_choose_i[j - 1])*out[j - 1] + previous_attention[j])
      return p_choose_i*np.array(out)

    # Generate a random batch of choosing probabilities for seq. len. 20
    p_choose_i = np.random.uniform(size=(10, 20)).astype(np.float32)
    # Generate random previous attention distributions
    previous_attention = np.random.uniform(size=(10, 20)).astype(np.float32)
    previous_attention /= previous_attention.sum(axis=1).reshape((-1, 1))

    # Create the output to test against
    explicit_output = np.array([
        monotonic_attention_explicit(p, a)
        for p, a in zip(p_choose_i, previous_attention)])

    # Compute output with TensorFlow function, for both calculation types
    with self.test_session():
      recursive_output = wrapper.monotonic_attention(
          p_choose_i, previous_attention, 'recursive').eval()

    self.assertEqual(recursive_output.ndim, explicit_output.ndim)
    for x, y in zip(recursive_output.shape, explicit_output.shape):
      self.assertEqual(x, y)
    for x, y in zip(recursive_output.flatten(), explicit_output.flatten()):
      # Use assertAlmostEqual for the actual values due to floating point
      self.assertAlmostEqual(x, y, places=5)

    # Generate new p_choose_i for parallel, which is unstable when p_choose_i[n]
    # is close to 1
    p_choose_i = np.random.uniform(0, 0.9, size=(10, 20)).astype(np.float32)

    # Create new output to test against
    explicit_output = np.array([
        monotonic_attention_explicit(p, a)
        for p, a in zip(p_choose_i, previous_attention)])

    # Compute output with TensorFlow function, for both calculation types
    with self.test_session():
      parallel_output = wrapper.monotonic_attention(
          p_choose_i, previous_attention, 'parallel').eval()

    self.assertEqual(parallel_output.ndim, explicit_output.ndim)
    for x, y in zip(parallel_output.shape, explicit_output.shape):
      self.assertEqual(x, y)
    for x, y in zip(parallel_output.flatten(), explicit_output.flatten()):
      # Use assertAlmostEqual for the actual values due to floating point
      self.assertAlmostEqual(x, y, places=5)

    # Now, test hard mode, where probabilities must be 0 or 1
    p_choose_i = np.random.choice(np.array([0, 1], np.float32), (10, 20))
    previous_attention = np.zeros((10, 20), np.float32)
    # Randomly choose input sequence indices at each timestep
    random_idx = np.random.randint(0, previous_attention.shape[1],
                                   previous_attention.shape[0])
    previous_attention[np.arange(previous_attention.shape[0]), random_idx] = 1

    # Create the output to test against
    explicit_output = np.array([
        monotonic_attention_explicit(p, a)
        for p, a in zip(p_choose_i, previous_attention)])

    # Compute output with TensorFlow function, for both calculation types
    with self.test_session():
      hard_output = wrapper.monotonic_attention(
          # TensorFlow is unhappy when these are not wrapped as tf.constant
          constant_op.constant(p_choose_i),
          constant_op.constant(previous_attention),
          'hard').eval()

    self.assertEqual(hard_output.ndim, explicit_output.ndim)
    for x, y in zip(hard_output.shape, explicit_output.shape):
      self.assertEqual(x, y)
    for x, y in zip(hard_output.flatten(), explicit_output.flatten()):
      # Use assertAlmostEqual for the actual values due to floating point
      self.assertAlmostEqual(x, y, places=5)

    # Now, test recursively computing attention distributions vs. sampling
    def sample(p_choose_i):
      """Generate a sequence of emit-ingest decisions from p_choose_i."""
      output = np.zeros(p_choose_i.shape)
      t_im1 = 0
      for i in range(p_choose_i.shape[0]):
        for j in range(t_im1, p_choose_i.shape[1]):
          if np.random.uniform() <= p_choose_i[i, j]:
            output[i, j] = 1
            t_im1 = j
            break
        else:
          t_im1 = p_choose_i.shape[1]
      return output

    # Now, the first axis is output timestep and second is input timestep
    p_choose_i = np.random.uniform(size=(4, 5)).astype(np.float32)
    # Generate the average of a bunch of samples
    n_samples = 100000
    sampled_output = np.mean(
        [sample(p_choose_i) for _ in range(n_samples)], axis=0)

    # Create initial previous_attention base case
    recursive_output = [np.array([1] + [0]*(p_choose_i.shape[1] - 1),
                                 np.float32)]
    # Compute output with TensorFlow function, for both calculation types
    with self.test_session():
      for j in range(p_choose_i.shape[0]):
        # Compute attention distribution for this output time step
        recursive_output.append(wrapper.monotonic_attention(
            # newaxis is for adding the expected batch dimension
            p_choose_i[j][np.newaxis],
            recursive_output[-1][np.newaxis], 'recursive').eval()[0])
      # Stack together distributions; remove basecase
      recursive_output = np.array(recursive_output[1:])

    self.assertEqual(recursive_output.ndim, sampled_output.ndim)
    for x, y in zip(recursive_output.shape, sampled_output.shape):
      self.assertEqual(x, y)
    for x, y in zip(recursive_output.flatten(), sampled_output.flatten()):
      # Use a very forgiving threshold since we are sampling
      self.assertAlmostEqual(x, y, places=2)

  def testBahdanauMonotonicNotNormalized(self):
    create_attention_mechanism = functools.partial(
        wrapper.BahdanauMonotonicAttention, sigmoid_noise=1.0,
        sigmoid_noise_seed=3)

    expected_final_output = BasicDecoderOutput(
        rnn_output=ResultSummary(
            shape=(5, 3, 6), dtype=dtype('float32'), mean=-0.002122893),
        sample_id=ResultSummary(
            shape=(5, 3), dtype=dtype('int32'), mean=1.7333333333333334))
    expected_final_state = AttentionWrapperState(
        cell_state=LSTMStateTuple(
            c=ResultSummary(
                shape=(5, 9), dtype=dtype('float32'), mean=-0.0040002423),
            h=ResultSummary(
                shape=(5, 9), dtype=dtype('float32'), mean=-0.0019968653)),
        attention=ResultSummary(
            shape=(5, 6), dtype=dtype('float32'), mean=-5.9313523e-05),
        time=3,
        alignments=ResultSummary(
            shape=(5, 8), dtype=dtype('float32'), mean=0.032228071),
        attention_state=ResultSummary(
            shape=(5, 8), dtype=dtype('float32'), mean=0.032228071),
        alignment_history=())
    expected_final_alignment_history = ResultSummary(
        shape=(3, 5, 8), dtype=dtype('float32'), mean=0.050430927)

    self._testWithAttention(
        create_attention_mechanism,
        expected_final_output,
        expected_final_state,
        alignment_history=True,
        expected_final_alignment_history=expected_final_alignment_history,
        name='testBahdanauMonotonicNotNormalized')

  def testBahdanauMonotonicNormalized(self):
    create_attention_mechanism = functools.partial(
        wrapper.BahdanauMonotonicAttention, normalize=True,
        sigmoid_noise=1.0, sigmoid_noise_seed=3)

    expected_final_output = BasicDecoderOutput(
        rnn_output=ResultSummary(
            shape=(5, 3, 6), dtype=dtype('float32'), mean=-0.0025896581),
        sample_id=ResultSummary(
            shape=(5, 3), dtype=dtype('int32'), mean=1.6))
    expected_final_state = AttentionWrapperState(
        cell_state=LSTMStateTuple(
            c=ResultSummary(
                shape=(5, 9), dtype=dtype('float32'), mean=-0.0040013152),
            h=ResultSummary(
                shape=(5, 9), dtype=dtype('float32'), mean=-0.0019973689)),
        attention=ResultSummary(
            shape=(5, 6), dtype=dtype('float32'), mean=-0.00069823361),
        time=3,
        alignments=ResultSummary(
            shape=(5, 8), dtype=dtype('float32'), mean=0.028698336),
        attention_state=ResultSummary(
            shape=(5, 8), dtype=dtype('float32'), mean=0.028698336),
        alignment_history=())
    expected_final_alignment_history = ResultSummary(
        shape=(3, 5, 8), dtype=dtype('float32'), mean=0.04865776002407074)

    self._testWithAttention(
        create_attention_mechanism,
        expected_final_output,
        expected_final_state,
        alignment_history=True,
        expected_final_alignment_history=expected_final_alignment_history,
        name='testBahdanauMonotonicNormalized')

  def testBahdanauMonotonicHard(self):
    # Run attention mechanism with mode='hard', make sure probabilities are hard
    b, t, u, d = 10, 20, 30, 40
    with self.test_session(use_gpu=True) as sess:
      a = wrapper.BahdanauMonotonicAttention(
          d,
          random_ops.random_normal((b, t, u)),
          mode='hard')
      # Just feed previous attention as [1, 0, 0, ...]
      attn, unused_state = a(
          random_ops.random_normal((b, d)), array_ops.one_hot([0]*b, t))
      sess.run(variables.global_variables_initializer())
      attn_out = attn.eval()
      # All values should be 0 or 1
      self.assertTrue(np.all(np.logical_or(attn_out == 0, attn_out == 1)))
      # Sum of distributions should be 0 or 1 (0 when all p_choose_i are 0)
      self.assertTrue(np.all(np.logical_or(attn_out.sum(axis=1) == 1,
                                           attn_out.sum(axis=1) == 0)))

  def testLuongMonotonicNotNormalized(self):
    create_attention_mechanism = functools.partial(
        wrapper.LuongMonotonicAttention, sigmoid_noise=1.0,
        sigmoid_noise_seed=3)

    expected_final_output = BasicDecoderOutput(
        rnn_output=ResultSummary(
            shape=(5, 3, 6), dtype=dtype('float32'), mean=-0.0021257224),
        sample_id=ResultSummary(
            shape=(5, 3), dtype=dtype('int32'), mean=1.7333333333333334))
    expected_final_state = AttentionWrapperState(
        cell_state=LSTMStateTuple(
            c=ResultSummary(
                shape=(5, 9), dtype=dtype('float32'), mean=-0.0040003359),
            h=ResultSummary(
                shape=(5, 9), dtype=dtype('float32'), mean=-0.001996913)),
        attention=ResultSummary(
            shape=(5, 6), dtype=dtype('float32'), mean=-5.2024145e-05),
        time=3,
        alignments=ResultSummary(
            shape=(5, 8), dtype=dtype('float32'), mean=0.032198936),
        attention_state=ResultSummary(
            shape=(5, 8), dtype=dtype('float32'), mean=0.032198936),
        alignment_history=())
    expected_final_alignment_history = ResultSummary(
        shape=(3, 5, 8), dtype=dtype('float32'), mean=0.050387777)

    self._testWithAttention(
        create_attention_mechanism,
        expected_final_output,
        expected_final_state,
        attention_mechanism_depth=9,
        alignment_history=True,
        expected_final_alignment_history=expected_final_alignment_history,
        name='testLuongMonotonicNotNormalized')

  def testLuongMonotonicScaled(self):
    create_attention_mechanism = functools.partial(
        wrapper.LuongMonotonicAttention, scale=True, sigmoid_noise=1.0,
        sigmoid_noise_seed=3)

    expected_final_output = BasicDecoderOutput(
        rnn_output=ResultSummary(
            shape=(5, 3, 6), dtype=dtype('float32'), mean=-0.0021257224),
        sample_id=ResultSummary(
            shape=(5, 3), dtype=dtype('int32'), mean=1.7333333333333334))
    expected_final_state = AttentionWrapperState(
        cell_state=LSTMStateTuple(
            c=ResultSummary(
                shape=(5, 9), dtype=dtype('float32'), mean=-0.0040003359),
            h=ResultSummary(
                shape=(5, 9), dtype=dtype('float32'), mean=-0.001996913)),
        attention=ResultSummary(
            shape=(5, 6), dtype=dtype('float32'), mean=-5.2024145e-05),
        time=3,
        alignments=ResultSummary(
            shape=(5, 8), dtype=dtype('float32'), mean=0.032198936),
        attention_state=ResultSummary(
            shape=(5, 8), dtype=dtype('float32'), mean=0.032198936),
        alignment_history=())
    expected_final_alignment_history = ResultSummary(
        shape=(3, 5, 8), dtype=dtype('float32'), mean=0.050387777)

    self._testWithAttention(
        create_attention_mechanism,
        expected_final_output,
        expected_final_state,
        attention_mechanism_depth=9,
        alignment_history=True,
        expected_final_alignment_history=expected_final_alignment_history,
        name='testLuongMonotonicScaled')

  def testMultiAttention(self):
    create_attention_mechanisms = (
        wrapper.BahdanauAttention, wrapper.LuongAttention)

    expected_final_output = BasicDecoderOutput(
        rnn_output=ResultSummary(
            shape=(5, 3, 7), dtype=dtype('float32'), mean=0.0011709079),
        sample_id=ResultSummary(
            shape=(5, 3), dtype=dtype('int32'), mean=3.2000000000000002))
    expected_final_state = AttentionWrapperState(
        cell_state=LSTMStateTuple(
            c=ResultSummary(
                shape=(5, 9), dtype=dtype('float32'), mean=-0.0038725811),
            h=ResultSummary(
                shape=(5, 9), dtype=dtype('float32'), mean=-0.0019329828)),
        attention=ResultSummary(
            shape=(5, 7), dtype=dtype('float32'), mean=0.001174294),
        time=3,
        alignments=(
            ResultSummary(shape=(5, 8), dtype=dtype('float32'), mean=0.125),
            ResultSummary(shape=(5, 8), dtype=dtype('float32'), mean=0.125)),
        attention_state=(
            ResultSummary(shape=(5, 8), dtype=dtype('float32'), mean=0.125),
            ResultSummary(shape=(5, 8), dtype=dtype('float32'), mean=0.125)),
        alignment_history=())

    expected_final_alignment_history = (
        ResultSummary(shape=(3, 5, 8), dtype=dtype('float32'), mean=0.125),
        ResultSummary(shape=(3, 5, 8), dtype=dtype('float32'), mean=0.125))

    self._testWithMaybeMultiAttention(
        True,
        create_attention_mechanisms,
        expected_final_output,
        expected_final_state,
        attention_mechanism_depths=[9, 9],
        attention_layer_sizes=[3, 4],
        alignment_history=True,
        expected_final_alignment_history=expected_final_alignment_history,
        name='testMultiAttention')

  def testLuongMonotonicHard(self):
    # Run attention mechanism with mode='hard', make sure probabilities are hard
    b, t, u, d = 10, 20, 30, 40
    with self.test_session(use_gpu=True) as sess:
      a = wrapper.LuongMonotonicAttention(
          d,
          random_ops.random_normal((b, t, u)),
          mode='hard')
      # Just feed previous attention as [1, 0, 0, ...]
      attn, unused_state = a(
          random_ops.random_normal((b, d)), array_ops.one_hot([0]*b, t))
      sess.run(variables.global_variables_initializer())
      attn_out = attn.eval()
      # All values should be 0 or 1
      self.assertTrue(np.all(np.logical_or(attn_out == 0, attn_out == 1)))
      # Sum of distributions should be 0 or 1 (0 when all p_choose_i are 0)
      self.assertTrue(np.all(np.logical_or(attn_out.sum(axis=1) == 1,
                                           attn_out.sum(axis=1) == 0)))

  def testMultiAttentionNoAttentionLayer(self):
    create_attention_mechanisms = (
        wrapper.BahdanauAttention, wrapper.LuongAttention)

    expected_final_output = BasicDecoderOutput(
        rnn_output=ResultSummary(
            shape=(5, 3, 20), dtype=dtype('float32'), mean=0.11798714846372604),
        sample_id=ResultSummary(
            shape=(5, 3), dtype=dtype('int32'), mean=7.933333333333334))
    expected_final_state = AttentionWrapperState(
        cell_state=LSTMStateTuple(
            c=ResultSummary(
                shape=(5, 9), dtype=dtype('float32'), mean=-0.0036486709),
            h=ResultSummary(
                shape=(5, 9), dtype=dtype('float32'), mean=-0.0018835809)),
        attention=ResultSummary(
            shape=(5, 20), dtype=dtype('float32'), mean=0.11798714846372604),
        time=3,
        alignments=(
            ResultSummary(shape=(5, 8), dtype=dtype('float32'), mean=0.125),
            ResultSummary(shape=(5, 8), dtype=dtype('float32'), mean=0.125)),
        attention_state=(
            ResultSummary(shape=(5, 8), dtype=dtype('float32'), mean=0.125),
            ResultSummary(shape=(5, 8), dtype=dtype('float32'), mean=0.125)),
        alignment_history=())
    expected_final_alignment_history = (
        ResultSummary(shape=(3, 5, 8), dtype=dtype('float32'), mean=0.125),
        ResultSummary(shape=(3, 5, 8), dtype=dtype('float32'), mean=0.125))

    self._testWithMaybeMultiAttention(
        is_multi=True,
        create_attention_mechanisms=create_attention_mechanisms,
        expected_final_output=expected_final_output,
        expected_final_state=expected_final_state,
        attention_mechanism_depths=[9, 9],
        alignment_history=True,
        expected_final_alignment_history=expected_final_alignment_history,
        name='testMultiAttention')

  def testSingleAttentionAsList(self):
    create_attention_mechanisms = [wrapper.BahdanauAttention]

    expected_final_output = BasicDecoderOutput(
        rnn_output=ResultSummary(
            shape=(5, 3, 3), dtype=dtype('float32'), mean=-0.0098485695),
        sample_id=ResultSummary(
            shape=(5, 3), dtype=dtype('int32'), mean=1.8))
    expected_final_state = AttentionWrapperState(
        cell_state=LSTMStateTuple(
            c=ResultSummary(
                shape=(5, 9), dtype=dtype('float32'), mean=-0.0040023471),
            h=ResultSummary(
                shape=(5, 9), dtype=dtype('float32'), mean=-0.0019979973)),
        attention=ResultSummary(
            shape=(5, 3), dtype=dtype('float32'), mean=-0.0098808752),
        time=3,
        alignments=(
            ResultSummary(shape=(5, 8), dtype=dtype('float32'), mean=0.125),),
        attention_state=(
            ResultSummary(shape=(5, 8), dtype=dtype('float32'), mean=0.125),),
        alignment_history=())

    expected_final_alignment_history = (
        ResultSummary(shape=(3, 5, 8), dtype=dtype('float32'), mean=0.125),)

    self._testWithMaybeMultiAttention(
        is_multi=True,  # pass the AttentionMechanism wrapped in a list
        create_attention_mechanisms=create_attention_mechanisms,
        expected_final_output=expected_final_output,
        expected_final_state=expected_final_state,
        attention_mechanism_depths=[9],
        attention_layer_sizes=[3],
        alignment_history=True,
        expected_final_alignment_history=expected_final_alignment_history,
        name='testMultiAttention')

if __name__ == '__main__':
  test.main()
