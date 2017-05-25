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
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import init_ops
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
      return super(AttentionWrapperTest, self).assertAllClose(x, y, **kwargs)
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
    encoder_sequence_length = [3, 2, 3, 1, 0]
    decoder_sequence_length = [2, 0, 1, 2, 3]
    batch_size = 5
    encoder_max_time = 8
    decoder_max_time = 4
    input_depth = 7
    encoder_output_depth = 10
    cell_depth = 9

    if attention_layer_size is not None:
      attention_depth = attention_layer_size
    else:
      attention_depth = encoder_output_depth

    decoder_inputs = np.random.randn(batch_size, decoder_max_time,
                                     input_depth).astype(np.float32)
    encoder_outputs = np.random.randn(batch_size, encoder_max_time,
                                      encoder_output_depth).astype(np.float32)

    attention_mechanism = create_attention_mechanism(
        num_units=attention_mechanism_depth,
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length)

    with self.test_session(use_gpu=True) as sess:
      with vs.variable_scope(
          'root',
          initializer=init_ops.random_normal_initializer(stddev=0.01, seed=3)):
        cell = rnn_cell.LSTMCell(cell_depth)
        cell = wrapper.AttentionWrapper(
            cell,
            attention_mechanism,
            attention_layer_size=attention_layer_size,
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
        state_alignment_history = final_state.alignment_history.stack()
        # Remove the history from final_state for purposes of the
        # remainder of the tests.
        final_state = final_state._replace(alignment_history=())  # pylint: disable=protected-access
        self.assertEqual((None, batch_size, encoder_max_time),
                         tuple(state_alignment_history.get_shape().as_list()))
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
            shape=(5, 3, 6), dtype=dtype('float32'), mean=-0.00083043973),
        sample_id=ResultSummary(shape=(5, 3), dtype=dtype('int32'), mean=2.0))
    expected_final_state = AttentionWrapperState(
        cell_state=LSTMStateTuple(
            c=ResultSummary(
                shape=(5, 9), dtype=dtype('float32'), mean=-0.0039763632),
            h=ResultSummary(
                shape=(5, 9), dtype=dtype('float32'), mean=-0.0019849765)),
        attention=ResultSummary(
            shape=(5, 6), dtype=dtype('float32'), mean=-0.00081052497),
        time=3,
        alignments=ResultSummary(
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
            shape=(5, 3, 6), dtype=dtype('float32'), mean=-0.00040482997),
        sample_id=ResultSummary(
            shape=(5, 3), dtype=dtype('int32'), mean=1.8666666666666667))
    expected_final_state = AttentionWrapperState(
        cell_state=LSTMStateTuple(
            c=ResultSummary(
                shape=(5, 9), dtype=dtype('float32'), mean=-0.0039785588),
            h=ResultSummary(
                shape=(5, 9), dtype=dtype('float32'), mean=-0.0019861322)),
        attention=ResultSummary(
            shape=(5, 6), dtype=dtype('float32'), mean=-0.00038488387),
        time=3,
        alignments=ResultSummary(
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
            shape=(5, 3, 6), dtype=dtype('float32'), mean=-0.00084602338),
        sample_id=ResultSummary(shape=(5, 3), dtype=dtype('int32'), mean=2.0))
    expected_final_state = AttentionWrapperState(
        cell_state=LSTMStateTuple(
            c=ResultSummary(
                shape=(5, 9), dtype=dtype('float32'), mean=-0.0039764317),
            h=ResultSummary(
                shape=(5, 9), dtype=dtype('float32'), mean=-0.0019850098)),
        attention=ResultSummary(
            shape=(5, 6), dtype=dtype('float32'), mean=-0.00080144603),
        time=3,
        alignments=ResultSummary(
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
            shape=(5, 3, 6), dtype=dtype('float32'), mean=-0.00084602338),
        sample_id=ResultSummary(shape=(5, 3), dtype=dtype('int32'), mean=2.0))
    expected_final_state = AttentionWrapperState(
        cell_state=LSTMStateTuple(
            c=ResultSummary(
                shape=(5, 9), dtype=dtype('float32'), mean=-0.0039764317),
            h=ResultSummary(
                shape=(5, 9), dtype=dtype('float32'), mean=-0.0019850098)),
        attention=ResultSummary(
            shape=(5, 6), dtype=dtype('float32'), mean=-0.00080144603),
        time=3,
        alignments=ResultSummary(
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
            shape=(5, 3, 10), dtype=dtype('float32'), mean=0.019546926),
        sample_id=ResultSummary(
            shape=(5, 3), dtype=dtype('int32'), mean=2.7999999999999998))
    expected_final_state = AttentionWrapperState(
        cell_state=LSTMStateTuple(
            c=ResultSummary(
                shape=(5, 9), dtype=dtype('float32'), mean=-0.0041728448),
            h=ResultSummary(
                shape=(5, 9), dtype=dtype('float32'), mean=-0.002085865)),
        attention=ResultSummary(
            shape=(5, 10), dtype=dtype('float32'), mean=0.019546915),
        time=3,
        alignments=ResultSummary(
            shape=(5, 8), dtype=dtype('float32'), mean=0.125),
        alignment_history=())

    self._testWithAttention(
        create_attention_mechanism,
        expected_final_output,
        expected_final_state,
        attention_layer_size=None,
        name='testNotUseAttentionLayer')


if __name__ == '__main__':
  test.main()
