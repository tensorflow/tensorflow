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
"""Tests for contrib.seq2seq.python.ops.attention_wrapper."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl.testing import parameterized
import numpy as np

from tensorflow.contrib.seq2seq.python.ops import attention_wrapper as wrapper
from tensorflow.contrib.seq2seq.python.ops import basic_decoder
from tensorflow.contrib.seq2seq.python.ops import sampler as sampler_py
from tensorflow.python import keras
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.keras import initializers
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.util import nest


@test_util.run_all_in_graph_and_eager_modes
class AttentionMechanismTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(AttentionMechanismTest, self).setUp()
    self.batch = 10
    self.timestep = 5
    self.memory_size = 6
    self.units = 8

    self.memory = np.random.randn(self.batch, self.timestep,
                                  self.memory_size).astype(np.float32)
    self.query = np.random.randn(self.batch, self.units).astype(np.float32)
    self.state = np.random.randn(self.batch, self.timestep).astype(np.float32)

  @parameterized.named_parameters(
      ("luong", wrapper.LuongAttentionV2),
      ("luong_monotonic", wrapper.LuongMonotonicAttentionV2),
      ("bahdanau", wrapper.BahdanauAttentionV2),
      ("bahdanau_monotonic", wrapper.BahdanauMonotonicAttentionV2),
  )
  def test_attention_shape_inference(self, attention_cls):
    attention = attention_cls(self.units, self.memory)
    attention_score = attention([self.query, self.state])
    self.assertLen(attention_score, 2)
    self.assertEqual(attention_score[0].shape, (self.batch, self.timestep))
    self.assertEqual(attention_score[1].shape, (self.batch, self.timestep))

  @parameterized.named_parameters(
      ("luong", wrapper.LuongAttentionV2),
      ("luong_monotonic", wrapper.LuongMonotonicAttentionV2),
      ("bahdanau", wrapper.BahdanauAttentionV2),
      ("bahdanau_monotonic", wrapper.BahdanauMonotonicAttentionV2),
  )
  def test_get_config(self, attention_cls):
    attention = attention_cls(self.units, self.memory)
    config = attention.get_config()

    attention_from_config = attention_cls.from_config(config)
    config_from_clone = attention_from_config.get_config()

    self.assertDictEqual(config, config_from_clone)

  @parameterized.named_parameters(
      ("luong", wrapper.LuongAttentionV2),
      ("luong_monotonic", wrapper.LuongMonotonicAttentionV2),
      ("bahdanau", wrapper.BahdanauAttentionV2),
      ("bahdanau_monotonic", wrapper.BahdanauMonotonicAttentionV2),
  )
  def test_layer_output(self, attention_cls):
    attention = attention_cls(self.units, self.memory)
    score = attention([self.query, self.state])
    self.evaluate(variables.variables_initializer(attention.variables))

    score_val = self.evaluate(score)
    self.assertLen(score_val, 2)
    self.assertEqual(score_val[0].shape, (self.batch, self.timestep))
    self.assertEqual(score_val[1].shape, (self.batch, self.timestep))

  @parameterized.named_parameters(
      ("luong", wrapper.LuongAttentionV2),
      ("luong_monotonic", wrapper.LuongMonotonicAttentionV2),
      ("bahdanau", wrapper.BahdanauAttentionV2),
      ("bahdanau_monotonic", wrapper.BahdanauMonotonicAttentionV2),
  )
  def test_passing_memory_from_call(self, attention_cls):
    attention = attention_cls(self.units, self.memory)
    weights_before_query = attention.get_weights()
    ref_score = attention([self.query, self.state])

    self.evaluate(variables.global_variables_initializer())
    ref_score_val = self.evaluate(ref_score)

    all_weights = attention.get_weights()
    config = attention.get_config()
    # Simulate the twice invocation of calls here.
    attention_from_config = attention_cls.from_config(config)
    attention_from_config.build(self.memory.shape)
    attention_from_config.set_weights(weights_before_query)
    attention_from_config(self.memory, setup_memory=True)
    attention_from_config.build([self.query.shape, self.state.shape])
    attention_from_config.set_weights(all_weights)
    score = attention_from_config([self.query, self.state])

    score_val = self.evaluate(score)
    self.assertAllClose(ref_score_val, score_val)

  @parameterized.named_parameters(
      ("luong", wrapper.LuongAttentionV2),
      ("luong_monotonic", wrapper.LuongMonotonicAttentionV2),
      ("bahdanau", wrapper.BahdanauAttentionV2),
      ("bahdanau_monotonic", wrapper.BahdanauMonotonicAttentionV2),
  )
  def test_save_load_layer(self, attention_cls):
    vocab = 20
    embedding_dim = 6
    inputs = keras.layers.Input(shape=[self.timestep])
    encoder_input = keras.layers.Embedding(
        vocab, embedding_dim, mask_zero=True)(
            inputs)
    encoder_output = keras.layers.LSTM(
        self.memory_size, return_sequences=True)(
            encoder_input)

    attention = attention_cls(self.units, encoder_output)
    query = keras.layers.Input(shape=[self.units])
    state = keras.layers.Input(shape=[self.timestep])

    score = attention([query, state])

    x = np.random.randint(vocab, size=(self.batch, self.timestep))
    x_test = np.random.randint(vocab, size=(self.batch, self.timestep))
    y = np.random.randn(self.batch, self.timestep)
    model = keras.models.Model([inputs, query, state], score)
    model.compile("rmsprop", "mse")
    model.fit([x, self.query, self.state], (y, y))
    y_ref = model.predict_on_batch([x_test, self.query, self.state])

    config = model.get_config()
    weights = model.get_weights()
    loaded_model = keras.models.Model.from_config(
        config, custom_objects={attention_cls.__name__: attention_cls})
    loaded_model.set_weights(weights)

    y = loaded_model.predict_on_batch([x_test, self.query, self.state])

    self.assertAllClose(y_ref, y)

  # TODO(scottzhu): Add tests for model.compile(run_eagerly=True)


class ResultSummary(
    collections.namedtuple("ResultSummary", ("shape", "dtype", "mean"))):
  pass


def get_result_summary(x):
  if isinstance(x, np.ndarray):
    return ResultSummary(x.shape, x.dtype, x.mean())
  return x


@test_util.run_all_in_graph_and_eager_modes
class AttentionWrapperV2Test(test.TestCase, parameterized.TestCase):

  def assertAllCloseOrEqual(self, x, y, **kwargs):
    if isinstance(x, np.ndarray) or isinstance(x, float):
      return super(AttentionWrapperV2Test, self).assertAllClose(
          x, y, atol=1e-3, **kwargs)
    else:
      self.assertAllEqual(x, y, **kwargs)

  def setUp(self):
    super(AttentionWrapperV2Test, self).setUp()
    self.batch = 64
    self.units = 128
    self.encoder_timestep = 10
    self.encoder_dim = 256
    self.decoder_timestep = 12
    self.encoder_outputs = np.random.randn(self.batch, self.encoder_timestep,
                                           self.encoder_dim)
    self.encoder_sequence_length = np.random.randint(
        self.encoder_timestep, size=(self.batch,)).astype(np.int32)
    self.decoder_inputs = np.random.randn(self.batch, self.decoder_timestep,
                                          self.units)
    self.decoder_sequence_length = np.random.randint(
        self.decoder_timestep, size=(self.batch,)).astype(np.int32)

  def _testWithAttention(self,
                         create_attention_mechanism,
                         expected_final_output,
                         expected_final_state,
                         attention_mechanism_depth=3,
                         alignment_history=False,
                         expected_final_alignment_history=None,
                         attention_layer_size=6,
                         attention_layer=None,
                         create_query_layer=False,
                         create_memory_layer=True,
                         create_attention_kwargs=None):
    attention_layer_sizes = ([attention_layer_size]
                             if attention_layer_size is not None else None)
    attention_layers = ([attention_layer]
                        if attention_layer is not None else None)
    self._testWithMaybeMultiAttention(
        is_multi=False,
        create_attention_mechanisms=[create_attention_mechanism],
        expected_final_output=expected_final_output,
        expected_final_state=expected_final_state,
        attention_mechanism_depths=[attention_mechanism_depth],
        alignment_history=alignment_history,
        expected_final_alignment_history=expected_final_alignment_history,
        attention_layer_sizes=attention_layer_sizes,
        attention_layers=attention_layers,
        create_query_layer=create_query_layer,
        create_memory_layer=create_memory_layer,
        create_attention_kwargs=create_attention_kwargs)

  def _testWithMaybeMultiAttention(self,
                                   is_multi,
                                   create_attention_mechanisms,
                                   expected_final_output,
                                   expected_final_state,
                                   attention_mechanism_depths,
                                   alignment_history=False,
                                   expected_final_alignment_history=None,
                                   attention_layer_sizes=None,
                                   attention_layers=None,
                                   create_query_layer=False,
                                   create_memory_layer=True,
                                   create_attention_kwargs=None):
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
    create_attention_kwargs = create_attention_kwargs or {}

    if attention_layer_sizes is not None:
      # Compute sum of attention_layer_sizes. Use encoder_output_depth if None.
      attention_depth = sum(attention_layer_size or encoder_output_depth
                            for attention_layer_size in attention_layer_sizes)
    elif attention_layers is not None:
      # Compute sum of attention_layers output depth.
      attention_depth = sum(
          attention_layer.compute_output_shape(
              [batch_size, cell_depth + encoder_output_depth]).dims[-1].value
          for attention_layer in attention_layers)
    else:
      attention_depth = encoder_output_depth * len(create_attention_mechanisms)

    decoder_inputs = np.random.randn(batch_size, decoder_max_time,
                                     input_depth).astype(np.float32)
    encoder_outputs = np.random.randn(batch_size, encoder_max_time,
                                      encoder_output_depth).astype(np.float32)

    attention_mechanisms = []
    for creator, depth in zip(create_attention_mechanisms,
                              attention_mechanism_depths):
      # Create a memory layer with deterministic initializer to avoid randomness
      # in the test between graph and eager.
      if create_query_layer:
        create_attention_kwargs["query_layer"] = keras.layers.Dense(
            depth, kernel_initializer="ones", use_bias=False)
      if create_memory_layer:
        create_attention_kwargs["memory_layer"] = keras.layers.Dense(
            depth, kernel_initializer="ones", use_bias=False)

      attention_mechanisms.append(
          creator(
              units=depth,
              memory=encoder_outputs,
              memory_sequence_length=encoder_sequence_length,
              **create_attention_kwargs))

    with self.cached_session(use_gpu=True):
      attention_layer_size = attention_layer_sizes
      attention_layer = attention_layers
      if not is_multi:
        if attention_layer_size is not None:
          attention_layer_size = attention_layer_size[0]
        if attention_layer is not None:
          attention_layer = attention_layer[0]
      cell = keras.layers.LSTMCell(cell_depth,
                                   recurrent_activation="sigmoid",
                                   kernel_initializer="ones",
                                   recurrent_initializer="ones")
      cell = wrapper.AttentionWrapper(
          cell,
          attention_mechanisms if is_multi else attention_mechanisms[0],
          attention_layer_size=attention_layer_size,
          alignment_history=alignment_history,
          attention_layer=attention_layer)
      if cell._attention_layers is not None:
        for layer in cell._attention_layers:
          if getattr(layer, "kernel_initializer") is None:
            layer.kernel_initializer = initializers.glorot_uniform(seed=1337)

      sampler = sampler_py.TrainingSampler()
      my_decoder = basic_decoder.BasicDecoderV2(cell=cell, sampler=sampler)
      initial_state = cell.get_initial_state(
          dtype=dtypes.float32, batch_size=batch_size)
      final_outputs, final_state, _ = my_decoder(
          decoder_inputs,
          initial_state=initial_state,
          sequence_length=decoder_sequence_length)

      self.assertIsInstance(final_outputs, basic_decoder.BasicDecoderOutput)
      self.assertIsInstance(final_state, wrapper.AttentionWrapperState)

      expected_time = (
          expected_final_state.time if context.executing_eagerly() else None)
      self.assertEqual((batch_size, expected_time, attention_depth),
                       tuple(final_outputs.rnn_output.get_shape().as_list()))
      self.assertEqual((batch_size, expected_time),
                       tuple(final_outputs.sample_id.get_shape().as_list()))

      self.assertEqual((batch_size, attention_depth),
                       tuple(final_state.attention.get_shape().as_list()))
      self.assertEqual((batch_size, cell_depth),
                       tuple(final_state.cell_state[0].get_shape().as_list()))
      self.assertEqual((batch_size, cell_depth),
                       tuple(final_state.cell_state[1].get_shape().as_list()))

      if alignment_history:
        if is_multi:
          state_alignment_history = []
          for history_array in final_state.alignment_history:
            history = history_array.stack()
            self.assertEqual((expected_time, batch_size, encoder_max_time),
                             tuple(history.get_shape().as_list()))
            state_alignment_history.append(history)
          state_alignment_history = tuple(state_alignment_history)
        else:
          state_alignment_history = final_state.alignment_history.stack()
          self.assertEqual((expected_time, batch_size, encoder_max_time),
                           tuple(state_alignment_history.get_shape().as_list()))
        nest.assert_same_structure(cell.state_size,
                                   cell.zero_state(batch_size, dtypes.float32))
        # Remove the history from final_state for purposes of the
        # remainder of the tests.
        final_state = final_state._replace(alignment_history=())  # pylint: disable=protected-access
      else:
        state_alignment_history = ()

      self.evaluate(variables.global_variables_initializer())
      eval_result = self.evaluate({
          "final_outputs": final_outputs,
          "final_state": final_state,
          "state_alignment_history": state_alignment_history,
      })

      final_output_info = nest.map_structure(get_result_summary,
                                             eval_result["final_outputs"])
      final_state_info = nest.map_structure(get_result_summary,
                                            eval_result["final_state"])
      print("final_output_info: ", final_output_info)
      print("final_state_info: ", final_state_info)

      nest.map_structure(self.assertAllCloseOrEqual, expected_final_output,
                         final_output_info)
      nest.map_structure(self.assertAllCloseOrEqual, expected_final_state,
                         final_state_info)
      if alignment_history:  # by default, the wrapper emits attention as output
        final_alignment_history_info = nest.map_structure(
            get_result_summary, eval_result["state_alignment_history"])
        print("final_alignment_history_info: ", final_alignment_history_info)
        nest.map_structure(
            self.assertAllCloseOrEqual,
            # outputs are batch major but the stacked TensorArray is time major
            expected_final_alignment_history,
            final_alignment_history_info)

  # TODO(b/126893309): reenable np.float16 once the bug is fixed.
  @parameterized.parameters([np.float32, np.float64])
  def testBahdanauNormalizedDType(self, dtype):
    encoder_outputs = self.encoder_outputs.astype(dtype)
    decoder_inputs = self.decoder_inputs.astype(dtype)
    attention_mechanism = wrapper.BahdanauAttentionV2(
        units=self.units,
        memory=encoder_outputs,
        memory_sequence_length=self.encoder_sequence_length,
        normalize=True,
        dtype=dtype)
    cell = keras.layers.LSTMCell(self.units, recurrent_activation="sigmoid")
    cell = wrapper.AttentionWrapper(cell, attention_mechanism)

    sampler = sampler_py.TrainingSampler()
    my_decoder = basic_decoder.BasicDecoderV2(cell=cell, sampler=sampler)

    final_outputs, final_state, _ = my_decoder(
        decoder_inputs,
        initial_state=cell.zero_state(dtype=dtype, batch_size=self.batch),
        sequence_length=self.decoder_sequence_length)
    self.assertIsInstance(final_outputs, basic_decoder.BasicDecoderOutput)
    self.assertEqual(final_outputs.rnn_output.dtype, dtype)
    self.assertIsInstance(final_state, wrapper.AttentionWrapperState)

  # TODO(b/126893309): reenable np.float16 once the bug is fixed.
  @parameterized.parameters([np.float32, np.float64])
  def testLuongScaledDType(self, dtype):
    # Test case for GitHub issue 18099
    encoder_outputs = self.encoder_outputs.astype(dtype)
    decoder_inputs = self.decoder_inputs.astype(dtype)
    attention_mechanism = wrapper.LuongAttentionV2(
        units=self.units,
        memory=encoder_outputs,
        memory_sequence_length=self.encoder_sequence_length,
        scale=True,
        dtype=dtype,
    )
    cell = keras.layers.LSTMCell(self.units, recurrent_activation="sigmoid")
    cell = wrapper.AttentionWrapper(cell, attention_mechanism)

    sampler = sampler_py.TrainingSampler()
    my_decoder = basic_decoder.BasicDecoderV2(cell=cell, sampler=sampler)

    final_outputs, final_state, _ = my_decoder(
        decoder_inputs,
        initial_state=cell.zero_state(dtype=dtype, batch_size=self.batch),
        sequence_length=self.decoder_sequence_length)
    self.assertIsInstance(final_outputs, basic_decoder.BasicDecoderOutput)
    self.assertEqual(final_outputs.rnn_output.dtype, dtype)
    self.assertIsInstance(final_state, wrapper.AttentionWrapperState)

  def testBahdanauNotNormalized(self):
    create_attention_mechanism = wrapper.BahdanauAttentionV2
    create_attention_kwargs = {"kernel_initializer": "ones"}
    expected_final_output = basic_decoder.BasicDecoderOutput(
        rnn_output=ResultSummary(
            shape=(5, 3, 6), dtype=np.dtype(np.float32), mean=0.051747426),
        sample_id=ResultSummary(
            shape=(5, 3), dtype=np.dtype(np.int32), mean=3.33333333))
    expected_final_state = wrapper.AttentionWrapperState(
        cell_state=[
            ResultSummary(
                shape=(5, 9), dtype=np.dtype(np.float32), mean=0.44189346),
            ResultSummary(
                shape=(5, 9), dtype=np.dtype(np.float32), mean=0.65429491)],
        attention=ResultSummary(
            shape=(5, 6), dtype=np.dtype(np.float32), mean=0.073610783),
        time=3,
        alignments=ResultSummary(
            shape=(5, 8), dtype=np.dtype(np.float32), mean=0.125),
        attention_state=ResultSummary(
            shape=(5, 8), dtype=np.dtype(np.float32), mean=0.125),
        alignment_history=())
    expected_final_alignment_history = ResultSummary(
        shape=(3, 5, 8), dtype=np.dtype(np.float32), mean=0.125)

    self._testWithAttention(
        create_attention_mechanism,
        expected_final_output,
        expected_final_state,
        alignment_history=True,
        create_query_layer=True,
        expected_final_alignment_history=expected_final_alignment_history,
        create_attention_kwargs=create_attention_kwargs)

  def testBahdanauNormalized(self):
    create_attention_mechanism = wrapper.BahdanauAttentionV2
    create_attention_kwargs = {"kernel_initializer": "ones", "normalize": True}

    expected_final_output = basic_decoder.BasicDecoderOutput(
        rnn_output=ResultSummary(
            shape=(5, 3, 6), dtype=np.dtype("float32"), mean=0.047594748),
        sample_id=ResultSummary(
            shape=(5, 3), dtype=np.dtype("int32"), mean=3.6))
    expected_final_state = wrapper.AttentionWrapperState(
        cell_state=[
            ResultSummary(
                shape=(5, 9), dtype=np.dtype("float32"), mean=0.41311637),
            ResultSummary(
                shape=(5, 9), dtype=np.dtype("float32"), mean=0.61683208)],
        attention=ResultSummary(
            shape=(5, 6), dtype=np.dtype("float32"), mean=0.090581432),
        time=3,
        alignments=ResultSummary(
            shape=(5, 8), dtype=np.dtype("float32"), mean=0.125),
        attention_state=ResultSummary(
            shape=(5, 8), dtype=np.dtype("float32"), mean=0.125),
        alignment_history=())

    self._testWithAttention(
        create_attention_mechanism,
        expected_final_output,
        expected_final_state,
        create_query_layer=True,
        create_attention_kwargs=create_attention_kwargs)

  def testLuongNotNormalized(self):
    create_attention_mechanism = wrapper.LuongAttentionV2

    expected_final_output = basic_decoder.BasicDecoderOutput(
        rnn_output=ResultSummary(
            shape=(5, 3, 6), dtype=np.dtype("float32"), mean=0.05481226),
        sample_id=ResultSummary(
            shape=(5, 3), dtype=np.dtype("int32"), mean=3.13333333))
    expected_final_state = wrapper.AttentionWrapperState(
        cell_state=[
            ResultSummary(
                shape=(5, 9), dtype=np.dtype("float32"), mean=0.38453412),
            ResultSummary(
                shape=(5, 9), dtype=np.dtype("float32"), mean=0.5785929)],
        attention=ResultSummary(
            shape=(5, 6), dtype=np.dtype("float32"), mean=0.16311775),
        time=3,
        alignments=ResultSummary(
            shape=(5, 8), dtype=np.dtype("float32"), mean=0.125),
        attention_state=ResultSummary(
            shape=(5, 8), dtype=np.dtype("float32"), mean=0.125),
        alignment_history=())

    self._testWithAttention(
        create_attention_mechanism,
        expected_final_output,
        expected_final_state,
        attention_mechanism_depth=9)

  def testLuongScaled(self):
    create_attention_mechanism = wrapper.LuongAttentionV2
    create_attention_kwargs = {"scale": True}

    expected_final_output = basic_decoder.BasicDecoderOutput(
        rnn_output=ResultSummary(
            shape=(5, 3, 6), dtype=np.dtype("float32"), mean=0.05481226),
        sample_id=ResultSummary(
            shape=(5, 3), dtype=np.dtype("int32"), mean=3.13333333))
    expected_final_state = wrapper.AttentionWrapperState(
        cell_state=[
            ResultSummary(
                shape=(5, 9), dtype=np.dtype("float32"), mean=0.38453412),
            ResultSummary(
                shape=(5, 9), dtype=np.dtype("float32"), mean=0.5785929)],
        attention=ResultSummary(
            shape=(5, 6), dtype=np.dtype("float32"), mean=0.16311775),
        time=3,
        alignments=ResultSummary(
            shape=(5, 8), dtype=np.dtype("float32"), mean=0.125),
        attention_state=ResultSummary(
            shape=(5, 8), dtype=np.dtype("float32"), mean=0.125),
        alignment_history=())

    self._testWithAttention(
        create_attention_mechanism,
        expected_final_output,
        expected_final_state,
        attention_mechanism_depth=9,
        create_attention_kwargs=create_attention_kwargs)

  def testNotUseAttentionLayer(self):
    create_attention_mechanism = wrapper.BahdanauAttentionV2
    create_attention_kwargs = {"kernel_initializer": "ones"}

    expected_final_output = basic_decoder.BasicDecoderOutput(
        rnn_output=ResultSummary(
            shape=(5, 3, 10), dtype=np.dtype("float32"), mean=0.072406612),
        sample_id=ResultSummary(
            shape=(5, 3), dtype=np.dtype("int32"), mean=3.86666666))
    expected_final_state = wrapper.AttentionWrapperState(
        cell_state=[
            ResultSummary(
                shape=(5, 9), dtype=np.dtype("float32"), mean=0.61177742),
            ResultSummary(
                shape=(5, 9), dtype=np.dtype("float32"), mean=1.032002)],
        attention=ResultSummary(
            shape=(5, 10), dtype=np.dtype("float32"), mean=0.011346335),
        time=3,
        alignments=ResultSummary(
            shape=(5, 8), dtype=np.dtype("float32"), mean=0.125),
        attention_state=ResultSummary(
            shape=(5, 8), dtype=np.dtype("float32"), mean=0.125),
        alignment_history=())

    self._testWithAttention(
        create_attention_mechanism,
        expected_final_output,
        expected_final_state,
        attention_layer_size=None,
        create_query_layer=True,
        create_attention_kwargs=create_attention_kwargs)

  def testBahdanauMonotonicNotNormalized(self):
    create_attention_mechanism = wrapper.BahdanauMonotonicAttentionV2
    create_attention_kwargs = {"kernel_initializer": "ones"}

    expected_final_output = basic_decoder.BasicDecoderOutput(
        rnn_output=ResultSummary(
            shape=(5, 3, 6), dtype=np.dtype("float32"), mean=0.041342419),
        sample_id=ResultSummary(
            shape=(5, 3), dtype=np.dtype("int32"), mean=3.53333333))
    expected_final_state = wrapper.AttentionWrapperState(
        cell_state=[
            ResultSummary(
                shape=(5, 9), dtype=np.dtype("float32"), mean=0.33866978),
            ResultSummary(
                shape=(5, 9), dtype=np.dtype("float32"), mean=0.46913195)],
        attention=ResultSummary(
            shape=(5, 6), dtype=np.dtype("float32"), mean=0.092498459),
        time=3,
        alignments=ResultSummary(
            shape=(5, 8), dtype=np.dtype("float32"), mean=0.12079944),
        attention_state=ResultSummary(
            shape=(5, 8), dtype=np.dtype("float32"), mean=0.12079944),
        alignment_history=())
    expected_final_alignment_history = ResultSummary(
        shape=(3, 5, 8), dtype=np.dtype("float32"), mean=0.121448785067)

    self._testWithAttention(
        create_attention_mechanism,
        expected_final_output,
        expected_final_state,
        alignment_history=True,
        expected_final_alignment_history=expected_final_alignment_history,
        create_query_layer=True,
        create_attention_kwargs=create_attention_kwargs)

  def testBahdanauMonotonicNormalized(self):
    create_attention_mechanism = wrapper.BahdanauMonotonicAttentionV2
    create_attention_kwargs = {"kernel_initializer": "ones",
                               "normalize": True}
    expected_final_output = basic_decoder.BasicDecoderOutput(
        rnn_output=ResultSummary(
            shape=(5, 3, 6), dtype=np.dtype("float32"), mean=0.043294173),
        sample_id=ResultSummary(
            shape=(5, 3), dtype=np.dtype("int32"), mean=3.53333333))
    expected_final_state = wrapper.AttentionWrapperState(
        cell_state=[
            ResultSummary(
                shape=(5, 9), dtype=np.dtype("float32"), mean=0.40034312),
            ResultSummary(
                shape=(5, 9), dtype=np.dtype("float32"), mean=0.5925445)],
        attention=ResultSummary(
            shape=(5, 6), dtype=np.dtype("float32"), mean=0.096119694),
        time=3,
        alignments=ResultSummary(
            shape=(5, 8), dtype=np.dtype("float32"), mean=0.1211452),
        attention_state=ResultSummary(
            shape=(5, 8), dtype=np.dtype("float32"), mean=0.1211452),
        alignment_history=())
    expected_final_alignment_history = ResultSummary(
        shape=(3, 5, 8), dtype=np.dtype("float32"), mean=0.12258384)

    self._testWithAttention(
        create_attention_mechanism,
        expected_final_output,
        expected_final_state,
        alignment_history=True,
        expected_final_alignment_history=expected_final_alignment_history,
        create_query_layer=True,
        create_attention_kwargs=create_attention_kwargs)

  def testLuongMonotonicNotNormalized(self):
    create_attention_mechanism = wrapper.LuongMonotonicAttentionV2

    expected_final_output = basic_decoder.BasicDecoderOutput(
        rnn_output=ResultSummary(
            shape=(5, 3, 6), dtype=np.dtype("float32"), mean=0.027387079),
        sample_id=ResultSummary(
            shape=(5, 3), dtype=np.dtype("int32"), mean=3.133333333))
    expected_final_state = wrapper.AttentionWrapperState(
        cell_state=[
            ResultSummary(
                shape=(5, 9), dtype=np.dtype("float32"), mean=0.32660431),
            ResultSummary(
                shape=(5, 9), dtype=np.dtype("float32"), mean=0.52464348)],
        attention=ResultSummary(
            shape=(5, 6), dtype=np.dtype("float32"), mean=0.089345723),
        time=3,
        alignments=ResultSummary(
            shape=(5, 8), dtype=np.dtype("float32"), mean=0.11831035),
        attention_state=ResultSummary(
            shape=(5, 8), dtype=np.dtype("float32"), mean=0.11831035),
        alignment_history=())
    expected_final_alignment_history = ResultSummary(
        shape=(3, 5, 8), dtype=np.dtype("float32"), mean=0.12194442004)

    self._testWithAttention(
        create_attention_mechanism,
        expected_final_output,
        expected_final_state,
        attention_mechanism_depth=9,
        alignment_history=True,
        expected_final_alignment_history=expected_final_alignment_history)

  def testLuongMonotonicScaled(self):
    create_attention_mechanism = wrapper.LuongMonotonicAttentionV2
    create_attention_kwargs = {"scale": True}

    expected_final_output = basic_decoder.BasicDecoderOutput(
        rnn_output=ResultSummary(
            shape=(5, 3, 6), dtype=np.dtype("float32"), mean=0.027387079),
        sample_id=ResultSummary(
            shape=(5, 3), dtype=np.dtype("int32"), mean=3.13333333))
    expected_final_state = wrapper.AttentionWrapperState(
        cell_state=[
            ResultSummary(
                shape=(5, 9), dtype=np.dtype("float32"), mean=0.32660431),
            ResultSummary(
                shape=(5, 9), dtype=np.dtype("float32"), mean=0.52464348)],
        attention=ResultSummary(
            shape=(5, 6), dtype=np.dtype("float32"), mean=0.089345723),
        time=3,
        alignments=ResultSummary(
            shape=(5, 8), dtype=np.dtype("float32"), mean=0.11831035),
        attention_state=ResultSummary(
            shape=(5, 8), dtype=np.dtype("float32"), mean=0.11831035),
        alignment_history=())
    expected_final_alignment_history = ResultSummary(
        shape=(3, 5, 8), dtype=np.dtype("float32"), mean=0.12194442004)

    self._testWithAttention(
        create_attention_mechanism,
        expected_final_output,
        expected_final_state,
        attention_mechanism_depth=9,
        alignment_history=True,
        expected_final_alignment_history=expected_final_alignment_history,
        create_attention_kwargs=create_attention_kwargs)

if __name__ == "__main__":
  test.main()
