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
"""Tests for contrib.seq2seq.python.ops.seq2seq."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, "getdlopenflags") and hasattr(sys, "setdlopenflags"):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

from tensorflow.contrib import layers
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.contrib.seq2seq.python.ops import attention_decoder_fn
from tensorflow.contrib.seq2seq.python.ops import decoder_fn as decoder_fn_lib
from tensorflow.contrib.seq2seq.python.ops import seq2seq
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class Seq2SeqTest(test.TestCase):

  # test a default call of rnn_decoder
  def test_rnn_decoder(self):
    pass

  # test default call with time_major=True
  def test_dynamic_rnn_decoder_time_major(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)) as varscope:
        # Define inputs/outputs to model
        batch_size = 2
        encoder_embedding_size = 3
        decoder_embedding_size = 4
        encoder_hidden_size = 5
        decoder_hidden_size = encoder_hidden_size
        input_sequence_length = 6
        decoder_sequence_length = 7
        num_decoder_symbols = 20
        start_of_sequence_id = end_of_sequence_id = 1
        decoder_embeddings = variable_scope.get_variable(
            "decoder_embeddings", [num_decoder_symbols, decoder_embedding_size],
            initializer=init_ops.random_normal_initializer(stddev=0.1))
        inputs = constant_op.constant(
            0.5,
            shape=[input_sequence_length, batch_size, encoder_embedding_size])
        decoder_inputs = constant_op.constant(
            0.4,
            shape=[decoder_sequence_length, batch_size, decoder_embedding_size])
        decoder_length = constant_op.constant(
            decoder_sequence_length, dtype=dtypes.int32, shape=[batch_size,])
        with variable_scope.variable_scope("rnn") as scope:
          # setting up weights for computing the final output
          output_fn = lambda x: layers.linear(x, num_decoder_symbols,
                                              scope=scope)

          # Define model
          encoder_outputs, encoder_state = rnn.dynamic_rnn(
              cell=core_rnn_cell_impl.GRUCell(encoder_hidden_size),
              inputs=inputs,
              dtype=dtypes.float32,
              time_major=True,
              scope=scope)

        with variable_scope.variable_scope("decoder") as scope:
          # Train decoder
          decoder_cell = core_rnn_cell_impl.GRUCell(decoder_hidden_size)
          decoder_fn_train = Seq2SeqTest._decoder_fn_with_context_state(
              decoder_fn_lib.simple_decoder_fn_train(
                  encoder_state=encoder_state))
          (decoder_outputs_train, decoder_state_train,
           decoder_context_state_train) = (seq2seq.dynamic_rnn_decoder(
               cell=decoder_cell,
               decoder_fn=decoder_fn_train,
               inputs=decoder_inputs,
               sequence_length=decoder_length,
               time_major=True,
               scope=scope))
          decoder_outputs_train = output_fn(decoder_outputs_train)

          # Setup variable reuse
          scope.reuse_variables()

          # Inference decoder
          decoder_fn_inference = Seq2SeqTest._decoder_fn_with_context_state(
              decoder_fn_lib.simple_decoder_fn_inference(
                  output_fn=output_fn,
                  encoder_state=encoder_state,
                  embeddings=decoder_embeddings,
                  start_of_sequence_id=start_of_sequence_id,
                  end_of_sequence_id=end_of_sequence_id,
                  #TODO: find out why it goes to +1
                  maximum_length=decoder_sequence_length - 1,
                  num_decoder_symbols=num_decoder_symbols,
                  dtype=dtypes.int32))
          (decoder_outputs_inference, decoder_state_inference,
           decoder_context_state_inference) = (seq2seq.dynamic_rnn_decoder(
               cell=decoder_cell,
               decoder_fn=decoder_fn_inference,
               time_major=True,
               scope=scope))

        # Run model
        variables.global_variables_initializer().run()
        (decoder_outputs_train_res, decoder_state_train_res,
         decoder_context_state_train_res) = sess.run([
             decoder_outputs_train, decoder_state_train,
             decoder_context_state_train
         ])
        (decoder_outputs_inference_res, decoder_state_inference_res,
         decoder_context_state_inference_res) = sess.run([
             decoder_outputs_inference, decoder_state_inference,
             decoder_context_state_inference
         ])

        # Assert outputs
        self.assertEqual((decoder_sequence_length, batch_size,
                          num_decoder_symbols), decoder_outputs_train_res.shape)
        self.assertEqual((batch_size, num_decoder_symbols),
                         decoder_outputs_inference_res.shape[1:3])
        self.assertEqual(decoder_sequence_length,
                         decoder_context_state_inference_res)
        self.assertEqual((batch_size, decoder_hidden_size),
                         decoder_state_train_res.shape)
        self.assertEqual((batch_size, decoder_hidden_size),
                         decoder_state_inference_res.shape)
        self.assertEqual(decoder_sequence_length,
                         decoder_context_state_train_res)
        # The dynamic decoder might end earlier than `maximal_length`
        # under inference
        self.assertGreaterEqual(decoder_sequence_length,
                                decoder_state_inference_res.shape[0])

  # test attention
  def test_attention(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        # Define inputs/outputs to model
        batch_size = 2
        encoder_embedding_size = 3
        decoder_embedding_size = 4
        encoder_hidden_size = 5
        decoder_hidden_size = encoder_hidden_size
        input_sequence_length = 6
        decoder_sequence_length = 7
        num_decoder_symbols = 20
        start_of_sequence_id = end_of_sequence_id = 1
        decoder_embeddings = variable_scope.get_variable(
            "decoder_embeddings", [num_decoder_symbols, decoder_embedding_size],
            initializer=init_ops.random_normal_initializer(stddev=0.1))
        inputs = constant_op.constant(
            0.5,
            shape=[input_sequence_length, batch_size, encoder_embedding_size])
        decoder_inputs = constant_op.constant(
            0.4,
            shape=[decoder_sequence_length, batch_size, decoder_embedding_size])
        decoder_length = constant_op.constant(
            decoder_sequence_length, dtype=dtypes.int32, shape=[batch_size,])

        # attention
        attention_option = "luong"  # can be "bahdanau"

        with variable_scope.variable_scope("rnn") as scope:
          # Define model
          encoder_outputs, encoder_state = rnn.dynamic_rnn(
              cell=core_rnn_cell_impl.GRUCell(encoder_hidden_size),
              inputs=inputs,
              dtype=dtypes.float32,
              time_major=True,
              scope=scope)

          # attention_states: size [batch_size, max_time, num_units]
          attention_states = array_ops.transpose(encoder_outputs, [1, 0, 2])

        with variable_scope.variable_scope("decoder") as scope:
          # Prepare attention
          (attention_keys, attention_values, attention_score_fn,
           attention_construct_fn) = (attention_decoder_fn.prepare_attention(
               attention_states, attention_option, decoder_hidden_size))
          decoder_fn_train = attention_decoder_fn.attention_decoder_fn_train(
              encoder_state=encoder_state,
              attention_keys=attention_keys,
              attention_values=attention_values,
              attention_score_fn=attention_score_fn,
              attention_construct_fn=attention_construct_fn)

          # setting up weights for computing the final output
          def create_output_fn():

            def output_fn(x):
              return layers.linear(x, num_decoder_symbols, scope=scope)

            return output_fn

          output_fn = create_output_fn()

          # Train decoder
          decoder_cell = core_rnn_cell_impl.GRUCell(decoder_hidden_size)
          (decoder_outputs_train, decoder_state_train, _) = (
              seq2seq.dynamic_rnn_decoder(
                  cell=decoder_cell,
                  decoder_fn=decoder_fn_train,
                  inputs=decoder_inputs,
                  sequence_length=decoder_length,
                  time_major=True,
                  scope=scope))
          decoder_outputs_train = output_fn(decoder_outputs_train)
          # Setup variable reuse
          scope.reuse_variables()

          # Inference decoder
          decoder_fn_inference = (
              attention_decoder_fn.attention_decoder_fn_inference(
                  output_fn=output_fn,
                  encoder_state=encoder_state,
                  attention_keys=attention_keys,
                  attention_values=attention_values,
                  attention_score_fn=attention_score_fn,
                  attention_construct_fn=attention_construct_fn,
                  embeddings=decoder_embeddings,
                  start_of_sequence_id=start_of_sequence_id,
                  end_of_sequence_id=end_of_sequence_id,
                  maximum_length=decoder_sequence_length - 1,
                  num_decoder_symbols=num_decoder_symbols,
                  dtype=dtypes.int32))
          (decoder_outputs_inference, decoder_state_inference, _) = (
              seq2seq.dynamic_rnn_decoder(
                  cell=decoder_cell,
                  decoder_fn=decoder_fn_inference,
                  time_major=True,
                  scope=scope))

        # Run model
        variables.global_variables_initializer().run()
        (decoder_outputs_train_res, decoder_state_train_res) = sess.run(
            [decoder_outputs_train, decoder_state_train])
        (decoder_outputs_inference_res, decoder_state_inference_res) = sess.run(
            [decoder_outputs_inference, decoder_state_inference])

        # Assert outputs
        self.assertEqual((decoder_sequence_length, batch_size,
                          num_decoder_symbols), decoder_outputs_train_res.shape)
        self.assertEqual((batch_size, num_decoder_symbols),
                         decoder_outputs_inference_res.shape[1:3])
        self.assertEqual((batch_size, decoder_hidden_size),
                         decoder_state_train_res.shape)
        self.assertEqual((batch_size, decoder_hidden_size),
                         decoder_state_inference_res.shape)
        # The dynamic decoder might end earlier than `maximal_length`
        # under inference
        self.assertGreaterEqual(decoder_sequence_length,
                                decoder_state_inference_res.shape[0])

  @staticmethod
  def _decoder_fn_with_context_state(inner_decoder_fn, name=None):
    """Wraps a given decoder function, adding context state to it.

    Given a valid `inner_decoder_fn`, returns another valid `decoder_fn` which
    first calls `inner_decoder_fn`, then overwrites the context_state, setting
    it to the current time.

    Args:
      inner_decoder_fn: A valid `decoder_fn` of the type passed into
        `dynamic_rnn_decoder`.

    Returns:
      A valid `decoder_fn` to be passed into `dynamic_rnn_decoder`.
    """

    def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
      with ops.name_scope(
          name, "decoder_fn_with_context_state",
          [time, cell_state, cell_input, cell_output, context_state]):
        done, next_state, next_input, emit_output, next_context_state = (
            inner_decoder_fn(time, cell_state, cell_input, cell_output,
                             context_state))
        next_context_state = time
        return done, next_state, next_input, emit_output, next_context_state

    return decoder_fn


if __name__ == "__main__":
  test.main()
