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
# pylint: disable=unused-import,g-bad-import-order
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# pylint: enable=unused-import

import tensorflow as tf
from tensorflow.contrib import layers

class Seq2SeqTest(tf.test.TestCase):

  # test a default call of rnn_decoder
  def test_rnn_decoder(self):
    pass

  # test default call with time_major=True
  def test_dynamic_rnn_decoder_time_major(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=
                             tf.constant_initializer(0.5)) as varscope:
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
        decoder_embeddings = tf.get_variable('decoder_embeddings',
            [num_decoder_symbols, decoder_embedding_size],
            initializer=tf.random_normal_initializer(stddev=0.1))
        inputs = tf.constant(0.5, shape=[input_sequence_length, batch_size,
                                         encoder_embedding_size])
        decoder_inputs = tf.constant(0.4, shape=[decoder_sequence_length,
                                                 batch_size,
                                                 decoder_embedding_size])
        decoder_length = tf.constant(decoder_sequence_length, dtype=tf.int32,
                                     shape=[batch_size,])
        with tf.variable_scope("rnn") as scope:
          # setting up weights for computing the final output
          output_fn = lambda x: layers.linear(x, num_decoder_symbols,
                                              scope=scope)

          # Define model
          encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
              cell=tf.contrib.rnn.GRUCell(encoder_hidden_size), inputs=inputs,
              dtype=tf.float32, time_major=True, scope=scope)


        with tf.variable_scope("decoder") as scope:
          # Train decoder
          decoder_cell = tf.contrib.rnn.GRUCell(decoder_hidden_size)
          decoder_fn_train = Seq2SeqTest._decoder_fn_with_context_state(
              tf.contrib.seq2seq.simple_decoder_fn_train(
                  encoder_state=encoder_state))
          (decoder_outputs_train, decoder_state_train,
           decoder_context_state_train) = (
               tf.contrib.seq2seq.dynamic_rnn_decoder(
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
              tf.contrib.seq2seq.simple_decoder_fn_inference(
                output_fn=output_fn,
                encoder_state=encoder_state,
                embeddings=decoder_embeddings,
                start_of_sequence_id=start_of_sequence_id,
                end_of_sequence_id=end_of_sequence_id,
                #TODO: find out why it goes to +1
                maximum_length=decoder_sequence_length-1,
                num_decoder_symbols=num_decoder_symbols,
                dtype=tf.int32))
          (decoder_outputs_inference, decoder_state_inference,
           decoder_context_state_inference) = (
               tf.contrib.seq2seq.dynamic_rnn_decoder(
                   cell=decoder_cell,
                   decoder_fn=decoder_fn_inference,
                   time_major=True,
                   scope=scope))

        # Run model
        tf.global_variables_initializer().run()
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
                          num_decoder_symbols),
                         decoder_outputs_train_res.shape)
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
      with tf.name_scope(
          name, "decoder_fn_with_context_state",
          [time, cell_state, cell_input, cell_output, context_state]):
        done, next_state, next_input, emit_output, next_context_state = (
            inner_decoder_fn(time, cell_state, cell_input, cell_output,
                             context_state))
        next_context_state = time
        return done, next_state, next_input, emit_output, next_context_state

    return decoder_fn


if __name__ == '__main__':
  tf.test.main()
