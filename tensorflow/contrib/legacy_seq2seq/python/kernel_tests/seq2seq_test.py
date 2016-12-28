# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for functional style sequence-to-sequence models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random

import numpy as np
import tensorflow as tf


class Seq2SeqTest(tf.test.TestCase):

  def testRNNDecoder(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        inp = [tf.constant(0.5, shape=[2, 2])] * 2
        _, enc_state = tf.contrib.rnn.static_rnn(
            tf.contrib.rnn.GRUCell(2), inp, dtype=tf.float32)
        dec_inp = [tf.constant(0.4, shape=[2, 2])] * 3
        cell = tf.contrib.rnn.OutputProjectionWrapper(
            tf.contrib.rnn.GRUCell(2), 4)
        dec, mem = tf.contrib.legacy_seq2seq.rnn_decoder(
            dec_inp, enc_state, cell)
        sess.run([tf.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 4), res[0].shape)

        res = sess.run([mem])
        self.assertEqual((2, 2), res[0].shape)

  def testBasicRNNSeq2Seq(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        inp = [tf.constant(0.5, shape=[2, 2])] * 2
        dec_inp = [tf.constant(0.4, shape=[2, 2])] * 3
        cell = tf.contrib.rnn.OutputProjectionWrapper(
            tf.contrib.rnn.GRUCell(2), 4)
        dec, mem = tf.contrib.legacy_seq2seq.basic_rnn_seq2seq(
            inp, dec_inp, cell)
        sess.run([tf.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 4), res[0].shape)

        res = sess.run([mem])
        self.assertEqual((2, 2), res[0].shape)

  def testTiedRNNSeq2Seq(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        inp = [tf.constant(0.5, shape=[2, 2])] * 2
        dec_inp = [tf.constant(0.4, shape=[2, 2])] * 3
        cell = tf.contrib.rnn.OutputProjectionWrapper(
            tf.contrib.rnn.GRUCell(2), 4)
        dec, mem = tf.contrib.legacy_seq2seq.tied_rnn_seq2seq(
            inp, dec_inp, cell)
        sess.run([tf.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 4), res[0].shape)

        res = sess.run([mem])
        self.assertEqual(1, len(res))
        self.assertEqual((2, 2), res[0].shape)

  def testEmbeddingRNNDecoder(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        inp = [tf.constant(0.5, shape=[2, 2])] * 2
        cell = tf.contrib.rnn.BasicLSTMCell(2, state_is_tuple=True)
        _, enc_state = tf.contrib.rnn.static_rnn(cell, inp, dtype=tf.float32)
        dec_inp = [tf.constant(i, tf.int32, shape=[2]) for i in range(3)]
        dec, mem = tf.contrib.legacy_seq2seq.embedding_rnn_decoder(
            dec_inp, enc_state, cell, num_symbols=4, embedding_size=2)
        sess.run([tf.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 2), res[0].shape)

        res = sess.run([mem])
        self.assertEqual(1, len(res))
        self.assertEqual((2, 2), res[0].c.shape)
        self.assertEqual((2, 2), res[0].h.shape)

  def testEmbeddingRNNSeq2Seq(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        enc_inp = [tf.constant(1, tf.int32, shape=[2]) for i in range(2)]
        dec_inp = [tf.constant(i, tf.int32, shape=[2]) for i in range(3)]
        cell = tf.contrib.rnn.BasicLSTMCell(2, state_is_tuple=True)
        dec, mem = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
            enc_inp, dec_inp, cell, num_encoder_symbols=2,
            num_decoder_symbols=5, embedding_size=2)
        sess.run([tf.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 5), res[0].shape)

        res = sess.run([mem])
        self.assertEqual((2, 2), res[0].c.shape)
        self.assertEqual((2, 2), res[0].h.shape)

        # Test with state_is_tuple=False.
        with tf.variable_scope("no_tuple"):
          cell1 = tf.contrib.rnn.BasicLSTMCell(2, state_is_tuple=False)
          dec, mem = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
              enc_inp, dec_inp, cell1, num_encoder_symbols=2,
              num_decoder_symbols=5, embedding_size=2)
          sess.run([tf.global_variables_initializer()])
          res = sess.run(dec)
          self.assertEqual(3, len(res))
          self.assertEqual((2, 5), res[0].shape)

          res = sess.run([mem])
          self.assertEqual((2, 4), res[0].shape)

        # Test externally provided output projection.
        w = tf.get_variable("proj_w", [2, 5])
        b = tf.get_variable("proj_b", [5])
        with tf.variable_scope("proj_seq2seq"):
          dec, _ = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
              enc_inp, dec_inp, cell, num_encoder_symbols=2,
              num_decoder_symbols=5, embedding_size=2, output_projection=(w, b))
        sess.run([tf.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 2), res[0].shape)

        # Test that previous-feeding model ignores inputs after the first.
        dec_inp2 = [tf.constant(0, tf.int32, shape=[2]) for _ in range(3)]
        with tf.variable_scope("other"):
          d3, _ = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
              enc_inp, dec_inp2, cell, num_encoder_symbols=2,
              num_decoder_symbols=5, embedding_size=2,
              feed_previous=tf.constant(True))
        sess.run([tf.global_variables_initializer()])
        tf.get_variable_scope().reuse_variables()
        d1, _ = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
            enc_inp, dec_inp, cell, num_encoder_symbols=2,
            num_decoder_symbols=5, embedding_size=2, feed_previous=True)
        d2, _ = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
            enc_inp, dec_inp2, cell, num_encoder_symbols=2,
            num_decoder_symbols=5, embedding_size=2, feed_previous=True)
        res1 = sess.run(d1)
        res2 = sess.run(d2)
        res3 = sess.run(d3)
        self.assertAllClose(res1, res2)
        self.assertAllClose(res1, res3)

  def testEmbeddingTiedRNNSeq2Seq(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        enc_inp = [tf.constant(1, tf.int32, shape=[2]) for i in range(2)]
        dec_inp = [tf.constant(i, tf.int32, shape=[2]) for i in range(3)]
        cell = tf.contrib.rnn.BasicLSTMCell(2, state_is_tuple=True)
        dec, mem = tf.contrib.legacy_seq2seq.embedding_tied_rnn_seq2seq(
            enc_inp, dec_inp, cell, num_symbols=5, embedding_size=2)
        sess.run([tf.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 5), res[0].shape)

        res = sess.run([mem])
        self.assertEqual((2, 2), res[0].c.shape)
        self.assertEqual((2, 2), res[0].h.shape)

        # Test when num_decoder_symbols is provided, the size of decoder output
        # is num_decoder_symbols.
        with tf.variable_scope("decoder_symbols_seq2seq"):
          dec, mem = tf.contrib.legacy_seq2seq.embedding_tied_rnn_seq2seq(
              enc_inp, dec_inp, cell, num_symbols=5, num_decoder_symbols=3,
              embedding_size=2)
        sess.run([tf.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 3), res[0].shape)

        # Test externally provided output projection.
        w = tf.get_variable("proj_w", [2, 5])
        b = tf.get_variable("proj_b", [5])
        with tf.variable_scope("proj_seq2seq"):
          dec, _ = tf.contrib.legacy_seq2seq.embedding_tied_rnn_seq2seq(
              enc_inp, dec_inp, cell, num_symbols=5, embedding_size=2,
              output_projection=(w, b))
        sess.run([tf.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 2), res[0].shape)

        # Test that previous-feeding model ignores inputs after the first.
        dec_inp2 = [tf.constant(0, tf.int32, shape=[2])] * 3
        with tf.variable_scope("other"):
          d3, _ = tf.contrib.legacy_seq2seq.embedding_tied_rnn_seq2seq(
              enc_inp, dec_inp2, cell, num_symbols=5, embedding_size=2,
              feed_previous=tf.constant(True))
        sess.run([tf.global_variables_initializer()])
        tf.get_variable_scope().reuse_variables()
        d1, _ = tf.contrib.legacy_seq2seq.embedding_tied_rnn_seq2seq(
            enc_inp, dec_inp, cell, num_symbols=5, embedding_size=2,
            feed_previous=True)
        d2, _ = tf.contrib.legacy_seq2seq.embedding_tied_rnn_seq2seq(
            enc_inp, dec_inp2, cell, num_symbols=5, embedding_size=2,
            feed_previous=True)
        res1 = sess.run(d1)
        res2 = sess.run(d2)
        res3 = sess.run(d3)
        self.assertAllClose(res1, res2)
        self.assertAllClose(res1, res3)

  def testAttentionDecoder1(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        cell = tf.contrib.rnn.GRUCell(2)
        inp = [tf.constant(0.5, shape=[2, 2])] * 2
        enc_outputs, enc_state = tf.contrib.rnn.static_rnn(
            cell, inp, dtype=tf.float32)
        attn_states = tf.concat_v2(
            [tf.reshape(e, [-1, 1, cell.output_size]) for e in enc_outputs], 1)
        dec_inp = [tf.constant(0.4, shape=[2, 2])] * 3
        dec, mem = tf.contrib.legacy_seq2seq.attention_decoder(
            dec_inp, enc_state,
            attn_states, cell, output_size=4)
        sess.run([tf.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 4), res[0].shape)

        res = sess.run([mem])
        self.assertEqual((2, 2), res[0].shape)

  def testAttentionDecoder2(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        cell = tf.contrib.rnn.GRUCell(2)
        inp = [tf.constant(0.5, shape=[2, 2])] * 2
        enc_outputs, enc_state = tf.contrib.rnn.static_rnn(
            cell, inp, dtype=tf.float32)
        attn_states = tf.concat_v2(
            [tf.reshape(e, [-1, 1, cell.output_size]) for e in enc_outputs], 1)
        dec_inp = [tf.constant(0.4, shape=[2, 2])] * 3
        dec, mem = tf.contrib.legacy_seq2seq.attention_decoder(
            dec_inp, enc_state,
            attn_states, cell, output_size=4,
            num_heads=2)
        sess.run([tf.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 4), res[0].shape)

        res = sess.run([mem])
        self.assertEqual((2, 2), res[0].shape)

  def testDynamicAttentionDecoder1(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        cell = tf.contrib.rnn.GRUCell(2)
        inp = tf.constant(0.5, shape=[2, 2, 2])
        enc_outputs, enc_state = tf.nn.dynamic_rnn(cell, inp, dtype=tf.float32)
        attn_states = enc_outputs
        dec_inp = [tf.constant(0.4, shape=[2, 2])] * 3
        dec, mem = tf.contrib.legacy_seq2seq.attention_decoder(
            dec_inp, enc_state,
            attn_states, cell, output_size=4)
        sess.run([tf.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 4), res[0].shape)

        res = sess.run([mem])
        self.assertEqual((2, 2), res[0].shape)

  def testDynamicAttentionDecoder2(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        cell = tf.contrib.rnn.GRUCell(2)
        inp = tf.constant(0.5, shape=[2, 2, 2])
        enc_outputs, enc_state = tf.nn.dynamic_rnn(cell, inp, dtype=tf.float32)
        attn_states = enc_outputs
        dec_inp = [tf.constant(0.4, shape=[2, 2])] * 3
        dec, mem = tf.contrib.legacy_seq2seq.attention_decoder(
            dec_inp, enc_state,
            attn_states, cell, output_size=4,
            num_heads=2)
        sess.run([tf.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 4), res[0].shape)

        res = sess.run([mem])
        self.assertEqual((2, 2), res[0].shape)

  def testAttentionDecoderStateIsTuple(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        cell = tf.contrib.rnn.BasicLSTMCell(2, state_is_tuple=True)
        cell = tf.contrib.rnn.MultiRNNCell(cells=[cell] * 2,
                                           state_is_tuple=True)
        inp = [tf.constant(0.5, shape=[2, 2])] * 2
        enc_outputs, enc_state = tf.contrib.rnn.static_rnn(
            cell, inp, dtype=tf.float32)
        attn_states = tf.concat_v2(
            [tf.reshape(e, [-1, 1, cell.output_size]) for e in enc_outputs], 1)
        dec_inp = [tf.constant(0.4, shape=[2, 2])] * 3
        dec, mem = tf.contrib.legacy_seq2seq.attention_decoder(
            dec_inp, enc_state,
            attn_states, cell, output_size=4)
        sess.run([tf.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 4), res[0].shape)

        res = sess.run([mem])
        self.assertEqual(2, len(res[0]))
        self.assertEqual((2, 2), res[0][0].c.shape)
        self.assertEqual((2, 2), res[0][0].h.shape)
        self.assertEqual((2, 2), res[0][1].c.shape)
        self.assertEqual((2, 2), res[0][1].h.shape)

    def testDynamicAttentionDecoderStateIsTuple(self):
      with self.test_session() as sess:
        with tf.variable_scope("root",
                               initializer=tf.constant_initializer(0.5)):
          cell = tf.contrib.rnn.BasicLSTMCell(2, state_is_tuple=True)
          cell = tf.contrib.rnn.MultiRNNCell(cells=[cell] * 2,
                                             state_is_tuple=True)
          inp = tf.constant(0.5, shape=[2, 2, 2])
          enc_outputs, enc_state = tf.contrib.rnn.static_rnn(
              cell, inp, dtype=tf.float32)
          attn_states = tf.concat_v2(
              [tf.reshape(e, [-1, 1, cell.output_size]) for e in enc_outputs],
              1)
          dec_inp = [tf.constant(0.4, shape=[2, 2])] * 3
          dec, mem = tf.contrib.legacy_seq2seq.attention_decoder(
              dec_inp, enc_state,
              attn_states, cell, output_size=4)
          sess.run([tf.global_variables_initializer()])
          res = sess.run(dec)
          self.assertEqual(3, len(res))
          self.assertEqual((2, 4), res[0].shape)

          res = sess.run([mem])
          self.assertEqual(2, len(res[0]))
          self.assertEqual((2, 2), res[0][0].c.shape)
          self.assertEqual((2, 2), res[0][0].h.shape)
          self.assertEqual((2, 2), res[0][1].c.shape)
          self.assertEqual((2, 2), res[0][1].h.shape)

  def testEmbeddingAttentionDecoder(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        inp = [tf.constant(0.5, shape=[2, 2])] * 2
        cell = tf.contrib.rnn.GRUCell(2)
        enc_outputs, enc_state = tf.contrib.rnn.static_rnn(
            cell, inp, dtype=tf.float32)
        attn_states = tf.concat_v2(
            [tf.reshape(e, [-1, 1, cell.output_size]) for e in enc_outputs], 1)
        dec_inp = [tf.constant(i, tf.int32, shape=[2]) for i in range(3)]
        dec, mem = tf.contrib.legacy_seq2seq.embedding_attention_decoder(
            dec_inp, enc_state, attn_states, cell, num_symbols=4,
            embedding_size=2, output_size=3)
        sess.run([tf.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 3), res[0].shape)

        res = sess.run([mem])
        self.assertEqual((2, 2), res[0].shape)

  def testEmbeddingAttentionSeq2Seq(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        enc_inp = [tf.constant(1, tf.int32, shape=[2]) for i in range(2)]
        dec_inp = [tf.constant(i, tf.int32, shape=[2]) for i in range(3)]
        cell = tf.contrib.rnn.BasicLSTMCell(2, state_is_tuple=True)
        dec, mem = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
            enc_inp, dec_inp, cell, num_encoder_symbols=2,
            num_decoder_symbols=5, embedding_size=2)
        sess.run([tf.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 5), res[0].shape)

        res = sess.run([mem])
        self.assertEqual((2, 2), res[0].c.shape)
        self.assertEqual((2, 2), res[0].h.shape)

        # Test with state_is_tuple=False.
        with tf.variable_scope("no_tuple"):
          cell = tf.contrib.rnn.BasicLSTMCell(2, state_is_tuple=False)
          dec, mem = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
              enc_inp, dec_inp, cell, num_encoder_symbols=2,
              num_decoder_symbols=5, embedding_size=2)
          sess.run([tf.global_variables_initializer()])
          res = sess.run(dec)
          self.assertEqual(3, len(res))
          self.assertEqual((2, 5), res[0].shape)

          res = sess.run([mem])
          self.assertEqual((2, 4), res[0].shape)

        # Test externally provided output projection.
        w = tf.get_variable("proj_w", [2, 5])
        b = tf.get_variable("proj_b", [5])
        with tf.variable_scope("proj_seq2seq"):
          dec, _ = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
              enc_inp, dec_inp, cell, num_encoder_symbols=2,
              num_decoder_symbols=5, embedding_size=2, output_projection=(w, b))
        sess.run([tf.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 2), res[0].shape)

        # Test that previous-feeding model ignores inputs after the first.
        dec_inp2 = [tf.constant(0, tf.int32, shape=[2]) for _ in range(3)]
        with tf.variable_scope("other"):
          d3, _ = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
              enc_inp, dec_inp2, cell, num_encoder_symbols=2,
              num_decoder_symbols=5, embedding_size=2,
              feed_previous=tf.constant(True))
        sess.run([tf.global_variables_initializer()])
        tf.get_variable_scope().reuse_variables()
        d1, _ = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
            enc_inp, dec_inp, cell, num_encoder_symbols=2,
            num_decoder_symbols=5, embedding_size=2, feed_previous=True)
        d2, _ = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
            enc_inp, dec_inp2, cell, num_encoder_symbols=2,
            num_decoder_symbols=5, embedding_size=2, feed_previous=True)
        res1 = sess.run(d1)
        res2 = sess.run(d2)
        res3 = sess.run(d3)
        self.assertAllClose(res1, res2)
        self.assertAllClose(res1, res3)

  def testOne2ManyRNNSeq2Seq(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        enc_inp = [tf.constant(1, tf.int32, shape=[2]) for i in range(2)]
        dec_inp_dict = {}
        dec_inp_dict["0"] = [
            tf.constant(i, tf.int32, shape=[2]) for i in range(3)]
        dec_inp_dict["1"] = [
            tf.constant(i, tf.int32, shape=[2]) for i in range(4)]
        dec_symbols_dict = {"0": 5, "1": 6}
        cell = tf.contrib.rnn.BasicLSTMCell(2, state_is_tuple=True)
        outputs_dict, state_dict = (
            tf.contrib.legacy_seq2seq.one2many_rnn_seq2seq(
                enc_inp, dec_inp_dict, cell, 2, dec_symbols_dict,
                embedding_size=2))

        sess.run([tf.global_variables_initializer()])
        res = sess.run(outputs_dict["0"])
        self.assertEqual(3, len(res))
        self.assertEqual((2, 5), res[0].shape)
        res = sess.run(outputs_dict["1"])
        self.assertEqual(4, len(res))
        self.assertEqual((2, 6), res[0].shape)
        res = sess.run([state_dict["0"]])
        self.assertEqual((2, 2), res[0].c.shape)
        self.assertEqual((2, 2), res[0].h.shape)
        res = sess.run([state_dict["1"]])
        self.assertEqual((2, 2), res[0].c.shape)
        self.assertEqual((2, 2), res[0].h.shape)

        # Test that previous-feeding model ignores inputs after the first, i.e.
        # dec_inp_dict2 has different inputs from dec_inp_dict after the first
        # time-step.
        dec_inp_dict2 = {}
        dec_inp_dict2["0"] = [
            tf.constant(0, tf.int32, shape=[2]) for _ in range(3)]
        dec_inp_dict2["1"] = [
            tf.constant(0, tf.int32, shape=[2]) for _ in range(4)]
        with tf.variable_scope("other"):
          outputs_dict3, _ = tf.contrib.legacy_seq2seq.one2many_rnn_seq2seq(
              enc_inp, dec_inp_dict2, cell, 2, dec_symbols_dict,
              embedding_size=2, feed_previous=tf.constant(True))
        sess.run([tf.global_variables_initializer()])
        tf.get_variable_scope().reuse_variables()
        outputs_dict1, _ = tf.contrib.legacy_seq2seq.one2many_rnn_seq2seq(
            enc_inp, dec_inp_dict, cell, 2, dec_symbols_dict,
            embedding_size=2, feed_previous=True)
        outputs_dict2, _ = tf.contrib.legacy_seq2seq.one2many_rnn_seq2seq(
            enc_inp, dec_inp_dict2, cell, 2, dec_symbols_dict,
            embedding_size=2, feed_previous=True)
        res1 = sess.run(outputs_dict1["0"])
        res2 = sess.run(outputs_dict2["0"])
        res3 = sess.run(outputs_dict3["0"])
        self.assertAllClose(res1, res2)
        self.assertAllClose(res1, res3)

  def testSequenceLoss(self):
    with self.test_session() as sess:
      logits = [tf.constant(i + 0.5, shape=[2, 5]) for i in range(3)]
      targets = [tf.constant(i, tf.int32, shape=[2]) for i in range(3)]
      weights = [tf.constant(1.0, shape=[2]) for i in range(3)]

      average_loss_per_example = tf.contrib.legacy_seq2seq.sequence_loss(
          logits, targets, weights,
          average_across_timesteps=True,
          average_across_batch=True)
      res = sess.run(average_loss_per_example)
      self.assertAllClose(1.60944, res)

      average_loss_per_sequence = tf.contrib.legacy_seq2seq.sequence_loss(
          logits, targets, weights,
          average_across_timesteps=False,
          average_across_batch=True)
      res = sess.run(average_loss_per_sequence)
      self.assertAllClose(4.828314, res)

      total_loss = tf.contrib.legacy_seq2seq.sequence_loss(
          logits, targets, weights,
          average_across_timesteps=False,
          average_across_batch=False)
      res = sess.run(total_loss)
      self.assertAllClose(9.656628, res)

  def testSequenceLossByExample(self):
    with self.test_session() as sess:
      output_classes = 5
      logits = [tf.constant(i + 0.5, shape=[2, output_classes])
                for i in range(3)]
      targets = [tf.constant(i, tf.int32, shape=[2]) for i in range(3)]
      weights = [tf.constant(1.0, shape=[2]) for i in range(3)]

      average_loss_per_example = (
          tf.contrib.legacy_seq2seq.sequence_loss_by_example(
              logits, targets, weights,
              average_across_timesteps=True))
      res = sess.run(average_loss_per_example)
      self.assertAllClose(np.asarray([1.609438, 1.609438]), res)

      loss_per_sequence = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
          logits, targets, weights,
          average_across_timesteps=False)
      res = sess.run(loss_per_sequence)
      self.assertAllClose(np.asarray([4.828314, 4.828314]), res)

  def testModelWithBucketsScopeAndLoss(self):
    """Test that variable scope reuse is not reset after model_with_buckets."""
    classes = 10
    buckets = [(4, 4), (8, 8)]

    with self.test_session():
      # Here comes a sample Seq2Seq model using GRU cells.
      def SampleGRUSeq2Seq(enc_inp, dec_inp, weights, per_example_loss):
        """Example sequence-to-sequence model that uses GRU cells."""
        def GRUSeq2Seq(enc_inp, dec_inp):
          cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(24)] * 2,
                                             state_is_tuple=True)
          return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
              enc_inp, dec_inp, cell, num_encoder_symbols=classes,
              num_decoder_symbols=classes, embedding_size=24)
        targets = [dec_inp[i+1] for i in range(len(dec_inp) - 1)] + [0]
        return tf.contrib.legacy_seq2seq.model_with_buckets(
            enc_inp, dec_inp, targets, weights, buckets, GRUSeq2Seq,
            per_example_loss=per_example_loss)

      # Now we construct the copy model.
      inp = [tf.placeholder(tf.int32, shape=[None]) for _ in range(8)]
      out = [tf.placeholder(tf.int32, shape=[None]) for _ in range(8)]
      weights = [tf.ones_like(inp[0], dtype=tf.float32) for _ in range(8)]
      with tf.variable_scope("root"):
        _, losses1 = SampleGRUSeq2Seq(inp, out, weights, per_example_loss=False)
        # Now check that we did not accidentally set reuse.
        self.assertEqual(False, tf.get_variable_scope().reuse)
        # Construct one more model with per-example loss.
        tf.get_variable_scope().reuse_variables()
        _, losses2 = SampleGRUSeq2Seq(inp, out, weights, per_example_loss=True)
        # First loss is scalar, the second one is a 1-dimensinal tensor.
        self.assertEqual([], losses1[0].get_shape().as_list())
        self.assertEqual([None], losses2[0].get_shape().as_list())

  def testModelWithBuckets(self):
    """Larger tests that does full sequence-to-sequence model training."""
    # We learn to copy 10 symbols in 2 buckets: length 4 and length 8.
    classes = 10
    buckets = [(4, 4), (8, 8)]
    perplexities = [[], []]  # Results for each bucket.
    tf.set_random_seed(111)
    random.seed(111)
    np.random.seed(111)

    with self.test_session() as sess:
      # We use sampled softmax so we keep output projection separate.
      w = tf.get_variable("proj_w", [24, classes])
      w_t = tf.transpose(w)
      b = tf.get_variable("proj_b", [classes])
      # Here comes a sample Seq2Seq model using GRU cells.
      def SampleGRUSeq2Seq(enc_inp, dec_inp, weights):
        """Example sequence-to-sequence model that uses GRU cells."""
        def GRUSeq2Seq(enc_inp, dec_inp):
          cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(24)] * 2,
                                             state_is_tuple=True)
          return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
              enc_inp, dec_inp, cell, num_encoder_symbols=classes,
              num_decoder_symbols=classes, embedding_size=24,
              output_projection=(w, b))
        targets = [dec_inp[i+1] for i in range(len(dec_inp) - 1)] + [0]
        def SampledLoss(labels, inputs):
          labels = tf.reshape(labels, [-1, 1])
          return tf.nn.sampled_softmax_loss(
              weights=w_t,
              biases=b,
              labels=labels,
              inputs=inputs,
              num_sampled=8,
              num_classes=classes)
        return tf.contrib.legacy_seq2seq.model_with_buckets(
            enc_inp, dec_inp, targets, weights, buckets, GRUSeq2Seq,
            softmax_loss_function=SampledLoss)

      # Now we construct the copy model.
      batch_size = 8
      inp = [tf.placeholder(tf.int32, shape=[None]) for _ in range(8)]
      out = [tf.placeholder(tf.int32, shape=[None]) for _ in range(8)]
      weights = [tf.ones_like(inp[0], dtype=tf.float32) for _ in range(8)]
      with tf.variable_scope("root"):
        _, losses = SampleGRUSeq2Seq(inp, out, weights)
        updates = []
        params = tf.all_variables()
        optimizer = tf.train.AdamOptimizer(0.03, epsilon=1e-5)
        for i in range(len(buckets)):
          full_grads = tf.gradients(losses[i], params)
          grads, _ = tf.clip_by_global_norm(full_grads, 30.0)
          update = optimizer.apply_gradients(zip(grads, params))
          updates.append(update)
        sess.run([tf.global_variables_initializer()])
      steps = 6
      for _ in range(steps):
        bucket = random.choice(np.arange(len(buckets)))
        length = buckets[bucket][0]
        i = [np.array([np.random.randint(9) + 1 for _ in range(batch_size)],
                      dtype=np.int32) for _ in range(length)]
        # 0 is our "GO" symbol here.
        o = [np.array([0] * batch_size, dtype=np.int32)] + i
        feed = {}
        for i1, i2, o1, o2 in zip(inp[:length], i[:length],
                                  out[:length], o[:length]):
          feed[i1.name] = i2
          feed[o1.name] = o2
        if length < 8:  # For the 4-bucket, we need the 5th as target.
          feed[out[length].name] = o[length]
        res = sess.run([updates[bucket], losses[bucket]], feed)
        perplexities[bucket].append(math.exp(float(res[1])))
      for bucket in range(len(buckets)):
        if len(perplexities[bucket]) > 1:  # Assert that perplexity went down.
          self.assertLess(perplexities[bucket][-1], perplexities[bucket][0])

  def testModelWithBooleanFeedPrevious(self):
    """Test the model behavior when feed_previous is True.

    For example, the following two cases have the same effect:
      - Train `embedding_rnn_seq2seq` with `feed_previous=True`, which contains
        a `embedding_rnn_decoder` with `feed_previous=True` and
        `update_embedding_for_previous=True`. The decoder is fed with "<Go>"
        and outputs "A, B, C".
      - Train `embedding_rnn_seq2seq` with `feed_previous=False`. The decoder
        is fed with "<Go>, A, B".
    """
    num_encoder_symbols = 3
    num_decoder_symbols = 5
    batch_size = 2
    num_enc_timesteps = 2
    num_dec_timesteps = 3

    def TestModel(seq2seq):
      with self.test_session(graph=tf.Graph()) as sess:
        tf.set_random_seed(111)
        random.seed(111)
        np.random.seed(111)

        enc_inp = [tf.constant(i + 1, tf.int32, shape=[batch_size])
                     for i in range(num_enc_timesteps)]
        dec_inp_fp_true = [tf.constant(i, tf.int32, shape=[batch_size])
                           for i in range(num_dec_timesteps)]
        dec_inp_holder_fp_false = [tf.placeholder(tf.int32, shape=[batch_size])
                                   for _ in range(num_dec_timesteps)]
        targets = [tf.constant(i + 1, tf.int32, shape=[batch_size])
                   for i in range(num_dec_timesteps)]
        weights = [tf.constant(1.0, shape=[batch_size])
                   for i in range(num_dec_timesteps)]

        def ForwardBackward(enc_inp, dec_inp, feed_previous):
          scope_name = "fp_{}".format(feed_previous)
          with tf.variable_scope(scope_name):
            dec_op, _ = seq2seq(enc_inp, dec_inp, feed_previous=feed_previous)
            net_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                              scope_name)
          optimizer = tf.train.AdamOptimizer(0.03, epsilon=1e-5)
          update_op = optimizer.minimize(
              tf.contrib.legacy_seq2seq.sequence_loss(dec_op, targets, weights),
              var_list=net_variables)
          return dec_op, update_op, net_variables

        dec_op_fp_true, update_fp_true, variables_fp_true = ForwardBackward(
            enc_inp, dec_inp_fp_true, feed_previous=True)
        dec_op_fp_false, update_fp_false, variables_fp_false = ForwardBackward(
            enc_inp, dec_inp_holder_fp_false, feed_previous=False)

        sess.run(tf.global_variables_initializer())

        # We only check consistencies between the variables existing in both
        # the models with True and False feed_previous. Variables created by
        # the loop_function in the model with True feed_previous are ignored.
        v_false_name_dict = {v.name.split('/', 1)[-1]: v
                             for v in variables_fp_false}
        matched_variables = [(v, v_false_name_dict[v.name.split('/', 1)[-1]])
                             for v in variables_fp_true]
        for v_true, v_false in matched_variables:
          sess.run(tf.assign(v_false, v_true))

        # Take the symbols generated by the decoder with feed_previous=True as
        # the true input symbols for the decoder with feed_previous=False.
        dec_fp_true = sess.run(dec_op_fp_true)
        output_symbols_fp_true = np.argmax(dec_fp_true, axis=2)
        dec_inp_fp_false = np.vstack((dec_inp_fp_true[0].eval(),
                                      output_symbols_fp_true[:-1]))
        sess.run(update_fp_true)
        sess.run(update_fp_false,
                 {holder: inp for holder, inp in zip(dec_inp_holder_fp_false,
                                                     dec_inp_fp_false)})

        for v_true, v_false in matched_variables:
          self.assertAllClose(v_true.eval(), v_false.eval())

    def EmbeddingRNNSeq2SeqF(enc_inp, dec_inp, feed_previous):
      cell = tf.contrib.rnn.BasicLSTMCell(2, state_is_tuple=True)
      return tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
          enc_inp, dec_inp, cell, num_encoder_symbols,
          num_decoder_symbols, embedding_size=2, feed_previous=feed_previous)

    def EmbeddingRNNSeq2SeqNoTupleF(enc_inp, dec_inp, feed_previous):
      cell = tf.contrib.rnn.BasicLSTMCell(2, state_is_tuple=False)
      return tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
          enc_inp, dec_inp, cell, num_encoder_symbols,
          num_decoder_symbols, embedding_size=2, feed_previous=feed_previous)

    def EmbeddingTiedRNNSeq2Seq(enc_inp, dec_inp, feed_previous):
      cell = tf.contrib.rnn.BasicLSTMCell(2, state_is_tuple=True)
      return tf.contrib.legacy_seq2seq.embedding_tied_rnn_seq2seq(
          enc_inp, dec_inp, cell, num_decoder_symbols, embedding_size=2,
          feed_previous=feed_previous)

    def EmbeddingTiedRNNSeq2SeqNoTuple(enc_inp, dec_inp, feed_previous):
      cell = tf.contrib.rnn.BasicLSTMCell(2, state_is_tuple=False)
      return tf.contrib.legacy_seq2seq.embedding_tied_rnn_seq2seq(
          enc_inp, dec_inp, cell, num_decoder_symbols, embedding_size=2,
          feed_previous=feed_previous)

    def EmbeddingAttentionSeq2Seq(enc_inp, dec_inp, feed_previous):
      cell = tf.contrib.rnn.BasicLSTMCell(2, state_is_tuple=True)
      return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
          enc_inp, dec_inp, cell, num_encoder_symbols,
          num_decoder_symbols, embedding_size=2, feed_previous=feed_previous)

    def EmbeddingAttentionSeq2SeqNoTuple(enc_inp, dec_inp, feed_previous):
      cell = tf.contrib.rnn.BasicLSTMCell(2, state_is_tuple=False)
      return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
          enc_inp, dec_inp, cell, num_encoder_symbols,
          num_decoder_symbols, embedding_size=2, feed_previous=feed_previous)

    for model in (EmbeddingRNNSeq2SeqF, EmbeddingRNNSeq2SeqNoTupleF,
                  EmbeddingTiedRNNSeq2Seq, EmbeddingTiedRNNSeq2SeqNoTuple,
                  EmbeddingAttentionSeq2Seq, EmbeddingAttentionSeq2SeqNoTuple):
      TestModel(model)


if __name__ == "__main__":
  tf.test.main()
