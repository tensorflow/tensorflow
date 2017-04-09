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

import functools
import math
import random

import numpy as np

from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq as seq2seq_lib
from tensorflow.contrib.rnn.python.ops import core_rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import rnn
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adam


class Seq2SeqTest(test.TestCase):

  def testRNNDecoder(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        inp = [constant_op.constant(0.5, shape=[2, 2])] * 2
        _, enc_state = core_rnn.static_rnn(
            core_rnn_cell_impl.GRUCell(2), inp, dtype=dtypes.float32)
        dec_inp = [constant_op.constant(0.4, shape=[2, 2])] * 3
        cell = core_rnn_cell_impl.OutputProjectionWrapper(
            core_rnn_cell_impl.GRUCell(2), 4)
        dec, mem = seq2seq_lib.rnn_decoder(dec_inp, enc_state, cell)
        sess.run([variables.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 4), res[0].shape)

        res = sess.run([mem])
        self.assertEqual((2, 2), res[0].shape)

  def testBasicRNNSeq2Seq(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        inp = [constant_op.constant(0.5, shape=[2, 2])] * 2
        dec_inp = [constant_op.constant(0.4, shape=[2, 2])] * 3
        cell = core_rnn_cell_impl.OutputProjectionWrapper(
            core_rnn_cell_impl.GRUCell(2), 4)
        dec, mem = seq2seq_lib.basic_rnn_seq2seq(inp, dec_inp, cell)
        sess.run([variables.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 4), res[0].shape)

        res = sess.run([mem])
        self.assertEqual((2, 2), res[0].shape)

  def testTiedRNNSeq2Seq(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        inp = [constant_op.constant(0.5, shape=[2, 2])] * 2
        dec_inp = [constant_op.constant(0.4, shape=[2, 2])] * 3
        cell = core_rnn_cell_impl.OutputProjectionWrapper(
            core_rnn_cell_impl.GRUCell(2), 4)
        dec, mem = seq2seq_lib.tied_rnn_seq2seq(inp, dec_inp, cell)
        sess.run([variables.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 4), res[0].shape)

        res = sess.run([mem])
        self.assertEqual(1, len(res))
        self.assertEqual((2, 2), res[0].shape)

  def testEmbeddingRNNDecoder(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        inp = [constant_op.constant(0.5, shape=[2, 2])] * 2
        cell_fn = lambda: core_rnn_cell_impl.BasicLSTMCell(2)
        cell = cell_fn()
        _, enc_state = core_rnn.static_rnn(cell, inp, dtype=dtypes.float32)
        dec_inp = [
            constant_op.constant(
                i, dtypes.int32, shape=[2]) for i in range(3)
        ]
        # Use a new cell instance since the attention decoder uses a
        # different variable scope.
        dec, mem = seq2seq_lib.embedding_rnn_decoder(
            dec_inp, enc_state, cell_fn(), num_symbols=4, embedding_size=2)
        sess.run([variables.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 2), res[0].shape)

        res = sess.run([mem])
        self.assertEqual(1, len(res))
        self.assertEqual((2, 2), res[0].c.shape)
        self.assertEqual((2, 2), res[0].h.shape)

  def testEmbeddingRNNSeq2Seq(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        enc_inp = [
            constant_op.constant(
                1, dtypes.int32, shape=[2]) for i in range(2)
        ]
        dec_inp = [
            constant_op.constant(
                i, dtypes.int32, shape=[2]) for i in range(3)
        ]
        cell_fn = lambda: core_rnn_cell_impl.BasicLSTMCell(2)
        cell = cell_fn()
        dec, mem = seq2seq_lib.embedding_rnn_seq2seq(
            enc_inp,
            dec_inp,
            cell,
            num_encoder_symbols=2,
            num_decoder_symbols=5,
            embedding_size=2)
        sess.run([variables.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 5), res[0].shape)

        res = sess.run([mem])
        self.assertEqual((2, 2), res[0].c.shape)
        self.assertEqual((2, 2), res[0].h.shape)

        # Test with state_is_tuple=False.
        with variable_scope.variable_scope("no_tuple"):
          cell_nt = core_rnn_cell_impl.BasicLSTMCell(2, state_is_tuple=False)
          dec, mem = seq2seq_lib.embedding_rnn_seq2seq(
              enc_inp,
              dec_inp,
              cell_nt,
              num_encoder_symbols=2,
              num_decoder_symbols=5,
              embedding_size=2)
          sess.run([variables.global_variables_initializer()])
          res = sess.run(dec)
          self.assertEqual(3, len(res))
          self.assertEqual((2, 5), res[0].shape)

          res = sess.run([mem])
          self.assertEqual((2, 4), res[0].shape)

        # Test externally provided output projection.
        w = variable_scope.get_variable("proj_w", [2, 5])
        b = variable_scope.get_variable("proj_b", [5])
        with variable_scope.variable_scope("proj_seq2seq"):
          dec, _ = seq2seq_lib.embedding_rnn_seq2seq(
              enc_inp,
              dec_inp,
              cell_fn(),
              num_encoder_symbols=2,
              num_decoder_symbols=5,
              embedding_size=2,
              output_projection=(w, b))
        sess.run([variables.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 2), res[0].shape)

        # Test that previous-feeding model ignores inputs after the first.
        dec_inp2 = [
            constant_op.constant(
                0, dtypes.int32, shape=[2]) for _ in range(3)
        ]
        with variable_scope.variable_scope("other"):
          d3, _ = seq2seq_lib.embedding_rnn_seq2seq(
              enc_inp,
              dec_inp2,
              cell_fn(),
              num_encoder_symbols=2,
              num_decoder_symbols=5,
              embedding_size=2,
              feed_previous=constant_op.constant(True))
        with variable_scope.variable_scope("other_2"):
          d1, _ = seq2seq_lib.embedding_rnn_seq2seq(
              enc_inp,
              dec_inp,
              cell_fn(),
              num_encoder_symbols=2,
              num_decoder_symbols=5,
              embedding_size=2,
              feed_previous=True)
        with variable_scope.variable_scope("other_3"):
          d2, _ = seq2seq_lib.embedding_rnn_seq2seq(
              enc_inp,
              dec_inp2,
              cell_fn(),
              num_encoder_symbols=2,
              num_decoder_symbols=5,
              embedding_size=2,
              feed_previous=True)
        sess.run([variables.global_variables_initializer()])
        res1 = sess.run(d1)
        res2 = sess.run(d2)
        res3 = sess.run(d3)
        self.assertAllClose(res1, res2)
        self.assertAllClose(res1, res3)

  def testEmbeddingTiedRNNSeq2Seq(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        enc_inp = [
            constant_op.constant(
                1, dtypes.int32, shape=[2]) for i in range(2)
        ]
        dec_inp = [
            constant_op.constant(
                i, dtypes.int32, shape=[2]) for i in range(3)
        ]
        cell = functools.partial(
            core_rnn_cell_impl.BasicLSTMCell,
            2, state_is_tuple=True)
        dec, mem = seq2seq_lib.embedding_tied_rnn_seq2seq(
            enc_inp, dec_inp, cell(), num_symbols=5, embedding_size=2)
        sess.run([variables.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 5), res[0].shape)

        res = sess.run([mem])
        self.assertEqual((2, 2), res[0].c.shape)
        self.assertEqual((2, 2), res[0].h.shape)

        # Test when num_decoder_symbols is provided, the size of decoder output
        # is num_decoder_symbols.
        with variable_scope.variable_scope("decoder_symbols_seq2seq"):
          dec, mem = seq2seq_lib.embedding_tied_rnn_seq2seq(
              enc_inp,
              dec_inp,
              cell(),
              num_symbols=5,
              num_decoder_symbols=3,
              embedding_size=2)
        sess.run([variables.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 3), res[0].shape)

        # Test externally provided output projection.
        w = variable_scope.get_variable("proj_w", [2, 5])
        b = variable_scope.get_variable("proj_b", [5])
        with variable_scope.variable_scope("proj_seq2seq"):
          dec, _ = seq2seq_lib.embedding_tied_rnn_seq2seq(
              enc_inp,
              dec_inp,
              cell(),
              num_symbols=5,
              embedding_size=2,
              output_projection=(w, b))
        sess.run([variables.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 2), res[0].shape)

        # Test that previous-feeding model ignores inputs after the first.
        dec_inp2 = [constant_op.constant(0, dtypes.int32, shape=[2])] * 3
        with variable_scope.variable_scope("other"):
          d3, _ = seq2seq_lib.embedding_tied_rnn_seq2seq(
              enc_inp,
              dec_inp2,
              cell(),
              num_symbols=5,
              embedding_size=2,
              feed_previous=constant_op.constant(True))
        with variable_scope.variable_scope("other_2"):
          d1, _ = seq2seq_lib.embedding_tied_rnn_seq2seq(
              enc_inp,
              dec_inp,
              cell(),
              num_symbols=5,
              embedding_size=2,
              feed_previous=True)
        with variable_scope.variable_scope("other_3"):
          d2, _ = seq2seq_lib.embedding_tied_rnn_seq2seq(
              enc_inp,
              dec_inp2,
              cell(),
              num_symbols=5,
              embedding_size=2,
              feed_previous=True)
        sess.run([variables.global_variables_initializer()])
        res1 = sess.run(d1)
        res2 = sess.run(d2)
        res3 = sess.run(d3)
        self.assertAllClose(res1, res2)
        self.assertAllClose(res1, res3)

  def testAttentionDecoder1(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        cell_fn = lambda: core_rnn_cell_impl.GRUCell(2)
        cell = cell_fn()
        inp = [constant_op.constant(0.5, shape=[2, 2])] * 2
        enc_outputs, enc_state = core_rnn.static_rnn(
            cell, inp, dtype=dtypes.float32)
        attn_states = array_ops.concat([
            array_ops.reshape(e, [-1, 1, cell.output_size]) for e in enc_outputs
        ], 1)
        dec_inp = [constant_op.constant(0.4, shape=[2, 2])] * 3

        # Create a new cell instance for the decoder, since it uses a
        # different variable scope
        dec, mem = seq2seq_lib.attention_decoder(
            dec_inp, enc_state, attn_states, cell_fn(), output_size=4)
        sess.run([variables.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 4), res[0].shape)

        res = sess.run([mem])
        self.assertEqual((2, 2), res[0].shape)

  def testAttentionDecoder2(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        cell_fn = lambda: core_rnn_cell_impl.GRUCell(2)
        cell = cell_fn()
        inp = [constant_op.constant(0.5, shape=[2, 2])] * 2
        enc_outputs, enc_state = core_rnn.static_rnn(
            cell, inp, dtype=dtypes.float32)
        attn_states = array_ops.concat([
            array_ops.reshape(e, [-1, 1, cell.output_size]) for e in enc_outputs
        ], 1)
        dec_inp = [constant_op.constant(0.4, shape=[2, 2])] * 3

        # Use a new cell instance since the attention decoder uses a
        # different variable scope.
        dec, mem = seq2seq_lib.attention_decoder(
            dec_inp, enc_state, attn_states, cell_fn(),
            output_size=4, num_heads=2)
        sess.run([variables.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 4), res[0].shape)

        res = sess.run([mem])
        self.assertEqual((2, 2), res[0].shape)

  def testDynamicAttentionDecoder1(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        cell_fn = lambda: core_rnn_cell_impl.GRUCell(2)
        cell = cell_fn()
        inp = constant_op.constant(0.5, shape=[2, 2, 2])
        enc_outputs, enc_state = rnn.dynamic_rnn(
            cell, inp, dtype=dtypes.float32)
        attn_states = enc_outputs
        dec_inp = [constant_op.constant(0.4, shape=[2, 2])] * 3

        # Use a new cell instance since the attention decoder uses a
        # different variable scope.
        dec, mem = seq2seq_lib.attention_decoder(
            dec_inp, enc_state, attn_states, cell_fn(), output_size=4)
        sess.run([variables.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 4), res[0].shape)

        res = sess.run([mem])
        self.assertEqual((2, 2), res[0].shape)

  def testDynamicAttentionDecoder2(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        cell_fn = lambda: core_rnn_cell_impl.GRUCell(2)
        cell = cell_fn()
        inp = constant_op.constant(0.5, shape=[2, 2, 2])
        enc_outputs, enc_state = rnn.dynamic_rnn(
            cell, inp, dtype=dtypes.float32)
        attn_states = enc_outputs
        dec_inp = [constant_op.constant(0.4, shape=[2, 2])] * 3

        # Use a new cell instance since the attention decoder uses a
        # different variable scope.
        dec, mem = seq2seq_lib.attention_decoder(
            dec_inp, enc_state, attn_states, cell_fn(),
            output_size=4, num_heads=2)
        sess.run([variables.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 4), res[0].shape)

        res = sess.run([mem])
        self.assertEqual((2, 2), res[0].shape)

  def testAttentionDecoderStateIsTuple(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        single_cell = lambda: core_rnn_cell_impl.BasicLSTMCell(  # pylint: disable=g-long-lambda
            2, state_is_tuple=True)
        cell_fn = lambda: core_rnn_cell_impl.MultiRNNCell(  # pylint: disable=g-long-lambda
            cells=[single_cell() for _ in range(2)], state_is_tuple=True)
        cell = cell_fn()
        inp = [constant_op.constant(0.5, shape=[2, 2])] * 2
        enc_outputs, enc_state = core_rnn.static_rnn(
            cell, inp, dtype=dtypes.float32)
        attn_states = array_ops.concat([
            array_ops.reshape(e, [-1, 1, cell.output_size]) for e in enc_outputs
        ], 1)
        dec_inp = [constant_op.constant(0.4, shape=[2, 2])] * 3

        # Use a new cell instance since the attention decoder uses a
        # different variable scope.
        dec, mem = seq2seq_lib.attention_decoder(
            dec_inp, enc_state, attn_states, cell_fn(), output_size=4)
        sess.run([variables.global_variables_initializer()])
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
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        cell_fn = lambda: core_rnn_cell_impl.MultiRNNCell(  # pylint: disable=g-long-lambda
            cells=[core_rnn_cell_impl.BasicLSTMCell(2) for _ in range(2)])
        cell = cell_fn()
        inp = [constant_op.constant(0.5, shape=[2, 2])] * 2
        enc_outputs, enc_state = core_rnn.static_rnn(
            cell, inp, dtype=dtypes.float32)
        attn_states = array_ops.concat([
            array_ops.reshape(e, [-1, 1, cell.output_size])
            for e in enc_outputs
        ], 1)
        dec_inp = [constant_op.constant(0.4, shape=[2, 2])] * 3

        # Use a new cell instance since the attention decoder uses a
        # different variable scope.
        dec, mem = seq2seq_lib.attention_decoder(
            dec_inp, enc_state, attn_states, cell_fn(), output_size=4)
        sess.run([variables.global_variables_initializer()])
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
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        inp = [constant_op.constant(0.5, shape=[2, 2])] * 2
        cell_fn = lambda: core_rnn_cell_impl.GRUCell(2)
        cell = cell_fn()
        enc_outputs, enc_state = core_rnn.static_rnn(
            cell, inp, dtype=dtypes.float32)
        attn_states = array_ops.concat([
            array_ops.reshape(e, [-1, 1, cell.output_size]) for e in enc_outputs
        ], 1)
        dec_inp = [
            constant_op.constant(
                i, dtypes.int32, shape=[2]) for i in range(3)
        ]

        # Use a new cell instance since the attention decoder uses a
        # different variable scope.
        dec, mem = seq2seq_lib.embedding_attention_decoder(
            dec_inp,
            enc_state,
            attn_states,
            cell_fn(),
            num_symbols=4,
            embedding_size=2,
            output_size=3)
        sess.run([variables.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 3), res[0].shape)

        res = sess.run([mem])
        self.assertEqual((2, 2), res[0].shape)

  def testEmbeddingAttentionSeq2Seq(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        enc_inp = [
            constant_op.constant(
                1, dtypes.int32, shape=[2]) for i in range(2)
        ]
        dec_inp = [
            constant_op.constant(
                i, dtypes.int32, shape=[2]) for i in range(3)
        ]
        cell_fn = lambda: core_rnn_cell_impl.BasicLSTMCell(2)
        cell = cell_fn()
        dec, mem = seq2seq_lib.embedding_attention_seq2seq(
            enc_inp,
            dec_inp,
            cell,
            num_encoder_symbols=2,
            num_decoder_symbols=5,
            embedding_size=2)
        sess.run([variables.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 5), res[0].shape)

        res = sess.run([mem])
        self.assertEqual((2, 2), res[0].c.shape)
        self.assertEqual((2, 2), res[0].h.shape)

        # Test with state_is_tuple=False.
        with variable_scope.variable_scope("no_tuple"):
          cell_fn = functools.partial(
              core_rnn_cell_impl.BasicLSTMCell,
              2, state_is_tuple=False)
          cell_nt = cell_fn()
          dec, mem = seq2seq_lib.embedding_attention_seq2seq(
              enc_inp,
              dec_inp,
              cell_nt,
              num_encoder_symbols=2,
              num_decoder_symbols=5,
              embedding_size=2)
          sess.run([variables.global_variables_initializer()])
          res = sess.run(dec)
          self.assertEqual(3, len(res))
          self.assertEqual((2, 5), res[0].shape)

          res = sess.run([mem])
          self.assertEqual((2, 4), res[0].shape)

        # Test externally provided output projection.
        w = variable_scope.get_variable("proj_w", [2, 5])
        b = variable_scope.get_variable("proj_b", [5])
        with variable_scope.variable_scope("proj_seq2seq"):
          dec, _ = seq2seq_lib.embedding_attention_seq2seq(
              enc_inp,
              dec_inp,
              cell_fn(),
              num_encoder_symbols=2,
              num_decoder_symbols=5,
              embedding_size=2,
              output_projection=(w, b))
        sess.run([variables.global_variables_initializer()])
        res = sess.run(dec)
        self.assertEqual(3, len(res))
        self.assertEqual((2, 2), res[0].shape)

        # TODO(ebrevdo, lukaszkaiser): Re-enable once RNNCells allow reuse
        # within a variable scope that already has a weights tensor.
        #
        # # Test that previous-feeding model ignores inputs after the first.
        # dec_inp2 = [
        #     constant_op.constant(
        #         0, dtypes.int32, shape=[2]) for _ in range(3)
        # ]
        # with variable_scope.variable_scope("other"):
        #   d3, _ = seq2seq_lib.embedding_attention_seq2seq(
        #       enc_inp,
        #       dec_inp2,
        #       cell_fn(),
        #       num_encoder_symbols=2,
        #       num_decoder_symbols=5,
        #       embedding_size=2,
        #       feed_previous=constant_op.constant(True))
        # sess.run([variables.global_variables_initializer()])
        # variable_scope.get_variable_scope().reuse_variables()
        # cell = cell_fn()
        # d1, _ = seq2seq_lib.embedding_attention_seq2seq(
        #     enc_inp,
        #     dec_inp,
        #     cell,
        #     num_encoder_symbols=2,
        #     num_decoder_symbols=5,
        #     embedding_size=2,
        #     feed_previous=True)
        # d2, _ = seq2seq_lib.embedding_attention_seq2seq(
        #     enc_inp,
        #     dec_inp2,
        #     cell,
        #     num_encoder_symbols=2,
        #     num_decoder_symbols=5,
        #     embedding_size=2,
        #     feed_previous=True)
        # res1 = sess.run(d1)
        # res2 = sess.run(d2)
        # res3 = sess.run(d3)
        # self.assertAllClose(res1, res2)
        # self.assertAllClose(res1, res3)

  def testOne2ManyRNNSeq2Seq(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        enc_inp = [
            constant_op.constant(
                1, dtypes.int32, shape=[2]) for i in range(2)
        ]
        dec_inp_dict = {}
        dec_inp_dict["0"] = [
            constant_op.constant(
                i, dtypes.int32, shape=[2]) for i in range(3)
        ]
        dec_inp_dict["1"] = [
            constant_op.constant(
                i, dtypes.int32, shape=[2]) for i in range(4)
        ]
        dec_symbols_dict = {"0": 5, "1": 6}
        def EncCellFn():
          return core_rnn_cell_impl.BasicLSTMCell(2, state_is_tuple=True)
        def DecCellsFn():
          return dict(
              (k, core_rnn_cell_impl.BasicLSTMCell(2, state_is_tuple=True))
              for k in dec_symbols_dict)
        outputs_dict, state_dict = (seq2seq_lib.one2many_rnn_seq2seq(
            enc_inp, dec_inp_dict, EncCellFn(), DecCellsFn(),
            2, dec_symbols_dict, embedding_size=2))

        sess.run([variables.global_variables_initializer()])
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
            constant_op.constant(
                0, dtypes.int32, shape=[2]) for _ in range(3)
        ]
        dec_inp_dict2["1"] = [
            constant_op.constant(
                0, dtypes.int32, shape=[2]) for _ in range(4)
        ]
        with variable_scope.variable_scope("other"):
          outputs_dict3, _ = seq2seq_lib.one2many_rnn_seq2seq(
              enc_inp,
              dec_inp_dict2,
              EncCellFn(),
              DecCellsFn(),
              2,
              dec_symbols_dict,
              embedding_size=2,
              feed_previous=constant_op.constant(True))
        with variable_scope.variable_scope("other_2"):
          outputs_dict1, _ = seq2seq_lib.one2many_rnn_seq2seq(
              enc_inp,
              dec_inp_dict,
              EncCellFn(),
              DecCellsFn(),
              2,
              dec_symbols_dict,
              embedding_size=2,
              feed_previous=True)
        with variable_scope.variable_scope("other_3"):
          outputs_dict2, _ = seq2seq_lib.one2many_rnn_seq2seq(
              enc_inp,
              dec_inp_dict2,
              EncCellFn(),
              DecCellsFn(),
              2,
              dec_symbols_dict,
              embedding_size=2,
              feed_previous=True)
        sess.run([variables.global_variables_initializer()])
        res1 = sess.run(outputs_dict1["0"])
        res2 = sess.run(outputs_dict2["0"])
        res3 = sess.run(outputs_dict3["0"])
        self.assertAllClose(res1, res2)
        self.assertAllClose(res1, res3)

  def testSequenceLoss(self):
    with self.test_session() as sess:
      logits = [constant_op.constant(i + 0.5, shape=[2, 5]) for i in range(3)]
      targets = [
          constant_op.constant(
              i, dtypes.int32, shape=[2]) for i in range(3)
      ]
      weights = [constant_op.constant(1.0, shape=[2]) for i in range(3)]

      average_loss_per_example = seq2seq_lib.sequence_loss(
          logits,
          targets,
          weights,
          average_across_timesteps=True,
          average_across_batch=True)
      res = sess.run(average_loss_per_example)
      self.assertAllClose(1.60944, res)

      average_loss_per_sequence = seq2seq_lib.sequence_loss(
          logits,
          targets,
          weights,
          average_across_timesteps=False,
          average_across_batch=True)
      res = sess.run(average_loss_per_sequence)
      self.assertAllClose(4.828314, res)

      total_loss = seq2seq_lib.sequence_loss(
          logits,
          targets,
          weights,
          average_across_timesteps=False,
          average_across_batch=False)
      res = sess.run(total_loss)
      self.assertAllClose(9.656628, res)

  def testSequenceLossByExample(self):
    with self.test_session() as sess:
      output_classes = 5
      logits = [
          constant_op.constant(
              i + 0.5, shape=[2, output_classes]) for i in range(3)
      ]
      targets = [
          constant_op.constant(
              i, dtypes.int32, shape=[2]) for i in range(3)
      ]
      weights = [constant_op.constant(1.0, shape=[2]) for i in range(3)]

      average_loss_per_example = (seq2seq_lib.sequence_loss_by_example(
          logits, targets, weights, average_across_timesteps=True))
      res = sess.run(average_loss_per_example)
      self.assertAllClose(np.asarray([1.609438, 1.609438]), res)

      loss_per_sequence = seq2seq_lib.sequence_loss_by_example(
          logits, targets, weights, average_across_timesteps=False)
      res = sess.run(loss_per_sequence)
      self.assertAllClose(np.asarray([4.828314, 4.828314]), res)

  # TODO(ebrevdo, lukaszkaiser): Re-enable once RNNCells allow reuse
  # within a variable scope that already has a weights tensor.
  #
  # def testModelWithBucketsScopeAndLoss(self):
  #   """Test variable scope reuse is not reset after model_with_buckets."""
  #   classes = 10
  #   buckets = [(4, 4), (8, 8)]

  #   with self.test_session():
  #     # Here comes a sample Seq2Seq model using GRU cells.
  #     def SampleGRUSeq2Seq(enc_inp, dec_inp, weights, per_example_loss):
  #       """Example sequence-to-sequence model that uses GRU cells."""

  #       def GRUSeq2Seq(enc_inp, dec_inp):
  #         cell = core_rnn_cell_impl.MultiRNNCell(
  #             [core_rnn_cell_impl.GRUCell(24) for _ in range(2)])
  #         return seq2seq_lib.embedding_attention_seq2seq(
  #             enc_inp,
  #             dec_inp,
  #             cell,
  #             num_encoder_symbols=classes,
  #             num_decoder_symbols=classes,
  #             embedding_size=24)

  #       targets = [dec_inp[i + 1] for i in range(len(dec_inp) - 1)] + [0]
  #       return seq2seq_lib.model_with_buckets(
  #           enc_inp,
  #           dec_inp,
  #           targets,
  #           weights,
  #           buckets,
  #           GRUSeq2Seq,
  #           per_example_loss=per_example_loss)

  #     # Now we construct the copy model.
  #     inp = [
  #         array_ops.placeholder(
  #             dtypes.int32, shape=[None]) for _ in range(8)
  #     ]
  #     out = [
  #         array_ops.placeholder(
  #             dtypes.int32, shape=[None]) for _ in range(8)
  #     ]
  #     weights = [
  #         array_ops.ones_like(
  #             inp[0], dtype=dtypes.float32) for _ in range(8)
  #     ]
  #     with variable_scope.variable_scope("root"):
  #       _, losses1 = SampleGRUSeq2Seq(
  #           inp, out, weights, per_example_loss=False)
  #       # Now check that we did not accidentally set reuse.
  #       self.assertEqual(False, variable_scope.get_variable_scope().reuse)
  #     with variable_scope.variable_scope("new"):
  #       _, losses2 = SampleGRUSeq2Seq
  #           inp, out, weights, per_example_loss=True)
  #       # First loss is scalar, the second one is a 1-dimensinal tensor.
  #       self.assertEqual([], losses1[0].get_shape().as_list())
  #       self.assertEqual([None], losses2[0].get_shape().as_list())

  def testModelWithBuckets(self):
    """Larger tests that does full sequence-to-sequence model training."""
    # We learn to copy 10 symbols in 2 buckets: length 4 and length 8.
    classes = 10
    buckets = [(4, 4), (8, 8)]
    perplexities = [[], []]  # Results for each bucket.
    random_seed.set_random_seed(111)
    random.seed(111)
    np.random.seed(111)

    with self.test_session() as sess:
      # We use sampled softmax so we keep output projection separate.
      w = variable_scope.get_variable("proj_w", [24, classes])
      w_t = array_ops.transpose(w)
      b = variable_scope.get_variable("proj_b", [classes])

      # Here comes a sample Seq2Seq model using GRU cells.
      def SampleGRUSeq2Seq(enc_inp, dec_inp, weights):
        """Example sequence-to-sequence model that uses GRU cells."""

        def GRUSeq2Seq(enc_inp, dec_inp):
          cell = core_rnn_cell_impl.MultiRNNCell(
              [core_rnn_cell_impl.GRUCell(24) for _ in range(2)],
              state_is_tuple=True)
          return seq2seq_lib.embedding_attention_seq2seq(
              enc_inp,
              dec_inp,
              cell,
              num_encoder_symbols=classes,
              num_decoder_symbols=classes,
              embedding_size=24,
              output_projection=(w, b))

        targets = [dec_inp[i + 1] for i in range(len(dec_inp) - 1)] + [0]

        def SampledLoss(labels, logits):
          labels = array_ops.reshape(labels, [-1, 1])
          return nn_impl.sampled_softmax_loss(
              weights=w_t,
              biases=b,
              labels=labels,
              inputs=logits,
              num_sampled=8,
              num_classes=classes)

        return seq2seq_lib.model_with_buckets(
            enc_inp,
            dec_inp,
            targets,
            weights,
            buckets,
            GRUSeq2Seq,
            softmax_loss_function=SampledLoss)

      # Now we construct the copy model.
      batch_size = 8
      inp = [
          array_ops.placeholder(
              dtypes.int32, shape=[None]) for _ in range(8)
      ]
      out = [
          array_ops.placeholder(
              dtypes.int32, shape=[None]) for _ in range(8)
      ]
      weights = [
          array_ops.ones_like(
              inp[0], dtype=dtypes.float32) for _ in range(8)
      ]
      with variable_scope.variable_scope("root"):
        _, losses = SampleGRUSeq2Seq(inp, out, weights)
        updates = []
        params = variables.global_variables()
        optimizer = adam.AdamOptimizer(0.03, epsilon=1e-5)
        for i in range(len(buckets)):
          full_grads = gradients_impl.gradients(losses[i], params)
          grads, _ = clip_ops.clip_by_global_norm(full_grads, 30.0)
          update = optimizer.apply_gradients(zip(grads, params))
          updates.append(update)
        sess.run([variables.global_variables_initializer()])
      steps = 6
      for _ in range(steps):
        bucket = random.choice(np.arange(len(buckets)))
        length = buckets[bucket][0]
        i = [
            np.array(
                [np.random.randint(9) + 1 for _ in range(batch_size)],
                dtype=np.int32) for _ in range(length)
        ]
        # 0 is our "GO" symbol here.
        o = [np.array([0] * batch_size, dtype=np.int32)] + i
        feed = {}
        for i1, i2, o1, o2 in zip(inp[:length], i[:length], out[:length],
                                  o[:length]):
          feed[i1.name] = i2
          feed[o1.name] = o2
        if length < 8:  # For the 4-bucket, we need the 5th as target.
          feed[out[length].name] = o[length]
        res = sess.run([updates[bucket], losses[bucket]], feed)
        perplexities[bucket].append(math.exp(float(res[1])))
      for bucket in range(len(buckets)):
        if len(perplexities[bucket]) > 1:  # Assert that perplexity went down.
          self.assertLess(perplexities[bucket][-1],  # 10% margin of error.
                          1.1 * perplexities[bucket][0])

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
      with self.test_session(graph=ops.Graph()) as sess:
        random_seed.set_random_seed(111)
        random.seed(111)
        np.random.seed(111)

        enc_inp = [
            constant_op.constant(
                i + 1, dtypes.int32, shape=[batch_size])
            for i in range(num_enc_timesteps)
        ]
        dec_inp_fp_true = [
            constant_op.constant(
                i, dtypes.int32, shape=[batch_size])
            for i in range(num_dec_timesteps)
        ]
        dec_inp_holder_fp_false = [
            array_ops.placeholder(
                dtypes.int32, shape=[batch_size])
            for _ in range(num_dec_timesteps)
        ]
        targets = [
            constant_op.constant(
                i + 1, dtypes.int32, shape=[batch_size])
            for i in range(num_dec_timesteps)
        ]
        weights = [
            constant_op.constant(
                1.0, shape=[batch_size]) for i in range(num_dec_timesteps)
        ]

        def ForwardBackward(enc_inp, dec_inp, feed_previous):
          scope_name = "fp_{}".format(feed_previous)
          with variable_scope.variable_scope(scope_name):
            dec_op, _ = seq2seq(enc_inp, dec_inp, feed_previous=feed_previous)
            net_variables = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES,
                                               scope_name)
          optimizer = adam.AdamOptimizer(0.03, epsilon=1e-5)
          update_op = optimizer.minimize(
              seq2seq_lib.sequence_loss(dec_op, targets, weights),
              var_list=net_variables)
          return dec_op, update_op, net_variables

        dec_op_fp_true, update_fp_true, variables_fp_true = ForwardBackward(
            enc_inp, dec_inp_fp_true, feed_previous=True)
        _, update_fp_false, variables_fp_false = ForwardBackward(
            enc_inp, dec_inp_holder_fp_false, feed_previous=False)

        sess.run(variables.global_variables_initializer())

        # We only check consistencies between the variables existing in both
        # the models with True and False feed_previous. Variables created by
        # the loop_function in the model with True feed_previous are ignored.
        v_false_name_dict = {
            v.name.split("/", 1)[-1]: v
            for v in variables_fp_false
        }
        matched_variables = [(v, v_false_name_dict[v.name.split("/", 1)[-1]])
                             for v in variables_fp_true]
        for v_true, v_false in matched_variables:
          sess.run(state_ops.assign(v_false, v_true))

        # Take the symbols generated by the decoder with feed_previous=True as
        # the true input symbols for the decoder with feed_previous=False.
        dec_fp_true = sess.run(dec_op_fp_true)
        output_symbols_fp_true = np.argmax(dec_fp_true, axis=2)
        dec_inp_fp_false = np.vstack((dec_inp_fp_true[0].eval(),
                                      output_symbols_fp_true[:-1]))
        sess.run(update_fp_true)
        sess.run(update_fp_false, {
            holder: inp
            for holder, inp in zip(dec_inp_holder_fp_false, dec_inp_fp_false)
        })

        for v_true, v_false in matched_variables:
          self.assertAllClose(v_true.eval(), v_false.eval())

    def EmbeddingRNNSeq2SeqF(enc_inp, dec_inp, feed_previous):
      cell = core_rnn_cell_impl.BasicLSTMCell(2, state_is_tuple=True)
      return seq2seq_lib.embedding_rnn_seq2seq(
          enc_inp,
          dec_inp,
          cell,
          num_encoder_symbols,
          num_decoder_symbols,
          embedding_size=2,
          feed_previous=feed_previous)

    def EmbeddingRNNSeq2SeqNoTupleF(enc_inp, dec_inp, feed_previous):
      cell = core_rnn_cell_impl.BasicLSTMCell(2, state_is_tuple=False)
      return seq2seq_lib.embedding_rnn_seq2seq(
          enc_inp,
          dec_inp,
          cell,
          num_encoder_symbols,
          num_decoder_symbols,
          embedding_size=2,
          feed_previous=feed_previous)

    def EmbeddingTiedRNNSeq2Seq(enc_inp, dec_inp, feed_previous):
      cell = core_rnn_cell_impl.BasicLSTMCell(2, state_is_tuple=True)
      return seq2seq_lib.embedding_tied_rnn_seq2seq(
          enc_inp,
          dec_inp,
          cell,
          num_decoder_symbols,
          embedding_size=2,
          feed_previous=feed_previous)

    def EmbeddingTiedRNNSeq2SeqNoTuple(enc_inp, dec_inp, feed_previous):
      cell = core_rnn_cell_impl.BasicLSTMCell(2, state_is_tuple=False)
      return seq2seq_lib.embedding_tied_rnn_seq2seq(
          enc_inp,
          dec_inp,
          cell,
          num_decoder_symbols,
          embedding_size=2,
          feed_previous=feed_previous)

    def EmbeddingAttentionSeq2Seq(enc_inp, dec_inp, feed_previous):
      cell = core_rnn_cell_impl.BasicLSTMCell(2, state_is_tuple=True)
      return seq2seq_lib.embedding_attention_seq2seq(
          enc_inp,
          dec_inp,
          cell,
          num_encoder_symbols,
          num_decoder_symbols,
          embedding_size=2,
          feed_previous=feed_previous)

    def EmbeddingAttentionSeq2SeqNoTuple(enc_inp, dec_inp, feed_previous):
      cell = core_rnn_cell_impl.BasicLSTMCell(2, state_is_tuple=False)
      return seq2seq_lib.embedding_attention_seq2seq(
          enc_inp,
          dec_inp,
          cell,
          num_encoder_symbols,
          num_decoder_symbols,
          embedding_size=2,
          feed_previous=feed_previous)

    for model in (EmbeddingRNNSeq2SeqF, EmbeddingRNNSeq2SeqNoTupleF,
                  EmbeddingTiedRNNSeq2Seq, EmbeddingTiedRNNSeq2SeqNoTuple,
                  EmbeddingAttentionSeq2Seq, EmbeddingAttentionSeq2SeqNoTuple):
      TestModel(model)


if __name__ == "__main__":
  test.main()
