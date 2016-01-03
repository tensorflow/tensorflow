# encoding: utf-8

#  Copyright 2015-present Scikit Flow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np

import tensorflow as tf
from tensorflow.python.ops import rnn_cell, rnn, seq2seq

import skflow

# Get training data

# This dataset can be downloaded from http://www.statmt.org/europarl/v6/fr-en.tgz

def X_iter():
    while True:
        yield "some sentence"
        yield "some other sentence"

def X_predict_iter():
    yield "some sentence"
    yield "some other sentence"

def y_iter():
    while True:
        yield u"какое-то приложение" 
        yield u"какое-то другое приложение"

# Translation model

MAX_DOCUMENT_LENGTH = 3
HIDDEN_SIZE = 10

def rnn_decoder(decoder_inputs, initial_state, cell, scope=None):
    with tf.variable_scope(scope or "dnn_decoder"):
        states, sampling_states = [initial_state], [initial_state]
        outputs, sampling_outputs = [], []
        with tf.op_scope([decoder_inputs, initial_state], "training"):
            for i in xrange(len(decoder_inputs)):
                inp = decoder_inputs[i]
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                output, new_state = cell(inp, states[-1])
                outputs.append(output)
                states.append(new_state)
        with tf.op_scope([initial_state], "sampling"):
            for i in xrange(len(decoder_inputs)):
                if i == 0:
                    sampling_outputs.append(outputs[i])
                    sampling_states.append(states[i])
                else:
                    sampling_output, sampling_state = cell(sampling_outputs[-1], sampling_states[-1])
                    sampling_outputs.append(sampling_output)
                    sampling_states.append(sampling_state)
    return outputs, states, sampling_outputs, sampling_states


def rnn_seq2seq(encoder_inputs, decoder_inputs, cell, dtype=tf.float32, scope=None):
    with tf.variable_scope(scope or "rnn_seq2seq"):
        _, enc_states = rnn.rnn(cell, encoder_inputs, dtype=dtype)
        return rnn_decoder(decoder_inputs, enc_states[-1], cell)


def translate_model(X, y):
    byte_list = skflow.ops.one_hot_matrix(X, 256)
    in_X, in_y, out_y = skflow.ops.seq2seq_inputs(
        byte_list, y, MAX_DOCUMENT_LENGTH, MAX_DOCUMENT_LENGTH)
    cell = rnn_cell.OutputProjectionWrapper(rnn_cell.GRUCell(HIDDEN_SIZE), 256)
    decoding, _, sampling_decoding, _ = rnn_seq2seq(in_X, in_y, cell)
    return skflow.ops.sequence_classifier(decoding, out_y, sampling_decoding)


vocab_processor = skflow.preprocessing.ByteProcessor(
    max_document_length=MAX_DOCUMENT_LENGTH)

x_iter = vocab_processor.transform(X_iter())
y_iter = vocab_processor.transform(y_iter())
xpredict_iter = vocab_processor.transform(X_predict_iter())

translator = skflow.TensorFlowEstimator(model_fn=translate_model,
    n_classes=256)
translator.fit(x_iter, y_iter, logdir='/tmp/tf_examples/ntm/')
translator.save('/tmp/tf_examples/ntm/')
print translator.predict(xpredict_iter)

