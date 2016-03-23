# encoding: utf-8

#  Copyright 2015-present The Scikit Flow Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import math
import os
import numpy as np

import tensorflow as tf
from tensorflow.contrib import skflow

### Training data

CORPUS_FILENAME = "europarl-v6.fr-en.en"
MAX_DOC_LENGTH = 10

def training_data(filename):
    f = open(filename)
    for line in f:
        yield line


def iter_docs(docs):
  for doc in docs:
    n_parts = int(math.ceil(float(len(doc)) / MAX_DOC_LENGTH))
    for part in range(n_parts):
      offset_begin = part * MAX_DOC_LENGTH
      offset_end = offset_begin + MAX_DOC_LENGTH
      inp = np.zeros(MAX_DOC_LENGTH, dtype=np.int32)
      out = np.zeros(MAX_DOC_LENGTH, dtype=np.int32)
      inp[:min(offset_end - offset_begin, len(doc) - offset_begin)] = doc[offset_begin:offset_end]
      out[:min(offset_end - offset_begin, len(doc) - offset_begin - 1)] = doc[offset_begin + 1:offset_end + 1]
      yield inp, out


def unpack_xy(iter_obj):
  X, y = itertools.tee(iter_obj)
  return (item[0] for item in X), (item[1] for item in y)


byte_processor = skflow.preprocessing.ByteProcessor(
    max_document_length=MAX_DOC_LENGTH)

data = training_data(CORPUS_FILENAME)
data = byte_processor.transform(data)
X, y = unpack_xy(iter_docs(data))


### Model

HIDDEN_SIZE = 10


def seq_autoencoder(X, y):
    """Sequence auto-encoder with RNN."""
    inputs = skflow.ops.one_hot_matrix(X, 256)
    in_X, in_y, out_y = skflow.ops.seq2seq_inputs(inputs, y, MAX_DOC_LENGTH, MAX_DOC_LENGTH)
    encoder_cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
    decoder_cell = tf.nn.rnn_cell.OutputProjectionWrapper(tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE), 256)
    decoding, _, sampling_decoding, _ = skflow.ops.rnn_seq2seq(in_X, in_y, encoder_cell, decoder_cell)
    return skflow.ops.sequence_classifier(decoding, out_y, sampling_decoding)


def get_language_model(hidden_size):
    """Returns a language model with given hidden size."""

    def language_model(X, y):
        inputs = skflow.ops.one_hot_matrix(X, 256)
        inputs = skflow.ops.split_squeeze(1, MAX_DOC_LENGTH, inputs)
        target = skflow.ops.split_squeeze(1, MAX_DOC_LENGTH, y)
        encoder_cell = tf.nn.rnn_cell.OutputProjectionWrapper(tf.nn.rnn_cell.GRUCell(hidden_size),256)
        output, _ = tf.nn.rnn(encoder_cell, inputs, dtype=tf.float32)
        return skflow.ops.sequence_classifier(output, target)
  
    return language_model


### Training model.

estimator = skflow.TensorFlowEstimator(model_fn=get_language_model(HIDDEN_SIZE), 
                                       n_classes=256, 
                                       optimizer='Adam', learning_rate=0.01, 
                                       steps=1000, batch_size=64, continue_training=True)

estimator.fit(X, y)
