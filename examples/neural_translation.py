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


def sequence_classifier(decoding, labels):
    predictions, xent_list = [], []
    print decoding, labels
    for i, pred in enumerate(decoding):
        print pred, labels[i]
        xent_list.append(
            tf.nn.softmax_cross_entropy_with_logits(
                pred, labels[i], name="sequence_loss/xent_raw{0}".format(i)))
        predictions.append(tf.nn.softmax(pred))
    xent = tf.add_n(xent_list)
    loss = tf.reduce_mean(xent, name="loss")
    return predictions, loss


def seq2seq_inputs(X, y, input_length, output_length, sentinel=None):
    in_X = skflow.ops.split_squeeze(1, input_length, X)
    y = skflow.ops.split_squeeze(1, output_length, y)
    if not sentinel:
        # Set to zeros of shape of y[0]
        sentinel = tf.zeros(tf.shape(y[0]))
        sentinel.set_shape(y[0].get_shape())
    in_y = [sentinel] + y
    out_y = y + [sentinel]
    return in_X, in_y, out_y
 

def translate_model(X, y):
    byte_list = skflow.ops.one_hot_matrix(X, 256)
    print X.get_shape(), byte_list.get_shape(), y.get_shape()
    in_X, in_y, out_y = seq2seq_inputs(
        byte_list, y, MAX_DOCUMENT_LENGTH, MAX_DOCUMENT_LENGTH)
    cell = rnn_cell.OutputProjectionWrapper(rnn_cell.GRUCell(HIDDEN_SIZE), 256)
    print in_X[0].get_shape()
    for yyy in in_y:
        print yyy.get_shape()
    decoding, _ = seq2seq.basic_rnn_seq2seq(in_X, in_y, cell)
    return sequence_classifier(decoding, out_y)


vocab_processor = skflow.preprocessing.ByteProcessor(
    max_document_length=MAX_DOCUMENT_LENGTH)

x_iter = vocab_processor.transform(X_iter())
y_iter = vocab_processor.transform(y_iter())
xpredict_iter = vocab_processor.transform(X_predict_iter())

translator = skflow.TensorFlowEstimator(model_fn=translate_model,
    n_classes=256)
translator.fit(x_iter, y_iter)
print translator.predict(xpredict_iter)

