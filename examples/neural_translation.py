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
from tensorflow.models.rnn import rnn_cell, rnn, seq2seq

import skflow

# Get training data

# This dataset can be downloaded from http://www.statmt.org/europarl/v6/fr-en.tgz

def X_iter():
    while True:
        yield "some sentence"
        yield "some other sentence"

def y_iter():
    while True:
        yield u"какое-то приложение" 
        yield u"какое-то другое приложение"

# Translation model

hidden_size = 10

def translate_model(X, y):
    print X.get_shape(), y.get_shape()

#    in_X, in_y, out_y = skflow.ops.seq2seq_inputs(X, y)
#    cell = tf.rnn_cell.GRUCell(hidden_size)
#    decoding = seq2seq.basic_rnn_seq2seq(in_X, in_y, cell)
#    return skflow.ops.sequence_classifier(decoding, out_y)
    return X, y

vocab_processor = skflow.preprocessing.VocabularyProcessor(
    max_document_length=3)
vocab_processor.fit(["some sentence", "some other sentence"])

xiter = vocab_processor.transform(X_iter())
yiter = vocab_processor.transform(y_iter())

translator = skflow.TensorFlowEstimator(model_fn=translate_model,
    n_classes=128)
translator.fit(xiter, yiter)
print translator.predict(X_iter())

