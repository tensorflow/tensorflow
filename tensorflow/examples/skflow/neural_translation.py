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
import os
import numpy as np

import tensorflow as tf
from tensorflow.contrib import learn

# Get training data

# This dataset can be downloaded from http://www.statmt.org/europarl/v6/fr-en.tgz

ENGLISH_CORPUS = "europarl-v6.fr-en.en"
FRENCH_CORPUS = "europarl-v6.fr-en.fr"

def read_iterator(filename):
    f = open(filename)
    for line in f:
        yield line.strip()


def repeated_read_iterator(filename):
    while True:
        f = open(filename)
        for line in f:
            yield line.strip()


def split_train_test(data, partition=0.2, random_seed=42):
    rnd = np.random.RandomState(random_seed)
    for item in data:
        if rnd.uniform() > partition:
            yield (0, item)
        else:
            yield (1, item)


def save_partitions(data, filenames):
    files = [open(filename, 'w') for filename in filenames]
    for partition, item in data:
        files[partition].write(item + '\n')


def loop_iterator(data):
    while True:
        for item in data:
            yield item


if not (os.path.exists('train.data') and os.path.exists('test.data')):
    english_data = read_iterator(ENGLISH_CORPUS)
    french_data = read_iterator(FRENCH_CORPUS)
    parallel_data = ('%s;;;%s' % (eng, fr) for eng, fr in itertools.izip(english_data, french_data))
    save_partitions(split_train_test(parallel_data), ['train.data', 'test.data'])

def Xy(data):
    def split_lines(data):
        for item in data:
            yield item.split(';;;')
    X, y = itertools.tee(split_lines(data))
    return (item[0] for item in X), (item[1] for item in y)

X_train, y_train = Xy(repeated_read_iterator('train.data'))
X_test, y_test = Xy(read_iterator('test.data'))


# Translation model

MAX_DOCUMENT_LENGTH = 30
HIDDEN_SIZE = 100

def translate_model(X, y):
    byte_list = learn.ops.one_hot_matrix(X, 256)
    in_X, in_y, out_y = learn.ops.seq2seq_inputs(
        byte_list, y, MAX_DOCUMENT_LENGTH, MAX_DOCUMENT_LENGTH)
    cell = tf.nn.rnn_cell.OutputProjectionWrapper(tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE), 256)
    decoding, _, sampling_decoding, _ = learn.ops.rnn_seq2seq(in_X, in_y, cell)
    return learn.ops.sequence_classifier(decoding, out_y, sampling_decoding)


vocab_processor = learn.preprocessing.ByteProcessor(
    max_document_length=MAX_DOCUMENT_LENGTH)

x_iter = vocab_processor.transform(X_train)
y_iter = vocab_processor.transform(y_train)
xpred = np.array(list(vocab_processor.transform(X_test))[:20])
ygold = list(y_test)[:20]

PATH = '/tmp/tf_examples/ntm/'

if os.path.exists(PATH):
    translator = learn.TensorFlowEstimator.restore(PATH)
else:
    translator = learn.TensorFlowEstimator(model_fn=translate_model,
        n_classes=256,
        optimizer='Adam', learning_rate=0.01, batch_size=128,
        continue_training=True)

while True:
    translator.fit(x_iter, y_iter, logdir=PATH)
    translator.save(PATH)

    predictions = translator.predict(xpred, axis=2)
    xpred_inp = vocab_processor.reverse(xpred)
    text_outputs = vocab_processor.reverse(predictions)
    for inp_data, input_text, pred, output_text, gold in zip(xpred, xpred_inp,
        predictions, text_outputs, ygold):
        print('English: %s. French (pred): %s, French (gold): %s' %
            (input_text, output_text, gold.decode('utf-8')))
        print(inp_data, pred)
