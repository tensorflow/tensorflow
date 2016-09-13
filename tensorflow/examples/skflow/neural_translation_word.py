# encoding: utf-8

#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

import cPickle
import itertools
import os
import random

import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn

# Get training data

# This dataset can be downloaded from http://www.statmt.org/europarl/v6/fr-en.tgz

ENGLISH_CORPUS = "europarl-v6.fr-en.en"
FRENCH_CORPUS = "europarl-v6.fr-en.fr"

def read_iterator(filename, reporting=True):
  f = open(filename)
  line_count = 0
  for line in f:
    line_count += 1
    if reporting and line_count % 100000 == 0:
      print("%d lines read from %s" % (line_count, filename))
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
  parallel_data = ('%s;;;%s' % (eng, fr) for eng, fr in itertools.izip(
      english_data, french_data))
  save_partitions(split_train_test(parallel_data), ['train.data', 'test.data'])

def Xy(data):
  def split_lines(data):
    for item in data:
      yield item.split(';;;')
  X, y = itertools.tee(split_lines(data))
  return (item[0] for item in X), (item[1] for item in y)

X_train, y_train = Xy(repeated_read_iterator('train.data'))
X_test, y_test = Xy(read_iterator('test.data'))

# Preprocessing

MAX_DOCUMENT_LENGTH = 10

if not (os.path.exists('en.vocab') and os.path.exists('fr.vocab')):
  X_vocab_processor = learn.preprocessing.VocabularyProcessor(
      MAX_DOCUMENT_LENGTH, min_frequency=5)
  y_vocab_processor = learn.preprocessing.VocabularyProcessor(
      MAX_DOCUMENT_LENGTH, min_frequency=5)
  Xtrainff, ytrainff = Xy(read_iterator('train.data'))
  print('Fitting dictionary for English...')
  X_vocab_processor.fit(Xtrainff)
  print('Fitting dictionary for French...')
  y_vocab_processor.fit(ytrainff)
  open('en.vocab', 'w').write(cPickle.dumps(X_vocab_processor))
  open('fr.vocab', 'w').write(cPickle.dumps(y_vocab_processor))
else:
  X_vocab_processor = cPickle.loads(open('en.vocab').read())
  y_vocab_processor = cPickle.loads(open('fr.vocab').read())
print('Transforming...')
X_train = X_vocab_processor.transform(X_train)
y_train = y_vocab_processor.transform(y_train)
X_test = X_vocab_processor.transform(X_test)

# TODO: Expand this to use the whole test set.
X_test = np.array([X_test.next() for _ in range(1000)])
y_test = [y_test.next() for _ in range(1000)]

n_en_words = len(X_vocab_processor.vocabulary_)
n_fr_words = len(y_vocab_processor.vocabulary_)
print('Total words, en: %d, fr: %d' % (n_en_words, n_fr_words))

# Translation model

HIDDEN_SIZE = 20
EMBEDDING_SIZE = 20

def translate_model(X, y):
  word_vectors = learn.ops.categorical_variable(X, n_classes=n_en_words,
      embedding_size=EMBEDDING_SIZE, name='words')
  in_X, in_y, out_y = learn.ops.seq2seq_inputs(
      word_vectors, y, MAX_DOCUMENT_LENGTH, MAX_DOCUMENT_LENGTH)
  encoder_cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
  decoder_cell = tf.nn.rnn_cell.OutputProjectionWrapper(
      tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE), n_fr_words)
  decoding, _, sampling_decoding, _ = learn.ops.rnn_seq2seq(in_X, in_y,
      encoder_cell, decoder_cell=decoder_cell)
  return learn.ops.sequence_classifier(decoding, out_y, sampling_decoding)


PATH = '/tmp/tf_examples/ntm_words/'

if os.path.exists(os.path.join(PATH, 'graph.pbtxt')):
  translator = learn.TensorFlowEstimator.restore(PATH)
else:
  translator = learn.TensorFlowEstimator(model_fn=translate_model,
      n_classes=n_fr_words,
      optimizer='Adam', learning_rate=0.01, batch_size=128,
      continue_training=True, steps=100)

while True:
  translator.fit(X_train, y_train, logdir=PATH)
  translator.save(PATH)

  xpred, ygold = [], []
  for _ in range(10):
    idx = random.randint(0, len(X_test) - 1)
    xpred.append(X_test[idx])
    ygold.append(y_test[idx])
  xpred = np.array(xpred)
  predictions = translator.predict(xpred, axis=2)
  xpred_inp = X_vocab_processor.reverse(xpred)
  text_outputs = y_vocab_processor.reverse(predictions)
  for inp_data, input_text, pred, output_text, gold in zip(xpred, xpred_inp,
      predictions, text_outputs, ygold):
    print('English: %s. French (pred): %s, French (gold): %s' %
        (input_text, output_text, gold))
    print(inp_data, pred)
