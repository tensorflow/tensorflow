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

"""
This is an example of using recurrent neural networks over characters
for DBpedia dataset to predict class from description of an entity.

This model is similar to one described in this paper:
   "Character-level Convolutional Networks for Text Classification"
   http://arxiv.org/abs/1509.01626

and is somewhat alternative to the Lua code from here:
   https://github.com/zhangxiangxiao/Crepe
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import metrics
import pandas

import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
from tensorflow.contrib import skflow

### Training data

# Download dbpedia_csv.tar.gz from
# https://drive.google.com/folderview?id=0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M
# Unpack: tar -xvf dbpedia_csv.tar.gz

train = pandas.read_csv('dbpedia_csv/train.csv', header=None)
X_train, y_train = train[2], train[0]
test = pandas.read_csv('dbpedia_csv/test.csv', header=None)
X_test, y_test = test[2], test[0]

### Process vocabulary

MAX_DOCUMENT_LENGTH = 100

char_processor = skflow.preprocessing.ByteProcessor(MAX_DOCUMENT_LENGTH)
X_train = np.array(list(char_processor.fit_transform(X_train)))
X_test = np.array(list(char_processor.transform(X_test)))

### Models

HIDDEN_SIZE = 20

def char_rnn_model(X, y):
    byte_list = skflow.ops.one_hot_matrix(X, 256)
    byte_list = skflow.ops.split_squeeze(1, MAX_DOCUMENT_LENGTH, byte_list)
    cell = rnn_cell.GRUCell(HIDDEN_SIZE)
    _, encoding = rnn.rnn(cell, byte_list, dtype=tf.float32)
    return skflow.models.logistic_regression(encoding, y)

classifier = skflow.TensorFlowEstimator(model_fn=char_rnn_model, n_classes=15,
    steps=100, optimizer='Adam', learning_rate=0.01, continue_training=True)

# Continuously train for 1000 steps & predict on test set.
while True:
    classifier.fit(X_train, y_train)
    score = metrics.accuracy_score(y_test, classifier.predict(X_test))
    print("Accuracy: %f" % score)
