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

import random

from sklearn import datasets
from sklearn.metrics import accuracy_score, mean_squared_error

import tensorflow as tf

import skflow

class NonLinearTest(tf.test.TestCase):

    def testIrisDNN(self):
        random.seed(42)
        iris = datasets.load_iris()
        classifier = skflow.TensorFlowDNNClassifier(
            hidden_units=[10, 20, 10], n_classes=3)
        classifier.fit(iris.data, iris.target)
        score = accuracy_score(iris.target, classifier.predict(iris.data))
        self.assertGreater(score, 0.5, "Failed with score = {0}".format(score))

    def testBostonDNN(self):
        random.seed(42)
        boston = datasets.load_boston()
        regressor = skflow.TensorFlowDNNRegressor(
            hidden_units=[10, 20, 10], n_classes=0,
            batch_size=boston.data.shape[0],
            steps=200, learning_rate=0.001)
        regressor.fit(boston.data, boston.target)
        score = mean_squared_error(
            boston.target, regressor.predict(boston.data))
        self.assertLess(score, 100, "Failed with score = {0}".format(score))

    def testIrisRNN(self):
        import numpy as np
        data = ["I can do this", "I believe myself", "I am okay",
                "Not good, man", "Bad mood today", "Feeling sick now"]
        labels = [1, 1, 1, 0, 0, 0]
        MAX_DOCUMENT_LENGTH = 6
        EMBEDDING_SIZE = 10
        vocab_processor = skflow.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
        data = np.array(list(vocab_processor.fit_transform(data)))
        n_words = len(vocab_processor.vocabulary_)

        def input_op_fn(X):
            word_vectors = skflow.ops.categorical_variable(X, n_classes=n_words,
                embedding_size=EMBEDDING_SIZE, name='words')
            word_list = skflow.ops.split_squeeze(1, MAX_DOCUMENT_LENGTH, word_vectors)
            return word_list
        random.seed(42)
        # Only declare them for now
        # TODO: Add test case once we have data set in the repo
        classifier = skflow.TensorFlowRNNClassifier(
            rnn_size=5, cell_type='gru', input_op_fn=input_op_fn, n_classes=3)
        classifier = skflow.TensorFlowRNNRegressor(
            rnn_size=5, cell_type='gru', input_op_fn=input_op_fn, n_classes=0)

if __name__ == "__main__":
    tf.test.main()
