#  Copyright 2015 Google Inc. All Rights Reserved.
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
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss

import skflow

import tensorflow as tf
from tensorflow.python.platform import googletest

class SkFlowTest(googletest.TestCase):

    def testIris(self):
        random.seed(42)
        iris = datasets.load_iris()
        classifier = skflow.TensorFlowClassifier(n_classes=3)
        classifier.fit(iris.data, iris.target)
        score = accuracy_score(classifier.predict(iris.data), iris.target)
        self.assertGreater(score, 0.5, "Failed with score = {0}".format(score))

    def testIris_proba(self):
        random.seed(42)
        iris = datasets.load_iris()
        classifier = skflow.TensorFlowClassifier(n_classes=3)
        classifier.fit(iris.data, iris.target)
        score = log_loss(iris.target, classifier.predict_proba(iris.data))
        self.assertLess(score, 0.6, "Failed with score = {0}".format(score))

    def testBoston(self):
        random.seed(42)
        boston = datasets.load_boston()
        regressor = skflow.TensorFlowRegressor(n_classes=0,
                                               batch_size=boston.data.shape[0],
                                               steps=500,
                                               learning_rate=0.001)
        regressor.fit(boston.data, boston.target)
        score = mean_squared_error(boston.target, regressor.predict(boston.data))
        self.assertLess(score, 150, "Failed with score = {0}".format(score))


if __name__ == "__main__":
    googletest.main()
