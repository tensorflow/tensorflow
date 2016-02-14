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
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss

import numpy as np
import tensorflow as tf

import skflow
from skflow.estimators import base


class BaseTest(tf.test.TestCase):

    def testOneDim(self):
        random.seed(42)
        X = np.random.rand(1000)
        y = 2 * X + 3
        regressor = skflow.TensorFlowLinearRegressor()
        regressor.fit(X, y)
        score = mean_squared_error(regressor.predict(X), y)
        self.assertLess(score, 0.3, "Failed with score = {0}".format(score))

    def testIris(self):
        iris = datasets.load_iris()
        classifier = skflow.TensorFlowLinearClassifier(n_classes=3)
        classifier.fit(iris.data, iris.target)
        score = accuracy_score(iris.target, classifier.predict(iris.data))
        self.assertGreater(score, 0.7, "Failed with score = {0}".format(score))

    def testIrisClassWeight(self):
        iris = datasets.load_iris()
        classifier = skflow.TensorFlowLinearClassifier(
            n_classes=3, class_weight=[0.1, 0.8, 0.1])
        classifier.fit(iris.data, iris.target)
        score = accuracy_score(iris.target, classifier.predict(iris.data))
        self.assertLess(score, 0.7, "Failed with score = {0}".format(score))

    def testIrisSummaries(self):
        iris = datasets.load_iris()
        classifier = skflow.TensorFlowLinearClassifier(n_classes=3)
        classifier.fit(iris.data, iris.target, logdir='/tmp/skflow_tests/')
        score = accuracy_score(iris.target, classifier.predict(iris.data))
        self.assertGreater(score, 0.5, "Failed with score = {0}".format(score))


    def testIrisContinueTraining(self):
        iris = datasets.load_iris()
        classifier = skflow.TensorFlowLinearClassifier(n_classes=3,
            learning_rate=0.01, continue_training=True, steps=250)
        classifier.fit(iris.data, iris.target)
        score1 = accuracy_score(iris.target, classifier.predict(iris.data))
        classifier.fit(iris.data, iris.target)
        score2 = accuracy_score(iris.target, classifier.predict(iris.data))
        self.assertGreater(score2, score1,
                           "Failed with score = {0}".format(score2))

    def testIrisStreaming(self):
        iris = datasets.load_iris()

        def iris_data():
            while True:
                for x in iris.data:
                    yield x

        def iris_predict_data():
            for x in iris.data:
                yield x

        def iris_target():
            while True:
                for y in iris.target:
                    yield y

        classifier = skflow.TensorFlowLinearClassifier(n_classes=3, steps=100)
        classifier.fit(iris_data(), iris_target())
        score1 = accuracy_score(iris.target, classifier.predict(iris.data))
        score2 = accuracy_score(iris.target, classifier.predict(iris_predict_data()))
        self.assertGreater(score1, 0.5, "Failed with score = {0}".format(score1))
        self.assertEqual(score2, score1, "Scores from {0} iterator doesn't "
                                         "match score {1} from full "
                                         "data.".format(score2, score1))

    def testIris_proba(self):
        random.seed(42)
        iris = datasets.load_iris()
        classifier = skflow.TensorFlowClassifier(n_classes=3, steps=250)
        classifier.fit(iris.data, iris.target)
        score = log_loss(iris.target, classifier.predict_proba(iris.data))
        self.assertLess(score, 0.8, "Failed with score = {0}".format(score))

    def testBoston(self):
        random.seed(42)
        boston = datasets.load_boston()
        regressor = skflow.TensorFlowLinearRegressor(
            batch_size=boston.data.shape[0],
            steps=500,
            learning_rate=0.001)
        regressor.fit(boston.data, boston.target)
        score = mean_squared_error(
            boston.target, regressor.predict(boston.data))
        self.assertLess(score, 150, "Failed with score = {0}".format(score))

    def testUnfitted(self):
        estimator = skflow.TensorFlowEstimator(model_fn=None, n_classes=1)
        with self.assertRaises(base.NotFittedError):
            estimator.predict([1, 2, 3])
        with self.assertRaises(base.NotFittedError):
            estimator.save('/tmp/path')


if __name__ == '__main__':
    tf.test.main()
