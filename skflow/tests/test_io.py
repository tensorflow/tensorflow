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
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.python.platform import googletest

from skflow.io import *
import skflow

class BaseTest(googletest.TestCase):
    def test_pandas_dataframe(self):
        if HAS_PANDAS:
            random.seed(42)
            iris = datasets.load_iris()
            data = DataFrame(iris.data)
            labels = DataFrame(iris.target)
            classifier = skflow.TensorFlowLinearClassifier(n_classes=3)
            classifier.fit(data, labels)
            score = accuracy_score(classifier.predict(data), labels)
            self.assertGreater(score, 0.5, "Failed with score = {0}".format(score))
        else:
            print("No pandas installed. test_pandas_dataframe skipped.")

    def test_pandas_series(self):
        if HAS_PANDAS:
            random.seed(42)
            iris = datasets.load_iris()
            data = DataFrame(iris.data)
            labels = Series(iris.target)
            classifier = skflow.TensorFlowLinearClassifier(n_classes=3)
            classifier.fit(data, labels)
            score = accuracy_score(classifier.predict(data), labels)
            self.assertGreater(score, 0.5, "Failed with score = {0}".format(score))
        else:
            print("No pandas installed. test_pandas_series skipped.")

if __name__ == '__main__':
    tf.test.main()
