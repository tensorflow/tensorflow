# pylint: disable=g-bad-file-header
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Custom decay tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import tensorflow as tf

from tensorflow.contrib.learn.python import learn
from tensorflow.contrib.learn.python.learn import datasets
from tensorflow.contrib.learn.python.learn.estimators._sklearn import accuracy_score
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split


class CustomDecayTest(tf.test.TestCase):
  """Custom decay tests."""

  def testIrisExponentialDecay(self):
    random.seed(42)

    iris = datasets.load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data,
                                                        iris.target,
                                                        test_size=0.2,
                                                        random_state=42)

    # setup exponential decay function
    def exp_decay(global_step):
      return tf.train.exponential_decay(learning_rate=0.1,
                                        global_step=global_step,
                                        decay_steps=100,
                                        decay_rate=0.001)

    classifier = learn.TensorFlowDNNClassifier(hidden_units=[10, 20, 10],
                                               n_classes=3,
                                               steps=500,
                                               learning_rate=exp_decay)
    classifier.fit(x_train, y_train)
    score = accuracy_score(y_test, classifier.predict(x_test))

    self.assertGreater(score, 0.65, "Failed with score = {0}".format(score))


if __name__ == "__main__":
  tf.test.main()
