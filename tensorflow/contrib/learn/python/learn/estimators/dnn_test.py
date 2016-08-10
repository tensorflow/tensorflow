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

"""Tests for DNNEstimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# pylint: disable=g-import-not-at-top
try:
  from sklearn.cross_validation import cross_val_score
  HAS_SKLEARN = True
except ImportError:
  HAS_SKLEARN = False


def _iris_input_fn():
  iris = tf.contrib.learn.datasets.load_iris()
  return {
      'feature': tf.constant(iris.data, dtype=tf.float32)
  }, tf.constant(iris.target, shape=[150, 1], dtype=tf.int32)


class DNNClassifierTest(tf.test.TestCase):

  def testMultiClass(self):
    """Tests multi-class classification using matrix data as input."""
    cont_features = [
        tf.contrib.layers.real_valued_column('feature', dimension=4)]

    classifier = tf.contrib.learn.DNNClassifier(n_classes=3,
                                                feature_columns=cont_features,
                                                hidden_units=[3, 3])

    classifier.fit(input_fn=_iris_input_fn, steps=1000)
    classifier.evaluate(input_fn=_iris_input_fn, steps=100)
    self.assertTrue('centered_bias_weight' in classifier.get_variable_names())
    # TODO(ispir): Enable accuracy check after resolving the randomness issue.
    # self.assertGreater(scores['accuracy/mean'], 0.6)

  def testDisableCenteredBias(self):
    """Tests that we can disable centered bias."""
    cont_features = [
        tf.contrib.layers.real_valued_column('feature', dimension=4)]

    classifier = tf.contrib.learn.DNNClassifier(n_classes=3,
                                                feature_columns=cont_features,
                                                hidden_units=[3, 3],
                                                enable_centered_bias=False)

    classifier.fit(input_fn=_iris_input_fn, steps=1000)
    self.assertFalse('centered_bias_weight' in classifier.get_variable_names())

  def testSklearnCompatibility(self):
    """Tests compatibility with sklearn"""
    if not HAS_SKLEARN:
      return
    iris = tf.contrib.learn.datasets.load_iris()
    kwargs = {
            "n_classes": 3,
            "optimizer" : "Adam",
            "hidden_units" : [3, 4]
    }

    classifier = tf.contrib.learn.DNNClassifier(**kwargs)

    scores = cross_val_score(
      classifier,
      iris.data[1:5],
      iris.target[1:5],
      scoring="accuracy",
      fit_params={"steps": 2}
    )
    self.assertAllClose(scores, [1, 1, 1])


class DNNRegressorTest(tf.test.TestCase):

  def testRegression(self):
    """Tests multi-class classification using matrix data as input."""
    cont_features = [
        tf.contrib.layers.real_valued_column('feature', dimension=4)]

    regressor = tf.contrib.learn.DNNRegressor(feature_columns=cont_features,
                                              hidden_units=[3, 3])

    regressor.fit(input_fn=_iris_input_fn, steps=1000)
    regressor.evaluate(input_fn=_iris_input_fn, steps=100)


def boston_input_fn():
  boston = tf.contrib.learn.datasets.load_boston()
  features = tf.cast(tf.reshape(tf.constant(boston.data), [-1, 13]), tf.float32)
  target = tf.cast(tf.reshape(tf.constant(boston.target), [-1, 1]), tf.float32)
  return features, target


class FeatureColumnTest(tf.test.TestCase):

  # TODO(b/29580537): Remove when we deprecate feature column inference.
  def testTrainWithInferredFeatureColumns(self):
    est = tf.contrib.learn.DNNRegressor(hidden_units=[3, 3])
    est.fit(input_fn=boston_input_fn, steps=1)
    _ = est.evaluate(input_fn=boston_input_fn, steps=1)

  def testTrain(self):
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input_fn(
        boston_input_fn)
    est = tf.contrib.learn.DNNRegressor(
        feature_columns=feature_columns, hidden_units=[3, 3])
    est.fit(input_fn=boston_input_fn, steps=1)
    _ = est.evaluate(input_fn=boston_input_fn, steps=1)


if __name__ == '__main__':
  tf.test.main()
