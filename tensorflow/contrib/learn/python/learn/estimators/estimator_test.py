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
"""Tests for Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

import tensorflow as tf


def boston_input_fn():
  boston = tf.contrib.learn.datasets.load_boston()
  features = tf.cast(
      tf.reshape(
          tf.constant(boston.data), [-1, 13]), tf.float32)
  target = tf.cast(
      tf.reshape(
          tf.constant(boston.target), [-1, 1]), tf.float32)
  return features, target


def linear_model_fn(features, target, unused_mode):
  return tf.contrib.learn.models.linear_regression_zero_init(features, target)


class EstimatorTest(tf.test.TestCase):

  def testTrain(self):
    output_dir = tempfile.mkdtemp()
    est = tf.contrib.learn.Estimator(model_fn=linear_model_fn,
                                     classification=False, model_dir=output_dir)
    est.train(input_fn=boston_input_fn, steps=1)
    _ = est.evaluate(input_fn=boston_input_fn, steps=1)

  def testPredict(self):
    est = tf.contrib.learn.Estimator(model_fn=linear_model_fn,
                                     classification=False)
    boston = tf.contrib.learn.datasets.load_boston()
    est.train(input_fn=boston_input_fn, steps=1)
    output = est.predict(boston.data)
    self.assertEqual(output['predictions'].shape[0], boston.target.shape[0])

  def testWrongInput(self):
    def other_input_fn():
      return {'other': tf.constant([0, 0, 0])}, tf.constant([0, 0, 0])
    output_dir = tempfile.mkdtemp()
    est = tf.contrib.learn.Estimator(model_fn=linear_model_fn,
                                     classification=False, model_dir=output_dir)
    est.train(input_fn=boston_input_fn, steps=1)
    with self.assertRaises(ValueError):
      est.train(input_fn=other_input_fn, steps=1)


if __name__ == '__main__':
  tf.test.main()
