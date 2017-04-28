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
"""Custom optimizer tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np

from tensorflow.contrib.learn.python import learn
from tensorflow.contrib.learn.python.learn import datasets
from tensorflow.contrib.learn.python.learn import metric_spec
from tensorflow.contrib.learn.python.learn.estimators import estimator as estimator_lib
from tensorflow.contrib.learn.python.learn.estimators._sklearn import accuracy_score
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import momentum as momentum_lib


class FeatureEngineeringFunctionTest(test.TestCase):
  """Tests feature_engineering_fn."""

  def testFeatureEngineeringFn(self):

    def input_fn():
      return {
          "x": constant_op.constant([1.])
      }, {
          "y": constant_op.constant([11.])
      }

    def feature_engineering_fn(features, labels):
      _, _ = features, labels
      return {
          "transformed_x": constant_op.constant([9.])
      }, {
          "transformed_y": constant_op.constant([99.])
      }

    def model_fn(features, labels):
      # dummy variable:
      _ = variables.Variable([0.])
      _ = labels
      predictions = features["transformed_x"]
      loss = constant_op.constant([2.])
      return predictions, loss, control_flow_ops.no_op()

    estimator = estimator_lib.Estimator(
        model_fn=model_fn, feature_engineering_fn=feature_engineering_fn)
    estimator.fit(input_fn=input_fn, steps=1)
    prediction = next(estimator.predict(input_fn=input_fn, as_iterable=True))
    # predictions = transformed_x (9)
    self.assertEqual(9., prediction)
    metrics = estimator.evaluate(
        input_fn=input_fn, steps=1,
        metrics={"label":
                 metric_spec.MetricSpec(lambda predictions, labels: labels)})
    # labels = transformed_y (99)
    self.assertEqual(99., metrics["label"])

  def testNoneFeatureEngineeringFn(self):

    def input_fn():
      return {
          "x": constant_op.constant([1.])
      }, {
          "y": constant_op.constant([11.])
      }

    def feature_engineering_fn(features, labels):
      _, _ = features, labels
      return {
          "x": constant_op.constant([9.])
      }, {
          "y": constant_op.constant([99.])
      }

    def model_fn(features, labels):
      # dummy variable:
      _ = variables.Variable([0.])
      _ = labels
      predictions = features["x"]
      loss = constant_op.constant([2.])
      return predictions, loss, control_flow_ops.no_op()

    estimator_with_fe_fn = estimator_lib.Estimator(
        model_fn=model_fn, feature_engineering_fn=feature_engineering_fn)
    estimator_with_fe_fn.fit(input_fn=input_fn, steps=1)
    estimator_without_fe_fn = estimator_lib.Estimator(model_fn=model_fn)
    estimator_without_fe_fn.fit(input_fn=input_fn, steps=1)

    # predictions = x
    prediction_with_fe_fn = next(
        estimator_with_fe_fn.predict(
            input_fn=input_fn, as_iterable=True))
    self.assertEqual(9., prediction_with_fe_fn)
    prediction_without_fe_fn = next(
        estimator_without_fe_fn.predict(
            input_fn=input_fn, as_iterable=True))
    self.assertEqual(1., prediction_without_fe_fn)


class CustomOptimizer(test.TestCase):
  """Custom optimizer tests."""

  def testIrisMomentum(self):
    random.seed(42)

    iris = datasets.load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42)

    def custom_optimizer():
      return momentum_lib.MomentumOptimizer(learning_rate=0.01, momentum=0.9)

    classifier = learn.DNNClassifier(
        hidden_units=[10, 20, 10],
        feature_columns=learn.infer_real_valued_columns_from_input(x_train),
        n_classes=3,
        optimizer=custom_optimizer,
        config=learn.RunConfig(tf_random_seed=1))
    classifier.fit(x_train, y_train, steps=400)
    predictions = np.array(list(classifier.predict_classes(x_test)))
    score = accuracy_score(y_test, predictions)

    self.assertGreater(score, 0.65, "Failed with score = {0}".format(score))


if __name__ == "__main__":
  test.main()
