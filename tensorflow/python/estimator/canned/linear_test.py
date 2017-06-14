# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for linear.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.estimator.canned import linear
from tensorflow.python.estimator.canned import linear_testing_utils
from tensorflow.python.platform import test


def _linear_regressor_fn(*args, **kwargs):
  return linear.LinearRegressor(*args, **kwargs)


def _linear_classifier_fn(*args, **kwargs):
  return linear.LinearClassifier(*args, **kwargs)


# Tests for Linear Regressor.


class LinearRegressorPartitionerTest(
    linear_testing_utils.BaseLinearRegressorPartitionerTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearRegressorPartitionerTest.__init__(
        self, _linear_regressor_fn)


class LinearRegressorEvaluationTest(
    linear_testing_utils.BaseLinearRegressorEvaluationTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearRegressorEvaluationTest.__init__(
        self, _linear_regressor_fn)


class LinearRegressorPredictTest(
    linear_testing_utils.BaseLinearRegressorPredictTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearRegressorPredictTest.__init__(
        self, _linear_regressor_fn)


class LinearRegressorIntegrationTest(
    linear_testing_utils.BaseLinearRegressorIntegrationTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearRegressorIntegrationTest.__init__(
        self, _linear_regressor_fn)


class LinearRegressorTrainingTest(
    linear_testing_utils.BaseLinearRegressorTrainingTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearRegressorTrainingTest.__init__(
        self, _linear_regressor_fn)


# Tests for Linear Classifier.


class LinearClassifierTrainingTest(
    linear_testing_utils.BaseLinearClassifierTrainingTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearClassifierTrainingTest.__init__(
        self, linear_classifier_fn=_linear_classifier_fn)


class LinearClassifierEvaluationTest(
    linear_testing_utils.BaseLinearClassifierEvaluationTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearClassifierEvaluationTest.__init__(
        self, linear_classifier_fn=_linear_classifier_fn)


class LinearClassifierPredictTest(
    linear_testing_utils.BaseLinearClassifierPredictTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearClassifierPredictTest.__init__(
        self, linear_classifier_fn=_linear_classifier_fn)


class LinearClassifierIntegrationTest(
    linear_testing_utils.BaseLinearClassifierIntegrationTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearClassifierIntegrationTest.__init__(
        self, linear_classifier_fn=_linear_classifier_fn)


if __name__ == '__main__':
  test.main()
