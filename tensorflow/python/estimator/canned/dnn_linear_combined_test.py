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
"""Tests for dnn_linear_combined.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.estimator.canned import dnn_linear_combined
from tensorflow.python.estimator.canned import dnn_testing_utils
from tensorflow.python.estimator.canned import linear_testing_utils
from tensorflow.python.ops import nn
from tensorflow.python.platform import test


class DNNOnlyModelFnTest(dnn_testing_utils.BaseDNNModelFnTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    dnn_testing_utils.BaseDNNModelFnTest.__init__(self, self._dnn_only_model_fn)

  def _dnn_only_model_fn(
      self,
      features,
      labels,
      mode,
      head,
      hidden_units,
      feature_columns,
      optimizer='Adagrad',
      activation_fn=nn.relu,
      dropout=None,  # pylint: disable=redefined-outer-name
      input_layer_partitioner=None,
      config=None):
    return dnn_linear_combined._dnn_linear_combined_model_fn(
        features=features,
        labels=labels,
        mode=mode,
        head=head,
        linear_feature_columns=[],
        dnn_hidden_units=hidden_units,
        dnn_feature_columns=feature_columns,
        dnn_optimizer=optimizer,
        dnn_activation_fn=activation_fn,
        dnn_dropout=dropout,
        input_layer_partitioner=input_layer_partitioner,
        config=config)


# A function to mimic linear-regressor init reuse same tests.
def _linear_regressor_fn(feature_columns,
                         model_dir=None,
                         label_dimension=1,
                         weight_feature_key=None,
                         optimizer='Ftrl',
                         config=None,
                         partitioner=None):
  return dnn_linear_combined.DNNLinearCombinedRegressor(
      model_dir=model_dir,
      linear_feature_columns=feature_columns,
      linear_optimizer=optimizer,
      label_dimension=label_dimension,
      weight_feature_key=weight_feature_key,
      input_layer_partitioner=partitioner,
      config=config)


class LinearOnlyRegressorPartitionerTest(
    linear_testing_utils.BaseLinearRegressorPartitionerTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearRegressorPartitionerTest.__init__(
        self, _linear_regressor_fn)


class LinearOnlyRegressorEvaluationTest(
    linear_testing_utils.BaseLinearRegressorEvaluationTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearRegressorEvaluationTest.__init__(
        self, _linear_regressor_fn)


class LinearOnlyRegressorPredictTest(
    linear_testing_utils.BaseLinearRegressorPredictTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearRegressorPredictTest.__init__(
        self, _linear_regressor_fn)


class LinearOnlyRegressorIntegrationTest(
    linear_testing_utils.BaseLinearRegressorIntegrationTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearRegressorIntegrationTest.__init__(
        self, _linear_regressor_fn)


class LinearOnlyRegressorTrainingTest(
    linear_testing_utils.BaseLinearRegressorTrainingTest, test.TestCase):

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    linear_testing_utils.BaseLinearRegressorTrainingTest.__init__(
        self, _linear_regressor_fn)


if __name__ == '__main__':
  test.main()
