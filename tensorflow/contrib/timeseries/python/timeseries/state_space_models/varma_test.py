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
"""Tests for VARMA.

Tests VARMA model building and utility functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.timeseries.python.timeseries.feature_keys import TrainEvalFeatures
from tensorflow.contrib.timeseries.python.timeseries.state_space_models import state_space_model
from tensorflow.contrib.timeseries.python.timeseries.state_space_models import varma

from tensorflow.python.estimator import estimator_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class MakeModelTest(test.TestCase):

  def test_ar_smaller(self):
    model = varma.VARMA(
        autoregressive_order=0,
        moving_average_order=3)
    model.initialize_graph()
    outputs = model.define_loss(
        features={
            TrainEvalFeatures.TIMES: constant_op.constant([[1, 2]]),
            TrainEvalFeatures.VALUES: constant_op.constant([[[1.], [2.]]])
        },
        mode=estimator_lib.ModeKeys.TRAIN)
    initializer = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run([initializer])
      outputs.loss.eval()

  def test_ma_smaller(self):
    model = varma.VARMA(
        autoregressive_order=6,
        moving_average_order=3,
        configuration=state_space_model.StateSpaceModelConfiguration(
            num_features=7))
    model.initialize_graph()
    outputs = model.define_loss(
        features={
            TrainEvalFeatures.TIMES: constant_op.constant([[1, 2]]),
            TrainEvalFeatures.VALUES: constant_op.constant(
                [[[1.] * 7, [2.] * 7]])
        },
        mode=estimator_lib.ModeKeys.TRAIN)
    initializer = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run([initializer])
      outputs.loss.eval()

  def test_make_ensemble_no_errors(self):
    with variable_scope.variable_scope("model_one"):
      model_one = varma.VARMA(10, 5)
    with variable_scope.variable_scope("model_two"):
      model_two = varma.VARMA(0, 3)
    configuration = state_space_model.StateSpaceModelConfiguration()
    ensemble = state_space_model.StateSpaceIndependentEnsemble(
        ensemble_members=[model_one, model_two],
        configuration=configuration)
    ensemble.initialize_graph()
    outputs = ensemble.define_loss(
        features={
            TrainEvalFeatures.TIMES: constant_op.constant([[1, 2]]),
            TrainEvalFeatures.VALUES: constant_op.constant([[[1.], [2.]]])},
        mode=estimator_lib.ModeKeys.TRAIN)
    initializer = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run([initializer])
      outputs.loss.eval()


if __name__ == "__main__":
  test.main()
