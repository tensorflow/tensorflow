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
"""Tests for the structural state space ensembles."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from tensorflow.contrib import layers
from tensorflow.contrib.layers.python.layers import feature_column

from tensorflow.contrib.timeseries.python.timeseries import estimators
from tensorflow.contrib.timeseries.python.timeseries import input_pipeline
from tensorflow.contrib.timeseries.python.timeseries.feature_keys import TrainEvalFeatures
from tensorflow.contrib.timeseries.python.timeseries.state_space_models import state_space_model
from tensorflow.contrib.timeseries.python.timeseries.state_space_models import structural_ensemble

from tensorflow.python.estimator import estimator_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import test


class StructuralEnsembleEstimatorTests(test.TestCase):

  def simple_data(self, sample_every, dtype, period, num_samples, num_features):
    time = sample_every * numpy.arange(num_samples)
    noise = numpy.random.normal(
        scale=0.01, size=[num_samples, num_features])
    values = noise + numpy.sin(
        numpy.arange(num_features)[None, ...]
        + time[..., None] / float(period) * 2.0 * numpy.pi).astype(
            dtype.as_numpy_dtype)
    return {TrainEvalFeatures.TIMES: numpy.reshape(time, [1, -1]),
            TrainEvalFeatures.VALUES: numpy.reshape(
                values, [1, -1, num_features])}

  def dry_run_train_helper(
      self, sample_every, period, num_samples, model_type, model_args,
      num_features=1):
    numpy.random.seed(1)
    dtype = dtypes.float32
    features = self.simple_data(
        sample_every, dtype=dtype, period=period, num_samples=num_samples,
        num_features=num_features)
    model = model_type(
        configuration=(
            state_space_model.StateSpaceModelConfiguration(
                num_features=num_features,
                dtype=dtype,
                covariance_prior_fn=lambda _: 0.)),
        **model_args)

    class _RunConfig(estimator_lib.RunConfig):

      @property
      def tf_random_seed(self):
        return 4

    estimator = estimators.StateSpaceRegressor(model, config=_RunConfig())
    train_input_fn = input_pipeline.RandomWindowInputFn(
        input_pipeline.NumpyReader(features), num_threads=1, shuffle_seed=1,
        batch_size=16, window_size=16)
    eval_input_fn = input_pipeline.WholeDatasetInputFn(
        input_pipeline.NumpyReader(features))
    estimator.train(input_fn=train_input_fn, max_steps=1)
    first_evaluation = estimator.evaluate(input_fn=eval_input_fn, steps=1)
    estimator.train(input_fn=train_input_fn, max_steps=3)
    second_evaluation = estimator.evaluate(input_fn=eval_input_fn, steps=1)
    self.assertLess(second_evaluation["loss"], first_evaluation["loss"])

  def test_structural_multivariate(self):
    self.dry_run_train_helper(
        sample_every=3,
        period=5,
        num_samples=100,
        num_features=3,
        model_type=structural_ensemble.StructuralEnsemble,
        model_args={
            "periodicities": 2,
            "moving_average_order": 2,
            "autoregressive_order": 1
        })

  def test_exogenous_input(self):
    """Test that no errors are raised when using exogenous features."""
    dtype = dtypes.float64
    times = [1, 2, 3, 4, 5, 6]
    values = [[0.01], [5.10], [5.21], [0.30], [5.41], [0.50]]
    feature_a = [["off"], ["on"], ["on"], ["off"], ["on"], ["off"]]
    sparse_column_a = feature_column.sparse_column_with_keys(
        column_name="feature_a", keys=["on", "off"])
    one_hot_a = layers.one_hot_column(sparse_id_column=sparse_column_a)
    regressor = estimators.StructuralEnsembleRegressor(
        periodicities=[],
        num_features=1,
        moving_average_order=0,
        exogenous_feature_columns=[one_hot_a],
        dtype=dtype)
    features = {TrainEvalFeatures.TIMES: times,
                TrainEvalFeatures.VALUES: values,
                "feature_a": feature_a}
    train_input_fn = input_pipeline.RandomWindowInputFn(
        input_pipeline.NumpyReader(features),
        window_size=6, batch_size=1)
    regressor.train(input_fn=train_input_fn, steps=1)
    eval_input_fn = input_pipeline.WholeDatasetInputFn(
        input_pipeline.NumpyReader(features))
    evaluation = regressor.evaluate(input_fn=eval_input_fn, steps=1)
    predict_input_fn = input_pipeline.predict_continuation_input_fn(
        evaluation, times=[[7, 8, 9]],
        exogenous_features={"feature_a": [[["on"], ["off"], ["on"]]]})
    regressor.predict(input_fn=predict_input_fn)

  def test_no_periodicity(self):
    """Test that no errors are raised when periodicites is None."""
    dtype = dtypes.float64
    times = [1, 2, 3, 4, 5, 6]
    values = [[0.01], [5.10], [5.21], [0.30], [5.41], [0.50]]
    regressor = estimators.StructuralEnsembleRegressor(
        periodicities=None,
        num_features=1,
        moving_average_order=0,
        dtype=dtype)
    features = {TrainEvalFeatures.TIMES: times,
                TrainEvalFeatures.VALUES: values}
    train_input_fn = input_pipeline.RandomWindowInputFn(
        input_pipeline.NumpyReader(features),
        window_size=6, batch_size=1)
    regressor.train(input_fn=train_input_fn, steps=1)
    eval_input_fn = input_pipeline.WholeDatasetInputFn(
        input_pipeline.NumpyReader(features))
    evaluation = regressor.evaluate(input_fn=eval_input_fn, steps=1)
    predict_input_fn = input_pipeline.predict_continuation_input_fn(
        evaluation, times=[[7, 8, 9]])
    regressor.predict(input_fn=predict_input_fn)

if __name__ == "__main__":
  test.main()
