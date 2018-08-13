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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tempfile

import numpy
import six

from tensorflow.contrib.timeseries.python.timeseries import ar_model
from tensorflow.contrib.timeseries.python.timeseries import estimators
from tensorflow.contrib.timeseries.python.timeseries import feature_keys
from tensorflow.contrib.timeseries.python.timeseries import input_pipeline
from tensorflow.contrib.timeseries.python.timeseries import saved_model_utils

from tensorflow.python.client import session
from tensorflow.python.estimator import estimator_lib
from tensorflow.python.feature_column import feature_column
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import tag_constants


class _SeedRunConfig(estimator_lib.RunConfig):

  @property
  def tf_random_seed(self):
    return 3


class TimeSeriesRegressorTest(test.TestCase):

  def _fit_restore_fit_test_template(self, estimator_fn, dtype):
    """Tests restoring previously fit models."""
    model_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    exogenous_feature_columns = (
        feature_column.numeric_column("exogenous"),
    )
    first_estimator = estimator_fn(model_dir, exogenous_feature_columns)
    times = numpy.arange(20, dtype=numpy.int64)
    values = numpy.arange(20, dtype=dtype.as_numpy_dtype)
    exogenous = numpy.arange(20, dtype=dtype.as_numpy_dtype)
    features = {
        feature_keys.TrainEvalFeatures.TIMES: times,
        feature_keys.TrainEvalFeatures.VALUES: values,
        "exogenous": exogenous
    }
    train_input_fn = input_pipeline.RandomWindowInputFn(
        input_pipeline.NumpyReader(features), shuffle_seed=2, num_threads=1,
        batch_size=16, window_size=16)
    eval_input_fn = input_pipeline.RandomWindowInputFn(
        input_pipeline.NumpyReader(features), shuffle_seed=3, num_threads=1,
        batch_size=16, window_size=16)
    first_estimator.train(input_fn=train_input_fn, steps=1)
    first_loss_before_fit = first_estimator.evaluate(
        input_fn=eval_input_fn, steps=1)["loss"]
    self.assertAllEqual([], first_loss_before_fit.shape)
    first_estimator.train(input_fn=train_input_fn, steps=1)
    first_loss_after_fit = first_estimator.evaluate(
        input_fn=eval_input_fn, steps=1)["loss"]
    self.assertAllEqual([], first_loss_after_fit.shape)
    second_estimator = estimator_fn(model_dir, exogenous_feature_columns)
    second_estimator.train(input_fn=train_input_fn, steps=1)
    whole_dataset_input_fn = input_pipeline.WholeDatasetInputFn(
        input_pipeline.NumpyReader(features))
    whole_dataset_evaluation = second_estimator.evaluate(
        input_fn=whole_dataset_input_fn, steps=1)
    exogenous_values_ten_steps = {
        "exogenous": numpy.arange(
            10, dtype=dtype.as_numpy_dtype)[None, :, None]
    }
    predict_input_fn = input_pipeline.predict_continuation_input_fn(
        evaluation=whole_dataset_evaluation,
        exogenous_features=exogenous_values_ten_steps,
        steps=10)
    # Also tests that limit_epochs in predict_continuation_input_fn prevents
    # infinite iteration
    (estimator_predictions,
    ) = list(second_estimator.predict(input_fn=predict_input_fn))
    self.assertAllEqual([10, 1], estimator_predictions["mean"].shape)
    input_receiver_fn = first_estimator.build_raw_serving_input_receiver_fn()
    export_location = first_estimator.export_savedmodel(self.get_temp_dir(),
                                                        input_receiver_fn)
    with ops.Graph().as_default():
      with session.Session() as sess:
        signatures = loader.load(sess, [tag_constants.SERVING], export_location)
        # Test that prediction and filtering can continue from evaluation output
        saved_prediction = saved_model_utils.predict_continuation(
            continue_from=whole_dataset_evaluation,
            steps=10,
            exogenous_features=exogenous_values_ten_steps,
            signatures=signatures,
            session=sess)
        # Saved model predictions should be the same as Estimator predictions
        # starting from the same evaluation.
        for prediction_key, prediction_value in estimator_predictions.items():
          self.assertAllClose(prediction_value,
                              numpy.squeeze(
                                  saved_prediction[prediction_key], axis=0))
        first_filtering = saved_model_utils.filter_continuation(
            continue_from=whole_dataset_evaluation,
            features={
                feature_keys.FilteringFeatures.TIMES: times[None, -1] + 2,
                feature_keys.FilteringFeatures.VALUES: values[None, -1] + 2.,
                "exogenous": values[None, -1, None] + 12.
            },
            signatures=signatures,
            session=sess)
        # Test that prediction and filtering can continue from filtering output
        second_saved_prediction = saved_model_utils.predict_continuation(
            continue_from=first_filtering,
            steps=1,
            exogenous_features={
                "exogenous": numpy.arange(
                    1, dtype=dtype.as_numpy_dtype)[None, :, None]
            },
            signatures=signatures,
            session=sess)
        self.assertEqual(
            times[-1] + 3,
            numpy.squeeze(
                second_saved_prediction[feature_keys.PredictionResults.TIMES]))
        saved_model_utils.filter_continuation(
            continue_from=first_filtering,
            features={
                feature_keys.FilteringFeatures.TIMES: times[-1] + 3,
                feature_keys.FilteringFeatures.VALUES: values[-1] + 3.,
                "exogenous": values[-1, None] + 13.
            },
            signatures=signatures,
            session=sess)

        # Test cold starting
        six.assertCountEqual(
            self,
            [feature_keys.FilteringFeatures.TIMES,
             feature_keys.FilteringFeatures.VALUES,
             "exogenous"],
            signatures.signature_def[
                feature_keys.SavedModelLabels.COLD_START_FILTER].inputs.keys())
        batch_numpy_times = numpy.tile(
            numpy.arange(30, dtype=numpy.int64)[None, :], (10, 1))
        batch_numpy_values = numpy.ones([10, 30, 1])
        state = saved_model_utils.cold_start_filter(
            signatures=signatures,
            session=sess,
            features={
                feature_keys.FilteringFeatures.TIMES: batch_numpy_times,
                feature_keys.FilteringFeatures.VALUES: batch_numpy_values,
                "exogenous": 10. + batch_numpy_values
            }
        )
        predict_times = numpy.tile(
            numpy.arange(30, 45, dtype=numpy.int64)[None, :], (10, 1))
        predictions = saved_model_utils.predict_continuation(
            continue_from=state,
            times=predict_times,
            exogenous_features={
                "exogenous": numpy.tile(numpy.arange(
                    15, dtype=dtype.as_numpy_dtype), (10,))[None, :, None]
            },
            signatures=signatures,
            session=sess)
        self.assertAllEqual([10, 15, 1], predictions["mean"].shape)

  def test_fit_restore_fit_ar_flat(self):
    def _estimator_fn(model_dir, exogenous_feature_columns):
      return estimators.ARRegressor(
          periodicities=10, input_window_size=10, output_window_size=6,
          num_features=1, model_dir=model_dir, config=_SeedRunConfig(),
          # This test is flaky with normal likelihood loss (could add more
          # training iterations instead).
          loss=ar_model.ARModel.SQUARED_LOSS,
          exogenous_feature_columns=exogenous_feature_columns)
    self._fit_restore_fit_test_template(_estimator_fn, dtype=dtypes.float32)

  def test_fit_restore_fit_ar_lstm(self):
    def _estimator_fn(model_dir, exogenous_feature_columns):
      return estimators.TimeSeriesRegressor(
          model=ar_model.ARModel(
              periodicities=10, input_window_size=10, output_window_size=6,
              num_features=1,
              exogenous_feature_columns=exogenous_feature_columns,
              prediction_model_factory=functools.partial(
                  ar_model.LSTMPredictionModel,
                  num_units=10)),
          config=_SeedRunConfig(),
          model_dir=model_dir)
    self._fit_restore_fit_test_template(_estimator_fn, dtype=dtypes.float32)

  def test_fit_restore_fit_structural_ensemble_regressor(self):
    dtype = dtypes.float32
    def _estimator_fn(model_dir, exogenous_feature_columns):
      return estimators.StructuralEnsembleRegressor(
          num_features=1, periodicities=10, model_dir=model_dir, dtype=dtype,
          config=_SeedRunConfig(),
          exogenous_feature_columns=exogenous_feature_columns)
    self._fit_restore_fit_test_template(_estimator_fn, dtype=dtype)


if __name__ == "__main__":
  test.main()
