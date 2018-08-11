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
"""Tests for head."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

from absl.testing import parameterized
import numpy
import six

from tensorflow.contrib.estimator.python.estimator import extenders
from tensorflow.contrib.timeseries.examples import lstm as lstm_example
from tensorflow.contrib.timeseries.python.timeseries import ar_model
from tensorflow.contrib.timeseries.python.timeseries import estimators as ts_estimators
from tensorflow.contrib.timeseries.python.timeseries import feature_keys
from tensorflow.contrib.timeseries.python.timeseries import head as ts_head_lib
from tensorflow.contrib.timeseries.python.timeseries import input_pipeline
from tensorflow.contrib.timeseries.python.timeseries import model
from tensorflow.contrib.timeseries.python.timeseries import state_management
from tensorflow.core.example import example_pb2

from tensorflow.python.client import session as session_lib
from tensorflow.python.estimator import estimator_lib
from tensorflow.python.feature_column import feature_column
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.training import adam
from tensorflow.python.training import coordinator as coordinator_lib
from tensorflow.python.training import queue_runner_impl
from tensorflow.python.training import training as train


class HeadTest(test.TestCase):

  def test_labels_provided_error(self):
    model_fn = _stub_model_fn()
    for mode in [estimator_lib.ModeKeys.TRAIN, estimator_lib.ModeKeys.EVAL,
                 estimator_lib.ModeKeys.PREDICT]:
      with self.assertRaisesRegexp(ValueError, "received a `labels`"):
        model_fn(features={}, labels={"a": "b"}, mode=mode)

      with self.assertRaisesRegexp(ValueError, "received a `labels`"):
        model_fn(features={}, labels=array_ops.zeros([]), mode=mode)

  def test_unknown_mode(self):
    model_fn = _stub_model_fn()
    with self.assertRaisesRegexp(ValueError, "Unknown mode 'Not a mode'"):
      model_fn(features={}, labels={}, mode="Not a mode")


class _TickerModel(object):
  num_features = 1
  dtype = dtypes.float32

  def initialize_graph(self, input_statistics):
    pass

  def define_loss(self, features, mode):
    del mode  # unused
    return model.ModelOutputs(
        loss=features["ticker"],
        end_state=(features["ticker"], features["ticker"]),
        prediction_times=array_ops.zeros(()),
        predictions={"ticker": features["ticker"]})


class EvaluationMetricsTests(test.TestCase):

  def test_metrics_consistent(self):
    # Tests that the identity metrics used to report in-sample predictions match
    # the behavior of standard metrics.
    g = ops.Graph()
    with g.as_default():
      features = {
          feature_keys.TrainEvalFeatures.TIMES:
              array_ops.zeros((1, 1)),
          feature_keys.TrainEvalFeatures.VALUES:
              array_ops.zeros((1, 1, 1)),
          "ticker":
              array_ops.reshape(
                  math_ops.cast(
                      variables.Variable(
                          name="ticker",
                          initial_value=0,
                          dtype=dtypes.int64,
                          collections=[ops.GraphKeys.LOCAL_VARIABLES])
                      .count_up_to(10),
                      dtype=dtypes.float32), (1, 1, 1))
      }
      model_fn = ts_head_lib.TimeSeriesRegressionHead(
          model=_TickerModel(),
          state_manager=state_management.PassthroughStateManager(),
          optimizer=train.GradientDescentOptimizer(0.001)).create_estimator_spec
      outputs = model_fn(
          features=features, labels=None, mode=estimator_lib.ModeKeys.EVAL)
      metric_update_ops = [
          metric[1] for metric in outputs.eval_metric_ops.values()]
      loss_mean, loss_update = metrics.mean(outputs.loss)
      metric_update_ops.append(loss_update)
      with self.test_session() as sess:
        coordinator = coordinator_lib.Coordinator()
        queue_runner_impl.start_queue_runners(sess, coord=coordinator)
        variables.local_variables_initializer().run()
        sess.run(metric_update_ops)
        loss_evaled, metric_evaled, nested_metric_evaled = sess.run(
            (loss_mean, outputs.eval_metric_ops["ticker"][0],
             outputs.eval_metric_ops[feature_keys.FilteringResults.STATE_TUPLE][
                 0][0]))
        # The custom model_utils metrics for in-sample predictions should be in
        # sync with the Estimator's mean metric for model loss.
        self.assertAllClose(0., loss_evaled)
        self.assertAllClose((((0.,),),), metric_evaled)
        self.assertAllClose((((0.,),),), nested_metric_evaled)
        coordinator.request_stop()
        coordinator.join()

  def test_custom_metrics(self):
    """Tests that the custom metrics can be applied to the estimator."""
    model_dir = self.get_temp_dir()
    estimator = ts_estimators.TimeSeriesRegressor(
        model=lstm_example._LSTMModel(num_features=1, num_units=4),
        optimizer=adam.AdamOptimizer(0.001),
        config=estimator_lib.RunConfig(tf_random_seed=4),
        model_dir=model_dir)

    def input_fn():
      return {
          feature_keys.TrainEvalFeatures.TIMES: [[1, 2, 3], [7, 8, 9]],
          feature_keys.TrainEvalFeatures.VALUES:
              numpy.array([[[0.], [1.], [0.]], [[2.], [3.], [2.]]])
      }

    def metrics_fn(predictions, features):
      # checking that the inputs are properly passed.
      predict = predictions["mean"]
      target = features[feature_keys.TrainEvalFeatures.VALUES][:, -1, 0]
      return {
          "plain_boring_metric386":
              (math_ops.reduce_mean(math_ops.abs(predict - target)),
               control_flow_ops.no_op()),
          "fun_metric101": (math_ops.reduce_sum(predict + target),
                            control_flow_ops.no_op()),
      }

    # Evaluation without training is enough for testing custom metrics.
    estimator = extenders.add_metrics(estimator, metrics_fn)
    evaluation = estimator.evaluate(input_fn, steps=1)
    self.assertIn("plain_boring_metric386", evaluation)
    self.assertIn("fun_metric101", evaluation)
    # The values are deterministic because of fixed tf_random_seed.
    # However if they become flaky, remove such exacts comparisons.
    self.assertAllClose(evaluation["plain_boring_metric386"], 1.130380)
    self.assertAllClose(evaluation["fun_metric101"], 10.435442)


class _StubModel(object):
  num_features = 3
  dtype = dtypes.float64

  def initialize_graph(self, input_statistics):
    del input_statistics  # unused


def _stub_model_fn():
  return ts_head_lib.TimeSeriesRegressionHead(
      model=_StubModel(),
      state_manager=state_management.PassthroughStateManager(),
      optimizer=train.AdamOptimizer(0.001)).create_estimator_spec


class TrainEvalFeatureCheckingTests(test.TestCase):

  def test_no_time_feature(self):
    model_fn = _stub_model_fn()
    for mode in [estimator_lib.ModeKeys.TRAIN, estimator_lib.ModeKeys.EVAL]:
      with self.assertRaisesRegexp(ValueError, "Expected a '{}' feature".format(
          feature_keys.TrainEvalFeatures.TIMES)):
        model_fn(
            features={feature_keys.TrainEvalFeatures.VALUES: [[[1.]]]},
            labels=None,
            mode=mode)

  def test_no_value_feature(self):
    model_fn = _stub_model_fn()
    for mode in [estimator_lib.ModeKeys.TRAIN, estimator_lib.ModeKeys.EVAL]:
      with self.assertRaisesRegexp(ValueError, "Expected a '{}' feature".format(
          feature_keys.TrainEvalFeatures.VALUES)):
        model_fn(
            features={feature_keys.TrainEvalFeatures.TIMES: [[1]]},
            labels=None,
            mode=mode)

  def test_bad_time_rank(self):
    model_fn = _stub_model_fn()
    for mode in [estimator_lib.ModeKeys.TRAIN, estimator_lib.ModeKeys.EVAL]:
      with self.assertRaisesRegexp(ValueError,
                                   "Expected shape.*for feature '{}'".format(
                                       feature_keys.TrainEvalFeatures.TIMES)):
        model_fn(
            features={
                feature_keys.TrainEvalFeatures.TIMES: [[[1]]],
                feature_keys.TrainEvalFeatures.VALUES: [[[1.]]]
            },
            labels=None,
            mode=mode)

  def test_bad_value_rank(self):
    model_fn = _stub_model_fn()
    for mode in [estimator_lib.ModeKeys.TRAIN, estimator_lib.ModeKeys.EVAL]:
      with self.assertRaisesRegexp(ValueError,
                                   "Expected shape.*for feature '{}'".format(
                                       feature_keys.TrainEvalFeatures.VALUES)):
        model_fn(
            features={
                feature_keys.TrainEvalFeatures.TIMES: [[1]],
                feature_keys.TrainEvalFeatures.VALUES: [[1.]]
            },
            labels=None,
            mode=mode)

  def test_bad_value_num_features(self):
    model_fn = _stub_model_fn()
    for mode in [estimator_lib.ModeKeys.TRAIN, estimator_lib.ModeKeys.EVAL]:
      with self.assertRaisesRegexp(
          ValueError, "Expected shape.*, 3.*for feature '{}'".format(
              feature_keys.TrainEvalFeatures.VALUES)):
        model_fn(
            features={
                feature_keys.TrainEvalFeatures.TIMES: [[1]],
                feature_keys.TrainEvalFeatures.VALUES: [[[1.]]]
            },
            labels=None,
            mode=mode)

  def test_bad_exogenous_shape(self):
    model_fn = _stub_model_fn()
    for mode in [estimator_lib.ModeKeys.TRAIN, estimator_lib.ModeKeys.EVAL]:
      with self.assertRaisesRegexp(
          ValueError,
          "Features must have shape.*for feature 'exogenous'"):
        model_fn(
            features={
                feature_keys.TrainEvalFeatures.TIMES: [[1]],
                feature_keys.TrainEvalFeatures.VALUES: [[[1., 2., 3.]]],
                "exogenous": [[1], [2]]
            },
            labels=None,
            mode=mode)


class PredictFeatureCheckingTests(test.TestCase):

  def test_no_time_feature(self):
    model_fn = _stub_model_fn()
    with self.assertRaisesRegexp(ValueError, "Expected a '{}' feature".format(
        feature_keys.PredictionFeatures.TIMES)):
      model_fn(
          features={
              feature_keys.PredictionFeatures.STATE_TUPLE: ([[[1.]]], 1.)
          },
          labels=None,
          mode=estimator_lib.ModeKeys.PREDICT)

  def test_no_start_state_feature(self):
    model_fn = _stub_model_fn()
    with self.assertRaisesRegexp(ValueError, "Expected a '{}' feature".format(
        feature_keys.PredictionFeatures.STATE_TUPLE)):
      model_fn(
          features={feature_keys.PredictionFeatures.TIMES: [[1]]},
          labels=None,
          mode=estimator_lib.ModeKeys.PREDICT)

  def test_bad_time_rank(self):
    model_fn = _stub_model_fn()
    with self.assertRaisesRegexp(ValueError,
                                 "Expected shape.*for feature '{}'".format(
                                     feature_keys.PredictionFeatures.TIMES)):
      model_fn(
          features={
              feature_keys.PredictionFeatures.TIMES: 1,
              feature_keys.PredictionFeatures.STATE_TUPLE: (1, (2, 3.))
          },
          labels=None,
          mode=estimator_lib.ModeKeys.PREDICT)

  def test_bad_exogenous_shape(self):
    model_fn = _stub_model_fn()
    with self.assertRaisesRegexp(
        ValueError,
        "Features must have shape.*for feature 'exogenous'"):
      model_fn(
          features={
              feature_keys.PredictionFeatures.TIMES: [[1]],
              feature_keys.PredictionFeatures.STATE_TUPLE: (1, (2, 3.)),
              "exogenous": 1.
          },
          labels=None,
          mode=estimator_lib.ModeKeys.PREDICT)


def _custom_time_series_regressor(
    model_dir, head_type, exogenous_feature_columns):
  return ts_estimators.TimeSeriesRegressor(
      model=lstm_example._LSTMModel(
          num_features=5, num_units=128,
          exogenous_feature_columns=exogenous_feature_columns),
      optimizer=adam.AdamOptimizer(0.001),
      config=estimator_lib.RunConfig(tf_random_seed=4),
      state_manager=state_management.ChainingStateManager(),
      head_type=head_type,
      model_dir=model_dir)


def _structural_ensemble_regressor(
    model_dir, head_type, exogenous_feature_columns):
  return ts_estimators.StructuralEnsembleRegressor(
      periodicities=None,
      num_features=5,
      exogenous_feature_columns=exogenous_feature_columns,
      head_type=head_type,
      model_dir=model_dir)


def _ar_lstm_regressor(
    model_dir, head_type, exogenous_feature_columns):
  return ts_estimators.TimeSeriesRegressor(
      model=ar_model.ARModel(
          periodicities=10, input_window_size=10, output_window_size=6,
          num_features=5,
          exogenous_feature_columns=exogenous_feature_columns,
          prediction_model_factory=functools.partial(
              ar_model.LSTMPredictionModel,
              num_units=10)),
      head_type=head_type,
      model_dir=model_dir)


class OneShotTests(parameterized.TestCase):

  @parameterized.named_parameters(
      {"testcase_name": "ar_lstm_regressor",
       "estimator_factory": _ar_lstm_regressor},
      {"testcase_name": "custom_time_series_regressor",
       "estimator_factory": _custom_time_series_regressor},
      {"testcase_name": "structural_ensemble_regressor",
       "estimator_factory": _structural_ensemble_regressor})
  def test_one_shot_prediction_head_export(self, estimator_factory):
    def _new_temp_dir():
      return os.path.join(test.get_temp_dir(), str(ops.uid()))
    model_dir = _new_temp_dir()
    categorical_column = feature_column.categorical_column_with_hash_bucket(
        key="categorical_exogenous_feature", hash_bucket_size=16)
    exogenous_feature_columns = [
        feature_column.numeric_column(
            "2d_exogenous_feature", shape=(2,)),
        feature_column.embedding_column(
            categorical_column=categorical_column, dimension=10)]
    estimator = estimator_factory(
        model_dir=model_dir,
        exogenous_feature_columns=exogenous_feature_columns,
        head_type=ts_head_lib.OneShotPredictionHead)
    train_features = {
        feature_keys.TrainEvalFeatures.TIMES: numpy.arange(
            20, dtype=numpy.int64),
        feature_keys.TrainEvalFeatures.VALUES: numpy.tile(numpy.arange(
            20, dtype=numpy.float32)[:, None], [1, 5]),
        "2d_exogenous_feature": numpy.ones([20, 2]),
        "categorical_exogenous_feature": numpy.array(
            ["strkey"] * 20)[:, None]
    }
    train_input_fn = input_pipeline.RandomWindowInputFn(
        input_pipeline.NumpyReader(train_features), shuffle_seed=2,
        num_threads=1, batch_size=16, window_size=16)
    estimator.train(input_fn=train_input_fn, steps=5)
    result = estimator.evaluate(input_fn=train_input_fn, steps=1)
    self.assertNotIn(feature_keys.State.STATE_TUPLE, result)
    input_receiver_fn = estimator.build_raw_serving_input_receiver_fn()
    export_location = estimator.export_savedmodel(_new_temp_dir(),
                                                  input_receiver_fn)
    graph = ops.Graph()
    with graph.as_default():
      with session_lib.Session() as session:
        signatures = loader.load(
            session, [tag_constants.SERVING], export_location)
        self.assertEqual([feature_keys.SavedModelLabels.PREDICT],
                         list(signatures.signature_def.keys()))
        predict_signature = signatures.signature_def[
            feature_keys.SavedModelLabels.PREDICT]
        six.assertCountEqual(
            self,
            [feature_keys.FilteringFeatures.TIMES,
             feature_keys.FilteringFeatures.VALUES,
             "2d_exogenous_feature",
             "categorical_exogenous_feature"],
            predict_signature.inputs.keys())
        features = {
            feature_keys.TrainEvalFeatures.TIMES: numpy.tile(
                numpy.arange(35, dtype=numpy.int64)[None, :], [2, 1]),
            feature_keys.TrainEvalFeatures.VALUES: numpy.tile(numpy.arange(
                20, dtype=numpy.float32)[None, :, None], [2, 1, 5]),
            "2d_exogenous_feature": numpy.ones([2, 35, 2]),
            "categorical_exogenous_feature": numpy.tile(numpy.array(
                ["strkey"] * 35)[None, :, None], [2, 1, 1])
        }
        feeds = {
            graph.as_graph_element(input_value.name): features[input_key]
            for input_key, input_value in predict_signature.inputs.items()}
        fetches = {output_key: graph.as_graph_element(output_value.name)
                   for output_key, output_value
                   in predict_signature.outputs.items()}
        output = session.run(fetches, feed_dict=feeds)
        self.assertEqual((2, 15, 5), output["mean"].shape)
    # Build a parsing input function, then make a tf.Example for it to parse.
    export_location = estimator.export_savedmodel(
        _new_temp_dir(),
        estimator.build_one_shot_parsing_serving_input_receiver_fn(
            filtering_length=20, prediction_length=15))
    graph = ops.Graph()
    with graph.as_default():
      with session_lib.Session() as session:
        example = example_pb2.Example()
        times = example.features.feature[feature_keys.TrainEvalFeatures.TIMES]
        values = example.features.feature[feature_keys.TrainEvalFeatures.VALUES]
        times.int64_list.value.extend(range(35))
        for i in range(20):
          values.float_list.value.extend(
              [float(i) * 2. + feature_number
               for feature_number in range(5)])
        real_feature = example.features.feature["2d_exogenous_feature"]
        categortical_feature = example.features.feature[
            "categorical_exogenous_feature"]
        for i in range(35):
          real_feature.float_list.value.extend([1, 1])
          categortical_feature.bytes_list.value.append(b"strkey")
        # Serialize the tf.Example for feeding to the Session
        examples = [example.SerializeToString()] * 2
        signatures = loader.load(
            session, [tag_constants.SERVING], export_location)
        predict_signature = signatures.signature_def[
            feature_keys.SavedModelLabels.PREDICT]
        ((_, input_value),) = predict_signature.inputs.items()
        feeds = {graph.as_graph_element(input_value.name): examples}
        fetches = {output_key: graph.as_graph_element(output_value.name)
                   for output_key, output_value
                   in predict_signature.outputs.items()}
        output = session.run(fetches, feed_dict=feeds)
        self.assertEqual((2, 15, 5), output["mean"].shape)


if __name__ == "__main__":
  test.main()
