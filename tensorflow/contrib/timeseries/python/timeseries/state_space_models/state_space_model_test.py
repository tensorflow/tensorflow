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
"""Tests for state space model infrastructure."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy

from tensorflow.contrib import layers

from tensorflow.contrib.timeseries.python.timeseries import estimators
from tensorflow.contrib.timeseries.python.timeseries import feature_keys
from tensorflow.contrib.timeseries.python.timeseries import input_pipeline
from tensorflow.contrib.timeseries.python.timeseries import math_utils
from tensorflow.contrib.timeseries.python.timeseries import saved_model_utils
from tensorflow.contrib.timeseries.python.timeseries import state_management
from tensorflow.contrib.timeseries.python.timeseries import test_utils
from tensorflow.contrib.timeseries.python.timeseries.state_space_models import state_space_model

from tensorflow.python.estimator import estimator_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.training import coordinator as coordinator_lib
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import queue_runner_impl


class RandomStateSpaceModel(state_space_model.StateSpaceModel):

  def __init__(self,
               state_dimension,
               state_noise_dimension,
               configuration=state_space_model.StateSpaceModelConfiguration()):
    self.transition = numpy.random.normal(
        size=[state_dimension, state_dimension]).astype(
            configuration.dtype.as_numpy_dtype)
    self.noise_transform = numpy.random.normal(
        size=(state_dimension, state_noise_dimension)).astype(
            configuration.dtype.as_numpy_dtype)
    # Test batch broadcasting
    self.observation_model = numpy.random.normal(
        size=(configuration.num_features, state_dimension)).astype(
            configuration.dtype.as_numpy_dtype)
    super(RandomStateSpaceModel, self).__init__(
        configuration=configuration._replace(
            covariance_prior_fn=lambda _: 0.))

  def get_state_transition(self):
    return self.transition

  def get_noise_transform(self):
    return self.noise_transform

  def get_observation_model(self, times):
    return self.observation_model


class ConstructionTests(test.TestCase):

  def test_initialize_graph_error(self):
    with self.assertRaisesRegexp(ValueError, "initialize_graph"):
      model = RandomStateSpaceModel(2, 2)
      outputs = model.define_loss(
          features={
              feature_keys.TrainEvalFeatures.TIMES:
                  constant_op.constant([[1, 2]]),
              feature_keys.TrainEvalFeatures.VALUES:
                  constant_op.constant([[[1.], [2.]]])
          },
          mode=estimator_lib.ModeKeys.TRAIN)
      initializer = variables.global_variables_initializer()
      with self.test_session() as sess:
        sess.run([initializer])
        outputs.loss.eval()

  def test_initialize_graph_state_manager_error(self):
    with self.assertRaisesRegexp(ValueError, "initialize_graph"):
      model = RandomStateSpaceModel(2, 2)
      state_manager = state_management.ChainingStateManager()
      outputs = state_manager.define_loss(
          model=model,
          features={
              feature_keys.TrainEvalFeatures.TIMES:
                  constant_op.constant([[1, 2]]),
              feature_keys.TrainEvalFeatures.VALUES:
                  constant_op.constant([[[1.], [2.]]])
          },
          mode=estimator_lib.ModeKeys.TRAIN)
      initializer = variables.global_variables_initializer()
      with self.test_session() as sess:
        sess.run([initializer])
        outputs.loss.eval()


class GapTests(test.TestCase):

  def _gap_test_template(self, times, values):
    random_model = RandomStateSpaceModel(
        state_dimension=1, state_noise_dimension=1,
        configuration=state_space_model.StateSpaceModelConfiguration(
            num_features=1))
    random_model.initialize_graph()
    input_fn = input_pipeline.WholeDatasetInputFn(
        input_pipeline.NumpyReader({
            feature_keys.TrainEvalFeatures.TIMES: times,
            feature_keys.TrainEvalFeatures.VALUES: values
        }))
    features, _ = input_fn()
    times = features[feature_keys.TrainEvalFeatures.TIMES]
    values = features[feature_keys.TrainEvalFeatures.VALUES]
    model_outputs = random_model.get_batch_loss(
        features={
            feature_keys.TrainEvalFeatures.TIMES: times,
            feature_keys.TrainEvalFeatures.VALUES: values
        },
        mode=None,
        state=math_utils.replicate_state(
            start_state=random_model.get_start_state(),
            batch_size=array_ops.shape(times)[0]))
    with self.test_session() as session:
      variables.global_variables_initializer().run()
      coordinator = coordinator_lib.Coordinator()
      queue_runner_impl.start_queue_runners(session, coord=coordinator)
      model_outputs.loss.eval()
      coordinator.request_stop()
      coordinator.join()

  def test_start_gap(self):
    self._gap_test_template(times=[20, 21, 22], values=numpy.arange(3))

  def test_mid_gap(self):
    self._gap_test_template(times=[2, 60, 61], values=numpy.arange(3))

  def test_end_gap(self):
    self._gap_test_template(times=[2, 3, 73], values=numpy.arange(3))

  def test_all_gaps(self):
    self._gap_test_template(times=[2, 4, 8, 16, 32, 64, 128],
                            values=numpy.arange(7))


class StateSpaceEquivalenceTests(test.TestCase):

  def test_savedmodel_state_override(self):
    random_model = RandomStateSpaceModel(
        state_dimension=5,
        state_noise_dimension=4,
        configuration=state_space_model.StateSpaceModelConfiguration(
            exogenous_feature_columns=[layers.real_valued_column("exogenous")],
            dtype=dtypes.float64, num_features=1))
    estimator = estimators.StateSpaceRegressor(
        model=random_model,
        optimizer=gradient_descent.GradientDescentOptimizer(0.1))
    combined_input_fn = input_pipeline.WholeDatasetInputFn(
        input_pipeline.NumpyReader({
            feature_keys.FilteringFeatures.TIMES: [1, 2, 3, 4],
            feature_keys.FilteringFeatures.VALUES: [1., 2., 3., 4.],
            "exogenous": [-1., -2., -3., -4.]
        }))
    estimator.train(combined_input_fn, steps=1)
    export_location = estimator.export_savedmodel(
        self.get_temp_dir(),
        estimator.build_raw_serving_input_receiver_fn(
            exogenous_features={
                "exogenous": numpy.zeros((0, 0), dtype=numpy.float32)}))
    with ops.Graph().as_default() as graph:
      random_model.initialize_graph()
      with self.test_session(graph=graph) as session:
        variables.global_variables_initializer().run()
        evaled_start_state = session.run(random_model.get_start_state())
    evaled_start_state = [
        state_element[None, ...] for state_element in evaled_start_state]
    with ops.Graph().as_default() as graph:
      with self.test_session(graph=graph) as session:
        signatures = loader.load(
            session, [tag_constants.SERVING], export_location)
        first_split_filtering = saved_model_utils.filter_continuation(
            continue_from={
                feature_keys.FilteringResults.STATE_TUPLE: evaled_start_state},
            signatures=signatures,
            session=session,
            features={
                feature_keys.FilteringFeatures.TIMES: [1, 2],
                feature_keys.FilteringFeatures.VALUES: [1., 2.],
                "exogenous": [-1., -2.]})
        second_split_filtering = saved_model_utils.filter_continuation(
            continue_from=first_split_filtering,
            signatures=signatures,
            session=session,
            features={
                feature_keys.FilteringFeatures.TIMES: [3, 4],
                feature_keys.FilteringFeatures.VALUES: [3., 4.],
                "exogenous": [-3., -4.]
            })
        combined_filtering = saved_model_utils.filter_continuation(
            continue_from={
                feature_keys.FilteringResults.STATE_TUPLE: evaled_start_state},
            signatures=signatures,
            session=session,
            features={
                feature_keys.FilteringFeatures.TIMES: [1, 2, 3, 4],
                feature_keys.FilteringFeatures.VALUES: [1., 2., 3., 4.],
                "exogenous": [-1., -2., -3., -4.]
            })
        split_predict = saved_model_utils.predict_continuation(
            continue_from=second_split_filtering,
            signatures=signatures,
            session=session,
            steps=1,
            exogenous_features={
                "exogenous": [[-5.]]})
        combined_predict = saved_model_utils.predict_continuation(
            continue_from=combined_filtering,
            signatures=signatures,
            session=session,
            steps=1,
            exogenous_features={
                "exogenous": [[-5.]]})
    for state_key, combined_state_value in combined_filtering.items():
      if state_key == feature_keys.FilteringResults.TIMES:
        continue
      self.assertAllClose(
          combined_state_value, second_split_filtering[state_key])
    for prediction_key, combined_value in combined_predict.items():
      self.assertAllClose(combined_value, split_predict[prediction_key])

  def _equivalent_to_single_model_test_template(self, model_generator):
    with self.test_session() as session:
      random_model = RandomStateSpaceModel(
          state_dimension=5,
          state_noise_dimension=4,
          configuration=state_space_model.StateSpaceModelConfiguration(
              dtype=dtypes.float64, num_features=1))
      random_model.initialize_graph()
      series_length = 10
      model_data = random_model.generate(
          number_of_series=1, series_length=series_length,
          model_parameters=random_model.random_model_parameters())
      input_fn = input_pipeline.WholeDatasetInputFn(
          input_pipeline.NumpyReader(model_data))
      features, _ = input_fn()
      model_outputs = random_model.get_batch_loss(
          features=features,
          mode=None,
          state=math_utils.replicate_state(
              start_state=random_model.get_start_state(),
              batch_size=array_ops.shape(
                  features[feature_keys.TrainEvalFeatures.TIMES])[0]))
      variables.global_variables_initializer().run()
      compare_outputs_evaled_fn = model_generator(
          random_model, model_data)
      coordinator = coordinator_lib.Coordinator()
      queue_runner_impl.start_queue_runners(session, coord=coordinator)
      compare_outputs_evaled = compare_outputs_evaled_fn(session)
      model_outputs_evaled = session.run(
          (model_outputs.end_state, model_outputs.predictions))
      coordinator.request_stop()
      coordinator.join()
      model_posteriors, model_predictions = model_outputs_evaled
      (_, compare_posteriors,
       compare_predictions) = compare_outputs_evaled
      (model_posterior_mean, model_posterior_var,
       model_from_time) = model_posteriors
      (compare_posterior_mean, compare_posterior_var,
       compare_from_time) = compare_posteriors
      self.assertAllClose(model_posterior_mean, compare_posterior_mean[0])
      self.assertAllClose(model_posterior_var, compare_posterior_var[0])
      self.assertAllClose(model_from_time, compare_from_time)
      self.assertEqual(sorted(model_predictions.keys()),
                       sorted(compare_predictions.keys()))
      for prediction_name in model_predictions:
        if prediction_name == "loss":
          # Chunking means that losses will be different; skip testing them.
          continue
        # Compare the last chunk to their corresponding un-chunked model
        # predictions
        last_prediction_chunk = compare_predictions[prediction_name][-1]
        comparison_values = last_prediction_chunk.shape[0]
        model_prediction = (
            model_predictions[prediction_name][0, -comparison_values:])
        self.assertAllClose(model_prediction,
                            last_prediction_chunk)

  def _model_equivalent_to_chained_model_test_template(self, chunk_size):
    def chained_model_outputs(original_model, data):
      input_fn = test_utils.AllWindowInputFn(
          input_pipeline.NumpyReader(data), window_size=chunk_size)
      state_manager = state_management.ChainingStateManager(
          state_saving_interval=1)
      features, _ = input_fn()
      state_manager.initialize_graph(original_model)
      model_outputs = state_manager.define_loss(
          model=original_model,
          features=features,
          mode=estimator_lib.ModeKeys.TRAIN)
      def _eval_outputs(session):
        for _ in range(50):
          # Warm up saved state
          model_outputs.loss.eval()
        (posterior_mean, posterior_var,
         priors_from_time) = model_outputs.end_state
        posteriors = ((posterior_mean,), (posterior_var,), priors_from_time)
        outputs = (model_outputs.loss, posteriors,
                   model_outputs.predictions)
        chunked_outputs_evaled = session.run(outputs)
        return chunked_outputs_evaled
      return _eval_outputs
    self._equivalent_to_single_model_test_template(chained_model_outputs)

  def test_model_equivalent_to_chained_model_chunk_size_one(self):
    numpy.random.seed(2)
    random_seed.set_random_seed(3)
    self._model_equivalent_to_chained_model_test_template(1)

  def test_model_equivalent_to_chained_model_chunk_size_five(self):
    numpy.random.seed(4)
    random_seed.set_random_seed(5)
    self._model_equivalent_to_chained_model_test_template(5)


class PredictionTests(test.TestCase):

  def _check_predictions(
      self, predicted_mean, predicted_covariance, window_size):
    self.assertAllEqual(predicted_covariance.shape,
                        [1,   # batch
                         window_size,
                         1,   # num features
                         1])  # num features
    self.assertAllEqual(predicted_mean.shape,
                        [1,   # batch
                         window_size,
                         1])  # num features
    for position in range(window_size - 2):
      self.assertGreater(predicted_covariance[0, position + 2, 0, 0],
                         predicted_covariance[0, position, 0, 0])

  def test_predictions_direct(self):
    dtype = dtypes.float64
    with variable_scope.variable_scope(dtype.name):
      random_model = RandomStateSpaceModel(
          state_dimension=5, state_noise_dimension=4,
          configuration=state_space_model.StateSpaceModelConfiguration(
              dtype=dtype, num_features=1))
      random_model.initialize_graph()
      prediction_dict = random_model.predict(features={
          feature_keys.PredictionFeatures.TIMES: [[1, 3, 5, 6]],
          feature_keys.PredictionFeatures.STATE_TUPLE:
              math_utils.replicate_state(
                  start_state=random_model.get_start_state(), batch_size=1)
      })
      with self.test_session():
        variables.global_variables_initializer().run()
        predicted_mean = prediction_dict["mean"].eval()
        predicted_covariance = prediction_dict["covariance"].eval()
      self._check_predictions(predicted_mean, predicted_covariance,
                              window_size=4)

  def test_predictions_after_loss(self):
    dtype = dtypes.float32
    with variable_scope.variable_scope(dtype.name):
      random_model = RandomStateSpaceModel(
          state_dimension=5, state_noise_dimension=4,
          configuration=state_space_model.StateSpaceModelConfiguration(
              dtype=dtype, num_features=1))
      features = {
          feature_keys.TrainEvalFeatures.TIMES: [[1, 2, 3, 4]],
          feature_keys.TrainEvalFeatures.VALUES:
              array_ops.ones([1, 4, 1], dtype=dtype)
      }
      passthrough = state_management.PassthroughStateManager()
      random_model.initialize_graph()
      passthrough.initialize_graph(random_model)
      model_outputs = passthrough.define_loss(
          model=random_model,
          features=features,
          mode=estimator_lib.ModeKeys.EVAL)
      predictions = random_model.predict({
          feature_keys.PredictionFeatures.TIMES: [[5, 7, 8]],
          feature_keys.PredictionFeatures.STATE_TUPLE: model_outputs.end_state
      })
      with self.test_session():
        variables.global_variables_initializer().run()
        predicted_mean = predictions["mean"].eval()
        predicted_covariance = predictions["covariance"].eval()
      self._check_predictions(predicted_mean, predicted_covariance,
                              window_size=3)


class ExogenousTests(test.TestCase):

  def test_noise_increasing(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      with variable_scope.variable_scope(dtype.name):
        random_model = RandomStateSpaceModel(
            state_dimension=5, state_noise_dimension=4,
            configuration=state_space_model.StateSpaceModelConfiguration(
                dtype=dtype, num_features=1))
        original_covariance = array_ops.diag(array_ops.ones(shape=[5]))
        _, new_covariance, _ = random_model._exogenous_noise_increasing(
            current_times=[[1]],
            exogenous_values=[[5.]],
            state=[
                array_ops.ones(shape=[1, 5]), original_covariance[None], [0]
            ])
        with self.test_session() as session:
          variables.global_variables_initializer().run()
          evaled_new_covariance, evaled_original_covariance = session.run(
              [new_covariance[0], original_covariance])
          new_variances = numpy.diag(evaled_new_covariance)
          original_variances = numpy.diag(evaled_original_covariance)
          for i in range(5):
            self.assertGreater(new_variances[i], original_variances[i])

  def test_noise_decreasing(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      with variable_scope.variable_scope(dtype.name):
        random_model = RandomStateSpaceModel(
            state_dimension=5, state_noise_dimension=4,
            configuration=state_space_model.StateSpaceModelConfiguration(
                dtype=dtype, num_features=1))
        random_model.initialize_graph()
        original_covariance = array_ops.diag(
            array_ops.ones(shape=[5], dtype=dtype))
        _, new_covariance, _ = random_model._exogenous_noise_decreasing(
            current_times=[[1]],
            exogenous_values=constant_op.constant([[-2.]], dtype=dtype),
            state=[
                -array_ops.ones(shape=[1, 5], dtype=dtype),
                original_covariance[None], [0]
            ])
        with self.test_session() as session:
          variables.global_variables_initializer().run()
          evaled_new_covariance, evaled_original_covariance = session.run(
              [new_covariance[0], original_covariance])
          new_variances = numpy.diag(evaled_new_covariance)
          original_variances = numpy.diag(evaled_original_covariance)
          for i in range(5):
            self.assertLess(new_variances[i], original_variances[i])


class StubStateSpaceModel(state_space_model.StateSpaceModel):

  def __init__(self,
               transition,
               state_noise_dimension,
               configuration=state_space_model.StateSpaceModelConfiguration()):
    self.transition = transition
    self.noise_transform = numpy.random.normal(
        size=(transition.shape[0], state_noise_dimension)).astype(numpy.float32)
    # Test feature + batch broadcasting
    self.observation_model = numpy.random.normal(
        size=(transition.shape[0])).astype(numpy.float32)
    super(StubStateSpaceModel, self).__init__(
        configuration=configuration)

  def get_state_transition(self):
    return self.transition

  def get_noise_transform(self):
    return self.noise_transform

  def get_observation_model(self, times):
    return self.observation_model


GeneratedModel = collections.namedtuple(
    "GeneratedModel", ["model", "data", "true_parameters"])


class PosteriorTests(test.TestCase):

  def _get_cycle_transition(self, period):
    cycle_transition = numpy.zeros([period - 1, period - 1],
                                   dtype=numpy.float32)
    cycle_transition[0, :] = -1
    cycle_transition[1:, :-1] = numpy.identity(period - 2)
    return cycle_transition

  _adder_transition = numpy.array([[1, 1],
                                   [0, 1]], dtype=numpy.float32)

  def _get_single_model(self):
    numpy.random.seed(8)
    stub_model = StubStateSpaceModel(
        transition=self._get_cycle_transition(5), state_noise_dimension=0)
    series_length = 1000
    stub_model.initialize_graph()
    true_params = stub_model.random_model_parameters()
    data = stub_model.generate(
        number_of_series=1, series_length=series_length,
        model_parameters=true_params)
    return GeneratedModel(
        model=stub_model, data=data, true_parameters=true_params)

  def test_exact_posterior_recovery_no_transition_noise(self):
    with self.test_session() as session:
      stub_model, data, true_params = self._get_single_model()
      input_fn = input_pipeline.WholeDatasetInputFn(
          input_pipeline.NumpyReader(data))
      features, _ = input_fn()
      model_outputs = stub_model.get_batch_loss(
          features=features,
          mode=None,
          state=math_utils.replicate_state(
              start_state=stub_model.get_start_state(),
              batch_size=array_ops.shape(
                  features[feature_keys.TrainEvalFeatures.TIMES])[0]))
      variables.global_variables_initializer().run()
      coordinator = coordinator_lib.Coordinator()
      queue_runner_impl.start_queue_runners(session, coord=coordinator)
      posterior_mean, posterior_var, posterior_times = session.run(
          # Feed the true model parameters so that this test doesn't depend on
          # the generated parameters being close to the variable initializations
          # (an alternative would be training steps to fit the noise values,
          # which would be slow).
          model_outputs.end_state, feed_dict=true_params)
      coordinator.request_stop()
      coordinator.join()

      self.assertAllClose(numpy.zeros([1, 4, 4]), posterior_var,
                          atol=1e-2)
      self.assertAllClose(
          numpy.dot(
              numpy.linalg.matrix_power(
                  stub_model.transition,
                  data[feature_keys.TrainEvalFeatures.TIMES].shape[1]),
              true_params[stub_model.prior_state_mean]),
          posterior_mean[0],
          rtol=1e-1)
      self.assertAllClose(
          math_utils.batch_end_time(
              features[feature_keys.TrainEvalFeatures.TIMES]).eval(),
          posterior_times)

  def test_chained_exact_posterior_recovery_no_transition_noise(self):
    with self.test_session() as session:
      stub_model, data, true_params = self._get_single_model()
      chunk_size = 10
      input_fn = test_utils.AllWindowInputFn(
          input_pipeline.NumpyReader(data), window_size=chunk_size)
      features, _ = input_fn()
      state_manager = state_management.ChainingStateManager(
          state_saving_interval=1)
      state_manager.initialize_graph(stub_model)
      model_outputs = state_manager.define_loss(
          model=stub_model,
          features=features,
          mode=estimator_lib.ModeKeys.TRAIN)
      variables.global_variables_initializer().run()
      coordinator = coordinator_lib.Coordinator()
      queue_runner_impl.start_queue_runners(session, coord=coordinator)
      for _ in range(
          data[feature_keys.TrainEvalFeatures.TIMES].shape[1] // chunk_size):
        model_outputs.loss.eval()
      posterior_mean, posterior_var, posterior_times = session.run(
          model_outputs.end_state, feed_dict=true_params)
      coordinator.request_stop()
      coordinator.join()
      self.assertAllClose(numpy.zeros([1, 4, 4]), posterior_var,
                          atol=1e-2)
      self.assertAllClose(
          numpy.dot(
              numpy.linalg.matrix_power(
                  stub_model.transition,
                  data[feature_keys.TrainEvalFeatures.TIMES].shape[1]),
              true_params[stub_model.prior_state_mean]),
          posterior_mean[0],
          rtol=1e-1)
      self.assertAllClose(data[feature_keys.TrainEvalFeatures.TIMES][:, -1],
                          posterior_times)


class TimeDependentStateSpaceModel(state_space_model.StateSpaceModel):
  """A mostly trivial model which predicts values = times + 1."""

  def __init__(self, static_unrolling_window_size_threshold=None):
    super(TimeDependentStateSpaceModel, self).__init__(
        configuration=state_space_model.StateSpaceModelConfiguration(
            use_observation_noise=False,
            static_unrolling_window_size_threshold=
            static_unrolling_window_size_threshold))

  def get_state_transition(self):
    return array_ops.ones(shape=[1, 1])

  def get_noise_transform(self):
    return array_ops.ones(shape=[1, 1])

  def get_observation_model(self, times):
    return array_ops.reshape(
        tensor=math_ops.cast(times + 1, dtypes.float32), shape=[-1, 1, 1])

  def make_priors(self):
    return (ops.convert_to_tensor([1.]), ops.convert_to_tensor([[0.]]))


class UnknownShapeModel(TimeDependentStateSpaceModel):

  def get_observation_model(self, times):
    parent_model = super(UnknownShapeModel, self).get_observation_model(times)
    parent_model._shape = tensor_shape.unknown_shape()
    assert parent_model.get_shape().ndims is None
    return parent_model


class TimeDependentTests(test.TestCase):

  def _time_dependency_test_template(self, model_type):
    """Test that a time-dependent observation model influences predictions."""
    model = model_type()
    estimator = estimators.StateSpaceRegressor(
        model=model, optimizer=gradient_descent.GradientDescentOptimizer(0.1))
    values = numpy.reshape([1., 2., 3., 4.],
                           newshape=[1, 4, 1])
    input_fn = input_pipeline.WholeDatasetInputFn(
        input_pipeline.NumpyReader({
            feature_keys.TrainEvalFeatures.TIMES: [[0, 1, 2, 3]],
            feature_keys.TrainEvalFeatures.VALUES: values
        }))
    estimator.train(input_fn=input_fn, max_steps=1)
    predicted_values = estimator.evaluate(input_fn=input_fn, steps=1)["mean"]
    # Throw out the first value so we don't test the prior
    self.assertAllEqual(values[1:], predicted_values[1:])

  def test_undefined_shape_time_dependency(self):
    self._time_dependency_test_template(UnknownShapeModel)

  def test_loop_unrolling(self):
    """Tests running/restoring from a checkpoint with static unrolling."""
    model = TimeDependentStateSpaceModel(
        # Unroll during training, but not evaluation
        static_unrolling_window_size_threshold=2)
    estimator = estimators.StateSpaceRegressor(model=model)
    times = numpy.arange(100)
    values = numpy.arange(100)
    dataset = {
        feature_keys.TrainEvalFeatures.TIMES: times,
        feature_keys.TrainEvalFeatures.VALUES: values
    }
    train_input_fn = input_pipeline.RandomWindowInputFn(
        input_pipeline.NumpyReader(dataset), batch_size=16, window_size=2)
    eval_input_fn = input_pipeline.WholeDatasetInputFn(
        input_pipeline.NumpyReader(dataset))
    estimator.train(input_fn=train_input_fn, max_steps=1)
    estimator.evaluate(input_fn=eval_input_fn, steps=1)


class LevelOnlyModel(state_space_model.StateSpaceModel):

  def get_state_transition(self):
    return linalg_ops.eye(1, dtype=self.dtype)

  def get_noise_transform(self):
    return linalg_ops.eye(1, dtype=self.dtype)

  def get_observation_model(self, times):
    return [1]


class MultivariateLevelModel(
    state_space_model.StateSpaceCorrelatedFeaturesEnsemble):

  def __init__(self, configuration):
    univariate_component_configuration = configuration._replace(
        num_features=1)
    components = []
    for feature in range(configuration.num_features):
      with variable_scope.variable_scope("feature{}".format(feature)):
        components.append(
            LevelOnlyModel(configuration=univariate_component_configuration))
    super(MultivariateLevelModel, self).__init__(
        ensemble_members=components, configuration=configuration)


class MultivariateTests(test.TestCase):

  def test_multivariate(self):
    dtype = dtypes.float32
    num_features = 3
    covariance = numpy.eye(num_features)
    # A single off-diagonal has a non-zero value in the true transition
    # noise covariance.
    covariance[-1, 0] = 1.
    covariance[0, -1] = 1.
    dataset_size = 100
    values = numpy.cumsum(
        numpy.random.multivariate_normal(
            mean=numpy.zeros(num_features),
            cov=covariance,
            size=dataset_size),
        axis=0)
    times = numpy.arange(dataset_size)
    model = MultivariateLevelModel(
        configuration=state_space_model.StateSpaceModelConfiguration(
            num_features=num_features,
            dtype=dtype,
            use_observation_noise=False,
            transition_covariance_initial_log_scale_bias=5.))
    estimator = estimators.StateSpaceRegressor(
        model=model, optimizer=gradient_descent.GradientDescentOptimizer(0.1))
    data = {
        feature_keys.TrainEvalFeatures.TIMES: times,
        feature_keys.TrainEvalFeatures.VALUES: values
    }
    train_input_fn = input_pipeline.RandomWindowInputFn(
        input_pipeline.NumpyReader(data), batch_size=16, window_size=16)
    estimator.train(input_fn=train_input_fn, steps=1)
    for component in model._ensemble_members:
      # Check that input statistics propagated to component models
      self.assertTrue(component._input_statistics)

  def test_ensemble_observation_noise(self):
    model = MultivariateLevelModel(
        configuration=state_space_model.StateSpaceModelConfiguration())
    model.initialize_graph()
    outputs = model.define_loss(
        features={
            feature_keys.TrainEvalFeatures.TIMES:
                constant_op.constant([[1, 2]]),
            feature_keys.TrainEvalFeatures.VALUES:
                constant_op.constant([[[1.], [2.]]])
        },
        mode=estimator_lib.ModeKeys.TRAIN)
    initializer = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run([initializer])
      outputs.loss.eval()

if __name__ == "__main__":
  test.main()
