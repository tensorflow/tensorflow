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
"""Tests for state management."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from tensorflow.contrib.timeseries.python.timeseries import feature_keys
from tensorflow.contrib.timeseries.python.timeseries import input_pipeline
from tensorflow.contrib.timeseries.python.timeseries import math_utils
from tensorflow.contrib.timeseries.python.timeseries import model
from tensorflow.contrib.timeseries.python.timeseries import state_management
from tensorflow.contrib.timeseries.python.timeseries import test_utils

from tensorflow.python.estimator import estimator_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator as coordinator_lib
from tensorflow.python.training import queue_runner_impl
from tensorflow.python.training import training as train
from tensorflow.python.util import nest


class StubTimeSeriesModel(model.TimeSeriesModel):

  def __init__(self, correct_offset=False):
    self._correct_offset = correct_offset
    super(StubTimeSeriesModel, self).__init__(1)

  def initialize_graph(self, input_statistics=None):
    super(StubTimeSeriesModel, self).initialize_graph(
        input_statistics=input_statistics)
    self.prior_var = variable_scope.get_variable(
        "prior", [], initializer=init_ops.constant_initializer(0.))

  def generate(self, *args):
    pass

  def predict(self, *args):
    pass

  def get_start_state(self):
    return (array_ops.zeros([], dtype=dtypes.int64), self.prior_var)

  def get_batch_loss(self, features, mode, state):
    raise NotImplementedError("This stub only supports managed state.")

  def per_step_batch_loss(self, features, mode, state):
    times = features[feature_keys.TrainEvalFeatures.TIMES]
    values = features[feature_keys.TrainEvalFeatures.VALUES]
    (priors_from_time, prior) = state
    time_corrected_priors = prior + math_ops.cast(
        math_utils.batch_start_time(times) - priors_from_time, dtypes.float32)
    posterior = time_corrected_priors[:, None] + math_ops.cast(
        times - math_utils.batch_start_time(times)[:, None], dtypes.float32)
    batch_end_values = array_ops.squeeze(
        array_ops.slice(values, [0, array_ops.shape(times)[1] - 1, 0],
                        [-1, 1, -1]),
        axis=[1, 2])
    # A pretty odd but easy to think about loss: L1 loss on the batch end
    # values.
    loss = math_ops.reduce_sum(
        math_ops.abs(
            array_ops.reshape(posterior[:, -1], [-1]) - batch_end_values))
    if self._correct_offset:
      posterior += batch_end_values[0] - posterior[0, -1]
    posteriors = (times, posterior)
    return loss, posteriors, {"dummy_predictions": array_ops.zeros_like(values)}


class ChainingStateManagerTest(test.TestCase):

  def _make_test_data(self, length, cut_start, cut_end, offset, step=1):
    times_full = step * numpy.arange(length, dtype=numpy.int64)
    values_full = offset + step * numpy.arange(length, dtype=numpy.float32)
    if cut_start is not None:
      times = numpy.concatenate((times_full[:cut_start],
                                 times_full[cut_end:]))
      values = numpy.concatenate((values_full[:cut_start],
                                  values_full[cut_end:]))
    else:
      times = times_full
      values = values_full
    return {
        feature_keys.TrainEvalFeatures.TIMES: times,
        feature_keys.TrainEvalFeatures.VALUES: values
    }

  def _test_initialization(self, warmup_iterations, batch_size):
    stub_model = StubTimeSeriesModel()
    data = self._make_test_data(length=20, cut_start=None, cut_end=None,
                                offset=0.)
    if batch_size == -1:
      input_fn = test_utils.AllWindowInputFn(
          input_pipeline.NumpyReader(data), window_size=10)
    else:
      input_fn = input_pipeline.RandomWindowInputFn(
          input_pipeline.NumpyReader(data),
          window_size=10,
          batch_size=batch_size)
    chainer = state_management.ChainingStateManager(
        state_saving_interval=1)
    features, _ = input_fn()
    stub_model.initialize_graph()
    chainer.initialize_graph(model=stub_model)
    model_outputs = chainer.define_loss(
        model=stub_model, features=features, mode=estimator_lib.ModeKeys.TRAIN)
    with self.cached_session() as session:
      variables.global_variables_initializer().run()
      coordinator = coordinator_lib.Coordinator()
      queue_runner_impl.start_queue_runners(session, coord=coordinator)
      for _ in range(warmup_iterations):
        # Warm up saved state
        model_outputs.loss.eval()
      outputs = model_outputs.loss.eval()
      coordinator.request_stop()
      coordinator.join()
      return outputs

  def test_zero_initializations(self):
    # Even with no initialization, we are imputing values up to each chunk,
    # which in this case gives exact values.
    self.assertEqual(0., self._test_initialization(
        warmup_iterations=0, batch_size=-1))

  def test_one_initializations(self):
    # Further initialization should still be correct, if redundant
    self.assertEqual(0., self._test_initialization(
        warmup_iterations=1, batch_size=-1))

  def test_stochastic_batch(self):
    # It shouldn't matter whether we're using a full deterministic batch or a
    # smaller stochastic batch.
    self.assertEqual(0., self._test_initialization(
        warmup_iterations=1, batch_size=5))

  def _test_pass_to_next(self, read_offset, step, correct_offset):
    stub_model = StubTimeSeriesModel(correct_offset=correct_offset)
    data = self._make_test_data(
        length=100 + read_offset, cut_start=None, cut_end=None, offset=100.,
        step=step)
    init_input_fn = input_pipeline.WholeDatasetInputFn(
        input_pipeline.NumpyReader(
            {k: v[:-read_offset] for k, v in data.items()}))
    result_input_fn = input_pipeline.WholeDatasetInputFn(
        input_pipeline.NumpyReader(
            {k: v[read_offset:] for k, v in data.items()}))

    chainer = state_management.ChainingStateManager(
        state_saving_interval=1)
    stub_model.initialize_graph()
    chainer.initialize_graph(model=stub_model)
    init_model_outputs = chainer.define_loss(
        model=stub_model, features=init_input_fn()[0],
        mode=estimator_lib.ModeKeys.TRAIN)
    result_model_outputs = chainer.define_loss(
        model=stub_model, features=result_input_fn()[0],
        mode=estimator_lib.ModeKeys.TRAIN)
    with self.cached_session() as session:
      variables.global_variables_initializer().run()
      coordinator = coordinator_lib.Coordinator()
      queue_runner_impl.start_queue_runners(session, coord=coordinator)
      init_model_outputs.loss.eval()
      returned_loss = result_model_outputs.loss.eval()
      coordinator.request_stop()
      coordinator.join()
      return returned_loss

  def test_pass_to_next_step_one_no_correction(self):
    self.assertEqual(100., self._test_pass_to_next(
        read_offset=1, step=1, correct_offset=False))

  def test_pass_to_next_step_one_with_correction(self):
    self.assertEqual(0., self._test_pass_to_next(
        read_offset=1, step=1, correct_offset=True))

  def test_pass_to_next_step_three_with_correction(self):
    self.assertEqual(0., self._test_pass_to_next(
        read_offset=1, step=3, correct_offset=True))

  def test_large_read_offset(self):
    self.assertEqual(0., self._test_pass_to_next(
        read_offset=50, step=20, correct_offset=True))

  def test_past_init_offset(self):
    self.assertEqual(100., self._test_pass_to_next(
        read_offset=100, step=20, correct_offset=True))

  def _test_missing_values(self, cut_start, cut_end, offset):
    stub_model = StubTimeSeriesModel()
    data = self._make_test_data(
        length=100, cut_start=cut_start, cut_end=cut_end, offset=offset)
    input_fn = test_utils.AllWindowInputFn(
        input_pipeline.NumpyReader(data), window_size=10)
    chainer = state_management.ChainingStateManager(
        state_saving_interval=1)
    features, _ = input_fn()
    stub_model.initialize_graph()
    chainer.initialize_graph(model=stub_model)
    model_outputs = chainer.define_loss(
        model=stub_model, features=features, mode=estimator_lib.ModeKeys.TRAIN)
    with self.cached_session() as session:
      variables.global_variables_initializer().run()
      coordinator = coordinator_lib.Coordinator()
      queue_runner_impl.start_queue_runners(session, coord=coordinator)
      for _ in range(10):
        model_outputs.loss.eval()
      returned_loss = model_outputs.loss.eval()
      coordinator.request_stop()
      coordinator.join()
      return returned_loss

  def test_missing_values_ten(self):
    # Each posterior should be off by 10 from the offset in the values. 90
    # values with a chunk size of 10 means 90 - 10 + 1 possible chunks.
    self.assertEqual((90 - 10 + 1) * 10, self._test_missing_values(
        cut_start=20, cut_end=30, offset=10.))

  def test_missing_values_five(self):
    self.assertEqual((95 - 10 + 1) * 10, self._test_missing_values(
        cut_start=15, cut_end=20, offset=10.))


class _StateOverrideModel(model.TimeSeriesModel):

  def __init__(self):
    super(_StateOverrideModel, self).__init__(num_features=1)

  def generate(self, *args):
    pass

  def predict(self, *args):
    pass

  def get_start_state(self):
    return (constant_op.constant([20, 30, 40], dtype=dtypes.int64),
            (constant_op.constant(-10, dtype=dtypes.int64),
             constant_op.constant([30., 50.], dtype=dtypes.float64)))

  def get_batch_loss(self, features, mode, state):
    per_observation_loss, state, outputs = self.per_step_batch_loss(
        features, mode, state)
    state = nest.map_structure(lambda element: element[:, -1], state)
    outputs["observed"] = features[feature_keys.TrainEvalFeatures.VALUES]
    return model.ModelOutputs(
        loss=per_observation_loss,
        end_state=state,
        predictions=outputs,
        prediction_times=features[feature_keys.TrainEvalFeatures.TIMES])

  def per_step_batch_loss(self, features, mode, state):
    return (
        constant_op.constant(1.),
        # Assumes only one step: this is the per-step loss.
        nest.map_structure(
            lambda element: ops.convert_to_tensor(element)[:, None], state),
        {
            "dummy_predictions":
                array_ops.zeros_like(
                    features[feature_keys.TrainEvalFeatures.VALUES])
        })


class _StateOverrideTest(test.TestCase):

  def test_state_override(self):
    test_start_state = (numpy.array([[2, 3, 4]]), (numpy.array([2]),
                                                   numpy.array([[3., 5.]])))
    data = {
        feature_keys.FilteringFeatures.TIMES: numpy.arange(5),
        feature_keys.FilteringFeatures.VALUES: numpy.zeros(shape=[5, 3])
    }
    features, _ = input_pipeline.WholeDatasetInputFn(
        input_pipeline.NumpyReader(data))()
    features[feature_keys.FilteringFeatures.STATE_TUPLE] = test_start_state
    stub_model = _StateOverrideModel()
    chainer = state_management.ChainingStateManager()
    stub_model.initialize_graph()
    chainer.initialize_graph(model=stub_model)
    model_outputs = chainer.define_loss(
        model=stub_model, features=features, mode=estimator_lib.ModeKeys.EVAL)
    with train.MonitoredSession() as session:
      end_state = session.run(model_outputs.end_state)
    nest.assert_same_structure(test_start_state, end_state)
    for expected, received in zip(
        nest.flatten(test_start_state), nest.flatten(end_state)):
      self.assertAllEqual(expected, received)


if __name__ == "__main__":
  test.main()
