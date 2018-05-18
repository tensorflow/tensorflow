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
"""Utilities for testing time series models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.timeseries.python.timeseries import estimators
from tensorflow.contrib.timeseries.python.timeseries import input_pipeline
from tensorflow.contrib.timeseries.python.timeseries import state_management
from tensorflow.contrib.timeseries.python.timeseries.feature_keys import TrainEvalFeatures

from tensorflow.python.client import session
from tensorflow.python.estimator import estimator_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import adam
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import coordinator as coordinator_lib
from tensorflow.python.training import queue_runner_impl
from tensorflow.python.util import nest


class AllWindowInputFn(input_pipeline.TimeSeriesInputFn):
  """Returns all contiguous windows of data from a full dataset.

  In contrast to WholeDatasetInputFn, which does basic shape checking but
  maintains the flat sequencing of data, this `TimeSeriesInputFn` creates
  batches of windows. However, unlike `RandomWindowInputFn` these windows are
  deterministic, starting at every possible offset (i.e. batches of size
  series_length - window_size + 1 are produced).
  """

  def __init__(self, time_series_reader, window_size):
    """Initialize the input_pipeline.

    Args:
      time_series_reader: A `input_pipeline.TimeSeriesReader` object.
      window_size: The size of contiguous windows of data to produce.
    """
    self._window_size = window_size
    self._reader = time_series_reader
    super(AllWindowInputFn, self).__init__()

  def create_batch(self):
    features = self._reader.read_full()
    times = features[TrainEvalFeatures.TIMES]
    num_windows = array_ops.shape(times)[0] - self._window_size + 1
    indices = array_ops.reshape(math_ops.range(num_windows), [num_windows, 1])
    # indices contains the starting point for each window. We now extend these
    # indices to include the elements inside the windows as well by doing a
    # broadcast addition.
    increments = array_ops.reshape(math_ops.range(self._window_size), [1, -1])
    all_indices = array_ops.reshape(indices + increments, [-1])
    # Select the appropriate elements in the batch and reshape the output to 3D.
    features = {
        key: array_ops.reshape(
            array_ops.gather(value, all_indices),
            array_ops.concat(
                [[num_windows, self._window_size], array_ops.shape(value)[1:]],
                axis=0))
        for key, value in features.items()
    }
    return (features, None)


class _SavingTensorHook(basic_session_run_hooks.LoggingTensorHook):
  """A hook to save Tensors during training."""

  def __init__(self, tensors, every_n_iter=None, every_n_secs=None):
    self.tensor_values = {}
    super(_SavingTensorHook, self).__init__(
        tensors=tensors, every_n_iter=every_n_iter,
        every_n_secs=every_n_secs)

  def after_run(self, run_context, run_values):
    del run_context
    if self._should_trigger:
      for tag in self._current_tensors.keys():
        self.tensor_values[tag] = run_values.results[tag]
      self._timer.update_last_triggered_step(self._iter_count)
    self._iter_count += 1


def _train_on_generated_data(
    generate_fn, generative_model, train_iterations, seed,
    learning_rate=0.1, ignore_params_fn=lambda _: (),
    derived_param_test_fn=lambda _: (),
    train_input_fn_type=input_pipeline.WholeDatasetInputFn,
    train_state_manager=state_management.PassthroughStateManager()):
  """The training portion of parameter recovery tests."""
  random_seed.set_random_seed(seed)
  generate_graph = ops.Graph()
  with generate_graph.as_default():
    with session.Session(graph=generate_graph):
      generative_model.initialize_graph()
      time_series_reader, true_parameters = generate_fn(generative_model)
      true_parameters = {
          tensor.name: value for tensor, value in true_parameters.items()}
  eval_input_fn = input_pipeline.WholeDatasetInputFn(time_series_reader)
  eval_state_manager = state_management.PassthroughStateManager()
  true_parameter_eval_graph = ops.Graph()
  with true_parameter_eval_graph.as_default():
    generative_model.initialize_graph()
    ignore_params = ignore_params_fn(generative_model)
    feature_dict, _ = eval_input_fn()
    eval_state_manager.initialize_graph(generative_model)
    feature_dict[TrainEvalFeatures.VALUES] = math_ops.cast(
        feature_dict[TrainEvalFeatures.VALUES], generative_model.dtype)
    model_outputs = eval_state_manager.define_loss(
        model=generative_model,
        features=feature_dict,
        mode=estimator_lib.ModeKeys.EVAL)
    with session.Session(graph=true_parameter_eval_graph) as sess:
      variables.global_variables_initializer().run()
      coordinator = coordinator_lib.Coordinator()
      queue_runner_impl.start_queue_runners(sess, coord=coordinator)
      true_param_loss = model_outputs.loss.eval(feed_dict=true_parameters)
      true_transformed_params = {
          param: param.eval(feed_dict=true_parameters)
          for param in derived_param_test_fn(generative_model)}
      coordinator.request_stop()
      coordinator.join()

  saving_hook = _SavingTensorHook(
      tensors=true_parameters.keys(),
      every_n_iter=train_iterations - 1)

  class _RunConfig(estimator_lib.RunConfig):

    @property
    def tf_random_seed(self):
      return seed

  estimator = estimators.TimeSeriesRegressor(
      model=generative_model,
      config=_RunConfig(),
      state_manager=train_state_manager,
      optimizer=adam.AdamOptimizer(learning_rate))
  train_input_fn = train_input_fn_type(time_series_reader=time_series_reader)
  trained_loss = (estimator.train(
      input_fn=train_input_fn,
      max_steps=train_iterations,
      hooks=[saving_hook]).evaluate(
          input_fn=eval_input_fn, steps=1))["loss"]
  logging.info("Final trained loss: %f", trained_loss)
  logging.info("True parameter loss: %f", true_param_loss)
  return (ignore_params, true_parameters, true_transformed_params,
          trained_loss, true_param_loss, saving_hook,
          true_parameter_eval_graph)


def test_parameter_recovery(
    generate_fn, generative_model, train_iterations, test_case, seed,
    learning_rate=0.1, rtol=0.2, atol=0.1, train_loss_tolerance_coeff=0.99,
    ignore_params_fn=lambda _: (),
    derived_param_test_fn=lambda _: (),
    train_input_fn_type=input_pipeline.WholeDatasetInputFn,
    train_state_manager=state_management.PassthroughStateManager()):
  """Test that a generative model fits generated data.

  Args:
    generate_fn: A function taking a model and returning a `TimeSeriesReader`
        object and dictionary mapping parameters to their
        values. model.initialize_graph() will have been called on the model
        before it is passed to this function.
    generative_model: A timeseries.model.TimeSeriesModel instance to test.
    train_iterations: Number of training steps.
    test_case: A tf.test.TestCase to run assertions on.
    seed: Same as for TimeSeriesModel.unconditional_generate().
    learning_rate: Step size for optimization.
    rtol: Relative tolerance for tests.
    atol: Absolute tolerance for tests.
    train_loss_tolerance_coeff: Trained loss times this value must be less
        than the loss evaluated using the generated parameters.
    ignore_params_fn: Function mapping from a Model to a list of parameters
        which are not tested for accurate recovery.
    derived_param_test_fn: Function returning a list of derived parameters
        (Tensors) which are checked for accurate recovery (comparing the value
        evaluated with trained parameters to the value under the true
        parameters).

        As an example, for VARMA, in addition to checking AR and MA parameters,
        this function can be used to also check lagged covariance. See
        varma_ssm.py for details.
    train_input_fn_type: The `TimeSeriesInputFn` type to use when training
        (likely `WholeDatasetInputFn` or `RandomWindowInputFn`). If None, use
        `WholeDatasetInputFn`.
    train_state_manager: The state manager to use when training (likely
        `PassthroughStateManager` or `ChainingStateManager`). If None, use
        `PassthroughStateManager`.
  """
  (ignore_params, true_parameters, true_transformed_params,
   trained_loss, true_param_loss, saving_hook, true_parameter_eval_graph
  ) = _train_on_generated_data(
      generate_fn=generate_fn, generative_model=generative_model,
      train_iterations=train_iterations, seed=seed, learning_rate=learning_rate,
      ignore_params_fn=ignore_params_fn,
      derived_param_test_fn=derived_param_test_fn,
      train_input_fn_type=train_input_fn_type,
      train_state_manager=train_state_manager)
  trained_parameter_substitutions = {}
  for param in true_parameters.keys():
    evaled_value = saving_hook.tensor_values[param]
    trained_parameter_substitutions[param] = evaled_value
    true_value = true_parameters[param]
    logging.info("True %s: %s, learned: %s",
                 param, true_value, evaled_value)
  with session.Session(graph=true_parameter_eval_graph):
    for transformed_param, true_value in true_transformed_params.items():
      trained_value = transformed_param.eval(
          feed_dict=trained_parameter_substitutions)
      logging.info("True %s [transformed parameter]: %s, learned: %s",
                   transformed_param, true_value, trained_value)
      test_case.assertAllClose(true_value, trained_value,
                               rtol=rtol, atol=atol)

  if ignore_params is None:
    ignore_params = []
  else:
    ignore_params = nest.flatten(ignore_params)
  ignore_params = [tensor.name for tensor in ignore_params]
  if trained_loss > 0:
    test_case.assertLess(trained_loss * train_loss_tolerance_coeff,
                         true_param_loss)
  else:
    test_case.assertLess(trained_loss / train_loss_tolerance_coeff,
                         true_param_loss)
  for param in true_parameters.keys():
    if param in ignore_params:
      continue
    evaled_value = saving_hook.tensor_values[param]
    true_value = true_parameters[param]
    test_case.assertAllClose(true_value, evaled_value,
                             rtol=rtol, atol=atol)


def parameter_recovery_dry_run(
    generate_fn, generative_model, seed,
    learning_rate=0.1,
    train_input_fn_type=input_pipeline.WholeDatasetInputFn,
    train_state_manager=state_management.PassthroughStateManager()):
  """Test that a generative model can train on generated data.

  Args:
    generate_fn: A function taking a model and returning a
        `input_pipeline.TimeSeriesReader` object and a dictionary mapping
        parameters to their values. model.initialize_graph() will have been
        called on the model before it is passed to this function.
    generative_model: A timeseries.model.TimeSeriesModel instance to test.
    seed: Same as for TimeSeriesModel.unconditional_generate().
    learning_rate: Step size for optimization.
    train_input_fn_type: The type of `TimeSeriesInputFn` to use when training
        (likely `WholeDatasetInputFn` or `RandomWindowInputFn`). If None, use
        `WholeDatasetInputFn`.
    train_state_manager: The state manager to use when training (likely
        `PassthroughStateManager` or `ChainingStateManager`). If None, use
        `PassthroughStateManager`.
  """
  _train_on_generated_data(
      generate_fn=generate_fn, generative_model=generative_model,
      seed=seed, learning_rate=learning_rate,
      train_input_fn_type=train_input_fn_type,
      train_state_manager=train_state_manager,
      train_iterations=2)
