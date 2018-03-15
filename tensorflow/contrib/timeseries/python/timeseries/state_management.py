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
"""Classes for wrapping a model to operate on different data shapes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from tensorflow.contrib.timeseries.python.timeseries import feature_keys
from tensorflow.contrib.timeseries.python.timeseries import math_utils
from tensorflow.contrib.timeseries.python.timeseries.model import ModelOutputs

from tensorflow.python.estimator import estimator_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest


class PassthroughStateManager(object):
  """A minimal wrapper for models which do not need state management."""

  def __init__(self):
    self._input_statistics = None
    self._graph_initialized = False

  def initialize_graph(self, model, input_statistics=None):
    """Adds required operations to the graph."""
    del model  # unused
    self._graph_initialized = True
    self._input_statistics = input_statistics

  def define_loss(self, model, features, mode):
    """Wrap "model" with StateManager-specific operations.

    Args:
      model: The model (inheriting from TimeSeriesModel) to manage state for.
      features: A dictionary with the following key/value pairs:
        feature_keys.TrainEvalFeatures.TIMES: A [batch size x window size]
            Tensor with times for each observation.
        feature_keys.TrainEvalFeatures.VALUES: A [batch size x window size x num
            features] Tensor with values for each observation.
      mode: The tf.estimator.ModeKeys mode to use (TRAIN or EVAL).
    Returns:
      A ModelOutputs object.
    Raises:
      ValueError: If start state was specified.
    """
    if feature_keys.State.STATE_TUPLE in features:
      raise ValueError(
          "Overriding start state is not supported for this model.")
    return model.define_loss(features, mode)


class _OverridableStateManager(PassthroughStateManager):
  """Base class for state managers which support overriding model state."""

  @abc.abstractmethod
  def _define_loss_with_saved_state(self, model, features, mode):
    pass

  def define_loss(self, model, features, mode):
    """Switches between explicit start state and managed state."""
    if feature_keys.FilteringFeatures.STATE_TUPLE in features:
      # Explicit start state has been provided, so we should use that.
      if mode == estimator_lib.ModeKeys.TRAIN:
        raise ValueError(
            "Overriding saved state for training is not supported (but a value "
            "for feature {} was specified).".format(
                feature_keys.FilteringFeatures.STATE_TUPLE))
      start_state = features[feature_keys.FilteringFeatures.STATE_TUPLE]
      del features[feature_keys.FilteringFeatures.STATE_TUPLE]
      return model.get_batch_loss(
          features=features, mode=mode, state=start_state)
    else:
      # No explicit start state; use managed state.
      return self._define_loss_with_saved_state(
          model=model, features=features, mode=mode)


class FilteringOnlyStateManager(_OverridableStateManager):
  """State manager for models which use state only for filtering.

  Window-based models (ARModel) do not require state to be fed during training
  (instead requiring a specific window size). Rather than requiring a minimum
  window size for filtering, these models maintain this window in their state,
  and so need state to be fed.
  """

  def _define_loss_with_saved_state(self, model, features, mode):
    return model.define_loss(features, mode)


class ChainingStateManager(_OverridableStateManager):
  """Maintains state across a batch for SequentialTimeSeriesModel subclasses.

  The batch dimension is treated as indexing sequential chunks of the same
  timeseries. End state from each chunk is fed as start state to the next chunk
  during the next timestep. This is an approximation to full-batch training for
  sequential models, but is typically much faster while still accurately
  recovering parameters. The speedup comes from reduced scheduling overhead of
  TensorFlow ops, since each operation can do much more work.
  """

  def __init__(self, state_saving_interval=20, checkpoint_state=False):
    """Initialize the state manager.

    Args:
      state_saving_interval: This state manager saves intermediate model state
          every `state_saving_interval` times. Larger values save memory, and
          checkpoint size if `checkpoint_state` is enabled, but models
          will need to impute across artificial gaps of up to this size
          (i.e. gaps not appearing in the original data). This imputation may
          affect training. Set state_saving_interval to 1 to avoid any
          artificial imputation.
      checkpoint_state: If True, saved intermediate model state will be
          written to checkpoints. Checkpoints will then scale with dataset
          size. If False, state will be freshly imputed from the beginning of a
          series each time the model is restored, which means it may take a few
          iterations for state to warm up.
    """
    super(ChainingStateManager, self).__init__()
    self._checkpoint_state = checkpoint_state
    self._state_saving_interval = state_saving_interval
    self._start_state = None
    self._cached_states = None

  def initialize_graph(self, model, input_statistics=None):
    """Adds required operations to the graph."""
    super(ChainingStateManager, self).initialize_graph(
        model=model, input_statistics=input_statistics)
    self._start_state = model.get_start_state()
    self._cached_states = math_utils.TupleOfTensorsLookup(
        key_dtype=dtypes.int64,
        default_values=self._start_state,
        empty_key=-1,
        name="cached_states",
        checkpoint=self._checkpoint_state)

  def _define_loss_with_saved_state(self, model, features, mode):
    """Feeds end state from one training iteration into the next.

    Args:
      model: The model to wrap. Compatible with children of TimeSeriesModel.
      features: Dictionary with Tensor values defining the data to be
        processed. The expected key/value pairs are at minimum:
          feature_keys.TrainEvalFeatures.TIMES: A [number of chunks x window
            size] Tensor with times for each observation, the result of chunking
            a single longer time series.
          feature_keys.TrainEvalFeatures.VALUES: A [number of chunks x window
            size x num features] Tensor with values for each observation,
            corresponding to times.
      mode: The tf.estimator.ModeKeys mode to use. For EVAL and INFER, no
          batching is performed, which may be slow. This is to avoid giving
          cached and almost certainly stale values.
    Returns:
      A ModelOutputs object.
    Raises:
      ValueError: If initialize_graph has not been called.
    """
    if not self._graph_initialized:
      raise ValueError("ChainingStateManager requires initialize_graph() to be "
                       "called before use.")
    (loss_op, end_state, batch_predictions) = self._update_cached_states(
        model=model,
        features=features,
        mode=mode)
    # Add a batch dimension so state can be used directly (e.g. for predictions)
    # without the user manually reshaping it.
    last_end_state_flat = [end_state_value[-1][None]
                           for end_state_value in nest.flatten(end_state)]
    batch_predictions["observed"] = features[
        feature_keys.TrainEvalFeatures.VALUES]
    return ModelOutputs(
        loss=loss_op,
        end_state=nest.pack_sequence_as(end_state, last_end_state_flat),
        predictions=batch_predictions,
        prediction_times=features[feature_keys.TrainEvalFeatures.TIMES])

  def _get_chunk_number(self, time):
    return time // self._state_saving_interval

  def _get_cached_states(self, times):
    """Retrieve cached states for a batch of times."""
    read_chunk_numbers = self._get_chunk_number(times)
    looked_up_state = list(self._cached_states.lookup(
        math_ops.cast(read_chunk_numbers, dtypes.int64)))
    looked_up_state = tuple(looked_up_state)
    # We need to special-case the first chunk in a series to explicitly rely on
    # the model's starting state so that gradients flow back to it. Otherwise it
    # would affect only initialization, and would not be read from or updated
    # during training. Not doing this also isolates that part of the graph,
    # leading to errors on model reload if there are trainable variables
    # affecting a model's start state.
    if self._input_statistics is not None:
      start_time = self._input_statistics.start_time
    else:
      start_time = 0
    set_to_start_state = math_ops.equal(read_chunk_numbers,
                                        self._get_chunk_number(start_time))
    new_states = []
    for start_state_value, cache_variable in zip(
        nest.flatten(
            math_utils.replicate_state(self._start_state,
                                       array_ops.shape(times)[0])),
        nest.flatten(looked_up_state)):

      new_states.append(
          array_ops.where(set_to_start_state, start_state_value,
                          cache_variable))
    looked_up_state = nest.pack_sequence_as(looked_up_state, new_states)
    return looked_up_state

  def _update_cached_states(self, model, features, mode):
    """Read, process, and write chunks to the cache."""
    times = features[feature_keys.TrainEvalFeatures.TIMES]
    looked_up_state = self._get_cached_states(times[:, 0])
    (model_loss, intermediate_states,
     batch_predictions) = model.per_step_batch_loss(
         features=features,
         mode=mode,
         state=looked_up_state)
    # We need to at least write to the bucket after the one we read from.
    min_chunk_numbers = self._get_chunk_number(times) + 1
    # We write to the bucket that would have been read had the window started at
    # the next sample (except for the last sample in the window, which gets
    # written to the next bucket). This assumes fixed missing times (i.e. if we
    # were presented with times [10, 50] we will never see times [30, 50]).
    #
    # TODO(allenl): Retrieve the highest time less than the current time rather
    # than relying on fixed bucketing.
    write_chunk_numbers = math_ops.maximum(
        self._get_chunk_number(array_ops.concat(
            [times[:, 1:], times[:, -1:] + 1], axis=1)),
        min_chunk_numbers)
    # Write once for every computed state; this may mean that we write multiple
    # times to the same cell, but later writes will take precedence.
    save_ops = [
        self._cached_states.insert(
            keys=write_chunk_numbers,
            values=intermediate_states)]
    end_state = nest.pack_sequence_as(
        intermediate_states,
        [state_element[:, -1]
         for state_element in nest.flatten(intermediate_states)])
    with ops.control_dependencies(save_ops):
      # Make sure end states get saved at each iteration
      loss_op = array_ops.identity(model_loss)
    return loss_op, end_state, batch_predictions
