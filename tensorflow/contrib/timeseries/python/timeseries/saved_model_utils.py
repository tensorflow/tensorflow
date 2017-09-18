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
"""Convenience functions for working with time series saved_models.

@@predict_continuation
@@filter_continuation
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.timeseries.python.timeseries import feature_keys as _feature_keys
from tensorflow.contrib.timeseries.python.timeseries import input_pipeline as _input_pipeline
from tensorflow.contrib.timeseries.python.timeseries import model_utils as _model_utils

from tensorflow.python.util.all_util import remove_undocumented


def _colate_features_to_feeds_and_fetches(continue_from, signature, features,
                                          graph):
  """Uses a saved model signature to construct feed and fetch dictionaries."""
  if _feature_keys.FilteringResults.STATE_TUPLE in continue_from:
    # We're continuing from an evaluation, so we need to unpack/flatten state.
    state_values = _model_utils.state_to_dictionary(
        continue_from[_feature_keys.FilteringResults.STATE_TUPLE])
  else:
    state_values = continue_from
  input_feed_tensors_by_name = {
      input_key: graph.as_graph_element(input_value.name)
      for input_key, input_value in signature.inputs.items()
  }
  output_tensors_by_name = {
      output_key: graph.as_graph_element(output_value.name)
      for output_key, output_value in signature.outputs.items()
  }
  feed_dict = {}
  for state_key, state_value in state_values.items():
    feed_dict[input_feed_tensors_by_name[state_key]] = state_value
  for feature_key, feature_value in features.items():
    feed_dict[input_feed_tensors_by_name[feature_key]] = feature_value
  return output_tensors_by_name, feed_dict


def predict_continuation(continue_from,
                         signatures,
                         session,
                         steps=None,
                         times=None,
                         exogenous_features=None):
  """Perform prediction using an exported saved model.

  Analogous to _input_pipeline.predict_continuation_input_fn, but operates on a
  saved model rather than feeding into Estimator's predict method.

  Args:
    continue_from: A dictionary containing the results of either an Estimator's
      evaluate method or filter_continuation. Used to determine the model
      state to make predictions starting from.
    signatures: The `MetaGraphDef` protocol buffer returned from
      `tf.saved_model.loader.load`. Used to determine the names of Tensors to
      feed and fetch. Must be from the same model as `continue_from`.
    session: The session to use. The session's graph must be the one into which
      `tf.saved_model.loader.load` loaded the model.
    steps: The number of steps to predict (scalar), starting after the
      evaluation or filtering. If `times` is specified, `steps` must not be; one
      is required.
    times: A [batch_size x window_size] array of integers (not a Tensor)
      indicating times to make predictions for. These times must be after the
      corresponding evaluation or filtering. If `steps` is specified, `times`
      must not be; one is required. If the batch dimension is omitted, it is
      assumed to be 1.
    exogenous_features: Optional dictionary. If specified, indicates exogenous
      features for the model to use while making the predictions. Values must
      have shape [batch_size x window_size x ...], where `batch_size` matches
      the batch dimension used when creating `continue_from`, and `window_size`
      is either the `steps` argument or the `window_size` of the `times`
      argument (depending on which was specified).
  Returns:
    A dictionary with model-specific predictions (typically having keys "mean"
    and "covariance") and a feature_keys.PredictionResults.TIMES key indicating
    the times for which the predictions were computed.
  Raises:
    ValueError: If `times` or `steps` are misspecified.
  """
  if exogenous_features is None:
    exogenous_features = {}
  predict_times = _model_utils.canonicalize_times_or_steps_from_output(
      times=times, steps=steps, previous_model_output=continue_from)
  features = {_feature_keys.PredictionFeatures.TIMES: predict_times}
  features.update(exogenous_features)
  predict_signature = signatures.signature_def[
      _feature_keys.SavedModelLabels.PREDICT]
  output_tensors_by_name, feed_dict = _colate_features_to_feeds_and_fetches(
      continue_from=continue_from,
      signature=predict_signature,
      features=features,
      graph=session.graph)
  output = session.run(output_tensors_by_name, feed_dict=feed_dict)
  output[_feature_keys.PredictionResults.TIMES] = features[
      _feature_keys.PredictionFeatures.TIMES]
  return output


def filter_continuation(continue_from, signatures, session, features):
  """Perform filtering using an exported saved model.

  Filtering refers to updating model state based on new observations.
  Predictions based on the returned model state will be conditioned on these
  observations.

  Args:
    continue_from: A dictionary containing the results of either an Estimator's
      evaluate method or a previous filter_continuation. Used to determine the
      model state to start filtering from.
    signatures: The `MetaGraphDef` protocol buffer returned from
      `tf.saved_model.loader.load`. Used to determine the names of Tensors to
      feed and fetch. Must be from the same model as `continue_from`.
    session: The session to use. The session's graph must be the one into which
      `tf.saved_model.loader.load` loaded the model.
    features: A dictionary mapping keys to Numpy arrays, with several possible
      shapes (requires keys `FilteringFeatures.TIMES` and
      `FilteringFeatures.VALUES`):
        Single example; `TIMES` is a scalar and `VALUES` is either a scalar or a
          vector of length [number of features].
        Sequence; `TIMES` is a vector of shape [series length], `VALUES` either
          has shape [series length] (univariate) or [series length x number of
          features] (multivariate).
        Batch of sequences; `TIMES` is a vector of shape [batch size x series
          length], `VALUES` has shape [batch size x series length] or [batch
          size x series length x number of features].
      In any case, `VALUES` and any exogenous features must have their shapes
      prefixed by the shape of the value corresponding to the `TIMES` key.
  Returns:
    A dictionary containing model state updated to account for the observations
    in `features`.
  """
  filter_signature = signatures.signature_def[
      _feature_keys.SavedModelLabels.FILTER]
  features = _input_pipeline._canonicalize_numpy_data(  # pylint: disable=protected-access
      data=features,
      require_single_batch=False)
  output_tensors_by_name, feed_dict = _colate_features_to_feeds_and_fetches(
      continue_from=continue_from,
      signature=filter_signature,
      features=features,
      graph=session.graph)
  output = session.run(output_tensors_by_name, feed_dict=feed_dict)
  # Make it easier to chain filter -> predict by keeping track of the current
  # time.
  output[_feature_keys.FilteringResults.TIMES] = features[
      _feature_keys.FilteringFeatures.TIMES]
  return output

remove_undocumented(module_name=__name__)
