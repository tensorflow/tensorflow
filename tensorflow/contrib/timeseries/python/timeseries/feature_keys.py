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
"""Commonly used special feature names for time series models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.saved_model import signature_constants


class State(object):
  """Key formats for accepting/returning state."""
  # The model-dependent state to start from, as a single tuple.
  STATE_TUPLE = "start_tuple"
  # Same meaning as STATE_TUPLE, but prefixes keys representing flattened model
  # state rather than mapping to a nested tuple containing model state,
  # primarily for use with export_savedmodel.
  STATE_PREFIX = "model_state"


class Times(object):
  """Key formats for accepting/returning times."""
  # An increasing vector of integers.
  TIMES = "times"


class Values(object):
  """Key formats for accepting/returning values."""
  # Floating point, with one or more values corresponding to each time in TIMES.
  VALUES = "values"


class TrainEvalFeatures(Times, Values):
  """Feature names used during training and evaluation."""
  pass


class PredictionFeatures(Times, State):
  """Feature names used during prediction."""
  pass


class FilteringFeatures(Times, Values, State):
  """Special feature names for filtering."""
  pass


class PredictionResults(Times):
  """Keys returned when predicting (not comprehensive)."""
  pass


class FilteringResults(Times, State):
  """Keys returned from evaluation/filtering."""
  pass


class SavedModelLabels(object):
  """Names of signatures exported with export_savedmodel."""
  PREDICT = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
  FILTER = "filter"
