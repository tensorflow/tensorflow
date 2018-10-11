# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Debug and model explainability logic for boosted trees."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.core.kernels.boosted_trees import boosted_trees_pb2

# For directional feature contributions.
_DEBUG_PROTO_KEY = '_serialized_debug_outputs_proto'
_BIAS_ID = 0


def _parse_debug_proto_string(example_proto_serialized):
  example_debug_outputs = boosted_trees_pb2.DebugOutput()
  example_debug_outputs.ParseFromString(example_proto_serialized)
  feature_ids = example_debug_outputs.feature_ids
  logits_path = example_debug_outputs.logits_path
  return feature_ids, logits_path


def _compute_directional_feature_contributions(example_feature_ids,
                                               example_logits_paths, activation,
                                               num_bucketized_features):
  """Directional feature contributions and bias, per example."""
  # Initialize contributions to 0.
  dfcs = {k: 0 for k in range(num_bucketized_features)}

  # Traverse tree subtracting child prediction from parent prediction and
  # associating change with feature id used to split.
  predictions = np.array(activation(example_logits_paths))
  delta_pred = predictions[_BIAS_ID + 1:] - predictions[:-1]
  # Group by feature id, then sum delta_pred.
  contribs = np.bincount(
      example_feature_ids,
      weights=delta_pred,
      minlength=num_bucketized_features)
  for f, dfc in zip(range(num_bucketized_features), contribs):
    dfcs[f] = dfc
  return predictions[_BIAS_ID], dfcs


def _identity(logits):
  return logits


def _sigmoid(logits):
  # TODO(crawles): Change to softmax once multiclass support is available.
  return 1 / (1 + np.exp(-np.array(logits)))


def _parse_explanations_from_prediction(serialized_debug_proto,
                                        n_features,
                                        classification=False):
  """Parse serialized explanability proto, compute dfc, and return bias, dfc."""
  feature_ids, logits_path = _parse_debug_proto_string(serialized_debug_proto)
  if classification:
    activation = _sigmoid
  else:
    activation = _identity
  bias, dfcs = _compute_directional_feature_contributions(
      feature_ids, logits_path, activation, n_features)
  # TODO(crawles): Prediction path and leaf IDs.
  return bias, dfcs
