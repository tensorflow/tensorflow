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
"""Enum for model prediction keys."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.estimator import model_fn


class MetricKeys(object):
  """Metric key strings."""
  LOSS = model_fn.LOSS_METRIC_KEY
  LOSS_MEAN = model_fn.AVERAGE_LOSS_METRIC_KEY

  ACCURACY = 'accuracy'
  # This is the best the model could do by always predicting one class.
  # Should be < ACCURACY in a trained model.
  ACCURACY_BASELINE = 'accuracy_baseline'
  AUC = 'auc'
  AUC_PR = 'auc_precision_recall'
  LABEL_MEAN = 'label/mean'
  PREDICTION_MEAN = 'prediction/mean'

  # The following require a threshold applied, should be float in range (0, 1).
  ACCURACY_AT_THRESHOLD = 'accuracy/positive_threshold_%g'
  PRECISION_AT_THRESHOLD = 'precision/positive_threshold_%g'
  RECALL_AT_THRESHOLD = 'recall/positive_threshold_%g'
