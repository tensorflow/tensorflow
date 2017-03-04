# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Enum for metric keys."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class MetricKey(object):
  """Metric key strings."""
  LOSS = "loss"
  AUC = "auc"
  CLASS_AUC = "auc/class%d"
  PREDICTION_MEAN = "labels/prediction_mean"
  CLASS_PREDICTION_MEAN = "labels/prediction_mean/class%d"
  CLASS_LOGITS_MEAN = "labels/logits_mean/class%d"
  CLASS_PROBABILITY_MEAN = "labels/probability_mean/class%d"
  LABEL_MEAN = "labels/actual_label_mean"
  CLASS_LABEL_MEAN = "labels/actual_label_mean/class%d"
  ACCURACY = "accuracy"
  ACCURACY_BASELINE = "accuracy/baseline_label_mean"
  ACCURACY_MEAN = "accuracy/threshold_%f_mean"
  PRECISION_MEAN = "precision/positive_threshold_%f_mean"
  RECALL_MEAN = "recall/positive_threshold_%f_mean"
