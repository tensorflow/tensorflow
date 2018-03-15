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

"""Evaluation-related metrics.

@@accuracy
@@auc
@@false_negatives
@@false_negatives_at_thresholds
@@false_positives
@@false_positives_at_thresholds
@@mean
@@mean_absolute_error
@@mean_cosine_distance
@@mean_iou
@@mean_per_class_accuracy
@@mean_relative_error
@@mean_squared_error
@@mean_tensor
@@percentage_below
@@precision
@@precision_at_thresholds
@@recall
@@recall_at_k
@@recall_at_top_k
@@recall_at_thresholds
@@root_mean_squared_error
@@sensitivity_at_specificity
@@sparse_average_precision_at_k
@@average_precision_at_k
@@sparse_precision_at_k
@@precision_at_k
@@precision_at_top_k
@@specificity_at_sensitivity
@@true_negatives
@@true_negatives_at_thresholds
@@true_positives
@@true_positives_at_thresholds

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.metrics_impl import *
# pylint: enable=wildcard-import

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = []
remove_undocumented(__name__, _allowed_symbols)
