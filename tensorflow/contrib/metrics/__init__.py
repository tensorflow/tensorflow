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
"""Ops for evaluation metrics and summary statistics.

See the @{$python/contrib.metrics} guide.

@@auc_with_confidence_intervals
@@streaming_accuracy
@@streaming_mean
@@streaming_recall
@@streaming_recall_at_thresholds
@@streaming_precision
@@streaming_precision_at_thresholds
@@streaming_false_positive_rate
@@streaming_false_positive_rate_at_thresholds
@@streaming_false_negative_rate
@@streaming_false_negative_rate_at_thresholds
@@streaming_auc
@@streaming_dynamic_auc
@@streaming_curve_points
@@streaming_recall_at_k
@@streaming_mean_absolute_error
@@streaming_mean_iou
@@streaming_mean_relative_error
@@streaming_mean_squared_error
@@streaming_mean_tensor
@@streaming_root_mean_squared_error
@@streaming_covariance
@@streaming_pearson_correlation
@@streaming_mean_cosine_distance
@@streaming_percentage_less
@@streaming_sensitivity_at_specificity
@@streaming_sparse_average_precision_at_k
@@streaming_sparse_average_precision_at_top_k
@@streaming_sparse_precision_at_k
@@streaming_sparse_precision_at_top_k
@@streaming_sparse_recall_at_k
@@streaming_specificity_at_sensitivity
@@streaming_concat
@@streaming_false_negatives
@@streaming_false_negatives_at_thresholds
@@streaming_false_positives
@@streaming_false_positives_at_thresholds
@@streaming_true_negatives
@@streaming_true_negatives_at_thresholds
@@streaming_true_positives
@@streaming_true_positives_at_thresholds
@@sparse_recall_at_top_k
@@auc_using_histogram
@@accuracy
@@aggregate_metrics
@@aggregate_metric_map
@@confusion_matrix
@@f1_score
@@set_difference
@@set_intersection
@@set_size
@@set_union
@@cohen_kappa
@@count
@@precision_recall_at_equal_thresholds
@@recall_at_precision
@@precision_at_recall

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,line-too-long,g-importing-member,wildcard-import
from tensorflow.contrib.metrics.python.metrics import *
# pylint: enable=wildcard-import
from tensorflow.contrib.metrics.python.ops.confusion_matrix_ops import confusion_matrix
from tensorflow.contrib.metrics.python.ops.histogram_ops import auc_using_histogram
from tensorflow.contrib.metrics.python.ops.metric_ops import aggregate_metric_map
from tensorflow.contrib.metrics.python.ops.metric_ops import aggregate_metrics
from tensorflow.contrib.metrics.python.ops.metric_ops import auc_with_confidence_intervals
from tensorflow.contrib.metrics.python.ops.metric_ops import cohen_kappa
from tensorflow.contrib.metrics.python.ops.metric_ops import count
from tensorflow.contrib.metrics.python.ops.metric_ops import precision_at_recall
from tensorflow.contrib.metrics.python.ops.metric_ops import precision_recall_at_equal_thresholds
from tensorflow.contrib.metrics.python.ops.metric_ops import recall_at_precision
from tensorflow.contrib.metrics.python.ops.metric_ops import sparse_recall_at_top_k
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_accuracy
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_auc
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_concat
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_covariance
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_curve_points
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_dynamic_auc
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_false_negative_rate
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_false_negative_rate_at_thresholds
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_false_negatives
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_false_negatives_at_thresholds
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_false_positive_rate
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_false_positive_rate_at_thresholds
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_false_positives
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_false_positives_at_thresholds
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_mean
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_mean_absolute_error
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_mean_cosine_distance
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_mean_iou
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_mean_relative_error
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_mean_squared_error
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_mean_tensor
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_pearson_correlation
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_percentage_less
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_precision
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_precision_at_thresholds
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_recall
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_recall_at_k
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_recall_at_thresholds
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_root_mean_squared_error
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_sensitivity_at_specificity
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_sparse_average_precision_at_k
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_sparse_average_precision_at_top_k
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_sparse_precision_at_k
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_sparse_precision_at_top_k
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_sparse_recall_at_k
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_specificity_at_sensitivity
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_true_negatives
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_true_negatives_at_thresholds
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_true_positives
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_true_positives_at_thresholds
from tensorflow.contrib.metrics.python.ops.set_ops import set_difference
from tensorflow.contrib.metrics.python.ops.set_ops import set_intersection
from tensorflow.contrib.metrics.python.ops.set_ops import set_size
from tensorflow.contrib.metrics.python.ops.set_ops import set_union
# pylint: enable=unused-import,line-too-long

from tensorflow.python.util.all_util import remove_undocumented
remove_undocumented(__name__)
