# Copyright 2016 Google Inc. All Rights Reserved.
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

## This package provides Ops for evaluation metrics and summary statistics.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,line-too-long,g-importing-member,wildcard-import
from tensorflow.contrib.metrics.python.metrics import *
from tensorflow.contrib.metrics.python.ops.confusion_matrix_ops import confusion_matrix
from tensorflow.contrib.metrics.python.ops.histogram_ops import auc_using_histogram
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_accuracy
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_auc
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_mean
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_mean_absolute_error
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_mean_cosine_distance
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_mean_relative_error
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_mean_squared_error
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_percentage_less
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_precision
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_recall
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_recall_at_k
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_root_mean_squared_error
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_sparse_precision_at_k
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_sparse_recall_at_k
from tensorflow.contrib.metrics.python.ops.set_ops import set_difference
from tensorflow.contrib.metrics.python.ops.set_ops import set_intersection
from tensorflow.contrib.metrics.python.ops.set_ops import set_size
from tensorflow.contrib.metrics.python.ops.set_ops import set_union

