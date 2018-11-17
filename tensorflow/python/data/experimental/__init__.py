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
"""Experimental API for building input pipelines.

This module contains experimental `Dataset` sources and transformations that can
be used in conjunction with the `tf.data.Dataset` API. Note that the
`tf.data.experimental` API is not subject to the same backwards compatibility
guarantees as `tf.data`, but we will provide deprecation advice in advance of
removing existing functionality.

See [Importing Data](https://tensorflow.org/guide/datasets) for an overview.

@@Counter
@@CheckpointInputPipelineHook
@@CsvDataset
@@Optional
@@RandomDataset
@@Reducer
@@SqlDataset
@@StatsAggregator
@@StatsOptions
@@TFRecordWriter

@@bucket_by_sequence_length
@@choose_from_datasets
@@copy_to_device
@@dense_to_sparse_batch
@@enumerate_dataset
@@get_next_as_optional
@@get_single_element
@@group_by_reducer
@@group_by_window
@@ignore_errors
@@latency_stats
@@make_batched_features_dataset
@@make_csv_dataset
@@make_saveable_from_iterator
@@map_and_batch
@@parallel_interleave
@@parse_example_dataset
@@prefetch_to_device
@@rejection_resample
@@sample_from_datasets
@@scan
@@shuffle_and_repeat
@@unbatch
@@unique

@@AUTOTUNE
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import

from tensorflow.python.data.experimental.ops.batching import dense_to_sparse_batch
from tensorflow.python.data.experimental.ops.batching import map_and_batch
from tensorflow.python.data.experimental.ops.batching import unbatch
from tensorflow.python.data.experimental.ops.counter import Counter
from tensorflow.python.data.experimental.ops.enumerate_ops import enumerate_dataset
from tensorflow.python.data.experimental.ops.error_ops import ignore_errors
from tensorflow.python.data.experimental.ops.get_single_element import get_single_element
from tensorflow.python.data.experimental.ops.grouping import bucket_by_sequence_length
from tensorflow.python.data.experimental.ops.grouping import group_by_reducer
from tensorflow.python.data.experimental.ops.grouping import group_by_window
from tensorflow.python.data.experimental.ops.grouping import Reducer
from tensorflow.python.data.experimental.ops.interleave_ops import choose_from_datasets
from tensorflow.python.data.experimental.ops.interleave_ops import parallel_interleave
from tensorflow.python.data.experimental.ops.interleave_ops import sample_from_datasets
from tensorflow.python.data.experimental.ops.iterator_ops import CheckpointInputPipelineHook
from tensorflow.python.data.experimental.ops.iterator_ops import make_saveable_from_iterator

# Optimization constant that can be used to enable auto-tuning.
from tensorflow.python.data.experimental.ops.optimization import AUTOTUNE

from tensorflow.python.data.experimental.ops.parsing_ops import parse_example_dataset
from tensorflow.python.data.experimental.ops.prefetching_ops import copy_to_device
from tensorflow.python.data.experimental.ops.prefetching_ops import prefetch_to_device
from tensorflow.python.data.experimental.ops.random_ops import RandomDataset
from tensorflow.python.data.experimental.ops.readers import CsvDataset
from tensorflow.python.data.experimental.ops.readers import make_batched_features_dataset
from tensorflow.python.data.experimental.ops.readers import make_csv_dataset
from tensorflow.python.data.experimental.ops.readers import SqlDataset
from tensorflow.python.data.experimental.ops.resampling import rejection_resample
from tensorflow.python.data.experimental.ops.scan_ops import scan
from tensorflow.python.data.experimental.ops.shuffle_ops import shuffle_and_repeat
from tensorflow.python.data.experimental.ops.stats_aggregator import StatsAggregator
from tensorflow.python.data.experimental.ops.stats_ops import latency_stats
from tensorflow.python.data.experimental.ops.stats_options import StatsOptions
from tensorflow.python.data.experimental.ops.unique import unique
from tensorflow.python.data.experimental.ops.writers import TFRecordWriter
from tensorflow.python.data.ops.iterator_ops import get_next_as_optional
from tensorflow.python.data.ops.optional_ops import Optional
# pylint: enable=unused-import

from tensorflow.python.util.all_util import remove_undocumented
remove_undocumented(__name__)
