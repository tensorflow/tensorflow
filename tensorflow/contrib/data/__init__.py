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
be used in conjunction with the @{tf.data.Dataset} API. Note that the
`tf.contrib.data` API is not subject to the same backwards compatibility
guarantees as `tf.data`, but we will provide deprecation advice in advance of
removing existing functionality.

See @{$guide/datasets$Importing Data} for an overview.

@@Counter
@@CheckpointInputPipelineHook
@@CsvDataset
@@RandomDataset
@@Reducer
@@SqlDataset
@@TFRecordWriter

@@assert_element_shape
@@batch_and_drop_remainder
@@bucket_by_sequence_length
@@choose_from_datasets
@@copy_to_device
@@dense_to_sparse_batch
@@enumerate_dataset

@@get_single_element
@@group_by_reducer
@@group_by_window
@@ignore_errors
@@make_batched_features_dataset
@@make_csv_dataset
@@make_saveable_from_iterator

@@map_and_batch
@@padded_batch_and_drop_remainder
@@parallel_interleave
@@prefetch_to_device
@@read_batch_features
@@rejection_resample
@@reduce_dataset
@@sample_from_datasets
@@scan
@@shuffle_and_repeat
@@sliding_window_batch
@@sloppy_interleave
@@unbatch
@@unique
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import

from tensorflow.contrib.data.python.ops.batching import assert_element_shape
from tensorflow.contrib.data.python.ops.batching import batch_and_drop_remainder
from tensorflow.contrib.data.python.ops.batching import dense_to_sparse_batch
from tensorflow.contrib.data.python.ops.batching import map_and_batch
from tensorflow.contrib.data.python.ops.batching import padded_batch_and_drop_remainder
from tensorflow.contrib.data.python.ops.batching import unbatch
from tensorflow.contrib.data.python.ops.counter import Counter
from tensorflow.contrib.data.python.ops.enumerate_ops import enumerate_dataset
from tensorflow.contrib.data.python.ops.error_ops import ignore_errors
from tensorflow.contrib.data.python.ops.get_single_element import get_single_element
from tensorflow.contrib.data.python.ops.get_single_element import reduce_dataset
from tensorflow.contrib.data.python.ops.grouping import bucket_by_sequence_length
from tensorflow.contrib.data.python.ops.grouping import group_by_reducer
from tensorflow.contrib.data.python.ops.grouping import group_by_window
from tensorflow.contrib.data.python.ops.grouping import Reducer
from tensorflow.contrib.data.python.ops.interleave_ops import choose_from_datasets
from tensorflow.contrib.data.python.ops.interleave_ops import parallel_interleave
from tensorflow.contrib.data.python.ops.interleave_ops import sample_from_datasets
from tensorflow.contrib.data.python.ops.interleave_ops import sloppy_interleave
from tensorflow.contrib.data.python.ops.iterator_ops import CheckpointInputPipelineHook
from tensorflow.contrib.data.python.ops.iterator_ops import make_saveable_from_iterator
from tensorflow.contrib.data.python.ops.prefetching_ops import copy_to_device
from tensorflow.contrib.data.python.ops.prefetching_ops import prefetch_to_device
from tensorflow.contrib.data.python.ops.random_ops import RandomDataset
from tensorflow.contrib.data.python.ops.readers import CsvDataset
from tensorflow.contrib.data.python.ops.readers import make_batched_features_dataset
from tensorflow.contrib.data.python.ops.readers import make_csv_dataset
from tensorflow.contrib.data.python.ops.readers import read_batch_features
from tensorflow.contrib.data.python.ops.readers import SqlDataset
from tensorflow.contrib.data.python.ops.resampling import rejection_resample
from tensorflow.contrib.data.python.ops.scan_ops import scan
from tensorflow.contrib.data.python.ops.shuffle_ops import shuffle_and_repeat
from tensorflow.contrib.data.python.ops.sliding import sliding_window_batch
from tensorflow.contrib.data.python.ops.unique import unique
from tensorflow.contrib.data.python.ops.writers import TFRecordWriter
# pylint: enable=unused-import

from tensorflow.python.util.all_util import remove_undocumented
remove_undocumented(__name__)

# A constant that can be used to enable auto-tuning.
AUTOTUNE = -1
