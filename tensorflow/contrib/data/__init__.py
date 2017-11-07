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
"""`tf.contrib.data.Dataset` API for input pipelines.

See the @{$datasets$Importing Data} Programmer's Guide for an overview.

@@Dataset
@@Iterator
@@TFRecordDataset
@@FixedLengthRecordDataset
@@TextLineDataset

@@batch_and_drop_remainder
@@dense_to_sparse_batch
@@enumerate_dataset
@@group_by_window
@@ignore_errors
@@make_saveable_from_iterator
@@read_batch_features
@@unbatch
@@parallel_interleave
@@rejection_resample
@@sloppy_interleave

@@get_single_element
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import

from tensorflow.contrib.data.python.ops.batching import batch_and_drop_remainder
from tensorflow.contrib.data.python.ops.batching import dense_to_sparse_batch
from tensorflow.contrib.data.python.ops.batching import unbatch
from tensorflow.contrib.data.python.ops.dataset_ops import Dataset
from tensorflow.contrib.data.python.ops.dataset_ops import get_single_element
from tensorflow.contrib.data.python.ops.enumerate_ops import enumerate_dataset
from tensorflow.contrib.data.python.ops.error_ops import ignore_errors
from tensorflow.contrib.data.python.ops.grouping import group_by_window
from tensorflow.contrib.data.python.ops.interleave_ops import parallel_interleave
from tensorflow.contrib.data.python.ops.interleave_ops import sloppy_interleave
from tensorflow.contrib.data.python.ops.iterator_ops import make_saveable_from_iterator
from tensorflow.contrib.data.python.ops.readers import FixedLengthRecordDataset
from tensorflow.contrib.data.python.ops.readers import read_batch_features
from tensorflow.contrib.data.python.ops.readers import SqlDataset
from tensorflow.contrib.data.python.ops.readers import TextLineDataset
from tensorflow.contrib.data.python.ops.readers import TFRecordDataset
from tensorflow.contrib.data.python.ops.resampling import rejection_resample
from tensorflow.python.data.ops.iterator_ops import Iterator
# pylint: enable=unused-import

from tensorflow.python.util.all_util import remove_undocumented
remove_undocumented(__name__)
