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
"""`tf.data.Dataset` API for input pipelines.

See [Importing Data](https://tensorflow.org/guide/data) for an overview.
"""

# pylint: disable=unused-import
from tensorflow.python.data import experimental
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.data.ops.dataset_ops import INFINITE as INFINITE_CARDINALITY
from tensorflow.python.data.ops.dataset_ops import make_initializable_iterator
from tensorflow.python.data.ops.dataset_ops import make_one_shot_iterator
from tensorflow.python.data.ops.dataset_ops import UNKNOWN as UNKNOWN_CARDINALITY
from tensorflow.python.data.ops.iterator_ops import Iterator
from tensorflow.python.data.ops.options import Options
from tensorflow.python.data.ops.readers import FixedLengthRecordDataset
from tensorflow.python.data.ops.readers import TextLineDataset
from tensorflow.python.data.ops.readers import TFRecordDataset
# pylint: enable=unused-import

# Additional tf.data readers
from tensorflow.python.data.ops.readers import CsvDataset
from tensorflow.python.data.ops.readers import SqlDataset
# Experimental high-level helpers
from tensorflow.python.data.experimental.ops.readers import make_csv_dataset
from tensorflow.python.data.experimental.ops.readers import make_batched_features_dataset
# Experimental transformations
from tensorflow.python.data.experimental.ops.batching import dense_to_ragged_batch
from tensorflow.python.data.experimental.ops.prefetching_ops import prefetch_to_device
from tensorflow.python.data.experimental.ops.interleave_ops import parallel_interleave
# Optimization / threading options
from tensorflow.python.data.experimental.ops.optimization_options import OptimizationOptions
from tensorflow.python.data.experimental.ops.threading_options import ThreadingOptions
