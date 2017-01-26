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
"""Tools to allow different io formats."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.learn.python.learn.learn_io.dask_io import extract_dask_data
from tensorflow.contrib.learn.python.learn.learn_io.dask_io import extract_dask_labels
from tensorflow.contrib.learn.python.learn.learn_io.dask_io import HAS_DASK
from tensorflow.contrib.learn.python.learn.learn_io.graph_io import _read_keyed_batch_examples_shared_queue
from tensorflow.contrib.learn.python.learn.learn_io.graph_io import _read_keyed_batch_features_shared_queue
from tensorflow.contrib.learn.python.learn.learn_io.graph_io import queue_parsed_features
from tensorflow.contrib.learn.python.learn.learn_io.graph_io import read_batch_examples
from tensorflow.contrib.learn.python.learn.learn_io.graph_io import read_batch_features
from tensorflow.contrib.learn.python.learn.learn_io.graph_io import read_batch_record_features
from tensorflow.contrib.learn.python.learn.learn_io.graph_io import read_keyed_batch_examples
from tensorflow.contrib.learn.python.learn.learn_io.graph_io import read_keyed_batch_features
from tensorflow.contrib.learn.python.learn.learn_io.numpy_io import numpy_input_fn
from tensorflow.contrib.learn.python.learn.learn_io.pandas_io import extract_pandas_data
from tensorflow.contrib.learn.python.learn.learn_io.pandas_io import extract_pandas_labels
from tensorflow.contrib.learn.python.learn.learn_io.pandas_io import extract_pandas_matrix
from tensorflow.contrib.learn.python.learn.learn_io.pandas_io import HAS_PANDAS
from tensorflow.contrib.learn.python.learn.learn_io.pandas_io import pandas_input_fn
from tensorflow.contrib.learn.python.learn.learn_io.generator_io import generator_input_fn