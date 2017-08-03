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

"""DataFrames for ingesting and preprocessing data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.learn.python.learn.dataframe.dataframe import DataFrame
from tensorflow.contrib.learn.python.learn.dataframe.series import PredefinedSeries
from tensorflow.contrib.learn.python.learn.dataframe.series import Series
from tensorflow.contrib.learn.python.learn.dataframe.series import TransformedSeries
from tensorflow.contrib.learn.python.learn.dataframe.tensorflow_dataframe import TensorFlowDataFrame
from tensorflow.contrib.learn.python.learn.dataframe.transform import parameter
from tensorflow.contrib.learn.python.learn.dataframe.transform import TensorFlowTransform
from tensorflow.contrib.learn.python.learn.dataframe.transform import Transform

# Transforms
from tensorflow.contrib.learn.python.learn.dataframe.transforms.boolean_mask import BooleanMask
from tensorflow.contrib.learn.python.learn.dataframe.transforms.difference import Difference
from tensorflow.contrib.learn.python.learn.dataframe.transforms.hashes import HashFast
from tensorflow.contrib.learn.python.learn.dataframe.transforms.in_memory_source import NumpySource
from tensorflow.contrib.learn.python.learn.dataframe.transforms.in_memory_source import PandasSource
from tensorflow.contrib.learn.python.learn.dataframe.transforms.reader_source import ReaderSource
# Coming soon; multichange client hassle due to no DIFFBASE in Cider
# from tensorflow.contrib.learn.python.learn.dataframe \
#     .transforms.split_mask import SplitMask
from tensorflow.contrib.learn.python.learn.dataframe.transforms.sum import Sum

# pylint: disable=g-bad-import-order
from tensorflow.contrib.learn.python.learn.dataframe.transforms import unary_transforms as _ut
from tensorflow.contrib.learn.python.learn.dataframe.transforms import binary_transforms as _bt

from tensorflow.python.util import deprecation


# Suppress deprecation warnings in these registrations.
with deprecation.silence():
  # Unary Transform registration
  for ut_def in _ut.UNARY_TRANSFORMS:
    _ut.register_unary_op(*ut_def)

  # Comparison Transform registration
  for bt_def in _bt.BINARY_TRANSFORMS:
    _bt.register_binary_op(*bt_def)


__all__ = ['DataFrame', 'Series', 'PredefinedSeries', 'TransformedSeries',
           'TensorFlowDataFrame', 'TensorFlowTransform', 'parameter',
           'Transform']
