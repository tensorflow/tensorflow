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
"""FeatureColumns: tools for ingesting and representing features."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,line-too-long,wildcard-import
from tensorflow.python.feature_column.feature_column import *

from tensorflow.python.util.all_util import remove_undocumented
# pylint: enable=unused-import,line-too-long

_allowed_symbols = [
    'input_layer',
    'linear_model',
    'make_parse_example_spec',
    'embedding_column',
    'shared_embedding_columns',
    'crossed_column',
    'numeric_column',
    'bucketized_column',
    'categorical_column_with_hash_bucket',
    'categorical_column_with_vocabulary_file',
    'categorical_column_with_vocabulary_list',
    'categorical_column_with_identity',
    'weighted_categorical_column',
    'indicator_column',
]

remove_undocumented(__name__, allowed_exception_list=_allowed_symbols)
