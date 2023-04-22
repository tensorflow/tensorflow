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

# pylint: disable=unused-import,line-too-long,wildcard-import,g-bad-import-order
from tensorflow.python.feature_column.feature_column import *
from tensorflow.python.feature_column.feature_column_v2 import *
from tensorflow.python.feature_column.sequence_feature_column import *
from tensorflow.python.feature_column.serialization import *
# We import dense_features_v2 first so that the V1 DenseFeatures is the default
# if users directly import feature_column_lib.
from tensorflow.python.keras.feature_column.dense_features_v2 import *
from tensorflow.python.keras.feature_column.dense_features import *
from tensorflow.python.keras.feature_column.sequence_feature_column import *
# pylint: enable=unused-import,line-too-long
