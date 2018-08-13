# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for preprocessing sequence data.
"""
# pylint: disable=invalid-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras_preprocessing import sequence

from tensorflow.python.util.tf_export import tf_export

pad_sequences = sequence.pad_sequences
make_sampling_table = sequence.make_sampling_table
skipgrams = sequence.skipgrams
# TODO(fchollet): consider making `_remove_long_seq` public.
_remove_long_seq = sequence._remove_long_seq  # pylint: disable=protected-access
TimeseriesGenerator = sequence.TimeseriesGenerator

tf_export('keras.preprocessing.sequence.pad_sequences')(pad_sequences)
tf_export(
    'keras.preprocessing.sequence.make_sampling_table')(make_sampling_table)
tf_export('keras.preprocessing.sequence.skipgrams')(skipgrams)
tf_export(
    'keras.preprocessing.sequence.TimeseriesGenerator')(TimeseriesGenerator)
