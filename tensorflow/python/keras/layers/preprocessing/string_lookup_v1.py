# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Keras string lookup preprocessing layer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.engine import base_preprocessing_layer
from tensorflow.python.keras.layers.preprocessing import index_lookup_v1
from tensorflow.python.keras.layers.preprocessing import string_lookup
from tensorflow.python.util.tf_export import keras_export


@keras_export(v1=["keras.layers.experimental.preprocessing.StringLookup"])
class StringLookup(string_lookup.StringLookup, index_lookup_v1.IndexLookup):
  """Maps strings from a vocabulary to integer indices."""

  def __init__(self,
               max_tokens=None,
               num_oov_indices=1,
               mask_token="",
               oov_token="[UNK]",
               vocabulary=None,
               encoding=None,
               invert=False,
               **kwargs):
    super(StringLookup, self).__init__(
        max_tokens=max_tokens,
        num_oov_indices=num_oov_indices,
        mask_token=mask_token,
        oov_token=oov_token,
        vocabulary=vocabulary,
        invert=invert,
        **kwargs)
    base_preprocessing_layer.keras_kpl_gauge.get_cell(
        "StringLookup_V1").set(True)
