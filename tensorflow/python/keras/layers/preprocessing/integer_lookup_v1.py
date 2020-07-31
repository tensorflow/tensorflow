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
from tensorflow.python.keras.layers.preprocessing import integer_lookup
from tensorflow.python.util.tf_export import keras_export


@keras_export(v1=["keras.layers.experimental.preprocessing.IntegerLookup"])
class IntegerLookup(integer_lookup.IntegerLookup, index_lookup_v1.IndexLookup):
  """Maps integers from a vocabulary to integer indices."""

  def __init__(self,
               max_values=None,
               num_oov_indices=1,
               mask_value=0,
               oov_value=-1,
               vocabulary=None,
               invert=False,
               **kwargs):
    super(IntegerLookup, self).__init__(max_values, num_oov_indices, mask_value,
                                        oov_value, vocabulary, invert, **kwargs)
    base_preprocessing_layer._kpl_gauge.get_cell("V1").set("IntegerLookup")
