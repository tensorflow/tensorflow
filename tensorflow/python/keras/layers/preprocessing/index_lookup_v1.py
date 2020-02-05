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
"""Tensorflow V1 version of the text vectorization preprocessing layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import base_preprocessing_layer_v1
from tensorflow.python.keras.layers.preprocessing import index_lookup
from tensorflow.python.ops.ragged import ragged_tensor_value


class IndexLookup(index_lookup.IndexLookup,
                  base_preprocessing_layer_v1.CombinerPreprocessingLayer):
  """IndexLookup layer.

  This layer translates a set of arbitray strings or integers into an integer
  output via a table-based lookup, with optional out-of-vocabulary handling.

  If desired, the user can call this layer's adapt() method on a data set.
  When this layer is adapted, it will analyze the dataset, determine the
  frequency of individual string or integer values, and create a vocabulary
  from them. This vocabulary can have unlimited size or be capped, depending on
  the configuration options for this layer; if there are more unique values in
  the input than the maximum vocabulary size, the most frequent terms will be
  used to create the vocabulary.

  Attributes:
    max_vocab_size: The maximum size of the vocabulary for this layer. If None,
      there is no cap on the size of the vocabulary. Note that the vocabulary
      does include OOV buckets, so the effective number of unique values in the
      vocabulary is (max_vocab_size - num_oov_buckets) when this value is set.
    num_oov_buckets: The number of out-of-vocabulary tokens to use; defaults to
      1. If this value is more than 1, OOV inputs are hashed to determine their
      OOV value; if this value is 0, passing an OOV input will result in a
      runtime error.
    reserve_zero: Whether to reserve the index '0', which has a special meaning
      in the Keras masking system. If True, the output of this layer will be in
      the range [1...max_vocab_size+1); if False, the output will be in the
      range [0...max_vocab_size). Defaults to True.
    mask_inputs: If True, input values of 0 (for integers) and "" (for strings)
      will be treated as masked values and assigned an output value of 0. If
      this option is set, reserve_zero must also be set. Defaults to False.
  """

  def _get_table_data(self):
    keys, values = self._table.export()
    np_keys = K.get_session().run(keys)
    np_values = K.get_session().run(values)
    return (np_keys, np_values)

  def vocab_size(self):
    return K.get_session().run(self._table.size())

  def _clear_table(self):
    keys, _ = self._table.export()
    K.get_session().run(self._table.remove(keys))

  def _insert_table_data(self, keys, values):
    K.get_session().run(self._table.insert(keys, values))

  def _to_numpy(self, data):
    """Converts preprocessed inputs into numpy arrays."""
    if isinstance(data, np.ndarray):
      return data
    session = K.get_session()
    data = session.run(data)
    if isinstance(data, ragged_tensor_value.RaggedTensorValue):
      data = np.array(data.to_list())
    return data
