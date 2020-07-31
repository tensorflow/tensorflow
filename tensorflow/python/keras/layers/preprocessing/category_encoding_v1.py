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
"""Tensorflow V1 version of the text category_encoding preprocessing layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.engine import base_preprocessing_layer_v1
from tensorflow.python.keras.layers.preprocessing import category_encoding
from tensorflow.python.util.tf_export import keras_export


@keras_export(v1=["keras.layers.experimental.preprocessing.CategoryEncoding"])
class CategoryEncoding(category_encoding.CategoryEncoding,
                       base_preprocessing_layer_v1.CombinerPreprocessingLayer):
  """CategoryEncoding layer.

  This layer provides options for condensing input data into denser
  representations. It accepts either integer values or strings as inputs,
  allows users to map those inputs into a contiguous integer space, and
  outputs either those integer values (one sample = 1D tensor of integer token
  indices) or a dense representation (one sample = 1D tensor of float values
  representing data about the sample's tokens).

  If desired, the user can call this layer's adapt() method on a dataset.
  When this layer is adapted, it will analyze the dataset, determine the
  frequency of individual integer or string values, and create a 'vocabulary'
  from them. This vocabulary can have unlimited size or be capped, depending
  on the configuration options for this layer; if there are more unique
  values in the input than the maximum vocabulary size, the most frequent
  terms will be used to create the vocabulary.

  Attributes:
    max_elements: The maximum size of the vocabulary for this layer. If None,
      there is no cap on the size of the vocabulary.
    output_mode: Optional specification for the output of the layer. Values can
      be "int", "binary", "count" or "tf-idf", configuring the layer as follows:
        "int": Outputs integer indices, one integer index per split string
          token.
        "binary": Outputs a single int array per batch, of either vocab_size or
          max_elements size, containing 1s in all elements where the token
          mapped to that index exists at least once in the batch item.
        "count": As "binary", but the int array contains a count of the number
          of times the token at that index appeared in the batch item.
        "tf-idf": As "binary", but the TF-IDF algorithm is applied to find the
          value in each token slot.
    output_sequence_length: Only valid in INT mode. If set, the output will have
      its time dimension padded or truncated to exactly `output_sequence_length`
      values, resulting in a tensor of shape [batch_size,
      output_sequence_length] regardless of the input shape.
    pad_to_max_elements: Only valid in  "binary", "count", and "tf-idf" modes.
      If True, the output will have its feature axis padded to `max_elements`
      even if the number of unique values in the vocabulary is less than
      max_elements, resulting in a tensor of shape [batch_size, max_elements]
      regardless of vocabulary size. Defaults to False.
  """
