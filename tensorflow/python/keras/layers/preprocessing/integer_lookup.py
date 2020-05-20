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

from tensorflow.python.framework import dtypes
from tensorflow.python.keras.layers.preprocessing import index_lookup
from tensorflow.python.keras.layers.preprocessing import table_utils


class IntegerLookup(index_lookup.IndexLookup):
  """Maps integers from a vocabulary to integer indices.

  This layer translates a set of arbitrary integers into an integer output via a
  table-based lookup, with optional out-of-vocabulary handling.

  If desired, the user can call this layer's `adapt()` method on a data set,
  which will analyze the data set, determine the frequency of individual string
  values, and create a vocabulary from them. This vocabulary can have
  unlimited size or be capped, depending on the configuration options for this
  layer; if there are more unique values in the input than the maximum
  vocabulary size, the most frequent terms will be used to create the
  vocabulary.

  Attributes:
    max_values: The maximum size of the vocabulary for this layer. If None,
      there is no cap on the size of the vocabulary. Note that this vocabulary
      includes the OOV and mask tokens, so the effective number of tokens is
      (max_tokens - num_oov_tokens - (1 if mask_token else 0))
    num_oov_indices: The number of out-of-vocabulary values to use; defaults to
      1. If this value is more than 1, OOV inputs are hashed to determine their
      OOV value; if this value is 0, passing an OOV input will result in a '-1'
      being returned for that value in the output tensor. (Note that, because
      the value is -1 and not 0, this will allow you to effectively drop OOV
      values from categorical encodings.)
    mask_value: A value that represents masked inputs, and which is mapped to
      index 0. Defaults to 0. If set to None, no mask term will be added and the
      OOV tokens, if any, will be indexed from (0...num_oov_tokens) instead of
      (1...num_oov_tokens+1).
    oov_value: The value representing an out-of-vocabulary value. Defaults to
      -1.
    vocabulary: An optional list of values, or a path to a text file containing
      a vocabulary to load into this layer. The file should contain one value
      per line. If the list or file contains the same token multiple times, an
      error will be thrown.
    invert: If true, this layer will map indices to vocabulary items instead
      of mapping vocabulary items to indices.
  """

  def __init__(self,
               max_values=None,
               num_oov_indices=1,
               mask_value=0,
               oov_value=-1,
               vocabulary=None,
               invert=False,
               **kwargs):
    allowed_dtypes = [dtypes.int64]

    if "dtype" in kwargs and kwargs["dtype"] not in allowed_dtypes:
      raise ValueError("IntegerLookup may only have a dtype in %s." %
                       allowed_dtypes)

    if "dtype" not in kwargs:
      kwargs["dtype"] = dtypes.int64

    # If max_values is set, the value must be greater than 1 - otherwise we
    # are creating a 0-element vocab, which doesn't make sense.
    if max_values is not None and max_values <= 1:
      raise ValueError("If set, max_values must be greater than 1.")

    if num_oov_indices < 0:
      raise ValueError("num_oov_indices must be greater than 0. You passed %s" %
                       num_oov_indices)

    if vocabulary is not None:
      if isinstance(vocabulary, str):
        vocabulary = table_utils.get_vocabulary_from_file(vocabulary)
        vocabulary = [int(v) for v in vocabulary]

    super(IntegerLookup, self).__init__(
        max_tokens=max_values,
        num_oov_indices=num_oov_indices,
        mask_token=mask_value,
        oov_token=oov_value,
        vocabulary=vocabulary,
        invert=invert,
        **kwargs)

  def get_config(self):
    base_config = super(IntegerLookup, self).get_config()
    # Because the super config has a bunch of args we're also passing,
    # we need to rename and remove them from the config dict.
    base_config["max_values"] = base_config["max_tokens"]
    del base_config["max_tokens"]

    base_config["mask_value"] = base_config["mask_token"]
    del base_config["mask_token"]

    base_config["oov_value"] = base_config["oov_token"]
    del base_config["oov_token"]
    return base_config
