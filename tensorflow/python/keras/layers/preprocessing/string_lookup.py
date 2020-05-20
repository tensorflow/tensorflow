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


class StringLookup(index_lookup.IndexLookup):
  """Maps strings from a vocabulary to integer indices.

  This layer translates a set of arbitrary strings into an integer output via a
  table-based lookup, with optional out-of-vocabulary handling.

  If desired, the user can call this layer's `adapt()` method on a data set,
  which will analyze the data set, determine the frequency of individual string
  values, and create a vocabulary from them. This vocabulary can have
  unlimited size or be capped, depending on the configuration options for this
  layer; if there are more unique values in the input than the maximum
  vocabulary size, the most frequent terms will be used to create the
  vocabulary.

  Attributes:
    max_tokens: The maximum size of the vocabulary for this layer. If None,
      there is no cap on the size of the vocabulary. Note that this vocabulary
      includes the OOV and mask tokens, so the effective number of tokens is
      (max_tokens - num_oov_indices - (1 if mask_token else 0))
    num_oov_indices: The number of out-of-vocabulary tokens to use; defaults to
      1. If this value is more than 1, OOV inputs are hashed to determine their
      OOV value; if this value is 0, passing an OOV input will result in a '-1'
      being returned for that value in the output tensor. (Note that, because
      the value is -1 and not 0, this will allow you to effectively drop OOV
      values from categorical encodings.)
    mask_token: A token that represents masked values, and which is mapped to
      index 0. Defaults to the empty string "". If set to None, no mask term
      will be added and the OOV tokens, if any, will be indexed from
      (0...num_oov_indices) instead of (1...num_oov_indices+1).
    oov_token: The token representing an out-of-vocabulary value. Defaults to
      "[OOV]".
    vocabulary: An optional list of vocabulary terms, or a path to a text file
      containing a vocabulary to load into this layer. The file should contain
      one token per line. If the list or file contains the same token multiple
      times, an error will be thrown.
    encoding: The Python string encoding to use. Defaults to `'utf-8'`.
    invert: If true, this layer will map indices to vocabulary items instead
      of mapping vocabulary items to indices.
  """

  def __init__(self,
               max_tokens=None,
               num_oov_indices=1,
               mask_token="",
               oov_token="[OOV]",
               vocabulary=None,
               encoding="utf-8",
               invert=False,
               **kwargs):
    allowed_dtypes = [dtypes.string]

    if "dtype" in kwargs and kwargs["dtype"] not in allowed_dtypes:
      raise ValueError("StringLookup may only have a dtype in %s." %
                       allowed_dtypes)

    if "dtype" not in kwargs:
      kwargs["dtype"] = dtypes.string

    if vocabulary is not None:
      if isinstance(vocabulary, str):
        vocabulary = table_utils.get_vocabulary_from_file(vocabulary, encoding)

    self.encoding = encoding

    super(StringLookup, self).__init__(
        max_tokens=max_tokens,
        num_oov_indices=num_oov_indices,
        mask_token=mask_token,
        oov_token=oov_token,
        vocabulary=vocabulary,
        invert=invert,
        **kwargs)

  def get_config(self):
    config = {"encoding": self.encoding}
    base_config = super(StringLookup, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_vocabulary(self):
    if self._table_handler.vocab_size() == 0:
      return []

    keys, values = self._table_handler.data()
    # This is required because the MutableHashTable doesn't preserve insertion
    # order, but we rely on the order of the array to assign indices.
    return [x.decode(self.encoding) for _, x in sorted(zip(values, keys))]
