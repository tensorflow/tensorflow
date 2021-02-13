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
# pylint: disable=g-classes-have-attributes
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.keras.engine import base_preprocessing_layer
from tensorflow.python.keras.layers.preprocessing import index_lookup
from tensorflow.python.keras.layers.preprocessing import table_utils
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.experimental.preprocessing.StringLookup", v1=[])
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
  vocabulary (and the terms that don't make the cut will be treated as OOV).

  Args:
    max_tokens: The maximum size of the vocabulary for this layer. If None,
      there is no cap on the size of the vocabulary. Note that this vocabulary
      includes the OOV and mask tokens, so the effective number of tokens is
      `(max_tokens - num_oov_indices - (1 if mask_token else 0))`.
    num_oov_indices: The number of out-of-vocabulary tokens to use; defaults to
      1. If this value is more than 1, OOV inputs are hashed to determine their
      OOV value; if this value is 0, passing an OOV input will result in a '-1'
      being returned for that value in the output tensor. (Note that, because
      the value is -1 and not 0, this will allow you to effectively drop OOV
      values from categorical encodings.)
    mask_token: A token that represents masked values, and which is mapped to
      index 0. Defaults to the empty string `""`. If set to None, no mask term
      will be added and the OOV tokens, if any, will be indexed from
      `(0...num_oov_indices)` instead of `(1...num_oov_indices+1)`.
    oov_token: The token representing an out-of-vocabulary value. Defaults to
      `"[UNK]"`.
    vocabulary: An optional list of vocabulary terms, or a path to a text file
      containing a vocabulary to load into this layer. The file should contain
      one token per line. If the list or file contains the same token multiple
      times, an error will be thrown.
    encoding: The Python string encoding to use. Defaults to `"utf-8"`.
    invert: If true, this layer will map indices to vocabulary items instead
      of mapping vocabulary items to indices.
    output_mode: Specification for the output of the layer. Only applicable
      when `invert` is False. Defaults to "int". Values can be "int", "binary",
      or "count", configuring the layer as follows:
        "int": Return the raw integer indices of the input values.
        "binary": Outputs a single int array per batch, of either vocab_size or
          max_tokens size, containing 1s in all elements where the token mapped
          to that index exists at least once in the batch item.
        "count": Like "binary", but the int array contains a count of the number
          of times the token at that index appeared in the batch item.
        "tf-idf": As "binary", but the TF-IDF algorithm is applied to find the
          value in each token slot.
    pad_to_max_tokens: Only valid in  "binary", "count", and "tf-idf" modes. If
      True, the output will have its feature axis padded to `max_tokens` even if
      the number of unique tokens in the vocabulary is less than max_tokens,
      resulting in a tensor of shape [batch_size, max_tokens] regardless of
      vocabulary size. Defaults to True.
    sparse: Boolean. Only applicable to "binary" and "count" output modes.
      If true, returns a `SparseTensor` instead of a dense `Tensor`.
      Defaults to `False`.

  Examples:

  **Creating a lookup layer with a known vocabulary**

  This example creates a lookup layer with a pre-existing vocabulary.

  >>> vocab = ["a", "b", "c", "d"]
  >>> data = tf.constant([["a", "c", "d"], ["d", "z", "b"]])
  >>> layer = StringLookup(vocabulary=vocab)
  >>> layer(data)
  <tf.Tensor: shape=(2, 3), dtype=int64, numpy=
  array([[2, 4, 5],
         [5, 1, 3]])>

  **Creating a lookup layer with an adapted vocabulary**

  This example creates a lookup layer and generates the vocabulary by analyzing
  the dataset.

  >>> data = tf.constant([["a", "c", "d"], ["d", "z", "b"]])
  >>> layer = StringLookup()
  >>> layer.adapt(data)
  >>> layer.get_vocabulary()
  ['', '[UNK]', 'd', 'z', 'c', 'b', 'a']

  Note how the mask token '' and the OOV token [UNK] have been added to the
  vocabulary. The remaining tokens are sorted by frequency ('d', which has
  2 occurrences, is first) then by inverse sort order.

  >>> data = tf.constant([["a", "c", "d"], ["d", "z", "b"]])
  >>> layer = StringLookup()
  >>> layer.adapt(data)
  >>> layer(data)
  <tf.Tensor: shape=(2, 3), dtype=int64, numpy=
  array([[6, 4, 2],
         [2, 3, 5]])>

  **Lookups with multiple OOV indices**

  This example demonstrates how to use a lookup layer with multiple OOV indices.
  When a layer is created with more than one OOV index, any OOV values are
  hashed into the number of OOV buckets, distributing OOV values in a
  deterministic fashion across the set.

  >>> vocab = ["a", "b", "c", "d"]
  >>> data = tf.constant([["a", "c", "d"], ["m", "z", "b"]])
  >>> layer = StringLookup(vocabulary=vocab, num_oov_indices=2)
  >>> layer(data)
  <tf.Tensor: shape=(2, 3), dtype=int64, numpy=
  array([[3, 5, 6],
         [1, 2, 4]])>

  Note that the output for OOV value 'm' is 1, while the output for OOV value
  'z' is 2. The in-vocab terms have their output index increased by 1 from
  earlier examples (a maps to 3, etc) in order to make space for the extra OOV
  value.

  **Multi-hot output**

  Configure the layer with `output_mode='binary'`. Note that the first two
  dimensions in the binary encoding represent the mask token and the OOV token,
  respectively.

  >>> vocab = ["a", "b", "c", "d"]
  >>> data = tf.constant([["a", "c", "d", "d"], ["d", "z", "b", "z"]])
  >>> layer = StringLookup(vocabulary=vocab, output_mode='binary')
  >>> layer(data)
  <tf.Tensor: shape=(2, 6), dtype=float32, numpy=
    array([[0., 0., 1., 0., 1., 1.],
           [0., 1., 0., 1., 0., 1.]], dtype=float32)>

  **Token count output**

  Configure the layer with `output_mode='count'`. As with binary output, the
  first two dimensions in the output represent the mask token and the OOV token,
  respectively.

  >>> vocab = ["a", "b", "c", "d"]
  >>> data = tf.constant([["a", "c", "d", "d"], ["d", "z", "b", "z"]])
  >>> layer = StringLookup(vocabulary=vocab, output_mode='count')
  >>> layer(data)
  <tf.Tensor: shape=(2, 6), dtype=float32, numpy=
    array([[0., 0., 1., 0., 1., 2.],
           [0., 2., 0., 1., 0., 1.]], dtype=float32)>

  **TF-IDF output**

  Configure the layer with `output_mode='tf-idf'`. As with binary output, the
  first two dimensions in the output represent the mask token and the OOV token,
  respectively.

  Each token bin will output `token_count * idf_weight`, where the idf weights
  are the inverse document frequency weights per token. These should be provided
  along with the vocabulary. Note that the `idf_weight` for mask tokens and OOV
  tokens will default to the average of all idf weights passed in.

  >>> vocab = ["a", "b", "c", "d"]
  >>> idf_weights = [0.25, 0.75, 0.6, 0.4]
  >>> data = tf.constant([["a", "c", "d", "d"], ["d", "z", "b", "z"]])
  >>> layer = StringLookup(output_mode='tf-idf')
  >>> layer.set_vocabulary(vocab, idf_weights=idf_weights)
  >>> layer(data)
  <tf.Tensor: shape=(2, 6), dtype=float32, numpy=
    array([[0.  , 0.  , 0.25, 0.  , 0.6 , 0.8 ],
           [0.  , 1.0 , 0.  , 0.75, 0.  , 0.4 ]], dtype=float32)>

  To specify the idf weights for mask and oov values, you will need to pass the
  entire vocabularly including these values.

  >>> vocab = ["", "[UNK]", "a", "b", "c", "d"]
  >>> idf_weights = [0.0, 0.9, 0.25, 0.75, 0.6, 0.4]
  >>> data = tf.constant([["a", "c", "d", "d"], ["d", "z", "b", "z"]])
  >>> layer = StringLookup(output_mode='tf-idf')
  >>> layer.set_vocabulary(vocab, idf_weights=idf_weights)
  >>> layer(data)
  <tf.Tensor: shape=(2, 6), dtype=float32, numpy=
    array([[0.  , 0.  , 0.25, 0.  , 0.6 , 0.8 ],
           [0.  , 1.8 , 0.  , 0.75, 0.  , 0.4 ]], dtype=float32)>

  When adapting the layer in tf-idf mode, each input sample will be considered a
  document, and idf weight per token will be calculated as
  `log(1 + num_documents / (1 + token_document_count))`.

  **Inverse lookup**

  This example demonstrates how to map indices to strings using this layer. (You
  can also use adapt() with inverse=True, but for simplicity we'll pass the
  vocab in this example.)

  >>> vocab = ["a", "b", "c", "d"]
  >>> data = tf.constant([[1, 3, 4], [4, 5, 2]])
  >>> layer = StringLookup(vocabulary=vocab, invert=True)
  >>> layer(data)
  <tf.Tensor: shape=(2, 3), dtype=string, numpy=
  array([[b'a', b'c', b'd'],
         [b'd', b'[UNK]', b'b']], dtype=object)>

  Note that the integer 5, which is out of the vocabulary space, returns an OOV
  token.


  **Forward and inverse lookup pairs**

  This example demonstrates how to use the vocabulary of a standard lookup
  layer to create an inverse lookup layer.

  >>> vocab = ["a", "b", "c", "d"]
  >>> data = tf.constant([["a", "c", "d"], ["d", "z", "b"]])
  >>> layer = StringLookup(vocabulary=vocab)
  >>> i_layer = StringLookup(vocabulary=layer.get_vocabulary(), invert=True)
  >>> int_data = layer(data)
  >>> i_layer(int_data)
  <tf.Tensor: shape=(2, 3), dtype=string, numpy=
  array([[b'a', b'c', b'd'],
         [b'd', b'[UNK]', b'b']], dtype=object)>

  In this example, the input value 'z' resulted in an output of '[UNK]', since
  1000 was not in the vocabulary - it got represented as an OOV, and all OOV
  values are returned as '[OOV}' in the inverse layer. Also, note that for the
  inverse to work, you must have already set the forward layer vocabulary
  either directly or via fit() before calling get_vocabulary().
  """

  def __init__(self,
               max_tokens=None,
               num_oov_indices=1,
               mask_token="",
               oov_token="[UNK]",
               vocabulary=None,
               encoding=None,
               invert=False,
               output_mode=index_lookup.INT,
               sparse=False,
               pad_to_max_tokens=False,
               **kwargs):
    allowed_dtypes = [dtypes.string]

    if "dtype" in kwargs and kwargs["dtype"] not in allowed_dtypes:
      raise ValueError("The value of the dtype argument for StringLookup may "
                       "only be one of %s." % (allowed_dtypes,))

    if "dtype" not in kwargs:
      kwargs["dtype"] = dtypes.string

    if encoding is None:
      encoding = "utf-8"

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
        output_mode=output_mode,
        sparse=sparse,
        pad_to_max_tokens=pad_to_max_tokens,
        **kwargs)
    base_preprocessing_layer.keras_kpl_gauge.get_cell("StringLookup").set(True)

  def get_config(self):
    config = {"encoding": self.encoding}
    base_config = super(StringLookup, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_vocabulary(self):
    if self._table_handler.vocab_size() == 0:
      return []

    if self.invert:
      ids, strings = self._table_handler.data()
    else:
      strings, ids = self._table_handler.data()

    # This is required because the MutableHashTable doesn't preserve insertion
    # order, but we rely on the order of the array to assign indices.
    return [x.decode(self.encoding) for _, x in sorted(zip(ids, strings))]

  def set_vocabulary(self, vocab, idf_weights=None):
    if isinstance(vocab, str):
      if self.output_mode == index_lookup.TFIDF:
        raise RuntimeError("Setting vocabulary directly from a file is not "
                           "supported in TF-IDF mode, since this layer cannot "
                           "read files containing TF-IDF weight data. Please "
                           "read the file using Python and set the vocab "
                           "and weights by passing lists or arrays to the "
                           "set_vocabulary function's `vocab` and "
                           "`idf_weights` args.")
      vocab = table_utils.get_vocabulary_from_file(vocab, self.encoding)
    super().set_vocabulary(vocab, idf_weights=idf_weights)
