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


@keras_export("keras.layers.experimental.preprocessing.IntegerLookup", v1=[])
class IntegerLookup(index_lookup.IndexLookup):
  """Maps integers from a vocabulary to integer indices.

  This layer translates a set of arbitrary integers into an integer output via a
  table-based lookup, with optional out-of-vocabulary handling.

  If desired, the user can call this layer's `adapt()` method on a data set,
  which will analyze the data set, determine the frequency of individual integer
  values, and create a vocabulary from them. This vocabulary can have
  unlimited size or be capped, depending on the configuration options for this
  layer; if there are more unique values in the input than the maximum
  vocabulary size, the most frequent values will be used to create the
  vocabulary (and the values that don't make the cut will be treated as OOV).

  Arguments:
    max_values: The maximum size of the vocabulary for this layer. If None,
      there is no cap on the size of the vocabulary. Note that this vocabulary
      includes the OOV and mask values, so the effective number of values is
      `(max_values - num_oov_values - (1 if mask_token else 0))`.
    num_oov_indices: The number of out-of-vocabulary values to use; defaults to
      1. If this value is more than 1, OOV inputs are modulated to determine
      their OOV value; if this value is 0, passing an OOV input will result in
      a '-1' being returned for that value in the output tensor. (Note that,
      because the value is -1 and not 0, this will allow you to effectively drop
      OOV values from categorical encodings.)
    mask_value: A value that represents masked inputs, and which is mapped to
      index 0. Defaults to 0. If set to None, no mask term will be added and the
      OOV values, if any, will be indexed from `(0...num_oov_values)` instead of
      `(1...num_oov_values + 1)`.
    oov_value: The value representing an out-of-vocabulary value. Defaults to
      -1.
    vocabulary: An optional list of values, or a path to a text file containing
      a vocabulary to load into this layer. The file should contain one value
      per line. If the list or file contains the same token multiple times, an
      error will be thrown.
    invert: If true, this layer will map indices to vocabulary items instead
      of mapping vocabulary items to indices.
    output_mode: Specification for the output of the layer. Only applicable
      when `invert` is False.
      Defaults to "int". Values can
      be "int", "binary", or "count", configuring the layer as follows:
        "int": Return the raw integer indices of the input values.
        "binary": Outputs a single int array per batch, of either vocab_size or
          max_tokens size, containing 1s in all elements where the token mapped
          to that index exists at least once in the batch item.
        "count": Like "binary", but the int array contains a count of the number
          of times the token at that index appeared in the batch item.
    sparse: Boolean. Only applicable to "binary" and "count" output modes.
      If true, returns a `SparseTensor` instead of a dense `Tensor`.
      Defaults to `False`.

  Examples:

  **Creating a lookup layer with a known vocabulary**

  This example creates a lookup layer with a pre-existing vocabulary.

  >>> vocab = [12, 36, 1138, 42]
  >>> data = tf.constant([[12, 1138, 42], [42, 1000, 36]])  # Note OOV values
  >>> layer = IntegerLookup(vocabulary=vocab)
  >>> layer(data)
  <tf.Tensor: shape=(2, 3), dtype=int64, numpy=
  array([[2, 4, 5],
         [5, 1, 3]])>

  **Configuring the layer to apply multi-hot encoding after lookup**

  Just set `output_mode='binary'`. Note that the first two dimensions
  in the binary encoding represent the mask value and the OOV value,
  respectively.

  >>> vocab = [12, 36, 1138, 42]
  >>> data = tf.constant([[12, 1138, 42], [42, 1000, 36]])  # Note OOV values
  >>> layer = IntegerLookup(vocabulary=vocab, output_mode='binary')
  >>> layer(data)
  <tf.Tensor: shape=(2, 6), dtype=float32, numpy=
    array([[0., 0., 1., 0., 1., 1.],
           [0., 1., 0., 1., 0., 1.]], dtype=float32)>


  **Creating a lookup layer with an adapted vocabulary**

  This example creates a lookup layer and generates the vocabulary by analyzing
  the dataset.

  >>> data = tf.constant([[12, 1138, 42], [42, 1000, 36]])
  >>> layer = IntegerLookup()
  >>> layer.adapt(data)
  >>> layer.get_vocabulary()
  [0, -1, 42, 1138, 1000, 36, 12]

  Note how the mask value 0 and the OOV value -1 have been added to the
  vocabulary. The remaining values are sorted by frequency (1138, which has
  2 occurrences, is first) then by inverse sort order.

  >>> data = tf.constant([[12, 1138, 42], [42, 1000, 36]])
  >>> layer = IntegerLookup()
  >>> layer.adapt(data)
  >>> layer(data)
  <tf.Tensor: shape=(2, 3), dtype=int64, numpy=
  array([[6, 3, 2],
         [2, 4, 5]])>


  **Lookups with multiple OOV indices**

  This example demonstrates how to use a lookup layer with multiple OOV indices.
  When a layer is created with more than one OOV index, any OOV values are
  hashed into the number of OOV buckets, distributing OOV values in a
  deterministic fashion across the set.

  >>> vocab = [12, 36, 1138, 42]
  >>> data = tf.constant([[12, 1138, 42], [37, 1000, 36]])
  >>> layer = IntegerLookup(vocabulary=vocab, num_oov_indices=2)
  >>> layer(data)
  <tf.Tensor: shape=(2, 3), dtype=int64, numpy=
  array([[3, 5, 6],
         [2, 1, 4]])>

  Note that the output for OOV value 37 is 2, while the output for OOV value
  1000 is 1. The in-vocab terms have their output index increased by 1 from
  earlier examples (12 maps to 3, etc) in order to make space for the extra OOV
  value.


  **Inverse lookup**

  This example demonstrates how to map indices to values using this layer. (You
  can also use adapt() with inverse=True, but for simplicity we'll pass the
  vocab in this example.)

  >>> vocab = [12, 36, 1138, 42]
  >>> data = tf.constant([[1, 3, 4], [4, 5, 2]])
  >>> layer = IntegerLookup(vocabulary=vocab, invert=True)
  >>> layer(data)
  <tf.Tensor: shape=(2, 3), dtype=int64, numpy=
  array([[  12, 1138,   42],
         [  42,   -1,   36]])>

  Note that the integer 5, which is out of the vocabulary space, returns an OOV
  token.


  **Forward and inverse lookup pairs**

  This example demonstrates how to use the vocabulary of a standard lookup
  layer to create an inverse lookup layer.

  >>> vocab = [12, 36, 1138, 42]
  >>> data = tf.constant([[12, 1138, 42], [42, 1000, 36]])
  >>> layer = IntegerLookup(vocabulary=vocab)
  >>> i_layer = IntegerLookup(vocabulary=layer.get_vocabulary(), invert=True)
  >>> int_data = layer(data)
  >>> i_layer(int_data)
  <tf.Tensor: shape=(2, 3), dtype=int64, numpy=
  array([[  12, 1138,   42],
         [  42,   -1,   36]])>

  In this example, the input value 1000 resulted in an output of -1, since
  1000 was not in the vocabulary - it got represented as an OOV, and all OOV
  values are returned as -1 in the inverse layer. Also, note that for the
  inverse to work, you must have already set the forward layer vocabulary
  either directly or via `fit()` before calling `get_vocabulary()`.
  """

  def __init__(self,
               max_values=None,
               num_oov_indices=1,
               mask_value=0,
               oov_value=-1,
               vocabulary=None,
               invert=False,
               output_mode=index_lookup.INT,
               sparse=False,
               **kwargs):
    allowed_dtypes = [dtypes.int64]

    if "dtype" in kwargs and kwargs["dtype"] not in allowed_dtypes:
      raise ValueError("The value of the dtype argument for IntegerLookup may "
                       "only be one of %s." % (allowed_dtypes,))

    if "dtype" not in kwargs:
      kwargs["dtype"] = dtypes.int64

    # If max_values is set, the value must be greater than 1 - otherwise we
    # are creating a 0-element vocab, which doesn't make sense.
    if max_values is not None and max_values <= 1:
      raise ValueError("If set, max_values must be greater than 1. "
                       "You passed %s" % (max_values,))

    if num_oov_indices < 0:
      raise ValueError("num_oov_indices must be greater than 0. You passed %s" %
                       (num_oov_indices,))

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
        output_mode=output_mode,
        sparse=sparse,
        **kwargs)
    base_preprocessing_layer.keras_kpl_gauge.get_cell("IntegerLookup").set(True)

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

  def set_vocabulary(self, vocab):
    if isinstance(vocab, str):
      vocab = table_utils.get_vocabulary_from_file(vocab)
      vocab = [int(v) for v in vocab]
    super().set_vocabulary(vocab)
