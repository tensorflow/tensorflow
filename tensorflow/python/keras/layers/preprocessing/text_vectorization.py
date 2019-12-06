# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Keras text vectorization preprocessing layer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_preprocessing_layer import Combiner
from tensorflow.python.keras.engine.base_preprocessing_layer import CombinerPreprocessingLayer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_string_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import compat

LOWER_AND_STRIP_PUNCTUATION = "lower_and_strip_punctuation"

SPLIT_ON_WHITESPACE = "whitespace"

TFIDF = "tf-idf"
INT = "int"
BINARY = "binary"
COUNT = "count"

# The string tokens in the extracted vocabulary
_VOCAB_NAME = "vocab"
# The inverse-document-frequency weights
_IDF_NAME = "idf"
# The IDF data for the OOV token
_OOV_IDF_NAME = "oov_idf"

# The string tokens in the full vocabulary
_ACCUMULATOR_VOCAB_NAME = "vocab"
# The total counts of each token in the vocabulary
_ACCUMULATOR_COUNTS_NAME = "counts"
# The number of doccumeents / examples that each token appears in.
_ACCUMULATOR_DOCUMENT_COUNTS = "document_counts"
# The total number of documents / examples in the dataset.
_ACCUMULATOR_NUM_DOCUMENTS = "num_documents"


class TextVectorization(CombinerPreprocessingLayer):
  """Text vectorization layer.

  This layer has basic options for managing text in a Keras model. It
  transforms a batch of strings (one sample = one string) into either a list of
  token indices (one sample = 1D tensor of integer token indices) or a dense
  representation (one sample = 1D tensor of float values representing data about
  the sample's tokens).

  The processing of each sample contains the following steps:
    1) standardize each sample (usually lowercasing + punctuation stripping)
    2) split each sample into substrings (usually words)
    3) recombine substrings into tokens (usually ngrams)
    4) index tokens (associate a unique int value with each token)
    5) transform each sample using this index, either into a vector of ints or
       a dense float vector.

  Attributes:
    max_tokens: The maximum size of the vocabulary for this layer. If None,
      there is no cap on the size of the vocabulary.
    standardize: Optional specification for standardization to apply to the
      input text. Values can be None (no standardization),
      LOWER_AND_STRIP_PUNCTUATION (lowercase and remove punctuation) or a
      Callable. Default is LOWER_AND_STRIP_PUNCTUATION.
    split: Optional specification for splitting the input text. Values can be
      None (no splitting), SPLIT_ON_WHITESPACE (split on ASCII whitespace), or a
      Callable. Default is SPLIT_ON_WHITESPACE.
    ngrams: Optional specification for ngrams to create from the possibly-split
      input text. Values can be None, an integer or tuple of integers; passing
      an integer will create ngrams up to that integer, and passing a tuple of
      integers will create ngrams for the specified values in the tuple. Passing
      None means that no ngrams will be created.
    output_mode: Optional specification for the output of the layer. Values can
      be INT, BINARY, COUNT or TFIDF, which control the outputs as follows:
        INT: Outputs integer indices, one integer index per split string token.
        BINARY: Outputs a single int array per batch, of either vocab_size or
          max_tokens size, containing 1s in all elements where the token mapped
          to that index exists at least once in the batch item.
        COUNT: As BINARY, but the int array contains a count of the number of
          times the token at that index appeared in the batch item.
        TFIDF: As BINARY, but the TF-IDF algorithm is applied to find the value
          in each token slot.
    output_sequence_length: Only valid in INT mode. If set, the output will have
      its time dimension padded or truncated to exactly `output_sequence_length`
      values, resulting in a tensor of shape [batch_size,
      output_sequence_length] regardless of how many tokens resulted from the
      splitting step. Defaults to None.
    pad_to_max_tokens: Only valid in  BINARY, COUNT, and TFIDF modes. If True,
      the output will have its feature axis padded to `max_tokens` even if the
      number of unique tokens in the vocabulary is less than max_tokens,
      resulting in a tensor of shape [batch_size, max_tokens] regardless of
      vocabulary size. Defaults to True.
  """
  # TODO(momernick): Add an examples section to the docstring.

  def __init__(self,
               max_tokens=None,
               standardize=LOWER_AND_STRIP_PUNCTUATION,
               split=SPLIT_ON_WHITESPACE,
               ngrams=None,
               output_mode=INT,
               output_sequence_length=None,
               pad_to_max_tokens=True,
               **kwargs):

    # This layer only applies to string processing, and so should only have
    # a dtype of 'string'.
    if "dtype" in kwargs and kwargs["dtype"] != dtypes.string:
      raise ValueError("TextVectorization may only have a dtype of string.")
    elif "dtype" not in kwargs:
      kwargs["dtype"] = dtypes.string

    # TODO(momernick): Validate the inputs. The following must apply:
    # 'standardize' must be one of (None, LOWER_AND_STRIP, callable)
    # 'split' must be one of (None, WHITESPACE, callable)
    # 'ngrams' must be one of (None, int, tuple(int))
    # 'output_mode' must be one of (None, INT, COUNT, BINARY, TFIDF)
    # 'output_sequence_length' must be one of (None, int) and is only
    # set if output_mode is INT.

    self._max_tokens = max_tokens

    # In INT mode, we have two reserved values (PAD and OOV). However, non-INT
    # modes don't have a PAD value, so we only need to reserve one value.
    self._reserved_values = 2 if output_mode == INT else 1

    # In INT mode, the zero value is reserved for padding (per Keras standard
    # padding approaches). In non-INT modes, there is no padding so we can set
    # the OOV value to zero instead of one.
    self._oov_value = 1 if output_mode == INT else 0

    # We always reduce the max token number by 1 to account for the OOV token
    # if it is set. The PAD marker isn't really a token (it's the absence of a
    # token) so we don't account for it here.
    self._max_vocab_size = max_tokens - 1 if max_tokens is not None else None

    # This is an explicit regex of all the tokens that will be stripped if
    # LOWER_AND_STRIP_PUNCTUATION is set. If an application requires other
    # stripping, a Callable should be passed into the 'standardize' arg.
    self._strip_regex = r'[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\t\n\']'

    self._standardize = standardize
    self._split = split
    self._ngrams_arg = ngrams
    if isinstance(ngrams, int):
      self._ngrams = tuple(range(1, ngrams + 1))
    else:
      self._ngrams = ngrams

    self._output_mode = output_mode
    self._output_sequence_length = output_sequence_length
    self._pad_to_max = pad_to_max_tokens
    self._has_vocab = False

    super(TextVectorization, self).__init__(
        combiner=_TextVectorizationCombiner(
            self._max_vocab_size, compute_idf=output_mode == TFIDF),
        **kwargs)

    self._table = lookup_ops.MutableHashTable(
        key_dtype=dtypes.string,
        value_dtype=dtypes.int64,
        default_value=self._oov_value,
        name=(self._name + "_index_table"))

    self._add_trackable(self._table, trainable=False)

    # We are adding this here instead of in build() since it does not depend
    # on the input shape at all.
    if self._output_mode == TFIDF:
      # Create the TFIDF weight, but use a (None,) tensorshape. This creates
      # a 1D variable with arbitrary shape, which we can assign any weight to
      # so long as it has 1 dimension. In order to properly initialize this
      # weight in Keras, we need to provide a custom callable initializer which
      # does not depend on the shape of the weight (as all other initializers
      # do) since the weight is not known. Hence the lambda shape, dtype: [0].
      self._tf_idf_weights = self.add_weight(
          name="tfidf_data",
          shape=tensor_shape.TensorShape((None,)),
          dtype=K.floatx(),
          trainable=False,
          initializer=lambda shape, dtype: [0])

  # These are V1/V2 shim points. There are V1 implementations in the V1 class.
  def _get_table_data(self):
    keys, values = self._table.export()
    return (keys.numpy(), values.numpy())

  def _get_table_size(self):
    return self._table.size()

  def _clear_table(self):
    keys, _ = self._table.export()
    self._table.remove(keys)
    self._has_vocab = False

  def _insert_table_data(self, keys, values):
    if len(values) != len(keys):
      raise RuntimeError("Size mismatch between values and key arrays. "
                         "Keys had size %s, values had size %s." %
                         (len(keys), len(values)))
    self._table.insert(keys, values)
    self._has_vocab = True

  def _to_numpy(self, preprocessed_data):
    """Converts preprocessed inputs into numpy arrays."""
    if isinstance(preprocessed_data, np.ndarray):
      return preprocessed_data
    return np.array(preprocessed_data.to_list())
  # End of V1/V2 shim points.

  def _assert_same_type(self, expected_type, values, value_name):
    if dtypes.as_dtype(expected_type) != dtypes.as_dtype(values.dtype):
      raise RuntimeError("Expected %s type %s, got %s" %
                         (value_name, expected_type, values.dtype))

  def _convert_to_ndarray(self, x):
    return np.array(x) if isinstance(x, (list, tuple)) else x

  def compute_output_shape(self, input_shape):
    if self._output_mode != INT:
      return tensor_shape.TensorShape([input_shape[0], self._max_tokens])

    if self._output_mode == INT and self._split is None:
      return input_shape

    if self._output_mode == INT and self._split is not None:
      input_shape = list(input_shape)
      input_shape[1] = self._output_sequence_length
      return tensor_shape.TensorShape(input_shape)

  def compute_output_signature(self, input_spec):
    output_shape = self.compute_output_shape(input_spec.shape.as_list())
    output_dtype = K.floatx() if self._output_mode == TFIDF else dtypes.int64
    return tensor_spec.TensorSpec(shape=output_shape, dtype=output_dtype)

  def adapt(self, data, reset_state=True):
    """Fits the state of the preprocessing layer to the dataset.

    Overrides the default adapt method to apply relevant preprocessing to the
    inputs before passing to the combiner.

    Arguments:
      data: The data to train on. It can be passed either as a tf.data Dataset,
        or as a numpy array.
      reset_state: Optional argument specifying whether to clear the state of
        the layer at the start of the call to `adapt`. This must be True for
        this layer, which does not support repeated calls to `adapt`.
    """
    if not reset_state:
      raise ValueError("TextVectorization does not support streaming adapts.")
    self.build(data.shape)
    # TODO(askerryryan): Look into making preprocessing a model that can be
    # passed as a subgraph to dataset.map.
    preprocessed_inputs = self._preprocess(data)
    super(TextVectorization,
          self).adapt(self._to_numpy(preprocessed_inputs), reset_state)

  def get_vocabulary(self):
    if not self._has_vocab:
      return []

    keys, values = self._get_table_data()
    # This is required because the MutableHashTable doesn't preserve insertion
    # order, but we rely on the order of the array to assign indices.
    return [x for _, x in sorted(zip(values, keys))]

  def get_config(self):
    config = {
        "max_tokens": self._max_tokens,
        "standardize": self._standardize,
        "split": self._split,
        "ngrams": self._ngrams_arg,
        "output_mode": self._output_mode,
        "output_sequence_length": self._output_sequence_length,
        "pad_to_max_tokens": self._pad_to_max,
    }
    base_config = super(TextVectorization, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def set_vocabulary(self,
                     vocab,
                     df_data=None,
                     oov_df_value=None,
                     append=False):
    """Sets vocabulary (and optionally document frequency) data for this layer.

    This method sets the vocabulary and DF data for this layer directly, instead
    of analyzing a dataset through 'adapt'. It should be used whenever the vocab
    (and optionally document frequency) information is already known. If
    vocabulary data is already present in the layer, this method will either
    replace it, if 'append' is set to False, or append to it (if 'append' is set
    to True).

    Arguments:
      vocab: An array of string tokens.
      df_data: An array of document frequency data. Only necessary if the layer
        output_mode is TFIDF.
      oov_df_value: The document frequency of the OOV token. Only necessary if
        output_mode is TFIDF. OOV data is optional when appending additional
        data in TFIDF mode; if an OOV value is supplied it will overwrite the
        existing OOV value.
      append: Whether to overwrite or append any existing vocabulary data.

    Raises:
      ValueError: If there are too many inputs, the inputs do not match, or
        input data is missing.
    """
    total_vocab_size = len(vocab) + (self._get_table_size() if append else 0)
    if self._max_tokens is not None and total_vocab_size > self._max_vocab_size:
      raise ValueError(
          "Attempted to set a vocabulary larger than the maximum vocab size. "
          "Passed vocab size is %s, max vocab size is %s. Note that the OOV "
          "token is automatically added to the number of tokens." %
          (total_vocab_size, self._max_vocab_size))

    # We're only _really_ appending if the table_size is nonzero. This is
    # important for some sanity checks in tfidf mode (specifically, checking if
    # oov_df_value is set or not) and handling existing tfidf weight data.
    append = append if self._get_table_size() > 0 else False

    if self._output_mode == TFIDF:
      if df_data is None:
        raise ValueError("df_data must be set if output_mode is TFIDF")
      if len(vocab) != len(df_data):
        raise ValueError("df_data must be the same length as vocab. "
                         "len(df_data) is %s, len(vocab) is %s" %
                         (len(vocab), len(df_data)))
      if not append and oov_df_value is None:
        raise ValueError("You must pass an oov_df_value the first time "
                         "'set_vocabulary' is called when output_mode is "
                         "TFIDF.")
    else:
      if df_data is not None:
        raise ValueError("df_data should only be set if output_mode is TFIDF. "
                         "output_mode is %s." % self._output_mode)

    start_index = self._reserved_values + (
        self._get_table_size() if append else 0)
    values = np.arange(start_index, len(vocab) + start_index, dtype=np.int64)

    vocab = self._convert_to_ndarray(vocab)
    self._assert_same_type(dtypes.string, vocab, "vocab")

    values = self._convert_to_ndarray(values)
    self._assert_same_type(dtypes.int64, values, "values")

    if self._output_mode == TFIDF:
      df_data = self._convert_to_ndarray(df_data)
      if append:
        # The existing IDF data is stored in a Keras weight, so we can get it
        # by calling K.get_value() on the weight object. Take the first
        # table_size+1 values in case we're padding the weight with zeros.
        existing_df_data = K.get_value(
            self._tf_idf_weights)[:self._get_table_size() + 1]
        df_data = np.append(existing_df_data, df_data, axis=0)
        # If we are appending and need to replace the OOV DF value, we can just
        # assign it over the existing OOV DF value at index 0 of the (already-
        # concatenated) DF value array.
        if oov_df_value is not None:
          df_data[0] = oov_df_value
      else:
        # If we are not appending (that is, we have only new data) we need to
        # insert the OOV value to the front of the array. (This is an append to
        # the head, not a replacement of the zeroth value.)
        if not isinstance(oov_df_value, np.ndarray):
          oov_df_value = np.array([oov_df_value])
        df_data = np.insert(df_data, 0, oov_df_value)

      if self._pad_to_max:
        padding_size = self._max_tokens - len(df_data)
        df_data = np.pad(df_data, (0, padding_size), "constant")

      # As above, we're using the fact that df_data is a Keras weight to
      # simplify storing the value back into the TF variable.
      K.set_value(self._tf_idf_weights, df_data)

    if not append and self._has_vocab:
      self._clear_table()

    self._insert_table_data(vocab, values)

  def build(self, input_shape):
    # We have to use 'and not ==' here, because input_shape[1] !/== 1 can result
    # in None for undefined shape axes. If using 'and !=', this causes the
    # expression to evaluate to False instead of True if the shape is undefined;
    # the expression needs to evaluate to True in that case.
    if self._split is not None and not input_shape[1] == 1:  # pylint: disable=g-comparison-negation
      raise RuntimeError(
          "When using TextVectorization to tokenize strings, the first "
          "dimension of the input array must be 1, got shape "
          "{}".format(input_shape))

    super(TextVectorization, self).build(input_shape)

  def _set_state_variables(self, updates):
    if not self.built:
      raise RuntimeError("_set_state_variables() must be called after build().")
    if self._output_mode == TFIDF:
      self.set_vocabulary(updates[_VOCAB_NAME], updates[_IDF_NAME],
                          updates[_OOV_IDF_NAME])
    else:
      self.set_vocabulary(updates[_VOCAB_NAME])

  def _preprocess(self, inputs):
    if self._standardize is LOWER_AND_STRIP_PUNCTUATION:
      lowercase_inputs = gen_string_ops.string_lower(inputs)
      inputs = string_ops.regex_replace(lowercase_inputs, self._strip_regex, "")
    elif self._standardize is not None:
      # TODO(momernick): Support callables here.
      raise RuntimeError("Not a supported standardization.")

    if self._split is SPLIT_ON_WHITESPACE:
      # If split isn't None, we validate that the 1st axis is of dimension 1 and
      # so can be squeezed out. We do this here instead of after splitting for
      # performance reasons - it's more expensive to squeeze a ragged tensor.
      inputs = array_ops.squeeze(inputs, axis=1)
      # This treats multiple whitespaces as one whitespace, and strips leading
      # and trailing whitespace.
      inputs = ragged_string_ops.string_split_v2(inputs)
    elif self._split is not None:
      # TODO(momernick): Support callables here.
      raise RuntimeError("Not a supported splitting.")

    # Note that 'inputs' here can be either ragged or dense depending on the
    # configuration choices for this Layer. The strings.ngrams op, however, does
    # support both ragged and dense inputs.
    if self._ngrams is not None:
      inputs = ragged_string_ops.ngrams(
          inputs, ngram_width=self._ngrams, separator=" ")

    return inputs

  def call(self, inputs):
    inputs = self._preprocess(inputs)

    # If we're not doing any output processing, return right away.
    if self._output_mode is None:
      return inputs

    # The table lookup ops don't natively support ragged tensors, so if we have
    # a RT we need to use map_flat_values to look up every element.
    if ragged_tensor.is_ragged(inputs):
      indexed_data = ragged_functional_ops.map_flat_values(
          self._table.lookup, inputs)
    else:
      indexed_data = self._table.lookup(inputs)

    if self._output_mode == INT:
      # Once we have the dense tensor, we can return it if we weren't given a
      # fixed output sequence length. If we were, though, we have to dynamically
      # choose whether to pad or trim it based on each tensor.

      # We need to convert to dense if we have a ragged tensor.
      if ragged_tensor.is_ragged(indexed_data):
        dense_data = indexed_data.to_tensor(default_value=0)
      else:
        dense_data = indexed_data

      if self._output_sequence_length is None:
        return dense_data
      else:
        sequence_len = K.shape(dense_data)[1]
        pad_amt = self._output_sequence_length - sequence_len
        pad_fn = lambda: array_ops.pad(dense_data, [[0, 0], [0, pad_amt]])
        slice_fn = lambda: dense_data[:, :self._output_sequence_length]
        return control_flow_ops.cond(
            sequence_len < self._output_sequence_length,
            true_fn=pad_fn,
            false_fn=slice_fn)

    out_depth = self._max_tokens if self._pad_to_max else math_ops.cast(
        (self._get_table_size() + self._reserved_values), dtypes.int32)

    if self._output_mode == BINARY:
      bool_one_hot_data = array_ops.one_hot(
          indexed_data, depth=out_depth, on_value=True, off_value=False)
      reduced_bool_data = math_ops.reduce_any(bool_one_hot_data, axis=1)
      binary_data = math_ops.cast(reduced_bool_data, dtypes.int64)
      return binary_data

    one_hot_data = array_ops.one_hot(indexed_data, depth=out_depth)
    counts = math_ops.reduce_sum(one_hot_data, axis=1)
    if self._output_mode == COUNT:
      return math_ops.cast(counts, dtypes.int64)

    tf_idf_data = math_ops.multiply(counts, self._tf_idf_weights)
    if self._output_mode == TFIDF:
      return tf_idf_data

    # We can only get here if we didn't recognize the passed mode.
    raise ValueError("Unknown output mode %s" % self._output_mode)


class _TextVectorizationCombiner(Combiner):
  """Combiner for the TextVectorization preprocessing layer.

  This class encapsulates the logic for computing a vocabulary based on the
  frequency of each token.

  Attributes:
    vocab_size: (Optional) If set, only the top `vocab_size` tokens (based on
      frequency across the dataset) are retained in the vocabulary. If None, or
      set to a value greater than the total number of distinct tokens in the
      dataset, all tokens are retained.
  """

  def __init__(self, vocab_size=None, compute_idf=False):
    self._vocab_size = vocab_size
    self._compute_idf = compute_idf
    self._input_dtype = dtypes.string

  def compute(self, values, accumulator=None):
    """Compute a step in this computation, returning a new accumulator."""
    # The batch dimension is irrelevant for counting token occurences, so we
    # concat into a single token vector
    if dtypes.as_dtype(self._input_dtype) != dtypes.as_dtype(values.dtype):
      raise RuntimeError("Expected input type %s, got %s" %
                         (self._input_dtype, values.dtype))
    flattened_batch = np.concatenate(values)
    vocab, counts = np.unique(flattened_batch, return_counts=True)
    if self._compute_idf:
      document_counts = np.sum(
          [np.in1d(vocab, document).astype(int) for document in values], axis=0)
      num_documents = len(values)
    else:
      document_counts = None
      num_documents = None
    batch_accumulator = self._create_accumulator(vocab, counts, document_counts,
                                                 num_documents)
    if accumulator is None:
      return batch_accumulator
    else:
      return self.merge([accumulator, batch_accumulator])

  def merge(self, accumulators):
    """Merge several accumulators to a single accumulator."""
    # TODO(askerryryan): Think about performance and benchmark different options
    # for the merge algo.
    concat_vocab = np.concatenate(
        [getattr(acc, _ACCUMULATOR_VOCAB_NAME) for acc in accumulators])
    concat_counts = np.concatenate(
        [getattr(acc, _ACCUMULATOR_COUNTS_NAME) for acc in accumulators])
    concat_document_counts = np.concatenate(
        [getattr(acc, _ACCUMULATOR_DOCUMENT_COUNTS) for acc in accumulators])
    merged_values, merged_indices = np.unique(concat_vocab, return_inverse=True)

    def sum_segment(index, array_to_segment):
      """Sum the counts from a segment specified by merged_indices == index."""
      indices = np.nonzero(merged_indices == index)
      return np.sum(array_to_segment[indices])

    segmented_sum = np.vectorize(sum_segment, excluded=["array_to_segment"])
    indices = np.arange(np.max(merged_indices) + 1)
    merged_counts = segmented_sum(indices, array_to_segment=concat_counts)
    sorted_indices = np.argsort(-merged_counts)
    vocab = merged_values[sorted_indices]
    counts = merged_counts[sorted_indices]
    if self._compute_idf:
      document_counts = segmented_sum(
          indices, array_to_segment=concat_document_counts)[sorted_indices]
      num_documents = np.sum(
          [getattr(acc, _ACCUMULATOR_NUM_DOCUMENTS) for acc in accumulators])
    else:
      document_counts = None
      num_documents = None
    return self._create_accumulator(vocab, counts, document_counts,
                                    num_documents)

  def _inverse_document_frequency(self, document_counts, num_documents):
    """Compute the inverse-document-frequency (IDF) component of TFIDF.

    Uses the default weighting scheme described in
    https://en.wikipedia.org/wiki/Tf%E2%80%93idf.

    Args:
      document_counts: An array of the # of documents each token appears in.
      num_documents: An int representing the total number of documents

    Returns:
      An array of "inverse document frequency" weights.
    """
    return np.log(1 + num_documents / (1 + document_counts))

  def extract(self, accumulator):
    """Convert an accumulator into a dict of output values.

    Args:
      accumulator: An accumulator aggregating over the full dataset.

    Returns:
      A dict of:
        "vocab": A list of the retained items in the vocabulary.
        "idf": The inverse-document-frequency for each item in vocab.
          idf[vocab_idx] is the IDF value for the corresponding vocab item.
        "oov_idf": The inverse-document-frequency for the OOV token.
    """
    if self._compute_idf:
      vocab, _, document_counts, num_documents = accumulator
    else:
      vocab, _ = accumulator
    vocab = vocab[:self._vocab_size] if self._vocab_size is not None else vocab
    if self._compute_idf:
      if self._vocab_size is not None:
        document_counts = document_counts[:self._vocab_size]
      idf = self._inverse_document_frequency(document_counts, num_documents)
      oov_idf = np.array([np.log(1 + num_documents)])
      return {_VOCAB_NAME: vocab, _IDF_NAME: idf, _OOV_IDF_NAME: oov_idf}
    else:
      return {_VOCAB_NAME: vocab}

  def restore(self, output):
    """Create an accumulator based on 'output'."""
    raise NotImplementedError(
        "TextVectorization does not restore or support streaming updates.")

  def _accumulator_fields(self):
    """Returns the list of fields stored on the accumulator."""
    fields = [_ACCUMULATOR_VOCAB_NAME, _ACCUMULATOR_COUNTS_NAME]
    if self._compute_idf:
      fields += [_ACCUMULATOR_DOCUMENT_COUNTS, _ACCUMULATOR_NUM_DOCUMENTS]
    return fields

  def serialize(self, accumulator):
    """Serialize an accumulator for a remote call."""
    fields = self._accumulator_fields()
    output_dict = {name: getattr(accumulator, name).tolist() for name in fields}
    return compat.as_bytes(json.dumps(output_dict))

  def deserialize(self, encoded_accumulator):
    """Deserialize an accumulator received from 'serialize()'."""
    accumulator_dict = json.loads(compat.as_text(encoded_accumulator))
    args = [accumulator_dict[field] for field in self._accumulator_fields()]
    return self._create_accumulator(*args)

  def _create_accumulator(self,
                          vocab,
                          counts,
                          document_counts=None,
                          num_documents=None):
    """Accumulate a sorted array of vocab tokens and corresponding counts."""
    accumulator = collections.namedtuple("Accumulator",
                                         self._accumulator_fields())
    counts = np.array(counts)
    vocab = np.array(vocab)
    if dtypes.as_dtype(vocab.dtype) != dtypes.as_dtype(self._input_dtype):
      raise ValueError("Expected vocab type %s, got %s" %
                       (self._input_dtype, vocab.dtype))
    if not np.issubdtype(counts.dtype, np.number):
      raise ValueError("Expected counts to be numeric")

    sorted_indices = np.argsort(-counts)
    counts = counts[sorted_indices]
    vocab = vocab[sorted_indices]
    if self._compute_idf:
      document_counts = np.array(document_counts)[sorted_indices]
      num_documents = np.array(num_documents)
      return accumulator(vocab, counts, document_counts, num_documents)
    else:
      return accumulator(vocab, counts)
