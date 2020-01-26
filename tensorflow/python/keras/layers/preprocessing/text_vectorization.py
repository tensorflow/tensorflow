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
import operator

import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_preprocessing_layer import Combiner
from tensorflow.python.keras.engine.base_preprocessing_layer import CombinerPreprocessingLayer
from tensorflow.python.keras.layers.preprocessing import categorical_encoding
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_string_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import keras_export

LOWER_AND_STRIP_PUNCTUATION = "lower_and_strip_punctuation"

SPLIT_ON_WHITESPACE = "whitespace"

TFIDF = categorical_encoding.TFIDF
INT = categorical_encoding.INT
BINARY = categorical_encoding.BINARY
COUNT = categorical_encoding.COUNT

# This is an explicit regex of all the tokens that will be stripped if
# LOWER_AND_STRIP_PUNCTUATION is set. If an application requires other
# stripping, a Callable should be passed into the 'standardize' arg.
DEFAULT_STRIP_REGEX = r'[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']'

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


@keras_export(
    "keras.layers.experimental.preprocessing.TextVectorization", v1=[])
class TextVectorization(CombinerPreprocessingLayer):
  """Text vectorization layer.

  This layer has basic options for managing text in a Keras model. It
  transforms a batch of strings (one sample = one string) into either a list of
  token indices (one sample = 1D tensor of integer token indices) or a dense
  representation (one sample = 1D tensor of float values representing data about
  the sample's tokens).

  If desired, the user can call this layer's adapt() method on a dataset.
  When this layer is adapted, it will analyze the dataset, determine the
  frequency of individual string values, and create a 'vocabulary' from them.
  This vocabulary can have unlimited size or be capped, depending on the
  configuration options for this layer; if there are more unique values in the
  input than the maximum vocabulary size, the most frequent terms will be used
  to create the vocabulary.

  The processing of each sample contains the following steps:
    1) standardize each sample (usually lowercasing + punctuation stripping)
    2) split each sample into substrings (usually words)
    3) recombine substrings into tokens (usually ngrams)
    4) index tokens (associate a unique int value with each token)
    5) transform each sample using this index, either into a vector of ints or
       a dense float vector.

  Some notes on passing Callables to customize splitting and normalization for
  this layer:
    1) Any callable can be passed to this Layer, but if you want to serialize
       this object you should only pass functions that are registered Keras
       serializables (see `tf.keras.utils.register_keras_serializable` for more
       details).
    2) When using a custom callable for `standardize`, the data received
       by the callable will be exactly as passed to this layer. The callable
       should return a tensor of the same shape as the input.
    3) When using a custom callable for `split`, the data received by the
       callable will have the 1st dimension squeezed out - instead of
       `[["string to split"], ["another string to split"]]`, the Callable will
       see `["string to split", "another string to split"]`. The callable should
       return a Tensor with the first dimension containing the split tokens -
       in this example, we should see something like `[["string", "to", "split],
       ["another", "string", "to", "split"]]`. This makes the callable site
       natively compatible with `tf.strings.split()`.

  Attributes:
    max_tokens: The maximum size of the vocabulary for this layer. If None,
      there is no cap on the size of the vocabulary.
    standardize: Optional specification for standardization to apply to the
      input text. Values can be None (no standardization),
      'lower_and_strip_punctuation' (lowercase and remove punctuation) or a
      Callable. Default is 'lower_and_strip_punctuation'.
    split: Optional specification for splitting the input text. Values can be
      None (no splitting), 'whitespace' (split on ASCII whitespace), or a
      Callable. The default is 'whitespace'.
    ngrams: Optional specification for ngrams to create from the possibly-split
      input text. Values can be None, an integer or tuple of integers; passing
      an integer will create ngrams up to that integer, and passing a tuple of
      integers will create ngrams for the specified values in the tuple. Passing
      None means that no ngrams will be created.
    output_mode: Optional specification for the output of the layer. Values can
      be "int", "binary", "count" or "tf-idf", configuring the layer as follows:
        "int": Outputs integer indices, one integer index per split string
          token.
        "binary": Outputs a single int array per batch, of either vocab_size or
          max_tokens size, containing 1s in all elements where the token mapped
          to that index exists at least once in the batch item.
        "count": As "binary", but the int array contains a count of the number
          of times the token at that index appeared in the batch item.
        "tf-idf": As "binary", but the TF-IDF algorithm is applied to find the
          value in each token slot.
    output_sequence_length: Only valid in INT mode. If set, the output will have
      its time dimension padded or truncated to exactly `output_sequence_length`
      values, resulting in a tensor of shape [batch_size,
      output_sequence_length] regardless of how many tokens resulted from the
      splitting step. Defaults to None.
    pad_to_max_tokens: Only valid in  "binary", "count", and "tf-idf" modes. If
      True, the output will have its feature axis padded to `max_tokens` even if
      the number of unique tokens in the vocabulary is less than max_tokens,
      resulting in a tensor of shape [batch_size, max_tokens] regardless of
      vocabulary size. Defaults to True.

  Example:
  This example instantiates a TextVectorization layer that lowercases text,
  splits on whitespace, strips punctuation, and outputs integer vocab indices.
  ```
  max_features = 5000  # Maximum vocab size.
  max_len = 40  # Sequence length to pad the outputs to.

  # Create the layer.
  vectorize_layer = text_vectorization.TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=max_len)

  # Now that the vocab layer has been created, call `adapt` on the text-only
  # dataset to create the vocabulary. You don't have to batch, but for large
  # datasets this means we're not keeping spare copies of the dataset in memory.
  vectorize_layer.adapt(text_dataset.batch(64))

  # Create the model that uses the vectorize text layer
  model = tf.keras.models.Sequential()

  # Start by creating an explicit input layer. It needs to have a shape of (1,)
  # (because we need to guarantee that there is exactly one string input per
  # batch), and the dtype needs to be 'string'.
  model.add(tf.keras.Input(shape=(1,), dtype=tf.string))

  # The first layer in our model is the vectorization layer. After this layer,
  # we have a tensor of shape (batch_size, max_len) containing vocab indices.
  model.add(vectorize_layer)

  # Next, we add a layer to map those vocab indices into a space of
  # dimensionality 'embedding_dims'. Note that we're using max_features+1 here,
  # since there's an OOV token that gets added to the vocabulary in
  # vectorize_layer.
  model.add(tf.keras.layers.Embedding(max_features+1, embedding_dims))

  # At this point, you have embedded float data representing your tokens, and
  # can add whatever other layers you need to create your model.
  ```
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

    # 'standardize' must be one of (None, LOWER_AND_STRIP_PUNCTUATION, callable)
    layer_utils.validate_string_arg(
        standardize,
        allowable_strings=[LOWER_AND_STRIP_PUNCTUATION],
        layer_name="TextVectorization",
        arg_name="standardize",
        allow_none=True,
        allow_callables=True)

    # 'split' must be one of (None, SPLIT_ON_WHITESPACE, callable)
    layer_utils.validate_string_arg(
        split,
        allowable_strings=[SPLIT_ON_WHITESPACE],
        layer_name="TextVectorization",
        arg_name="split",
        allow_none=True,
        allow_callables=True)

    # 'output_mode' must be one of (None, INT, COUNT, BINARY, TFIDF)
    layer_utils.validate_string_arg(
        output_mode,
        allowable_strings=[INT, COUNT, BINARY, TFIDF],
        layer_name="TextVectorization",
        arg_name="output_mode",
        allow_none=True)

    # 'ngrams' must be one of (None, int, tuple(int))
    if not (ngrams is None or
            isinstance(ngrams, int) or
            isinstance(ngrams, tuple) and
            all(isinstance(item, int) for item in ngrams)):
      raise ValueError(("`ngrams` must be None, an integer, or a tuple of "
                        "integers. Got %s") % (ngrams,))

    # 'output_sequence_length' must be one of (None, int) and is only
    # set if output_mode is INT.
    if (output_mode == INT and not (isinstance(output_sequence_length, int) or
                                    (output_sequence_length is None))):
      raise ValueError("`output_sequence_length` must be either None or an "
                       "integer when `output_mode` is 'int'. "
                       "Got %s" % output_sequence_length)

    if output_mode != INT and output_sequence_length is not None:
      raise ValueError("`output_sequence_length` must not be set if "
                       "`output_mode` is not 'int'.")

    # If max_tokens is set, the value must be greater than 1 - otherwise we
    # are creating a 0-element vocab, which doesn't make sense.
    if max_tokens is not None and max_tokens < 1:
      raise ValueError("max_tokens must be > 1.")

    self._max_tokens = max_tokens

    # In INT mode, we have two reserved values (PAD and OOV). However, non-INT
    # modes don't have a PAD value, so we only need to reserve one value.
    self._reserved_values = 2 if output_mode == INT else 1

    # In INT mode, the zero value is reserved for padding (per Keras standard
    # padding approaches). In non-INT modes, there is no padding so we can set
    # the OOV value to zero instead of one.
    self._oov_value = 1 if output_mode == INT else 0

    # We always reduce the max token number by 1 to account for the OOV token
    # if it is set. Keras' use of the reserved number 0 for padding tokens,
    # if the output is in INT mode, does not really count as a 'token' for
    # vocabulary purposes, so we only reduce vocab size by 1 here.
    self._max_vocab_size = max_tokens - 1 if max_tokens is not None else None

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
    self._vocab_size = 0
    self._called = False

    super(TextVectorization, self).__init__(
        combiner=_TextVectorizationCombiner(
            self._max_vocab_size, compute_idf=output_mode == TFIDF),
        **kwargs)

    self._table = lookup_ops.MutableHashTable(
        key_dtype=dtypes.string,
        value_dtype=dtypes.int64,
        default_value=self._oov_value,
        name=(self._name + "_index_table"))

    def fail(_):
      raise NotImplementedError(
          "Saving is not yet supported for TextVectorization layers.")
    self._table._list_extra_dependencies_for_serialization = fail  # pylint: disable=protected-access

    tracked_table = self._add_trackable(self._table, trainable=False)

    # This is a workaround for summary() on this layer. Because the table is
    # not mutable during training, the effective number of parameters (and so
    # the weight shape) is 0; we add this as an attr so that the parameter
    # counting code in the Model object doesn't throw an attribute error.
    tracked_table.shape = tensor_shape.TensorShape((0,))

    # If this layer is configured for string or integer output, we do not
    # create a vectorization layer (as the output is not vectorized).
    if self._output_mode in [None, INT]:
      return

    if max_tokens is not None and self._pad_to_max:
      vectorize_max_tokens = max_tokens
    else:
      vectorize_max_tokens = None
    self._vectorize_layer = self._get_vectorization_class()(
        max_tokens=vectorize_max_tokens, output_mode=self._output_mode)

  # These are V1/V2 shim points. There are V1 implementations in the V1 class.
  def _get_vectorization_class(self):
    return categorical_encoding.CategoricalEncoding

  def _get_table_data(self):
    keys, values = self._table.export()
    return (keys.numpy(), values.numpy())

  def _get_table_size(self):
    return self._table.size().numpy()

  def _clear_table(self):
    if (self._output_mode in [BINARY, COUNT, TFIDF] and self._called and
        not self._pad_to_max):
      raise RuntimeError(("When using TextVectorization in {mode} mode, the "
                          "vocabulary cannot be changed after the layer is "
                          "called.").format(mode=self._output_mode))
    keys, _ = self._table.export()
    self._table.remove(keys)
    self._vocab_size = 0

  def _insert_table_data(self, keys, values):
    if (self._output_mode in [BINARY, COUNT, TFIDF] and self._called and
        not self._pad_to_max):
      raise RuntimeError(("When using TextVectorization in {mode} mode, the "
                          "vocabulary cannot be changed after the layer is "
                          "called.").format(mode=self._output_mode))
    if len(values) != len(keys):
      raise RuntimeError("Size mismatch between values and key arrays. "
                         "Keys had size %s, values had size %s." %
                         (len(keys), len(values)))
    self._table.insert(keys, values)
    self._vocab_size += len(keys)

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

    # Build the layer explicitly with the original data shape instead of relying
    # on an implicit call to `build` in the base layer's `adapt`, since
    # preprocessing changes the input shape.
    if isinstance(data, np.ndarray):
      if data.ndim == 1:
        data = np.expand_dims(data, axis=-1)
      self.build(data.shape)
      preprocessed_inputs = self._to_numpy(self._preprocess(data))
    elif isinstance(data, dataset_ops.DatasetV2):
      # TODO(momernick): Replace this with a more V2-friendly API.
      shape = dataset_ops.get_legacy_output_shapes(data)
      if not isinstance(shape, tensor_shape.TensorShape):
        raise ValueError("The dataset passed to 'adapt' must contain a single "
                         "tensor value.")
      if shape.rank == 1:
        data = data.map(lambda tensor: array_ops.expand_dims(tensor, -1))
      self.build(dataset_ops.get_legacy_output_shapes(data))
      preprocessed_inputs = data.map(self._preprocess)
    else:
      raise ValueError(
          "adapt() requires a Dataset or a Numpy array as input, got {}".format(
              type(data)))
    super(TextVectorization, self).adapt(preprocessed_inputs, reset_state)

  def get_vocabulary(self):
    if self._vocab_size == 0:
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

  def count_params(self):
    # This method counts the number of scalars in the weights of this layer.
    # Since this layer doesn't have any /actual/ weights (in that there's
    # nothing in this layer that can be trained - we only use the weight
    # abstraction for ease of saving!) we return 0.
    return 0

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
    current_table_size = self._get_table_size()
    total_vocab_size = len(vocab) + (current_table_size if append else 0)
    if self._max_tokens is not None and total_vocab_size > self._max_vocab_size:
      raise ValueError(
          "Attempted to set a vocabulary larger than the maximum vocab size. "
          "Passed vocab size is %s, max vocab size is %s. Note that the OOV "
          "token is automatically added to the number of tokens." %
          (total_vocab_size, self._max_vocab_size))

    # We're only _really_ appending if the table_size is nonzero. This is
    # important for some sanity checks in tfidf mode (specifically, checking if
    # oov_df_value is set or not) and handling existing tfidf weight data.
    append = append if current_table_size > 0 else False

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

    if not append and self._vocab_size > 0:
      self._clear_table()
    self._insert_table_data(vocab, values)

    # When doing raw or integer output, we don't have a Vectorize layer to
    # manage. In this case, we can return directly.
    if self._output_mode in [None, INT]:
      return

    if not self._pad_to_max or self._max_tokens is None:
      num_tokens = total_vocab_size + self._reserved_values
      self._vectorize_layer.set_num_elements(num_tokens)

    if self._output_mode == TFIDF:
      df_data = self._convert_to_ndarray(df_data)
      if append:
        # The existing IDF data is stored in a Keras weight, so we can get it
        # by calling K.get_value() on the weight object. Take the first
        # table_size+1 values in case we're padding the weight with zeros
        existing_df_data = K.get_value(
            self._vectorize_layer.tf_idf_weights)[:current_table_size + 1]
        df_data = np.append(existing_df_data, df_data, axis=0)
        # If we are appending and need to replace the OOV DF value, we can
        # assign it over the existing OOV DF value at index 0 of the (already-
        # concatenated) DF value array.
        if oov_df_value is not None:
          df_data[0] = oov_df_value
      else:
        # If we are not appending (that is, we have only new data) we need to
        # insert the OOV value to the front of the array. (This is a append to
        # the head, not a replacement of the zeroth value.)
        if not isinstance(oov_df_value, np.ndarray):
          oov_df_value = np.array([oov_df_value])
        df_data = np.insert(df_data, 0, oov_df_value)
      self._vectorize_layer.set_tfidf_data(df_data)

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

    # This handles a corner case where, if restored from weights or SavedModel,
    # the layer might not have accurate vocab size information.
    self._vocab_size = self._get_table_size()
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
    if self._standardize == LOWER_AND_STRIP_PUNCTUATION:
      lowercase_inputs = gen_string_ops.string_lower(inputs)
      inputs = string_ops.regex_replace(lowercase_inputs, DEFAULT_STRIP_REGEX,
                                        "")
    elif callable(self._standardize):
      inputs = self._standardize(inputs)
    elif self._standardize is not None:
      raise ValueError(("%s is not a supported standardization. "
                        "TextVectorization supports the following options "
                        "for `standardize`: None, "
                        "'lower_and_strip_punctuation', or a "
                        "Callable.") % self._standardize)

    if self._split is not None:
      # If we are splitting, we validate that the 1st axis is of dimension 1 and
      # so can be squeezed out. We do this here instead of after splitting for
      # performance reasons - it's more expensive to squeeze a ragged tensor.
      inputs = array_ops.squeeze(inputs, axis=1)
      if self._split == SPLIT_ON_WHITESPACE:
        # This treats multiple whitespaces as one whitespace, and strips leading
        # and trailing whitespace.
        inputs = ragged_string_ops.string_split_v2(inputs)
      elif callable(self._split):
        inputs = self._split(inputs)
      else:
        raise ValueError(
            ("%s is not a supported splitting."
             "TextVectorization supports the following options "
             "for `split`: None, 'whitespace', or a Callable.") % self._split)

    # Note that 'inputs' here can be either ragged or dense depending on the
    # configuration choices for this Layer. The strings.ngrams op, however, does
    # support both ragged and dense inputs.
    if self._ngrams is not None:
      inputs = ragged_string_ops.ngrams(
          inputs, ngram_width=self._ngrams, separator=" ")

    return inputs

  def call(self, inputs):
    self._called = True
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
        dense_data.set_shape(tensor_shape.TensorShape((None, None)))
        return dense_data
      else:
        sequence_len = K.shape(dense_data)[1]
        pad_amt = self._output_sequence_length - sequence_len
        pad_fn = lambda: array_ops.pad(dense_data, [[0, 0], [0, pad_amt]])
        slice_fn = lambda: dense_data[:, :self._output_sequence_length]
        output_tensor = control_flow_ops.cond(
            sequence_len < self._output_sequence_length,
            true_fn=pad_fn,
            false_fn=slice_fn)
        output_tensor.set_shape(
            tensor_shape.TensorShape((None, self._output_sequence_length)))
        return output_tensor

    # If we're not returning integers here, we rely on the vectorization layer
    # to create the output.
    return self._vectorize_layer(indexed_data)


# A note on this combiner: This contains functionality that will be extracted
# into the Vectorization and Lookup combiner objects. At that point,
# TextVectorization can become a PreprocessingStage instead of a Layer and
# this combiner can be retired. Until then, we leave this as is instead of
# attempting a refactor of what will soon be deleted.
class _TextVectorizationCombiner(Combiner):
  """Combiner for the TextVectorization preprocessing layer.

  This class encapsulates the logic for computing a vocabulary based on the
  frequency of each token.

  Attributes:
    vocab_size: (Optional) If set, only the top `vocab_size` tokens (based on
      frequency across the dataset) are retained in the vocabulary. If None, or
      set to a value greater than the total number of distinct tokens in the
      dataset, all tokens are retained.
    compute_idf: (Optional) If set, the inverse document frequency will be
      computed for each value.
  """

  def __init__(self, vocab_size=None, compute_idf=False):
    self._vocab_size = vocab_size
    self._compute_idf = compute_idf
    self._input_dtype = dtypes.string

  def compute(self, values, accumulator=None):
    """Compute a step in this computation, returning a new accumulator."""
    if dtypes.as_dtype(self._input_dtype) != dtypes.as_dtype(values.dtype):
      raise RuntimeError("Expected input type %s, got %s" %
                         (self._input_dtype, values.dtype))
    if ragged_tensor.is_ragged(values):
      values = values.to_list()
    if isinstance(values, ops.EagerTensor):
      values = values.numpy()
    if isinstance(values, np.ndarray):
      values = values.tolist()

    if accumulator is None:
      accumulator = self._create_accumulator()

    # TODO(momernick): Benchmark improvements to this algorithm.
    for document in values:
      current_doc_id = accumulator.metadata[0]
      for token in document:
        accumulator.count_dict[token] += 1
        if self._compute_idf:
          doc_count = accumulator.per_doc_count_dict[token]
          if doc_count["last_doc_id"] != current_doc_id:
            doc_count["count"] += 1
            doc_count["last_doc_id"] = current_doc_id
      accumulator.metadata[0] += 1

    return accumulator

  def merge(self, accumulators):
    """Merge several accumulators to a single accumulator."""
    if not accumulators:
      return accumulators

    base_accumulator = accumulators[0]

    for accumulator in accumulators[1:]:
      base_accumulator.metadata[0] += accumulator.metadata[0]
      for token, value in accumulator.count_dict.items():
        base_accumulator.count_dict[token] += value
      if self._compute_idf:
        for token, value in accumulator.per_doc_count_dict.items():
          # Any newly created token counts in 'base_accumulator''s
          # per_doc_count_dict will have a last_doc_id of -1. This is always
          # less than the next doc id (which are strictly positive), so any
          # future occurrences are guaranteed to be counted.
          base_accumulator.per_doc_count_dict[token]["count"] += value["count"]

    return base_accumulator

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
    return np.log(1 + num_documents / (1 + np.array(document_counts)))

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
      vocab_counts, document_counts, num_documents = accumulator
    else:
      vocab_counts, _, _ = accumulator

    sorted_counts = sorted(
        vocab_counts.items(), key=operator.itemgetter(1, 0), reverse=True)
    vocab_data = (
        sorted_counts[:self._vocab_size] if self._vocab_size else sorted_counts)
    vocab = [data[0] for data in vocab_data]

    if self._compute_idf:
      doc_counts = [document_counts[token]["count"] for token in vocab]
      idf = self._inverse_document_frequency(doc_counts, num_documents[0])
      oov_idf = np.array([np.log(1 + num_documents[0])])
      return {_VOCAB_NAME: vocab, _IDF_NAME: idf, _OOV_IDF_NAME: oov_idf}
    else:
      return {_VOCAB_NAME: vocab}

  def restore(self, output):
    """Create an accumulator based on 'output'."""
    raise NotImplementedError(
        "TextVectorization does not restore or support streaming updates.")

  def serialize(self, accumulator):
    """Serialize an accumulator for a remote call."""
    output_dict = {}
    output_dict["metadata"] = accumulator.metadata
    output_dict["vocab"] = list(accumulator.count_dict.keys())
    output_dict["vocab_counts"] = list(accumulator.count_dict.values())
    if self._compute_idf:
      output_dict["idf_vocab"] = list(accumulator.per_doc_count_dict.keys())
      output_dict["idf_counts"] = [
          counter["count"]
          for counter in accumulator.per_doc_count_dict.values()
      ]
    return compat.as_bytes(json.dumps(output_dict))

  def deserialize(self, encoded_accumulator):
    """Deserialize an accumulator received from 'serialize()'."""
    accumulator_dict = json.loads(compat.as_text(encoded_accumulator))

    accumulator = self._create_accumulator()
    accumulator.metadata[0] = accumulator_dict["metadata"][0]

    count_dict = dict(
        zip(accumulator_dict["vocab"], accumulator_dict["vocab_counts"]))
    accumulator.count_dict.update(count_dict)

    if self._compute_idf:
      create_dict = lambda x: {"count": x, "last_doc_id": -1}
      idf_count_dicts = [
          create_dict(count) for count in accumulator_dict["idf_counts"]
      ]
      idf_dict = dict(zip(accumulator_dict["idf_vocab"], idf_count_dicts))
      accumulator.per_doc_count_dict.update(idf_dict)

    return accumulator

  def _create_accumulator(self):
    """Accumulate a sorted array of vocab tokens and corresponding counts."""
    accumulator = collections.namedtuple(
        "Accumulator", ["count_dict", "per_doc_count_dict", "metadata"])

    count_dict = collections.defaultdict(int)
    if self._compute_idf:
      create_default_dict = lambda: {"count": 0, "last_doc_id": -1}
      per_doc_count_dict = collections.defaultdict(create_default_dict)
    else:
      per_doc_count_dict = None
    metadata = [0]
    return accumulator(count_dict, per_doc_count_dict, metadata)
