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

import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import base_preprocessing_layer
from tensorflow.python.keras.layers.preprocessing import category_encoding
from tensorflow.python.keras.layers.preprocessing import string_lookup
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_string_ops
from tensorflow.python.util.tf_export import keras_export

LOWER_AND_STRIP_PUNCTUATION = "lower_and_strip_punctuation"

SPLIT_ON_WHITESPACE = "whitespace"

TFIDF = category_encoding.TFIDF
INT = category_encoding.INT
BINARY = category_encoding.BINARY
COUNT = category_encoding.COUNT

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
# The number of documents / examples that each token appears in.
_ACCUMULATOR_DOCUMENT_COUNTS = "document_counts"
# The total number of documents / examples in the dataset.
_ACCUMULATOR_NUM_DOCUMENTS = "num_documents"


@keras_export(
    "keras.layers.experimental.preprocessing.TextVectorization", v1=[])
class TextVectorization(base_preprocessing_layer.CombinerPreprocessingLayer):
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

    1. standardize each sample (usually lowercasing + punctuation stripping)
    2. split each sample into substrings (usually words)
    3. recombine substrings into tokens (usually ngrams)
    4. index tokens (associate a unique int value with each token)
    5. transform each sample using this index, either into a vector of ints or
       a dense float vector.

  Some notes on passing Callables to customize splitting and normalization for
  this layer:

    1. Any callable can be passed to this Layer, but if you want to serialize
       this object you should only pass functions that are registered Keras
       serializables (see `tf.keras.utils.register_keras_serializable` for more
       details).
    2. When using a custom callable for `standardize`, the data received
       by the callable will be exactly as passed to this layer. The callable
       should return a tensor of the same shape as the input.
    3. When using a custom callable for `split`, the data received by the
       callable will have the 1st dimension squeezed out - instead of
       `[["string to split"], ["another string to split"]]`, the Callable will
       see `["string to split", "another string to split"]`. The callable should
       return a Tensor with the first dimension containing the split tokens -
       in this example, we should see something like `[["string", "to", "split],
       ["another", "string", "to", "split"]]`. This makes the callable site
       natively compatible with `tf.strings.split()`.

  Attributes:
    max_tokens: The maximum size of the vocabulary for this layer. If None,
      there is no cap on the size of the vocabulary. Note that this vocabulary
      contains 1 OOV token, so the effective number of tokens is `(max_tokens -
      1 - (1 if output == "int" else 0))`.
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
          token. When output == "int", 0 is reserved for masked locations;
          this reduces the vocab size to max_tokens-2 instead of max_tokens-1
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
    vocabulary: An optional list of vocabulary terms, or a path to a text file
      containing a vocabulary to load into this layer. The file should contain
      one token per line. If the list or file contains the same token multiple
      times, an error will be thrown.

  Example:
  This example instantiates a TextVectorization layer that lowercases text,
  splits on whitespace, strips punctuation, and outputs integer vocab indices.

  >>> text_dataset = tf.data.Dataset.from_tensor_slices(["foo", "bar", "baz"])
  >>> max_features = 5000  # Maximum vocab size.
  >>> max_len = 4  # Sequence length to pad the outputs to.
  >>> embedding_dims = 2
  >>>
  >>> # Create the layer.
  >>> vectorize_layer = TextVectorization(
  ...  max_tokens=max_features,
  ...  output_mode='int',
  ...  output_sequence_length=max_len)
  >>>
  >>> # Now that the vocab layer has been created, call `adapt` on the text-only
  >>> # dataset to create the vocabulary. You don't have to batch, but for large
  >>> # datasets this means we're not keeping spare copies of the dataset.
  >>> vectorize_layer.adapt(text_dataset.batch(64))
  >>>
  >>> # Create the model that uses the vectorize text layer
  >>> model = tf.keras.models.Sequential()
  >>>
  >>> # Start by creating an explicit input layer. It needs to have a shape of
  >>> # (1,) (because we need to guarantee that there is exactly one string
  >>> # input per batch), and the dtype needs to be 'string'.
  >>> model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
  >>>
  >>> # The first layer in our model is the vectorization layer. After this
  >>> # layer, we have a tensor of shape (batch_size, max_len) containing vocab
  >>> # indices.
  >>> model.add(vectorize_layer)
  >>>
  >>> # Now, the model can map strings to integers, and you can add an embedding
  >>> # layer to map these integers to learned embeddings.
  >>> input_data = [["foo qux bar"], ["qux baz"]]
  >>> model.predict(input_data)
  array([[2, 1, 4, 0],
         [1, 3, 0, 0]])

  Example:
  This example instantiates a TextVectorization layer by passing a list
  of vocabulary terms to the layer's __init__ method.

    input_array = np.array([["earth", "wind", "and", "fire"],
                            ["fire", "and", "earth", "michigan"]])
    expected_output = [[2, 3, 4, 5], [5, 4, 2, 1]]

    input_data = keras.Input(shape=(None,), dtype=dtypes.string)
    layer = get_layer_class()(
        max_tokens=None,
        standardize=None,
        split=None,
        output_mode=text_vectorization.INT,
        vocabulary=vocab_data)
    int_data = layer(input_data)
    model = keras.Model(inputs=input_data, outputs=int_data)

    output_dataset = model.predict(input_array)
  >>> vocab_data = ["earth", "wind", "and", "fire"]
  >>> max_len = 4  # Sequence length to pad the outputs to.
  >>>
  >>> # Create the layer, passing the vocab directly. You can also pass the
  >>> # vocabulary arg a path to a file containing one vocabulary word per
  >>> # line.
  >>> vectorize_layer = TextVectorization(
  ...  max_tokens=max_features,
  ...  output_mode='int',
  ...  output_sequence_length=max_len,
  ...  vocabulary=vocab_data)
  >>>
  >>> # Because we've passed the vocabulary directly, we don't need to adapt
  >>> # the layer - the vocabulary is already set. The vocabulary contains the
  >>> # padding token ('') and OOV token ('[UNK]') as well as the passed tokens.
  >>> vectorize_layer.get_vocabulary()
  ['', '[UNK]', 'earth', 'wind', 'and', 'fire']

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
               vocabulary=None,
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
        allowable_strings=(LOWER_AND_STRIP_PUNCTUATION),
        layer_name="TextVectorization",
        arg_name="standardize",
        allow_none=True,
        allow_callables=True)

    # 'split' must be one of (None, SPLIT_ON_WHITESPACE, callable)
    layer_utils.validate_string_arg(
        split,
        allowable_strings=(SPLIT_ON_WHITESPACE),
        layer_name="TextVectorization",
        arg_name="split",
        allow_none=True,
        allow_callables=True)

    # 'output_mode' must be one of (None, INT, COUNT, BINARY, TFIDF)
    layer_utils.validate_string_arg(
        output_mode,
        allowable_strings=(INT, COUNT, BINARY, TFIDF),
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

    # In INT mode, the zero value is reserved for padding (per Keras standard
    # padding approaches). In non-INT modes, there is no padding so we can set
    # the OOV value to zero instead of one.
    self._oov_value = 1 if output_mode == INT else 0

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
        combiner=None,
        **kwargs)
    base_preprocessing_layer._kpl_gauge.get_cell("V2").set("TextVectorization")

    mask_token = "" if output_mode in [None, INT] else None
    self._index_lookup_layer = self._get_index_lookup_class()(
        max_tokens=max_tokens, mask_token=mask_token, vocabulary=vocabulary)

    # If this layer is configured for string or integer output, we do not
    # create a vectorization layer (as the output is not vectorized).
    if self._output_mode in [None, INT]:
      self._vectorize_layer = None
    else:
      if max_tokens is not None and self._pad_to_max:
        max_elements = max_tokens
      else:
        max_elements = None
      self._vectorize_layer = self._get_vectorization_class()(
          max_tokens=max_elements, output_mode=self._output_mode)

  # These are V1/V2 shim points. There are V1 implementations in the V1 class.
  def _get_vectorization_class(self):
    return category_encoding.CategoryEncoding

  def _get_index_lookup_class(self):
    return string_lookup.StringLookup
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
      if len(input_shape) == 1:
        input_shape = tuple(input_shape) + (1,)
      return tensor_shape.TensorShape(input_shape)

    if self._output_mode == INT and self._split is not None:
      input_shape = list(input_shape)
      if len(input_shape) == 1:
        input_shape = input_shape + [self._output_sequence_length]
      else:
        input_shape[1] = self._output_sequence_length
      return tensor_shape.TensorShape(input_shape)

  def compute_output_signature(self, input_spec):
    output_shape = self.compute_output_shape(input_spec.shape.as_list())
    output_dtype = dtypes.int64 if self._output_mode == INT else K.floatx()
    return tensor_spec.TensorSpec(shape=output_shape, dtype=output_dtype)

  def adapt(self, data, reset_state=True):
    """Fits the state of the preprocessing layer to the dataset.

    Overrides the default adapt method to apply relevant preprocessing to the
    inputs before passing to the combiner.

    Arguments:
      data: The data to train on. It can be passed either as a tf.data Dataset,
        as a NumPy array, a string tensor, or as a list of texts.
      reset_state: Optional argument specifying whether to clear the state of
        the layer at the start of the call to `adapt`. This must be True for
        this layer, which does not support repeated calls to `adapt`.
    """
    if not reset_state:
      raise ValueError("TextVectorization does not support streaming adapts.")

    # Build the layer explicitly with the original data shape instead of relying
    # on an implicit call to `build` in the base layer's `adapt`, since
    # preprocessing changes the input shape.
    if isinstance(data, (list, tuple, np.ndarray)):
      data = ops.convert_to_tensor_v2_with_dispatch(data)

    if isinstance(data, ops.Tensor):
      if data.shape.rank == 1:
        data = array_ops.expand_dims(data, axis=-1)
      self.build(data.shape)
      preprocessed_inputs = self._preprocess(data)
    elif isinstance(data, dataset_ops.DatasetV2):
      # TODO(momernick): Replace this with a more V2-friendly API.
      shape = dataset_ops.get_legacy_output_shapes(data)
      if not isinstance(shape, tensor_shape.TensorShape):
        raise ValueError("The dataset passed to 'adapt' must contain a single "
                         "tensor value.")
      if shape.rank == 0:
        data = data.map(lambda tensor: array_ops.expand_dims(tensor, 0))
        shape = dataset_ops.get_legacy_output_shapes(data)
      if shape.rank == 1:
        data = data.map(lambda tensor: array_ops.expand_dims(tensor, -1))
      self.build(dataset_ops.get_legacy_output_shapes(data))
      preprocessed_inputs = data.map(self._preprocess)
    else:
      raise ValueError(
          "adapt() requires a Dataset or an array as input, got {}".format(
              type(data)))

    self._index_lookup_layer.adapt(preprocessed_inputs)
    if self._vectorize_layer:
      if isinstance(data, ops.Tensor):
        integer_data = self._index_lookup_layer(preprocessed_inputs)
      else:
        integer_data = preprocessed_inputs.map(self._index_lookup_layer)
      self._vectorize_layer.adapt(integer_data)

  def get_vocabulary(self):
    return self._index_lookup_layer.get_vocabulary()

  def get_config(self):
    # This does not include the 'vocabulary' arg, since if the vocab was passed
    # at init time it's now stored in variable state - we don't need to
    # pull it off disk again.
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
                     oov_df_value=None):
    """Sets vocabulary (and optionally document frequency) data for this layer.

    This method sets the vocabulary and DF data for this layer directly, instead
    of analyzing a dataset through 'adapt'. It should be used whenever the vocab
    (and optionally document frequency) information is already known. If
    vocabulary data is already present in the layer, this method will replace
    it.

    Arguments:
      vocab: An array of string tokens.
      df_data: An array of document frequency data. Only necessary if the layer
        output_mode is TFIDF.
      oov_df_value: The document frequency of the OOV token. Only necessary if
        output_mode is TFIDF.

    Raises:
      ValueError: If there are too many inputs, the inputs do not match, or
        input data is missing.
      RuntimeError: If the vocabulary cannot be set when this function is
        called. This happens when "binary", "count", and "tfidf" modes,
        if "pad_to_max_tokens" is False and the layer itself has already been
        called.
    """
    if self._output_mode != TFIDF and df_data is not None:
      raise ValueError("df_data should only be set if output_mode is TFIDF. "
                       "output_mode is %s." % self._output_mode)

    if (self._output_mode in [BINARY, COUNT, TFIDF] and self._called and
        not self._pad_to_max):
      raise RuntimeError(("When using TextVectorization in {mode} mode and "
                          "pad_to_max_tokens is False, the vocabulary cannot "
                          "be changed after the layer is "
                          "called.").format(mode=self._output_mode))

    self._index_lookup_layer.set_vocabulary(vocab)

    # When doing raw or integer output, we don't have a Vectorize layer to
    # manage. In this case, we can return directly.
    if self._output_mode in [None, INT]:
      return

    if not self._pad_to_max or self._max_tokens is None:
      num_tokens = self._index_lookup_layer.vocab_size()
      self._vectorize_layer.set_num_elements(num_tokens)

    if self._output_mode == TFIDF:
      if df_data is None:
        raise ValueError("df_data must be set if output_mode is TFIDF")
      if len(vocab) != len(df_data):
        raise ValueError("df_data must be the same length as vocab. "
                         "len(df_data) is %s, len(vocab) is %s" %
                         (len(vocab), len(df_data)))
      if oov_df_value is None:
        raise ValueError("You must pass an oov_df_value when output_mode is "
                         "TFIDF.")

      df_data = self._convert_to_ndarray(df_data)
      if not isinstance(oov_df_value, np.ndarray):
        oov_df_value = np.array([oov_df_value])
      df_data = np.insert(df_data, 0, oov_df_value)
      self._vectorize_layer.set_tfidf_data(df_data)

  def build(self, input_shape):
    # We have to use 'and not ==' here, because input_shape[1] !/== 1 can result
    # in None for undefined shape axes. If using 'and !=', this causes the
    # expression to evaluate to False instead of True if the shape is undefined;
    # the expression needs to evaluate to True in that case.
    if self._split is not None:
      if input_shape.ndims > 1 and not input_shape[-1] == 1:  # pylint: disable=g-comparison-negation
        raise RuntimeError(
            "When using TextVectorization to tokenize strings, the innermost "
            "dimension of the input array must be 1, got shape "
            "{}".format(input_shape))

    super(TextVectorization, self).build(input_shape)

  def _set_state_variables(self, updates):
    if not self.built:
      raise RuntimeError("_set_state_variables() must be called after build().")
    if self._output_mode == TFIDF:
      self.set_vocabulary(
          updates[_VOCAB_NAME],
          updates[_IDF_NAME],
          updates[_OOV_IDF_NAME])
    else:
      self.set_vocabulary(updates[_VOCAB_NAME])

  def _preprocess(self, inputs):
    if self._standardize == LOWER_AND_STRIP_PUNCTUATION:
      if tf_utils.is_ragged(inputs):
        lowercase_inputs = ragged_functional_ops.map_flat_values(
            gen_string_ops.string_lower, inputs)
        # Depending on configuration, we may never touch the non-data tensor
        # in the ragged inputs tensor. If that is the case, and this is the
        # only layer in the keras model, running it will throw an error.
        # To get around this, we wrap the result in an identity.
        lowercase_inputs = array_ops.identity(lowercase_inputs)
      else:
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
      if inputs.shape.ndims > 1:
        inputs = array_ops.squeeze(inputs, axis=-1)
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
    if isinstance(inputs, (list, tuple, np.ndarray)):
      inputs = ops.convert_to_tensor_v2_with_dispatch(inputs)

    self._called = True
    inputs = self._preprocess(inputs)

    # If we're not doing any output processing, return right away.
    if self._output_mode is None:
      return inputs
    indexed_data = self._index_lookup_layer(inputs)
    if self._output_mode == INT:
      # Once we have the dense tensor, we can return it if we weren't given a
      # fixed output sequence length. If we were, though, we have to dynamically
      # choose whether to pad or trim it based on each tensor.

      # We need to convert to dense if we have a ragged tensor.
      if tf_utils.is_ragged(indexed_data):
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
        output_tensor = control_flow_ops.cond(
            sequence_len < self._output_sequence_length,
            true_fn=pad_fn,
            false_fn=slice_fn)
        output_shape = output_tensor.shape.as_list()
        output_shape[-1] = self._output_sequence_length
        output_tensor.set_shape(tensor_shape.TensorShape(output_shape))
        return output_tensor

    # If we're not returning integers here, we rely on the vectorization layer
    # to create the output.
    return self._vectorize_layer(indexed_data)
