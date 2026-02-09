# coding=utf-8
# Copyright 2025 TF.Text Authors.
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

"""keras tokenization layers."""

import os

import tensorflow as tf

from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops.ragged import ragged_conversion_ops
from tensorflow_text.python.ops import unicode_script_tokenizer
from tensorflow_text.python.ops import whitespace_tokenizer
from tensorflow_text.python.ops import wordpiece_tokenizer


class TokenizerBase(tf.keras.layers.Layer):
  """Abstract Layer for tensorflow_text tokenizers.

  Input shape:
    N-D (2D as default) tensor with shape: `(batch_size, input_length)`.

  Output shape:
    (N+1)-D (3D as default) tensor with shape:
      `(batch_size, input_length, token_dim)`.
  """

  def __init__(self, tokenizer_instance, pad_value, squeeze_token_dim,
               **kwargs):
    if kwargs.get('dtype') is not None and kwargs.get('dtype') != 'string':
      raise ValueError('The only valid dtype for %s is string, but got: %s' %
                       (self.__class__.__name__, kwargs.get('dtype')))
    kwargs['dtype'] = 'string'
    super(TokenizerBase, self).__init__(**kwargs)

    self._tokenizer = tokenizer_instance
    self._pad_value = pad_value
    self._squeeze_token_dim = squeeze_token_dim

  def build(self, input_shape):
    # We have to use 'and not ==' here, because input_shape[1] !/== 1 can result
    # in None for undefined shape axes. If using 'and !=', this causes the
    # expression to evaluate to False instead of True if the shape is undefined;
    # the expression needs to evaluate to True in that case.
    if ((self._squeeze_token_dim) and (input_shape.ndims > 1) and
        (not input_shape[1] == 1)):  # pylint: disable=g-comparison-negation
      raise RuntimeError(
          '`squeeze_token_dim` should be set to False if you are calling this '
          'layer on a Tensor with inner dimension not equal to 1 (got '
          'input_shape: %s).' % input_shape)
    super(TokenizerBase, self).build(input_shape)

  def _set_tokenizer(self, tokenizer):
    self._tokenizer = tokenizer

  def call(self, text_to_be_tokenized):
    if self._squeeze_token_dim and (text_to_be_tokenized.shape.ndims > 1):
      text_to_be_tokenized = tf.compat.v1.squeeze(text_to_be_tokenized, axis=1)
    text = self._tokenizer.tokenize(text_to_be_tokenized)
    if self._pad_value is not None:
      text = ragged_conversion_ops.to_tensor(
          text, default_value=self._pad_value)
    return text

  def compute_output_shape(self, input_shape):
    """Computes output shape for the layer.

    Args:
      input_shape: Shape tuple (tuple of integers) or list of shape tuples (one
        per output tensor of the layer). Shape tuples can include None for free
        dimensions, instead of an integer.

    Returns:
      Computed output shape(s).
    """
    input_shape = tf.TensorShape(input_shape).as_list()
    shape = [dim for dim in input_shape]
    # because the output of tokenization is ragged, the added dimension should
    # be set as None
    shape.append(None)
    return tf.TensorShape(shape)

  def compute_output_signature(self, input_signature):
    """Compute the output tensor signature of the layer based on the inputs.

    Args:
      input_signature: Single TensorSpec or nested structure of TensorSpec
        objects, describing a candidate input for the layer.

    Returns:
      Single TensorSpec or nested structure of TensorSpec objects, describing
        how the layer would transform the provided input.

    Raises:
      TypeError: If input_signature contains a non-TensorSpec object.
    """

    def CheckAndReturnShape(s):
      if not isinstance(s, tf.TensorSpec):
        raise TypeError('Only TensorSpec signature types are supported, '
                        'but saw signature signature entry: {}.'.format(s))
      return s.shape

    input_shape = CheckAndReturnShape(input_signature)
    output_shape = self.compute_output_shape(input_shape)
    return tf.TensorSpec(dtype=tf.string, shape=output_shape)

  def get_config(self):
    config = {
        'pad_value': self._pad_value,
        'squeeze_token_dim': self._squeeze_token_dim
    }
    base_config = super(TokenizerBase, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package='Text')
class UnicodeScriptTokenizer(TokenizerBase):
  """Unicode script tokenization layer.

  Splits a string when successive tokens change their Unicode script or change
  being whitespace or not. By not keeping the whitespace tokens, this allows
  you to split on whitespace, and also to split out tokens from different
  scripts (so, for instance, a string with both Latin and Japanese characters
  would be split at the boundary of the Latin and Japanese characters in
  addition to any whitespace boundaries).

  Attributes:
    keep_whitespace: A boolean that specifices whether to emit whitespace
      tokens (default `False`).
    pad_value: if not None, performs the padding (using pad_value) at the
      inner-most dimension (i.e. token dimension) and outputs a padded dense
      tensor (default=None).
    squeeze_token_dim: Whether to squeeze the dimension added by tokenization.
      When this arg is set to False, the output will have an additional inner
      dimension added, containing the tokens in each string; when this arg is
      True, the layer will attempt to squeeze that dimension out. If you are
      passing one string per batch, you probably want to keep this as True; if
      you are passing more than one string per batch or are using this layer in
      a context like the Keras `TextVectorization` layer which expects a
      tf.strings.split()-stype output, this should be False. Defaults to True.
  """

  def __init__(self,
               keep_whitespace=False,
               pad_value=None,
               squeeze_token_dim=True,
               **kwargs):
    tokenizer_fn = unicode_script_tokenizer.UnicodeScriptTokenizer(
        keep_whitespace=keep_whitespace)
    super(UnicodeScriptTokenizer, self).__init__(
        tokenizer_instance=tokenizer_fn,
        squeeze_token_dim=squeeze_token_dim,
        pad_value=pad_value,
        **kwargs)


@tf.keras.utils.register_keras_serializable(package='Text')
class WhitespaceTokenizer(TokenizerBase):
  """Whitespace tokenization layer.

  Splits a string into substrings at ICU whitespace boundaries.

  Attributes:
    pad_value: if not None, performs the padding (using pad_value) at the
      inner-most dimension (i.e. token dimension) and outputs a padded dense
      tensor (default=None).
    squeeze_token_dim: Whether to squeeze the dimension added by tokenization.
      When this arg is set to False, the output will have an additional inner
      dimension added, containing the tokens in each string; when this arg is
      True, the layer will attempt to squeeze that dimension out. If you are
      passing one string per batch, you probably want to keep this as True; if
      you are passing more than one string per batch or are using this layer in
      a context like the Keras `TextVectorization` layer which expects a
      tf.strings.split()-stype output, this should be False. Defaults to True.
  """

  def __init__(self, pad_value=None, squeeze_token_dim=True, **kwargs):
    tokenizer_fn = whitespace_tokenizer.WhitespaceTokenizer()
    super(WhitespaceTokenizer, self).__init__(
        tokenizer_instance=tokenizer_fn,
        squeeze_token_dim=squeeze_token_dim,
        pad_value=pad_value,
        **kwargs)


@tf.keras.utils.register_keras_serializable(package='Text')
class WordpieceTokenizer(TokenizerBase):
  """Splits an already-tokenized tensor of tokens further into WordPiece tokens.

  Splits a set of string tokens into subwords as described in
  https://arxiv.org/pdf/1609.08144.pdf. This layer does not build the WordPiece
  vocabulary; instead, users should set the vocabulary by either passing it to
  the init call or by calling set_vocabulary() after the layer is constructed.

  Attributes:
    vocabulary: An optional list of vocabulary terms, or a path to a text file
      containing a vocabulary to load into this layer. The file should contain
      one token per line. If the list or file contains the same token multiple
      times, an error will be thrown.
    suffix_indicator: (optional) The characters prepended to a wordpiece to
      indicate that it is a suffix to another subword. Default is '##'.
    max_bytes_per_word: (optional) Max size of input token. Default is 100.
    token_out_type: (optional) The type of the token to return. This can be
      `tf.int64` IDs, or `tf.string` subwords. The default is `tf.int64`.
    unknown_token: (optional) The string value to substitute for an unknown
      token. Default is "[UNK]". If set to `None`, no substitution occurs.
      If `token_out_type` is `tf.int64`, the `vocabulary` is used (after
      substitution) to convert the unknown token to an integer, resulting in -1
      if `unknown_token` is set to `None` or not contained in the `vocabulary`.
    pad_value: if not None, performs the padding (using pad_value) at the
      inner-most dimension (i.e. token dimension) and outputs a padded dense
      tensor (default=None).
    merge_wordpiece_dim: If False, this layer will output a RaggedTensor
      with an additional inner 'wordpiece' dimension, containing the wordpieces
      for each token. If set to True, this layer will concatenate and squeeze
      along that dimension. Defaults to True.
  """

  def __init__(self,
               vocabulary=None,
               suffix_indicator='##',
               max_bytes_per_word=100,
               token_out_type=tf.string,
               unknown_token='[UNK]',
               pad_value=None,
               merge_wordpiece_dim=True,
               **kwargs):
    self._suffix_indicator = suffix_indicator
    self._max_bytes_per_word = max_bytes_per_word
    self._token_out_type = tf.dtypes.as_dtype(token_out_type)
    self._unknown_token = unknown_token
    self._merge_wordpiece_dim = merge_wordpiece_dim

    self._table = lookup_ops.MutableHashTable(
        key_dtype=tf.string, value_dtype=tf.int64, default_value=-1)

    tokenizer_instance = wordpiece_tokenizer.WordpieceTokenizer(
        vocab_lookup_table=self._table,
        suffix_indicator=self._suffix_indicator,
        max_bytes_per_word=self._max_bytes_per_word,
        token_out_type=self._token_out_type,
        unknown_token=self._unknown_token)

    super(WordpieceTokenizer, self).__init__(
        tokenizer_instance=tokenizer_instance,
        squeeze_token_dim=False,
        pad_value=pad_value,
        **kwargs)

    # We need to add the trackable after the superclass was called, since
    # it adds the table to a list that is created there.
    tracked_table = self._add_trackable(self._table, trainable=False)
    # This is a workaround for summary() on this layer. Because the table is
    # not mutable during training, the effective number of parameters (and so
    # the weight shape) is 0; we add this as an attr so that the parameter
    # counting code in the Model object doesn't throw an attribute error.
    tracked_table.shape = tf.TensorShape((0,))

    if vocabulary is not None:
      self.set_vocabulary(vocabulary)

  def set_vocabulary(self, vocab):
    if isinstance(vocab, (str, bytes)):
      vocab = _GetVocabularyFromFile(vocab)
    keys = tf.convert_to_tensor(vocab, dtype=tf.string)
    values = tf.range(len(vocab), dtype=tf.int64)
    op = self._table.insert(keys, values)
    if not tf.executing_eagerly():
      tf.compat.v1.get_default_session().run(op)

  def get_config(self):
    config = {
        'suffix_indicator': self._suffix_indicator,
        'max_bytes_per_word': self._max_bytes_per_word,
        'token_out_type': self._token_out_type.name,
        'unknown_token': self._unknown_token,
        'vocabulary': None,
    }
    base_config = super(WordpieceTokenizer, self).get_config()
    del base_config['squeeze_token_dim']
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs):
    wordpiece_tensor = super(WordpieceTokenizer, self).call(inputs)
    if self._merge_wordpiece_dim:
      wordpiece_tensor = tf.concat(wordpiece_tensor, -1)
    return wordpiece_tensor


def _GetVocabularyFromFile(vocabulary_path):
  """Read a vocabulary in from a file."""
  vocab = []
  with tf.io.gfile.GFile(vocabulary_path, 'r') as reader:
    while True:
      # Get the next line (incl. \n), and break if nothing is left to read.
      text = reader.readline()
      if not text:
        break

      # Convert the raw text and strip whitespace.
      if isinstance(text, str):
        token = text
      elif isinstance(text, bytes):
        token = text.decode('utf-8', 'ignore')
      token = token.rstrip(os.linesep)
      vocab.append(token)
  return vocab
