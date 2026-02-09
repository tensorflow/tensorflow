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

"""Fast BERT tokenization with TFLite support."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import monitoring
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow_text.python.ops.fast_bert_normalizer import FastBertNormalizer
from tensorflow_text.python.ops.fast_wordpiece_tokenizer import FastWordpieceTokenizer
from tensorflow_text.python.ops.tokenization import Detokenizer
from tensorflow_text.python.ops.tokenization import TokenizerWithOffsets

_tf_text_fast_bert_tokenizer_op_create_counter = monitoring.Counter(
    '/nlx/api/python/fast_bert_tokenizer_create_counter',
    'Counter for number of FastBertTokenizers created in Python.')


class FastBertTokenizer(TokenizerWithOffsets, Detokenizer):
  r"""Tokenizer used for BERT, a faster version with TFLite support.

    This tokenizer applies an end-to-end, text string to wordpiece tokenization.
    It is equivalent to `BertTokenizer` for most common scenarios while running
    faster and supporting TFLite. It does not support certain special settings
    (see the docs below).

    See `WordpieceTokenizer` for details on the subword tokenization.

    For an example of use, see
    https://www.tensorflow.org/text/guide/bert_preprocessing_guide

  Attributes:
    vocab: (optional) The list of tokens in the vocabulary.
    suffix_indicator: (optional) The characters prepended to a wordpiece to
      indicate that it is a suffix to another subword.
    max_bytes_per_word: (optional) Max size of input token.
    token_out_type: (optional) The type of the token to return. This can be
      `tf.int64` or `tf.int32` IDs, or `tf.string` subwords.
    unknown_token: (optional) The string value to substitute for an unknown
      token. It must be included in `vocab`.
    no_pretokenization: (optional) By default, the input is split on
      whitespaces and punctuations before applying the Wordpiece tokenization.
      When true, the input is assumed to be pretokenized already.
    support_detokenization: (optional) Whether to make the tokenizer support
      doing detokenization. Setting it to true expands the size of the model
      flatbuffer. As a reference, when using 120k multilingual BERT WordPiece
      vocab, the flatbuffer's size increases from ~5MB to ~6MB.
    fast_wordpiece_model_buffer: (optional) Bytes object (or a uint8 tf.Tenosr)
      that contains the wordpiece model in flatbuffer format (see
      fast_wordpiece_tokenizer_model.fbs). If not `None`, all other arguments
      related to FastWordPieceTokenizer (except `token_output_type`) are
      ignored.
    lower_case_nfd_strip_accents: (optional) .
      - If true, it first lowercases the text, applies NFD normalization, strips
      accents characters, and then replaces control characters with whitespaces.
      - If false, it only replaces control characters with whitespaces.
    fast_bert_normalizer_model_buffer: (optional) bytes object (or a uint8
      tf.Tenosr) that contains the fast bert normalizer model in flatbuffer
      format (see fast_bert_normalizer_model.fbs). If not `None`,
      `lower_case_nfd_strip_accents` is ignored.
  """

  def __init__(
      self,
      vocab=None,
      suffix_indicator='##',
      max_bytes_per_word=100,
      token_out_type=dtypes.int64,
      unknown_token='[UNK]',
      no_pretokenization=False,
      support_detokenization=False,
      fast_wordpiece_model_buffer=None,
      lower_case_nfd_strip_accents=False,
      fast_bert_normalizer_model_buffer=None,
  ):
    super(FastBertTokenizer, self).__init__()
    _tf_text_fast_bert_tokenizer_op_create_counter.get_cell().increase_by(1)

    self._fast_bert_normalizer = FastBertNormalizer(
        lower_case_nfd_strip_accents=lower_case_nfd_strip_accents,
        model_buffer=fast_bert_normalizer_model_buffer)
    self._fast_wordpiece_tokenizer = FastWordpieceTokenizer(
        vocab,
        suffix_indicator,
        max_bytes_per_word,
        token_out_type,
        unknown_token,
        no_pretokenization,
        support_detokenization,
        model_buffer=fast_wordpiece_model_buffer)

  def tokenize_with_offsets(self, text_input):
    r"""Tokenizes a tensor of string tokens into subword tokens for BERT.

    Example:
    >>> vocab = ['they', "##'", '##re', 'the', 'great', '##est', '[UNK]']
    >>> tokenizer = FastBertTokenizer(vocab=vocab)
    >>> text_inputs = tf.constant(['greatest'.encode('utf-8')])
    >>> tokenizer.tokenize_with_offsets(text_inputs)
    (<tf.RaggedTensor [[4, 5]]>,
     <tf.RaggedTensor [[0, 5]]>,
     <tf.RaggedTensor [[5, 8]]>)

    Args:
      text_input: input: A `Tensor` or `RaggedTensor` of untokenized UTF-8
        strings.

    Returns:
      A tuple of `RaggedTensor`s where the first element is the tokens where
      `tokens[i1...iN, j]`, the second element is the starting offsets, the
      third element is the end offset. (Please look at `tokenize` for details
      on tokens.)

    """
    normalized_input, offsets = self._fast_bert_normalizer.normalize_with_offsets(
        text_input)
    wordpieces, post_norm_offsets_starts, post_norm_offsets_ends = (
        self._fast_wordpiece_tokenizer.tokenize_with_offsets(normalized_input))
    pre_norm_offsets_starts = array_ops.gather(
        offsets, post_norm_offsets_starts, axis=-1, batch_dims=-1)
    pre_norm_offsets_ends = array_ops.gather(
        offsets, post_norm_offsets_ends, axis=-1, batch_dims=-1)
    return wordpieces, pre_norm_offsets_starts, pre_norm_offsets_ends

  def tokenize(self, text_input):
    r"""Tokenizes a tensor of string tokens into subword tokens for BERT.

    Example:
    >>> vocab = ['they', "##'", '##re', 'the', 'great', '##est', '[UNK]']
    >>> tokenizer = FastBertTokenizer(vocab=vocab)
    >>> text_inputs = tf.constant(['greatest'.encode('utf-8') ])
    >>> tokenizer.tokenize(text_inputs)
    <tf.RaggedTensor [[4, 5]]>

    Args:
      text_input: input: A `Tensor` or `RaggedTensor` of untokenized UTF-8
        strings.

    Returns:
      A `RaggedTensor` of tokens where `tokens[i1...iN, j]` is the string
      contents (or ID in the vocab_lookup_table representing that string)
      of the `jth` token in `input[i1...iN]`
    """
    normalized_input = self._fast_bert_normalizer.normalize(text_input)
    return self._fast_wordpiece_tokenizer.tokenize(normalized_input)

  def detokenize(self, token_ids):
    r"""Convert a `Tensor` or `RaggedTensor` of wordpiece IDs to string-words.

    See `WordpieceTokenizer.detokenize` for details.

    Note: `FastBertTokenizer.tokenize`/`FastBertTokenizer.detokenize` does not
    round trip losslessly. The result of `detokenize` will not, in general, have
    the same content or offsets as the input to `tokenize`. This is because the
    "basic tokenization" step, that splits the strings into words before
    applying the `WordpieceTokenizer`, includes irreversible steps like
    lower-casing and splitting on punctuation. `WordpieceTokenizer` on the other
    hand **is** reversible.

    Note: This method assumes wordpiece IDs are dense on the interval
    `[0, vocab_size)`.

    Example:
    >>> vocab = ['they', "##'", '##re', 'the', 'great', '##est', '[UNK]']
    >>> tokenizer = FastBertTokenizer(vocab=vocab, support_detokenization=True)
    >>> tokenizer.detokenize([[4, 5]])
    <tf.Tensor: shape=(1,), dtype=string, numpy=array([b'greatest'],
    dtype=object)>

    Args:
      token_ids: A `RaggedTensor` or `Tensor` with an int dtype.

    Returns:
      A `RaggedTensor` with dtype `string` and the same rank as the input
      `token_ids`.
    """
    return self._fast_wordpiece_tokenizer.detokenize(token_ids)
