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

"""Basic tokenization ops for BERT preprocessing."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy


from tensorflow.python.eager import monitoring
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import string_ops
from tensorflow_text.python.ops import regex_split_ops
from tensorflow_text.python.ops.normalize_ops import case_fold_utf8
from tensorflow_text.python.ops.normalize_ops import normalize_utf8
from tensorflow_text.python.ops.tokenization import Detokenizer
from tensorflow_text.python.ops.tokenization import TokenizerWithOffsets
from tensorflow_text.python.ops.wordpiece_tokenizer import WordpieceTokenizer

_tf_text_bert_tokenizer_op_create_counter = monitoring.Counter(
    "/nlx/api/python/bert_tokenizer_create_counter",
    "Counter for number of BertTokenizers created in Python.")

_DELIM_REGEX = [
    r"\s+",
    r"|".join([
        r"[!-/]",
        r"[:-@]",
        r"[\[-`]",
        r"[{-~]",
        r"[\p{P}]",
    ]),
    r"|".join([
        r"[\x{4E00}-\x{9FFF}]",
        r"[\x{3400}-\x{4DBF}]",
        r"[\x{20000}-\x{2A6DF}]",
        r"[\x{2A700}-\x{2B73F}]",
        r"[\x{2B740}-\x{2B81F}]",
        r"[\x{2B820}-\x{2CEAF}]",
        r"[\x{F900}-\x{FAFF}]",
        r"[\x{2F800}-\x{2FA1F}]",
    ]),
]

_DELIM_REGEX_PATTERN = "|".join(_DELIM_REGEX)
_KEEP_DELIM_NO_WHITESPACE = copy.deepcopy(_DELIM_REGEX)
_KEEP_DELIM_NO_WHITESPACE.remove(r"\s+")
_UNUSED_TOKEN_REGEX = "\\[unused\\d+\\]"
_KEEP_DELIM_NO_WHITESPACE_PATTERN = "|".join(_KEEP_DELIM_NO_WHITESPACE)


class BasicTokenizer(TokenizerWithOffsets):
  r"""Basic tokenizer for for tokenizing text.

  A basic tokenizer that tokenizes using some deterministic rules:
  - For most languages, this tokenizer will split on whitespace.
  - For Chinese, Japanese, and Korean characters, this tokenizer will split on
      Unicode characters.

  Example:
  >>> text_inputs =  [b'taste the rustisc indiefrost']
  >>> tokenizer = BasicTokenizer(
  ...     lower_case=False, normalization_form='NFC')
  >>> tokenizer.tokenize(text_inputs)
  <tf.RaggedTensor [[b'taste', b'the', b'rustisc', b'indiefrost']]>

  Attributes:
    lower_case: bool - If true, a preprocessing step is added to lowercase the
      text, which also applies NFD normalization and strip accents from
      characters.
    keep_whitespace: bool - If true, preserves whitespace characters instead of
      stripping them away.
    normalization_form: If set to a valid value and lower_case=False, the input
      text will be normalized to `normalization_form`. See normalize_utf8() op
      for a list of valid values.
    preserve_unused_token: If true, text in the regex format "\\[unused\\d+\\]"
      will be treated as a token and thus remain preserved as-is to be looked up
      in the vocabulary.
  """

  def __init__(self,
               lower_case=False,
               keep_whitespace=False,
               normalization_form=None,
               preserve_unused_token=False):
    self._lower_case = lower_case
    if not keep_whitespace:
      self._keep_delim_regex_pattern = _KEEP_DELIM_NO_WHITESPACE_PATTERN
    else:
      self._keep_delim_regex_pattern = _DELIM_REGEX_PATTERN

    if lower_case and normalization_form not in [None, "NFD"]:
      raise ValueError("`lower_case` strips accents. When `lower_case` is set, "
                       "`normalization_form` is 'NFD'.")
    self._normalization_form = normalization_form

    if preserve_unused_token:
      self._delim_regex_pattern = "|".join(
          [_UNUSED_TOKEN_REGEX, _DELIM_REGEX_PATTERN])
      self._keep_delim_regex_pattern = "|".join(
          [_UNUSED_TOKEN_REGEX, self._keep_delim_regex_pattern])
    else:
      self._delim_regex_pattern = _DELIM_REGEX_PATTERN

  def tokenize(self, text_input):
    tokens, _, _ = self.tokenize_with_offsets(text_input)
    return tokens

  def tokenize_with_offsets(self, text_input):
    """Performs basic word tokenization for BERT.

    Args:
      text_input: A `Tensor` or `RaggedTensor` of untokenized UTF-8 strings.

    Returns:
      A `RaggedTensor` of tokenized strings from text_input.
    """
    # lowercase and strip accents (if option is set)
    if self._lower_case:
      text_input = self.lower_case(text_input)
    else:
      # utf8 normalization
      if self._normalization_form is not None:
        text_input = normalize_utf8(text_input, self._normalization_form)

    # strip out control characters
    text_input = string_ops.regex_replace(text_input, r"\p{Cc}|\p{Cf}", " ")
    return regex_split_ops.regex_split_with_offsets(
        text_input, self._delim_regex_pattern, self._keep_delim_regex_pattern,
        "BertBasicTokenizer")

  def lower_case(self, text_input):
    """Lower-cases the `text_input'."""
    text_input = case_fold_utf8(text_input)
    text_input = normalize_utf8(text_input, "NFD")
    text_input = string_ops.regex_replace(text_input, r"\p{Mn}", "")
    return text_input


class AccentPreservingBasicTokenizer(BasicTokenizer):
  """I18n-friendly tokenizer that keeps accent characters during lowercasing."""

  def __init__(self, *args, **kwargs):
    super(AccentPreservingBasicTokenizer, self).__init__(*args, **kwargs)

  def lower_case(self, text_input):
    return string_ops.string_lower(text_input, encoding="utf-8")


class BertTokenizer(TokenizerWithOffsets, Detokenizer):
  r"""Tokenizer used for BERT.

    This tokenizer applies an end-to-end, text string to wordpiece tokenization.
    It first applies basic tokenization, followed by wordpiece
    tokenization.

    See `WordpieceTokenizer` for details on the subword tokenization.

    For an example of use, see
    https://www.tensorflow.org/text/guide/bert_preprocessing_guide

  Attributes:
    vocab_lookup_table: A lookup table implementing the LookupInterface
      containing the vocabulary of subwords or a string which is the file path
      to the vocab.txt file.
    suffix_indicator: (optional) The characters prepended to a wordpiece to
      indicate that it is a suffix to another subword. Default is '##'.
    max_bytes_per_word: (optional) Max size of input token. Default is 100.
    max_chars_per_token: (optional) Max size of subwords, excluding suffix
      indicator. If known, providing this improves the efficiency of decoding
      long words.
    token_out_type: (optional) The type of the token to return. This can be
      `tf.int64` IDs, or `tf.string` subwords. The default is `tf.int64`.
    unknown_token: (optional) The value to use when an unknown token is found.
      Default is "[UNK]". If this is set to a string, and `token_out_type` is
      `tf.int64`, the `vocab_lookup_table` is used to convert the
      `unknown_token` to an integer. If this is set to `None`, out-of-vocabulary
      tokens are left as is.
    split_unknown_characters: (optional) Whether to split out single unknown
      characters as subtokens. If False (default), words containing unknown
      characters will be treated as single unknown tokens.
    lower_case: bool - If true, a preprocessing step is added to lowercase the
      text, apply NFD normalization, and strip accents characters.
    keep_whitespace: bool - If true, preserves whitespace characters instead of
      stripping them away.
    normalization_form: If set to a valid value and lower_case=False, the input
      text will be normalized to `normalization_form`. See normalize_utf8() op
      for a list of valid values.
    preserve_unused_token: If true, text in the regex format `\\[unused\\d+\\]`
      will be treated as a token and thus remain preserved as is to be looked up
      in the vocabulary.
    basic_tokenizer_class: If set, the class to use instead of BasicTokenizer
  """

  def __init__(self,
               vocab_lookup_table,
               suffix_indicator="##",
               max_bytes_per_word=100,
               max_chars_per_token=None,
               token_out_type=dtypes.int64,
               unknown_token="[UNK]",
               split_unknown_characters=False,
               lower_case=False,
               keep_whitespace=False,
               normalization_form=None,
               preserve_unused_token=False,
               basic_tokenizer_class=BasicTokenizer):
    super(BertTokenizer, self).__init__()
    _tf_text_bert_tokenizer_op_create_counter.get_cell().increase_by(1)

    self._basic_tokenizer = basic_tokenizer_class(lower_case, keep_whitespace,
                                                  normalization_form,
                                                  preserve_unused_token)
    self._wordpiece_tokenizer = WordpieceTokenizer(
        vocab_lookup_table, suffix_indicator, max_bytes_per_word,
        max_chars_per_token, token_out_type, unknown_token,
        split_unknown_characters)

  def tokenize_with_offsets(self, text_input):
    r"""Tokenizes a tensor of string tokens into subword tokens for BERT.

    Example:
    >>> import pathlib
    >>> pathlib.Path('/tmp/tok_vocab.txt').write_text(
    ...     "they ##' ##re the great ##est".replace(' ', '\n'))
    >>> tokenizer = BertTokenizer(
    ...     vocab_lookup_table='/tmp/tok_vocab.txt')
    >>> text_inputs = tf.constant(['greatest'.encode('utf-8')])
    >>> tokenizer.tokenize_with_offsets(text_inputs)
    (<tf.RaggedTensor [[[4, 5]]]>,
     <tf.RaggedTensor [[[0, 5]]]>,
     <tf.RaggedTensor [[[5, 8]]]>)

    Args:
      text_input: input: A `Tensor` or `RaggedTensor` of untokenized UTF-8
        strings.

    Returns:
      A tuple of `RaggedTensor`s where the first element is the tokens where
      `tokens[i1...iN, j]`, the second element is the starting offsets, the
      third element is the end offset. (Please look at `tokenize` for details
      on tokens.)

    """
    tokens, begin, _ = self._basic_tokenizer.tokenize_with_offsets(text_input)
    wordpieces, wp_begin, wp_end = (
        self._wordpiece_tokenizer.tokenize_with_offsets(tokens))
    begin_expanded = array_ops.expand_dims(begin, axis=2)
    final_begin = begin_expanded + wp_begin
    final_end = begin_expanded + wp_end
    return wordpieces, final_begin, final_end

  def tokenize(self, text_input):
    r"""Tokenizes a tensor of string tokens into subword tokens for BERT.

    Example:
    >>> import pathlib
    >>> pathlib.Path('/tmp/tok_vocab.txt').write_text(
    ...     "they ##' ##re the great ##est".replace(' ', '\n'))
    >>> tokenizer = BertTokenizer(
    ...     vocab_lookup_table='/tmp/tok_vocab.txt')
    >>> text_inputs = tf.constant(['greatest'.encode('utf-8') ])
    >>> tokenizer.tokenize(text_inputs)
    <tf.RaggedTensor [[[4, 5]]]>

    Args:
      text_input: input: A `Tensor` or `RaggedTensor` of untokenized UTF-8
        strings.

    Returns:
      A `RaggedTensor` of tokens where `tokens[i1...iN, j]` is the string
      contents (or ID in the vocab_lookup_table representing that string)
      of the `jth` token in `input[i1...iN]`
    """
    tokens = self._basic_tokenizer.tokenize(text_input)
    return self._wordpiece_tokenizer.tokenize(tokens)

  def detokenize(self, token_ids):
    r"""Convert a `Tensor` or `RaggedTensor` of wordpiece IDs to string-words.

    See `WordpieceTokenizer.detokenize` for details.

    Note: `BertTokenizer.tokenize`/`BertTokenizer.detokenize` does not round
    trip losslessly. The result of `detokenize` will not, in general, have the
    same content or offsets as the input to `tokenize`. This is because the
    "basic tokenization" step, that splits the strings into words before
    applying the `WordpieceTokenizer`, includes irreversible
    steps like lower-casing and splitting on punctuation. `WordpieceTokenizer`
    on the other hand **is** reversible.

    Note: This method assumes wordpiece IDs are dense on the interval
    `[0, vocab_size)`.

    Example:
    >>> import pathlib
    >>> pathlib.Path('/tmp/tok_vocab.txt').write_text(
    ...    "they ##' ##re the great ##est".replace(' ', '\n'))
    >>> tokenizer = BertTokenizer(
    ...    vocab_lookup_table='/tmp/tok_vocab.txt')
    >>> text_inputs = tf.constant(['greatest'.encode('utf-8')])
    >>> tokenizer.detokenize([[4, 5]])
    <tf.RaggedTensor [[b'greatest']]>

    Args:
      token_ids: A `RaggedTensor` or `Tensor` with an int dtype.

    Returns:
      A `RaggedTensor` with dtype `string` and the same rank as the input
      `token_ids`.
    """
    return self._wordpiece_tokenizer.detokenize(token_ids)
