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

"""Ops to tokenize words into subwords."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import monitoring
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow_text.core.pybinds import pywrap_fast_wordpiece_tokenizer_model_builder
from tensorflow_text.python.ops.tokenization import Detokenizer
from tensorflow_text.python.ops.tokenization import TokenizerWithOffsets

# pylint: disable=g-bad-import-order
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
gen_fast_wordpiece_tokenizer = load_library.load_op_library(resource_loader.get_path_to_datafile('_fast_wordpiece_tokenizer.so'))

_tf_text_fast_wordpiece_tokenizer_op_create_counter = monitoring.Counter(
    '/nlx/api/python/fast_wordpiece_tokenizer_create_counter',
    'Counter for number of FastWordpieceTokenizers created in Python.')


class FastWordpieceTokenizer(TokenizerWithOffsets, Detokenizer):
  """Tokenizes a tensor of UTF-8 string tokens into subword pieces.

  It employs the linear (as opposed to quadratic) WordPiece algorithm (see the
  [paper](http://go/arxiv/2012.15524)).

  Differences compared to the classic
  [WordpieceTokenizer](https://www.tensorflow.org/text/api_docs/python/text/WordpieceTokenizer)
  are as follows (as of 11/2021):

    * `unknown_token` cannot be None or empty. That means if a word is too long
      or cannot be tokenized, FastWordpieceTokenizer always returns
      `unknown_token`. In constrast, the original
      [WordpieceTokenizer](https://www.tensorflow.org/text/api_docs/python/text/WordpieceTokenizer)
      would return the original word if `unknown_token` is empty or None.

    * `unknown_token` must be included in the vocabulary.

    * When `unknown_token` is returned, in tokenize_with_offsets(), the result
      end_offset is set to be the length of the original input word. In
      contrast, when `unknown_token` is returned by the original
      [WordpieceTokenizer](https://www.tensorflow.org/text/api_docs/python/text/WordpieceTokenizer),
      the end_offset is set to be the length of the `unknown_token` string.

    * `split_unknown_characters` is not supported.

    * `max_chars_per_token` is not used or needed.

    * By default the input is assumed to be general text (i.e., sentences), and
      FastWordpieceTokenizer first splits it on whitespaces and punctuations and
      then applies the Wordpiece tokenization (see the parameter
      `no_pretokenization`). If the input already contains single words only,
      please set `no_pretokenization=True` to be consistent with the classic
      [WordpieceTokenizer](https://www.tensorflow.org/text/api_docs/python/text/WordpieceTokenizer).

  """

  def __init__(self,
               vocab=None,
               suffix_indicator='##',
               max_bytes_per_word=100,
               token_out_type=dtypes.int64,
               unknown_token='[UNK]',
               no_pretokenization=False,
               support_detokenization=False,
               model_buffer=None):
    """Initializes the FastWordpieceTokenizer.

    Two ways to initialize:
      * (preferred) use a precompiled `model_buffer`.
      * use `vocab`, `suffix_indicator`, `max_bytes_per_word`, `unknown_token`,
        and `no_pretokenization`.

    Args:
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
      model_buffer: (optional) Bytes object (or a uint8 tf.Tenosr) that contains
        the wordpiece model in flatbuffer format (see
        fast_wordpiece_tokenizer_model.fbs). If not `None`, all other arguments
        (except `token_output_type`) are ignored.
    """
    super(FastWordpieceTokenizer, self).__init__()
    _tf_text_fast_wordpiece_tokenizer_op_create_counter.get_cell().increase_by(
        1)

    if model_buffer is None:
      model_buffer = (
          pywrap_fast_wordpiece_tokenizer_model_builder
          .build_fast_wordpiece_model(vocab, max_bytes_per_word,
                                      suffix_indicator, unknown_token,
                                      no_pretokenization,
                                      support_detokenization))
    # Use uint8 tensor as a buffer for the model to avoid any possible changes,
    # for example truncation by '\0'.
    if isinstance(model_buffer, tensor.Tensor):
      self._model = model_buffer
    else:
      self._model = constant_op.constant(list(model_buffer), dtype=dtypes.uint8)

    self._token_out_type = token_out_type

  def tokenize(self, input):  # pylint: disable=redefined-builtin
    """Tokenizes a tensor of UTF-8 string tokens further into subword tokens.

    ### Example 1, single word tokenization:
    >>> vocab = ["they", "##'", "##re", "the", "great", "##est", "[UNK]"]
    >>> tokenizer = FastWordpieceTokenizer(vocab, token_out_type=tf.string,
    ...                                    no_pretokenization=True)
    >>> tokens = [["they're", "the", "greatest"]]
    >>> tokenizer.tokenize(tokens)
    <tf.RaggedTensor [[[b'they', b"##'", b'##re'], [b'the'],
                       [b'great', b'##est']]]>

    ### Example 2, general text tokenization (pre-tokenization on
    ### punctuation and whitespace followed by WordPiece tokenization):
    >>> vocab = ["they", "##'", "##re", "the", "great", "##est", "[UNK]",
    ...          "'", "re"]
    >>> tokenizer = FastWordpieceTokenizer(vocab, token_out_type=tf.string)
    >>> tokens = [["they're the greatest", "the greatest"]]
    >>> tokenizer.tokenize(tokens)
    <tf.RaggedTensor [[[b'they', b"'", b're', b'the', b'great', b'##est'],
                       [b'the', b'great', b'##est']]]>

    Args:
      input: An N-dimensional `Tensor` or `RaggedTensor` of UTF-8 strings.

    Returns:
      A `RaggedTensor` of tokens where `tokens[i, j]` is the j-th token
      (i.e., wordpiece) for `input[i]` (i.e., the i-th input word). This token
      is either the actual token string content, or the corresponding integer
      id, i.e., the index of that token string in the vocabulary.  This choice
      is controlled by the `token_out_type` parameter passed to the initializer
      method.
    """
    # TODO(xysong): Optimize below by calling different overload kernels.
    subword, _, _ = self.tokenize_with_offsets(input)
    return subword

  def tokenize_with_offsets(self, input):  # pylint: disable=redefined-builtin
    """Tokenizes a tensor of UTF-8 string tokens further into subword tokens.

    ### Example 1, single word tokenization:
    >>> vocab = ["they", "##'", "##re", "the", "great", "##est", "[UNK]"]
    >>> tokenizer = FastWordpieceTokenizer(vocab, token_out_type=tf.string,
    ...                                    no_pretokenization=True)
    >>> tokens = [["they're", "the", "greatest"]]
    >>> subtokens, starts, ends = tokenizer.tokenize_with_offsets(tokens)
    >>> subtokens
    <tf.RaggedTensor [[[b'they', b"##'", b'##re'], [b'the'],
                       [b'great', b'##est']]]>
    >>> starts
    <tf.RaggedTensor [[[0, 4, 5], [0], [0, 5]]]>
    >>> ends
    <tf.RaggedTensor [[[4, 5, 7], [3], [5, 8]]]>

    ### Example 2, general text tokenization (pre-tokenization on
    ### punctuation and whitespace followed by WordPiece tokenization):
    >>> vocab = ["they", "##'", "##re", "the", "great", "##est", "[UNK]",
    ...          "'", "re"]
    >>> tokenizer = FastWordpieceTokenizer(vocab, token_out_type=tf.string)
    >>> tokens = [["they're the greatest", "the greatest"]]
    >>> subtokens, starts, ends = tokenizer.tokenize_with_offsets(tokens)
    >>> subtokens
    <tf.RaggedTensor [[[b'they', b"'", b're', b'the', b'great', b'##est'],
                       [b'the', b'great', b'##est']]]>
    >>> starts
    <tf.RaggedTensor [[[0, 4, 5, 8, 12, 17], [0, 4, 9]]]>
    >>> ends
    <tf.RaggedTensor [[[4, 5, 7, 11, 17, 20], [3, 9, 12]]]>

    Args:
      input: An N-dimensional `Tensor` or `RaggedTensor` of UTF-8 strings.

    Returns:
      A tuple `(tokens, start_offsets, end_offsets)` where:

      tokens: is a `RaggedTensor`, where `tokens[i, j]` is the j-th token
          (i.e., wordpiece) for `input[i]` (i.e., the i-th input word). This
          token is either the actual token string content, or the corresponding
          integer id, i.e., the index of that token string in the vocabulary.
          This choice is controlled by the `token_out_type` parameter passed to
          the initializer method.
      start_offsets[i1...iN, j]: is a `RaggedTensor` of the byte offsets
          for the inclusive start of the `jth` token in `input[i1...iN]`.
      end_offsets[i1...iN, j]: is a `RaggedTensor` of the byte offsets for
          the exclusive end of the `jth` token in `input[i`...iN]` (exclusive,
          i.e., first byte after the end of the token).
    """
    name = None
    with ops.name_scope(name, 'FastWordpieceTokenizeWithOffsets',
                        [input, self._model]):
      # Check that the types are expected and the ragged rank is appropriate.
      tokens = ragged_tensor.convert_to_tensor_or_ragged_tensor(input)
      rank = tokens.shape.ndims
      if rank is None:
        raise ValueError('input must have a known rank.')

      if rank == 0:
        wordpieces, starts, ends = self.tokenize_with_offsets(
            array_ops_stack.stack([tokens]))
        return wordpieces.values, starts.values, ends.values

      elif rank > 1:
        if not ragged_tensor.is_ragged(tokens):
          tokens = ragged_tensor.RaggedTensor.from_tensor(
              tokens, ragged_rank=rank - 1)
        wordpieces, starts, ends = self.tokenize_with_offsets(
            tokens.flat_values)
        wordpieces = wordpieces.with_row_splits_dtype(tokens.row_splits.dtype)
        starts = starts.with_row_splits_dtype(tokens.row_splits.dtype)
        ends = ends.with_row_splits_dtype(tokens.row_splits.dtype)
        return (tokens.with_flat_values(wordpieces),
                tokens.with_flat_values(starts), tokens.with_flat_values(ends))

      # Tokenize the tokens into subwords.
      # TODO(xysong): Optimize below by calling different overload kernels.
      subwords, subword_ids, row_splits, starts, ends = (
          gen_fast_wordpiece_tokenizer.fast_wordpiece_tokenize_with_offsets(
              input_values=tokens, wp_model=self._model))

      if self._token_out_type == dtypes.int64:
        values = math_ops.cast(subword_ids, dtypes.int64)
      elif self._token_out_type == dtypes.int32:
        values = math_ops.cast(subword_ids, dtypes.int32)
      else:
        values = subwords

      wordpieces = RaggedTensor.from_row_splits(
          values, row_splits, validate=False)
      starts = RaggedTensor.from_row_splits(starts, row_splits, validate=False)
      ends = RaggedTensor.from_row_splits(ends, row_splits, validate=False)

      return wordpieces, starts, ends

  def detokenize(self, input):  # pylint: disable=redefined-builtin
    """Detokenizes a tensor of int64 or int32 subword ids into sentences.

    Detokenize and tokenize an input string returns itself when the input string
    is normalized and the tokenized wordpieces don't contain `<unk>`.

    ### Example:
    >>> vocab = ["they", "##'", "##re", "the", "great", "##est", "[UNK]",
    ...          "'", "re", "ok"]
    >>> tokenizer = FastWordpieceTokenizer(vocab, support_detokenization=True)
    >>> ids = tf.ragged.constant([[0, 1, 2, 3, 4, 5], [9]])
    >>> tokenizer.detokenize(ids)
    <tf.Tensor: shape=(2,), dtype=string,
    ...         numpy=array([b"they're the greatest", b'ok'], dtype=object)>
    >>> ragged_ids = tf.ragged.constant([[[0, 1, 2, 3, 4, 5], [9]], [[4, 5]]])
    >>> tokenizer.detokenize(ragged_ids)
    <tf.RaggedTensor [[b"they're the greatest", b'ok'], [b'greatest']]>

    Args:
      input: An N-dimensional `Tensor` or `RaggedTensor` of int64 or int32.

    Returns:
      A `RaggedTensor` of sentences that has N - 1 dimension when N > 1.
      Otherwise, a string tensor.
    """
    name = None
    with ops.name_scope(name, 'FastWordpieceDetokenize', [input, self._model]):
      # Check that the types are expected and the ragged rank is appropriate.
      subword_ids = ragged_tensor.convert_to_tensor_or_ragged_tensor(input)
      subword_ids = math_ops.cast(subword_ids, dtypes.int32)
      rank = subword_ids.shape.ndims
      if rank is None:
        raise ValueError('input must have a known rank.')

      if rank < 2:
        words = self.detokenize(array_ops_stack.stack([subword_ids]))
        return words[0]

      if not ragged_tensor.is_ragged(subword_ids):
        subword_ids = ragged_tensor.RaggedTensor.from_tensor(
            subword_ids, ragged_rank=rank - 1)
      nested_row_splits = subword_ids.nested_row_splits
      # Detokenize the wordpiece ids to texts.
      words = (
          gen_fast_wordpiece_tokenizer.tf_text_fast_wordpiece_detokenize(
              input_values=subword_ids.flat_values,
              input_row_splits=nested_row_splits[-1],
              wp_model=self._model))
      words = RaggedTensor.from_nested_row_splits(
          words, nested_row_splits[:-1], validate=False)

      return words
