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

"""Ops to tokenize words into phrases."""

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
from tensorflow_text.core.pybinds import pywrap_phrase_tokenizer_model_builder
from tensorflow_text.python.ops.tokenization import Detokenizer
from tensorflow_text.python.ops.tokenization import Tokenizer

# pylint: disable=g-bad-import-order
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
gen_phrase_tokenizer = load_library.load_op_library(resource_loader.get_path_to_datafile('_phrase_tokenizer.so'))

_tf_text_phrase_tokenizer_op_create_counter = monitoring.Counter(
    '/nlx/api/python/phrase_tokenizer_create_counter',
    'Counter for number of PhraseTokenizers created in Python.')


class PhraseTokenizer(Tokenizer, Detokenizer):
  """Tokenizes a tensor of UTF-8 string tokens into phrases."""

  def __init__(
      self,
      vocab=None,
      token_out_type=dtypes.int32,
      unknown_token='<UNK>',
      support_detokenization=True,
      prob=0,
      split_end_punctuation=False,
      model_buffer=None,
  ):
    """Initializes the PhraseTokenizer.

    Args:
      vocab: (optional) The list of tokens in the vocabulary.
      token_out_type: (optional) The type of the token to return. This can be
        `tf.int64` or `tf.int32` IDs, or `tf.string` subwords.
      unknown_token: (optional) The string value to substitute for an unknown
        token. It must be included in `vocab`.
      support_detokenization: (optional) Whether to make the tokenizer support
        doing detokenization. Setting it to true expands the size of the model
        flatbuffer.
      prob: Probability of emitting a phrase when there is a match.
      split_end_punctuation: Split the end punctuation.
      model_buffer: (optional) Bytes object (or a uint8 tf.Tenosr) that contains
        the phrase model in flatbuffer format (see phrase_tokenizer_model.fbs).
        If not `None`, all other arguments (except `token_output_type`) are
        ignored.
    """
    super().__init__()
    _tf_text_phrase_tokenizer_op_create_counter.get_cell().increase_by(1)

    if model_buffer is None:
      model_buffer = (
          pywrap_phrase_tokenizer_model_builder.build_phrase_model(
              vocab, unknown_token, support_detokenization, prob,
              split_end_punctuation))
    # Use uint8 tensor as a buffer for the model to avoid any possible changes,
    # for example truncation by '\0'.
    if isinstance(model_buffer, tensor.Tensor):
      self._model = model_buffer
    else:
      self._model = constant_op.constant(list(model_buffer), dtype=dtypes.uint8)

    self._token_out_type = token_out_type

  def tokenize(self, input):  # pylint: disable=redefined-builtin
    """Tokenizes a tensor of UTF-8 string tokens further into phrase tokens.

    ### Example, single string tokenization:
    >>> vocab = ["I", "have", "a", "dream", "a dream", "I have a", "<UNK>"]
    >>> tokenizer = PhraseTokenizer(vocab, token_out_type=tf.string)
    >>> tokens = [["I have a dream"]]
    >>> phrases = tokenizer.tokenize(tokens)
    >>> phrases
    <tf.RaggedTensor [[[b'I have a', b'dream']]]>

    Args:
      input: An N-dimensional `Tensor` or `RaggedTensor` of UTF-8 strings.

    Returns:
      tokens: is a `RaggedTensor`, where `tokens[i, j]` is the j-th token
          (i.e., phrase) for `input[i]` (i.e., the i-th input word). This
          token is either the actual token string content, or the corresponding
          integer id, i.e., the index of that token string in the vocabulary.
          This choice is controlled by the `token_out_type` parameter passed to
          the initializer method.
    """
    name = None
    with ops.name_scope(name, 'PhraseTokenize', [input, self._model]):
      # Check that the types are expected and the ragged rank is appropriate.
      tokens = ragged_tensor.convert_to_tensor_or_ragged_tensor(input)
      rank = tokens.shape.ndims
      if rank is None:
        raise ValueError('input must have a known rank.')

      if rank == 0:
        phrases = self.tokenize(array_ops_stack.stack([tokens]))
        return phrases.values

      elif rank > 1:
        if not ragged_tensor.is_ragged(tokens):
          tokens = ragged_tensor.RaggedTensor.from_tensor(
              tokens, ragged_rank=rank - 1)
        phrases = self.tokenize(tokens.flat_values)
        phrases = phrases.with_row_splits_dtype(tokens.row_splits.dtype)
        return tokens.with_flat_values(phrases)

      # Tokenize the tokens into phrases.
      subwords, phrase_ids, row_splits = (
          gen_phrase_tokenizer.phrase_tokenize(
              input_values=tokens, phrase_model=self._model))

      if self._token_out_type == dtypes.int64:
        values = math_ops.cast(phrase_ids, dtypes.int64)
      elif self._token_out_type == dtypes.int32:
        values = math_ops.cast(phrase_ids, dtypes.int32)
      else:
        values = subwords

      phrases = RaggedTensor.from_row_splits(values, row_splits, validate=False)

      return phrases

  def detokenize(self, input_t):  # pylint: disable=redefined-builtin
    """Detokenizes a tensor of int64 or int32 phrase ids into sentences.

    Detokenize and tokenize an input string returns itself when the input string
    is normalized and the tokenized phrases don't contain `<unk>`.

    ### Example:
    >>> vocab = ["I", "have", "a", "dream", "a dream", "I have a", "<UNK>"]
    >>> tokenizer = PhraseTokenizer(vocab, support_detokenization=True)
    >>> ids = tf.ragged.constant([[0, 1, 2], [5, 3]])
    >>> tokenizer.detokenize(ids)
    <tf.Tensor: shape=(2,), dtype=string,
    ...       numpy=array([b'I have a', b'I have a dream'], dtype=object)>

    Args:
      input_t: An N-dimensional `Tensor` or `RaggedTensor` of int64 or int32.

    Returns:
      A `RaggedTensor` of sentences that has N - 1 dimension when N > 1.
      Otherwise, a string tensor.
    """
    name = None
    with ops.name_scope(name, 'PhraseDetokenize', [input_t, self._model]):
      # Check that the types are expected and the ragged rank is appropriate.
      # ragged_tensor.convert_to_tensor_or_ragged_tensor(input)
      phrase_ids = math_ops.cast(input_t, dtypes.int32)
      rank = phrase_ids.shape.ndims
      if rank is None:
        raise ValueError('input must have a known rank.')

      if rank < 2:
        words = self.detokenize(array_ops_stack.stack([phrase_ids]))
        return words[0]

      if not ragged_tensor.is_ragged(phrase_ids):
        phrase_ids = ragged_tensor.RaggedTensor.from_tensor(
            phrase_ids, ragged_rank=rank - 1)
      nested_row_splits = phrase_ids.nested_row_splits
      # Detokenize the wordpiece ids to texts.
      words = (
          gen_phrase_tokenizer.tf_text_phrase_detokenize(
              input_values=phrase_ids.flat_values,
              input_row_splits=nested_row_splits[-1],
              phrase_model=self._model))
      words = RaggedTensor.from_nested_row_splits(
          words, nested_row_splits[:-1], validate=False)

      return words
