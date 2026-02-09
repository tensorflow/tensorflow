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
from tensorflow.python.framework import ops
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow_text.python.ops.tokenization import TokenizerWithOffsets

# pylint: disable=g-bad-import-order
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
gen_split_merge_from_logits_tokenizer = load_library.load_op_library(resource_loader.get_path_to_datafile('_split_merge_from_logits_tokenizer.so'))


_tf_text_split_merge_from_logits_tokenizer_op_create_counter = monitoring.Counter(
    '/nlx/api/python/split_merge_from_logits_tokenizer_create_counter',
    'Counter for number of SplitMergeFromLogitsTokenizer instances '
    'created in Python.')


class SplitMergeFromLogitsTokenizer(TokenizerWithOffsets):
  """Tokenizes a tensor of UTF-8 string into words according to logits."""

  def __init__(self, force_split_at_break_character=True):
    """Initializes a new instance.

    Args:
      force_split_at_break_character: a bool that indicates whether to force
        start a new word after an ICU-defined whitespace character.  Regardless
        of this parameter, we never include a whitespace into a token, and we
        always ignore the split/merge action for the whitespace character
        itself.  This parameter indicates what happens after a whitespace.
        * if force_split_at_break_character is true, create a new word starting
            at the first non-space character, regardless of the 0/1 label for
            that character, for instance:

            ```python
            s = [2.0, 1.0]  # sample pair of logits indicating a split action
            m = [1.0, 3.0]  # sample pair of logits indicating a merge action

            strings=["New York"]
            logits=[[s, m, m, s, m, m, m, m]]
            output tokens=[["New", "York"]]

            strings=["New York"]
            logits=[[s, m, m, m, m, m, m, m]]
            output tokens=[["New", "York"]]

            strings=["New York"],
            logits=[[s, m, m, m, s, m, m, m]]
            output tokens=[["New", "York"]]
            ```
        * otherwise, create a new word / continue the current one depending on
            the action for the first non-whitespace character.

            ```python
            s = [2.0, 1.0]  # sample pair of logits indicating a split action
            m = [1.0, 3.0]  # sample pair of logits indicating a merge action

            strings=["New York"],
            logits=[[s, m, m, s, m, m, m, m]]
            output tokens=[["NewYork"]]

            strings=["New York"],
            logits=[[s, m, m, m, m, m, m, m]]
            output tokens=[["NewYork"]]

            strings=["New York"],
            logits=[[s, m, m, m, s, m, m, m]]
            output tokens=[["New", "York"]]
            ```
    """
    super(SplitMergeFromLogitsTokenizer, self).__init__()
    self._force_split_at_break_character = force_split_at_break_character
    counter = _tf_text_split_merge_from_logits_tokenizer_op_create_counter
    counter.get_cell().increase_by(1)

  def tokenize(self, strings, logits):
    """Tokenizes a tensor of UTF-8 strings according to logits.

    The logits refer to the split / merge action we should take for each
    character.  For more info, see the doc for the logits argument below.

    ### Example:

    >>> strings = ['IloveFlume!', 'and tensorflow']
    >>> logits = [
    ... [
    ...     # 'I'
    ...     [5.0, -3.2],  # I: split
    ...     # 'love'
    ...     [2.2, -1.0],  # l: split
    ...     [0.2, 12.0],  # o: merge
    ...     [0.0, 11.0],  # v: merge
    ...     [-3.0, 3.0],  # e: merge
    ...     # 'Flume'
    ...     [10.0, 0.0],  # F: split
    ...     [0.0, 11.0],  # l: merge
    ...     [0.0, 11.0],  # u: merge
    ...     [0.0, 12.0],  # m: merge
    ...     [0.0, 12.0],  # e: merge
    ...     # '!'
    ...     [5.2, -7.0],  # !: split
    ...     # padding:
    ...     [1.0, 0.0], [1.0, 1.0], [1.0, 0.0],
    ... ], [
    ...     # 'and'
    ...     [2.0, 0.7],  # a: split
    ...     [0.2, 1.5],  # n: merge
    ...     [0.5, 2.3],  # d: merge
    ...     # ' '
    ...     [1.7, 7.0],  # <space>: merge
    ...     # 'tensorflow'
    ...     [2.2, 0.1],  # t: split
    ...     [0.2, 3.1],  # e: merge
    ...     [1.1, 2.5],  # n: merge
    ...     [0.7, 0.9],  # s: merge
    ...     [0.6, 1.0],  # o: merge
    ...     [0.3, 1.0],  # r: merge
    ...     [0.2, 2.2],  # f: merge
    ...     [0.7, 3.1],  # l: merge
    ...     [0.4, 5.0],  # o: merge
    ...     [0.8, 6.0],  # w: merge
    ... ]]
    >>> tokenizer = SplitMergeFromLogitsTokenizer()
    >>> tokenizer.tokenize(strings, logits)
    <tf.RaggedTensor [[b'I', b'love', b'Flume', b'!'], [b'and', b'tensorflow']]>

    Args:
      strings: a 1D `Tensor` of UTF-8 strings.
      logits: 3D Tensor; logits[i,j,0] is the logit for the split action for
        j-th character of strings[i].  logits[i,j,1] is the logit for the merge
        action for that same character.  For each character, we pick the action
        with the greatest logit.  Split starts a new word at this character and
        merge adds this character to the previous word.  The shape of this
        tensor should be (n, m, 2) where n is the number of strings, and m is
        greater or equal with the number of characters from each strings[i].  As
        the elements of the strings tensor may have different lengths (in UTF-8
        chars), padding may be required to get a dense vector; for each row, the
        extra (padding) pairs of logits are ignored.

    Returns:
      A `RaggedTensor` of strings where `tokens[i, k]` is the string
      content of the `k-th` token in `strings[i]`

    Raises:
      InvalidArgumentError: if one of the input Tensors has the wrong shape.
        E.g., if the logits tensor does not have enough elements for one of the
        strings.
    """
    subword, _, _ = self.tokenize_with_offsets(strings, logits)
    return subword

  def tokenize_with_offsets(self, strings, logits):
    """Tokenizes a tensor of UTF-8 strings into tokens with [start,end) offsets.

    ### Example:

    >>> strings = ['IloveFlume!', 'and tensorflow']
    >>> logits = [
    ... [
    ...     # 'I'
    ...     [5.0, -3.2],  # I: split
    ...     # 'love'
    ...     [2.2, -1.0],  # l: split
    ...     [0.2, 12.0],  # o: merge
    ...     [0.0, 11.0],  # v: merge
    ...     [-3.0, 3.0],  # e: merge
    ...     # 'Flume'
    ...     [10.0, 0.0],  # F: split
    ...     [0.0, 11.0],  # l: merge
    ...     [0.0, 11.0],  # u: merge
    ...     [0.0, 12.0],  # m: merge
    ...     [0.0, 12.0],  # e: merge
    ...     # '!'
    ...     [5.2, -7.0],  # !: split
    ...     # padding:
    ...     [1.0, 0.0], [1.0, 1.0], [1.0, 0.0],
    ... ], [
    ...     # 'and'
    ...     [2.0, 0.7],  # a: split
    ...     [0.2, 1.5],  # n: merge
    ...     [0.5, 2.3],  # d: merge
    ...     # ' '
    ...     [1.7, 7.0],  # <space>: merge
    ...     # 'tensorflow'
    ...     [2.2, 0.1],  # t: split
    ...     [0.2, 3.1],  # e: merge
    ...     [1.1, 2.5],  # n: merge
    ...     [0.7, 0.9],  # s: merge
    ...     [0.6, 1.0],  # o: merge
    ...     [0.3, 1.0],  # r: merge
    ...     [0.2, 2.2],  # f: merge
    ...     [0.7, 3.1],  # l: merge
    ...     [0.4, 5.0],  # o: merge
    ...     [0.8, 6.0],  # w: merge
    ... ]]
    >>> tokenizer = SplitMergeFromLogitsTokenizer()
    >>> tokens, starts, ends = tokenizer.tokenize_with_offsets(strings, logits)
    >>> tokens
    <tf.RaggedTensor [[b'I', b'love', b'Flume', b'!'], [b'and', b'tensorflow']]>
    >>> starts
    <tf.RaggedTensor [[0, 1, 5, 10], [0, 4]]>
    >>> ends
    <tf.RaggedTensor [[1, 5, 10, 11], [3, 14]]>

    Args:
      strings: A 1D `Tensor` of UTF-8 strings.
      logits: 3D Tensor; logits[i,j,0] is the logit for the split action for
        j-th character of strings[i].  logits[i,j,1] is the logit for the merge
        action for that same character.  For each character, we pick the action
        with the greatest logit.  Split starts a new word at this character and
        merge adds this character to the previous word.  The shape of this
        tensor should be (n, m, 2) where n is the number of strings, and m is
        greater or equal with the number of characters from each strings[i].  As
        the elements of the strings tensor may have different lengths (in UTF-8
        chars), padding may be required to get a dense vector; for each row, the
        extra (padding) pairs of logits are ignored.

    Returns:
      A tuple `(tokens, start_offsets, end_offsets)` where:
        * `tokens` is a `RaggedTensor` of strings where `tokens[i, k]` is
          the string content of the `k-th` token in `strings[i]`
        * `start_offsets` is a `RaggedTensor` of int64s where
          `start_offsets[i, k]` is the byte offset for the start of the
          `k-th` token in `strings[i]`.
        * `end_offsets` is a `RaggedTensor` of int64s where
          `end_offsets[i, k]` is the byte offset immediately after the
          end of the `k-th` token in `strings[i]`.

    Raises:
      InvalidArgumentError: if one of the input Tensors has the wrong shape.
        E.g., if the tensor logits does not have enough elements for one of the
        strings.
    """
    name = None
    with ops.name_scope(name, 'SplitMergeFromLogitsTokenizer',
                        [strings, logits]):
      # Tokenize the strings into tokens.
      force_split = self._force_split_at_break_character
      token_values, token_row_splits, start_values, end_values = (
          gen_split_merge_from_logits_tokenizer.tokenizer_from_logits(
              strings=strings,
              logits=logits,
              force_split_at_break_character=force_split))

      # Put token info into RaggedTensors, as indicated by token_row_splits.
      def put_token_info_into_ragged_tensor(token_info_values):
        return RaggedTensor.from_row_splits(
            token_info_values, token_row_splits, validate=False)

      tokens = put_token_info_into_ragged_tensor(token_values)
      start_offsets = put_token_info_into_ragged_tensor(start_values)
      end_offsets = put_token_info_into_ragged_tensor(end_values)
      return tokens, start_offsets, end_offsets
