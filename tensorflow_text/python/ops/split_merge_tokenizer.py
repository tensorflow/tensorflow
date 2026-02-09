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
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow_text.python.ops.tokenization import TokenizerWithOffsets

# pylint: disable=g-bad-import-order
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
gen_split_merge_tokenizer = load_library.load_op_library(resource_loader.get_path_to_datafile('_split_merge_tokenizer.so'))

_tf_text_split_merge_tokenizer_op_create_counter = monitoring.Counter(
    '/nlx/api/python/split_merge_tokenizer_create_counter',
    'Counter for number of SplitMergeTokenizers created in Python.')


class SplitMergeTokenizer(TokenizerWithOffsets):
  """Tokenizes a tensor of UTF-8 string into words according to labels."""

  def __init__(self):
    """Initializes a new instance.
    """
    super(SplitMergeTokenizer, self).__init__()
    _tf_text_split_merge_tokenizer_op_create_counter.get_cell().increase_by(1)

  def tokenize(self,
               input,  # pylint: disable=redefined-builtin
               labels,
               force_split_at_break_character=True):
    """Tokenizes a tensor of UTF-8 strings according to labels.

    ### Example:

    >>> strings = ["HelloMonday", "DearFriday"]
    >>> labels = [[0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
    ...           [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]]
    >>> tokenizer = SplitMergeTokenizer()
    >>> tokenizer.tokenize(strings, labels)
    <tf.RaggedTensor [[b'Hello', b'Monday'], [b'Dear', b'Friday']]>

    Args:
      input: An N-dimensional `Tensor` or `RaggedTensor` of UTF-8 strings.
      labels: An (N+1)-dimensional `Tensor` or `RaggedTensor` of `int32`, with
        `labels[i1...iN, j]` being the split(0)/merge(1) label of the j-th
        character for `input[i1...iN]`.  Here split means create a new word with
        this character and merge means adding this character to the previous
        word.
      force_split_at_break_character: bool indicates whether to force start a
        new word after seeing a ICU defined whitespace character.  When seeing
        one or more ICU defined whitespace character:
        * if `force_split_at_break_character` is set true, then create a new
          word at the first non-space character, regardless of the label of that
          character, for instance:

          ```python
          input="New York"
          labels=[0, 1, 1, 0, 1, 1, 1, 1]
          output tokens=["New", "York"]
          ```

          ```python
          input="New York"
          labels=[0, 1, 1, 1, 1, 1, 1, 1]
          output tokens=["New", "York"]
          ```

          ```python
          input="New York",
          labels=[0, 1, 1, 1, 0, 1, 1, 1]
          output tokens=["New", "York"]
          ```

        * otherwise, whether to create a new word or not for the first non-space
          character depends on the label of that character, for instance:

          ```python
          input="New York",
          labels=[0, 1, 1, 0, 1, 1, 1, 1]
          output tokens=["NewYork"]
          ```

          ```python
          input="New York",
          labels=[0, 1, 1, 1, 1, 1, 1, 1]
          output tokens=["NewYork"]
          ```

          ```python
          input="New York",
          labels=[0, 1, 1, 1, 0, 1, 1, 1]
          output tokens=["New", "York"]
          ```

    Returns:
      A `RaggedTensor` of strings where `tokens[i1...iN, j]` is the string
      content of the `j-th` token in `input[i1...iN]`
    """
    subword, _, _ = self.tokenize_with_offsets(input, labels,
                                               force_split_at_break_character)
    return subword

  def tokenize_with_offsets(self,
                            input,  # pylint: disable=redefined-builtin
                            labels,
                            force_split_at_break_character=True):
    """Tokenizes a tensor of UTF-8 strings into tokens with [start,end) offsets.

    ### Example:

    >>> strings = ["HelloMonday", "DearFriday"]
    >>> labels = [[0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
    ...           [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]]
    >>> tokenizer = SplitMergeTokenizer()
    >>> tokens, starts, ends = tokenizer.tokenize_with_offsets(strings, labels)
    >>> tokens
    <tf.RaggedTensor [[b'Hello', b'Monday'], [b'Dear', b'Friday']]>
    >>> starts
    <tf.RaggedTensor [[0, 5], [0, 4]]>
    >>> ends
    <tf.RaggedTensor [[5, 11], [4, 10]]>

    Args:
      input: An N-dimensional `Tensor` or `RaggedTensor` of UTF-8 strings.
      labels: An (N+1)-dimensional `Tensor` or `RaggedTensor` of int32, with
        labels[i1...iN, j] being the split(0)/merge(1) label of the j-th
        character for input[i1...iN].  Here split means create a new word with
        this character and merge means adding this character to the previous
        word.
      force_split_at_break_character: bool indicates whether to force start a
        new word after seeing a ICU defined whitespace character.  When seeing
        one or more ICU defined whitespace character:
        * if `force_split_at_break_character` is set true, then create a new
          word at the first non-space character, regardless of the label of
          that character, for instance:

          ```python
          input="New York"
          labels=[0, 1, 1, 0, 1, 1, 1, 1]
          output tokens=["New", "York"]
          ```

          ```python
          input="New York"
          labels=[0, 1, 1, 1, 1, 1, 1, 1]
          output tokens=["New", "York"]
          ```

          ```python
          input="New York",
          labels=[0, 1, 1, 1, 0, 1, 1, 1]
          output tokens=["New", "York"]
          ```

        * otherwise, whether to create a new word or not for the first non-space
          character depends on the label of that character, for instance:

          ```python
          input="New York",
          labels=[0, 1, 1, 0, 1, 1, 1, 1]
          output tokens=["NewYork"]
          ```

          ```python
          input="New York",
          labels=[0, 1, 1, 1, 1, 1, 1, 1]
          output tokens=["NewYork"]
          ```

          ```python
          input="New York",
          labels=[0, 1, 1, 1, 0, 1, 1, 1]
          output tokens=["New", "York"]
          ```

    Returns:
      A tuple `(tokens, start_offsets, end_offsets)` where:

      tokens: is a `RaggedTensor` of strings where `tokens[i1...iN, j]` is
          the string content of the `j-th` token in `input[i1...iN]`
      start_offsets: is a `RaggedTensor` of int64s where
          `start_offsets[i1...iN, j]` is the byte offset for the start of the
          `j-th` token in `input[i1...iN]`.
      end_offsets: is a `RaggedTensor` of int64s where
          `end_offsets[i1...iN, j]` is the byte offset immediately after the
          end of the `j-th` token in `input[i...iN]`.
    """
    name = None
    with ops.name_scope(
        name, 'SplitMergeTokenizeWithOffsets',
        [input, labels, force_split_at_break_character]):
      # Check that the types are expected and the ragged rank is appropriate.
      tokens = ragged_tensor.convert_to_tensor_or_ragged_tensor(input)
      labels = ragged_tensor.convert_to_tensor_or_ragged_tensor(labels)
      rank = tokens.shape.ndims
      if rank is None:
        raise ValueError('input must have a known rank.')

      if rank == 0:
        words, starts, ends = self.tokenize_with_offsets(
            array_ops_stack.stack([tokens]),
            array_ops_stack.stack([labels]),
            force_split_at_break_character)
        return words.values, starts.values, ends.values

      elif rank > 1:
        if not ragged_tensor.is_ragged(tokens):
          tokens = ragged_tensor.RaggedTensor.from_tensor(
              tokens, ragged_rank=rank - 1)

        # Convert to a 2D ragged tensor from labels of shape
        # [#input_string, (labels per string)]
        if not ragged_tensor.is_ragged(labels):
          labels2d = array_ops.reshape(labels, [-1, labels.shape[-1]])
          labels_unpack = ragged_tensor.RaggedTensor.from_tensor(labels2d)
        else:
          labels_unpack = ragged_tensor.RaggedTensor.from_row_splits(
              values=labels.flat_values,
              row_splits=labels.nested_row_splits[-1])
        words, starts, ends = self.tokenize_with_offsets(
            tokens.flat_values,
            labels_unpack,
            force_split_at_break_character)
        words = words.with_row_splits_dtype(tokens.row_splits.dtype)
        starts = starts.with_row_splits_dtype(tokens.row_splits.dtype)
        ends = ends.with_row_splits_dtype(tokens.row_splits.dtype)
        return (tokens.with_flat_values(words),
                tokens.with_flat_values(starts),
                tokens.with_flat_values(ends))

      if not ragged_tensor.is_ragged(labels):
        ragged_labels = ragged_tensor.RaggedTensor.from_tensor(labels)
      else:
        ragged_labels = labels

      row_splits = math_ops.cast(ragged_labels.row_splits, dtypes.int32)

      # Tokenize the strings into tokens.
      values, row_splits, starts, ends = (
          gen_split_merge_tokenizer.split_merge_tokenize_with_offsets(
              input_values=tokens,
              labels=ragged_labels.flat_values,
              row_splits=row_splits,
              force_split_at_break_character=force_split_at_break_character))

      words = RaggedTensor.from_row_splits(values, row_splits, validate=False)
      starts = RaggedTensor.from_row_splits(starts, row_splits, validate=False)
      ends = RaggedTensor.from_row_splits(ends, row_splits, validate=False)
      return words, starts, ends
