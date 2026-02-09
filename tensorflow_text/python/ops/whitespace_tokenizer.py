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

"""Whitespace tokenizer for string tensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import monitoring
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops.ragged import ragged_conversion_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow_text.core.pybinds import pywrap_whitespace_tokenizer_config_builder
from tensorflow_text.python.ops.tokenization import TokenizerWithOffsets

# pylint: disable=g-bad-import-order,unused-import
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
gen_whitespace_tokenizer = load_library.load_op_library(resource_loader.get_path_to_datafile('_whitespace_tokenizer.so'))
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
gen_whitespace_tokenizer_v2 = load_library.load_op_library(resource_loader.get_path_to_datafile('_whitespace_tokenizer_v2.so'))

_tf_text_whitespace_tokenizer_op_create_counter = monitoring.Counter(
    "/nlx/api/python/whitespace_tokenizer_create_counter",
    "Counter for number of WhitespaceTokenizers created in Python.")


class WhitespaceTokenizer(TokenizerWithOffsets):
  """Tokenizes a tensor of UTF-8 strings on whitespaces."""

  def __init__(self):
    """Initializes the WhitespaceTokenizer.
    """
    super(WhitespaceTokenizer, self).__init__()
    self._config = (pywrap_whitespace_tokenizer_config_builder.
                    build_whitespace_tokenizer_config())
    _tf_text_whitespace_tokenizer_op_create_counter.get_cell().increase_by(1)

  def tokenize(self, input):  # pylint: disable=redefined-builtin
    """Tokenizes a tensor of UTF-8 strings on whitespaces.

    The strings are split on ICU defined whitespace characters. These
    whitespace characters are dropped.

    Example:

    >>> WhitespaceTokenizer().tokenize("small medium large")
    <tf.Tensor: shape=(3,), dtype=string, numpy=array([b'small', b'medium',
    b'large'], dtype=object)>

    Args:
      input: A `RaggedTensor` or `Tensor` of UTF-8 strings with any shape.

    Returns:
      A `RaggedTensor` of tokenized text. The returned shape is the shape of the
      input tensor with an added ragged dimension for tokens of each string.
    """
    (tokens, _, _) = self.tokenize_with_offsets(input)
    return tokens

  def tokenize_with_offsets(self, input):  # pylint: disable=redefined-builtin
    """Tokenizes a tensor of UTF-8 strings on whitespaces.

    The strings are split on ICU defined whitespace characters. These
    whitespace characters are dropped.

    Example:

    >>> splitter = WhitespaceTokenizer()
    >>> pieces, starts, ends = splitter.tokenize_with_offsets("a bb ccc")
    >>> print(pieces.numpy(), starts.numpy(), ends.numpy())
    [b'a' b'bb' b'ccc'] [0 2 5] [1 4 8]

    Args:
      input: A `RaggedTensor`or `Tensor` of UTF-8 strings with any shape.

    Returns:
      A tuple `(tokens, start_offsets, end_offsets)` where:

        * `tokens`: A `RaggedTensor` of tokenized text.
        * `start_offsets`: A `RaggedTensor` of the tokens' starting byte offset.
        * `end_offsets`: A `RaggedTensor` of the tokens' ending byte offset.
    """
    name = None
    with ops.name_scope(name, "WhitespaceTokenize", [input]):
      input_tensor = ragged_tensor.convert_to_tensor_or_ragged_tensor(input)
      if input_tensor.shape.ndims is None:
        raise ValueError("Rank of input_tensor must be statically known.")
      if ragged_tensor.is_ragged(input_tensor):
        if input_tensor.flat_values.shape.ndims > 1:
          # If the flat_values of our ragged tensor is multi-dimensional, we can
          # process it separately and our output will have the same nested
          # splits as our input.
          (tokens, starts,
           ends) = self.tokenize_with_offsets(input_tensor.flat_values)
          return (input_tensor.with_flat_values(tokens),
                  input_tensor.with_flat_values(starts),
                  input_tensor.with_flat_values(ends))
        else:
          # Recursively process the values of the ragged tensor.
          (tokens, starts,
           ends) = self.tokenize_with_offsets(input_tensor.values)
          return (input_tensor.with_values(tokens),
                  input_tensor.with_values(starts),
                  input_tensor.with_values(ends))
      else:
        if input_tensor.shape.ndims > 1:
          # Convert the input tensor to ragged and process it.
          return self.tokenize_with_offsets(
              ragged_conversion_ops.from_tensor(input_tensor))
        elif input_tensor.shape.ndims == 0:
          (tokens, starts, ends) = self.tokenize_with_offsets(
              array_ops_stack.stack([input_tensor]))
          return tokens.values, starts.values, ends.values
        else:
          # Our rank 1 tensor is the correct shape, so we can process it as
          # normal.
          return self._whitespace_tokenize_with_offsets(input_tensor)

  def _whitespace_tokenize_with_offsets(self, input_tensor):
    """Tokenizes a tensor of codepoints with rank of 1.

    Args:
      input_tensor: Single-dimension Tensor of strings to tokenize.

    Returns:
      Tuple of tokenized codepoints with offsets relative to the codepoints have
      a shape of [num_strings, (num_tokens or num_offsets)].
    """
    (values, row_splits, start_offsets, end_offsets) = (
        gen_whitespace_tokenizer_v2.tf_text_whitespace_tokenize_with_offsets_v2(
            input_values=input_tensor, input_config=self._config))
    values = RaggedTensor.from_nested_row_splits(
        flat_values=values,
        nested_row_splits=[row_splits])
    start_offsets = RaggedTensor.from_nested_row_splits(
        flat_values=start_offsets,
        nested_row_splits=[row_splits])
    end_offsets = RaggedTensor.from_nested_row_splits(
        flat_values=end_offsets,
        nested_row_splits=[row_splits])
    return (values, start_offsets, end_offsets)
