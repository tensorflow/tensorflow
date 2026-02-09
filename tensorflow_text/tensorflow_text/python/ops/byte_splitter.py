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

"""Byte splitter for string tensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import monitoring
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops.ragged import ragged_conversion_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow_text.python.ops.tokenization import SplitterWithOffsets

# pylint: disable=g-bad-import-order,unused-import
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
gen_byte_splitter = load_library.load_op_library(resource_loader.get_path_to_datafile('_byte_splitter.so'))

_tf_text_byte_splitter_op_create_counter = monitoring.Counter(
    "/nlx/api/python/byte_splitter_create_counter",
    "Counter for number of ByteSplitters created in Python.")


class ByteSplitter(SplitterWithOffsets):
  """Splits a string tensor into bytes."""

  def __init__(self):
    """Initializes the ByteSplitter.
    """
    super(ByteSplitter, self).__init__()
    _tf_text_byte_splitter_op_create_counter.get_cell().increase_by(1)

  def split(self, input):  # pylint: disable=redefined-builtin
    """Splits a string tensor into bytes.

    The strings are split bytes. Thus, some unicode characters may be split
    into multiple bytes.

    Example:

    >>> ByteSplitter().split("hello")
    <tf.Tensor: shape=(5,), dtype=uint8, numpy=array([104, 101, 108, 108, 111],
    dtype=uint8)>

    Args:
      input: A `RaggedTensor` or `Tensor` of strings with any shape.

    Returns:
      A `RaggedTensor` of bytes. The returned shape is the shape of the
      input tensor with an added ragged dimension for the bytes that make up
      each string.
    """
    (bytez, _, _) = self.split_with_offsets(input)
    return bytez

  def split_with_offsets(self, input):  # pylint: disable=redefined-builtin
    """Splits a string tensor into bytes.

    The strings are split bytes. Thus, some unicode characters may be split
    into multiple bytes.

    Example:

    >>> splitter = ByteSplitter()
    >>> bytes, starts, ends = splitter.split_with_offsets("hello")
    >>> print(bytes.numpy(), starts.numpy(), ends.numpy())
    [104 101 108 108 111] [0 1 2 3 4] [1 2 3 4 5]

    Args:
      input: A `RaggedTensor` or `Tensor` of strings with any shape.

    Returns:
      A `RaggedTensor` of bytes. The returned shape is the shape of the
      input tensor with an added ragged dimension for the bytes that make up
      each string.

    Returns:
      A tuple `(bytes, offsets)` where:

        * `bytes`: A `RaggedTensor` of bytes.
        * `start_offsets`: A `RaggedTensor` of the bytes' starting byte offset.
        * `end_offsets`: A `RaggedTensor` of the bytes' ending byte offset.
    """
    name = None
    with ops.name_scope(name, "ByteSplitter", [input]):
      input_tensor = ragged_tensor.convert_to_tensor_or_ragged_tensor(input)
      if input_tensor.shape.ndims is None:
        raise ValueError("Rank of input_tensor must be statically known.")
      if ragged_tensor.is_ragged(input_tensor):
        if input_tensor.flat_values.shape.ndims > 1:
          # If the flat_values of our ragged tensor is multi-dimensional, we can
          # process it separately and our output will have the same nested
          # splits as our input.
          (bytez, start_offsets, end_offsets) = self.split_with_offsets(
              input_tensor.flat_values)
          return (input_tensor.with_flat_values(bytez),
                  input_tensor.with_flat_values(start_offsets),
                  input_tensor.with_flat_values(end_offsets))
        else:
          # Recursively process the values of the ragged tensor.
          (bytez, start_offsets, end_offsets) = self.split_with_offsets(
              input_tensor.values)
          return (input_tensor.with_values(bytez),
                  input_tensor.with_values(start_offsets),
                  input_tensor.with_values(end_offsets))
      else:
        if input_tensor.shape.ndims > 1:
          # Convert the input tensor to ragged and process it.
          return self.split_with_offsets(
              ragged_conversion_ops.from_tensor(input_tensor))
        elif input_tensor.shape.ndims == 0:
          (bytez, start_offsets, end_offsets) = self.split_with_offsets(
              array_ops_stack.stack([input_tensor]))
          return bytez.values, start_offsets.values, end_offsets.values
        else:
          # Our rank 1 tensor is the correct shape, so we can process it as
          # normal.
          return self._byte_split_with_offsets(input_tensor)

  def _byte_split_with_offsets(self, input_tensor):
    """Splits a tensor of strings into bytes.

    Args:
      input_tensor: Single-dimension Tensor of strings to split.

    Returns:
      Tuple of tokenized codepoints with offsets relative to the codepoints have
      a shape of [num_strings, (num_tokens or num_offsets)].
    """
    (values, row_splits, start_offsets, end_offsets) = (
        gen_byte_splitter.tf_text_byte_split_with_offsets(
            input_values=input_tensor))
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

  def split_by_offsets(self, input, start_offsets, end_offsets):  # pylint: disable=redefined-builtin
    """Splits a string tensor into sub-strings.

    The strings are split based upon the provided byte offsets.

    Example:

    >>> splitter = ByteSplitter()
    >>> substrings = splitter.split_by_offsets("hello", [0, 4], [4, 5])
    >>> print(substrings.numpy())
    [b'hell' b'o']

    Args:
      input: `Tensor` or `RaggedTensor` of strings of any shape to split.
      start_offsets: `Tensor` or `RaggedTensor` of byte offsets to start splits
          on (inclusive). This should be one more than the rank of `input`.
      end_offsets: `Tensor` or `RaggedTensor` of byte offsets to end splits
          on (exclusive). This should be one more than the rank of `input`.

    Returns:
      A `RaggedTensor` or `Tensor` of substrings. The returned shape is the
      shape of the offsets.
    """
    name = None
    with ops.name_scope(name, "ByteSplitByOffsets",
                        [input, start_offsets, end_offsets]):
      input_tensor = ragged_tensor.convert_to_tensor_or_ragged_tensor(input)
      starts = ragged_tensor.convert_to_tensor_or_ragged_tensor(start_offsets)
      ends = ragged_tensor.convert_to_tensor_or_ragged_tensor(end_offsets)
      if input_tensor.shape.ndims is None:
        raise ValueError("Rank of input_tensor must be statically known.")
      if starts.shape.ndims is None:
        raise ValueError("Rank of start_offsets must be statically known.")
      if ends.shape.ndims is None:
        raise ValueError("Rank of end_offsets must be statically known.")
      if starts.shape.ndims != ends.shape.ndims:
        raise ValueError("Rank of start_offsets should be the same as ends.")
      if starts.shape.ndims != input_tensor.shape.ndims + 1:
        raise ValueError("Rank of offsets should be the as input values.")
      if ragged_tensor.is_ragged(input_tensor):
        if not ragged_tensor.is_ragged(starts):
          starts = ragged_conversion_ops.from_tensor(starts)
        if not ragged_tensor.is_ragged(ends):
          ends = ragged_conversion_ops.from_tensor(ends)
        if input_tensor.flat_values.shape.ndims > 1:
          # TODO(broken): Handle case where ragged_rank are not equal.
          if (starts.ragged_rank != input_tensor.ragged_rank or
              ends.ragged_rank != input_tensor.ragged_rank):
            raise ValueError("Ragged rank of inputs must be the same.")
          # If the flat_values of our ragged tensor is multi-dimensional, we can
          # process it separately and our output will have the same nested
          # splits as our input.
          result = self.split_by_offsets(input_tensor.flat_values,
                                         starts.flat_values,
                                         ends.flat_values)
          return input_tensor.with_flat_values(result)
        else:
          # Recursively process the values of the ragged tensor.
          result = self.split_by_offsets(input_tensor.values,
                                         starts.values,
                                         ends.values)
          return input_tensor.with_values(result)
      else:
        if input_tensor.shape.ndims > 1:
          # Convert the input tensor to ragged and process it.
          input_tensor = ragged_conversion_ops.from_tensor(input_tensor)
          return self.split_by_offsets(input_tensor, starts, ends)
        elif input_tensor.shape.ndims == 0:
          stacked_inputs = array_ops_stack.stack([input_tensor])
          stacked_starts = array_ops_stack.stack([starts])
          stacked_ends = array_ops_stack.stack([ends])
          result = self.split_by_offsets(
              stacked_inputs, stacked_starts, stacked_ends)
          return result.values
        else:
          # Our rank 1 tensor is the correct shape, so we can process it as
          # normal.
          if not ragged_tensor.is_ragged(starts):
            starts = ragged_conversion_ops.from_tensor(starts)
          if not ragged_tensor.is_ragged(ends):
            ends = ragged_conversion_ops.from_tensor(ends)
          (values, splits) = gen_byte_splitter.tf_text_byte_split_by_offsets(
              input_tensor, starts.values, ends.values, starts.row_splits)
          return RaggedTensor.from_nested_row_splits(values, [splits])
