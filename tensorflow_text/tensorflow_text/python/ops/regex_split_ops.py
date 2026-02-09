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

"""This file contains the python libraries for the regex_split op."""
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
gen_regex_split_ops = load_library.load_op_library(resource_loader.get_path_to_datafile('_regex_split_ops.so'))
from tensorflow_text.python.ops import splitter


# pylint: disable= redefined-builtin
def regex_split_with_offsets(input,
                             delim_regex_pattern,
                             keep_delim_regex_pattern="",
                             name=None):
  r"""Split `input` by delimiters that match a regex pattern; returns offsets.

  `regex_split_with_offsets` will split `input` using delimiters that match a
  regex pattern in `delim_regex_pattern`. It will return three tensors:
  one containing the split substrings ('result' in the examples below), one
  containing the offsets of the starts of each substring ('begin' in the
  examples below), and one containing the offsets of the ends of each substring
  ('end' in the examples below).

  Here is an example:

  >>> text_input=["hello there"]
  >>> # split by whitespace
  >>> result, begin, end = regex_split_with_offsets(input=text_input,
  ...                                               delim_regex_pattern="\s")
  >>> print("result: %s\nbegin: %s\nend: %s" % (result, begin, end))
  result: <tf.RaggedTensor [[b'hello', b'there']]>
  begin: <tf.RaggedTensor [[0, 6]]>
  end: <tf.RaggedTensor [[5, 11]]>

  By default, delimiters are not included in the split string results.
  Delimiters may be included by specifying a regex pattern
  `keep_delim_regex_pattern`. For example:

  >>> text_input=["hello there"]
  >>> # split by whitespace
  >>> result, begin, end = regex_split_with_offsets(input=text_input,
  ...                                             delim_regex_pattern="\s",
  ...                                             keep_delim_regex_pattern="\s")
  >>> print("result: %s\nbegin: %s\nend: %s" % (result, begin, end))
  result: <tf.RaggedTensor [[b'hello', b' ', b'there']]>
  begin: <tf.RaggedTensor [[0, 5, 6]]>
  end: <tf.RaggedTensor [[5, 6, 11]]>

  If there are multiple delimiters in a row, there are no empty splits emitted.
  For example:

  >>> text_input=["hello  there"]  #  Note the two spaces between the words.
  >>> # split by whitespace
  >>> result, begin, end = regex_split_with_offsets(input=text_input,
  ...                                               delim_regex_pattern="\s")
  >>> print("result: %s\nbegin: %s\nend: %s" % (result, begin, end))
  result: <tf.RaggedTensor [[b'hello', b'there']]>
  begin: <tf.RaggedTensor [[0, 7]]>
  end: <tf.RaggedTensor [[5, 12]]>

  See https://github.com/google/re2/wiki/Syntax for the full list of supported
  expressions.

  Args:
    input: A Tensor or RaggedTensor of string input.
    delim_regex_pattern: A string containing the regex pattern of a delimiter.
    keep_delim_regex_pattern: (optional) Regex pattern of delimiters that should
      be kept in the result.
    name: (optional) Name of the op.

  Returns:
    A tuple of RaggedTensors containing:
      (split_results, begin_offsets, end_offsets)
    where tokens is of type string, begin_offsets and end_offsets are of type
    int64.
  """
  # Convert input to ragged or tensor
  input = ragged_tensor.convert_to_tensor_or_ragged_tensor(
      input, dtype=dtypes.string)

  # Handle RaggedTensor inputs by recursively processing the `flat_values`.
  if ragged_tensor.is_ragged(input):
    # Split the `flat_values` of the input.
    tokens, begin_offsets, end_offsets = regex_split_with_offsets(
        input.flat_values, delim_regex_pattern, keep_delim_regex_pattern, name)
    # Copy outer dimenion partitions from `input` to the output tensors.
    tokens_rt = input.with_flat_values(
        tokens.with_row_splits_dtype(input.row_splits.dtype))
    begin_offsets_rt = input.with_flat_values(
        begin_offsets.with_row_splits_dtype(input.row_splits.dtype))
    end_offsets_rt = input.with_flat_values(
        end_offsets.with_row_splits_dtype(input.row_splits.dtype))
    return tokens_rt, begin_offsets_rt, end_offsets_rt

  delim_regex_pattern = b"".join(
      [b"(", delim_regex_pattern.encode("utf-8"), b")"])
  keep_delim_regex_pattern = b"".join(
      [b"(", keep_delim_regex_pattern.encode("utf-8"), b")"])

  # reshape to a flat Tensor (if not already)
  input_shape = math_ops.cast(array_ops.shape(input), dtypes.int64)
  input_reshaped = array_ops.reshape(input, [-1])

  # send flat_values to regex_split op.
  tokens, begin_offsets, end_offsets, row_splits = (
      gen_regex_split_ops.regex_split_with_offsets(input_reshaped,
                                                   delim_regex_pattern,
                                                   keep_delim_regex_pattern))
  # Pack back into ragged tensors
  tokens_rt = ragged_tensor.RaggedTensor.from_row_splits(
      tokens, row_splits=row_splits)
  begin_offsets_rt = ragged_tensor.RaggedTensor.from_row_splits(
      begin_offsets,
      row_splits=row_splits)
  end_offsets_rt = ragged_tensor.RaggedTensor.from_row_splits(
      end_offsets, row_splits=row_splits)

  # If the original input was a multi-dimensional Tensor, add back the
  # dimensions
  static_rank = input.get_shape().ndims
  if static_rank is not None and static_rank > 1:
    i = array_ops.get_positive_axis(-1, input.get_shape().ndims)
    for i in range(
        array_ops.get_positive_axis(-1,
                                    input.get_shape().ndims), 0, -1):
      tokens_rt = ragged_tensor.RaggedTensor.from_uniform_row_length(
          values=tokens_rt, uniform_row_length=input_shape[i])
      begin_offsets_rt = ragged_tensor.RaggedTensor.from_uniform_row_length(
          values=begin_offsets_rt, uniform_row_length=input_shape[i])
      end_offsets_rt = ragged_tensor.RaggedTensor.from_uniform_row_length(
          values=end_offsets_rt, uniform_row_length=input_shape[i])
  return tokens_rt, begin_offsets_rt, end_offsets_rt


# pylint: disable= redefined-builtin
def regex_split(input,
                delim_regex_pattern,
                keep_delim_regex_pattern="",
                name=None):
  r"""Split `input` by delimiters that match a regex pattern.

  `regex_split` will split `input` using delimiters that match a
  regex pattern in `delim_regex_pattern`. Here is an example:

  >>> text_input=["hello there"]
  >>> # split by whitespace
  >>> regex_split(input=text_input,
  ...             delim_regex_pattern="\s")
  <tf.RaggedTensor [[b'hello', b'there']]>

  By default, delimiters are not included in the split string results.
  Delimiters may be included by specifying a regex pattern
  `keep_delim_regex_pattern`. For example:

  >>> text_input=["hello there"]
  >>> # split by whitespace
  >>> regex_split(input=text_input,
  ...             delim_regex_pattern="\s",
  ...             keep_delim_regex_pattern="\s")
  <tf.RaggedTensor [[b'hello', b' ', b'there']]>

  If there are multiple delimiters in a row, there are no empty splits emitted.
  For example:

  >>> text_input=["hello  there"]  #  Note the two spaces between the words.
  >>> # split by whitespace
  >>> regex_split(input=text_input,
  ...             delim_regex_pattern="\s")
  <tf.RaggedTensor [[b'hello', b'there']]>


  See https://github.com/google/re2/wiki/Syntax for the full list of supported
  expressions.

  Args:
    input: A Tensor or RaggedTensor of string input.
    delim_regex_pattern: A string containing the regex pattern of a delimiter.
    keep_delim_regex_pattern: (optional) Regex pattern of delimiters that should
      be kept in the result.
    name: (optional) Name of the op.

  Returns:
    A RaggedTensors containing of type string containing the split string
    pieces.
  """
  tokens, _, _ = regex_split_with_offsets(input, delim_regex_pattern,
                                          keep_delim_regex_pattern, name)
  return tokens


class RegexSplitter(splitter.SplitterWithOffsets):
  r"""`RegexSplitter` splits text on the given regular expression.

  The default is a newline character pattern. It can also return the beginning
  and ending byte offsets as well.

  By default, this splitter will break on newlines, ignoring any trailing ones.
  >>> splitter = RegexSplitter()
  >>> text_input=[
  ...       b"Hi there.\nWhat time is it?\nIt is gametime.",
  ...       b"Who let the dogs out?\nWho?\nWho?\nWho?\n\n",
  ...   ]
  >>> splitter.split(text_input)
  <tf.RaggedTensor [[b'Hi there.', b'What time is it?', b'It is gametime.'],
                    [b'Who let the dogs out?', b'Who?', b'Who?', b'Who?']]>

  The splitter can be passed a custom split pattern, as well. The pattern
  can be any string, but we're using a single character (tab) in this example.
  >>> splitter = RegexSplitter(split_regex='\t')
  >>> text_input=[
  ...       b"Hi there.\tWhat time is it?\tIt is gametime.",
  ...       b"Who let the dogs out?\tWho?\tWho?\tWho?\t\t",
  ...   ]
  >>> splitter.split(text_input)
  <tf.RaggedTensor [[b'Hi there.', b'What time is it?', b'It is gametime.'],
                    [b'Who let the dogs out?', b'Who?', b'Who?', b'Who?']]>

  """

  def __init__(self, split_regex=None):
    r"""Creates an instance of `RegexSplitter`.

    Args:
      split_regex: (optional) A string containing the regex pattern of a
        delimiter to split on. Default is '\r?\n'.
    """
    if not split_regex:
      split_regex = "\r?\n"
    self._split_regex = split_regex

  def split(self, input):  # pylint: disable=redefined-builtin
    return regex_split(input, self._split_regex)

  def split_with_offsets(self, input):  # pylint: disable=redefined-builtin
    return regex_split_with_offsets(input, self._split_regex)
