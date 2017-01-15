# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Command parsing module for TensorFlow Debugger (tfdbg)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import re
import sys


_BRACKETS_PATTERN = re.compile(r"\[[^\]]*\]")
_QUOTES_PATTERN = re.compile(r"\"[^\"]*\"")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def parse_command(command):
  """Parse command string into a list of arguments.

  - Disregards whitespace inside double quotes and brackets.
  - Strips paired leading and trailing double quotes in arguments.
  - Splits the command at whitespace.

  Nested double quotes and brackets are not handled.

  Args:
    command: (str) Input command.

  Returns:
    (list of str) List of arguments.
  """

  command = command.strip()
  if not command:
    return []

  brackets_intervals = [f.span() for f in _BRACKETS_PATTERN.finditer(command)]
  quotes_intervals = [f.span() for f in _QUOTES_PATTERN.finditer(command)]
  whitespaces_intervals = [
      f.span() for f in _WHITESPACE_PATTERN.finditer(command)
  ]

  if not whitespaces_intervals:
    return [command]

  arguments = []
  idx0 = 0
  for start, end in whitespaces_intervals + [(len(command), None)]:
    # Skip whitespace stretches enclosed in brackets or double quotes.

    if not any(interval[0] < start < interval[1]
               for interval in brackets_intervals + quotes_intervals):
      argument = command[idx0:start]

      # Strip leading and trailing double quote if they are paired.
      if argument.startswith("\"") and argument.endswith("\""):
        argument = argument[1:-1]
      arguments.append(argument)
      idx0 = end

  return arguments


def extract_output_file_path(args):
  """Extract output file path from command arguments.

  Args:
    args: (list of str) command arguments.

  Returns:
    (list of str) Command arguments with the output file path part stripped.
    (str or None) Output file path (if any).

  Raises:
    SyntaxError: If there is no file path after the last ">" character.
  """

  if args and args[-1].endswith(">"):
    raise SyntaxError("Redirect file path is empty")
  elif args and args[-1].startswith(">"):
    output_file_path = args[-1][1:]
    args = args[:-1]
  elif len(args) > 1 and args[-2] == ">":
    output_file_path = args[-1]
    args = args[:-2]
  elif args and args[-1].count(">") == 1:
    gt_index = args[-1].index(">")
    output_file_path = args[-1][gt_index + 1:]
    args[-1] = args[-1][:gt_index]
  elif len(args) > 1 and args[-2].endswith(">"):
    output_file_path = args[-1]
    args = args[:-1]
    args[-1] = args[-1][:-1]
  else:
    output_file_path = None

  return args, output_file_path


def parse_tensor_name_with_slicing(in_str):
  """Parse tensor name, potentially suffixed by slicing string.

  Args:
    in_str: (str) Input name of the tensor, potentially followed by a slicing
      string. E.g.: Without slicing string: "hidden/weights/Variable:0", with
      slicing string: "hidden/weights/Variable:0[1, :]"

  Returns:
    (str) name of the tensor
    (str) slicing string, if any. If no slicing string is present, return "".
  """

  if in_str.count("[") == 1 and in_str.endswith("]"):
    tensor_name = in_str[:in_str.index("[")]
    tensor_slicing = in_str[in_str.index("["):]
  else:
    tensor_name = in_str
    tensor_slicing = ""

  return tensor_name, tensor_slicing


def validate_slicing_string(slicing_string):
  """Validate a slicing string.

  Check if the input string contains only brackets, digits, commas and
  colons that are valid characters in numpy-style array slicing.

  Args:
    slicing_string: (str) Input slicing string to be validated.

  Returns:
    (bool) True if and only if the slicing string is valid.
  """

  return bool(re.search(r"^\[(\d|,|\s|:)+\]$", slicing_string))


def parse_indices(indices_string):
  """Parse a string representing indices.

  For example, if the input is "[1, 2, 3]", the return value will be a list of
  indices: [1, 2, 3]

  Args:
    indices_string: (str) a string representing indices. Can optionally be
      surrounded by a pair of brackets.

  Returns:
    (list of int): Parsed indices.
  """

  # Strip whitespace.
  indices_string = re.sub(r"\s+", "", indices_string)

  # Strip any brackets at the two ends.
  if indices_string.startswith("[") and indices_string.endswith("]"):
    indices_string = indices_string[1:-1]

  return [int(element) for element in indices_string.split(",")]


def parse_ranges(range_string):
  """Parse a string representing numerical range(s).

  Args:
    range_string: (str) A string representing a numerical range or a list of
      them. For example:
        "[-1.0,1.0]", "[-inf, 0]", "[[-inf, -1.0], [1.0, inf]]"

  Returns:
    (list of list of float) A list of numerical ranges parsed from the input
      string.

  Raises:
    ValueError: If the input doesn't represent a range or a list of ranges.
  """

  range_string = range_string.strip()
  if not range_string:
    return []

  if "inf" in range_string:
    range_string = re.sub(r"inf", repr(sys.float_info.max), range_string)

  ranges = ast.literal_eval(range_string)
  if isinstance(ranges, list) and not isinstance(ranges[0], list):
    ranges = [ranges]

  # Verify that ranges is a list of list of numbers.
  for item in ranges:
    if len(item) != 2:
      raise ValueError("Incorrect number of elements in range")
    elif not isinstance(item[0], (int, float)):
      raise ValueError("Incorrect type in the 1st element of range: %s" %
                       type(item[0]))
    elif not isinstance(item[1], (int, float)):
      raise ValueError("Incorrect type in the 2nd element of range: %s" %
                       type(item[0]))

  return ranges


def parse_readable_size_str(size_str):
  """Convert a human-readable str representation to number of bytes.

  Only the units "kB", "MB", "GB" are supported. The "B character at the end
  of the input `str` may be omitted.

  Args:
    size_str: (`str`) A human-readable str representing a number of bytes
      (e.g., "0", "1023", "1.1kB", "24 MB", "23GB", "100 G".

  Returns:
    (`int`) The parsed number of bytes.

  Raises:
    ValueError: on failure to parse the input `size_str`.
  """

  size_str = size_str.strip()
  if size_str.endswith("B"):
    size_str = size_str[:-1]

  if size_str.isdigit():
    return int(size_str)
  elif size_str.endswith("k"):
    return int(float(size_str[:-1]) * 1024)
  elif size_str.endswith("M"):
    return int(float(size_str[:-1]) * 1048576)
  elif size_str.endswith("G"):
    return int(float(size_str[:-1]) * 1073741824)
  else:
    raise ValueError("Failed to parsed human-readable byte size str: \"%s\"" %
                     size_str)


def evaluate_tensor_slice(tensor, tensor_slicing):
  """Call eval on the slicing of a tensor, with validation.

  Args:
    tensor: (numpy ndarray) The tensor value.
    tensor_slicing: (str or None) Slicing of the tensor, e.g., "[:, 1]". If
      None, no slicing will be performed on the tensor.

  Returns:
    (numpy ndarray) The sliced tensor.

  Raises:
    ValueError: If tensor_slicing is not a valid numpy ndarray slicing str.
  """

  _ = tensor

  if not validate_slicing_string(tensor_slicing):
    raise ValueError("Invalid tensor-slicing string.")

  return eval("tensor" + tensor_slicing)  # pylint: disable=eval-used
