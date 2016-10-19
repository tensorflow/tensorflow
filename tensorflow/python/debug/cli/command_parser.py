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

import re

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


def parse_tensor_name_with_slicing(in_str):
  """Parse tensor name, potentially suffixed by slicing string.

  Args:
    in_str: (str) Input name of the tensor, potentially followed by a slicing
      string. E.g.: Without slicing string: "hidden/weights/Variable:0", with
      slicing string: "hidden/weights/Varaible:0[1, :]"

  Returns:
    (str) name of the tensor
    (str) sliciing string, if any. If no slicing string is present, return "".
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
