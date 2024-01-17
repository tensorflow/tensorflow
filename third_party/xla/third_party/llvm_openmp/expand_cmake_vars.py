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

"""Expands CMake variables in a text file."""

import re
import sys

_CMAKE_DEFINE_REGEX = re.compile(r"\s*#cmakedefine\s+([A-Za-z_0-9]*)(\s.*)?$")
_CMAKE_DEFINE01_REGEX = re.compile(r"\s*#cmakedefine01\s+([A-Za-z_0-9]*)")
_CMAKE_VAR_REGEX = re.compile(r"\${([A-Za-z_0-9]*)}")
_CMAKE_ATVAR_REGEX = re.compile(r"@([A-Za-z_0-9]*)@")


def _parse_args(argv):
  """Parses arguments with the form KEY=VALUE into a dictionary."""
  result = {}
  for arg in argv:
    k, v = arg.split("=")
    result[k] = v
  return result


def _expand_variables(input_str, cmake_vars):
  """Expands ${VARIABLE}s and @VARIABLE@s in 'input_str', using dictionary 'cmake_vars'.

  Args:
    input_str: the string containing ${VARIABLE} or @VARIABLE@ expressions to expand.
    cmake_vars: a dictionary mapping variable names to their values.

  Returns:
    The expanded string.
  """
  def replace(match):
    if match.group(1) in cmake_vars:
      return cmake_vars[match.group(1)]
    return ""
  return _CMAKE_ATVAR_REGEX.sub(replace,_CMAKE_VAR_REGEX.sub(replace, input_str))


def _expand_cmakedefines(line, cmake_vars):
  """Expands #cmakedefine declarations, using a dictionary 'cmake_vars'."""

  # Handles #cmakedefine lines
  match = _CMAKE_DEFINE_REGEX.match(line)
  if match:
    name = match.group(1)
    suffix = match.group(2) or ""
    if name in cmake_vars:
      return "#define {}{}\n".format(name,
                                     _expand_variables(suffix, cmake_vars))
    else:
      return "/* #undef {} */\n".format(name)

  # Handles #cmakedefine01 lines
  match = _CMAKE_DEFINE01_REGEX.match(line)
  if match:
    name = match.group(1)
    value = cmake_vars.get(name, "0")
    return "#define {} {}\n".format(name, value)

  # Otherwise return the line unchanged.
  return _expand_variables(line, cmake_vars)


def main():
  cmake_vars = _parse_args(sys.argv[1:])
  for line in sys.stdin:
    sys.stdout.write(_expand_cmakedefines(line, cmake_vars))


if __name__ == "__main__":
  main()
