#!/usr/bin/python3
# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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


"""Converts MSYS Linux-like paths stored in env variables to Windows paths.

This is necessary on Windows, because some applications do not understand/handle
Linux-like paths MSYS uses, for example, Docker.
"""

import argparse
import os


def should_convert(var_name: str,
                   blacklist: list[str] | None,
                   whitelist_prefix: list[str] | None):
  """Check the variable name against white/black lists."""
  if blacklist and var_name in blacklist:
    return False
  if not whitelist_prefix:
    return True

  for prefix in whitelist_prefix:
    if var_name.startswith(prefix):
      return True
  return False


def main(parsed_args: argparse.Namespace):
  converted_vars = {}

  for var, value in os.environ.items():
    if not value or not should_convert(var,
                                       parsed_args.blacklist,
                                       parsed_args.whitelist_prefix):
      continue

    # In Python, MSYS, Linux-like paths are automatically read as Windows paths
    # with forward slashes, e.g. 'C:/Program Files', instead of
    # '/c/Program Files', thus becoming converted simply by virtue of having
    # been read.
    converted_vars[var] = value

  var_str = '\n'.join(f'{k}="{v}"'
                      for k, v in converted_vars.items())
  # The string can then be piped into `source`, to re-set the
  # 'converted' variables.
  print(var_str)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=(
      'Convert MSYS paths in environment variables to Windows paths.'))
  parser.add_argument('--blacklist',
                      nargs='*',
                      help='List of variables to ignore')
  parser.add_argument('--whitelist-prefix',
                      nargs='*',
                      help='Prefix for variables to include')
  args = parser.parse_args()

  main(args)
