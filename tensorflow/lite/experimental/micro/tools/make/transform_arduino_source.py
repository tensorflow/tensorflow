# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Resolves non-system C/C++ includes to their full paths to help Arduino."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import re
import sys


def main(unused_args, flags):
  """Resolves third party headers to their full paths in source code."""
  input_file_lines = sys.stdin.read().split('\n')

  supplied_headers_list = flags.third_party_headers.split(' ')

  output_lines = []
  for line in input_file_lines:
    include_match = re.match(r'(.*#include.*")(.*)(")', line)
    if include_match:
      path = include_match.group(2)
      for supplied_header in supplied_headers_list:
        if supplied_header.endswith(path):
          path = supplied_header
          break
      line = include_match.group(1) + path + include_match.group(3)
    output_lines.append(line)
  output_text = '\n'.join(output_lines)

  sys.stdout.write(output_text)


def parse_args():
  """Converts the raw arguments into accessible flags."""
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--third_party_headers',
      type=str,
      default='',
      help='Space-separated list of headers to resolve.')
  flags, unparsed = parser.parse_known_args()

  main(unparsed, flags)


if __name__ == '__main__':
  parse_args()
