# Lint as: python2, python3
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

import six


def replace_includes(line, supplied_headers_list):
  """Updates any includes to reference the new Arduino library paths."""
  include_match = re.match(r'(.*#include.*")(.*)(")', line)
  if include_match:
    path = include_match.group(2)
    for supplied_header in supplied_headers_list:
      if six.ensure_str(supplied_header).endswith(path):
        path = supplied_header
        break
    line = include_match.group(1) + six.ensure_str(path) + include_match.group(
        3)
  return line


def replace_main(line):
  """Updates any occurrences of a bare main definition to the Arduino equivalent."""
  main_match = re.match(r'(.*int )(main)(\(.*)', line)
  if main_match:
    line = main_match.group(1) + 'tflite_micro_main' + main_match.group(3)
  return line


def check_ino_functions(input_text):
  """Ensures the required functions exist."""
  # We're moving to an Arduino-friendly structure for all our examples, so they
  # have to have a setup() and loop() function, just like their IDE expects.
  if not re.search(r'void setup\(\) \{', input_text):
    raise Exception(
        'All examples must have a setup() function for Arduino compatibility\n'
        + input_text)
  if not re.search(r'void loop\(\) \{', input_text):
    raise Exception(
        'All examples must have a loop() function for Arduino compatibility')
  return input_text


def add_example_ino_library_include(input_text):
  """Makes sure the example includes the header that loads the library."""
  return re.sub(r'#include ', '#include <TensorFlowLite.h>\n\n#include ',
                input_text, 1)


def replace_example_includes(line, _):
  """Updates any includes for local example files."""
  # Because the export process moves the example source and header files out of
  # their default locations into the top-level 'examples' folder in the Arduino
  # library, we have to update any include references to match.
  dir_path = 'tensorflow/lite/micro/examples/'
  include_match = re.match(
      r'(.*#include.*")' + six.ensure_str(dir_path) + r'([^/]+)/(.*")', line)
  if include_match:
    flattened_name = re.sub(r'/', '_', include_match.group(3))
    line = include_match.group(1) + flattened_name
  return line


def main(unused_args, flags):
  """Transforms the input source file to work when exported to Arduino."""
  input_file_lines = sys.stdin.read().split('\n')

  supplied_headers_list = six.ensure_str(flags.third_party_headers).split(' ')

  output_lines = []
  for line in input_file_lines:
    line = replace_includes(line, supplied_headers_list)
    if flags.is_example_ino or flags.is_example_source:
      line = replace_example_includes(line, flags.source_path)
    else:
      line = replace_main(line)
    output_lines.append(line)
  output_text = '\n'.join(output_lines)

  if flags.is_example_ino:
    output_text = check_ino_functions(output_text)
    output_text = add_example_ino_library_include(output_text)

  sys.stdout.write(output_text)


def parse_args():
  """Converts the raw arguments into accessible flags."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--third_party_headers',
      type=str,
      default='',
      help='Space-separated list of headers to resolve.')
  parser.add_argument(
      '--is_example_ino',
      dest='is_example_ino',
      action='store_true',
      help='Whether the destination is an example main ino.')
  parser.add_argument(
      '--is_example_source',
      dest='is_example_source',
      action='store_true',
      help='Whether the destination is an example cpp or header file.')
  parser.add_argument(
      '--source_path',
      type=str,
      default='',
      help='The relative path of the source code file.')
  flags, unparsed = parser.parse_known_args()

  main(unparsed, flags)


if __name__ == '__main__':
  parse_args()
