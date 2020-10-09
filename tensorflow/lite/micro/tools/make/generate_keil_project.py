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
"""Generates a Keil uVision project file from a template."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re

import six


def sanitize_xml(unsanitized):
  """Uses a allowlist to avoid generating bad XML."""
  return re.sub(r'[^a-zA-Z0-9+_\-/\\.]', '', six.ensure_str(unsanitized))


def main(unused_args, flags):
  """Generates a Keil project file from a template source."""
  with open(flags.input_template, 'r') as input_template_file:
    template_file_text = input_template_file.read()

  template_file_text = re.sub(r'%{EXECUTABLE}%',
                              six.ensure_str(flags.executable),
                              template_file_text)

  srcs_list = six.ensure_str(flags.srcs).split(' ')
  hdrs_list = six.ensure_str(flags.hdrs).split(' ')
  all_srcs_list = srcs_list + hdrs_list
  all_srcs_list.sort()

  replace_srcs = ''
  for src in all_srcs_list:
    if not src:
      continue
    ext = os.path.splitext(src)[1]
    # These extension indexes are used by uVision to keep track of the type
    # of files. I determined them by experimentation, since the file format
    # isn't documented.
    if ext == '.h':
      ext_index = '5'
    elif ext == '.c':
      ext_index = '1'
    elif ext == '.cc' or ext == '.cpp':
      ext_index = '8'
    else:
      ext_index = '5'
    basename = sanitize_xml(os.path.basename(src))
    clean_src = sanitize_xml(src)
    replace_srcs += '            <File>\n'
    replace_srcs += '              <FileName>' + basename + '</FileName>\n'
    replace_srcs += '              <FileType>' + ext_index + '</FileType>\n'
    replace_srcs += '              <FilePath>' + clean_src + '</FilePath>\n'
    replace_srcs += '            </File>\n'
  template_file_text = re.sub(r'%{SRCS}%', replace_srcs,
                              six.ensure_str(template_file_text))

  include_paths = re.sub(' ', ';', six.ensure_str(flags.include_paths))
  template_file_text = re.sub(r'%{INCLUDE_PATHS}%', include_paths,
                              template_file_text)

  with open(flags.output_file, 'w') as output_file:
    output_file.write(template_file_text)


def parse_args():
  """Converts the raw arguments into accessible flags."""
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--input_template',
      type=str,
      default='',
      help='Path to template project file to build from.')
  parser.add_argument(
      '--output_file',
      type=str,
      default='',
      help='Path to write the completed project file to.')
  parser.add_argument(
      '--executable',
      type=str,
      default='',
      help='Name of the executable the project will build.')
  parser.add_argument(
      '--hdrs',
      type=str,
      default='',
      help='Space-separated list of C or C++ source files to compile.')
  parser.add_argument(
      '--srcs',
      type=str,
      default='',
      help='Space-separated list of C or C++ header files to include.')
  parser.add_argument(
      '--include_paths',
      type=str,
      default='',
      help='Space-separated list of paths to look for header files on.')
  flags, unparsed = parser.parse_known_args()

  main(unparsed, flags)


if __name__ == '__main__':
  parse_args()
