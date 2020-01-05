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
"""Moves source files to match Arduino library conventions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import os

import six


def rename_example_subfolder_files(library_dir):
  """Moves source files in example subfolders to equivalents at root."""
  patterns = ['*.h', '*.cpp']
  for pattern in patterns:
    search_path = os.path.join(library_dir, 'examples/*/*', pattern)
    for source_file_path in glob.glob(search_path):
      source_file_dir = os.path.dirname(source_file_path)
      source_file_base = os.path.basename(source_file_path)
      new_source_file_path = source_file_dir + '_' + source_file_base
      os.rename(source_file_path, new_source_file_path)


def move_person_data(library_dir):
  """Moves the downloaded person model into the examples folder."""
  old_person_data_path = os.path.join(
      library_dir, 'src/tensorflow/lite/micro/tools/make/downloads/' +
      'person_model_grayscale/person_detect_model_data.cpp')
  new_person_data_path = os.path.join(
      library_dir, 'examples/person_detection/person_detect_model_data.cpp')
  if os.path.exists(old_person_data_path):
    os.rename(old_person_data_path, new_person_data_path)
    # Update include.
    with open(new_person_data_path, 'r') as source_file:
      file_contents = source_file.read()
    file_contents = file_contents.replace(
        six.ensure_str('#include "tensorflow/lite/micro/examples/' +
                       'person_detection/person_detect_model_data.h"'),
        '#include "person_detect_model_data.h"')
    with open(new_person_data_path, 'w') as source_file:
      source_file.write(file_contents)


def rename_example_main_inos(library_dir):
  """Makes sure the .ino sketch files match the example name."""
  search_path = os.path.join(library_dir, 'examples/*', 'main.ino')
  for ino_path in glob.glob(search_path):
    example_path = os.path.dirname(ino_path)
    example_name = os.path.basename(example_path)
    new_ino_path = os.path.join(example_path, example_name + '.ino')
    os.rename(ino_path, new_ino_path)


def main(unparsed_args):
  """Control the rewriting of source files."""
  library_dir = unparsed_args[0]
  rename_example_subfolder_files(library_dir)
  rename_example_main_inos(library_dir)
  move_person_data(library_dir)


def parse_args():
  """Converts the raw arguments into accessible flags."""
  parser = argparse.ArgumentParser()
  _, unparsed_args = parser.parse_known_args()

  main(unparsed_args)


if __name__ == '__main__':
  parse_args()
