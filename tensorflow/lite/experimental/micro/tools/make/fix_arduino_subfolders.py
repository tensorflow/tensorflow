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


def rename_example_subfolder_files(library_dir):
  """Moves source files in example subfolders to equivalents at root."""
  search_path = os.path.join(library_dir, 'examples/*/*', '*.cpp')
  for cpp_path in glob.glob(search_path):
    cpp_dir = os.path.dirname(cpp_path)
    cpp_base = os.path.basename(cpp_path)
    new_cpp_path = cpp_dir + '_' + cpp_base
    os.rename(cpp_path, new_cpp_path)


def move_person_data(library_dir):
  """Moves the downloaded person model into the examples folder."""
  old_person_data_path = os.path.join(
      library_dir,
      'src/tensorflow/lite/experimental/micro/tools/make/downloads/' +
      'person_model_grayscale/person_detect_model_data.cpp'
  )
  new_person_data_path = os.path.join(
      library_dir, 'examples/micro_vision/person_detect_model_data.cpp')
  if os.path.exists(old_person_data_path):
    os.rename(old_person_data_path, new_person_data_path)
    # Update include.
    with open(new_person_data_path, 'r') as source_file:
      file_contents = source_file.read()
    file_contents = file_contents.replace(
        '#include "tensorflow/lite/experimental/micro/examples/' +
        'micro_vision/person_detect_model_data.h"',
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
