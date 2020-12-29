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
import shutil
import tempfile
import zipfile


def main(unparsed_args):
  """Merges multiple Arduino zipfiles into a single result."""
  output_zip_path = unparsed_args[0]
  input_zip_paths = unparsed_args[1::]
  working_dir = tempfile.mkdtemp()
  for input_zip_path in input_zip_paths:
    with zipfile.ZipFile(input_zip_path, 'r') as input_zip:
      input_zip.extractall(path=working_dir)
  output_path_without_zip = output_zip_path.replace('.zip', '')
  shutil.make_archive(output_path_without_zip, 'zip', working_dir)


def parse_args():
  """Converts the raw arguments into accessible flags."""
  parser = argparse.ArgumentParser()
  _, unparsed_args = parser.parse_known_args()

  main(unparsed_args)


if __name__ == '__main__':
  parse_args()
