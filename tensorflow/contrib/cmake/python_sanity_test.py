# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest


def abs_path(path):
  root = os.path.dirname(__file__)

  for _ in range(3):
    root = os.path.join(root, os.pardir)

  path = os.path.join(root, path)
  path = os.path.abspath(path)
  return path

def read_entries(test):
  with open(abs_path(test.entries_file), "r") as f:
    lines = f.readlines()

  lines = [line.strip() for line in lines]

  test.entries = []
  test.whitelist = []

  for line in lines:
    if line.startswith('#'):
      line = line[1:].strip()
      test.whitelist.append(line)
    elif line.find('#') != -1:
      line = line[:line.find('#')].strip()
      test.entries.append(line)
    else:
      test.entries.append(line)

def test_invalid_directories(test):
  for entry in test.entries:
    test.assertTrue(os.path.isdir(abs_path(entry)),
      "'" + test.entries_file + "' contains invalid '" + entry + "'")

def test_missing_directory(test, path):
  if path in test.whitelist:
    return

  dir_exists = os.path.isdir(abs_path(path))
  entry_exists = path in test.entries

  test.assertFalse(dir_exists and not entry_exists,
    "'" + test.entries_file + "' is missing '" + path + "'")


class PythonModuleTest(unittest.TestCase):

  def setUp(self):
    self.entries_file = "tensorflow/contrib/cmake/python_modules.txt"
    read_entries(self)

  def testInvalidEntries(self):
    test_invalid_directories(self)

  def testMissingModules(self):
    module_names = next(os.walk(abs_path("tensorflow/contrib")))[1]

    for module_name in module_names:
      path = "tensorflow/contrib/" + module_name

      test_missing_directory(self, path + "/python")
      test_missing_directory(self, path + "/python/ops")
      test_missing_directory(self, path + "/python/kernels")
      test_missing_directory(self, path + "/python/layers")


class PythonProtoTest(unittest.TestCase):

  def setUp(self):
    self.entries_file = "tensorflow/contrib/cmake/python_protos.txt"
    read_entries(self)

  def testInvalidEntries(self):
    test_invalid_directories(self)


class PythonProtoCCTest(unittest.TestCase):

  def setUp(self):
    self.entries_file = "tensorflow/contrib/cmake/python_protos_cc.txt"
    read_entries(self)

  def testInvalidEntries(self):
    test_invalid_directories(self)


if __name__ == "__main__":
  unittest.main()
