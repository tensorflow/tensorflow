# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

"""Testing File IO operations in file_io.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf

from tensorflow.python.lib.io import file_io


class FileIoTest(tf.test.TestCase):

  def testFileDoesntExist(self):
    file_path = os.path.join(self.get_temp_dir(), "temp_file")
    self.assertFalse(file_io.file_exists(file_path))

  def testFileWrite(self):
    file_path = os.path.join(self.get_temp_dir(), "temp_file")
    file_io.write_string_to_file(file_path, "testing")
    self.assertTrue(file_io.file_exists(file_path))
    file_contents = file_io.read_file_to_string(file_path)
    self.assertEqual(b"testing", file_contents)
    file_io.delete_file(file_path)

  def testFileDelete(self):
    file_path = os.path.join(self.get_temp_dir(), "temp_file")
    file_io.write_string_to_file(file_path, "testing")
    file_io.delete_file(file_path)
    self.assertFalse(file_io.file_exists(file_path))

  def testGetMatchingFiles(self):
    dir_path = os.path.join(self.get_temp_dir(), "temp_dir")
    file_io.create_dir(dir_path)
    files = ["file1.txt", "file2.txt", "file3.txt"]
    for name in files:
      file_path = os.path.join(dir_path, name)
      file_io.write_string_to_file(file_path, "testing")
    expected_match = [os.path.join(dir_path, name) for name in files]
    self.assertItemsEqual(file_io.get_matching_files(os.path.join(dir_path,
                                                                  "file*.txt")),
                          expected_match)
    for name in files:
      file_path = os.path.join(dir_path, name)
      file_io.delete_file(file_path)

  def testCreateRecursiveDir(self):
    dir_path = os.path.join(self.get_temp_dir(), "temp_dir/temp_dir1/temp_dir2")
    file_io.recursive_create_dir(dir_path)
    file_path = os.path.join(dir_path, "temp_file")
    file_io.write_string_to_file(file_path, "testing")
    self.assertTrue(file_io.file_exists(file_path))


if __name__ == "__main__":
  tf.test.main()
