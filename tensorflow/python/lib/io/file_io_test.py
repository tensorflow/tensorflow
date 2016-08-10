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

from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io
from tensorflow.python.util import compat


class FileIoTest(tf.test.TestCase):

  def setUp(self):
    self._base_dir = os.path.join(self.get_temp_dir(), "base_dir")
    file_io.create_dir(self._base_dir)

  def tearDown(self):
    file_io.delete_recursively(self._base_dir)

  def testFileDoesntExist(self):
    file_path = os.path.join(self._base_dir, "temp_file")
    self.assertFalse(file_io.file_exists(file_path))

  def testFileWrite(self):
    file_path = os.path.join(self._base_dir, "temp_file")
    file_io.write_string_to_file(file_path, "testing")
    self.assertTrue(file_io.file_exists(file_path))
    file_contents = file_io.read_file_to_string(file_path)
    self.assertEqual(b"testing", file_contents)

  def testFileDelete(self):
    file_path = os.path.join(self._base_dir, "temp_file")
    file_io.write_string_to_file(file_path, "testing")
    file_io.delete_file(file_path)
    self.assertFalse(file_io.file_exists(file_path))

  def testGetMatchingFiles(self):
    dir_path = os.path.join(self._base_dir, "temp_dir")
    file_io.create_dir(dir_path)
    files = ["file1.txt", "file2.txt", "file3.txt"]
    for name in files:
      file_path = os.path.join(dir_path, name)
      file_io.write_string_to_file(file_path, "testing")
    expected_match = [os.path.join(dir_path, name) for name in files]
    self.assertItemsEqual(
        file_io.get_matching_files(os.path.join(dir_path, "file*.txt")),
        expected_match)
    file_io.delete_recursively(dir_path)
    self.assertFalse(file_io.file_exists(os.path.join(dir_path, "file3.txt")))

  def testCreateRecursiveDir(self):
    dir_path = os.path.join(self._base_dir, "temp_dir/temp_dir1/temp_dir2")
    file_io.recursive_create_dir(dir_path)
    file_path = os.path.join(dir_path, "temp_file")
    file_io.write_string_to_file(file_path, "testing")
    self.assertTrue(file_io.file_exists(file_path))
    file_io.delete_recursively(os.path.join(self._base_dir, "temp_dir"))
    self.assertFalse(file_io.file_exists(file_path))

  def testCopy(self):
    file_path = os.path.join(self._base_dir, "temp_file")
    file_io.write_string_to_file(file_path, "testing")
    copy_path = os.path.join(self._base_dir, "copy_file")
    file_io.copy(file_path, copy_path)
    self.assertTrue(file_io.file_exists(copy_path))
    self.assertEqual(b"testing", file_io.read_file_to_string(file_path))

  def testCopyOverwrite(self):
    file_path = os.path.join(self._base_dir, "temp_file")
    file_io.write_string_to_file(file_path, "testing")
    copy_path = os.path.join(self._base_dir, "copy_file")
    file_io.write_string_to_file(copy_path, "copy")
    file_io.copy(file_path, copy_path, overwrite=True)
    self.assertTrue(file_io.file_exists(copy_path))
    self.assertEqual(b"testing", file_io.read_file_to_string(file_path))

  def testCopyOverwriteFalse(self):
    file_path = os.path.join(self._base_dir, "temp_file")
    file_io.write_string_to_file(file_path, "testing")
    copy_path = os.path.join(self._base_dir, "copy_file")
    file_io.write_string_to_file(copy_path, "copy")
    with self.assertRaises(errors.AlreadyExistsError):
      file_io.copy(file_path, copy_path, overwrite=False)

  def testRename(self):
    file_path = os.path.join(self._base_dir, "temp_file")
    file_io.write_string_to_file(file_path, "testing")
    rename_path = os.path.join(self._base_dir, "rename_file")
    file_io.rename(file_path, rename_path)
    self.assertTrue(file_io.file_exists(rename_path))
    self.assertFalse(file_io.file_exists(file_path))

  def testRenameOverwrite(self):
    file_path = os.path.join(self._base_dir, "temp_file")
    file_io.write_string_to_file(file_path, "testing")
    rename_path = os.path.join(self._base_dir, "rename_file")
    file_io.write_string_to_file(rename_path, "rename")
    file_io.rename(file_path, rename_path, overwrite=True)
    self.assertTrue(file_io.file_exists(rename_path))
    self.assertFalse(file_io.file_exists(file_path))

  def testRenameOverwriteFalse(self):
    file_path = os.path.join(self._base_dir, "temp_file")
    file_io.write_string_to_file(file_path, "testing")
    rename_path = os.path.join(self._base_dir, "rename_file")
    file_io.write_string_to_file(rename_path, "rename")
    with self.assertRaises(errors.AlreadyExistsError):
      file_io.rename(file_path, rename_path, overwrite=False)
    self.assertTrue(file_io.file_exists(rename_path))
    self.assertTrue(file_io.file_exists(file_path))

  def testDeleteRecursivelyFail(self):
    fake_dir_path = os.path.join(self._base_dir, "temp_dir")
    with self.assertRaises(errors.NotFoundError):
      file_io.delete_recursively(fake_dir_path)

  def testIsDirectory(self):
    dir_path = os.path.join(self._base_dir, "test_dir")
    # Failure for a non-existing dir.
    with self.assertRaises(errors.NotFoundError):
      file_io.is_directory(dir_path)
    file_io.create_dir(dir_path)
    self.assertTrue(file_io.is_directory(dir_path))
    file_path = os.path.join(dir_path, "test_file")
    file_io.write_string_to_file(file_path, "test")
    # False for a file.
    self.assertFalse(file_io.is_directory(file_path))

  def testListDirectory(self):
    dir_path = os.path.join(self._base_dir, "test_dir")
    file_io.create_dir(dir_path)
    files = [b"file1.txt", b"file2.txt", b"file3.txt"]
    for name in files:
      file_path = os.path.join(dir_path, compat.as_str_any(name))
      file_io.write_string_to_file(file_path, "testing")
    subdir_path = os.path.join(dir_path, "sub_dir")
    file_io.create_dir(subdir_path)
    subdir_file_path = os.path.join(subdir_path, "file4.txt")
    file_io.write_string_to_file(subdir_file_path, "testing")
    dir_list = file_io.list_directory(dir_path)
    self.assertItemsEqual(files + [b"sub_dir"], dir_list)

  def testListDirectoryFailure(self):
    dir_path = os.path.join(self._base_dir, "test_dir")
    with self.assertRaises(errors.NotFoundError):
      file_io.list_directory(dir_path)

  def _setupWalkDirectories(self, dir_path):
    # Creating a file structure as follows
    # test_dir -> file: file1.txt; dirs: subdir1_1, subdir1_2, subdir1_3
    # subdir1_1 -> file: file3.txt
    # subdir1_2 -> dir: subdir2
    file_io.create_dir(dir_path)
    file_io.write_string_to_file(os.path.join(dir_path, "file1.txt"), "testing")
    sub_dirs1 = ["subdir1_1", "subdir1_2", "subdir1_3"]
    for name in sub_dirs1:
      file_io.create_dir(os.path.join(dir_path, name))
    file_io.write_string_to_file(
        os.path.join(dir_path, "subdir1_1/file2.txt"), "testing")
    file_io.create_dir(os.path.join(dir_path, "subdir1_2/subdir2"))

  def testWalkInOrder(self):
    dir_path = os.path.join(self._base_dir, "test_dir")
    self._setupWalkDirectories(dir_path)
    # Now test the walk (in_order = True)
    all_dirs = []
    all_subdirs = []
    all_files = []
    for (w_dir, w_subdirs, w_files) in file_io.walk(dir_path, in_order=True):
      all_dirs.append(w_dir)
      all_subdirs.append(w_subdirs)
      all_files.append(w_files)
    self.assertItemsEqual(all_dirs, [compat.as_bytes(dir_path)] + [
        compat.as_bytes(os.path.join(dir_path, item))
        for item in ["subdir1_1", "subdir1_2", "subdir1_2/subdir2", "subdir1_3"]
    ])
    self.assertEqual(compat.as_bytes(dir_path), all_dirs[0])
    self.assertLess(
        all_dirs.index(compat.as_bytes(os.path.join(dir_path, "subdir1_2"))),
        all_dirs.index(
            compat.as_bytes(os.path.join(dir_path, "subdir1_2/subdir2"))))
    self.assertItemsEqual(all_subdirs[1:5], [[], [b"subdir2"], [], []])
    self.assertItemsEqual(all_subdirs[0],
                          [b"subdir1_1", b"subdir1_2", b"subdir1_3"])
    self.assertItemsEqual(all_files, [[b"file1.txt"], [b"file2.txt"], [], [],
                                      []])
    self.assertLess(
        all_files.index([b"file1.txt"]), all_files.index([b"file2.txt"]))

  def testWalkPostOrder(self):
    dir_path = os.path.join(self._base_dir, "test_dir")
    self._setupWalkDirectories(dir_path)
    # Now test the walk (in_order = False)
    all_dirs = []
    all_subdirs = []
    all_files = []
    for (w_dir, w_subdirs, w_files) in file_io.walk(dir_path, in_order=False):
      all_dirs.append(w_dir)
      all_subdirs.append(w_subdirs)
      all_files.append(w_files)
    self.assertItemsEqual(all_dirs, [
        compat.as_bytes(os.path.join(dir_path, item))
        for item in ["subdir1_1", "subdir1_2/subdir2", "subdir1_2", "subdir1_3"]
    ] + [compat.as_bytes(dir_path)])
    self.assertEqual(compat.as_bytes(dir_path), all_dirs[4])
    self.assertLess(
        all_dirs.index(
            compat.as_bytes(os.path.join(dir_path, "subdir1_2/subdir2"))),
        all_dirs.index(compat.as_bytes(os.path.join(dir_path, "subdir1_2"))))
    self.assertItemsEqual(all_subdirs[0:4], [[], [], [b"subdir2"], []])
    self.assertItemsEqual(all_subdirs[4],
                          [b"subdir1_1", b"subdir1_2", b"subdir1_3"])
    self.assertItemsEqual(all_files, [[b"file2.txt"], [], [], [],
                                      [b"file1.txt"]])
    self.assertLess(
        all_files.index([b"file2.txt"]), all_files.index([b"file1.txt"]))

  def testWalkFailure(self):
    dir_path = os.path.join(self._base_dir, "test_dir")
    # Try walking a directory that wasn't created.
    all_dirs = []
    all_subdirs = []
    all_files = []
    for (w_dir, w_subdirs, w_files) in file_io.walk(dir_path, in_order=False):
      all_dirs.append(w_dir)
      all_subdirs.append(w_subdirs)
      all_files.append(w_files)
    self.assertItemsEqual(all_dirs, [])
    self.assertItemsEqual(all_subdirs, [])
    self.assertItemsEqual(all_files, [])

  def testStat(self):
    file_path = os.path.join(self._base_dir, "temp_file")
    file_io.write_string_to_file(file_path, "testing")
    file_statistics = file_io.stat(file_path)
    os_statistics = os.stat(file_path)
    self.assertEquals(7, file_statistics.length)
    self.assertEqual(
        int(os_statistics.st_mtime), int(file_statistics.mtime_nsec / 1e9))

if __name__ == "__main__":
  tf.test.main()
