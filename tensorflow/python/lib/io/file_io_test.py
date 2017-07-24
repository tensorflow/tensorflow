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

from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import test


class FileIoTest(test.TestCase):

  def setUp(self):
    self._base_dir = os.path.join(self.get_temp_dir(), "base_dir")
    file_io.create_dir(self._base_dir)

  def tearDown(self):
    file_io.delete_recursively(self._base_dir)

  def testFileDoesntExist(self):
    file_path = os.path.join(self._base_dir, "temp_file")
    self.assertFalse(file_io.file_exists(file_path))
    with self.assertRaises(errors.NotFoundError):
      _ = file_io.read_file_to_string(file_path)

  def testWriteToString(self):
    file_path = os.path.join(self._base_dir, "temp_file")
    file_io.write_string_to_file(file_path, "testing")
    self.assertTrue(file_io.file_exists(file_path))
    file_contents = file_io.read_file_to_string(file_path)
    self.assertEqual("testing", file_contents)

  def testAtomicWriteStringToFile(self):
    file_path = os.path.join(self._base_dir, "temp_file")
    file_io.atomic_write_string_to_file(file_path, "testing")
    self.assertTrue(file_io.file_exists(file_path))
    file_contents = file_io.read_file_to_string(file_path)
    self.assertEqual("testing", file_contents)

  def testReadBinaryMode(self):
    file_path = os.path.join(self._base_dir, "temp_file")
    file_io.write_string_to_file(file_path, "testing")
    with file_io.FileIO(file_path, mode="rb") as f:
      self.assertEqual(b"testing", f.read())

  def testWriteBinaryMode(self):
    file_path = os.path.join(self._base_dir, "temp_file")
    file_io.FileIO(file_path, "wb").write("testing")
    with file_io.FileIO(file_path, mode="r") as f:
      self.assertEqual("testing", f.read())

  def testAppend(self):
    file_path = os.path.join(self._base_dir, "temp_file")
    with file_io.FileIO(file_path, mode="w") as f:
      f.write("begin\n")
    with file_io.FileIO(file_path, mode="a") as f:
      f.write("a1\n")
    with file_io.FileIO(file_path, mode="a") as f:
      f.write("a2\n")
    with file_io.FileIO(file_path, mode="r") as f:
      file_contents = f.read()
      self.assertEqual("begin\na1\na2\n", file_contents)

  def testMultipleFiles(self):
    file_prefix = os.path.join(self._base_dir, "temp_file")
    for i in range(5000):
      f = file_io.FileIO(file_prefix + str(i), mode="w+")
      f.write("testing")
      f.flush()
      self.assertEqual("testing", f.read())
      f.close()

  def testMultipleWrites(self):
    file_path = os.path.join(self._base_dir, "temp_file")
    with file_io.FileIO(file_path, mode="w") as f:
      f.write("line1\n")
      f.write("line2")
    file_contents = file_io.read_file_to_string(file_path)
    self.assertEqual("line1\nline2", file_contents)

  def testFileWriteBadMode(self):
    file_path = os.path.join(self._base_dir, "temp_file")
    with self.assertRaises(errors.PermissionDeniedError):
      file_io.FileIO(file_path, mode="r").write("testing")

  def testFileReadBadMode(self):
    file_path = os.path.join(self._base_dir, "temp_file")
    file_io.FileIO(file_path, mode="w").write("testing")
    self.assertTrue(file_io.file_exists(file_path))
    with self.assertRaises(errors.PermissionDeniedError):
      file_io.FileIO(file_path, mode="w").read()

  def testFileDelete(self):
    file_path = os.path.join(self._base_dir, "temp_file")
    file_io.FileIO(file_path, mode="w").write("testing")
    file_io.delete_file(file_path)
    self.assertFalse(file_io.file_exists(file_path))

  def testFileDeleteFail(self):
    file_path = os.path.join(self._base_dir, "temp_file")
    with self.assertRaises(errors.NotFoundError):
      file_io.delete_file(file_path)

  def testGetMatchingFiles(self):
    dir_path = os.path.join(self._base_dir, "temp_dir")
    file_io.create_dir(dir_path)
    files = ["file1.txt", "file2.txt", "file3.txt"]
    for name in files:
      file_path = os.path.join(dir_path, name)
      file_io.FileIO(file_path, mode="w").write("testing")
    expected_match = [os.path.join(dir_path, name) for name in files]
    self.assertItemsEqual(
        file_io.get_matching_files(os.path.join(dir_path, "file*.txt")),
        expected_match)
    self.assertItemsEqual(file_io.get_matching_files(tuple()), [])
    files_subset = [
        os.path.join(dir_path, files[0]), os.path.join(dir_path, files[2])
    ]
    self.assertItemsEqual(
        file_io.get_matching_files(files_subset), files_subset)
    file_io.delete_recursively(dir_path)
    self.assertFalse(file_io.file_exists(os.path.join(dir_path, "file3.txt")))

  def testCreateRecursiveDir(self):
    dir_path = os.path.join(self._base_dir, "temp_dir/temp_dir1/temp_dir2")
    file_io.recursive_create_dir(dir_path)
    file_io.recursive_create_dir(dir_path)  # repeat creation
    file_path = os.path.join(dir_path, "temp_file")
    file_io.FileIO(file_path, mode="w").write("testing")
    self.assertTrue(file_io.file_exists(file_path))
    file_io.delete_recursively(os.path.join(self._base_dir, "temp_dir"))
    self.assertFalse(file_io.file_exists(file_path))

  def testCopy(self):
    file_path = os.path.join(self._base_dir, "temp_file")
    file_io.FileIO(file_path, mode="w").write("testing")
    copy_path = os.path.join(self._base_dir, "copy_file")
    file_io.copy(file_path, copy_path)
    self.assertTrue(file_io.file_exists(copy_path))
    f = file_io.FileIO(file_path, mode="r")
    self.assertEqual("testing", f.read())
    self.assertEqual(7, f.tell())

  def testCopyOverwrite(self):
    file_path = os.path.join(self._base_dir, "temp_file")
    file_io.FileIO(file_path, mode="w").write("testing")
    copy_path = os.path.join(self._base_dir, "copy_file")
    file_io.FileIO(copy_path, mode="w").write("copy")
    file_io.copy(file_path, copy_path, overwrite=True)
    self.assertTrue(file_io.file_exists(copy_path))
    self.assertEqual("testing", file_io.FileIO(file_path, mode="r").read())

  def testCopyOverwriteFalse(self):
    file_path = os.path.join(self._base_dir, "temp_file")
    file_io.FileIO(file_path, mode="w").write("testing")
    copy_path = os.path.join(self._base_dir, "copy_file")
    file_io.FileIO(copy_path, mode="w").write("copy")
    with self.assertRaises(errors.AlreadyExistsError):
      file_io.copy(file_path, copy_path, overwrite=False)

  def testRename(self):
    file_path = os.path.join(self._base_dir, "temp_file")
    file_io.FileIO(file_path, mode="w").write("testing")
    rename_path = os.path.join(self._base_dir, "rename_file")
    file_io.rename(file_path, rename_path)
    self.assertTrue(file_io.file_exists(rename_path))
    self.assertFalse(file_io.file_exists(file_path))

  def testRenameOverwrite(self):
    file_path = os.path.join(self._base_dir, "temp_file")
    file_io.FileIO(file_path, mode="w").write("testing")
    rename_path = os.path.join(self._base_dir, "rename_file")
    file_io.FileIO(rename_path, mode="w").write("rename")
    file_io.rename(file_path, rename_path, overwrite=True)
    self.assertTrue(file_io.file_exists(rename_path))
    self.assertFalse(file_io.file_exists(file_path))

  def testRenameOverwriteFalse(self):
    file_path = os.path.join(self._base_dir, "temp_file")
    file_io.FileIO(file_path, mode="w").write("testing")
    rename_path = os.path.join(self._base_dir, "rename_file")
    file_io.FileIO(rename_path, mode="w").write("rename")
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
    self.assertFalse(file_io.is_directory(dir_path))
    file_io.create_dir(dir_path)
    self.assertTrue(file_io.is_directory(dir_path))
    file_path = os.path.join(dir_path, "test_file")
    file_io.FileIO(file_path, mode="w").write("test")
    # False for a file.
    self.assertFalse(file_io.is_directory(file_path))
    # Test that the value returned from `stat()` has `is_directory` set.
    file_statistics = file_io.stat(dir_path)
    self.assertTrue(file_statistics.is_directory)

  def testListDirectory(self):
    dir_path = os.path.join(self._base_dir, "test_dir")
    file_io.create_dir(dir_path)
    files = ["file1.txt", "file2.txt", "file3.txt"]
    for name in files:
      file_path = os.path.join(dir_path, name)
      file_io.FileIO(file_path, mode="w").write("testing")
    subdir_path = os.path.join(dir_path, "sub_dir")
    file_io.create_dir(subdir_path)
    subdir_file_path = os.path.join(subdir_path, "file4.txt")
    file_io.FileIO(subdir_file_path, mode="w").write("testing")
    dir_list = file_io.list_directory(dir_path)
    self.assertItemsEqual(files + ["sub_dir"], dir_list)

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
    file_io.FileIO(
        os.path.join(dir_path, "file1.txt"), mode="w").write("testing")
    sub_dirs1 = ["subdir1_1", "subdir1_2", "subdir1_3"]
    for name in sub_dirs1:
      file_io.create_dir(os.path.join(dir_path, name))
    file_io.FileIO(
        os.path.join(dir_path, "subdir1_1/file2.txt"),
        mode="w").write("testing")
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
    self.assertItemsEqual(all_dirs, [dir_path] + [
        os.path.join(dir_path, item)
        for item in
        ["subdir1_1", "subdir1_2", "subdir1_2/subdir2", "subdir1_3"]
    ])
    self.assertEqual(dir_path, all_dirs[0])
    self.assertLess(
        all_dirs.index(os.path.join(dir_path, "subdir1_2")),
        all_dirs.index(os.path.join(dir_path, "subdir1_2/subdir2")))
    self.assertItemsEqual(all_subdirs[1:5], [[], ["subdir2"], [], []])
    self.assertItemsEqual(all_subdirs[0],
                          ["subdir1_1", "subdir1_2", "subdir1_3"])
    self.assertItemsEqual(all_files, [["file1.txt"], ["file2.txt"], [], [], []])
    self.assertLess(
        all_files.index(["file1.txt"]), all_files.index(["file2.txt"]))

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
        os.path.join(dir_path, item)
        for item in
        ["subdir1_1", "subdir1_2/subdir2", "subdir1_2", "subdir1_3"]
    ] + [dir_path])
    self.assertEqual(dir_path, all_dirs[4])
    self.assertLess(
        all_dirs.index(os.path.join(dir_path, "subdir1_2/subdir2")),
        all_dirs.index(os.path.join(dir_path, "subdir1_2")))
    self.assertItemsEqual(all_subdirs[0:4], [[], [], ["subdir2"], []])
    self.assertItemsEqual(all_subdirs[4],
                          ["subdir1_1", "subdir1_2", "subdir1_3"])
    self.assertItemsEqual(all_files, [["file2.txt"], [], [], [], ["file1.txt"]])
    self.assertLess(
        all_files.index(["file2.txt"]), all_files.index(["file1.txt"]))

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
    file_io.FileIO(file_path, mode="w").write("testing")
    file_statistics = file_io.stat(file_path)
    os_statistics = os.stat(file_path)
    self.assertEqual(7, file_statistics.length)
    self.assertEqual(
        int(os_statistics.st_mtime), int(file_statistics.mtime_nsec / 1e9))
    self.assertFalse(file_statistics.is_directory)

  def testReadLine(self):
    file_path = os.path.join(self._base_dir, "temp_file")
    with file_io.FileIO(file_path, mode="r+") as f:
      f.write("testing1\ntesting2\ntesting3\n\ntesting5")
    self.assertEqual(36, f.size())
    self.assertEqual("testing1\n", f.readline())
    self.assertEqual("testing2\n", f.readline())
    self.assertEqual("testing3\n", f.readline())
    self.assertEqual("\n", f.readline())
    self.assertEqual("testing5", f.readline())
    self.assertEqual("", f.readline())

  def testRead(self):
    file_path = os.path.join(self._base_dir, "temp_file")
    with file_io.FileIO(file_path, mode="r+") as f:
      f.write("testing1\ntesting2\ntesting3\n\ntesting5")
    self.assertEqual(36, f.size())
    self.assertEqual("testing1\n", f.read(9))
    self.assertEqual("testing2\n", f.read(9))
    self.assertEqual("t", f.read(1))
    self.assertEqual("esting3\n\ntesting5", f.read())

  def testTell(self):
    file_path = os.path.join(self._base_dir, "temp_file")
    with file_io.FileIO(file_path, mode="r+") as f:
      f.write("testing1\ntesting2\ntesting3\n\ntesting5")
    self.assertEqual(0, f.tell())
    self.assertEqual("testing1\n", f.readline())
    self.assertEqual(9, f.tell())
    self.assertEqual("testing2\n", f.readline())
    self.assertEqual(18, f.tell())
    self.assertEqual("testing3\n", f.readline())
    self.assertEqual(27, f.tell())
    self.assertEqual("\n", f.readline())
    self.assertEqual(28, f.tell())
    self.assertEqual("testing5", f.readline())
    self.assertEqual(36, f.tell())
    self.assertEqual("", f.readline())
    self.assertEqual(36, f.tell())

  def testSeek(self):
    file_path = os.path.join(self._base_dir, "temp_file")
    with file_io.FileIO(file_path, mode="r+") as f:
      f.write("testing1\ntesting2\ntesting3\n\ntesting5")
    self.assertEqual("testing1\n", f.readline())
    self.assertEqual(9, f.tell())

    # Seek to 18
    f.seek(18)
    self.assertEqual(18, f.tell())
    self.assertEqual("testing3\n", f.readline())

    # Seek back to 9
    f.seek(9)
    self.assertEqual(9, f.tell())
    self.assertEqual("testing2\n", f.readline())

    f.seek(0)
    self.assertEqual(0, f.tell())
    self.assertEqual("testing1\n", f.readline())

    with self.assertRaises(errors.InvalidArgumentError):
      f.seek(-1)

    with self.assertRaises(TypeError):
      f.seek()

    # TODO(jhseu): Delete after position deprecation.
    with self.assertRaises(TypeError):
      f.seek(offset=0, position=0)
    f.seek(position=9)
    self.assertEqual(9, f.tell())
    self.assertEqual("testing2\n", f.readline())

  def testSeekFromWhat(self):
    file_path = os.path.join(self._base_dir, "temp_file")
    with file_io.FileIO(file_path, mode="r+") as f:
      f.write("testing1\ntesting2\ntesting3\n\ntesting5")
    self.assertEqual("testing1\n", f.readline())
    self.assertEqual(9, f.tell())

    # Seek to 18
    f.seek(9, 1)
    self.assertEqual(18, f.tell())
    self.assertEqual("testing3\n", f.readline())

    # Seek back to 9
    f.seek(9, 0)
    self.assertEqual(9, f.tell())
    self.assertEqual("testing2\n", f.readline())

    f.seek(-f.size(), 2)
    self.assertEqual(0, f.tell())
    self.assertEqual("testing1\n", f.readline())

    with self.assertRaises(errors.InvalidArgumentError):
      f.seek(0, 3)

  def testReadingIterator(self):
    file_path = os.path.join(self._base_dir, "temp_file")
    data = ["testing1\n", "testing2\n", "testing3\n", "\n", "testing5"]
    with file_io.FileIO(file_path, mode="r+") as f:
      f.write("".join(data))
    actual_data = []
    for line in f:
      actual_data.append(line)
    self.assertSequenceEqual(actual_data, data)

  def testReadlines(self):
    file_path = os.path.join(self._base_dir, "temp_file")
    data = ["testing1\n", "testing2\n", "testing3\n", "\n", "testing5"]
    f = file_io.FileIO(file_path, mode="r+")
    f.write("".join(data))
    f.flush()
    lines = f.readlines()
    self.assertSequenceEqual(lines, data)

  def testEof(self):
    """Test that reading past EOF does not raise an exception."""

    file_path = os.path.join(self._base_dir, "temp_file")
    f = file_io.FileIO(file_path, mode="r+")
    content = "testing"
    f.write(content)
    f.flush()
    self.assertEqual(content, f.read(len(content) + 1))


if __name__ == "__main__":
  test.main()
