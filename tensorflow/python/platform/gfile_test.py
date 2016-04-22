# Copyright 2015 Google Inc. All Rights Reserved.
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

import contextlib
import os
import shutil
import time

from tensorflow.python.platform import gfile
from tensorflow.python.platform import googletest
from tensorflow.python.platform import tf_logging as logging


class _BaseTest(object):

  @property
  def tmp(self):
    return self._tmp_dir

  def setUp(self):
    self._orig_dir = os.getcwd()
    self._tmp_dir = googletest.GetTempDir() + "/"
    try:
      os.makedirs(self._tmp_dir)
    except OSError:
      pass  # Directory already exists

  def tearDown(self):
    try:
      shutil.rmtree(self._tmp_dir)
    except OSError:
      logging.warn("[%s] Post-test directory cleanup failed: %s",
                   self, self._tmp_dir)


class _GFileBaseTest(_BaseTest):

  @property
  def gfile(self):
    raise NotImplementedError("Do not use _GFileBaseTest directly.")

  def testWith(self):
    with self.gfile(self.tmp + "test_with", "w") as fh:
      fh.write("hi")
    with self.gfile(self.tmp + "test_with", "r") as fh:
      self.assertEqual(fh.read(), "hi")

  def testSizeAndTellAndSeek(self):
    with self.gfile(self.tmp + "test_tell", "w") as fh:
      fh.write("".join(["0"] * 1000))
    with self.gfile(self.tmp + "test_tell", "r") as fh:
      self.assertEqual(1000, fh.Size())
      self.assertEqual(0, fh.tell())
      fh.seek(0, 2)
      self.assertEqual(1000, fh.tell())
      fh.seek(0)
      self.assertEqual(0, fh.tell())

  def testReadAndWritelines(self):
    with self.gfile(self.tmp + "test_writelines", "w") as fh:
      fh.writelines(["%d\n" % d for d in range(10)])
    with self.gfile(self.tmp + "test_writelines", "r") as fh:
      self.assertEqual(["%d\n" % x for x in range(10)], fh.readlines())

  def testWriteAndTruncate(self):
    with self.gfile(self.tmp + "test_truncate", "w") as fh:
      fh.write("ababab")
    with self.gfile(self.tmp + "test_truncate", "a+") as fh:
      fh.seek(0, 2)
      fh.write("hjhjhj")
    with self.gfile(self.tmp + "test_truncate", "a+") as fh:
      self.assertEqual(fh.Size(), 12)
      fh.truncate(6)
    with self.gfile(self.tmp + "test_truncate", "r") as fh:
      self.assertEqual(fh.read(), "ababab")

  def testErrors(self):
    self.assertRaises(
        IOError, lambda: self.gfile(self.tmp + "doesnt_exist", "r"))
    with self.gfile(self.tmp + "test_error", "w") as fh:
      # Raises FileError inside Google and ValueError outside, so we
      # can only test for Exception.
      self.assertRaises(Exception, lambda: fh.seek(-1))
    # test_error now exists, we can read from it:
    with self.gfile(self.tmp + "test_error", "r") as fh:
      self.assertRaises(IOError, lambda: fh.write("ack"))
    fh = self.gfile(self.tmp + "test_error", "w")
    self.assertFalse(fh.closed)
    fh.close()
    self.assertTrue(fh.closed)
    self.assertRaises(ValueError, lambda: fh.write("ack"))

  def testIteration(self):
    with self.gfile(self.tmp + "test_iter", "w") as fh:
      fh.writelines(["a\n", "b\n", "c\n"])
    with self.gfile(self.tmp + "test_iter", "r") as fh:
      lines = list(fh)
      self.assertEqual(["a\n", "b\n", "c\n"], lines)


class GFileTest(_GFileBaseTest, googletest.TestCase):

  @property
  def gfile(self):
    return gfile.GFile


class FastGFileTest(_GFileBaseTest, googletest.TestCase):

  @property
  def gfile(self):
    return gfile.FastGFile


class FunctionTests(_BaseTest, googletest.TestCase):

  def testExists(self):
    self.assertFalse(gfile.Exists(self.tmp + "test_exists"))
    with gfile.GFile(self.tmp + "test_exists", "w"):
      pass
    self.assertTrue(gfile.Exists(self.tmp + "test_exists"))

  def testMkDirsGlobAndRmDirs(self):
    self.assertFalse(gfile.Exists(self.tmp + "test_dir"))
    gfile.MkDir(self.tmp + "test_dir")
    self.assertTrue(gfile.Exists(self.tmp + "test_dir"))
    gfile.RmDir(self.tmp + "test_dir")
    self.assertFalse(gfile.Exists(self.tmp + "test_dir"))
    gfile.MakeDirs(self.tmp + "test_dir/blah0")
    gfile.MakeDirs(self.tmp + "test_dir/blah1")
    self.assertEqual([self.tmp + "test_dir/blah0", self.tmp + "test_dir/blah1"],
                     sorted(gfile.Glob(self.tmp + "test_dir/*")))
    gfile.DeleteRecursively(self.tmp + "test_dir")
    self.assertFalse(gfile.Exists(self.tmp + "test_dir"))

  @contextlib.contextmanager
  def _working_directory(self, wd):
    original_cwd = os.getcwd()
    os.chdir(wd)
    try:
      yield
    finally:
      os.chdir(original_cwd)

  def testMakeDirsWithEmptyString(self):
    gfile.MakeDirs(self.tmp + "test_dir")
    with self._working_directory(self.tmp + "test_dir"):
      gfile.MakeDirs("")
    # Should succeed because MakeDirs("") is a no-op.
    gfile.RmDir(self.tmp + "test_dir")

  def testErrors(self):
    self.assertRaises(
        OSError, lambda: gfile.RmDir(self.tmp + "dir_doesnt_exist"))
    self.assertRaises(
        OSError, lambda: gfile.Remove(self.tmp + "file_doesnt_exist"))
    gfile.MkDir(self.tmp + "error_dir")
    with gfile.GFile(self.tmp + "error_dir/file", "w"):
      pass  # Create file
    self.assertRaises(
        OSError, lambda: gfile.Remove(self.tmp + "error_dir"))
    self.assertRaises(
        OSError, lambda: gfile.RmDir(self.tmp + "error_dir"))
    self.assertTrue(gfile.Exists(self.tmp + "error_dir"))
    gfile.DeleteRecursively(self.tmp + "error_dir")
    self.assertFalse(gfile.Exists(self.tmp + "error_dir"))

  def testStat(self):
    with gfile.GFile(self.tmp + "test_stat", "w"):
      pass
    creation_time = time.time()
    statinfo = gfile.Stat(self.tmp + "test_stat")
    # Test the modification timestamp is within 20 seconds of closing the file.
    self.assertLessEqual(statinfo.mtime, creation_time + 10)
    self.assertGreaterEqual(statinfo.mtime, creation_time - 10)

  def testRename(self):
    gfile.MkDir(self.tmp + "dir1")
    gfile.MkDir(self.tmp + "dir2")
    with gfile.GFile(self.tmp + "file1", "w"):
      pass  # Create file
    with gfile.GFile(self.tmp + "file2", "w"):
      pass  # Create file

    # Dest file already exists, overwrite=False (default).
    self.assertRaises(
        OSError, lambda: gfile.Rename(self.tmp + "file1", self.tmp + "file2"))
    gfile.Rename(self.tmp + "file1", self.tmp + "file2", overwrite=True)
    self.assertFalse(gfile.Exists(self.tmp + "file1"))
    gfile.Rename(self.tmp + "file2", self.tmp + "newfile")
    self.assertTrue(gfile.Exists(self.tmp + "newfile"))

    gfile.Rename(self.tmp + "dir1", self.tmp + "dir2")
    self.assertFalse(gfile.Exists(self.tmp + "dir1"))
    gfile.Rename(self.tmp + "dir2", self.tmp + "newdir")
    self.assertTrue(gfile.Exists(self.tmp + "newdir"))

  def testCopy(self):
    gfile.MkDir(self.tmp + "dir1")
    gfile.MkDir(self.tmp + "dir2")
    with gfile.GFile(self.tmp + "dir1/file1", "w"):
      pass  # Create file
    with gfile.GFile(self.tmp + "dir2/file2", "w"):
      pass  # Create file

    # Dest file already exists, overwrite=False (default).
    self.assertRaises(
        OSError, lambda: gfile.Copy(self.tmp + "dir1/file1",
                                    self.tmp + "dir2/file2"))
    # Overwrite succeeds
    gfile.Copy(self.tmp + "dir1/file1", self.tmp + "dir2/file2",
               overwrite=True)
    self.assertTrue(gfile.Exists(self.tmp + "dir2/file2"))

    # Normal copy.
    gfile.Rename(self.tmp + "dir1/file1", self.tmp + "dir2/file1")
    self.assertTrue(gfile.Exists(self.tmp + "dir2/file1"))

    # Normal copy to non-existent dir
    self.assertRaises(OSError,
                      lambda: gfile.Rename(self.tmp + "dir1/file1",
                                           self.tmp + "newdir/file1"))

  def testOpen(self):
    with gfile.Open(self.tmp + "test_open", "wb") as f:
      f.write(b"foo")
    with gfile.Open(self.tmp + "test_open") as f:
      result = f.readlines()
    self.assertEqual(["foo"], result)

if __name__ == "__main__":
  googletest.main()
