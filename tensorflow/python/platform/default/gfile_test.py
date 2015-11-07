import os
import shutil

from tensorflow.python.platform.default import _gfile as gfile
from tensorflow.python.platform.default import _googletest as googletest
from tensorflow.python.platform.default import _logging as logging


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
      logging.warn("[%s] Post-test directory cleanup failed: %s"
                   % (self, self._tmp_dir))


class _GFileBaseTest(_BaseTest):

  @property
  def gfile(self):
    raise NotImplementedError("Do not use _GFileBaseTest directly.")

  def testWith(self):
    with self.gfile(self.tmp + "test_with", "w") as fh:
      fh.write("hi")
    with self.gfile(self.tmp + "test_with", "r") as fh:
      self.assertEquals(fh.read(), "hi")

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
        gfile.FileError, lambda: self.gfile(self.tmp + "doesnt_exist", "r"))
    with self.gfile(self.tmp + "test_error", "w") as fh:
      self.assertRaises(gfile.FileError, lambda: fh.seek(-1))
    # test_error now exists, we can read from it:
    with self.gfile(self.tmp + "test_error", "r") as fh:
      self.assertRaises(gfile.FileError, lambda: fh.write("ack"))
    fh = self.gfile(self.tmp + "test_error", "w")
    self.assertFalse(fh.closed)
    fh.close()
    self.assertTrue(fh.closed)
    self.assertRaises(gfile.FileError, lambda: fh.write("ack"))

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

  def testErrors(self):
    self.assertRaises(
        gfile.GOSError, lambda: gfile.RmDir(self.tmp + "dir_doesnt_exist"))
    self.assertRaises(
        gfile.GOSError, lambda: gfile.Remove(self.tmp + "file_doesnt_exist"))
    gfile.MkDir(self.tmp + "error_dir")
    with gfile.GFile(self.tmp + "error_dir/file", "w"):
      pass  # Create file
    self.assertRaises(
        gfile.GOSError, lambda: gfile.Remove(self.tmp + "error_dir"))
    self.assertRaises(
        gfile.GOSError, lambda: gfile.RmDir(self.tmp + "error_dir"))
    self.assertTrue(gfile.Exists(self.tmp + "error_dir"))
    gfile.DeleteRecursively(self.tmp + "error_dir")
    self.assertFalse(gfile.Exists(self.tmp + "error_dir"))


if __name__ == "__main__":
  googletest.main()
