# This Python file uses the following encoding: utf-8
# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Portable recursive glob tests, including Windows path semantics."""

import collections
import contextlib
import glob
import ntpath
import os
import re
import tempfile
import types
from unittest import mock
import uuid

from absl.testing import parameterized

from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import test


class _WindowsFileSystem:
  """Small filesystem model with native Windows paths and wildcard behavior.

  Only the filesystem boundary is mocked. Calls go through the real public
  glob function and recursive traversal, including native result normalization.
  The Windows backend accepts either separator and returns backslashes; unlike
  POSIX glob, its PathMatchSpec matcher treats brackets literally.
  """

  def __init__(self, root):
    self.root = ntpath.normpath(root)
    self.directories = {
        self.path(name) for name in ("", "sub", "sub/nested", ".hidden",
                                    "bracket[1]")
    }
    self.files = {
        self.path(name) for name in ("root.txt", "sub/sub.txt",
                                    "sub/nested/deep.txt", ".hidden/secret.txt",
                                    ".secret.txt", "bracket[1]/literal.txt")
    }

  def path(self, name):
    return ntpath.normpath(ntpath.join(self.root, name))

  def is_directory(self, path):
    if re.match(r"^[a-zA-Z]://", path):
      raise AssertionError("A Windows drive root must not become a URI")
    return ntpath.normpath(path) in self.directories

  def children(self, path):
    path = ntpath.normpath(path)
    return [ntpath.basename(child)
            for child in self.directories | self.files
            if child != path and ntpath.normpath(ntpath.dirname(child)) == path]

  def matching_files(self, pattern):
    pattern = pattern.decode("utf-8")
    normalized = ntpath.normpath(pattern)
    components = normalized.split("\\")
    # Matching each component separately keeps '*' from crossing a separator.
    expressions = [re.compile(re.escape(part).replace(r"\*", ".*")
                              .replace(r"\?", "."), re.IGNORECASE)
                   for part in components]
    result = []
    for path in self.directories | self.files:
      parts = path.split("\\")
      if len(parts) == len(expressions) and all(
          expression.fullmatch(part)
          for expression, part in zip(expressions, parts)):
        if pattern.startswith(("./", ".\\")) and not path.startswith(".\\"):
          path = ".\\" + path
        result.append(path)
    return sorted(result)


@contextlib.contextmanager
def _windows_filesystem(root):
  fs = _WindowsFileSystem(root)
  windows_os = types.SimpleNamespace(name="nt", path=ntpath, sep="\\")
  with mock.patch.object(file_io, "os", windows_os), mock.patch.object(
      file_io, "is_directory_v2", fs.is_directory), mock.patch.object(
          file_io, "list_directory_v2", fs.children), mock.patch.object(
              file_io._pywrap_file_io, "GetMatchingFiles", fs.matching_files):
    yield fs


class RecursiveGlobTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    directory = tempfile.TemporaryDirectory(prefix="tf-recursive-glob-")
    self.addCleanup(directory.cleanup)
    self.root = directory.name
    for name in ("root.txt", "sub/sub.txt", "sub/nested/deep.txt"):
      path = file_io.join(self.root, *name.split("/"))
      file_io.recursive_create_dir(os.path.dirname(path))
      file_io.write_string_to_file(path, "test")

  def testNativeFilesystem(self):
    patterns = [file_io.join(self.root, "**", "*.txt")]
    if os.name == "nt":
      patterns.append(patterns[0].replace("\\", "/"))
      patterns.append(self.root + "/**\\*.txt")
    expected = [file_io.join(self.root, *name.split("/"))
                for name in ("root.txt", "sub/sub.txt", "sub/nested/deep.txt")]
    for pattern in patterns:
      with self.subTest(pattern=pattern):
        self.assertCountEqual(
            expected, file_io.get_matching_files_v2(pattern, recursive=True))

  def testZeroDepthAndIterablePatterns(self):
    pattern = file_io.join(self.root, "**", "*.txt")
    expected = glob.glob(pattern, recursive=True)
    self.assertEqual(
        [file_io.join(self.root, "root.txt")],
        file_io.get_matching_files_v2(
            file_io.join(self.root, "**", "root.txt"), recursive=True))
    self.assertCountEqual(
        expected, file_io.get_matching_files_v2([pattern], recursive=True))
    self.assertCountEqual(
        expected,
        file_io.get_matching_files_v2(iter([pattern]), recursive=True))
    self.assertEmpty(file_io.get_matching_files_v2(
        file_io.join(self.root, "**", "deep.txt"), recursive=False))

  def testGlobstarClassifiesEachChildOnce(self):
    # Isolate traversal classification from listdir's own validation checks.
    with _windows_filesystem(r"C:\data") as fs:
      with mock.patch.object(
          file_io, "is_directory_v2", wraps=file_io.is_directory_v2) as is_dir:
        self.assertEqual(
            [fs.path("sub/nested/deep.txt")],
            file_io.get_matching_files_v2(
                r"C:\data\**\deep.txt", recursive=True))
      calls = collections.Counter(
          ntpath.normpath(call.args[0]) for call in is_dir.call_args_list)
      for relative in ("root.txt", "sub", "sub/sub.txt", "sub/nested",
                       "sub/nested/deep.txt", "bracket[1]",
                       "bracket[1]/literal.txt"):
        path = fs.path(relative)
        with self.subTest(path=path):
          self.assertEqual(1, calls[path])

  def testRelativePaths(self):
    cwd = os.getcwd()
    try:
      os.chdir(self.root)
      for pattern in ("**/*.txt", "./**/*.txt", "sub/**/deep.txt"):
        expected = glob.glob(pattern, recursive=True)
        actual = file_io.get_matching_files_v2(pattern, recursive=True)
        self.assertCountEqual(
            [os.path.normpath(path) for path in expected],
            [os.path.normpath(path) for path in actual])
    finally:
      os.chdir(cwd)

  @parameterized.parameters(
      (r"C:\data", r"C:\data\**\*.txt"),
      (r"C:\data", "C:/data/**/*.txt"),
      (r"C:\data", r"C:\data/**\*.txt"),
      (r"C:\data", r"C:\d*\**\*.txt"),
      ("C:\\", r"C:\**\*.txt"),
      (r"\\server\share\data", r"\\server\share\data\**\*.txt"),
      (r"\\server\share\data", "//server/share/data/**/*.txt"),
      ("data", r"data\**\*.txt"),
      ("C:", r"C:**\*.txt"),
  )
  def testWindowsTraversal(self, root, pattern):
    with _windows_filesystem(root) as fs:
      expected = [fs.path(name) for name in ("root.txt", "sub/sub.txt",
                                            "sub/nested/deep.txt",
                                            "bracket[1]/literal.txt")]
      self.assertCountEqual(
          expected, file_io.get_matching_files_v2(pattern, recursive=True))

  @parameterized.parameters(r"C:\data", "C:\\")
  def testWindowsDirectoryOnlyAndTerminalGlobstar(self, root):
    with _windows_filesystem(root) as fs:
      self.assertCountEqual(
          [ntpath.join(fs.path(name), "") for name in ("", "sub", "sub/nested",
                                                       "bracket[1]")],
          file_io.get_matching_files_v2(
              ntpath.join(root, "**", ""), recursive=True))
      actual = file_io.get_matching_files_v2(
          ntpath.join(root, "**"), recursive=True)
      self.assertCountEqual(
          [ntpath.join(fs.root, "")] + [fs.path(name) for name in (
              "sub", "sub/nested", "bracket[1]", "root.txt", "sub/sub.txt",
              "sub/nested/deep.txt", "bracket[1]/literal.txt")], actual)

  def testWindowsHiddenAndDefault(self):
    with _windows_filesystem(r"C:\data") as fs:
      self.assertEqual(
          [fs.path(".hidden/secret.txt")], file_io.get_matching_files_v2(
              r"C:\data\**\.hidden\*.txt", recursive=True))
      self.assertEmpty(file_io.get_matching_files_v2(
          r"C:\data\**\deep.txt", recursive=False))
      pattern = r"C:\data\*.txt"
      self.assertEqual(file_io.get_matching_files_v2(pattern),
                       file_io.get_matching_files_v2(pattern, recursive=True))

  def testWindowsRelativeAndDriveRelative(self):
    for root, pattern in ((".", r"**\*.txt"), (".", r".\**\*.txt"),
                          ("C:", r"C:**\root.txt")):
      with self.subTest(pattern=pattern), _windows_filesystem(root) as fs:
        names = (["root.txt"] if root == "C:" else
                 ["root.txt", "sub/sub.txt", "sub/nested/deep.txt",
                  "bracket[1]/literal.txt"])
        expected = [fs.path(name) for name in names]
        if pattern.startswith(".\\"):
          expected = [".\\" + path for path in expected]
        self.assertCountEqual(
            expected, file_io.get_matching_files_v2(pattern, recursive=True))

  def testPosixLiteralBackslash(self):
    if os.name == "nt":
      self.skipTest("POSIX literal backslash filenames")
    directory = file_io.join(self.root, "literal\\directory")
    file_io.recursive_create_dir(directory)
    path = file_io.join(directory, "data.txt")
    file_io.write_string_to_file(path, "test")
    self.assertEqual([path], file_io.get_matching_files_v2(
        file_io.join(self.root, "**", "data.txt"), recursive=True))

  def testRegisteredFilesystemOnWindows(self):
    root = "ram://glob_windows_" + uuid.uuid4().hex
    path = root + "/a/b/data.txt"
    file_io.recursive_create_dir(root + "/a/b")
    file_io.write_string_to_file(path, "test")
    try:
      windows_os = types.SimpleNamespace(name="nt", path=ntpath, sep="\\")
      with mock.patch.object(file_io, "os", windows_os):
        self.assertEqual([path], file_io.get_matching_files_v2(
            root + "/**/*.txt", recursive=True))
    finally:
      file_io.delete_recursively(root)


if __name__ == "__main__":
  test.main()
