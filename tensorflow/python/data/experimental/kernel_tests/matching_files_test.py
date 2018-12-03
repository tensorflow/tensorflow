# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the private `MatchingFilesDataset`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile

from tensorflow.python.data.experimental.ops import matching_files
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class MatchingFilesTest(test_base.DatasetTestBase):

  def setUp(self):
    self.tmp_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.tmp_dir, ignore_errors=True)

  def _touchTempFiles(self, filenames):
    for filename in filenames:
      open(os.path.join(self.tmp_dir, filename), 'a').close()

  @test_util.run_deprecated_v1
  def testNonExistingDirectory(self):
    """Test the MatchingFiles dataset with a non-existing directory."""

    self.tmp_dir = os.path.join(self.tmp_dir, 'nonexistingdir')
    dataset = matching_files.MatchingFilesDataset(
        os.path.join(self.tmp_dir, '*'))
    with self.cached_session() as sess:
      next_element = dataset.make_one_shot_iterator().get_next()
      with self.assertRaises(errors.NotFoundError):
        sess.run(next_element)

  @test_util.run_deprecated_v1
  def testEmptyDirectory(self):
    """Test the MatchingFiles dataset with an empty directory."""

    dataset = matching_files.MatchingFilesDataset(
        os.path.join(self.tmp_dir, '*'))
    with self.cached_session() as sess:
      next_element = dataset.make_one_shot_iterator().get_next()
      with self.assertRaises(errors.NotFoundError):
        sess.run(next_element)

  @test_util.run_deprecated_v1
  def testSimpleDirectory(self):
    """Test the MatchingFiles dataset with a simple directory."""

    filenames = ['a', 'b', 'c']
    self._touchTempFiles(filenames)

    dataset = matching_files.MatchingFilesDataset(
        os.path.join(self.tmp_dir, '*'))
    with self.cached_session() as sess:
      next_element = dataset.make_one_shot_iterator().get_next()

      expected_filenames = []
      actual_filenames = []
      for filename in filenames:
        expected_filenames.append(
            compat.as_bytes(os.path.join(self.tmp_dir, filename)))
        actual_filenames.append(compat.as_bytes(sess.run(next_element)))

      self.assertItemsEqual(expected_filenames, actual_filenames)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  @test_util.run_deprecated_v1
  def testFileSuffixes(self):
    """Test the MatchingFiles dataset using the suffixes of filename."""

    filenames = ['a.txt', 'b.py', 'c.py', 'd.pyc']
    self._touchTempFiles(filenames)

    dataset = matching_files.MatchingFilesDataset(
        os.path.join(self.tmp_dir, '*.py'))
    with self.cached_session() as sess:
      next_element = dataset.make_one_shot_iterator().get_next()
      expected_filenames = []
      actual_filenames = []
      for filename in filenames[1:-1]:
        expected_filenames.append(
            compat.as_bytes(os.path.join(self.tmp_dir, filename)))
        actual_filenames.append(compat.as_bytes(sess.run(next_element)))

      self.assertItemsEqual(expected_filenames, actual_filenames)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  @test_util.run_deprecated_v1
  def testFileMiddles(self):
    """Test the MatchingFiles dataset using the middles of filename."""

    filenames = ['aa.txt', 'bb.py', 'bbc.pyc', 'cc.pyc']
    self._touchTempFiles(filenames)

    dataset = matching_files.MatchingFilesDataset(
        os.path.join(self.tmp_dir, 'b*.py*'))
    with self.cached_session() as sess:
      next_element = dataset.make_one_shot_iterator().get_next()
      expected_filenames = []
      actual_filenames = []
      for filename in filenames[1:3]:
        expected_filenames.append(
            compat.as_bytes(os.path.join(self.tmp_dir, filename)))
        actual_filenames.append(compat.as_bytes(sess.run(next_element)))

      self.assertItemsEqual(expected_filenames, actual_filenames)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  @test_util.run_deprecated_v1
  def testNestedDirectories(self):
    """Test the MatchingFiles dataset with nested directories."""

    filenames = []
    width = 8
    depth = 4
    for i in range(width):
      for j in range(depth):
        new_base = os.path.join(self.tmp_dir, str(i),
                                *[str(dir_name) for dir_name in range(j)])
        os.makedirs(new_base)
        child_files = ['a.py', 'b.pyc'] if j < depth - 1 else ['c.txt', 'd.log']
        for f in child_files:
          filename = os.path.join(new_base, f)
          filenames.append(filename)
          open(filename, 'w').close()

    patterns = [
        os.path.join(self.tmp_dir, os.path.join(*['**' for _ in range(depth)]),
                     suffix) for suffix in ['*.txt', '*.log']
    ]

    dataset = matching_files.MatchingFilesDataset(patterns)
    with self.cached_session() as sess:
      next_element = dataset.make_one_shot_iterator().get_next()
      expected_filenames = [
          compat.as_bytes(filename)
          for filename in filenames
          if filename.endswith('.txt') or filename.endswith('.log')
      ]
      actual_filenames = []
      while True:
        try:
          actual_filenames.append(compat.as_bytes(sess.run(next_element)))
        except errors.OutOfRangeError:
          break

      self.assertItemsEqual(expected_filenames, actual_filenames)


if __name__ == '__main__':
  test.main()
