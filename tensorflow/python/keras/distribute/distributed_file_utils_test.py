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
"""Tests for distributed_file_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.keras.distribute import distributed_file_utils
from tensorflow.python.platform import test


class DistributedFileUtilsTest(test.TestCase):

  class MockedExtended(object):
    pass

  class MockedChiefStrategy(object):

    def __init__(self):
      self.extended = DistributedFileUtilsTest.MockedExtended()
      self.extended._in_multi_worker_mode = lambda: True
      self.extended.should_checkpoint = True

  class MockedWorkerStrategy(object):

    def __init__(self):
      self.extended = DistributedFileUtilsTest.MockedExtended()
      self.extended._in_multi_worker_mode = lambda: True
      self.extended.should_checkpoint = False
      self.extended._task_id = 3

  class MockedSingleWorkerStrategy(object):

    def __init__(self):
      self.extended = DistributedFileUtilsTest.MockedExtended()
      self.extended._in_multi_worker_mode = lambda: False

  def _write_dummy_file(self, file_to_write):
    with open(file_to_write, 'w') as f:
      f.write('foo bar')

  def testChiefWriteDirAndFilePath(self):
    dirpath = self.get_temp_dir()
    filepath = os.path.join(dirpath, 'foo.bar')
    strategy = DistributedFileUtilsTest.MockedChiefStrategy()
    self.assertEqual(
        distributed_file_utils.write_filepath(filepath, strategy), filepath)
    self.assertEqual(
        distributed_file_utils.write_dirpath(dirpath, strategy), dirpath)

  def testWorkerWriteDirAndFilePath(self):
    dirpath = self.get_temp_dir()
    filepath = os.path.join(dirpath, 'foo.bar')
    strategy = DistributedFileUtilsTest.MockedWorkerStrategy()
    self.assertEqual(
        distributed_file_utils.write_filepath(filepath, strategy),
        os.path.join(dirpath, 'workertemp_3', 'foo.bar'))
    self.assertEqual(
        distributed_file_utils.write_dirpath(dirpath, strategy),
        os.path.join(dirpath, 'workertemp_3'))

  def testChiefDoesNotRemoveDirAndFilePath(self):
    temp_dir = self.get_temp_dir()
    strategy = DistributedFileUtilsTest.MockedChiefStrategy()
    dir_to_write = distributed_file_utils.write_dirpath(temp_dir, strategy)
    file_to_write = os.path.join(dir_to_write, 'tmp')
    self.assertFalse(os.path.exists(file_to_write))
    self._write_dummy_file(file_to_write)
    self.assertTrue(os.path.exists(file_to_write))
    distributed_file_utils.remove_temp_dir_with_filepath(
        file_to_write, strategy)
    self.assertTrue(os.path.exists(file_to_write))

  def testWorkerDoesRemoveFilePath(self):
    temp_dir = self.get_temp_dir()
    strategy = DistributedFileUtilsTest.MockedWorkerStrategy()
    dir_to_write = distributed_file_utils.write_dirpath(temp_dir, strategy)
    file_to_write = os.path.join(dir_to_write, 'tmp')
    self.assertFalse(os.path.exists(file_to_write))
    self._write_dummy_file(file_to_write)
    self.assertTrue(os.path.exists(file_to_write))
    distributed_file_utils.remove_temp_dir_with_filepath(
        file_to_write, strategy)
    self.assertFalse(os.path.exists(file_to_write))

  def testWorkerDoesRemoveDirPath(self):
    temp_dir = self.get_temp_dir()
    strategy = DistributedFileUtilsTest.MockedWorkerStrategy()
    dir_to_write = distributed_file_utils.write_dirpath(temp_dir, strategy)
    file_to_write = os.path.join(dir_to_write, 'tmp')
    self.assertFalse(os.path.exists(file_to_write))
    self._write_dummy_file(file_to_write)
    self.assertTrue(os.path.exists(file_to_write))
    distributed_file_utils.remove_temp_dirpath(temp_dir, strategy)
    self.assertFalse(os.path.exists(file_to_write))
    self.assertFalse(os.path.exists(os.path.dirname(file_to_write)))

  def testMultipleRemoveOrigDirPathIsFine(self):
    temp_dir = self.get_temp_dir()
    strategy = DistributedFileUtilsTest.MockedWorkerStrategy()
    dir_to_write = distributed_file_utils.write_dirpath(temp_dir, strategy)
    file_to_write = os.path.join(dir_to_write, 'tmp')
    self._write_dummy_file(file_to_write)
    distributed_file_utils.remove_temp_dirpath(temp_dir, strategy)
    distributed_file_utils.remove_temp_dirpath(temp_dir, strategy)
    distributed_file_utils.remove_temp_dirpath(temp_dir, strategy)

  def testMultipleRemoveDirToWritePathIsFine(self):
    temp_dir = self.get_temp_dir()
    strategy = DistributedFileUtilsTest.MockedWorkerStrategy()
    dir_to_write = distributed_file_utils.write_dirpath(temp_dir, strategy)
    file_to_write = os.path.join(dir_to_write, 'tmp')
    self._write_dummy_file(file_to_write)
    distributed_file_utils.remove_temp_dirpath(dir_to_write, strategy)
    distributed_file_utils.remove_temp_dirpath(dir_to_write, strategy)
    distributed_file_utils.remove_temp_dirpath(dir_to_write, strategy)


if __name__ == '__main__':
  test.main()
