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
# ==============================================================================

"""Tests for directory_watcher."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

import tensorflow as tf

from tensorflow.tensorboard.backend.event_processing import directory_watcher
from tensorflow.tensorboard.backend.event_processing import io_wrapper


class _ByteLoader(object):
  """A loader that loads individual bytes from a file."""

  def __init__(self, path):
    self._f = open(path)
    self.bytes_read = 0

  def Load(self):
    while True:
      self._f.seek(self.bytes_read)
      byte = self._f.read(1)
      if byte:
        self.bytes_read += 1
        yield byte
      else:
        return


class DirectoryWatcherTest(tf.test.TestCase):

  def setUp(self):
    # Put everything in a directory so it's easier to delete.
    self._directory = os.path.join(self.get_temp_dir(), 'monitor_dir')
    os.mkdir(self._directory)
    self._watcher = directory_watcher.DirectoryWatcher(self._directory,
                                                       _ByteLoader)
    self.stubs = tf.test.StubOutForTesting()

  def tearDown(self):
    self.stubs.CleanUp()
    try:
      shutil.rmtree(self._directory)
    except OSError:
      # Some tests delete the directory.
      pass

  def _WriteToFile(self, filename, data):
    path = os.path.join(self._directory, filename)
    with open(path, 'a') as f:
      f.write(data)

  def _LoadAllEvents(self):
    """Loads all events in the watcher."""
    for _ in self._watcher.Load():
      pass

  def assertWatcherYields(self, values):
    self.assertEqual(list(self._watcher.Load()), values)

  def testRaisesWithBadArguments(self):
    with self.assertRaises(ValueError):
      directory_watcher.DirectoryWatcher(None, lambda x: None)
    with self.assertRaises(ValueError):
      directory_watcher.DirectoryWatcher('dir', None)

  def testEmptyDirectory(self):
    self.assertWatcherYields([])

  def testSingleWrite(self):
    self._WriteToFile('a', 'abc')
    self.assertWatcherYields(['a', 'b', 'c'])
    self.assertFalse(self._watcher.OutOfOrderWritesDetected())

  def testMultipleWrites(self):
    self._WriteToFile('a', 'abc')
    self.assertWatcherYields(['a', 'b', 'c'])
    self._WriteToFile('a', 'xyz')
    self.assertWatcherYields(['x', 'y', 'z'])
    self.assertFalse(self._watcher.OutOfOrderWritesDetected())

  def testMultipleLoads(self):
    self._WriteToFile('a', 'a')
    self._watcher.Load()
    self._watcher.Load()
    self.assertWatcherYields(['a'])
    self.assertFalse(self._watcher.OutOfOrderWritesDetected())

  def testMultipleFilesAtOnce(self):
    self._WriteToFile('b', 'b')
    self._WriteToFile('a', 'a')
    self.assertWatcherYields(['a', 'b'])
    self.assertFalse(self._watcher.OutOfOrderWritesDetected())

  def testFinishesLoadingFileWhenSwitchingToNewFile(self):
    self._WriteToFile('a', 'a')
    # Empty the iterator.
    self.assertEquals(['a'], list(self._watcher.Load()))
    self._WriteToFile('a', 'b')
    self._WriteToFile('b', 'c')
    # The watcher should finish its current file before starting a new one.
    self.assertWatcherYields(['b', 'c'])
    self.assertFalse(self._watcher.OutOfOrderWritesDetected())

  def testIntermediateEmptyFiles(self):
    self._WriteToFile('a', 'a')
    self._WriteToFile('b', '')
    self._WriteToFile('c', 'c')
    self.assertWatcherYields(['a', 'c'])
    self.assertFalse(self._watcher.OutOfOrderWritesDetected())

  def testPathFilter(self):
    self._watcher = directory_watcher.DirectoryWatcher(
        self._directory, _ByteLoader,
        lambda path: 'do_not_watch_me' not in path)

    self._WriteToFile('a', 'a')
    self._WriteToFile('do_not_watch_me', 'b')
    self._WriteToFile('c', 'c')
    self.assertWatcherYields(['a', 'c'])
    self.assertFalse(self._watcher.OutOfOrderWritesDetected())

  def testDetectsNewOldFiles(self):
    self._WriteToFile('b', 'a')
    self._LoadAllEvents()
    self._WriteToFile('a', 'a')
    self._LoadAllEvents()
    self.assertTrue(self._watcher.OutOfOrderWritesDetected())

  def testIgnoresNewerFiles(self):
    self._WriteToFile('a', 'a')
    self._LoadAllEvents()
    self._WriteToFile('q', 'a')
    self._LoadAllEvents()
    self.assertFalse(self._watcher.OutOfOrderWritesDetected())

  def testDetectsChangingOldFiles(self):
    self._WriteToFile('a', 'a')
    self._WriteToFile('b', 'a')
    self._LoadAllEvents()
    self._WriteToFile('a', 'c')
    self._LoadAllEvents()
    self.assertTrue(self._watcher.OutOfOrderWritesDetected())

  def testDoesntCrashWhenFileIsDeleted(self):
    self._WriteToFile('a', 'a')
    self._LoadAllEvents()
    os.remove(os.path.join(self._directory, 'a'))
    self._WriteToFile('b', 'b')
    self.assertWatcherYields(['b'])

  def testRaisesRightErrorWhenDirectoryIsDeleted(self):
    self._WriteToFile('a', 'a')
    self._LoadAllEvents()
    shutil.rmtree(self._directory)
    with self.assertRaises(directory_watcher.DirectoryDeletedError):
      self._LoadAllEvents()

  def testDoesntRaiseDirectoryDeletedErrorIfOutageIsTransient(self):
    self._WriteToFile('a', 'a')
    self._LoadAllEvents()
    shutil.rmtree(self._directory)

    # Fake a single transient I/O error.
    def FakeFactory(original):

      def Fake(*args, **kwargs):
        if FakeFactory.has_been_called:
          original(*args, **kwargs)
        else:
          raise OSError('lp0 temporarily on fire')

      return Fake

    FakeFactory.has_been_called = False

    for stub_name in ['ListDirectoryAbsolute', 'ListRecursively']:
      self.stubs.Set(io_wrapper, stub_name,
                     FakeFactory(getattr(io_wrapper, stub_name)))
    for stub_name in ['IsDirectory', 'Exists', 'Stat']:
      self.stubs.Set(tf.gfile, stub_name,
                     FakeFactory(getattr(tf.gfile, stub_name)))

    with self.assertRaises((IOError, OSError)):
      self._LoadAllEvents()


if __name__ == '__main__':
  tf.test.main()
