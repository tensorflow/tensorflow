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

"""Tests for event_file_loader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.summary.impl import event_file_loader


class EventFileLoaderTest(test_util.TensorFlowTestCase):
  # A record containing a simple event.
  RECORD = (b'\x18\x00\x00\x00\x00\x00\x00\x00\xa3\x7fK"\t\x00\x00\xc0%\xddu'
            b'\xd5A\x1a\rbrain.Event:1\xec\xf32\x8d')

  def _WriteToFile(self, filename, data):
    with open(filename, 'ab') as f:
      f.write(data)

  def _LoaderForTestFile(self, filename):
    return event_file_loader.EventFileLoader(
        os.path.join(self.get_temp_dir(), filename))

  def testEmptyEventFile(self):
    filename = tempfile.NamedTemporaryFile(dir=self.get_temp_dir()).name
    self._WriteToFile(filename, b'')
    loader = self._LoaderForTestFile(filename)
    self.assertEqual(len(list(loader.Load())), 0)

  def testSingleWrite(self):
    filename = tempfile.NamedTemporaryFile(dir=self.get_temp_dir()).name
    self._WriteToFile(filename, EventFileLoaderTest.RECORD)
    loader = self._LoaderForTestFile(filename)
    events = list(loader.Load())
    self.assertEqual(len(events), 1)
    self.assertEqual(events[0].wall_time, 1440183447.0)
    self.assertEqual(len(list(loader.Load())), 0)

  def testMultipleWrites(self):
    filename = tempfile.NamedTemporaryFile(dir=self.get_temp_dir()).name
    self._WriteToFile(filename, EventFileLoaderTest.RECORD)
    loader = self._LoaderForTestFile(filename)
    self.assertEqual(len(list(loader.Load())), 1)
    self._WriteToFile(filename, EventFileLoaderTest.RECORD)
    self.assertEqual(len(list(loader.Load())), 1)

  def testMultipleLoads(self):
    filename = tempfile.NamedTemporaryFile(dir=self.get_temp_dir()).name
    self._WriteToFile(filename, EventFileLoaderTest.RECORD)
    loader = self._LoaderForTestFile(filename)
    loader.Load()
    loader.Load()
    self.assertEqual(len(list(loader.Load())), 1)

  def testMultipleWritesAtOnce(self):
    filename = tempfile.NamedTemporaryFile(dir=self.get_temp_dir()).name
    self._WriteToFile(filename, EventFileLoaderTest.RECORD)
    self._WriteToFile(filename, EventFileLoaderTest.RECORD)
    loader = self._LoaderForTestFile(filename)
    self.assertEqual(len(list(loader.Load())), 2)


if __name__ == '__main__':
  googletest.main()
