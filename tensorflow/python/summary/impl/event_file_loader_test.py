"""Tests for event_file_loader."""

import os

from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.summary.impl import event_file_loader


class EventFileLoaderTest(test_util.TensorFlowTestCase):
  # A record containing a simple event.
  RECORD = ('\x18\x00\x00\x00\x00\x00\x00\x00\xa3\x7fK"\t\x00\x00\xc0%\xddu'
            '\xd5A\x1a\rbrain.Event:1\xec\xf32\x8d')

  def _WriteToFile(self, filename, data):
    path = os.path.join(self.get_temp_dir(), filename)
    with open(path, 'ab') as f:
      f.write(data)

  def _LoaderForTestFile(self, filename):
    return event_file_loader.EventFileLoader(
        os.path.join(self.get_temp_dir(), filename))

  def testEmptyEventFile(self):
    self._WriteToFile('empty_event_file', '')
    loader = self._LoaderForTestFile('empty_event_file')
    self.assertEquals(len(list(loader.Load())), 0)

  def testSingleWrite(self):
    self._WriteToFile('single_event_file', EventFileLoaderTest.RECORD)
    loader = self._LoaderForTestFile('single_event_file')
    events = list(loader.Load())
    self.assertEquals(len(events), 1)
    self.assertEquals(events[0].wall_time, 1440183447.0)
    self.assertEquals(len(list(loader.Load())), 0)

  def testMultipleWrites(self):
    self._WriteToFile('staggered_event_file', EventFileLoaderTest.RECORD)
    loader = self._LoaderForTestFile('staggered_event_file')
    self.assertEquals(len(list(loader.Load())), 1)
    self._WriteToFile('staggered_event_file', EventFileLoaderTest.RECORD)
    self.assertEquals(len(list(loader.Load())), 1)

  def testMultipleLoads(self):
    self._WriteToFile('multiple_loads_event_file', EventFileLoaderTest.RECORD)
    loader = self._LoaderForTestFile('multiple_loads_event_file')
    loader.Load()
    loader.Load()
    self.assertEquals(len(list(loader.Load())), 1)

  def testMultipleWritesAtOnce(self):
    self._WriteToFile('multiple_event_file', EventFileLoaderTest.RECORD)
    self._WriteToFile('multiple_event_file', EventFileLoaderTest.RECORD)
    loader = self._LoaderForTestFile('staggered_event_file')
    self.assertEquals(len(list(loader.Load())), 2)


if __name__ == '__main__':
  googletest.main()
