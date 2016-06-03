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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.platform import googletest
from tensorflow.python.summary.impl import gcs
from tensorflow.python.summary.impl import gcs_file_loader


class GCSFileLoaderTest(tf.test.TestCase):

  def setUp(self):
    self._append_contents_call_count = 0
    # A record containing a simple event.
    self._stubs = googletest.StubOutForTesting()
    self._stubs.Set(gcs, 'CopyContents', self._MockCopyContents)

  def tearDown(self):
    self._stubs.CleanUp()

  def testLoad(self):
    loader = gcs_file_loader.GCSFileLoader('gs://some-fake-url')
    events = list(loader.Load())
    self.assertEqual(len(events), 1)
    self.assertEqual(events[0].file_version, 'brain.Event:1')
    events = list(loader.Load())
    self.assertEqual(len(events), 1)
    self.assertEqual(events[0].file_version, 'brain.Event:2')
    events = list(loader.Load())
    self.assertEqual(len(events), 0)
    self.assertEqual(self._append_contents_call_count, 3)

  # A couple of simple records.
  MOCK_RECORDS = [
      b'\x18\x00\x00\x00\x00\x00\x00\x00\xa3\x7fK"\t\x00\x00\xc0%\xddu'
      b'\xd5A\x1a\rbrain.Event:1\xec\xf32\x8d',
      b'\x18\x00\x00\x00\x00\x00\x00\x00\xa3\x7fK"\t\x00\x00\x00\'\xe6'
      b'\xb3\xd5A\x1a\rbrain.Event:2jM\x0b\x15'
  ]

  def _MockCopyContents(self, gcs_path, offset, local_file):
    if self._append_contents_call_count == 0:
      self.assertEqual(offset, 0)
    elif self._append_contents_call_count == 1:
      self.assertEqual(offset, len(self.MOCK_RECORDS[0]))
    else:
      self.assertEqual(offset,
                       len(self.MOCK_RECORDS[0]) + len(self.MOCK_RECORDS[1]))

    if self._append_contents_call_count < len(self.MOCK_RECORDS):
      local_file.write(self.MOCK_RECORDS[self._append_contents_call_count])
      local_file.flush()
    self._append_contents_call_count += 1


if __name__ == '__main__':
  tf.test.main()
