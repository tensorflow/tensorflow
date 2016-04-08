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
"""Loads events from a file stored on Google Cloud Storage."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

from tensorflow.core.util import event_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import app
from tensorflow.python.platform import logging
from tensorflow.python.summary.impl import gcs
from tensorflow.python.util import compat


class GCSFileLoader(object):
  """A GCSFileLoader loads Event protos from a path to GCS storage.

  The GCSFileLoader keeps track of the offset in the file, copies the contents
  of the file to local disk, reads it, and then immediately deletes the file.
  """

  def __init__(self, gcs_path):
    if not gcs.IsGCSPath(gcs_path):
      raise ValueError('A GCS path is required')
    self._gcs_path = gcs_path
    self._gcs_offset = 0

  def Load(self):
    # Create a temp file to hold the contents that we haven't seen yet.
    with tempfile.NamedTemporaryFile(prefix='tf-gcs-') as temp_file:
      name = temp_file.name
      logging.debug('Temp file created at %s', name)
      gcs.CopyContents(self._gcs_path, self._gcs_offset, temp_file)
      reader = pywrap_tensorflow.PyRecordReader_New(compat.as_bytes(name), 0)
      while reader.GetNext():
        event = event_pb2.Event()
        event.ParseFromString(reader.record())
        yield event
      logging.debug('No more events in %s', name)
      self._gcs_offset += reader.offset()


def main(argv):
  if len(argv) != 2:
    print('Usage: gcs_file_loader <path-to-gcs-object>')
    return 1
  loader = GCSFileLoader(argv[1])
  for event in loader.Load():
    print(event)


if __name__ == '__main__':
  app.run()
