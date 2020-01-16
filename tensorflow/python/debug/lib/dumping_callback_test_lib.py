# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Shared library for testing tfdbg v2 dumping callback."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile

from tensorflow.python.debug.lib import check_numerics_callback
from tensorflow.python.debug.lib import debug_events_reader
from tensorflow.python.debug.lib import dumping_callback
from tensorflow.python.framework import test_util
from tensorflow.python.framework import versions


class DumpingCallbackTestBase(test_util.TensorFlowTestCase):
  """Base test-case class for tfdbg v2 callbacks."""

  def setUp(self):
    super(DumpingCallbackTestBase, self).setUp()
    self.dump_root = tempfile.mkdtemp()

  def tearDown(self):
    if os.path.isdir(self.dump_root):
      shutil.rmtree(self.dump_root, ignore_errors=True)
    check_numerics_callback.disable_check_numerics()
    dumping_callback.disable_dump_debug_info()
    super(DumpingCallbackTestBase, self).tearDown()

  def _readAndCheckMetadataFile(self):
    """Read and check the .metadata debug-events file."""
    with debug_events_reader.DebugEventsReader(self.dump_root) as reader:
      metadata_iter = reader.metadata_iterator()
      metadata = next(metadata_iter).debug_event.debug_metadata
      self.assertEqual(metadata.tensorflow_version, versions.__version__)
      self.assertTrue(metadata.file_version.startswith("debug.Event"))
