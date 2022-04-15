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

import os
import shutil
import tempfile
import uuid

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
    self.tfdbg_run_id = str(uuid.uuid4())

  def tearDown(self):
    if os.path.isdir(self.dump_root):
      shutil.rmtree(self.dump_root, ignore_errors=True)
    check_numerics_callback.disable_check_numerics()
    dumping_callback.disable_dump_debug_info()
    super(DumpingCallbackTestBase, self).tearDown()

  def _readAndCheckMetadataFile(self):
    """Read and check the .metadata debug-events file."""
    with debug_events_reader.DebugEventsReader(self.dump_root) as reader:
      self.assertTrue(reader.tfdbg_run_id())
      self.assertEqual(reader.tensorflow_version(), versions.__version__)
      self.assertTrue(reader.tfdbg_file_version().startswith("debug.Event"))
