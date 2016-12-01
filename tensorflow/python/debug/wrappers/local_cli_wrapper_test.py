# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Unit tests for local command-line-interface debug wrapper session."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile

from tensorflow.python.client import session
from tensorflow.python.debug.wrappers import local_cli_wrapper
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class LocalCLIDebugWrapperSessionTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self._tmp_dir = tempfile.mktemp()

  def tearDown(self):
    if os.path.isdir(self._tmp_dir):
      shutil.rmtree(self._tmp_dir)

  def testConstructWrapper(self):
    local_cli_wrapper.LocalCLIDebugWrapperSession(
        session.Session(), log_usage=False)

  def testConstructWrapperWithExistingEmptyDumpRoot(self):
    os.mkdir(self._tmp_dir)
    self.assertTrue(os.path.isdir(self._tmp_dir))

    local_cli_wrapper.LocalCLIDebugWrapperSession(
        session.Session(), dump_root=self._tmp_dir, log_usage=False)

  def testConstructWrapperWithExistingNonEmptyDumpRoot(self):
    os.mkdir(self._tmp_dir)
    dir_path = os.path.join(self._tmp_dir, "foo")
    os.mkdir(dir_path)
    self.assertTrue(os.path.isdir(dir_path))

    with self.assertRaisesRegexp(
        ValueError, "dump_root path points to a non-empty directory"):
      local_cli_wrapper.LocalCLIDebugWrapperSession(
          session.Session(), dump_root=self._tmp_dir, log_usage=False)

  def testConstructWrapperWithExistingFileDumpRoot(self):
    os.mkdir(self._tmp_dir)
    file_path = os.path.join(self._tmp_dir, "foo")
    open(file_path, "a").close()  # Create the file
    self.assertTrue(os.path.isfile(file_path))
    with self.assertRaisesRegexp(
        ValueError, "dump_root path points to a file"):
      local_cli_wrapper.LocalCLIDebugWrapperSession(
          session.Session(), dump_root=file_path, log_usage=False)


if __name__ == "__main__":
  googletest.main()
