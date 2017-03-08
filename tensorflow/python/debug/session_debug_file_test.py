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
"""Tests for debugger functionalities in tf.Session with file:// URLs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.debug import session_debug_testlib
from tensorflow.python.platform import googletest


class SessionDebugTest(session_debug_testlib.SessionDebugTestBase):

  def _debug_urls(self, run_number=None):
    return ["file://%s" % self._debug_dump_dir(run_number=run_number)]

  def _debug_dump_dir(self, run_number=None):
    if run_number is None:
      return self._dump_root
    else:
      return os.path.join(self._dump_root, "run_%d" % run_number)


if __name__ == "__main__":
  googletest.main()
