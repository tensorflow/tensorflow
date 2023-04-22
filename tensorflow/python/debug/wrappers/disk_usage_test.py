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
"""Debugger Wrapper Session Consisting of a Local Curses-based CLI."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from tensorflow.python.client import session
from tensorflow.python.debug.wrappers import dumping_wrapper
from tensorflow.python.debug.wrappers import hooks
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import monitored_session


@test_util.run_v1_only("Sessions are not available in TF 2.x")
class DumpingDebugWrapperDiskUsageLimitTest(test_util.TensorFlowTestCase):

  @classmethod
  def setUpClass(cls):
    # For efficient testing, set the disk usage bytes limit to a small
    # number (10).
    os.environ["TFDBG_DISK_BYTES_LIMIT"] = "10"

  def setUp(self):
    self.session_root = tempfile.mkdtemp()

    self.v = variables.Variable(10.0, dtype=dtypes.float32, name="v")
    self.delta = constant_op.constant(1.0, dtype=dtypes.float32, name="delta")
    self.eta = constant_op.constant(-1.4, dtype=dtypes.float32, name="eta")
    self.inc_v = state_ops.assign_add(self.v, self.delta, name="inc_v")
    self.dec_v = state_ops.assign_add(self.v, self.eta, name="dec_v")

    self.sess = session.Session()
    self.sess.run(self.v.initializer)

  def testWrapperSessionNotExceedingLimit(self):
    def _watch_fn(fetches, feeds):
      del fetches, feeds
      return "DebugIdentity", r"(.*delta.*|.*inc_v.*)", r".*"
    sess = dumping_wrapper.DumpingDebugWrapperSession(
        self.sess, session_root=self.session_root,
        watch_fn=_watch_fn, log_usage=False)
    sess.run(self.inc_v)

  def testWrapperSessionExceedingLimit(self):
    def _watch_fn(fetches, feeds):
      del fetches, feeds
      return "DebugIdentity", r".*delta.*", r".*"
    sess = dumping_wrapper.DumpingDebugWrapperSession(
        self.sess, session_root=self.session_root,
        watch_fn=_watch_fn, log_usage=False)
    # Due to the watch function, each run should dump only 1 tensor,
    # which has a size of 4 bytes, which corresponds to the dumped 'delta:0'
    # tensor of scalar shape and float32 dtype.
    # 1st run should pass, after which the disk usage is at 4 bytes.
    sess.run(self.inc_v)
    # 2nd run should also pass, after which 8 bytes are used.
    sess.run(self.inc_v)
    # 3rd run should fail, because the total byte count (12) exceeds the
    # limit (10)
    with self.assertRaises(ValueError):
      sess.run(self.inc_v)

  def testHookNotExceedingLimit(self):
    def _watch_fn(fetches, feeds):
      del fetches, feeds
      return "DebugIdentity", r".*delta.*", r".*"
    dumping_hook = hooks.DumpingDebugHook(
        self.session_root, watch_fn=_watch_fn, log_usage=False)
    mon_sess = monitored_session._HookedSession(self.sess, [dumping_hook])
    mon_sess.run(self.inc_v)

  def testHookExceedingLimit(self):
    def _watch_fn(fetches, feeds):
      del fetches, feeds
      return "DebugIdentity", r".*delta.*", r".*"
    dumping_hook = hooks.DumpingDebugHook(
        self.session_root, watch_fn=_watch_fn, log_usage=False)
    mon_sess = monitored_session._HookedSession(self.sess, [dumping_hook])
    # Like in `testWrapperSessionExceedingLimit`, the first two calls
    # should be within the byte limit, but the third one should error
    # out due to exceeding the limit.
    mon_sess.run(self.inc_v)
    mon_sess.run(self.inc_v)
    with self.assertRaises(ValueError):
      mon_sess.run(self.inc_v)


if __name__ == "__main__":
  googletest.main()
