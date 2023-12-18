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
import os
import tempfile
import tensorflow as tf
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
class DumpingDebugWrapperVariableTypesTest(test_util.TensorFlowTestCase):

    @classmethod
    def setUpClass(cls):
        # For efficient testing, set the disk usage bytes limit to a small number (10).
        os.environ["TFDBG_DISK_BYTES_LIMIT"] = "50"

    def setUp(self):
        self.session_root = tempfile.mkdtemp()

        # Variable with float32 dtype
        self.v_float32 = variables.Variable(10.0, dtype=dtypes.float32, name="v_float32")

        # Variable with int32 dtype
        self.v_int32 = variables.Variable(5, dtype=dtypes.int32, name="v_int32")

        # Tensor with float64 dtype and shape (3, 3)
        self.tensor_float64 = constant_op.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                                                   dtype=dtypes.float64, name="tensor_float64")

        # Tensor with int64 dtype and shape (2, 4)
        self.tensor_int64 = constant_op.constant([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=dtypes.int64, name="tensor_int64")

        self.inc_v_float32 = state_ops.assign_add(self.v_float32, 1.0, name="inc_v_float32")
        self.inc_v_int32 = state_ops.assign_add(self.v_int32, 1, name="inc_v_int32")

        self.sess = tf.Session()
        self.sess.run([self.v_float32.initializer, self.v_int32.initializer])

    def testWrapperSessionVariableTypes(self):
        def _watch_fn(fetches, feeds):
            del fetches, feeds
            return "DebugIdentity", r".*", r".*"

        sess = dumping_wrapper.DumpingDebugWrapperSession(
            self.sess, session_root=self.session_root, watch_fn=_watch_fn)

        # Test with a float32 variable
        sess.run(self.inc_v_float32)

        # Test with an int32 variable
        sess.run(self.inc_v_int32)

        # Test with a float64 tensor
        sess.run(tf.reduce_sum(self.tensor_float64))

        # Test with an int64 tensor
        sess.run(tf.reduce_sum(self.tensor_int64))

    def testHookVariableTypes(self):
        def _watch_fn(fetches, feeds):
            del fetches, feeds
            return "DebugIdentity", r".*", r".*"

        dumping_hook = hooks.DumpingDebugHook(
            self.session_root, watch_fn=_watch_fn)
        mon_sess = monitored_session._HookedSession(self.sess, [dumping_hook])

        # Test with a float32 variable
        mon_sess.run(self.inc_v_float32)

        # Test with an int32 variable
        mon_sess.run(self.inc_v_int32)

        # Test with a float64 tensor
        mon_sess.run(tf.reduce_sum(self.tensor_float64))

        # Test with an int64 tensor
        mon_sess.run(tf.reduce_sum(self.tensor_int64))


if __name__ == "__main__":
    googletest.main()
