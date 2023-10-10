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
# =============================================================================
"""Tests for tensorflow.python.training.saver.py."""

import os

from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import saver


class SaverLargeVariableTest(test.TestCase):

  # NOTE: This is in a separate file from saver_test.py because the
  # large allocations do not play well with TSAN, and cause flaky
  # failures.
  def testLargeVariable(self):
    save_path = os.path.join(self.get_temp_dir(), "large_variable")
    with session.Session("", graph=ops.Graph()) as sess:
      # Declare a variable that is exactly 2GB. This should fail,
      # because a serialized checkpoint includes other header
      # metadata.
      with ops.device("/cpu:0"):
        var = variables.Variable(
            constant_op.constant(
                False, shape=[2, 1024, 1024, 1024], dtype=dtypes.bool))
      save = saver.Saver(
          {
              var.op.name: var
          }, write_version=saver_pb2.SaverDef.V1)
      var.initializer.run()
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "Tensor slice is too large to serialize"):
        save.save(sess, save_path)


if __name__ == "__main__":
  test.main()
