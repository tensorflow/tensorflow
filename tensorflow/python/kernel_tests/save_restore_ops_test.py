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
"""Tests for tensorflow.ops.io_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.platform import test


class ShardedFileOpsTest(test.TestCase):

  def testShardedFileName(self):
    with session.Session(
        target="", config=config_pb2.ConfigProto(device_count={"CPU": 2})):
      self.assertEqual(
          gen_io_ops._sharded_filename("foo", 4, 100).eval(),
          b"foo-00004-of-00100")
      self.assertEqual(
          gen_io_ops._sharded_filespec("foo", 100).eval(),
          b"foo-?????-of-00100")


if __name__ == "__main__":
  test.main()
