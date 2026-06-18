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
import os

from absl.testing import parameterized

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import test

_TEST_DTYPES = [dtypes.float32, dtypes.int32, dtypes.int4, dtypes.uint4]


class SaveRestoreTest(test.TestCase, parameterized.TestCase):

  @parameterized.parameters(_TEST_DTYPES)
  @test_util.run_in_graph_and_eager_modes
  def testRelativePath(self, dtype):
    os.chdir(self.get_temp_dir())
    self.evaluate(
        io_ops.save_v2(
            "ckpt", ["x"], [""], [constant_op.constant(2, dtype=dtype)]
        )
    )
    self.assertAllEqual(
        [2], self.evaluate(io_ops.restore_v2("ckpt", ["x"], [""], [dtype]))
    )

  @parameterized.parameters(_TEST_DTYPES)
  def testWithSliceInput(self, dtype):
    os.chdir(self.get_temp_dir())
    self.evaluate(
        io_ops.save_v2(
            "ckpt",
            ["x"],
            [""],
            [constant_op.constant([[1, 2, 3], [2, 3, 4]], dtype=dtype)],
        )
    )
    self.assertAllEqual(
        [[2], [3]],
        self.evaluate(io_ops.restore_v2("ckpt", ["x"], ["2 3 -:1,1"], [dtype]))[
            0
        ],
    )


class ShardedFileOpsTest(test.TestCase):

  def testShardedFileName(self):
    with session.Session(
        target="", config=config_pb2.ConfigProto(device_count={"CPU": 2})):
      self.assertEqual(
          gen_io_ops.sharded_filename("foo", 4, 100).eval(),
          b"foo-00004-of-00100")
      self.assertEqual(
          gen_io_ops.sharded_filespec("foo", 100).eval(), b"foo-?????-of-00100")


class ShapeInferenceTest(test.TestCase, parameterized.TestCase):

  @parameterized.parameters(_TEST_DTYPES)
  def testRestoreV2WithSliceInput(self, dtype):
    with ops.Graph().as_default():
      op = io_ops.restore_v2(
          "model", ["var1", "var2"], ["", "3 4 0,1:-"], [dtype, dtype]
      )
      self.assertEqual(2, len(op))
      self.assertFalse(op[0].get_shape().is_fully_defined())
      self.assertEqual([1, 4], op[1].get_shape())

  @parameterized.parameters(_TEST_DTYPES)
  def testRestoreV2NumSlicesNotMatch(self, dtype):
    with ops.Graph().as_default():
      with self.assertRaises(ValueError):
        io_ops.restore_v2(
            "model", ["var1", "var2", "var3"], ["", "3 4 0,1:-"], [dtype, dtype]
        )

  @parameterized.parameters(_TEST_DTYPES)
  def testRestoreSlice(self, dtype):
    with ops.Graph().as_default():
      op = gen_io_ops.restore_slice("model", "var", "3 4 0,1:-", dtype)
      self.assertEqual([1, 4], op.get_shape())


if __name__ == "__main__":
  test.main()
