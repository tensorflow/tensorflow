# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for DLPack functions."""
from absl.testing import parameterized
import numpy as np


from tensorflow.python.dlpack import dlpack
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.ops import array_ops

int_dtypes = [
    np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
    np.uint64
]
float_dtypes = [np.float16, np.float32, np.float64]
complex_dtypes = [np.complex64, np.complex128]
dlpack_dtypes = int_dtypes + float_dtypes + [dtypes.bfloat16]

testcase_shapes = [(), (1,), (2, 3), (2, 0), (0, 7), (4, 1, 2)]


def FormatShapeAndDtype(shape, dtype):
  return "_{}[{}]".format(str(dtype), ",".join(map(str, shape)))


def GetNamedTestParameters():
  result = []
  for dtype in dlpack_dtypes:
    for shape in testcase_shapes:
      result.append({
          "testcase_name": FormatShapeAndDtype(shape, dtype),
          "dtype": dtype,
          "shape": shape
      })
  return result


class DLPackTest(parameterized.TestCase, test.TestCase):

  @parameterized.named_parameters(GetNamedTestParameters())
  def testRoundTrip(self, dtype, shape):
    np.random.seed(42)
    np_array = np.random.randint(0, 10, shape)
    # copy to gpu if available
    tf_tensor = array_ops.identity(constant_op.constant(np_array, dtype=dtype))
    tf_tensor_device = tf_tensor.device
    tf_tensor_dtype = tf_tensor.dtype
    dlcapsule = dlpack.to_dlpack(tf_tensor)
    del tf_tensor  # should still work
    tf_tensor2 = dlpack.from_dlpack(dlcapsule)
    self.assertAllClose(np_array, tf_tensor2)
    if tf_tensor_dtype == dtypes.int32:
      # int32 tensor is always on cpu for now
      self.assertEqual(tf_tensor2.device,
                       "/job:localhost/replica:0/task:0/device:CPU:0")
    else:
      self.assertEqual(tf_tensor_device, tf_tensor2.device)

  def testTensorsCanBeConsumedOnceOnly(self):
    np.random.seed(42)
    np_array = np.random.randint(0, 10, (2, 3, 4))
    tf_tensor = constant_op.constant(np_array, dtype=np.float32)
    dlcapsule = dlpack.to_dlpack(tf_tensor)
    del tf_tensor  # should still work
    _ = dlpack.from_dlpack(dlcapsule)

    def ConsumeDLPackTensor():
      dlpack.from_dlpack(dlcapsule)  # Should can be consumed only once

    self.assertRaisesRegex(Exception,
                           ".*a DLPack tensor may be consumed at most once.*",
                           ConsumeDLPackTensor)

  def testDLPackFromWithoutContextInitialization(self):
    tf_tensor = constant_op.constant(1)
    dlcapsule = dlpack.to_dlpack(tf_tensor)
    # Resetting the context doesn't cause an error.
    context._reset_context()
    _ = dlpack.from_dlpack(dlcapsule)

  def testUnsupportedTypeToDLPack(self):

    def UnsupportedQint16():
      tf_tensor = constant_op.constant([[1, 4], [5, 2]], dtype=dtypes.qint16)
      _ = dlpack.to_dlpack(tf_tensor)

    def UnsupportedComplex64():
      tf_tensor = constant_op.constant([[1, 4], [5, 2]], dtype=dtypes.complex64)
      _ = dlpack.to_dlpack(tf_tensor)

    self.assertRaisesRegex(Exception, ".* is not supported by dlpack",
                           UnsupportedQint16)
    self.assertRaisesRegex(Exception, ".* is not supported by dlpack",
                           UnsupportedComplex64)

  def testMustPassTensorArgumentToDLPack(self):
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "The argument to `to_dlpack` must be a TF tensor, not Python object"):
      dlpack.to_dlpack([1])


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
