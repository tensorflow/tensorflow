# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
"""Regression test for ArgMax/ArgMin CPU kernel registration of `Tidx`.

Covers the fix adding `.TypeConstraint<int32>("Tidx")` to the CPU
registrations in `tensorflow/core/kernels/argmax_op.cc`, mirroring the
GPU registrations, which already had it. Before the fix, a `dimension`
tensor of dtype int64 (Tidx=int64) on CPU would be matched by a kernel
whose `Compute()` unconditionally reads `dimension` as int32, hitting a
fatal `Tensor::scalar<T>()` dtype-mismatch CHECK and crashing the
process. After the fix, TensorFlow should instead fail this combination
cleanly at kernel-lookup time with a catchable error, exactly as it
already does on GPU.
"""

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class ArgMaxMinTidxRegistrationTest(test_util.TensorFlowTestCase):

  def testArgMaxInt64DimensionOnCpuFailsCleanly(self):
    """ArgMax with an int64 `dimension` tensor on CPU must not crash.

    Before the fix, this combination was incorrectly matched by the
    CPU kernel (which only supports int32 `dimension`), and would
    crash the process with a fatal CHECK failure rather than raising a
    catchable Python exception. This test only verifies the failure is
    now catchable and clean; it does not (and should not) attempt to
    assert the exact resulting value.
    """
    with test_util.use_cpu():
      input_tensor = constant_op.constant([1.0, 2.0, 3.0], dtype=dtypes.float32)
      dimension = constant_op.constant(0, dtype=dtypes.int64)
      with self.assertRaises((errors.InvalidArgumentError,
                              errors.NotFoundError)):
        self.evaluate(array_ops.argmax(input_tensor, axis=dimension))

  def testArgMinInt64DimensionOnCpuFailsCleanly(self):
    """Same as above, for ArgMin."""
    with test_util.use_cpu():
      input_tensor = constant_op.constant([1.0, 2.0, 3.0], dtype=dtypes.float32)
      dimension = constant_op.constant(0, dtype=dtypes.int64)
      with self.assertRaises((errors.InvalidArgumentError,
                              errors.NotFoundError)):
        self.evaluate(array_ops.argmin(input_tensor, axis=dimension))

  def testArgMaxInt32DimensionOnCpuStillWorks(self):
    """Sanity check: the fix must not regress the supported Tidx=int32 case."""
    with test_util.use_cpu():
      input_tensor = constant_op.constant([1.0, 3.0, 2.0], dtype=dtypes.float32)
      dimension = constant_op.constant(0, dtype=dtypes.int32)
      result = self.evaluate(array_ops.argmax(input_tensor, axis=dimension))
      self.assertAllEqual(result, 1)

  def testArgMinInt32DimensionOnCpuStillWorks(self):
    """Sanity check: the fix must not regress the supported Tidx=int32 case."""
    with test_util.use_cpu():
      input_tensor = constant_op.constant([1.0, 3.0, 2.0], dtype=dtypes.float32)
      dimension = constant_op.constant(0, dtype=dtypes.int32)
      result = self.evaluate(array_ops.argmin(input_tensor, axis=dimension))
      self.assertAllEqual(result, 0)


if __name__ == "__main__":
  test.main()
