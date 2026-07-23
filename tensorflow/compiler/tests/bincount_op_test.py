# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for bincount using the XLA JIT."""
from tensorflow.compiler.tests import xla_test
from tensorflow.python.compat import v2_compat
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.platform import googletest


class BincountTest(xla_test.XLATestCase):

  def testInputRank0(self):
    with self.session():
      with self.test_scope():
        bincount = gen_math_ops.bincount(arr=6, size=804, weights=[52, 351])

      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          (
              "`weights` must be the same shape as `arr` or a length-0"
              " `Tensor`, in which case it acts as all weights equal to 1."
          ),
      ):
        self.evaluate(bincount)

  def testNegativeInputConstant(self):
    with self.session():
      with self.test_scope():

        @def_function.function(jit_compile=True)
        def f():
          return gen_math_ops.dense_bincount(
              input=array_ops.constant([0, -1, 2]), size=3, weights=[]
          )

        with self.assertRaisesRegex(
            errors.InvalidArgumentError, "Input arr must be non-negative!"
        ):
          self.evaluate(f())

  def testNegativeInputConstantBincount(self):
    with self.session():
      with self.test_scope():

        @def_function.function(jit_compile=True)
        def f():
          return gen_math_ops.bincount(
              arr=array_ops.constant([0, -1, 2]), size=3, weights=[]
          )

        with self.assertRaisesRegex(
            errors.InvalidArgumentError, "Input arr must be non-negative!"
        ):
          self.evaluate(f())

  def testNegativeInputRuntimeIgnored(self):
    """Runtime (non-constant) negative values should be silently ignored.

    When the input is not a compile-time constant, we cannot raise a runtime
    error from within the XLA computation. Instead, negative values should be
    ignored (not counted) so they don't corrupt the output. This tests the
    fix for b/117960 where negative values under jit_compile=True silently
    produced wrong results.
    """
    with self.session():
      with self.test_scope():

        @def_function.function(
            jit_compile=True,
            input_signature=[tensor_spec.TensorSpec([5], dtypes.int32)])
        def f(arr):
          return gen_math_ops.dense_bincount(
              input=arr, size=5, weights=[]
          )

        # Input with negative values — they should be ignored, not counted.
        # Expected: [1, 1, 1, 0, 0] (counts of 0, 1, 2; -1 and -2 ignored)
        result = self.evaluate(
            f(array_ops.constant([0, -1, 1, -2, 2], dtype=dtypes.int32)))
        self.assertAllEqual([1, 1, 1, 0, 0], result)

  def testNegativeInputRuntimeIgnoredBinary(self):
    """Runtime negative values with binary_output should be ignored."""
    with self.session():
      with self.test_scope():

        @def_function.function(
            jit_compile=True,
            input_signature=[tensor_spec.TensorSpec([5], dtypes.int32)])
        def f(arr):
          return gen_math_ops.dense_bincount(
              input=arr, size=5, weights=[], binary_output=True
          )

        result = self.evaluate(
            f(array_ops.constant([0, -1, 0, -2, 4], dtype=dtypes.int32)))
        self.assertAllEqual([1, 0, 0, 0, 1], result)

  def testNegativeInputRuntime2d(self):
    """Runtime negative values in 2D input should be ignored."""
    with self.session():
      with self.test_scope():

        @def_function.function(
            jit_compile=True,
            input_signature=[tensor_spec.TensorSpec([2, 3], dtypes.int32)])
        def f(arr):
          return gen_math_ops.dense_bincount(
              input=arr, size=4, weights=[]
          )

        # Row 0: [0, -1, 2] -> counts: [1, 0, 1, 0]
        # Row 1: [1, -3, 3] -> counts: [0, 1, 0, 1]
        result = self.evaluate(
            f(array_ops.constant(
                [[0, -1, 2], [1, -3, 3]], dtype=dtypes.int32)))
        self.assertAllEqual([[1, 0, 1, 0], [0, 1, 0, 1]], result)


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  googletest.main()
