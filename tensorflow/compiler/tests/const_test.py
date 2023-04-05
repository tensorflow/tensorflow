# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for const op compilation."""

import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


# This test doesn't use XLATestCase like the other tests in this directory.
# The Const op xla op kernel is compilation only and therefore is not executed
# with XLA in the on demand compilation mode. Instead we use
# tf.function(jit_compile=True)
class ConstOpTest(test_util.TensorFlowTestCase):

  # Verifies that the Const op works
  # @test_util.run_v2_only
  def testConst(self):
    types = {
        dtypes.bool, dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64,
        dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64,
        dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64,
        dtypes.float8_e5m2, dtypes.float8_e4m3fn,
    }
    for dtype in types:
      with self.subTest(dtype=dtype):
        if dtype == dtypes.bool:
          values = [True, False]
        else:
          values = [0., 1., -1., dtype.min, dtype.max]
        if dtype.is_floating:
          values.extend([float("Inf"), -float("Inf"), float("NaN")])
        values = np.array(values, dtype=dtype.as_numpy_dtype)

        @def_function.function(jit_compile=True)
        def f():
          return constant_op.constant(values, dtype)  # pylint: disable=cell-var-from-loop

        result = f()
        self.assertAllEqual(self.evaluate(result), values)


if __name__ == "__main__":
  test.main()
