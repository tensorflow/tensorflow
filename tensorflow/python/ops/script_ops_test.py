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
"""Tests for script operations."""

from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import script_ops
from tensorflow.python.ops.script_ops import numpy_function
from tensorflow.python.platform import test


class NumpyFunctionTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_numpy_arguments(self):

    def plus(a, b):
      return a + b

    actual_result = script_ops.numpy_function(plus, [1, 2], dtypes.int32)
    expect_result = constant_op.constant(3, dtypes.int32)
    self.assertAllEqual(actual_result, expect_result)

  def test_stateless_flag(self):

    def plus(a, b):
      return a + b

    @def_function.function
    def tensor_plus_stateful(a, b):
      return numpy_function(plus, [a, b], dtypes.int32, stateful=True)

    @def_function.function
    def tensor_plus_stateless(a, b):
      return numpy_function(plus, [a, b], dtypes.int32, stateful=False)

    @def_function.function(autograph=False)
    def tensor_double_plus(a, b, c, d):
      sum_stateful = tensor_plus_stateful(a, b)
      assert sum_stateful.op.op_def.is_stateful

      sum_stateless = tensor_plus_stateless(c, d)
      assert not sum_stateless.op.op_def.is_stateful

      return sum_stateful, sum_stateless

    tensor_double_plus(
      constant_op.constant(1, dtype=dtypes.int32),
      constant_op.constant(2, dtype=dtypes.int32),
      constant_op.constant(3, dtype=dtypes.int32),
      constant_op.constant(4, dtype=dtypes.int32),
    )


if __name__ == "__main__":
  test.main()
