# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Tensorflow -> CPURT compilation."""

import numpy as np

from tensorflow.compiler.mlir.tfrt.jit.python_binding import tf_cpurt
from tensorflow.python.platform import test

cpurt = tf_cpurt.TfCpurtExecutor()


class TfStridedSliceTest(test.TestCase):

  def test_strided_slice_1d_to_0d(self):
    mlir_function = """
      func @test(%arg0: tensor<3xi32>) -> tensor<i32> {
        %cst_0 = "tf.Const"() {value = dense<1> : tensor<1xi32>}
                 : () -> tensor<1xi32>
        %cst_1 = "tf.Const"() {value = dense<0> : tensor<1xi32>}
                 : () -> tensor<1xi32>
        %0 = "tf.StridedSlice"(%arg0, %cst_1, %cst_0, %cst_0)
             {
               begin_mask       = 0 : i64,
               ellipsis_mask    = 0 : i64,
               end_mask         = 0 : i64,
               new_axis_mask    = 0 : i64,
               shrink_axis_mask = 1 : i64
             } : (tensor<3xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>)
              -> tensor<i32>
        return %0 : tensor<i32>
      }"""

    compiled = cpurt.compile(mlir_function, 'test')
    arg0 = np.array([1, 2, 3], dtype=np.int32)
    [res] = cpurt.execute(compiled, [arg0])
    np.testing.assert_allclose(res, arg0[0], atol=0.0)


if __name__ == '__main__':
  test.main()
