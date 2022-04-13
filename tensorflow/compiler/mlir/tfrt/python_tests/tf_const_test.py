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
"""Tests for Tensorflow -> jitrt compilation."""

import numpy as np

from tensorflow.compiler.mlir.tfrt.jit.python_binding import tf_jitrt
from tensorflow.python.platform import test

jitrt = tf_jitrt.TfJitRtExecutor()


class TfConstTest(test.TestCase):

  def test_const_i32(self):
    mlir_function = """
      func.func @test() -> tensor<1xi32> {
        %0 = "tf.Const"() {
               value = dense<1> : tensor<1xi32>
             } : () -> tensor<1xi32>
        func.return %0 : tensor<1xi32>
      }"""

    compiled = jitrt.compile(mlir_function, 'test')
    [res] = jitrt.execute(compiled, [])
    np.testing.assert_allclose(res, 1, rtol=0.0)

  def test_constant_folding_i32(self):
    mlir_function = """
      func.func @test() -> tensor<2xi32> {
        %0 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
        %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
        %2 = "tf.Pack"(%0, %1) {axis = 0 : i64}
             : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
        func.return %2 : tensor<2xi32>
      }"""

    compiled = jitrt.compile(mlir_function, 'test')
    [res] = jitrt.execute(compiled, [])
    np.testing.assert_allclose(res, [0, 1], rtol=0.0)

if __name__ == '__main__':
  test.main()
