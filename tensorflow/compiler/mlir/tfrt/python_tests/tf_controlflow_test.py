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
"""Tests for Tensorflow -> jitrt compilation."""

import numpy as np

from tensorflow.compiler.mlir.tfrt.jit.python_binding import tf_jitrt
from tensorflow.python.platform import test

specializations = [
    tf_jitrt.Specialization.ENABLED,
    tf_jitrt.Specialization.DISABLED,
    tf_jitrt.Specialization.ALWAYS,
]

jitrt = tf_jitrt.TfJitRtExecutor()


class TfControlflowTest(test.TestCase):

  def test_if(self):
    for specialize in specializations:
      mlir_function = """
        func.func @test(%arg0: tensor<i1>, %arg1: tensor<i1>, %arg2: tensor<?xf32>,
                   %arg3: tensor<?xf32>) -> tensor<?xf32> {
          %0 = "tf.IfRegion"(%arg0) ({
              %1 = "tf.If"(%arg1, %arg2, %arg3)
                 {then_branch = @add, else_branch = @sub, is_stateless = true}
                 : (tensor<i1>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
              "tf.Yield"(%1) : (tensor<?xf32>) -> ()
            }, {
              %2 = "tf.Mul"(%arg2, %arg3) : (tensor<?xf32>, tensor<?xf32>)
                 -> tensor<?xf32>
              "tf.Yield"(%2) : (tensor<?xf32>) -> ()
            }) {is_stateless = false} : (tensor<i1>) -> tensor<?xf32>
          func.return %0: tensor<?xf32>
        }

        func.func @add(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
          %0 = "tf.Add"(%arg0, %arg1): (tensor<?xf32>, tensor<?xf32>)
             -> tensor<?xf32>
          func.return %0 : tensor<?xf32>
        }

        func.func @sub(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
          %0 = "tf.Sub"(%arg0, %arg1) : (tensor<?xf32>, tensor<?xf32>)
             -> tensor<?xf32>
          func.return %0 : tensor<?xf32>
        }"""
      compiled = jitrt.compile(mlir_function, 'test', specialize)

      d0 = np.random.randint(1, 100)

      arg0 = np.random.uniform(0.0, 10.0, size=(d0)).astype(np.float32)
      arg1 = np.random.uniform(0.0, 10.0, size=(d0)).astype(np.float32)

      true = np.array(True)
      false = np.array(False)
      [res] = jitrt.execute(compiled, [false, false, arg0, arg1])
      np.testing.assert_allclose(res, arg0 * arg1)
      [res] = jitrt.execute(compiled, [false, true, arg0, arg1])
      np.testing.assert_allclose(res, arg0 * arg1)
      [res] = jitrt.execute(compiled, [true, false, arg0, arg1])
      np.testing.assert_allclose(res, arg0 - arg1)
      [res] = jitrt.execute(compiled, [true, true, arg0, arg1])
      np.testing.assert_allclose(res, arg0 + arg1)

  def test_while(self):
    for specialize in specializations:
      # Square input until one element is over 100.
      mlir_function = """
        func.func @test(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
          %0 = "tf.While"(%arg0)
             {body = @while_body, cond = @while_cond, is_stateless = true}
             : (tensor<?x?xf32>) -> (tensor<?x?xf32>)
          func.return %0: tensor<?x?xf32>
        }

        func.func @while_body(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
          %0 = "tf.Square"(%arg0): (tensor<?x?xf32>) -> tensor<?x?xf32>
          func.return %0: tensor<?x?xf32>
        }

        func.func @while_cond(%arg0: tensor<?x?xf32>) -> tensor<i1> {
          %cst = "tf.Const"() {value = dense<100.0> : tensor<f32>}
             : () -> tensor<f32>
          %less = "tf.Less"(%arg0, %cst) {T = f32}
             : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xi1>
          %dim_to_reduce = "tf.Const"() {value = dense<[0, 1]> : tensor<2xi32>}
             : () -> tensor<2xi32>
          %all = "tf.All"(%less, %dim_to_reduce) {keep_dims = false}
             : (tensor<?x?xi1>, tensor<2xi32>) -> tensor<i1>
          func.return %all : tensor<i1>
        }"""
      compiled = jitrt.compile(mlir_function, 'test', specialize)

      d0 = np.random.randint(1, 100)
      d1 = np.random.randint(1, 100)

      arg0 = np.random.uniform(2.0, 10.0, size=(d0, d1)).astype(np.float32)

      np_res = arg0
      while np.all(np.less(np_res, 100)):
        np_res = np_res * np_res

      [res] = jitrt.execute(compiled, [arg0])
      np.testing.assert_allclose(res, np_res)


if __name__ == '__main__':
  test.main()
