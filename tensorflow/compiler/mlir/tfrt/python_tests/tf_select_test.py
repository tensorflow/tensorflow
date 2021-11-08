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

specializations = [
    tf_cpurt.Specialization.ENABLED,
    tf_cpurt.Specialization.DISABLED,
    tf_cpurt.Specialization.ALWAYS,
]

cpurt = tf_cpurt.TfCpurtExecutor()


class TfSelect(test.TestCase):

  def test_select_1d(self):
    for specialize in specializations:
      mlir_function = """
        func @test(%arg0: tensor<?xf32>)
               -> (tensor<?xf32>, tensor<?xi1>, tensor<?xf32>)
        {
          %c = "tf.Const"() {value = dense<0.0> : tensor<f32>}
               : () -> tensor<f32>
          %0 = "tf.ZerosLike"(%arg0)
               : (tensor<?xf32>) -> tensor<?xf32>
          %1 = "tf.Less"(%arg0, %c)
               : (tensor<?xf32>, tensor<f32>) -> tensor<?xi1>
          %2 = "tf.Select"(%1, %0, %arg0)
               : (tensor<?xi1>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
          return %0, %1, %2 : tensor<?xf32>, tensor<?xi1>, tensor<?xf32>
        }"""

      compiled = cpurt.compile(mlir_function, 'test', specialize)

      d0 = np.random.randint(1, 10)
      arg0 = np.random.uniform(0, 10.0, size=(d0)).astype(np.float32)

      [zeros, less, res] = cpurt.execute(compiled, [arg0])
      np.testing.assert_allclose(zeros, np.zeros_like(arg0), atol=0.0)
      np.testing.assert_allclose(less, np.less(arg0, 0.0), atol=0.0)
      np.testing.assert_allclose(res, np.clip(arg0, 0.0, None), atol=0.0)


if __name__ == '__main__':
  test.main()
