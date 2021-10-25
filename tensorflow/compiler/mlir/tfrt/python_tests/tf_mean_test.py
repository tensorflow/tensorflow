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
"""Tests for tf.Mean JIT compilation."""

import numpy as np

from tensorflow.compiler.mlir.tfrt.jit.python_binding import tf_cpurt
from tensorflow.python.platform import test

cpurt = tf_cpurt.TfCpurtExecutor()


class TfMeanTest(test.TestCase):

  def test_mean_2d(self):
    mlir_function = """
      func @mean(%arg0: tensor<?x?xf32>) -> tensor<?x1xf32> {
        %dim = "tf.Const"() {value = dense<1> : tensor<1xi32>}
               : () -> tensor<1xi32>
        %0 = "tf.Mean"(%arg0, %dim) { keep_dims = true }
             : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?x1xf32>
        return %0 : tensor<?x1xf32>
      }"""
    compiled = cpurt.compile(mlir_function, 'mean')
    arg0 = np.random.uniform(0.0, 1.0, size=(100, 200)).astype(np.float32)
    [res] = cpurt.execute(compiled, [arg0])
    mean = np.mean(arg0, axis=1, keepdims=True)
    np.testing.assert_allclose(res, mean, rtol=1e-05)


if __name__ == '__main__':
  np.random.seed(0)
  test.main()
