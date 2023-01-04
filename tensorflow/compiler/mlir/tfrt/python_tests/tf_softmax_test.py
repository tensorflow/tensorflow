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


def softmax(x):
  z = x - np.max(x, axis=-1, keepdims=True)
  numerator = np.exp(z)
  denominator = np.sum(numerator, axis=-1, keepdims=True)
  result = numerator / denominator
  return result


class TfSoftmaxTest(test.TestCase):

  def test_dynamic_softmax(self):
    mlir_function = """
        func.func @test(%input: tensor<?x?xf32>) -> tensor<?x?xf32> {
          %0 = "tf.Softmax"(%input) : (tensor<?x?xf32>) -> tensor<?x?xf32>
          func.return %0 : tensor<?x?xf32>
      }"""

    compiled = jitrt.compile(mlir_function, 'test', vectorize=True)

    arg0 = np.random.uniform(1, 5, size=(8, 8)).astype(np.float32)

    [res] = jitrt.execute(compiled, [arg0])
    np.testing.assert_allclose(res, softmax(arg0), atol=0.00001)

  def test_static_softmax(self):
    mlir_function = """
        func.func @test(%input: tensor<10x8xf32>) -> tensor<10x8xf32> {
          %0 = "tf.Softmax"(%input) : (tensor<10x8xf32>) -> tensor<10x8xf32>
          func.return %0 : tensor<10x8xf32>
      }"""

    compiled = jitrt.compile(mlir_function, 'test', vectorize=True)

    arg0 = np.random.uniform(1, 5, size=(10, 8)).astype(np.float32)

    [res] = jitrt.execute(compiled, [arg0])
    np.testing.assert_allclose(res, softmax(arg0), atol=0.00001)


if __name__ == '__main__':
  test.main()
