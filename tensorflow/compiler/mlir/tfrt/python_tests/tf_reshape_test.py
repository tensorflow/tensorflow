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


class TfReshapeTest(test.TestCase):

  def test_reshape_unknown_1d(self):
    # TODO(ezhulenev): Make it work without shape constraint.
    mlir_function = """
      func @test(%arg0: tensor<?xf32>
                {cpurt.constraint = "shape"}) -> tensor<?x?xf32> {
        %0 = "tf.Const"() { value = dense<[2, -1]> : tensor<2xi32> }
             : () -> tensor<2xi32>
        %1 = "tf.Reshape"(%arg0, %0)
             : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
        return %1 : tensor<?x?xf32>
      }"""

    # TODO(ezhulenev): Make it work with default executable.
    compiled = cpurt.compile(mlir_function, 'test',
                             tf_cpurt.Specialization.ALWAYS)

    d0 = np.random.randint(1, 10) * 2

    arg0 = np.random.uniform(0, 10.0, size=(d0)).astype(np.float32)

    [res] = cpurt.execute(compiled, [arg0])
    np.testing.assert_allclose(res, np.reshape(arg0, (2, -1)), atol=0.0)


if __name__ == '__main__':
  test.main()
