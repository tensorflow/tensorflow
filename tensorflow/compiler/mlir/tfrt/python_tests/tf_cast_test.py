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


class TfCastTest(test.TestCase):

  def test_cast_unsigned_signed(self):
    mlir_function = """
      func @test(%arg0: tensor<?xui32>)
           -> (tensor<?xi32>, tensor<?xui32>, tensor<?xi64>, tensor<?xui8>) {
        %0 = "tf.Cast"(%arg0) : (tensor<?xui32>) -> tensor<?xi32>
        %1 = "tf.Cast"(%0) : (tensor<?xi32>) -> tensor<?xui32>
        %2 = "tf.Cast"(%1) : (tensor<?xui32>) -> tensor<?xi64>
        %3 = "tf.Cast"(%2) : (tensor<?xi64>) -> tensor<?xui8>
        return %0, %1, %2, %3
          : tensor<?xi32>, tensor<?xui32>, tensor<?xi64>, tensor<?xui8>
      }"""

    compiled = cpurt.compile(mlir_function, 'test')

    arg0 = np.random.uniform(300, 3000, size=10).astype(np.uint32)

    [res0, res1, res2, res3] = cpurt.execute(compiled, [arg0])
    np.testing.assert_equal(res0, arg0)
    np.testing.assert_equal(res0.dtype, np.int32)
    np.testing.assert_equal(res1, arg0)
    np.testing.assert_equal(res1.dtype, np.uint32)
    np.testing.assert_equal(res2, arg0)
    np.testing.assert_equal(res2.dtype, np.int64)
    np.testing.assert_equal(res3, arg0.astype(np.uint8))
    np.testing.assert_equal(res3.dtype, np.uint8)


if __name__ == '__main__':
  test.main()
