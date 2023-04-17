# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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


class TfScatterTest(test.TestCase):

  def test_scatter(self):
    mlir_function = """
      func.func @test(%index: tensor<5x2xi32>, %updates: tensor<5x8xf32>,
          %out: tensor<1x11x11xf32>) -> tensor<1x11x11xf32> {
        %1 = "tf.TensorScatterAdd"(%out, %index, %updates)
          : (tensor<1x11x11xf32>, tensor<5x2xi32>, tensor<5x8xf32>) ->
          tensor<1x11x11xf32>
        return %1 : tensor<1x11x11xf32>
      }
    """
    compiled = jitrt.compile(mlir_function, 'test', vectorize=True)
    index = np.array([[0, 0], [0, 0], [0, 5], [0, 5], [0, 10]], dtype=np.int32)
    updates = np.array(
        [[1] * 8, [2] * 8, [3] * 8, [4] * 8, [5] * 8], dtype=np.float32
    )
    out = np.zeros((1, 11, 11), dtype=np.float32)

    exp_res = np.zeros((1, 11, 11), dtype=np.float32)
    exp_res[0][0][:8] += 3
    exp_res[0][5][:8] += 7
    exp_res[0][10][:8] += 5

    [res] = jitrt.execute(compiled, [index, updates, out])
    np.testing.assert_allclose(res, exp_res)


if __name__ == '__main__':
  test.main()
