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


# Metadata operations that are noop at runtime, but exist in the Tensorflow
# graphs purely to facilitate graph construction and transformations.
class TfMetadataOpsTest(test.TestCase):

  def test_stop_gradient(self):
    mlir_function = """
      func.func @test(%arg0: tensor<?xf32>) -> tensor<?xf32> {
        %0 = "tf.StopGradient"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
        func.return %0 : tensor<?xf32>
      }"""
    compiled = jitrt.compile(mlir_function, 'test')
    arg0 = np.random.uniform(0.0, 1.0, size=(10)).astype(np.float32)
    [res] = jitrt.execute(compiled, [arg0])
    np.testing.assert_allclose(res, arg0, rtol=0.0)


if __name__ == '__main__':
  test.main()
