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
"""Tests for tf_cpurt python binding."""

import numpy as np

import unittest
from tensorflow.compiler.mlir.tfrt.jit.python_binding import tf_cpurt


def log_1d():
  return """
  func @log_1d(%arg0: tensor<?xf32>) -> tensor<?xf32> {
    %0 = "tf.Log"(%arg0): (tensor<?xf32>) -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }"""


cpurt = tf_cpurt.TfCpurtExecutor()


class TfCpurtTest(googletest.TestCase):

  def test_log_1d(self):
    compiled = cpurt.compile(log_1d(), "log_1d")

    shape = np.random.randint(0, 10, size=(1))
    arg = np.random.uniform(0, 10.0, size=shape).astype(np.float32)

    [res] = cpurt.execute(compiled, [arg])
    np.testing.assert_allclose(res, np.log(arg), atol=1e-07)


if __name__ == "__main__":
  googletest.main()
