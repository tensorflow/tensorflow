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
"""Tests for tf.Log1p JIT compilation."""

import numpy as np

from tensorflow.compiler.mlir.tfrt.jit.python_binding import tf_cpurt
from tensorflow.python.platform import test

specializations = [
    tf_cpurt.Specialization.ENABLED,
    tf_cpurt.Specialization.DISABLED,
    tf_cpurt.Specialization.ALWAYS,
]


def log1p_1d():
  return """
  func @log1p(%arg0: tensor<?xf32>) -> tensor<?xf32> {
    %0 = "tf.Log1p"(%arg0): (tensor<?xf32>) -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }"""


def log1p_2d():
  return """
  func @log1p(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = "tf.Log1p"(%arg0): (tensor<?x?xf32>) -> tensor<?x?xf32>
    return %0 : tensor<?x?xf32>
  }"""


cpurt = tf_cpurt.TfCpurtExecutor()


def test_log1p(fn, rank):
  for specialize in specializations:
    compiled = cpurt.compile(fn(), "log1p", specialize)

    for _ in range(100):
      shape = np.random.randint(0, 10, size=(rank))
      arg = np.random.uniform(0, 10.0, size=shape).astype(np.float32)

      [res] = cpurt.execute(compiled, [arg])
      np.testing.assert_allclose(res, np.log1p(arg), atol=1e-06)


class TfLog1PTest(test.TestCase):

  def test_1d(self):
    test_log1p(log1p_1d, 1)

  def test_2d(self):
    test_log1p(log1p_2d, 2)


if __name__ == "__main__":
  np.random.seed(0)
  test.main()
