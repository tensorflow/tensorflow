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
"""Tests for numerical correctness of tf.math operations."""

import numpy as np

from absl import flags
from absl.testing import parameterized
from tensorflow import math

from tensorflow.compiler.mlir.tfrt.jit.python_binding import tf_cpurt
from tensorflow.python.platform import test

cpurt = tf_cpurt.TfCpurtExecutor()

FLAGS = flags.FLAGS
flags.DEFINE_integer('iters', '1000', 'Number of test iterations')


def mlir_func_1d(op_name):
  return f"""
  func @test(%arg0: tensor<?xf32>) -> tensor<?xf32> {{
    %0 = "tf.{op_name}"(%arg0): (tensor<?xf32>) -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }}"""


def test_1d(op_name, fn, vectorize=False, lb=-1.0, ub=1.0, rtol=0.0):
  compiled = cpurt.compile(mlir_func_1d(op_name), 'test', vectorize=vectorize)

  for _ in range(FLAGS.iters):
    arg = np.random.uniform(lb, ub, size=(100)).astype(np.float32)

    [res] = cpurt.execute(compiled, [arg])
    np.testing.assert_allclose(res, fn(arg), rtol=rtol)


class TfMathOpsTest(parameterized.TestCase):
  # Not all approximations are identical to TF's.
  base_rtol = 1e-6
  # For some ops we can match TF with the right build flags.
  avx2_rtol = 0.0 if cpurt.built_with('AVX2') else base_rtol

  @parameterized.named_parameters(
      ('reciprocal_scalar', 'Reciprocal', math.reciprocal, False, 0.0),
      ('reciprocal_vector', 'Reciprocal', math.reciprocal, True, 0.0),
      # Rsqrt: The AVX2 intrinsic is only emitted with vectorization.
      ('rsqrt_scalar', 'Rsqrt', math.rsqrt, False, base_rtol),
      ('rsqrt_vector', 'Rsqrt', math.rsqrt, True, avx2_rtol),
      ('tanh_scalar', 'Tanh', math.tanh, False, avx2_rtol),
      ('tanh_vector', 'Tanh', math.tanh, True, avx2_rtol),
  )

  def test_op(self, op_name, fn, vectorize, rtol):
    test_1d(op_name, fn, vectorize=vectorize, rtol=rtol)

if __name__ == '__main__':
  np.random.seed(0)
  test.main()
