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

import enum
import numpy as np

from absl import flags
from absl.testing import parameterized
from tensorflow import math

from tensorflow.compiler.mlir.tfrt.jit.python_binding import tf_jitrt
from tensorflow.python.platform import test

jitrt = tf_jitrt.TfJitRtExecutor()

FLAGS = flags.FLAGS
flags.DEFINE_integer('iters', '1000', 'Number of test iterations')
flags.DEFINE_integer('vector_size', '128', 'Iteration vector size')


# We cannot read flags from the parameterized test, so we cannot pass a value
# for rtol to the test function (rtol does depend on the 'vector_size' flag).
# Pass this enum instead so that we read the flags only in the test function.
class Rtol(enum.Enum):
  ZERO = 0
  BASE = 1
  AVX2 = 2


def mlir_func_1d(op_name):
  return f"""
  func.func @test(%arg0: tensor<?xf32>) -> tensor<?xf32> {{
    %0 = "tf.{op_name}"(%arg0): (tensor<?xf32>) -> tensor<?xf32>
    func.return %0 : tensor<?xf32>
  }}"""


def test_1d(op_name, fn, vectorize=False, lb=-1.0, ub=1.0, rtol_enum=Rtol.BASE):
  compiled = jitrt.compile(mlir_func_1d(op_name), 'test', vectorize=vectorize)
  rtols = {}
  rtols[Rtol.ZERO] = 0.0
  # Not all approximations are identical to TF's.
  rtols[Rtol.BASE] = 1e-6
  # For some ops we can match TF with the right build flags.
  # Note that vector size also matters: for vectors whose size is not a multiple
  # of the machine's vector length, Eigen (and therefore TF) computes some
  # elements differently (e.g. via libc).
  rtols[Rtol.AVX2] = rtols[Rtol.BASE]
  # Use 16 as the machine vector's length to be both simple and future-proof.
  if jitrt.built_with('AVX2') and FLAGS.vector_size % 16 == 0:
    rtols[Rtol.AVX2] = 0.0

  rtol = rtols[rtol_enum]

  for _ in range(FLAGS.iters):
    arg = np.random.uniform(lb, ub, size=(FLAGS.vector_size)).astype(np.float32)

    [res] = jitrt.execute(compiled, [arg])
    np.testing.assert_allclose(res, fn(arg), rtol=rtol, atol=1e-7)


class TfMathOpsTest(parameterized.TestCase):
  @parameterized.named_parameters(
      # Note: for now we are testing for identical results to TF (and therefore
      # Eigen). In the short term, this will work because Eigen's approximations
      # don't change too often. However, in the long term could become a
      # maintenance burden.
      # TODO(ecg): relax tolerances to accommodate for changes in Eigen, and add
      # a flag to control the minimum tolerance, so that we can manually check
      # for identical results to Eigen.
      ('exp_scalar', 'Exp', math.exp, False, Rtol.AVX2),
      ('exp_vector', 'Exp', math.exp, True, Rtol.AVX2),
      ('reciprocal_scalar', 'Reciprocal', math.reciprocal, False, Rtol.BASE),
      ('reciprocal_vector', 'Reciprocal', math.reciprocal, True, Rtol.BASE),
      # Rsqrt: The AVX2 intrinsic is only emitted with vectorization.
      ('rsqrt_scalar', 'Rsqrt', math.rsqrt, False, Rtol.BASE),
      ('rsqrt_vector', 'Rsqrt', math.rsqrt, True, Rtol.AVX2),
      ('tanh_scalar', 'Tanh', math.tanh, False, Rtol.AVX2),
      ('tanh_vector', 'Tanh', math.tanh, True, Rtol.AVX2),
  )

  def test_op(self, op_name, fn, vectorize, rtol_enum):
    test_1d(op_name, fn, vectorize=vectorize, rtol_enum=rtol_enum)

if __name__ == '__main__':
  np.random.seed(0)
  test.main()
