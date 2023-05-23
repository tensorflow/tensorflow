# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for testlib_runner."""

import pathlib

from absl.testing import absltest
import numpy as np

from tensorflow.compiler.xla.runtime.runner import runner

# We assume that the testlib runner is defined in the same project as this test.
r = runner.Runner(f'{pathlib.Path(__file__).parent.resolve()}/testlib_runner')


class TestlibRunnerTest(absltest.TestCase):

  def testScalarAdd(self):
    module = """
      func.func @add(%arg0: i32) -> i32 {
        %0 = arith.constant 42 : i32
        %1 = arith.addi %arg0, %0 : i32
        return %1 : i32
      }"""

    [res] = r.execute(module, 'add', [42])
    self.assertEqual(res, 84)

  def testTensorAdd(self):
    module = """
      func.func @addtensor(%arg0: memref<?xf32>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 3 : index
        %step = arith.constant 1 : index

        scf.for %i = %c0 to %c1 step %step {
          %0 = arith.constant 42.0 : f32
          %1 = memref.load %arg0[%i] : memref<?xf32>
          %2 = arith.addf %0, %1 : f32
          memref.store %2, %arg0[%i] : memref<?xf32>
        }
        
        func.return
      }"""

    arg = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    [res] = r.execute(module, 'addtensor', [arg], inout=[0])
    self.assertTrue(
        np.array_equal(res, np.array([43.0, 44.0, 45.0], dtype=np.float32)))

  def testTensorReturn(self):
    module = """
      func.func @returntensor(%arg0: memref<?xf32>) -> memref<4xf32> {
      %out = memref.alloc() : memref<4xf32>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 4 : index
      %step = arith.constant 1 : index

      scf.for %i = %c0 to %c1 step %step {
        %0 = memref.load %arg0[%i] : memref<?xf32>
        memref.store %0, %out[%i] : memref<4xf32>
      }

      return %out : memref<4xf32>
    }"""

    arg = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    [res] = r.execute(module, 'returntensor', [arg])

    self.assertTrue(
        np.array_equal(res, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)))

if __name__ == '__main__':
  absltest.main()
