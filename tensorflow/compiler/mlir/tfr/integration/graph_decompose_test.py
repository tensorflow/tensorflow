# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.compiler.mlir.tfr.integrattion.graph_decompose."""

import os

from tensorflow.compiler.mlir.tfr.resources import gen_composite_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test

_lib_dir = os.path.dirname(gen_composite_ops.__file__)
_lib_name = os.path.basename(gen_composite_ops.__file__)[4:].replace(
    '.py', '.so')
load_library.load_op_library(os.path.join(_lib_dir, _lib_name))


class GraphDecomposeTest(test.TestCase):

  def testAddN(self):
    add = def_function.function(gen_composite_ops.my_add_n)
    t1 = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    t2 = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    t3 = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    sq1 = add([t1])
    sq2 = add([t1, t2])
    sq3 = add([t1, t2, t3])
    self.assertAllEqual(sq1.numpy().reshape(-1), [1, 2, 3, 4])
    self.assertAllEqual(sq2.numpy().reshape(-1), [2, 4, 6, 8])
    self.assertAllEqual(sq3.numpy().reshape(-1), [3, 6, 9, 12])

  def testBiasedDense(self):
    biased_dense = def_function.function(gen_composite_ops.my_biased_dense)
    t1 = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    t2 = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    t3 = constant_op.constant([[-10.0, -10.0], [-10.0, -10.0]])
    sq = biased_dense(t1, t2, t3)
    self.assertAllEqual(sq.numpy().reshape(-1), [-3, 0, 5, 12])

  def testBiasedDenseRelu(self):
    biased_dense = def_function.function(gen_composite_ops.my_biased_dense)
    t1 = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    t2 = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    t3 = constant_op.constant([[-10.0, -10.0], [-10.0, -10.0]])
    sq = biased_dense(t1, t2, t3, act='relu')
    self.assertAllEqual(sq.numpy().reshape(-1), [0, 0, 5, 12])

  def testWithKnownKernel(self):

    @def_function.function
    def biasd_dense_elu(x, y, z):
      dot = gen_composite_ops.my_biased_dense(x, y, z)
      return nn_ops.elu(dot)  # with known kernel, should not expand.

    t1 = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    t2 = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    t3 = constant_op.constant([[-10.0, -10.0], [-10.0, -10.0]])
    sq = biasd_dense_elu(t1, t2, t3)
    self.assertAllClose(sq.numpy().reshape(-1), [-0.950213, 0, 5, 12])


if __name__ == '__main__':
  os.environ['TF_MLIR_TFR_LIB_DIR'] = 'tensorflow/compiler/mlir/tfr/resources'
  ops.enable_eager_execution()
  test.main()
