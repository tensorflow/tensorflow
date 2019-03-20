# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for vectorization of array kernels."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops.parallel_for import control_flow_ops as pfor_control_flow_ops
from tensorflow.python.ops.parallel_for.test_util import PForTestCase
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class ArrayTest(PForTestCase):

  def test_gather(self):
    x = random_ops.random_uniform([3, 3, 3])

    def loop_fn(i):
      outputs = []
      x_i = array_ops.gather(x, i)
      for y in [x, x_i]:
        axes = [0, 2, -1] if y == x else [0]
        for axis in axes:
          outputs.append(array_ops.gather(y, 2, axis=axis))
          outputs.append(array_ops.gather(y, i, axis=axis))
          outputs.append(array_ops.gather(y, [i], axis=axis))
          outputs.append(array_ops.gather(y, [i, 2], axis=axis))
          outputs.append(array_ops.gather(y, [[2, i], [i, 1]], axis=axis))
      return outputs

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.float32] * 20)

  def test_shape(self):
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      return array_ops.shape(x_i), array_ops.shape(x_i, out_type=dtypes.int64)

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.int32, dtypes.int64])

  def test_size(self):
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      return array_ops.size(x_i), array_ops.size(x_i, out_type=dtypes.int64)

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.int32, dtypes.int64])

  def test_rank(self):
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      return array_ops.rank(x_i)

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.int32])

  def test_shape_n(self):
    x = random_ops.random_uniform([3, 2, 3])
    y = random_ops.random_uniform([3])

    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      y_i = array_ops.gather(y, i)
      return array_ops.shape_n([x_i, x, y, y_i]), array_ops.shape_n(
          [x_i, x, y, y_i], out_type=dtypes.int64)

    self._test_loop_fn(
        loop_fn, 3, loop_fn_dtypes=[dtypes.int32] * 4 + [dtypes.int64] * 4)

  def test_reshape(self):
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return array_ops.reshape(x1, [-1]), array_ops.reshape(x1, [1, 3, 1, -1])

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.float32] * 2)

  def test_expand_dims(self):
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return array_ops.expand_dims(
          x1, axis=-1), array_ops.expand_dims(
              x1, axis=1)

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.float32] * 2)

  def test_slice(self):
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return array_ops.slice(x1, begin=(0, 1), size=(2, 1))

    self._test_loop_fn(loop_fn, 3)

  def test_tile(self):
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return array_ops.tile(x1, [2, 1])

    self._test_loop_fn(loop_fn, 3)

  def test_tile_loop_dependent(self):
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return array_ops.tile(x1, [i, 1])

    with self.assertRaisesRegexp(ValueError, "expected to be loop invariant"):
      pfor_control_flow_ops.pfor(loop_fn, 2)

  def test_pack(self):
    x = random_ops.random_uniform([3, 2, 3])
    y = random_ops.random_uniform([2, 3])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return array_ops.stack([x1, y], axis=-1)

    self._test_loop_fn(loop_fn, 1)

  def test_unpack(self):
    x = random_ops.random_uniform([3, 2, 3, 4])

    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      return array_ops.unstack(
          x_i, 4, axis=-1), array_ops.unstack(
              x_i, 3, axis=1)

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.float32] * 7)

  def test_pad(self):
    x = random_ops.random_uniform([3, 2, 3])
    padding = constant_op.constant([[1, 2], [3, 4]])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return array_ops.pad(x1, padding, mode="CONSTANT")

    self._test_loop_fn(loop_fn, 3)

  def test_split(self):
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return array_ops.split(x1, 2, axis=0), array_ops.split(x1, 3, axis=-1)

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.float32] * 5)

  def test_split_v(self):
    x = random_ops.random_uniform([3, 6, 3])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return (array_ops.split(x1, [2, 1, 3], axis=0),
              array_ops.split(x1, [3], axis=-1))

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.float32] * 4)

  def test_transpose(self):
    x = random_ops.random_uniform([3, 2, 3, 4])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return array_ops.transpose(x1, [2, 1, 0])

    self._test_loop_fn(loop_fn, 3)

  def test_zeros_like(self):
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      z = array_ops.zeros_like(x1),
      return z, z + x1

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.float32] * 2)

  def test_concat_v2(self):
    x = random_ops.random_uniform([3, 2, 3])
    y = random_ops.random_uniform([2, 3])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return array_ops.concat(
          [x1, x1, y], axis=0), array_ops.concat(
              [x1, x1, y], axis=-1)

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.float32] * 2)

  def test_unary_cwise_ops(self):
    for op in [array_ops.identity, array_ops.stop_gradient]:
      with backprop.GradientTape(persistent=True) as g:
        x = random_ops.random_uniform([3, 5])
        g.watch(x)

      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        with g:
          x1 = array_ops.gather(x, i)
          y = op(x1) + x1
          loss = nn.l2_loss(y)
        return op(x), y, g.gradient(loss, x1)

      # pylint: enable=cell-var-from-loop

      self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.float32] * 3)

  def test_identity_n(self):
    x = random_ops.random_uniform([3, 4])

    def loop_fn(i):
      return array_ops.identity_n([x, array_ops.gather(x, i)])

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.float32] * 2)

  def test_matrix_diag_part(self):
    x = random_ops.random_uniform([3, 4, 2])

    def loop_fn(i):
      return array_ops.matrix_diag_part(array_ops.gather(x, i))

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.float32])

  def test_strided_slice(self):
    with backprop.GradientTape(persistent=True) as g:
      x = random_ops.random_uniform([3, 3, 4, 4, 2, 2, 2])
      g.watch(x)

    def loop_fn(i):
      with g:
        x_i = array_ops.gather(x, i)
        y = x_i[:2, ::2, 1::3, ..., array_ops.newaxis, 1]
        loss = nn.l2_loss(y)
      return y, g.gradient(loss, x_i)

    self._test_loop_fn(loop_fn, 3, loop_fn_dtypes=[dtypes.float32] * 2)


if __name__ == "__main__":
  test.main()
