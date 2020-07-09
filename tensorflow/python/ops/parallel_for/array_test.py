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
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops.parallel_for import control_flow_ops as pfor_control_flow_ops
from tensorflow.python.ops.parallel_for.test_util import PForTestCase
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class ArrayTest(PForTestCase):

  def test_gather(self):
    x = random_ops.random_uniform([3, 3, 3, 3])
    x2 = array_ops.placeholder_with_default(x, shape=None)  # Has dynamic shape.

    def loop_fn(i):
      outputs = []
      x_i = array_ops.gather(x, i)
      for y in [x, x2, x_i]:
        for axis in [0, 2, -1]:
          outputs.append(array_ops.gather(y, 2, axis=axis))
          outputs.append(
              array_ops.gather(y, math_ops.cast(2, dtypes.int64), axis=axis))
          outputs.append(
              array_ops.gather(y, 2, axis=math_ops.cast(axis, dtypes.int64)))
          outputs.append(
              array_ops.gather(y, math_ops.cast(i, dtypes.int64), axis=axis))
          outputs.append(array_ops.gather(y, [i], axis=axis))
          outputs.append(array_ops.gather(y, [i, 2], axis=axis))
          outputs.append(array_ops.gather(y, [[2, i], [i, 1]], axis=axis))

        outputs.append(array_ops.gather(y, [0, 1, 2], axis=1, batch_dims=1))
        outputs.append(array_ops.gather(y, [i, 1, 2], axis=2, batch_dims=1))
        outputs.append(array_ops.gather(y, [[2, i], [i, 1], [2, 1]],
                                        axis=-1, batch_dims=1))

      return outputs

    self._test_loop_fn(loop_fn, 3)

  def test_gather_nd(self):
    x = random_ops.random_uniform([3, 3, 3])

    def loop_fn(i):
      outputs = []
      x_i = array_ops.gather(x, i)
      outputs.append(array_ops.gather_nd(x_i, [0], batch_dims=0))
      outputs.append(array_ops.gather_nd(x_i, [i], batch_dims=0))
      outputs.append(array_ops.gather_nd(x_i, [[i], [i], [i]], batch_dims=1))
      return outputs

    self._test_loop_fn(loop_fn, 3)

  def test_shape(self):
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      return array_ops.shape(x_i), array_ops.shape(x_i, out_type=dtypes.int64)

    self._test_loop_fn(loop_fn, 3)

  def test_size(self):
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      return array_ops.size(x_i), array_ops.size(x_i, out_type=dtypes.int64)

    self._test_loop_fn(loop_fn, 3)

  def test_rank(self):
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      return array_ops.rank(x_i)

    self._test_loop_fn(loop_fn, 3)

  def test_shape_n(self):
    x = random_ops.random_uniform([3, 2, 3])
    y = random_ops.random_uniform([3])

    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      y_i = array_ops.gather(y, i)
      return array_ops.shape_n([x_i, x, y,
                                y_i]), array_ops.shape_n([x_i, x, y, y_i],
                                                         out_type=dtypes.int64)

    self._test_loop_fn(loop_fn, 3)

  def test_reshape(self):
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return array_ops.reshape(x1, [-1]), array_ops.reshape(x1, [1, 3, 1, -1])

    self._test_loop_fn(loop_fn, 3)

  def test_fill(self):

    def loop_fn(i):
      return array_ops.fill((2, 3), i)

    self._test_loop_fn(loop_fn, 3)

  def test_broadcast_to(self):
    x = random_ops.random_uniform([3, 2, 1, 3])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return (array_ops.broadcast_to(x1, [2, 2, 3]),
              array_ops.broadcast_to(x1, [1, 2, 1, 3]))

    self._test_loop_fn(loop_fn, 3)

  def test_expand_dims(self):
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return array_ops.expand_dims(
          x1, axis=-1), array_ops.expand_dims(
              x1, axis=1)

    self._test_loop_fn(loop_fn, 3)

  def test_one_hot(self):
    indices = random_ops.random_uniform([3, 2, 3],
                                        minval=0,
                                        maxval=4,
                                        dtype=dtypes.int32)

    def loop_fn(i):
      indices_i = array_ops.gather(indices, i)
      return (array_ops.one_hot(indices_i, depth=4, on_value=2., off_value=-2.),
              array_ops.one_hot(indices_i, depth=4, axis=1))

    self._test_loop_fn(loop_fn, 3)

  def test_searchsorted(self):
    sorted_inputs = math_ops.cumsum(
        random_ops.random_uniform([3, 2, 4]), axis=-1)
    values = random_ops.random_uniform([2, 3], minval=-1, maxval=4.5)

    def loop_fn(i):
      inputs_i = array_ops.gather(sorted_inputs, i)
      return [
          array_ops.searchsorted(
              inputs_i, values, out_type=dtypes.int32,
              side="left"),  # creates LowerBound op.
          array_ops.searchsorted(
              inputs_i, values, out_type=dtypes.int64, side="right")
      ]  # creates UpperBound op.

    self._test_loop_fn(loop_fn, 3)

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

    with self.assertRaisesRegex(ValueError, "expected to be loop invariant"):
      pfor_control_flow_ops.pfor(loop_fn, 2, fallback_to_while_loop=False)

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

    self._test_loop_fn(loop_fn, 3)

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

    self._test_loop_fn(loop_fn, 3)

  def test_split_v(self):
    x = random_ops.random_uniform([3, 6, 3])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return (array_ops.split(x1, [2, 1, 3],
                              axis=0), array_ops.split(x1, [3], axis=-1))

    self._test_loop_fn(loop_fn, 3)

  def test_squeeze(self):
    x = random_ops.random_uniform([5, 1, 2, 1])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return (array_ops.squeeze(x1, axis=0), array_ops.squeeze(x1, axis=-1),
              array_ops.squeeze(x1))

    self._test_loop_fn(loop_fn, 3)

  def test_reverse(self):
    x = random_ops.random_uniform([3, 4, 2, 3])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return (array_ops.reverse(x1, axis=[0]),
              array_ops.reverse(x1, axis=[-1]),
              array_ops.reverse(x1, axis=[1, -1]))

    self._test_loop_fn(loop_fn, 3)

  def test_transpose(self):
    x = random_ops.random_uniform([3, 2, 3, 4])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return array_ops.transpose(x1, [2, 1, 0])

    self._test_loop_fn(loop_fn, 3)

  def test_conjugate_transpose(self):
    x = math_ops.complex(
        random_ops.random_uniform([3, 2, 3, 4]),
        random_ops.random_uniform([3, 2, 3, 4]))

    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      return array_ops.conjugate_transpose(x_i, [2, 1, 0])

    self._test_loop_fn(loop_fn, 3)

  def test_zeros_like(self):
    x = random_ops.random_uniform([3, 2, 3])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      z = array_ops.zeros_like(x1),
      return z, z + x1

    self._test_loop_fn(loop_fn, 3)

  def test_concat_v2(self):
    x = random_ops.random_uniform([3, 2, 3])
    y = random_ops.random_uniform([2, 3])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return array_ops.concat([x1, x1, y],
                              axis=0), array_ops.concat([x1, x1, y], axis=-1)

    self._test_loop_fn(loop_fn, 3)

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

      self._test_loop_fn(loop_fn, 3)

  def test_identity_n(self):
    x = random_ops.random_uniform([3, 4])

    def loop_fn(i):
      return array_ops.identity_n([x, array_ops.gather(x, i)])

    self._test_loop_fn(loop_fn, 3)

  def test_matrix_band_part(self):
    x = random_ops.random_uniform([3, 4, 2, 2])

    for num_lower, num_upper in ((0, -1), (-1, 0), (1, 1)):
      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        return array_ops.matrix_band_part(
            array_ops.gather(x, i), num_lower=num_lower, num_upper=num_upper)

      # pylint: enable=cell-var-from-loop

    self._test_loop_fn(loop_fn, 3)

  def test_matrix_diag(self):
    x = random_ops.random_uniform([3, 2, 4])

    def loop_fn(i):
      diagonal = array_ops.gather(x, i)
      return array_ops.matrix_diag(
          diagonal, k=(0, 1), num_rows=4, num_cols=5, align="RIGHT_LEFT")

    self._test_loop_fn(loop_fn, 3)

  def test_matrix_diag_part(self):
    x = random_ops.random_uniform([3, 4, 6])

    def loop_fn(i):
      input = array_ops.gather(x, i)  # pylint: disable=redefined-builtin
      return array_ops.matrix_diag_part(
          input, k=(-2, 0), padding_value=3, align="RIGHT_LEFT")

    self._test_loop_fn(loop_fn, 3)

  def test_diag(self):
    for x in (random_ops.random_uniform([3, 4]),
              random_ops.random_uniform([3, 4, 2])):
      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        inp = array_ops.gather(x, i)
        return array_ops.diag(inp)

      # pylint: disable=cell-var-from-loop
      self._test_loop_fn(loop_fn, 3)

  def test_diag_part(self):
    for x in (random_ops.random_uniform([3, 2, 2]),
              random_ops.random_uniform([3, 4, 2, 4, 2])):
      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        inp = array_ops.gather(x, i)  # pylint: disable=redefined-builtin
        return array_ops.diag_part(inp)

      # pylint: disable=cell-var-from-loop
      self._test_loop_fn(loop_fn, 3)

  def test_matrix_set_diag(self):
    matrices = random_ops.random_uniform([3, 4, 4])
    diags = random_ops.random_uniform([3, 4])
    bands = random_ops.random_uniform([3, 3, 4])

    def loop_fn(i):
      matrix_i = array_ops.gather(matrices, i)
      diag_i = array_ops.gather(diags, i)
      results = [
          array_ops.matrix_set_diag(matrix_i, diag_i),
          array_ops.matrix_set_diag(matrices[0, ...], diag_i),
          array_ops.matrix_set_diag(matrix_i, diags[0, ...]),
      ]

      k = (-1, 1)
      band_i = array_ops.gather(bands, i)
      for align in ["RIGHT_LEFT", "LEFT_RIGHT"]:
        results.extend([
            array_ops.matrix_set_diag(matrix_i, band_i, k=k, align=align),
            array_ops.matrix_set_diag(
                matrices[0, ...], band_i, k=k, align=align),
            array_ops.matrix_set_diag(
                matrix_i, bands[0, ...], k=k, align=align)
        ])
      return results

    self._test_loop_fn(loop_fn, 3)

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

    self._test_loop_fn(loop_fn, 3)

  def test_strided_slice_loop_variant(self):
    x = random_ops.random_uniform([3, 3, 4, 4, 2, 2, 2])

    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      return x_i[i:i+1, ...]

    # Test the fallback to while loop for a ConversionNotImplementedError is
    # handled.
    self._test_loop_fn(loop_fn, 3, fallback_to_while_loop=True)
    # Without fallback, ValueError is thrown.
    with self.assertRaisesRegex(ValueError, "expected to be loop invariant"):
      self._test_loop_fn(loop_fn, 3, fallback_to_while_loop=False)

  def test_depth_to_space(self):
    x = random_ops.random_uniform([2, 3, 2, 2, 12])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return array_ops.depth_to_space(x1, 2, data_format="NHWC")

    self._test_loop_fn(loop_fn, 2)

  def test_space_to_depth(self):
    x = random_ops.random_uniform([2, 3, 12, 12, 3])

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return array_ops.space_to_depth(x1, 2, data_format="NHWC")

    self._test_loop_fn(loop_fn, 2)

  def test_batch_to_space_nd(self):
    x = random_ops.random_uniform([7, 5 * 2 * 3, 2, 2, 3, 2])
    block_shapes = [2, 3]
    crops = [[1, 2], [1, 0]]

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return array_ops.batch_to_space_nd(x1, block_shapes, crops)

    self._test_loop_fn(loop_fn, 7)

  def test_space_to_batch_nd(self):
    x = random_ops.random_uniform([7, 5, 2 * 2 - 3, 2 * 3 - 1, 3, 2])
    block_shapes = [2, 3]
    paddings = [[1, 2], [1, 0]]

    def loop_fn(i):
      x1 = array_ops.gather(x, i)
      return array_ops.space_to_batch_nd(x1, block_shapes, paddings)

    self._test_loop_fn(loop_fn, 7)

  def test_check_numerics(self):
    x = random_ops.random_uniform([2, 3, 4])

    def loop_fn(i):
      x_i = array_ops.gather(x, i)
      return array_ops.check_numerics(x_i, "test_message")

    self._test_loop_fn(loop_fn, 2)


if __name__ == "__main__":
  test.main()
