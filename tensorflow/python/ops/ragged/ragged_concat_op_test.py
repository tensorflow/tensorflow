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
"""Tests for ragged_array_ops.concat."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_concat_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedConcatOpTest(test_util.TensorFlowTestCase,
                         parameterized.TestCase):

  def _rt_inputs_to_tensors(self, rt_inputs, ragged_ranks=None):
    if ragged_ranks is None:
      ragged_ranks = [None] * len(rt_inputs)
    return [  # pylint: disable=g-long-ternary
        ragged_factory_ops.constant(rt_input, ragged_rank=rrank)
        if rrank != 0 else constant_op.constant(rt_input)
        for (rt_input, rrank) in zip(rt_inputs, ragged_ranks)
    ]

  @parameterized.parameters(
      dict(
          descr='Two rank-2 inputs with empty value axis=1',
          rt_inputs=([[]], [[]]),
          axis=1,
          expected=[[]]),
      dict(
          descr='Two rank-2 inputs (ragged_rank=1), axis=0',
          rt_inputs=(
              [['a00', 'a01'], [], ['a20', 'a21']],   # shape=(3, None)
              [['b00'], ['b10']]),                    # shape=(2, None)
          axis=0,
          expected=[[b'a00', b'a01'], [], [b'a20', b'a21'], [b'b00'],
                    [b'b10']]),
      dict(
          descr='Two rank-2 inputs (ragged_rank=1), axis=1',
          rt_inputs=(
              [['a00', 'a01'], [], ['a20', 'a21', 'a22']],   # shape=(3, None)
              [['b00'], ['b10', 'b11', 'b12'], ['b20']]),    # shape=(3, None)
          axis=1,
          expected=[
              [b'a00', b'a01', b'b00'],
              [b'b10', b'b11', b'b12'],
              [b'a20', b'a21', b'a22', b'b20']]),
      dict(
          descr='Two rank-2 inputs (ragged_rank=1), axis=-2',
          rt_inputs=(
              [['a00', 'a01'], [], ['a20', 'a21']],   # shape=(3, None)
              [['b00'], ['b10']]),                    # shape=(2, None)
          axis=-2,
          expected=[[b'a00', b'a01'], [], [b'a20', b'a21'], [b'b00'],
                    [b'b10']]),
      dict(
          descr='Two rank-2 inputs (ragged_rank=1), axis=-1',
          rt_inputs=(
              [['a00', 'a01'], [], ['a20', 'a21', 'a22']],   # shape=(3, None)
              [['b00'], ['b10', 'b11', 'b12'], ['b20']]),    # shape=(3, None)
          axis=-1,
          expected=[
              [b'a00', b'a01', b'b00'],
              [b'b10', b'b11', b'b12'],
              [b'a20', b'a21', b'a22', b'b20']],
          expected_shape=[3, None]),
      dict(
          descr='Three rank-2 inputs (ragged_rank=1), axis=0',
          rt_inputs=(
              [['a00', 'a01'], [], ['a20', 'a21', 'a22']],   # shape=(3, None)
              [['b00'], ['b10']],                            # shape=(2, None)
              [['c00'], ['c10', 'c11'], ['c21']]),           # shape=(3, None)
          axis=0,
          expected=[[b'a00', b'a01'], [], [b'a20', b'a21', b'a22'], [b'b00'],
                    [b'b10'], [b'c00'], [b'c10', b'c11'], [b'c21']]),
      dict(
          descr='Three rank-2 inputs (ragged_rank=1), axis=1',
          rt_inputs=(
              [['a00', 'a01'], [], ['a20', 'a21', 'a22']],   # shape=(3, None)
              [['b00'], ['b10', 'b11', 'b12'], ['b20']],     # shape=(3, None)
              [[], ['c10', 'c11'], ['c20', 'c21']]),         # shape=(3, None)
          axis=1,
          expected=[
              [b'a00', b'a01', b'b00'],
              [b'b10', b'b11', b'b12', b'c10', b'c11'],
              [b'a20', b'a21', b'a22', b'b20', b'c20', b'c21']]),
      dict(
          descr='Three rank-3 inputs (ragged_rank=2), axis=0',
          rt_inputs=(
              [[['a000', 'a001'], ['a010']],
               [['a100', 'a101', 'a102'], ['a110', 'a111']]],
              [[['b000']], [['b100', 'b101'], ['b110']]],
              [[], [['c100', 'c101', 'c102', 'c103']], [[], ['c210', 'c211']]]),
          axis=0,
          expected=[
              [[b'a000', b'a001'], [b'a010']],
              [[b'a100', b'a101', b'a102'], [b'a110', b'a111']],
              [[b'b000']],
              [[b'b100', b'b101'], [b'b110']],
              [],
              [[b'c100', b'c101', b'c102', b'c103']],
              [[], [b'c210', b'c211']]]),
      dict(
          descr='Three rank-3 inputs (ragged_rank=2), axis=1',
          rt_inputs=(
              [[['a000', 'a001'], ['a010']],
               [['a100', 'a101', 'a102'], ['a110', 'a111']]],
              [[['b000']], [['b100', 'b101'], ['b110']]],
              [[], [[], ['c110', 'c111']]]),
          axis=1,
          expected=[
              [[b'a000', b'a001'], [b'a010'], [b'b000']],
              [[b'a100', b'a101', b'a102'], [b'a110', b'a111'],
               [b'b100', b'b101'], [b'b110'], [], [b'c110', b'c111']]]),
      dict(
          descr='Three rank-3 inputs (ragged_rank=2), axis=2',
          rt_inputs=(
              [[['a000', 'a001'], ['a010']],
               [['a100', 'a101', 'a102'], ['a110', 'a111']]],
              [[[], ['b010', 'b011']], [['b100', 'b101'], ['b110']]],
              [[['c000'], ['c010']], [[], ['c110', 'c111']]]),
          axis=2,
          expected=[
              [[b'a000', b'a001', b'c000'],
               [b'a010', b'b010', b'b011', b'c010']],
              [[b'a100', b'a101', b'a102', b'b100', b'b101'],
               [b'a110', b'a111', b'b110', b'c110', b'c111']]]),
      dict(
          descr='Three rank-3 inputs (ragged_rank=2), axis=-1',
          rt_inputs=(
              [[['a000', 'a001'], ['a010']],
               [['a100', 'a101', 'a102'], ['a110', 'a111']]],
              [[[], ['b010', 'b011']], [['b100', 'b101'], ['b110']]],
              [[['c000'], ['c010']], [[], ['c110', 'c111']]]),
          axis=-1,
          expected=[
              [[b'a000', b'a001', b'c000'],
               [b'a010', b'b010', b'b011', b'c010']],
              [[b'a100', b'a101', b'a102', b'b100', b'b101'],
               [b'a110', b'a111', b'b110', b'c110', b'c111']]]),
      dict(
          descr='ragged_concat([uniform, ragged, uniform], axis=1)',
          ragged_ranks=[0, 1, 0],
          rt_inputs=(
              [['0('], ['1('], ['2(']],                   # shape=(3, 1)
              [['b00'], ['b10', 'b11', 'b12'], ['b20']],  # shape=(3, None)
              [[')0'], [')1'], [')2']]),                  # shape=(3, 1)
          axis=1,
          expected=[
              [b'0(', b'b00', b')0'],
              [b'1(', b'b10', b'b11', b'b12', b')1'],
              [b'2(', b'b20', b')2']]),
      dict(
          descr='ragged_concat([uniform, uniform], axis=0)',
          ragged_ranks=[0, 0],
          rt_inputs=(
              [['a00', 'a01'], ['a10', 'a11'], ['a20', 'a21']],  # shape=(3, 2)
              [['b00', 'b01', 'b02'], ['b10', 'b11', 'b12']]),   # shape=(2, 3)
          axis=0,
          expected=[
              [b'a00', b'a01'], [b'a10', b'a11'], [b'a20', b'a21'],
              [b'b00', b'b01', b'b02'], [b'b10', b'b11', b'b12']],
          expected_ragged_rank=1),
      dict(
          descr='ragged_concat([uniform, ragged], axis=0)',
          ragged_ranks=[0, 1],
          rt_inputs=(
              [['a00', 'a01'], ['a10', 'a11'], ['a20', 'a21']],  # shape=(3, 2)
              [['b00', 'b01', 'b02'], ['b10', 'b11', 'b12']]),   # shape=(2, 3)
          axis=0,
          expected=[
              [b'a00', b'a01'], [b'a10', b'a11'], [b'a20', b'a21'],
              [b'b00', b'b01', b'b02'], [b'b10', b'b11', b'b12']]),
      dict(
          descr='ragged_concat([uniform, ragged], axis=0) with rank-3 inputs',
          ragged_ranks=[0, 2],
          rt_inputs=(
              [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],  # shape = (2, 2, 2)
              [[[8], [8, 8]]]),                      # shape = (2, None, None)
          axis=0,
          expected=[[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8], [8, 8]]]),
      dict(
          descr='Two rank-3 inputs with ragged_rank=1, axis=-1',
          ragged_ranks=[1, 1],
          rt_inputs=(
              [[[0, 1], [2, 3], [4, 5]], [], [[6, 7], [8, 9]]],
              [[[9, 8], [7, 6], [5, 4]], [], [[3, 2], [1, 0]]]),
          axis=-1,
          expected=[
              [[0, 1, 9, 8], [2, 3, 7, 6], [4, 5, 5, 4]], [],
              [[6, 7, 3, 2], [8, 9, 1, 0]]],
          expected_ragged_rank=1),
      dict(
          descr='ragged_concat([vector, vector], axis=0)',
          ragged_ranks=[0, 0],
          rt_inputs=([1, 2, 3], [4, 5, 6]),
          axis=0,
          expected=[1, 2, 3, 4, 5, 6]),
      dict(
          descr='One input (so ragged_concat is a noop)',
          rt_inputs=([['a00', 'a01'], [], ['a20', 'a21']],),
          axis=0,
          expected=[[b'a00', b'a01'], [], [b'a20', b'a21']]),
  )   # pyformat: disable
  def testRaggedConcat(self,
                       descr,
                       rt_inputs,
                       axis,
                       expected,
                       ragged_ranks=None,
                       expected_ragged_rank=None,
                       expected_shape=None):
    rt_inputs = self._rt_inputs_to_tensors(rt_inputs, ragged_ranks)
    concatenated = ragged_concat_ops.concat(rt_inputs, axis)
    if expected_ragged_rank is not None:
      self.assertEqual(concatenated.ragged_rank, expected_ragged_rank)
    if expected_shape is not None:
      self.assertEqual(concatenated.shape.as_list(), expected_shape)
    self.assertAllEqual(concatenated, expected)

  @parameterized.parameters(
      dict(
          rt_inputs=(),
          axis=0,
          error=ValueError,
          message=r'rt_inputs may not be empty\.'),
      dict(
          rt_inputs=([[1, 2]], [[3, 4]]),
          axis=r'foo',
          error=TypeError,
          message='axis must be an int'),
      dict(
          rt_inputs=([[1, 2]], [[3, 4]]),
          axis=-3,
          error=ValueError,
          message='axis=-3 out of bounds: expected -2<=axis<2'),
      dict(
          rt_inputs=([[1, 2]], [[3, 4]]),
          axis=2,
          error=ValueError,
          message='axis=2 out of bounds: expected -2<=axis<2'),
      dict(
          ragged_ranks=(0, 0),
          rt_inputs=([[1, 2]], [[3, 4], [5, 6]]),
          axis=1,
          error=(ValueError, errors.InvalidArgumentError)),
  )
  def testStaticError(self,
                      rt_inputs,
                      axis,
                      error,
                      message=None,
                      ragged_ranks=None):
    rt_inputs = self._rt_inputs_to_tensors(rt_inputs, ragged_ranks)
    self.assertRaisesRegex(error, message, ragged_concat_ops.concat, rt_inputs,
                           axis)

  @parameterized.parameters([
      dict(
          ragged_ranks=(1, 1),
          rt_inputs=([[1, 2]], [[3, 4], [5, 6]]),
          axis=1,
          error=errors.InvalidArgumentError,
          message='Input tensors have incompatible shapes'),
  ])
  def testRuntimeError(self, rt_inputs, axis, error, message,
                       ragged_ranks=None):
    if context.executing_eagerly():
      return
    rt_inputs = [
        array_ops.placeholder_with_default(rt, shape=None) for rt in rt_inputs
    ]
    concatenated = ragged_concat_ops.concat(rt_inputs, axis)
    with self.assertRaisesRegex(error, message):
      self.evaluate(concatenated)

  def testNegativeAxisWithUnknownRankError(self):
    if context.executing_eagerly():
      return
    rt_inputs = [
        array_ops.placeholder(dtypes.int64),
        array_ops.placeholder(dtypes.int64)
    ]
    self.assertRaisesRegex(
        ValueError, r'axis may only be negative if ndims is statically known.',
        ragged_concat_ops.concat, rt_inputs, -1)

  def testSingleTensorInput(self):
    """Tests ragged_concat with a single tensor input.

    Usually, we pass a list of values in for rt_inputs.  However, you can
    also pass in a single value (as with tf.concat), in which case it simply
    returns that tensor.  This test exercises that path.
    """
    rt_inputs = ragged_factory_ops.constant([[1, 2], [3, 4]])
    concatenated = ragged_concat_ops.concat(rt_inputs, 0)
    self.assertAllEqual(concatenated, [[1, 2], [3, 4]])


if __name__ == '__main__':
  googletest.main()
