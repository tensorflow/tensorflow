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
"""Tests for ragged.squeeze."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_conversion_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_squeeze_op
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedSqueezeTest(test_util.TensorFlowTestCase,
                        parameterized.TestCase):

  @parameterized.parameters([
      {
          'input_list': []
      },
      {
          'input_list': [[]],
          'squeeze_ranks': [0]
      },
      {
          'input_list': [[[[], []], [[], []]]],
          'squeeze_ranks': [0]
      },
  ])
  def test_passing_empty(self, input_list, squeeze_ranks=None):
    rt = ragged_squeeze_op.squeeze(
        ragged_factory_ops.constant(input_list), squeeze_ranks)
    dt = array_ops.squeeze(constant_op.constant(input_list), squeeze_ranks)
    self.assertAllEqual(ragged_conversion_ops.to_tensor(rt), dt)

  @parameterized.parameters([
      {
          'input_list': [[1]],
          'squeeze_ranks': [0]
      },
      {
          'input_list': [[1]],
          'squeeze_ranks': [0, 1]
      },
      {
          'input_list': [[1, 2]],
          'squeeze_ranks': [0]
      },
      {
          'input_list': [[1], [2]],
          'squeeze_ranks': [1]
      },
      {
          'input_list': [[[[12], [11]]]],
          'squeeze_ranks': [0]
      },
      {
          'input_list': [[[[12], [11]]]],
          'squeeze_ranks': [1]
      },
      {
          'input_list': [[[[12], [11]]]],
          'squeeze_ranks': [3]
      },
      {
          'input_list': [[[[12], [11]]]],
          'squeeze_ranks': [0, 3]
      },
      {
          'input_list': [[[[12], [11]]]],
          'squeeze_ranks': [0, 1]
      },
      {
          'input_list': [[[[12], [11]]]],
          'squeeze_ranks': [1, 3]
      },
      {
          'input_list': [[[[12], [11]]]],
          'squeeze_ranks': [0, 1, 3]
      },
      {
          'input_list': [[[1], [2]], [[3], [4]]],
          'squeeze_ranks': [2]
      },
      {
          'input_list': [[1], [2]],
          'squeeze_ranks': [-1]
      },
  ])
  def test_passing_simple(self, input_list, squeeze_ranks=None):
    rt = ragged_squeeze_op.squeeze(
        ragged_factory_ops.constant(input_list), squeeze_ranks)
    dt = array_ops.squeeze(constant_op.constant(input_list), squeeze_ranks)
    self.assertAllEqual(ragged_conversion_ops.to_tensor(rt), dt)

  @parameterized.parameters([
      # ragged_conversion_ops.from_tensor does not work for this
      # {'input_list': [1]},
      {
          'input_list': [[1]],
          'squeeze_ranks': [0]
      },
      {
          'input_list': [[1, 2]],
          'squeeze_ranks': [0]
      },
      {
          'input_list': [[1], [2]],
          'squeeze_ranks': [1]
      },
      {
          'input_list': [[[[12], [11]]]],
          'squeeze_ranks': [0]
      },
      {
          'input_list': [[[[12], [11]]]],
          'squeeze_ranks': [1]
      },
      {
          'input_list': [[[[12], [11]]]],
          'squeeze_ranks': [3]
      },
      {
          'input_list': [[[[12], [11]]]],
          'squeeze_ranks': [0, 3]
      },
      {
          'input_list': [[[[12], [11]]]],
          'squeeze_ranks': [0, 1]
      },
      {
          'input_list': [[[[12], [11]]]],
          'squeeze_ranks': [1, 3]
      },
      {
          'input_list': [[[[12], [11]]]],
          'squeeze_ranks': [0, 1, 3]
      },
      {
          'input_list': [[[1], [2]], [[3], [4]]],
          'squeeze_ranks': [2]
      },
  ])
  def test_passing_simple_from_dense(self, input_list, squeeze_ranks=None):
    dt = constant_op.constant(input_list)
    rt = ragged_conversion_ops.from_tensor(dt)
    rt_s = ragged_squeeze_op.squeeze(rt, squeeze_ranks)
    dt_s = array_ops.squeeze(dt, squeeze_ranks)
    self.assertAllEqual(ragged_conversion_ops.to_tensor(rt_s), dt_s)

  @parameterized.parameters([
      {
          'input_list': [[[[[[1]], [[1, 2]]]], [[[[]], [[]]]]]],
          'output_list': [[[1], [1, 2]], [[], []]],
          'squeeze_ranks': [0, 2, 4]
      },
      {
          'input_list': [[[[[[1]], [[1, 2]]]], [[[[]], [[]]]]]],
          'output_list': [[[[[1]], [[1, 2]]]], [[[[]], [[]]]]],
          'squeeze_ranks': [0]
      },
  ])
  def test_passing_ragged(self, input_list, output_list, squeeze_ranks=None):
    rt = ragged_factory_ops.constant(input_list)
    rt_s = ragged_squeeze_op.squeeze(rt, squeeze_ranks)
    ref = ragged_factory_ops.constant(output_list)
    self.assertAllEqual(rt_s, ref)

  def test_passing_text(self):
    rt = ragged_factory_ops.constant([[[[[[[['H']], [['e']], [['l']], [['l']],
                                           [['o']]],
                                          [[['W']], [['o']], [['r']], [['l']],
                                           [['d']], [['!']]]]],
                                        [[[[['T']], [['h']], [['i']], [['s']]],
                                          [[['i']], [['s']]],
                                          [[['M']], [['e']], [['h']], [['r']],
                                           [['d']], [['a']], [['d']]],
                                          [[['.']]]]]]]])
    output_list = [[['H', 'e', 'l', 'l', 'o'], ['W', 'o', 'r', 'l', 'd', '!']],
                   [['T', 'h', 'i', 's'], ['i', 's'],
                    ['M', 'e', 'h', 'r', 'd', 'a', 'd'], ['.']]]
    ref = ragged_factory_ops.constant(output_list)
    rt_s = ragged_squeeze_op.squeeze(rt, [0, 1, 3, 6, 7])
    self.assertAllEqual(rt_s, ref)

  @parameterized.parameters([
      {
          'input_list': [[]],
          'squeeze_ranks': [1]
      },
      {
          'input_list': [[1, 2]],
          'squeeze_ranks': [1]
      },
      {
          'input_list': [[1], [2]],
          'squeeze_ranks': [0]
      },
      {
          'input_list': [[[[12], [11]]]],
          'squeeze_ranks': [0, 2]
      },
      {
          'input_list': [[[[12], [11]]]],
          'squeeze_ranks': [2]
      },
      {
          'input_list': [[[1], [2]], [[3], [4]]],
          'squeeze_ranks': [0]
      },
      {
          'input_list': [[[1], [2]], [[3], [4]]],
          'squeeze_ranks': [1]
      },
      {
          'input_list': [[], []],
          'squeeze_ranks': [1]
      },
      {
          'input_list': [[[], []], [[], []]],
          'squeeze_ranks': [1]
      },
  ])
  def test_failing_InvalidArgumentError(self, input_list, squeeze_ranks):
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(
          ragged_squeeze_op.squeeze(
              ragged_factory_ops.constant(input_list), squeeze_ranks))

  @parameterized.parameters([
      {
          'input_list': [[]]
      },
      {
          'input_list': [[1]]
      },
      {
          'input_list': [[1, 2]]
      },
      {
          'input_list': [[[1], [2]], [[3], [4]]]
      },
      {
          'input_list': [[1]]
      },
      {
          'input_list': [[[1], [2]], [[3], [4]]]
      },
      {
          'input_list': [[[[12], [11]]]]
      },
  ])
  def test_failing_no_squeeze_dim_specified(self, input_list):
    with self.assertRaises(ValueError):
      ragged_squeeze_op.squeeze(ragged_factory_ops.constant(input_list))

  @parameterized.parameters([
      {
          'input_list': [[[[12], [11]]]],
          'squeeze_ranks': [0, 1, 3]
      },
  ])
  def test_failing_axis_is_not_a_list(self, input_list, squeeze_ranks):
    with self.assertRaises(TypeError):
      tensor_ranks = constant_op.constant(squeeze_ranks)
      ragged_squeeze_op.squeeze(
          ragged_factory_ops.constant(input_list), tensor_ranks)


if __name__ == '__main__':
  googletest.main()
