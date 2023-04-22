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
# ==============================================================================
"""Tests for unsorted_segment_join_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


class UnicodeTestCase(test.TestCase):
  """Test case with Python3-compatible string comparator."""

  def assertAllEqualUnicode(self, truth, actual):
    self.assertAllEqual(
        np.array(truth).astype('U'),
        np.array(actual).astype('U'))


@test_util.run_all_in_graph_and_eager_modes
class UnsortedSegmentJoinOpTest(UnicodeTestCase, parameterized.TestCase):

  def test_basic_np_array(self):
    inputs = [['Y', 'q', 'c'], ['Y', '6', '6'], ['p', 'G', 'a']]
    segment_ids = [1, 0, 1]
    num_segments = 2
    separator = ':'
    output_array = [['Y', '6', '6'], ['Y:p', 'q:G', 'c:a']]

    res = self.evaluate(
        string_ops.unsorted_segment_join(
            inputs=inputs,
            segment_ids=segment_ids,
            num_segments=num_segments,
            separator=separator))
    self.assertAllEqual(res.shape, np.array(output_array).shape)
    self.assertAllEqualUnicode(res, output_array)

  def test_segment_id_and_input_empty(self):
    inputs = np.array([], dtype=np.string_)
    segment_ids = np.array([], dtype=np.int32)
    num_segments = 3
    separator = ':'
    output_array = ['', '', '']
    res = self.evaluate(
        string_ops.unsorted_segment_join(
            inputs=inputs,
            segment_ids=segment_ids,
            num_segments=num_segments,
            separator=separator))
    self.assertAllEqual(res.shape, np.array(output_array).shape)
    self.assertAllEqualUnicode(res, output_array)

  def test_type_check(self):
    inputs = [['Y', 'q', 'c'], ['Y', '6', '6'], ['p', 'G', 'a']]
    segment_ids = np.array([1, 0, 1], dtype=np.int32)
    num_segments = np.array(2, dtype=np.int32)
    separator = ':'
    output_array = [['Y', '6', '6'], ['Y:p', 'q:G', 'c:a']]

    res = self.evaluate(
        string_ops.unsorted_segment_join(
            inputs=inputs,
            segment_ids=segment_ids,
            num_segments=num_segments,
            separator=separator))
    self.assertAllEqual(res.shape, np.array(output_array).shape)
    self.assertAllEqualUnicode(res, output_array)

    segment_ids = np.array([1, 0, 1], dtype=np.int64)
    num_segments = np.array(2, dtype=np.int64)
    res = self.evaluate(
        string_ops.unsorted_segment_join(
            inputs=inputs,
            segment_ids=segment_ids,
            num_segments=num_segments,
            separator=separator))
    self.assertAllEqual(res.shape, np.array(output_array).shape)
    self.assertAllEqualUnicode(res, output_array)

  def test_basic_tensor(self):
    inputs = constant_op.constant([['Y', 'q', 'c'], ['Y', '6', '6'],
                                   ['p', 'G', 'a']])
    segment_ids = constant_op.constant([1, 0, 1])
    num_segments = 2
    separator = ':'
    output_array = constant_op.constant([['Y', '6', '6'], ['Y:p', 'q:G',
                                                           'c:a']])

    res = self.evaluate(
        string_ops.unsorted_segment_join(
            inputs=inputs,
            segment_ids=segment_ids,
            num_segments=num_segments,
            separator=separator))
    self.assertAllEqual(res, output_array)
    self.assertAllEqual(res.shape, output_array.get_shape())

  def test_multiple_segment_join(self):
    inputs = [['Y', 'q', 'c'], ['Y', '6', '6'], ['p', 'G', 'a']]
    segment_ids_1 = [1, 0, 1]
    num_segments_1 = 2
    separator_1 = ':'
    output_array_1 = [['Y', '6', '6'], ['Y:p', 'q:G', 'c:a']]

    res = self.evaluate(
        string_ops.unsorted_segment_join(
            inputs=inputs,
            segment_ids=segment_ids_1,
            num_segments=num_segments_1,
            separator=separator_1))
    self.assertAllEqualUnicode(res, output_array_1)
    self.assertAllEqual(res.shape, np.array(output_array_1).shape)

    segment_ids_2 = [1, 1]
    num_segments_2 = 2
    separator_2 = ''
    output_array_2 = [['', '', ''], ['YY:p', '6q:G', '6c:a']]

    res = self.evaluate(
        string_ops.unsorted_segment_join(
            inputs=res,
            segment_ids=segment_ids_2,
            num_segments=num_segments_2,
            separator=separator_2))
    self.assertAllEqualUnicode(res, output_array_2)
    self.assertAllEqual(res.shape, np.array(output_array_2).shape)

  @parameterized.parameters([
      {
          'inputs': [[[['q'], ['s']], [['f'], ['F']], [['h'], ['0']]],
                     [[['E'], ['j']], [['2'], ['k']], [['N'], ['d']]],
                     [[['G'], ['M']], [['1'], ['S']], [['N'], ['7']]],
                     [[['8'], ['W']], [['W'], ['G']], [['j'], ['d']]]],
          'segment_ids': [1, 1, 0, 2],
          'num_segments':
              3,
          'separator':
              ':',
          'output_array': [[[['G'], ['M']], [['1'], ['S']], [['N'], ['7']]],
                           [[['q:E'], ['s:j']], [['f:2'], ['F:k']],
                            [['h:N'], ['0:d']]],
                           [[['8'], ['W']], [['W'], ['G']], [['j'], ['d']]]],
      },
      {
          'inputs': [[['Q', 'b'], ['c', 'p']], [['i', '9'], ['n', 'b']],
                     [['T', 'h'], ['g', 'z']]],
          'segment_ids': [[0, 1], [1, 0], [1, 0]],
          'num_segments': 2,
          'separator': ':',
          'output_array': [['Q:n:g', 'b:b:z'], ['c:i:T', 'p:9:h']]
      },
      {
          'inputs': [[['Q', 'b'], ['b', 'p']], [['i', '9'], ['n', 'b']],
                     [['T', 'h'], ['g', 'z']]],
          'segment_ids': [[[2, 1], [0, 0]], [[2, 0], [2, 2]], [[0, 2], [1, 0]]],
          'num_segments': 3,
          'separator': ':',
          'output_array': ['b:p:9:T:z', 'b:g', 'Q:i:n:b:h']
      },
      {
          'inputs': [[['z'], ['h']], [['c'], ['z']], [['V'], ['T']]],
          'segment_ids': [0, 1, 1],
          'num_segments': 3,
          'separator': ':',
          'output_array': [[['z'], ['h']], [['c:V'], ['z:T']], [[''], ['']]]
      },
  ])
  def test_multiple_cases_with_different_dims(self, inputs, segment_ids,
                                              num_segments, separator,
                                              output_array):
    res = self.evaluate(
        string_ops.unsorted_segment_join(
            inputs=inputs,
            segment_ids=segment_ids,
            num_segments=num_segments,
            separator=separator))
    self.assertAllEqualUnicode(res, output_array)
    self.assertAllEqual(res.shape, np.array(output_array).shape)

  @parameterized.parameters([
      {
          'separator': '',
          'output_array': ['thisisatest']
      },
      {
          'separator': ':',
          'output_array': ['this:is:a:test']
      },
      {
          'separator': 'UNK',
          'output_array': ['thisUNKisUNKaUNKtest']
      },
  ])
  def testSeparator(self, separator, output_array):
    inputs = ['this', 'is', 'a', 'test']
    segment_ids = [0, 0, 0, 0]
    num_segments = 1
    res = self.evaluate(
        string_ops.unsorted_segment_join(
            inputs=inputs,
            segment_ids=segment_ids,
            num_segments=num_segments,
            separator=separator))
    self.assertAllEqual(res.shape, np.array(output_array).shape)
    self.assertAllEqualUnicode(res, output_array)

  def test_fail_segment_id_exceeds_segment_nums(self):
    inputs = [['Y', 'q', 'c'], ['Y', '6', '6'], ['p', 'G', 'a']]
    segment_ids = [1, 0, 1]
    num_segments = 1
    separator = ':'

    with self.assertRaises(errors_impl.InvalidArgumentError):
      self.evaluate(
          string_ops.unsorted_segment_join(
              inputs=inputs,
              segment_ids=segment_ids,
              num_segments=num_segments,
              separator=separator))

  def test_fail_segment_id_dim_does_not_match(self):
    inputs = [['Y', 'q', 'c'], ['Y', '6', '6'], ['p', 'G', 'a']]
    segment_ids = [1, 0, 1, 1]
    num_segments = 2
    separator = ':'

    if not context.executing_eagerly():
      with self.assertRaises(ValueError):
        self.evaluate(
            string_ops.unsorted_segment_join(
                inputs=inputs,
                segment_ids=segment_ids,
                num_segments=num_segments,
                separator=separator))
    else:
      with self.assertRaises(errors_impl.InvalidArgumentError):
        self.evaluate(
            string_ops.unsorted_segment_join(
                inputs=inputs,
                segment_ids=segment_ids,
                num_segments=num_segments,
                separator=separator))

  def test_fail_segment_id_empty_input_non_empty(self):
    inputs = [['Y', 'q', 'c'], ['Y', '6', '6'], ['p', 'G', 'a']]
    segment_ids = np.array([], dtype=np.int32)
    num_segments = 2
    separator = ':'
    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      self.evaluate(
          string_ops.unsorted_segment_join(
              inputs=inputs,
              segment_ids=segment_ids,
              num_segments=num_segments,
              separator=separator))

  def test_empty_input(self):
    inputs = np.array([], dtype=np.string_)
    segment_ids = [1, 0, 1]
    num_segments = 2
    separator = ':'
    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      self.evaluate(
          string_ops.unsorted_segment_join(
              inputs=inputs,
              segment_ids=segment_ids,
              num_segments=num_segments,
              separator=separator))

  def test_fail_negative_segment_id(self):
    inputs = [['Y', 'q', 'c'], ['Y', '6', '6'], ['p', 'G', 'a']]
    segment_ids = [-1, 0, -1]
    num_segments = 1
    separator = ':'
    with self.assertRaises(errors_impl.InvalidArgumentError):
      self.evaluate(
          string_ops.unsorted_segment_join(
              inputs=inputs,
              segment_ids=segment_ids,
              num_segments=num_segments,
              separator=separator))


if __name__ == '__main__':
  test.main()
