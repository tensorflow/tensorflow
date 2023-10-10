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
"""Tests for tensorflow/python/util/type_annotations.py."""

import typing
from absl.testing import parameterized

from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.util import type_annotations


class TypeAnnotationsTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  @parameterized.parameters([
      (typing.Union[int, float], 'Union'),
      (typing.Tuple[int, ...], 'Tuple'),
      (typing.Tuple[int, float, float], 'Tuple'),
      (typing.Mapping[int, float], 'Mapping'),
      (typing.Union[typing.Tuple[int], typing.Tuple[int, ...]], 'Union'),
      # These predicates return False for Generic types w/ no parameters:
      (typing.Union, None),
      (typing.Tuple, None),
      (typing.Mapping, None),
      (int, None),
      (12, None),
  ])
  def testGenericTypePredicates(self, tp, expected):
    self.assertEqual(
        type_annotations.is_generic_union(tp), expected == 'Union')
    self.assertEqual(
        type_annotations.is_generic_tuple(tp), expected == 'Tuple')
    self.assertEqual(
        type_annotations.is_generic_mapping(tp), expected == 'Mapping')

  @parameterized.parameters([
      (typing.Union[int, float], (int, float)),
      (typing.Tuple[int, ...], (int, Ellipsis)),
      (typing.Tuple[int, float, float], (
          int,
          float,
          float,
      )),
      (typing.Mapping[int, float], (int, float)),
      (typing.Union[typing.Tuple[int],
                    typing.Tuple[int,
                                 ...]], (typing.Tuple[int], typing.Tuple[int,
                                                                         ...])),
  ])
  def testGetGenericTypeArgs(self, tp, expected):
    self.assertEqual(type_annotations.get_generic_type_args(tp), expected)

  def testIsForwardRef(self):
    tp = typing.Union['B', int]
    tp_args = type_annotations.get_generic_type_args(tp)
    self.assertTrue(type_annotations.is_forward_ref(tp_args[0]))
    self.assertFalse(type_annotations.is_forward_ref(tp_args[1]))


if __name__ == '__main__':
  googletest.main()
