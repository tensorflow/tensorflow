# -*- coding: utf-8 -*-
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
"""Tests for tf.strings.to_bytes op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized


from tensorflow.python.eager import def_function
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_string_ops
from tensorflow.python.platform import test


class StringsToBytesOpTest(test_util.TensorFlowTestCase,
                           parameterized.TestCase):

  @parameterized.parameters(
      # Scalar input -> vector output
      (b'hello', [b'h', b'e', b'l', b'l', b'o']),
      # Vector input -> 2D ragged output
      ([b'hello', b'123'],
       [[b'h', b'e', b'l', b'l', b'o'], [b'1', b'2', b'3']]),
      # 2D tensor input -> 3D ragged output
      ([[b'abc', b'de'], [b'fgh', b'']],
       [[[b'a', b'b', b'c'], [b'd', b'e']], [[b'f', b'g', b'h'], []]]),
      # 2D ragged input -> 3D ragged output
      (ragged_factory_ops.constant_value([[b'abc', b'de'], [b'f']]),
       [[[b'a', b'b', b'c'], [b'd', b'e']], [[b'f']]]),
      # 3D input -> 4D ragged output
      (ragged_factory_ops.constant_value(
          [[[b'big', b'small'], [b'red']], [[b'cat', b'dog'], [b'ox']]]),
       [[[[b'b', b'i', b'g'], [b's', b'm', b'a', b'l', b'l']],
         [[b'r', b'e', b'd']]],
        [[[b'c', b'a', b't'], [b'd', b'o', b'g']],
         [[b'o', b'x']]]]),
      # Empty string
      (b'', []),
      # Null byte
      (b'\x00', [b'\x00']),
      # Unicode
      (u'仅今年前'.encode('utf-8'),
       [b'\xe4', b'\xbb', b'\x85', b'\xe4', b'\xbb', b'\x8a', b'\xe5',
        b'\xb9', b'\xb4', b'\xe5', b'\x89', b'\x8d']),
  )
  def testStringToBytes(self, source, expected):
    expected = ragged_factory_ops.constant_value(expected, dtype=object)
    result = ragged_string_ops.string_bytes_split(source)
    self.assertAllEqual(expected, result)

  def testUnknownInputRankError(self):
    # Use a tf.function that erases shape information.
    @def_function.function(input_signature=[tensor_spec.TensorSpec(None)])
    def f(v):
      return ragged_string_ops.string_bytes_split(v)

    with self.assertRaisesRegex(ValueError,
                                'input must have a statically-known rank'):
      f(['foo'])


if __name__ == '__main__':
  test.main()
