# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `tensorflow::FunctionParameterCanonicalizer`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import test
from tensorflow.python.util import _function_parameter_canonicalizer_binding_for_test


class FunctionParameterCanonicalizerTest(test.TestCase):

  def setUp(self):
    super(FunctionParameterCanonicalizerTest, self).setUp()
    self._matmul_func = (
        _function_parameter_canonicalizer_binding_for_test
        .FunctionParameterCanonicalizer([
            'a', 'b', 'transpose_a', 'transpose_b', 'adjoint_a', 'adjoint_b',
            'a_is_sparse', 'b_is_sparse', 'name'
        ], (False, False, False, False, False, False, None)))

  def testPosOnly(self):
    self.assertEqual(
        self._matmul_func.canonicalize(2, 3),
        [2, 3, False, False, False, False, False, False, None])

  def testPosOnly2(self):
    self.assertEqual(
        self._matmul_func.canonicalize(2, 3, True, False, True),
        [2, 3, True, False, True, False, False, False, None])

  def testPosAndKwd(self):
    self.assertEqual(
        self._matmul_func.canonicalize(
            2, 3, transpose_a=True, name='my_matmul'),
        [2, 3, True, False, False, False, False, False, 'my_matmul'])

  def testPosAndKwd2(self):
    self.assertEqual(
        self._matmul_func.canonicalize(2, b=3),
        [2, 3, False, False, False, False, False, False, None])

  def testMissingPos(self):
    with self.assertRaisesRegex(TypeError,
                                'Missing required positional argument'):
      self._matmul_func.canonicalize(2)

  def testMissingPos2(self):
    with self.assertRaisesRegex(TypeError,
                                'Missing required positional argument'):
      self._matmul_func.canonicalize(
          transpose_a=True, transpose_b=True, adjoint_a=True)

  def testTooManyArgs(self):
    with self.assertRaisesRegex(TypeError, 'Too many arguments were given'):
      self._matmul_func.canonicalize(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

  def testInvalidKwd(self):
    with self.assertRaisesRegex(TypeError,
                                'Got an unexpected keyword argument'):
      self._matmul_func.canonicalize(2, 3, hohoho=True)

  def testDuplicatedArg(self):
    with self.assertRaisesRegex(TypeError,
                                "Got multiple values for argument 'b'"):
      self._matmul_func.canonicalize(2, 3, False, b=4)

  def testDuplicatedArg2(self):
    with self.assertRaisesRegex(
        TypeError, "Got multiple values for argument 'transpose_a'"):
      self._matmul_func.canonicalize(2, 3, False, transpose_a=True)


if __name__ == '__main__':
  test.main()
