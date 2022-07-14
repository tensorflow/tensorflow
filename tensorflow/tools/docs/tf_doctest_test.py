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
"""Tests for tf_doctest."""

import doctest

from absl.testing import absltest
from absl.testing import parameterized

from tensorflow.tools.docs import tf_doctest_lib


class TfDoctestOutputCheckerTest(parameterized.TestCase):

  @parameterized.parameters(
      # Don't match ints.
      ['result = 1', []],
      # Match floats.
      ['0.0', [0.]],
      ['text 1.0 text', [1.]],
      ['text 1. text', [1.]],
      ['text .1 text', [.1]],
      ['text 1e3 text', [1000.]],
      ['text 1.e3 text', [1000.]],
      ['text +1. text', [1.]],
      ['text -1. text', [-1.]],
      ['text 1e+3 text', [1000.]],
      ['text 1e-3 text', [0.001]],
      ['text +1E3 text', [1000.]],
      ['text -1E3 text', [-1000.]],
      ['text +1e-3 text', [0.001]],
      ['text -1e+3 text', [-1000.]],
      # Match at the start and end of a string.
      ['.1', [.1]],
      ['.1 text', [.1]],
      ['text .1', [.1]],
      ['0.1 text', [.1]],
      ['text 0.1', [.1]],
      ['0. text', [0.]],
      ['text 0.', [0.]],
      ['1e-1 text', [.1]],
      ['text 1e-1', [.1]],
      # Don't match floats mixed into text
      ['text1.0 text', []],
      ['text 1.0text', []],
      ['text1.0text', []],
      ['0x12e4', []],  #  not 12000
      ['TensorBoard: http://128.0.0.1:8888', []],
      # With a newline
      ['1.0 text\n 2.0 3.0 text', [1., 2., 3.]],
      # With ints and a float.
      ['shape (1,2,3) value -1e9', [-1e9]],
      # "." after a float.
      ['No floats at end of sentence: 1.0.', []],
      ['No floats with ellipsis: 1.0...', []],
      # A numpy array
      ["""array([[1., 2., 3.],
                 [4., 5., 6.]], dtype=float32)""", [1, 2, 3, 4, 5, 6]
      ],
      # Match both parts of a complex number
      # python style
      ['(0.0002+30000j)', [0.0002, 30000]],
      ['(2.3e-10-3.34e+9j)', [2.3e-10, -3.34e+9]],
      # numpy style
      ['array([1.27+5.j])', [1.27, 5]],
      ['(2.3e-10+3.34e+9j)', [2.3e-10, 3.34e+9]],
      ["""array([1.27e-09+5.e+00j,
                 2.30e+01-1.e-03j])""", [1.27e-09, 5.e+00, 2.30e+01, -1.e-03]],
      # Check examples in tolerence.
      ['1e-6', [0]],
      ['0.0', [1e-6]],
      ['1.000001e9', [1e9]],
      ['1e9', [1.000001e9]],
  )
  def test_extract_floats(self, text, expected_floats):
    extract_floats = tf_doctest_lib._FloatExtractor()
    output_checker = tf_doctest_lib.TfDoctestOutputChecker()

    (text_parts, extracted_floats) = extract_floats(text)
    text_with_wildcards = '...'.join(text_parts)

    # Check that the lengths match before doing anything else.
    try:
      self.assertLen(extracted_floats, len(expected_floats))
    except AssertionError as e:
      msg = '\n\n  expected: {}\n  found:     {}'.format(
          expected_floats, extracted_floats)
      e.args = (e.args[0] + msg,)
      raise e

    # The floats should match according to allclose
    try:
      self.assertTrue(
          output_checker._allclose(expected_floats, extracted_floats))
    except AssertionError as e:
      msg = '\n\nexpected:  {}\nfound:     {}'.format(expected_floats,
                                                      extracted_floats)
      e.args = (e.args[0] + msg,)
      raise e

    # The wildcard text should match the input text, according to the
    # OutputChecker base class.
    try:
      self.assertTrue(doctest.OutputChecker().check_output(
          want=text_with_wildcards, got=text, optionflags=doctest.ELLIPSIS))
    except AssertionError as e:
      msg = '\n\n  expected: {}\n  found:     {}'.format(
          text_with_wildcards, text)
      e.args = (e.args[0] + msg,)
      raise e

  @parameterized.parameters(
      # CHeck examples out of tolerence.
      ['1.001e-2', [0]],
      ['0.0', [1.001e-3]],
  )
  def test_fail_tolerences(self, text, expected_floats):
    extract_floats = tf_doctest_lib._FloatExtractor()
    output_checker = tf_doctest_lib.TfDoctestOutputChecker()

    (_, extracted_floats) = extract_floats(text)

    # These floats should not match according to allclose
    try:
      self.assertFalse(
          output_checker._allclose(expected_floats, extracted_floats))
    except AssertionError as e:
      msg = ('\n\nThese matched! They should not have.\n'
             '\n\n  Expected:  {}\n  found:     {}'.format(
                 expected_floats, extracted_floats))
      e.args = (e.args[0] + msg,)
      raise e

  def test_no_floats(self):
    want = 'text ... text'
    got = 'text 1.0 1.2 1.9 text'
    output_checker = tf_doctest_lib.TfDoctestOutputChecker()
    self.assertTrue(
        output_checker.check_output(
            want=want, got=got, optionflags=doctest.ELLIPSIS))

  @parameterized.parameters(['1.0, ..., 1.0', '1.0, 1.0, 1.0'],
                            ['1.0, 1.0..., 1.0', '1.0, 1.002, 1.0'])
  def test_warning_messages(self, want, got):
    output_checker = tf_doctest_lib.TfDoctestOutputChecker()

    output_checker.check_output(
        want=want, got=got, optionflags=doctest.ELLIPSIS)

    example = doctest.Example('None', want=want)
    result = output_checker.output_difference(
        example=example, got=got, optionflags=doctest.ELLIPSIS)
    self.assertIn("doesn't work if *some* of the", result)

  @parameterized.parameters(
      ['<...>', ('<...>', False)],
      ['TensorFlow', ('TensorFlow', False)],
      [
          'tf.Variable([[1, 2], [3, 4]])',
          ('tf.Variable([[1, 2], [3, 4]])', False)
      ],
      ['<tf.Tensor: shape=(), dtype=float32, numpy=inf>', ('inf', True)],
      [
          '<tf.RaggedTensor:... shape=(2, 2), numpy=1>',
          ('<tf.RaggedTensor:... shape=(2, 2), numpy=1>', False)
      ],
      [
          """<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
              array([[2, 2],
                     [3, 5]], dtype=int32)>""",
          ('\n              array([[2, 2],\n                     [3, 5]], '
           'dtype=int32)', True)
      ],
      [
          '[<tf.Tensor: shape=(2,), dtype=int32, numpy=array([1, 2], '
          'dtype=int32)>, '
          '<tf.Tensor: shape=(2,), dtype=int32, numpy=array([3, 4], '
          'dtype=int32)>]',
          ('[array([1, 2], dtype=int32), array([3, 4], dtype=int32)]', True)
      ],
  )
  def test_tf_tensor_numpy_output(self, string, expected_output):
    output_checker = tf_doctest_lib.TfDoctestOutputChecker()
    output = output_checker._tf_tensor_numpy_output(string)
    self.assertEqual(expected_output, output)

if __name__ == '__main__':
  absltest.main()
