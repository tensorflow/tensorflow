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
"""Tests for utils.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.platform import test


class UtilsTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(UtilsTest, self).setUp()
    self._old_np_doc_form = np_utils.get_np_doc_form()
    self._old_is_sig_mismatch_an_error = np_utils.is_sig_mismatch_an_error()

  def tearDown(self):
    np_utils.set_np_doc_form(self._old_np_doc_form)
    np_utils.set_is_sig_mismatch_an_error(self._old_is_sig_mismatch_an_error)
    super(UtilsTest, self).tearDown()

  # pylint: disable=unused-argument
  def testNpDocInlined(self):
    def np_fun(x):
      """np_fun docstring."""
      return
    np_utils.set_np_doc_form('inlined')
    @np_utils.np_doc(None, np_fun=np_fun)
    def f():
      """f docstring."""
      return
    expected = """TensorFlow variant of NumPy's `np_fun`.

Unsupported arguments: `x`.

f docstring.

Documentation for `numpy.np_fun`:

np_fun docstring."""
    self.assertEqual(expected, f.__doc__)

  @parameterized.named_parameters([
      (version, version, link) for version, link in  # pylint: disable=g-complex-comprehension
      [('dev',
        'https://numpy.org/devdocs/reference/generated/numpy.np_fun.html'),
       ('stable',
        'https://numpy.org/doc/stable/reference/generated/numpy.np_fun.html'),
       ('1.16',
        'https://numpy.org/doc/1.16/reference/generated/numpy.np_fun.html')
      ]])
  def testNpDocLink(self, version, link):
    def np_fun(x):
      """np_fun docstring."""
      return
    np_utils.set_np_doc_form(version)
    @np_utils.np_doc(None, np_fun=np_fun)
    def f():
      """f docstring."""
      return
    expected = """TensorFlow variant of NumPy's `np_fun`.

Unsupported arguments: `x`.

f docstring.

See the NumPy documentation for [`numpy.np_fun`](%s)."""
    expected = expected % (link)
    self.assertEqual(expected, f.__doc__)

  @parameterized.parameters([None, 1, 'a', '1a', '1.1a', '1.1.1a'])
  def testNpDocInvalid(self, invalid_flag):
    def np_fun(x):
      """np_fun docstring."""
      return
    np_utils.set_np_doc_form(invalid_flag)
    @np_utils.np_doc(None, np_fun=np_fun)
    def f():
      """f docstring."""
      return
    expected = """TensorFlow variant of NumPy's `np_fun`.

Unsupported arguments: `x`.

f docstring.

"""
    self.assertEqual(expected, f.__doc__)

  def testNpDocName(self):
    np_utils.set_np_doc_form('inlined')
    @np_utils.np_doc('foo')
    def f():
      """f docstring."""
      return
    expected = """TensorFlow variant of NumPy's `foo`.

f docstring.

"""
    self.assertEqual(expected, f.__doc__)

  # pylint: disable=unused-variable
  def testSigMismatchIsError(self):
    """Tests that signature mismatch is an error (when configured so)."""
    if not np_utils._supports_signature():
      self.skipTest('inspect.signature not supported')

    np_utils.set_is_sig_mismatch_an_error(True)

    def np_fun(x, y=1, **kwargs):
      return

    with self.assertRaisesRegex(TypeError, 'Cannot find parameter'):
      @np_utils.np_doc(None, np_fun=np_fun)
      def f1(a):
        return

    with self.assertRaisesRegex(TypeError, 'is of kind'):
      @np_utils.np_doc(None, np_fun=np_fun)
      def f2(x, kwargs):
        return

    with self.assertRaisesRegex(
        TypeError, 'Parameter "y" should have a default value'):
      @np_utils.np_doc(None, np_fun=np_fun)
      def f3(x, y):
        return

  def testSigMismatchIsNotError(self):
    """Tests that signature mismatch is not an error (when configured so)."""
    np_utils.set_is_sig_mismatch_an_error(False)

    def np_fun(x, y=1, **kwargs):
      return

    # The following functions all have signature mismatches, but they shouldn't
    # throw errors when is_sig_mismatch_an_error() is False.

    @np_utils.np_doc(None, np_fun=np_fun)
    def f1(a):
      return

    def f2(x, kwargs):
      return

    @np_utils.np_doc(None, np_fun=np_fun)
    def f3(x, y):
      return

  # pylint: enable=unused-variable


if __name__ == '__main__':
  test.main()
