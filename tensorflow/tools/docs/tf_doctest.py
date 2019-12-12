# Lint as: python3
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
"""Run doctests for tensorflow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import textwrap

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized

import numpy as np

import tensorflow.compat.v2 as tf

# We put doctest after absltest so that it picks up the unittest monkeypatch.
# Otherwise doctest tests aren't runnable at all.
import doctest  # pylint: disable=g-bad-import-order

tf.compat.v1.enable_v2_behavior()

FLAGS = flags.FLAGS

flags.DEFINE_string('module', None, 'A specific module to run doctest on.')
flags.DEFINE_boolean('list', None,
                     'List all the modules in the core package imported.')
flags.DEFINE_string('file', None, 'A specific file to run doctest on.')

flags.mark_flags_as_mutual_exclusive(['module', 'file'])
flags.mark_flags_as_mutual_exclusive(['list', 'file'])

PACKAGE = 'tensorflow.python.'


class _FloatExtractor(object):
  """Class for extracting floats from a string.

  For example:

  >>> text_parts, floats = _FloatExtractor()("Text 1.0 Text")
  >>> text_parts
  ["Text ", " Text"]
  >>> floats
  np.array([1.0])
  """

  # Note: non-capturing groups "(?" are not returned in matched groups, or by
  # re.split.
  _FLOAT_RE = re.compile(
      r"""
      (?:                        # (Non-capturing) Only start a match if:
         ^|                      # * At the Start of string, or
         (?<=[^\w.]              # * if the pervious character was not:
                                 #   * a word char or "."
         ))
      (                          # Captures the float value.
        [-+]?                    # Optional Sign
        (?:                      # Digits and exponent - something like:
          {digits_dot_maybe_digits}{exponent}?|   # "1.0" "1." "1.0e3", "1.e3"
          {dot_digits}{exponent}?|                # ".1" ".1e3"
          {digits}{exponent}                      # "1e3"
        )
      )
      (?=                        # Only accept the match if
        $|                       # * At the end of the string, or
        [^\w.]                   # * Next char is not a word char or "."
      )
      """.format(
          # Digits, a "." and optional more digits: "1.1".
          digits_dot_maybe_digits=r'(?:[0-9]+\.(?:[0-9]*))',
          # A "." with trailing digits ".23"
          dot_digits=r'(?:\.[0-9]+)',
          # digits: "12"
          digits=r'(?:[0-9]+)',
          # The exponent: An "e" or "E", optional sign, and at least one digit.
          # "e-123", "E+12", "e12"
          exponent=r'(?:[eE][-+]?[0-9]+)'),
      re.VERBOSE)

  def __call__(self, string):
    """Extracts floats from a string.

    >>> text_parts, floats = _FloatExtractor()("Text 1.0 Text")
    >>> text_parts
    ["Text ", " Text"]
    >>> floats
    np.array([1.0])

    Args:
      string: the string to extract floats from.

    Returns:
      A (string, array) pair, where `string` has each float replaced by "..."
      and `array` is a `float32` `numpy.array` containing the extracted floats.
    """
    texts = []
    floats = []
    for i, part in enumerate(self._FLOAT_RE.split(string)):
      if i % 2 == 0:
        texts.append(part)
      else:
        floats.append(float(part))

    return texts, np.array(floats)


class TfDoctestOutputChecker(doctest.OutputChecker, object):
  """Changes the `want` and `got` strings.

  This allows it to be customized before they are compared.
  """

  def __init__(self, *args, **kwargs):
    super(TfDoctestOutputChecker, self).__init__(*args, **kwargs)
    self.extract_floats = _FloatExtractor()
    self.text_good = None
    self.float_size_good = None

  _ADDRESS_RE = re.compile(r'\bat 0x[0-9a-f]*?>')

  def _allclose(self, want, got, rtol=1e-6, atol=1e-6):
    # Same default as: tensorflow/python/framework/test_util.py "assertAllClose"
    return np.allclose(want, got, rtol=rtol, atol=atol)

  def check_output(self, want, got, optionflags):
    """Compares the docstring output to the output gotten by running the code.

    Python addresses in the output are replaced with wildcards.

    Float values in the output compared as using `np.allclose`:

      * Float values are extracted from the text and replaced with wildcards.
      * The wildcard text is compared to the actual output.
      * The float values are compared using `np.allclose`.

    The method returns `True` if both the text comparison and the numeric
    comparison are successful.

    The numeric comparison will fail if either:

      * The wrong number of floats are found.
      * The float values are not within tolerence.

    Args:
      want: The output in the docstring.
      got: The output generated after running the snippet.
      optionflags: Flags passed to the doctest.

    Returns:
      A bool, indicating if the check was successful or not.
    """

    # Replace python's addresses with ellipsis (`...`) since it can change on
    # each execution.
    want = self._ADDRESS_RE.sub('at ...>', want)

    # Separate out the floats, and replace `want` with the wild-card version
    # "result=7.0" => "result=..."
    want_text_parts, self.want_floats = self.extract_floats(want)
    want_text_wild = '...'.join(want_text_parts)

    # Find the floats in the string returned by the test
    _, self.got_floats = self.extract_floats(got)

    self.text_good = super(TfDoctestOutputChecker, self).check_output(
        want=want_text_wild, got=got, optionflags=optionflags)
    if not self.text_good:
      return False

    if self.want_floats.size == 0:
      # If there are no floats in the "want" string, ignore all the floats in
      # the result. "np.array([ ... ])" matches "np.array([ 1.0, 2.0 ])"
      return True

    self.float_size_good = (self.want_floats.size == self.got_floats.size)

    if self.float_size_good:
      return self._allclose(self.want_floats, self.got_floats)
    else:
      return False

  def output_difference(self, example, got, optionflags):
    got = [got]

    # If the some of the float output is hidden with `...`, `float_size_good`
    # will be False. This is because the floats extracted from the string is
    # converted into a 1-D numpy array. Hence hidding floats is not allowed
    # anymore.
    if self.text_good:
      if not self.float_size_good:
        got.append("\n\nCAUTION: tf_doctest doesn't work if *some* of the "
                   "*float output* is hidden with a \"...\".")

    message = textwrap.dedent("""\n
        #############################################################
        Check the documentation
        (https://www.tensorflow.org/community/contribute/docs_ref) on how to write testable docstrings.
        #############################################################""")

    got.append(message)
    got = '\n'.join(got)
    return (super(TfDoctestOutputChecker,
                  self).output_difference(example, got, optionflags))


def find_modules():
  """Finds all the modules in the core package imported.

  Returns:
    A list containing all the modules in tensorflow.python.
  """

  tf_modules = []
  for name, module in sys.modules.items():
    if name.startswith(PACKAGE):
      tf_modules.append(module)

  return tf_modules


def filter_on_submodules(all_modules, submodule):
  """Filters all the modules based on the module flag.

  The module flag has to be relative to the core package imported.
  For example, if `submodule=keras.layers` then, this function will return
  all the modules in the submodule.

  Args:
    all_modules: All the modules in the core package.
    submodule: Submodule to filter from all the modules.

  Returns:
    All the modules in the submodule.
  """

  filtered_modules = [
      mod for mod in all_modules
      if PACKAGE + submodule in mod.__name__
  ]
  return filtered_modules


def get_module_and_inject_docstring(file_path):
  """Replaces the docstring of the module with the changed file's content.

  Args:
    file_path: Path to the file

  Returns:
    A list containing the module changed by the file.
  """

  file_path = os.path.abspath(file_path)
  mod_index = file_path.find(PACKAGE.replace('.', os.sep))
  file_mod_name, _ = os.path.splitext(file_path[mod_index:])
  file_module = sys.modules[file_mod_name.replace(os.sep, '.')]

  with open(file_path, 'r') as f:
    content = f.read()

  file_module.__doc__ = content

  return [file_module]


class TfTestCase(tf.test.TestCase):

  def set_up(self, test):
    self.setUp()

  def tear_down(self, test):
    self.tearDown()


def load_tests(unused_loader, tests, unused_ignore):
  """Loads all the tests in the docstrings and runs them."""

  tf_modules = find_modules()

  if FLAGS.module:
    tf_modules = filter_on_submodules(tf_modules, FLAGS.module)

  if FLAGS.list:
    print('**************************************************')
    for mod in tf_modules:
      print(mod.__name__)
    print('**************************************************')
    return tests

  if FLAGS.file:
    tf_modules = get_module_and_inject_docstring(FLAGS.file)

  for module in tf_modules:
    testcase = TfTestCase()
    tests.addTests(
        doctest.DocTestSuite(
            module,
            test_finder=doctest.DocTestFinder(exclude_empty=False),
            extraglobs={
                'tf': tf,
                'np': np,
                'os': os
            },
            setUp=testcase.set_up,
            tearDown=testcase.tear_down,
            checker=TfDoctestOutputChecker(),
            optionflags=(doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
                         | doctest.IGNORE_EXCEPTION_DETAIL
                         | doctest.DONT_ACCEPT_BLANKLINE),
        ))
  return tests


class TfDoctestOutputCheckerTest(parameterized.TestCase):
  """Tests for TFDoctestOutputChecker."""

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
      [
          textwrap.dedent("""
          array([[1., 2., 3.],
                 [4., 5., 6.]], dtype=float32)
          """), [1, 2, 3, 4, 5, 6]
      ],
      # Check examples in tolerence.
      ['1e-6', [0]],
      ['0.0', [1e-6]],
      ['1.000001e9', [1e9]],
      ['1e9', [1.000001e9]],
  )
  def test_extract_floats(self, text, expected_floats):
    extract_floats = _FloatExtractor()
    output_checker = TfDoctestOutputChecker()

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
      ['1.001e-6', [0]],
      ['0.0', [1.001e-6]],
      ['1.000001001e9', [1e9]],
      ['1e9', [1.000001001e9]],
  )
  def test_fail_tolerences(self, text, expected_floats):
    extract_floats = _FloatExtractor()
    output_checker = TfDoctestOutputChecker()

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
    output_checker = TfDoctestOutputChecker()
    self.assertTrue(
        output_checker.check_output(
            want=want, got=got, optionflags=doctest.ELLIPSIS))

  @parameterized.parameters(['1.0, ..., 1.0', '1.0, 1.0, 1.0'],
                            ['1.0, 1.0..., 1.0', '1.0, 1.002, 1.0'])
  def test_warning_messages(self, want, got):
    output_checker = TfDoctestOutputChecker()

    output_checker.check_output(
        want=want, got=got, optionflags=doctest.ELLIPSIS)

    example = doctest.Example('None', want=want)
    result = output_checker.output_difference(
        example=example, got=got, optionflags=doctest.ELLIPSIS)
    self.assertIn("doesn't work if *some* of the", result)


if __name__ == '__main__':
  absltest.main()
