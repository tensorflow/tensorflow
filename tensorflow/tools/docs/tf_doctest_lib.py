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

import doctest
import re
import textwrap

import numpy as np


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
      (                          # Captures the float value.
        (?:
           [-+]|                 # Start with a sign is okay anywhere.
           (?:                   # Otherwise:
               ^|                # Start after the start of string
               (?<=[^\w.])       # Not after a word char, or a .
           )
        )
        (?:                      # Digits and exponent - something like:
          {digits_dot_maybe_digits}{exponent}?|   # "1.0" "1." "1.0e3", "1.e3"
          {dot_digits}{exponent}?|                # ".1" ".1e3"
          {digits}{exponent}|                     # "1e3"
          {digits}(?=j)                           # "300j"
        )
      )
      j?                         # Optional j for cplx numbers, not captured.
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
  """Customizes how `want` and `got` are compared, see `check_output`."""

  def __init__(self, *args, **kwargs):
    super(TfDoctestOutputChecker, self).__init__(*args, **kwargs)
    self.extract_floats = _FloatExtractor()
    self.text_good = None
    self.float_size_good = None

  _ADDRESS_RE = re.compile(r'\bat 0x[0-9a-f]*?>')
  # TODO(yashkatariya): Add other tensor's string substitutions too.
  # tf.RaggedTensor doesn't need one.
  _NUMPY_OUTPUT_RE = re.compile(r'<tf.Tensor.*?numpy=(.*?)>', re.DOTALL)

  def _allclose(self, want, got, rtol=1e-3, atol=1e-3):
    return np.allclose(want, got, rtol=rtol, atol=atol)

  def _tf_tensor_numpy_output(self, string):
    modified_string = self._NUMPY_OUTPUT_RE.sub(r'\1', string)
    return modified_string, modified_string != string

  MESSAGE = textwrap.dedent("""\n
        #############################################################
        Check the documentation (https://www.tensorflow.org/community/contribute/docs_ref) on how to
        write testable docstrings.
        #############################################################""")

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

    # If the docstring's output is empty and there is some output generated
    # after running the snippet, return True. This is because if the user
    # doesn't want to display output, respect that over what the doctest wants.
    if got and not want:
      return True

    if want is None:
      want = ''

    # Replace python's addresses with ellipsis (`...`) since it can change on
    # each execution.
    want = self._ADDRESS_RE.sub('at ...>', want)

    # Replace tf.Tensor strings with only their numpy field values.
    want, want_changed = self._tf_tensor_numpy_output(want)
    if want_changed:
      got, _ = self._tf_tensor_numpy_output(got)

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

    got.append(self.MESSAGE)
    got = '\n'.join(got)
    return (super(TfDoctestOutputChecker,
                  self).output_difference(example, got, optionflags))
