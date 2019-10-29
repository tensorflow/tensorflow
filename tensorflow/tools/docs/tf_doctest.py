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
import numpy as np

from absl import flags
from absl.testing import absltest

import tensorflow.compat.v2 as tf
tf.compat.v1.enable_v2_behavior()

# We put doctest after absltest so that it picks up the unittest monkeypatch.
# Otherwise doctest tests aren't runnable at all.
import doctest  # pylint: disable=g-import-not-at-top, g-bad-import-order

FLAGS = flags.FLAGS

flags.DEFINE_string('module', None, 'A specific module to run doctest on.')
flags.DEFINE_boolean('list', None,
                     'List all the modules in the core package imported.')
flags.DEFINE_string('file', None, 'A specific file to run doctest on.')

flags.mark_flags_as_mutual_exclusive(['module', 'file'])
flags.mark_flags_as_mutual_exclusive(['list', 'file'])

PACKAGE = 'tensorflow.python.'


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


class CustomOutputChecker(doctest.OutputChecker):
  """Changes the `want` and `got` strings.

  This allows it to be customized before they are compared.
  """

  ADDRESS_RE = re.compile(r'\bat 0x[0-9a-f]*?>')

  def check_output(self, want, got, optionflags):
    # Replace python's addresses with ellipsis (`...`) since it can change on
    # each execution.
    want = self.ADDRESS_RE.sub('at ...>', want)
    return doctest.OutputChecker.check_output(self, want, got, optionflags)

  _MESSAGE = textwrap.dedent("""\n
        #############################################################
        Check the documentation
        (https://www.tensorflow.org/community/contribute/docs_ref) on how to write testable docstrings.
        #############################################################""")

  def output_difference(self, example, got, optionflags):
    got = got + self._MESSAGE
    return doctest.OutputChecker.output_difference(self, example, got,
                                                   optionflags)


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
            checker=CustomOutputChecker(),
            optionflags=(doctest.ELLIPSIS |
                         doctest.NORMALIZE_WHITESPACE |
                         doctest.IGNORE_EXCEPTION_DETAIL |
                         doctest.DONT_ACCEPT_BLANKLINE),
        ))
  return tests


if __name__ == '__main__':
  absltest.main()
