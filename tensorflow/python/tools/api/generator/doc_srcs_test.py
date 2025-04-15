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
# =============================================================================
"""Tests for tensorflow.python.tools.api.generator.doc_srcs."""

import argparse
import importlib
import sys

from tensorflow.python.platform import test
from tensorflow.python.tools.api.generator import doc_srcs


FLAGS = None


class DocSrcsTest(test.TestCase):

  def testModulesAreValidAPIModules(self):
    for module_name in doc_srcs.get_doc_sources(FLAGS.api_name):
      # Convert module_name to corresponding __init__.py file path.
      file_path = module_name.replace('.', '/')
      if file_path:
        file_path += '/'
      file_path += '__init__.py'

      self.assertIn(
          file_path, FLAGS.outputs,
          msg='%s is not a valid API module' % module_name)

  def testHaveDocstringOrDocstringModule(self):
    for module_name, docsrc in doc_srcs.get_doc_sources(FLAGS.api_name).items():
      self.assertFalse(
          docsrc.docstring and docsrc.docstring_module_name,
          msg=('%s contains DocSource has both a docstring and a '
               'docstring_module_name. Only one of "docstring" or '
               '"docstring_module_name" should be set.') % (module_name))

  def testDocstringModulesAreValidModules(self):
    for _, docsrc in doc_srcs.get_doc_sources(FLAGS.api_name).items():
      if docsrc.docstring_module_name:
        doc_module_name = '.'.join([
            FLAGS.package, docsrc.docstring_module_name])
        self.assertIn(
            doc_module_name, sys.modules,
            msg=('docsources_module %s is not a valid module under %s.' %
                 (docsrc.docstring_module_name, FLAGS.package)))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      'outputs', metavar='O', type=str, nargs='+',
      help='create_python_api output files.')
  parser.add_argument(
      '--package', type=str,
      help='Base package that imports modules containing the target tf_export '
           'decorators.')
  parser.add_argument(
      '--api_name', type=str,
      help='API name: tensorflow')
  FLAGS, unparsed = parser.parse_known_args()

  importlib.import_module(FLAGS.package)

  # Now update argv, so that unittest library does not get confused.
  sys.argv = [sys.argv[0]] + unparsed
  test.main()
