# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for create_python_api."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import imp
import sys

from tensorflow.python.platform import test
from tensorflow.python.tools.api.generator import create_python_api
from tensorflow.python.util.tf_export import tf_export


@tf_export('test_op', 'test_op1', 'test.test_op2')
def test_op():
  pass


@tf_export('test1.foo', v1=['test.foo'])
def deprecated_test_op():
  pass


@tf_export('TestClass', 'NewTestClass')
class TestClass(object):
  pass


_TEST_CONSTANT = 5
_MODULE_NAME = 'tensorflow.python.test_module'


class CreatePythonApiTest(test.TestCase):

  def setUp(self):
    # Add fake op to a module that has 'tensorflow' in the name.
    sys.modules[_MODULE_NAME] = imp.new_module(_MODULE_NAME)
    setattr(sys.modules[_MODULE_NAME], 'test_op', test_op)
    setattr(sys.modules[_MODULE_NAME], 'deprecated_test_op', deprecated_test_op)
    setattr(sys.modules[_MODULE_NAME], 'TestClass', TestClass)
    test_op.__module__ = _MODULE_NAME
    TestClass.__module__ = _MODULE_NAME
    tf_export('consts._TEST_CONSTANT').export_constant(
        _MODULE_NAME, '_TEST_CONSTANT')

  def tearDown(self):
    del sys.modules[_MODULE_NAME]

  def testFunctionImportIsAdded(self):
    imports, _ = create_python_api.get_api_init_text(
        packages=[create_python_api._DEFAULT_PACKAGE],
        output_package='tensorflow',
        api_name='tensorflow',
        api_version=1)
    if create_python_api._LAZY_LOADING:
      expected_import = (
          '\'test_op1\': '
          '(\'tensorflow.python.test_module\','
          ' \'test_op\')')
    else:
      expected_import = (
          'from tensorflow.python.test_module '
          'import test_op as test_op1')
    self.assertTrue(
        expected_import in str(imports),
        msg='%s not in %s' % (expected_import, str(imports)))

    if create_python_api._LAZY_LOADING:
      expected_import = (
          '\'test_op\': '
          '(\'tensorflow.python.test_module\','
          ' \'test_op\')')
    else:
      expected_import = (
          'from tensorflow.python.test_module '
          'import test_op')
    self.assertTrue(
        expected_import in str(imports),
        msg='%s not in %s' % (expected_import, str(imports)))
    # Also check that compat.v1 is not added to imports.
    self.assertFalse('compat.v1' in imports,
                     msg='compat.v1 in %s' % str(imports.keys()))

  def testClassImportIsAdded(self):
    imports, _ = create_python_api.get_api_init_text(
        packages=[create_python_api._DEFAULT_PACKAGE],
        output_package='tensorflow',
        api_name='tensorflow',
        api_version=2)
    if create_python_api._LAZY_LOADING:
      expected_import = (
          '\'NewTestClass\':'
          ' (\'tensorflow.python.test_module\','
          ' \'TestClass\')')
    else:
      expected_import = (
          'from tensorflow.python.test_module '
          'import TestClass')
    self.assertTrue(
        'TestClass' in str(imports),
        msg='%s not in %s' % (expected_import, str(imports)))

  def testConstantIsAdded(self):
    imports, _ = create_python_api.get_api_init_text(
        packages=[create_python_api._DEFAULT_PACKAGE],
        output_package='tensorflow',
        api_name='tensorflow',
        api_version=1)
    if create_python_api._LAZY_LOADING:
      expected = ('\'_TEST_CONSTANT\':'
                  ' (\'tensorflow.python.test_module\','
                  ' \'_TEST_CONSTANT\')')
    else:
      expected = ('from tensorflow.python.test_module '
                  'import _TEST_CONSTANT')
    self.assertTrue(expected in str(imports),
                    msg='%s not in %s' % (expected, str(imports)))

  def testCompatModuleIsAdded(self):
    imports, _ = create_python_api.get_api_init_text(
        packages=[create_python_api._DEFAULT_PACKAGE],
        output_package='tensorflow',
        api_name='tensorflow',
        api_version=2,
        compat_api_versions=[1])
    self.assertTrue('compat.v1' in imports,
                    msg='compat.v1 not in %s' % str(imports.keys()))
    self.assertTrue('compat.v1.test' in imports,
                    msg='compat.v1.test not in %s' % str(imports.keys()))

  def testNestedCompatModulesAreAdded(self):
    imports, _ = create_python_api.get_api_init_text(
        packages=[create_python_api._DEFAULT_PACKAGE],
        output_package='tensorflow',
        api_name='tensorflow',
        api_version=2,
        compat_api_versions=[1, 2])
    self.assertIn('compat.v1.compat.v1', imports,
                  msg='compat.v1.compat.v1 not in %s' % str(imports.keys()))
    self.assertIn('compat.v1.compat.v2', imports,
                  msg='compat.v1.compat.v2 not in %s' % str(imports.keys()))
    self.assertIn('compat.v2.compat.v1', imports,
                  msg='compat.v2.compat.v1 not in %s' % str(imports.keys()))
    self.assertIn('compat.v2.compat.v2', imports,
                  msg='compat.v2.compat.v2 not in %s' % str(imports.keys()))


if __name__ == '__main__':
  test.main()
