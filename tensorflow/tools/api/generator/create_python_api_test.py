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
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.api.generator import create_python_api


@tf_export('test_op', 'test_op1')
def test_op():
  pass


@tf_export('TestClass', 'NewTestClass')
class TestClass(object):
  pass


_TEST_CONSTANT = 5
_MODULE_NAME = 'test.tensorflow.test_module'


class CreatePythonApiTest(test.TestCase):

  def setUp(self):
    # Add fake op to a module that has 'tensorflow' in the name.
    sys.modules[_MODULE_NAME] = imp.new_module(_MODULE_NAME)
    setattr(sys.modules[_MODULE_NAME], 'test_op', test_op)
    setattr(sys.modules[_MODULE_NAME], 'TestClass', TestClass)
    test_op.__module__ = _MODULE_NAME
    TestClass.__module__ = _MODULE_NAME
    tf_export('consts._TEST_CONSTANT').export_constant(
        _MODULE_NAME, '_TEST_CONSTANT')

  def tearDown(self):
    del sys.modules[_MODULE_NAME]

  def testFunctionImportIsAdded(self):
    imports = create_python_api.get_api_imports()
    expected_import = (
        'from test.tensorflow.test_module import test_op as test_op1')
    self.assertTrue(
        expected_import in str(imports),
        msg='%s not in %s' % (expected_import, str(imports)))

    expected_import = 'from test.tensorflow.test_module import test_op'
    self.assertTrue(
        expected_import in str(imports),
        msg='%s not in %s' % (expected_import, str(imports)))

  def testClassImportIsAdded(self):
    imports = create_python_api.get_api_imports()
    expected_import = 'from test.tensorflow.test_module import TestClass'
    self.assertTrue(
        'TestClass' in str(imports),
        msg='%s not in %s' % (expected_import, str(imports)))

  def testConstantIsAdded(self):
    imports = create_python_api.get_api_imports()
    expected = 'from test.tensorflow.test_module import _TEST_CONSTANT'
    self.assertTrue(expected in str(imports),
                    msg='%s not in %s' % (expected, str(imports)))


if __name__ == '__main__':
  test.main()
