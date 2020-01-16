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
# ==============================================================================
"""tf_export tests."""

# pylint: disable=unused-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from tensorflow.python.platform import test
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_export


def _test_function(unused_arg=0):
  pass


def _test_function2(unused_arg=0):
  pass


class TestClassA(object):
  pass


class TestClassB(TestClassA):
  pass


class ValidateExportTest(test.TestCase):
  """Tests for tf_export class."""

  class MockModule(object):

    def __init__(self, name):
      self.__name__ = name

  def setUp(self):
    self._modules = []

  def tearDown(self):
    for name in self._modules:
      del sys.modules[name]
    self._modules = []
    for symbol in [_test_function, _test_function, TestClassA, TestClassB]:
      if hasattr(symbol, '_tf_api_names'):
        del symbol._tf_api_names
      if hasattr(symbol, '_tf_api_names_v1'):
        del symbol._tf_api_names_v1
      if hasattr(symbol, '_estimator_api_names'):
        del symbol._estimator_api_names
      if hasattr(symbol, '_estimator_api_names_v1'):
        del symbol._estimator_api_names_v1

  def _CreateMockModule(self, name):
    mock_module = self.MockModule(name)
    sys.modules[name] = mock_module
    self._modules.append(name)
    return mock_module

  def testExportSingleFunction(self):
    export_decorator = tf_export.tf_export('nameA', 'nameB')
    decorated_function = export_decorator(_test_function)
    self.assertEquals(decorated_function, _test_function)
    self.assertEquals(('nameA', 'nameB'), decorated_function._tf_api_names)
    self.assertEquals(['nameA', 'nameB'],
                      tf_export.get_v1_names(decorated_function))
    self.assertEquals(['nameA', 'nameB'],
                      tf_export.get_v2_names(decorated_function))

  def testExportMultipleFunctions(self):
    export_decorator1 = tf_export.tf_export('nameA', 'nameB')
    export_decorator2 = tf_export.tf_export('nameC', 'nameD')
    decorated_function1 = export_decorator1(_test_function)
    decorated_function2 = export_decorator2(_test_function2)
    self.assertEquals(decorated_function1, _test_function)
    self.assertEquals(decorated_function2, _test_function2)
    self.assertEquals(('nameA', 'nameB'), decorated_function1._tf_api_names)
    self.assertEquals(('nameC', 'nameD'), decorated_function2._tf_api_names)

  def testExportClasses(self):
    export_decorator_a = tf_export.tf_export('TestClassA1')
    export_decorator_a(TestClassA)
    self.assertEquals(('TestClassA1',), TestClassA._tf_api_names)
    self.assertTrue('_tf_api_names' not in TestClassB.__dict__)

    export_decorator_b = tf_export.tf_export('TestClassB1')
    export_decorator_b(TestClassB)
    self.assertEquals(('TestClassA1',), TestClassA._tf_api_names)
    self.assertEquals(('TestClassB1',), TestClassB._tf_api_names)
    self.assertEquals(['TestClassA1'], tf_export.get_v1_names(TestClassA))
    self.assertEquals(['TestClassB1'], tf_export.get_v1_names(TestClassB))

  def testExportClassInEstimator(self):
    export_decorator_a = tf_export.tf_export('TestClassA1')
    export_decorator_a(TestClassA)
    self.assertEquals(('TestClassA1',), TestClassA._tf_api_names)

    export_decorator_b = tf_export.estimator_export(
        'estimator.TestClassB1')
    export_decorator_b(TestClassB)
    self.assertTrue('_tf_api_names' not in TestClassB.__dict__)
    self.assertEquals(('TestClassA1',), TestClassA._tf_api_names)
    self.assertEquals(['TestClassA1'], tf_export.get_v1_names(TestClassA))
    self.assertEquals(['estimator.TestClassB1'],
                      tf_export.get_v1_names(TestClassB))

  def testExportSingleConstant(self):
    module1 = self._CreateMockModule('module1')

    export_decorator = tf_export.tf_export('NAME_A', 'NAME_B')
    export_decorator.export_constant('module1', 'test_constant')
    self.assertEquals([(('NAME_A', 'NAME_B'), 'test_constant')],
                      module1._tf_api_constants)
    self.assertEquals([(('NAME_A', 'NAME_B'), 'test_constant')],
                      tf_export.get_v1_constants(module1))
    self.assertEquals([(('NAME_A', 'NAME_B'), 'test_constant')],
                      tf_export.get_v2_constants(module1))

  def testExportMultipleConstants(self):
    module1 = self._CreateMockModule('module1')
    module2 = self._CreateMockModule('module2')

    test_constant1 = 123
    test_constant2 = 'abc'
    test_constant3 = 0.5

    export_decorator1 = tf_export.tf_export('NAME_A', 'NAME_B')
    export_decorator2 = tf_export.tf_export('NAME_C', 'NAME_D')
    export_decorator3 = tf_export.tf_export('NAME_E', 'NAME_F')
    export_decorator1.export_constant('module1', test_constant1)
    export_decorator2.export_constant('module2', test_constant2)
    export_decorator3.export_constant('module2', test_constant3)
    self.assertEquals([(('NAME_A', 'NAME_B'), 123)],
                      module1._tf_api_constants)
    self.assertEquals([(('NAME_C', 'NAME_D'), 'abc'),
                       (('NAME_E', 'NAME_F'), 0.5)],
                      module2._tf_api_constants)

  def testRaisesExceptionIfAlreadyHasAPINames(self):
    _test_function._tf_api_names = ['abc']
    export_decorator = tf_export.tf_export('nameA', 'nameB')
    with self.assertRaises(tf_export.SymbolAlreadyExposedError):
      export_decorator(_test_function)

  def testRaisesExceptionIfInvalidSymbolName(self):
    # TensorFlow code is not allowed to export symbols under package
    # tf.estimator
    with self.assertRaises(tf_export.InvalidSymbolNameError):
      tf_export.tf_export('estimator.invalid')

    # All symbols exported by Estimator must be under tf.estimator package.
    with self.assertRaises(tf_export.InvalidSymbolNameError):
      tf_export.estimator_export('invalid')
    with self.assertRaises(tf_export.InvalidSymbolNameError):
      tf_export.estimator_export('Estimator.invalid')
    with self.assertRaises(tf_export.InvalidSymbolNameError):
      tf_export.estimator_export('invalid.estimator')

  def testRaisesExceptionIfInvalidV1SymbolName(self):
    with self.assertRaises(tf_export.InvalidSymbolNameError):
      tf_export.tf_export('valid', v1=['estimator.invalid'])
    with self.assertRaises(tf_export.InvalidSymbolNameError):
      tf_export.estimator_export('estimator.valid', v1=['invalid'])

  def testOverridesFunction(self):
    _test_function2._tf_api_names = ['abc']

    export_decorator = tf_export.tf_export(
        'nameA', 'nameB', overrides=[_test_function2])
    export_decorator(_test_function)

    # _test_function overrides _test_function2. So, _tf_api_names
    # should be removed from _test_function2.
    self.assertFalse(hasattr(_test_function2, '_tf_api_names'))

  def testMultipleDecorators(self):
    def get_wrapper(func):
      def wrapper(*unused_args, **unused_kwargs):
        pass
      return tf_decorator.make_decorator(func, wrapper)
    decorated_function = get_wrapper(_test_function)

    export_decorator = tf_export.tf_export('nameA', 'nameB')
    exported_function = export_decorator(decorated_function)
    self.assertEquals(decorated_function, exported_function)
    self.assertEquals(('nameA', 'nameB'), _test_function._tf_api_names)


if __name__ == '__main__':
  test.main()
