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
import sys

from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_export


class ValidateExportTest(test.TestCase):
  """Tests for tf_export class."""

  class MockModule(object):

    def __init__(self, name):
      self.__name__ = name

  def setUp(self):
    super().setUp()
    self._modules = []

    def _test_function(unused_arg=0):
      pass

    def _test_function2(unused_arg=0):
      pass

    class TestClassA(object):
      pass

    class TestClassB(TestClassA):
      pass

    self._test_function = _test_function
    self._test_function2 = _test_function2
    self._test_class_a = TestClassA
    self._test_class_b = TestClassB

  def tearDown(self):
    super().tearDown()
    for name in self._modules:
      del sys.modules[name]
    self._modules = []

  def _CreateMockModule(self, name):
    mock_module = self.MockModule(name)
    sys.modules[name] = mock_module
    self._modules.append(name)
    return mock_module

  def testExportSingleFunction(self):
    export_decorator = tf_export.tf_export('nameA', 'nameB')
    decorated_function = export_decorator(self._test_function)
    self.assertEqual(decorated_function, self._test_function)
    self.assertEqual(('nameA', 'nameB'), decorated_function._tf_api_names)
    self.assertEqual(['nameA', 'nameB'],
                     tf_export.get_v1_names(decorated_function))
    self.assertEqual(['nameA', 'nameB'],
                     tf_export.get_v2_names(decorated_function))
    self.assertEqual(
        tf_export.get_symbol_from_name('nameA'), decorated_function)
    self.assertEqual(
        tf_export.get_symbol_from_name('nameB'), decorated_function)
    self.assertEqual(
        tf_export.get_symbol_from_name(
            tf_export.get_canonical_name_for_symbol(decorated_function)),
        decorated_function)

  def testExportSingleFunctionV1Only(self):
    export_decorator = tf_export.tf_export(v1=['nameA', 'nameB'])
    decorated_function = export_decorator(self._test_function)
    self.assertEqual(decorated_function, self._test_function)
    self.assertAllEqual(('nameA', 'nameB'), decorated_function._tf_api_names_v1)
    self.assertAllEqual(['nameA', 'nameB'],
                        tf_export.get_v1_names(decorated_function))
    self.assertEqual([], tf_export.get_v2_names(decorated_function))
    self.assertEqual(
        tf_export.get_symbol_from_name('compat.v1.nameA'), decorated_function)
    self.assertEqual(
        tf_export.get_symbol_from_name('compat.v1.nameB'), decorated_function)
    self.assertEqual(
        tf_export.get_symbol_from_name(
            tf_export.get_canonical_name_for_symbol(
                decorated_function, add_prefix_to_v1_names=True)),
        decorated_function)

  def testExportMultipleFunctions(self):
    export_decorator1 = tf_export.tf_export('nameA', 'nameB')
    export_decorator2 = tf_export.tf_export('nameC', 'nameD')
    decorated_function1 = export_decorator1(self._test_function)
    decorated_function2 = export_decorator2(self._test_function2)
    self.assertEqual(decorated_function1, self._test_function)
    self.assertEqual(decorated_function2, self._test_function2)
    self.assertEqual(('nameA', 'nameB'), decorated_function1._tf_api_names)
    self.assertEqual(('nameC', 'nameD'), decorated_function2._tf_api_names)
    self.assertEqual(
        tf_export.get_symbol_from_name('nameB'), decorated_function1)
    self.assertEqual(
        tf_export.get_symbol_from_name('nameD'), decorated_function2)
    self.assertEqual(
        tf_export.get_symbol_from_name(
            tf_export.get_canonical_name_for_symbol(decorated_function1)),
        decorated_function1)
    self.assertEqual(
        tf_export.get_symbol_from_name(
            tf_export.get_canonical_name_for_symbol(decorated_function2)),
        decorated_function2)

  def testExportClasses(self):
    export_decorator_a = tf_export.tf_export('TestClassA1')
    export_decorator_a(self._test_class_a)
    self.assertEqual(('TestClassA1',), self._test_class_a._tf_api_names)
    self.assertNotIn('_tf_api_names', self._test_class_b.__dict__)

    export_decorator_b = tf_export.tf_export('TestClassB1')
    export_decorator_b(self._test_class_b)
    self.assertEqual(('TestClassA1',), self._test_class_a._tf_api_names)
    self.assertEqual(('TestClassB1',), self._test_class_b._tf_api_names)
    self.assertEqual(
        ['TestClassA1'], tf_export.get_v1_names(self._test_class_a)
    )
    self.assertEqual(
        ['TestClassB1'], tf_export.get_v1_names(self._test_class_b)
    )

  def testExportSingleConstant(self):
    module1 = self._CreateMockModule('module1')

    export_decorator = tf_export.tf_export('NAME_A', 'NAME_B')
    export_decorator.export_constant('module1', 'test_constant')
    self.assertEqual([(('NAME_A', 'NAME_B'), 'test_constant')],
                     module1._tf_api_constants)
    self.assertEqual([(('NAME_A', 'NAME_B'), 'test_constant')],
                     tf_export.get_v1_constants(module1))
    self.assertEqual([(('NAME_A', 'NAME_B'), 'test_constant')],
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
    self.assertEqual([(('NAME_A', 'NAME_B'), 123)], module1._tf_api_constants)
    self.assertEqual([(('NAME_C', 'NAME_D'), 'abc'),
                      (('NAME_E', 'NAME_F'), 0.5)], module2._tf_api_constants)

  def testMultipleDecorators(self):

    def get_wrapper(func):

      def wrapper(*unused_args, **unused_kwargs):
        pass

      return tf_decorator.make_decorator(func, wrapper)

    decorated_function = get_wrapper(self._test_function)

    export_decorator = tf_export.tf_export('nameA', 'nameB')
    exported_function = export_decorator(decorated_function)
    self.assertEqual(decorated_function, exported_function)
    self.assertEqual(('nameA', 'nameB'), self._test_function._tf_api_names)


if __name__ == '__main__':
  test.main()
