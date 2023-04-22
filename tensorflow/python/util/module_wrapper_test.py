# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.python.util.module_wrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import types

from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import module_wrapper
from tensorflow.python.util import tf_inspect
from tensorflow.tools.compatibility import all_renames_v2

module_wrapper._PER_MODULE_WARNING_LIMIT = 5


class MockModule(types.ModuleType):
  pass


class DeprecationWrapperTest(test.TestCase):

  def testWrapperIsAModule(self):
    module = MockModule('test')
    wrapped_module = module_wrapper.TFModuleWrapper(module, 'test')
    self.assertTrue(tf_inspect.ismodule(wrapped_module))

  @test.mock.patch.object(logging, 'warning', autospec=True)
  def testDeprecationWarnings(self, mock_warning):
    module = MockModule('test')
    module.foo = 1
    module.bar = 2
    module.baz = 3
    all_renames_v2.symbol_renames['tf.test.bar'] = 'tf.bar2'
    all_renames_v2.symbol_renames['tf.test.baz'] = 'tf.compat.v1.baz'

    wrapped_module = module_wrapper.TFModuleWrapper(module, 'test')
    self.assertTrue(tf_inspect.ismodule(wrapped_module))

    self.assertEqual(0, mock_warning.call_count)
    bar = wrapped_module.bar
    self.assertEqual(1, mock_warning.call_count)
    foo = wrapped_module.foo
    self.assertEqual(1, mock_warning.call_count)
    baz = wrapped_module.baz  # pylint: disable=unused-variable
    self.assertEqual(2, mock_warning.call_count)
    baz = wrapped_module.baz
    self.assertEqual(2, mock_warning.call_count)

    # Check that values stayed the same
    self.assertEqual(module.foo, foo)
    self.assertEqual(module.bar, bar)


class LazyLoadingWrapperTest(test.TestCase):

  def testLazyLoad(self):
    module = MockModule('test')
    apis = {'cmd': ('', 'cmd'), 'ABCMeta': ('abc', 'ABCMeta')}
    wrapped_module = module_wrapper.TFModuleWrapper(
        module, 'test', public_apis=apis, deprecation=False)
    import cmd as _cmd  # pylint: disable=g-import-not-at-top
    from abc import ABCMeta as _ABCMeta  # pylint: disable=g-import-not-at-top, g-importing-member
    self.assertFalse(wrapped_module._fastdict_key_in('cmd'))
    self.assertEqual(wrapped_module.cmd, _cmd)
    # Verify that the APIs are added to the cache of FastModuleType object
    self.assertTrue(wrapped_module._fastdict_key_in('cmd'))
    self.assertFalse(wrapped_module._fastdict_key_in('ABCMeta'))
    self.assertEqual(wrapped_module.ABCMeta, _ABCMeta)
    self.assertTrue(wrapped_module._fastdict_key_in('ABCMeta'))

  def testLazyLoadLocalOverride(self):
    # Test that we can override and add fields to the wrapped module.
    module = MockModule('test')
    apis = {'cmd': ('', 'cmd')}
    wrapped_module = module_wrapper.TFModuleWrapper(
        module, 'test', public_apis=apis, deprecation=False)
    import cmd as _cmd  # pylint: disable=g-import-not-at-top
    self.assertEqual(wrapped_module.cmd, _cmd)
    setattr(wrapped_module, 'cmd', 1)
    setattr(wrapped_module, 'cgi', 2)
    self.assertEqual(wrapped_module.cmd, 1)  # override
    # Verify that the values are also updated in the cache
    # of the FastModuleType object
    self.assertEqual(wrapped_module._fastdict_get('cmd'), 1)
    self.assertEqual(wrapped_module.cgi, 2)  # add
    self.assertEqual(wrapped_module._fastdict_get('cgi'), 2)

  def testLazyLoadDict(self):
    # Test that we can override and add fields to the wrapped module.
    module = MockModule('test')
    apis = {'cmd': ('', 'cmd')}
    wrapped_module = module_wrapper.TFModuleWrapper(
        module, 'test', public_apis=apis, deprecation=False)
    import cmd as _cmd  # pylint: disable=g-import-not-at-top
    # At first cmd key does not exist in __dict__
    self.assertNotIn('cmd', wrapped_module.__dict__)
    # After it is referred (lazyloaded), it gets added to __dict__
    wrapped_module.cmd  # pylint: disable=pointless-statement
    self.assertEqual(wrapped_module.__dict__['cmd'], _cmd)
    # When we call setattr, it also gets added to __dict__
    setattr(wrapped_module, 'cmd2', _cmd)
    self.assertEqual(wrapped_module.__dict__['cmd2'], _cmd)

  def testLazyLoadWildcardImport(self):
    # Test that public APIs are in __all__.
    module = MockModule('test')
    module._should_not_be_public = 5
    apis = {'cmd': ('', 'cmd')}
    wrapped_module = module_wrapper.TFModuleWrapper(
        module, 'test', public_apis=apis, deprecation=False)
    setattr(wrapped_module, 'hello', 1)
    self.assertIn('hello', wrapped_module.__all__)
    self.assertIn('cmd', wrapped_module.__all__)
    self.assertNotIn('_should_not_be_public', wrapped_module.__all__)

  def testLazyLoadCorrectLiteModule(self):
    # If set, always load lite module from public API list.
    module = MockModule('test')
    apis = {'lite': ('', 'cmd')}
    module.lite = 5
    import cmd as _cmd  # pylint: disable=g-import-not-at-top
    wrapped_module = module_wrapper.TFModuleWrapper(
        module, 'test', public_apis=apis, deprecation=False, has_lite=True)
    self.assertEqual(wrapped_module.lite, _cmd)

  def testInitCachesAttributes(self):
    module = MockModule('test')
    wrapped_module = module_wrapper.TFModuleWrapper(module, 'test')
    self.assertTrue(wrapped_module._fastdict_key_in('_fastdict_key_in'))
    self.assertTrue(wrapped_module._fastdict_key_in('_tfmw_module_name'))
    self.assertTrue(wrapped_module._fastdict_key_in('__all__'))

  def testCompatV1APIInstrumenting(self):
    self.assertFalse(module_wrapper.TFModuleWrapper.compat_v1_usage_recorded)
    apis = {'cosh': ('', 'cmd')}

    mock_tf = MockModule('tensorflow')
    mock_tf_wrapped = module_wrapper.TFModuleWrapper(
        mock_tf, 'test', public_apis=apis)
    mock_tf_wrapped.cosh  # pylint: disable=pointless-statement
    self.assertFalse(module_wrapper.TFModuleWrapper.compat_v1_usage_recorded)

    mock_tf_v1 = MockModule('tensorflow.compat.v1')
    mock_tf_v1_wrapped = module_wrapper.TFModuleWrapper(
        mock_tf_v1, 'test', public_apis=apis)
    self.assertFalse(module_wrapper.TFModuleWrapper.compat_v1_usage_recorded)
    mock_tf_v1_wrapped.cosh  # pylint: disable=pointless-statement
    self.assertTrue(module_wrapper.TFModuleWrapper.compat_v1_usage_recorded)

    # 'Reset' the status before testing against 'tensorflow.compat.v2.compat.v1'
    module_wrapper.TFModuleWrapper.compat_v1_usage_recorded = False
    mock_tf_v2_v1 = mock_tf_v1 = MockModule('tensorflow.compat.v2.compat.v1')
    mock_tf_v2_v1_wrapped = module_wrapper.TFModuleWrapper(
        mock_tf_v2_v1, 'test', public_apis=apis)
    self.assertFalse(module_wrapper.TFModuleWrapper.compat_v1_usage_recorded)
    mock_tf_v2_v1_wrapped.cosh  # pylint: disable=pointless-statement
    self.assertTrue(module_wrapper.TFModuleWrapper.compat_v1_usage_recorded)


class PickleTest(test.TestCase):

  def testPickleSubmodule(self):
    name = PickleTest.__module__  # The current module is a submodule.
    module = module_wrapper.TFModuleWrapper(MockModule(name), name)
    restored = pickle.loads(pickle.dumps(module))
    self.assertEqual(restored.__name__, name)
    self.assertIsNotNone(restored.PickleTest)


if __name__ == '__main__':
  test.main()
