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
"""Tests for conversion module."""

import imp
import sys
import types
import weakref

import six

from tensorflow.python.autograph import utils
from tensorflow.python.autograph.core import config
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.impl import api
from tensorflow.python.autograph.impl import conversion
from tensorflow.python.autograph.impl.testing import pybind_for_testing
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test


class ConversionTest(test.TestCase):

  def _simple_program_ctx(self):
    return converter.ProgramContext(
        options=converter.ConversionOptions(recursive=True),
        autograph_module=api)

  def test_is_allowlisted(self):

    def test_fn():
      return constant_op.constant(1)

    self.assertFalse(conversion.is_allowlisted(test_fn))
    self.assertTrue(conversion.is_allowlisted(utils))
    self.assertTrue(conversion.is_allowlisted(constant_op.constant))

  def test_is_allowlisted_tensorflow_like(self):

    tf_like = imp.new_module('tensorflow_foo')
    def test_fn():
      pass
    tf_like.test_fn = test_fn
    test_fn.__module__ = tf_like

    self.assertFalse(conversion.is_allowlisted(tf_like.test_fn))

  def test_is_allowlisted_callable_allowlisted_call(self):

    allowlisted_mod = imp.new_module('test_allowlisted_call')
    sys.modules['test_allowlisted_call'] = allowlisted_mod
    config.CONVERSION_RULES = ((config.DoNotConvert('test_allowlisted_call'),) +
                               config.CONVERSION_RULES)

    class TestClass(object):

      def __call__(self):
        pass

      def allowlisted_method(self):
        pass

    TestClass.__module__ = 'test_allowlisted_call'
    if six.PY2:
      TestClass.__call__.__func__.__module__ = 'test_allowlisted_call'
    else:
      TestClass.__call__.__module__ = 'test_allowlisted_call'

    class Subclass(TestClass):

      def converted_method(self):
        pass

    tc = Subclass()

    self.assertTrue(conversion.is_allowlisted(TestClass.__call__))
    self.assertTrue(conversion.is_allowlisted(tc))
    self.assertTrue(conversion.is_allowlisted(tc.__call__))
    self.assertTrue(conversion.is_allowlisted(tc.allowlisted_method))
    self.assertFalse(conversion.is_allowlisted(Subclass))
    self.assertFalse(conversion.is_allowlisted(tc.converted_method))

  def test_is_allowlisted_tfmethodwrapper(self):

    class TestClass(object):

      def member_function(self):
        pass

    TestClass.__module__ = 'test_allowlisted_call'
    test_obj = TestClass()

    def test_fn(self):
      del self

    bound_method = types.MethodType(
        test_fn,
        function.TfMethodTarget(
            weakref.ref(test_obj), test_obj.member_function))

    self.assertTrue(conversion.is_allowlisted(bound_method))

  def test_is_allowlisted_pybind(self):
    test_object = pybind_for_testing.TestClassDef()
    with test.mock.patch.object(config, 'CONVERSION_RULES', ()):
      # TODO(mdan): This should return True for functions and methods.
      # Note: currently, native bindings are allowlisted by a separate check.
      self.assertFalse(conversion.is_allowlisted(test_object.method))


if __name__ == '__main__':
  test.main()
