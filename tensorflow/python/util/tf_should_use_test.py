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
"""Unit tests for tf_should_use."""

# pylint: disable=unused-import
import contextlib
import gc
import sys

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import tf_should_use


@contextlib.contextmanager
def reroute_error():
  """Temporarily reroute errors written to tf_logging.error into `captured`."""
  with test.mock.patch.object(tf_should_use.tf_logging, 'error') as error:
    yield error


class TfShouldUseTest(test.TestCase):

  def testAddShouldUseWarningWhenNotUsed(self):
    c = constant_op.constant(0, name='blah0')
    def in_this_function():
      h = tf_should_use._add_should_use_warning(c, warn_in_eager=True)
      del h
    with reroute_error() as error:
      in_this_function()
    msg = '\n'.join(error.call_args[0])
    self.assertIn('Object was never used', msg)
    if not context.executing_eagerly():
      self.assertIn('blah0:0', msg)
    self.assertIn('in_this_function', msg)
    self.assertFalse(gc.garbage)

  def testAddShouldUseExceptionInEagerAndFunction(self):
    def in_this_function():
      c = constant_op.constant(0, name='blah0')
      h = tf_should_use._add_should_use_warning(
          c, warn_in_eager=True, error_in_function=True)
      del h
    if context.executing_eagerly():
      with reroute_error() as error:
        in_this_function()
      msg = '\n'.join(error.call_args[0])
      self.assertIn('Object was never used', msg)
      self.assertIn('in_this_function', msg)
      self.assertFalse(gc.garbage)

    tf_fn_in_this_function = def_function.function(in_this_function)
    with self.assertRaisesRegex(RuntimeError,
                                r'Object was never used.*blah0:0'):
      tf_fn_in_this_function()
    self.assertFalse(gc.garbage)

  def _testAddShouldUseWarningWhenUsed(self, fn, name):
    c = constant_op.constant(0, name=name)
    with reroute_error() as error:
      h = tf_should_use._add_should_use_warning(c, warn_in_eager=True)
      fn(h)
      del h
    error.assert_not_called()

  def testAddShouldUseWarningWhenUsedWithAdd(self):
    def add(h):
      _ = h + 1
    self._testAddShouldUseWarningWhenUsed(add, name='blah_add')
    gc.collect()
    self.assertFalse(gc.garbage)

  def testAddShouldUseWarningWhenUsedWithGetShape(self):
    def get_shape(h):
      _ = h.shape
    self._testAddShouldUseWarningWhenUsed(get_shape, name='blah_get_name')
    gc.collect()
    self.assertFalse(gc.garbage)

  def testShouldUseResult(self):
    @tf_should_use.should_use_result(warn_in_eager=True)
    def return_const(value):
      return constant_op.constant(value, name='blah2')
    with reroute_error() as error:
      return_const(0.0)
    msg = '\n'.join(error.call_args[0])
    self.assertIn('Object was never used', msg)
    if not context.executing_eagerly():
      self.assertIn('blah2:0', msg)
    self.assertIn('return_const', msg)
    gc.collect()
    self.assertFalse(gc.garbage)

  def testShouldUseResultWhenNotReallyUsed(self):
    @tf_should_use.should_use_result(warn_in_eager=True)
    def return_const(value):
      return constant_op.constant(value, name='blah3')
    with reroute_error() as error:
      with self.cached_session():
        return_const(0.0)
        # Creating another op and executing it does not mark the
        # unused op as being "used".
        v = constant_op.constant(1.0, name='meh')
        self.evaluate(v)
    msg = '\n'.join(error.call_args[0])
    self.assertIn('Object was never used', msg)
    if not context.executing_eagerly():
      self.assertIn('blah3:0', msg)
    self.assertIn('return_const', msg)
    gc.collect()
    self.assertFalse(gc.garbage)

  # Tests that mark_used is available in the API.
  def testMarkUsed(self):
    @tf_should_use.should_use_result(warn_in_eager=True)
    def return_const(value):
      return constant_op.constant(value, name='blah3')

    with self.cached_session():
      return_const(0.0).mark_used()

if __name__ == '__main__':
  test.main()
