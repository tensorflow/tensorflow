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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import gc
import sys

from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import tf_should_use


@contextlib.contextmanager
def reroute_error(captured):
  """Temporarily reroute errors written to tf_logging.error into `captured`."""
  del captured[:]
  true_logger = tf_logging.error
  def capture_errors(*args, **unused_kwargs):
    captured.extend(args)
  tf_logging.error = capture_errors
  try:
    yield
  finally:
    tf_logging.error = true_logger


class TfShouldUseTest(test.TestCase):

  def testAddShouldUseWarningWhenNotUsed(self):
    c = constant_op.constant(0, name='blah0')
    captured = []
    with reroute_error(captured):
      def in_this_function():
        h = tf_should_use._add_should_use_warning(c)
        del h
      in_this_function()
    self.assertIn('Object was never used', '\n'.join(captured))
    self.assertIn('blah0:0', '\n'.join(captured))
    self.assertIn('in_this_function', '\n'.join(captured))
    gc.collect()
    self.assertFalse(gc.garbage)

  def _testAddShouldUseWarningWhenUsed(self, fn, name):
    c = constant_op.constant(0, name=name)
    captured = []
    with reroute_error(captured):
      h = tf_should_use._add_should_use_warning(c)
      fn(h)
      del h
    self.assertNotIn('Object was never used', '\n'.join(captured))
    self.assertNotIn('%s:0' % name, '\n'.join(captured))

  def testAddShouldUseWarningWhenUsedWithAdd(self):
    def add(h):
      _ = h + 1
    self._testAddShouldUseWarningWhenUsed(add, name='blah_add')
    gc.collect()
    self.assertFalse(gc.garbage)

  def testAddShouldUseWarningWhenUsedWithGetName(self):
    def get_name(h):
      _ = h.name
    self._testAddShouldUseWarningWhenUsed(get_name, name='blah_get_name')
    gc.collect()
    self.assertFalse(gc.garbage)

  def testShouldUseResult(self):
    @tf_should_use.should_use_result
    def return_const(value):
      return constant_op.constant(value, name='blah2')
    captured = []
    with reroute_error(captured):
      return_const(0.0)
    self.assertIn('Object was never used', '\n'.join(captured))
    self.assertIn('blah2:0', '\n'.join(captured))
    self.assertIn('return_const', '\n'.join(captured))
    gc.collect()
    self.assertFalse(gc.garbage)

  def testShouldUseResultWhenNotReallyUsed(self):
    @tf_should_use.should_use_result
    def return_const(value):
      return constant_op.constant(value, name='blah3')
    captured = []
    with reroute_error(captured):
      with self.test_session():
        return_const(0.0)
        # Creating another op and executing it does not mark the
        # unused op as being "used".
        v = constant_op.constant(1.0, name='meh')
        v.eval()
    self.assertIn('Object was never used', '\n'.join(captured))
    self.assertIn('blah3:0', '\n'.join(captured))
    self.assertIn('return_const', '\n'.join(captured))
    gc.collect()
    self.assertFalse(gc.garbage)


if __name__ == '__main__':
  test.main()
