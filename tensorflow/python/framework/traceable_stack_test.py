# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.python.framework.traceable_stack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import test_util
from tensorflow.python.framework import traceable_stack
from tensorflow.python.platform import googletest
from tensorflow.python.util import tf_inspect as inspect

_LOCAL_OBJECT = lambda x: x
_THIS_FILENAME = inspect.getsourcefile(_LOCAL_OBJECT)


class TraceableObjectTest(test_util.TensorFlowTestCase):

  def testSetFilenameAndLineFromCallerUsesCallersStack(self):
    t_obj = traceable_stack.TraceableObject(17)

    # Do not separate placeholder from the set_filename_and_line_from_caller()
    # call one line below it as it is used to calculate the latter's line
    # number.
    placeholder = lambda x: x
    result = t_obj.set_filename_and_line_from_caller()

    expected_lineno = inspect.getsourcelines(placeholder)[1] + 1
    self.assertEqual(expected_lineno, t_obj.lineno)
    self.assertEqual(_THIS_FILENAME, t_obj.filename)
    self.assertEqual(t_obj.SUCCESS, result)

  def testSetFilenameAndLineFromCallerRespectsOffset(self):

    def call_set_filename_and_line_from_caller(t_obj):
      # We expect to retrieve the line number from _our_ caller.
      return t_obj.set_filename_and_line_from_caller(offset=1)

    t_obj = traceable_stack.TraceableObject(None)
    # Do not separate placeholder from the
    # call_set_filename_and_line_from_caller() call one line below it as it is
    # used to calculate the latter's line number.
    placeholder = lambda x: x
    result = call_set_filename_and_line_from_caller(t_obj)

    expected_lineno = inspect.getsourcelines(placeholder)[1] + 1
    self.assertEqual(expected_lineno, t_obj.lineno)
    self.assertEqual(t_obj.SUCCESS, result)

  def testSetFilenameAndLineFromCallerHandlesRidiculousOffset(self):
    t_obj = traceable_stack.TraceableObject('The quick brown fox.')
    # This line shouldn't die.
    result = t_obj.set_filename_and_line_from_caller(offset=300)

    # We expect a heuristic to be used because we are not currently 300 frames
    # down on the stack.  The filename should be some wacky thing from the
    # outermost stack frame -- definitely not equal to this filename.
    self.assertEqual(t_obj.HEURISTIC_USED, result)
    self.assertNotEqual(_THIS_FILENAME, t_obj.filename)


class TraceableStackTest(test_util.TensorFlowTestCase):

  def testPushPeekPopObj(self):
    t_stack = traceable_stack.TraceableStack()
    t_stack.push_obj(42.0)
    t_stack.push_obj('hope')

    expected_lifo_peek = ['hope', 42.0]
    self.assertEqual(expected_lifo_peek, t_stack.peek_objs())

    self.assertEqual('hope', t_stack.pop_obj())
    self.assertEqual(42.0, t_stack.pop_obj())

  def testPushPopPreserveLifoOrdering(self):
    t_stack = traceable_stack.TraceableStack()
    t_stack.push_obj(0)
    t_stack.push_obj(1)
    t_stack.push_obj(2)
    t_stack.push_obj(3)

    obj_3 = t_stack.pop_obj()
    obj_2 = t_stack.pop_obj()
    obj_1 = t_stack.pop_obj()
    obj_0 = t_stack.pop_obj()

    self.assertEqual(3, obj_3)
    self.assertEqual(2, obj_2)
    self.assertEqual(1, obj_1)
    self.assertEqual(0, obj_0)

  def testPushObjSetsFilenameAndLineInfoForCaller(self):
    t_stack = traceable_stack.TraceableStack()

    # We expect that the line number recorded for the 1-object will come from
    # the call to t_stack.push_obj(1).  Do not separate the next two lines!
    placeholder_1 = lambda x: x
    t_stack.push_obj(1)

    # We expect that the line number recorded for the 2-object will come from
    # the call to call_push_obj() and _not_ the call to t_stack.push_obj().
    def call_push_obj(obj):
      t_stack.push_obj(obj, offset=1)

    # Do not separate the next two lines!
    placeholder_2 = lambda x: x
    call_push_obj(2)

    expected_lineno_1 = inspect.getsourcelines(placeholder_1)[1] + 1
    expected_lineno_2 = inspect.getsourcelines(placeholder_2)[1] + 1

    t_obj_2, t_obj_1 = t_stack.peek_traceable_objs()
    self.assertEqual(expected_lineno_2, t_obj_2.lineno)
    self.assertEqual(expected_lineno_1, t_obj_1.lineno)


if __name__ == '__main__':
  googletest.main()
