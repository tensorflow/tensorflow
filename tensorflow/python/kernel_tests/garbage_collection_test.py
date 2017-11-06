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
"""Tests which set DEBUG_SAVEALL and assert no garbage was created.

This flag seems to be sticky, so these tests have been isolated for now.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc

from tensorflow.python.eager import context
from tensorflow.python.framework import test_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test


def assert_no_garbage_created(f):
  """Test decorator to assert that no garbage has been created."""

  def decorator(self):
    """Sets DEBUG_SAVEALL, runs the test, and checks for new garbage."""
    gc.disable()
    previous_debug_flags = gc.get_debug()
    gc.set_debug(gc.DEBUG_SAVEALL)
    gc.collect()
    previous_garbage = len(gc.garbage)
    f(self)
    gc.collect()
    # This will fail if any garbage has been created, typically because of a
    # reference cycle.
    self.assertEqual(previous_garbage, len(gc.garbage))
    # TODO(allenl): Figure out why this debug flag reset doesn't work. It would
    # be nice to be able to decorate arbitrary tests in a large test suite and
    # not hold on to every object in other tests.
    gc.set_debug(previous_debug_flags)
    gc.enable()
  return decorator


class NoReferenceCycleTests(test_util.TensorFlowTestCase):

  @assert_no_garbage_created
  def testEagerResourceVariables(self):
    with context.eager_mode():
      resource_variable_ops.ResourceVariable(1.0, name="a")


if __name__ == "__main__":
  test.main()
