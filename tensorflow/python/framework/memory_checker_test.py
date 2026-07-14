# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework.memory_checker import MemoryChecker
from tensorflow.python.platform import test


@test_util.with_eager_op_as_function
class MemoryCheckerTest(test.TestCase):

  def testNoLeakEmpty(self):
    with MemoryChecker() as memory_checker:
      memory_checker.record_snapshot()
      memory_checker.record_snapshot()
      memory_checker.record_snapshot()
      memory_checker.record_snapshot()

    memory_checker.report()
    memory_checker.assert_no_leak_if_all_possibly_except_one()

  def testNoLeak1(self):
    with MemoryChecker() as memory_checker:
      memory_checker.record_snapshot()
      x = constant_op.constant(1)  # pylint: disable=unused-variable
      memory_checker.record_snapshot()
      memory_checker.record_snapshot()
      memory_checker.record_snapshot()

    memory_checker.report()
    memory_checker.assert_no_leak_if_all_possibly_except_one()

  def testNoLeak3(self):
    with MemoryChecker() as memory_checker:
      tensors = []
      for i in range(10):
        if i not in (5, 7):
          tensors.append(constant_op.constant(1))
        memory_checker.record_snapshot()

    memory_checker.report()
    memory_checker.assert_no_leak_if_all_possibly_except_one()

  def testLeak1(self):
    with MemoryChecker() as memory_checker:
      memory_checker.record_snapshot()
      x = constant_op.constant(1)  # pylint: disable=unused-variable
      memory_checker.record_snapshot()
      y = constant_op.constant(1)  # pylint: disable=unused-variable
      memory_checker.record_snapshot()
      memory_checker.record_snapshot()

    memory_checker.report()
    with self.assertRaises(AssertionError):
      memory_checker.assert_no_leak_if_all_possibly_except_one()

  def testLeak3(self):
    with MemoryChecker() as memory_checker:
      tensors = []
      for _ in range(10):
        tensors.append(constant_op.constant(1))
        memory_checker.record_snapshot()

    memory_checker.report()
    with self.assertRaises(AssertionError):
      memory_checker.assert_no_leak_if_all_possibly_except_one()

  def testNoNewPythonObjectsEmpty(self):
    with MemoryChecker() as memory_checker:
      memory_checker.record_snapshot()
      memory_checker.record_snapshot()

    memory_checker.assert_no_new_python_objects()

  def testNewPythonObjects(self):
    with MemoryChecker() as memory_checker:
      memory_checker.record_snapshot()
      x = constant_op.constant(1)
      memory_checker.record_snapshot()

    with self.assertRaisesRegex(AssertionError, 'New Python objects'):
      memory_checker.assert_no_new_python_objects()

    # use x to avoid any potential for optimizing it away.
    self.assertIsNot(x, None)

  def testNewPythonObjectBelowThreshold(self):

    class Foo(object):
      pass

    with MemoryChecker() as memory_checker:
      memory_checker.record_snapshot()
      foo = Foo()
      del foo
      memory_checker.record_snapshot()

    memory_checker.assert_no_new_python_objects()


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
