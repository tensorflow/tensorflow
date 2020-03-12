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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import _memory_checker_test_helper
from tensorflow.python import keras
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework.memory_checker import MemoryChecker
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


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

  def testNoLeak2(self):
    helper = _memory_checker_test_helper.MemoryCheckerTestHelper()
    with MemoryChecker() as memory_checker:
      memory_checker.record_snapshot()
      helper.list_push_back(10)
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

  def testLeak2(self):
    helper = _memory_checker_test_helper.MemoryCheckerTestHelper()
    with MemoryChecker() as memory_checker:
      memory_checker.record_snapshot()
      helper.list_push_back(10)
      memory_checker.record_snapshot()
      helper.list_push_back(11)
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
    self.skipTest('TODO(b/150324603): Flaky test.')
    with MemoryChecker() as memory_checker:
      memory_checker.record_snapshot()
      memory_checker.record_snapshot()

    # TODO(kkb): All the builtins below are unexpected, locate and fix it.
    memory_checker.assert_no_new_python_objects(
        threshold={'builtins.weakref': 1,
                   'builtins.function': 1})

  def testNewPythonObjects(self):
    with MemoryChecker() as memory_checker:
      memory_checker.record_snapshot()
      x = constant_op.constant(1)  # pylint: disable=unused-variable
      memory_checker.record_snapshot()

    with self.assertRaisesRegexp(AssertionError, 'New Python objects'):
      memory_checker.assert_no_new_python_objects()

  def testNewPythonObjectBelowThreshold(self):

    class Foo(object):
      pass

    with MemoryChecker() as memory_checker:
      memory_checker.record_snapshot()
      foo = Foo()  # pylint: disable=unused-variable
      memory_checker.record_snapshot()

    # TODO(kkb): `{'builtins.weakref': 1, 'builtins.function': 1}` is
    # unexpected, locate and fix it.
    memory_checker.assert_no_new_python_objects(threshold={
        '__main__.Foo': 1,
        'builtins.weakref': 1,
        'builtins.function': 1,
    })
    memory_checker.assert_no_new_python_objects(threshold={
        '__main__.Foo': 2,
        'builtins.weakref': 1,
        'builtins.function': 1,
    })

  def testKerasBasic(self):
    # TODO(kkb): Fix the the slowness on Forge.
    self.skipTest('This test is too slow on Forge so disabled for now.')

    x = array_ops.zeros([1, 1])
    y = constant_op.constant([[3]])
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(1, input_dim=1))
    model.compile(loss='mean_squared_error')

    with MemoryChecker() as memory_checker:
      for _ in range(10):
        model.fit(x, y)
        model.evaluate(x, y)
        memory_checker.record_snapshot()

    memory_checker.report()
    memory_checker.assert_no_leak_if_all_possibly_except_one()

  def testKerasAdvanced(self):
    # TODO(kkb): Fix the the slowness on Forge.
    self.skipTest('This test is too slow on Forge so disabled for now.')

    # A real world example taken from the following.
    # https://github.com/tensorflow/tensorflow/issues/32500
    # b/142150794

    with MemoryChecker() as memory_checker:
      rows = 6
      columns = 7
      model = keras.Sequential([
          keras.layers.Flatten(input_shape=[rows * columns, 3]),
          keras.layers.Dense(7, input_shape=[rows * columns * 3]),
      ])

      model.compile(
          optimizer=keras.optimizer_v2.gradient_descent.SGD(lr=0.01),
          loss='mean_squared_error',
          metrics=['accuracy'])
      states = [[1] * rows * columns for _ in range(20)]
      f = array_ops.one_hot(states, dtype='float32', depth=3)

      for _ in range(20):
        model.predict(f, steps=10)
        memory_checker.record_snapshot()

    memory_checker.report()
    memory_checker.assert_no_leak_if_all_possibly_except_one()


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
