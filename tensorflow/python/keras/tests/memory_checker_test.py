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

from tensorflow.python import keras
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework.memory_checker import MemoryChecker
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class MemoryCheckerTest(test.TestCase):

  def testKerasBasic(self):
    # TODO(kkb): Fix the slowness on Forge.
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
    # TODO(kkb): Fix the slowness on Forge.
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
