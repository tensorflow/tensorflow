# Lint as: python2, python3
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
"""Tests to improve the the of tensorflow.

Basic tests show how to use the consistency test to test against function,
eager, and xla function modes.
"""

import tensorflow as tf
from tensorflow.python.platform import test
from tensorflow.tools.consistency_integration_test.consistency_test_base import ConsistencyTestBase
from tensorflow.tools.consistency_integration_test.consistency_test_base import Example
from tensorflow.tools.consistency_integration_test.consistency_test_base import RunMode


class BasicTests(ConsistencyTestBase):
  """A few basic tests that are examples for other test writers."""

  def testSquare(self):
    """Test basic testing infrastructure."""

    def f(x):
      return x * x

    self._generic_test(f, [
        Example(arg=(3,), out=9., failure=[], bugs=[]),
        Example(arg=(3.2,), out=10.24, failure=[], bugs=[]),
        Example(
            arg=(tf.constant(3.),), out=tf.constant(9.), failure=[], bugs=[]),
    ])

  def testObjectInput(self):
    """Test taking a Python object. Should work in tf.function but not sm."""

    class A:

      def __init__(self):
        self.value = 3.0

    def f(x):
      return x.value

    self._generic_test(
        f, [Example(arg=(A(),), out=3.0, failure=[RunMode.SAVED], bugs=[])])
    return

  def testObjectOutput(self):
    """Test returning a Python object. Doesn't and shouldn't work."""

    class A:

      def __init__(self, x):
        self.value = x

    def f(x):
      return A(x)

    self._generic_test(f, [
        Example(
            arg=(3.,),
            out=3.0,
            failure=[RunMode.XLA, RunMode.FUNCTION, RunMode.SAVED],
            bugs=[])
    ])
    return


if __name__ == '__main__':
  test.main()
