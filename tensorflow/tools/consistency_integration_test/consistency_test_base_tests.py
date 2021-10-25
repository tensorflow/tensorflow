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

    # Tests involving type promotions are added to:
    # //tensorflow/tools/consistency_integration_test/type_promotion_tests.py
    self._generic_test(f, [
        Example(arg=(3,), out=9., failure=[], bugs=[]),
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
            # It should fail in all modes.
            failure=[RunMode.RAW, RunMode.XLA, RunMode.FUNCTION, RunMode.SAVED],
            bugs=[])
    ])
    return

  def testNotEqualOutput(self):
    """Tests that an error is thrown if the outputs are not equal.

    This test case is meant to test the consistency test infrastructure that the
    output of executing `f()` matches the groundtruth we provide as the `out`
    param in `_generic_test()`.
    """
    mock_func = test.mock.MagicMock(name='method')
    mock_func.return_value = 0  # This differs from the `expected` value below.
    mock_func.__doc__ = 'Tested with a mock function.'

    failure_modes = [RunMode.RAW, RunMode.FUNCTION, RunMode.XLA, RunMode.SAVED]
    input_args = [3, 3.2, tf.constant(3.)]
    expected = 1  # Randomly picked value just for testing purposes.

    for input_arg in input_args:
      self._generic_test(mock_func, [
          Example(
              arg=(input_arg,), out=expected, failure=failure_modes, bugs=[])
      ])

  def testSkipModes(self):
    """Tests `skip_modes` option available with `_generic_test`."""

    class A:

      def __init__(self, x):
        self.value = x

    def f(x):
      return A(x)

    self._generic_test(
        f, [Example(arg=(3.,), out=3.0, failure=[], bugs=[])],
        # Skip all tests as the test will fail in all modes.
        skip_modes=[RunMode.RAW, RunMode.XLA, RunMode.FUNCTION, RunMode.SAVED])
    return

  def testTensorArrayBasic(self):
    """Tests `_generic_test` with a `tf.TensorArray` as input to tf.function."""

    def f(x):
      return x.stack()

    ta = tf.TensorArray(dtype=tf.int32, dynamic_size=True, size=0)
    ta = ta.write(0, tf.constant([1, 2, 3]))
    ta = ta.write(1, tf.constant([4, 5, 6]))

    self._generic_test(
        f,
        [
            Example(
                arg=(ta,),
                out=tf.constant([[1, 2, 3], [4, 5, 6]]),
                failure=[RunMode.SAVED],  # TODO(b/187250924): Investigate.
                bugs=['b/180921284'])
        ])
    return

  def testFailureParamAsDict(self):
    """Tests passing in a `dict` for `failure` param to `_generic_test`."""

    def f(ta):
      return ta.stack()

    ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
    ta = ta.write(0, tf.constant([1.0, 2.0]))
    ta = ta.write(1, tf.constant([3.0, 4.0]))

    out_t = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    input_signature = [tf.TensorArraySpec(element_shape=None,
                                          dtype=tf.float32,
                                          dynamic_size=True)]

    self._generic_test(
        f,
        [
            Example(
                arg=(ta,),
                out=out_t,
                failure={
                    RunMode.FUNCTION:
                        'If shallow structure is a sequence, input must also '
                        'be a sequence',
                    RunMode.XLA:
                        'If shallow structure is a sequence, input must also '
                        'be a sequence',
                    RunMode.SAVED:
                        'Found zero restored functions for caller function',
                },
                bugs=['b/162452468'])
        ],
        input_signature=input_signature,
        skip_modes=[])
    return


if __name__ == '__main__':
  test.main()
