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
"""Tests to improve the consistency with tf.TensorArray."""

import io
import logging as std_logging

import tensorflow as tf
from tensorflow.python.platform import test
from tensorflow.tools.consistency_integration_test.consistency_test_base import ConsistencyTestBase


class TensorArrayConsistencyTests(ConsistencyTestBase):
  """Test cases for known issues or bugs related to tf.TensorArray."""

  def testConcat(self):
    """Tests inconsistent behavior with `.concat()`.

    Bugs:   b/180921284
    Status: Missing error message
    Issue:  Running functions eagerly, calling `.concat` on a `tf.TensorArray`
            should raise an error but it does not.

    Error message:
      Expected error message is "Concatenating scalars in `tf.TensorArray` is
      unsupported in eager mode. Please use `.stack()` instead".

    Improve error message? Needed. (b/187851559)

    Notes:
    * Inconsistent behavior between eager and non-eager mode execution of the
      `tf.function` decorated function. In graph mode, the error is thrown.
    * We need to improve graph mode error message though. The error gets thrown
      is "Shapes must be equal rank, but are 1 and 0" and is hard to grasp.
    * Using `.stack()` as a workaround is working as intended:
      ```
      def f(x):
        return x.write(1, tf.constant([4, 5, 6]))

      ta = tf.TensorArray(dtype=tf.int32, dynamic_size=True, size=0)
      ta = ta.write(0, tf.constant([1, 2, 3]))
      f(ta).stack()  # <tf.Tensor: shape=(2, 3), dtype=int32,
                     # numpy=array([[1, 2, 3], [4, 5, 6]], dtype=int32)>
      ```
    """
    self.skipTest('b/180921284')
    try:
      tf.config.run_functions_eagerly(True)

      @tf.function
      def f(x, y, z):
        ta = tf.TensorArray(size=3, dtype=tf.int32, element_shape=())
        ta = ta.write(0, x)
        ta = ta.write(1, y)
        ta = ta.write(2, z)
        return ta.concat()

      with self.assertRaisesWithPredicateMatch(
          BaseException,
          # TODO(hyey): Below is a placeholder error message of what we
          # probably want but it needs to be updated to specify what caused
          # the error and where.
          'Concatenating scalars in `tf.TensorArray` is unsupported in eager '
          'mode. Please use `.stack()` instead'):
        f(1, 2, 3)

    finally:
      tf.config.run_functions_eagerly(False)

  def testArrayReturnedFromTfFunction(self):
    """Tests bad handling of tf.TensorArray returned from tf.function.

    Bugs:   b/147450234
    Status: Broken
    Issue:  `tf.TensorArray` returned from tf.function is a `tf.variant` tensor
            (i.e. `tf.Tensor(<unprintable>, shape=(), dtype=variant)`). Calling
            `stack()` on it causes an AttributeError.

    Error message:
      "AttributeError: 'tensorflow.python.framework.ops.EagerTensor' object has"
      " no attribute 'stack'"

    Notes:
    * Note that XLA fails with a different error that is equally confusing:
      "Support for TensorList crossing the XLA/TF boundary is not implemented."
    """
    self.skipTest('b/147450234')
    num_rows = 2

    @tf.function
    def f(x):
      ta = tf.TensorArray(tf.float32, num_rows)
      for i in range(num_rows):
        ta = ta.write(i, x[i])

      return ta

    n = tf.constant([[1., 2.], [3., 4.]])
    ta0 = f(n)
    ta1 = tf.TensorArray(tf.float32, num_rows)
    ta1 = ta1.write(0, n[0])
    ta1 = ta1.write(1, n[1])

    # Output of `f(n)` is `tf.Tensor(<unprintable>, shape=(), dtype=variant)`.
    self.assertAllEqual(ta0.stack(), ta1.stack())

  def testTensorArraySpec(self):
    """Tests tf.TensorArray behavior with `TensorArraySpec` as input signature.

    Bugs:   b/162452468, b/187114287
    Status: Broken
    Issue:  Using `tf.TensorArraySpec` as the input signature to tf.function
            does not work. This is not documented anywhere.

    Error message:
      "If shallow structure is a sequence, input must also be a sequence."

    Notes:
    * Documentation for `tf.TensorArraySpec` appears to be minimal. Need to
      update it.
    """
    self.skipTest('b/187114287')
    input_signature = [
        tf.TensorArraySpec(
            element_shape=None, dtype=tf.float32, dynamic_size=True)
    ]

    @tf.function(input_signature=input_signature)
    def f(ta):
      return ta.stack()

    ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
    ta = ta.write(0, tf.constant([1.0, 2.0]))
    ta = ta.write(1, tf.constant([3.0, 4.0]))

    out_t = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    self.assertAllEqual(f(ta), out_t)

  def testTensorArrayConcreteFunction(self):
    """Tests ConcreteFunction retrieval of a tf.function with a tf.TensorArray.

    Bugs:   b/162452468, b/187114664
    Status: Broken
    Issue:  Calling tf.function with a proper argument (i.e. traced input)
            fails. More specifically, calling `cf(arr)` should work but doesn't
            and calling `cf()` works rather when it should fail.
    """
    self.skipTest('b/187114664')

    @tf.function
    def fun(x):
      return x.stack()

    ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
    ta = ta.write(0, tf.constant([1.0, 2.0]))
    ta = ta.write(1, tf.constant([3.0, 4.0]))

    cf = fun.get_concrete_function(ta)
    t0 = cf(ta)
    t1 = ta.stack()
    self.assertAllEqual(t0, t1)

  def testVariantTensorAsOutput(self):
    """Tests that tf.variant tensor returns from tf.function for tf.TensorArray.

    Bugs:   b/162452468, b/187115938
    Status: Broken
    Issue:  `tf.TensorArray` returned from tf.function is a tf.variant tensor
            and is limited in functionality. For e.g., as simple as trying to
            `print()` or call `.numpy()` on it does not work (see
            `testBadIOErrorMsg` test case above).

    Notes:
    * When tf.function returns a `tf.TensorArray`, output returned should be a
      `tf.TensorArray`.
    """
    self.skipTest('b/187115938')

    @tf.function
    def f():
      ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
      ta = ta.write(0, tf.constant([1.0, 2.0]))
      ta = ta.write(1, tf.constant([3.0, 4.0]))
      return ta

    rtn_ta = f()
    # Initialize a `tf.TensorArray` to check against `rtn_ta` that it is a
    # `tf.TensorArray`.
    a_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
    self.assertEqual(rtn_ta.__module__, a_ta.__module__)

  def testTensorArrayPassedInAndReturnedFromTfFunction(self):
    """Tests tf.TensorArray passed in as input and returned as output.

    Bugs:   b/162452468, b/187115435, b/147450234
    Status: Broken
    Issue:  Returning `tf.TensorArray` from a tf.function does not work when
            passing it in as an input works. This is not documented anywhere.

    Error message:
      "Attempting to build a graph-mode TF2-style TensorArray from either an
      eager-mode TensorArray or a TF1-style TensorArray."
    """
    self.skipTest('b/187115435')

    @tf.function
    def f(ta):
      ta = ta.write(1, tf.constant([3.0, 4.0]))
      return ta

    ta0 = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
    ta0 = ta0.write(0, tf.constant([1.0, 2.0]))
    ta0 = f(ta0)

    ta1 = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
    ta1 = ta1.write(0, tf.constant([1.0, 2.0]))
    ta1 = ta1.write(1, tf.constant([3.0, 4.0]))

    self.assertAllEqual(ta0.stack(), ta1.stack())

  def testMissingWarning(self):
    """Tests warnings when the output of tf.TensorArray methods is unused.

    Bugs:   b/150784251
    Status: Broken
    Issue:  tf.TensorArray API doc specifies that a warning should be present
            when the output of tf.TensorArray methods is unused but no warning
            is present for tf.function decorated functions.
            https://www.tensorflow.org/api_docs/python/tf/TensorArray

    Error message:
      'Object was never used ... If you want to mark it as used call its
      "mark_used()" method.'

    Improve error message? Needed. (b/187852489)

    Notes:
    * Inconsistent behavior between when a function is decorated with
      tf.function and not. For example, if `f()` is tf.function-decorated, then
      it will NOT print the warning. If `f()` is NOT tf.function-decorated, then
      it will print the warning.
        ```
        @tf.function
        def f(x):
          ta = tf.TensorArray(x.dtype, tf.shape(x)[0])
          ta.write(0, x[0])

        f(tf.constant([1, 2, 3, 4]))
        ```
    * As simple as assignment operation is enough to avoid the warning case.
        ```
        @tf.function
        def f(x):
          ta = tf.TensorArray(x.dtype, tf.shape(x)[0])
          ta = ta.write(0, x[0])

        f(tf.constant([1, 2, 3, 4]))
        ```
    """
    self.skipTest('b/150784251')

    log = io.StringIO()
    handler = std_logging.StreamHandler(log)
    std_logging.root.addHandler(handler)

    @tf.function
    def f(x):
      ta = tf.TensorArray(x.dtype, tf.shape(x)[0])
      # A warning should be thrown with the line below. This is the case only
      # when `f()` is not decorated with tf.function.
      ta.write(0, x[0])

    f(tf.constant([1, 2, 3, 4]))

    self.assertIn('Object was never used', log.getvalue())
    std_logging.root.removeHandler(handler)


if __name__ == '__main__':
  test.main()
