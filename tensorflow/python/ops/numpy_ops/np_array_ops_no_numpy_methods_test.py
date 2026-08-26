# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf numpy array methods that must not depend on Tensor methods.

`np_math_ops.enable_numpy_methods_on_tensor()` adds numpy methods such as
`astype` to `Tensor`. The other numpy_ops test modules call it from their
`__main__`, so a code path that only works once those methods are installed
still passes there. This module deliberately does not call it, which is the
configuration a user gets from a plain `import tensorflow`.
"""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.platform import test


class ArrayWithoutNumpyMethodsOnTensorTest(test.TestCase):

  def testAroundAcceptsFloatInputs(self):
    # `around` casts back to the argument's dtype on every path, so it is
    # broken for float arguments too, not only the promoted integer ones.
    arg = np.array([1.234, 5.678, -2.345], dtype=np.float32)
    for decimals in (0, 1, 2):
      actual = np_array_ops.around(arg, decimals)
      np.testing.assert_allclose(
          np.asarray(actual),
          np.around(arg, decimals),
          rtol=1e-6,
          atol=1e-6,
          err_msg='around({}, {})'.format(arg, decimals),
      )

  def testAroundAcceptsIntegerInputs(self):
    # An integer argument takes the promoting branch, which computes in a
    # float dtype and casts back to the integer dtype at the end.
    for arg in (
        np.array([11, 25, 37], dtype=np.int32),
        np.array([11, 25, 37], dtype=np.int64),
    ):
      for decimals in (0, 1):
        actual = np_array_ops.around(arg, decimals)
        np.testing.assert_array_equal(
            np.asarray(actual),
            np.around(arg, decimals),
            err_msg='around({}, {})'.format(arg, decimals),
        )

  def testAroundPreservesDtype(self):
    for dtype in (np.int32, np.int64, np.float32, np.float64):
      arg = np.array([1, 2, 3], dtype=dtype)
      self.assertEqual(np_array_ops.around(arg).dtype, dtype)

  def testRoundAcceptsFloatInputs(self):
    arg = np.array([1.234, 5.678], dtype=np.float32)
    np.testing.assert_allclose(
        np.asarray(np_array_ops.round(arg, 1)),
        np.round(arg, 1),
        rtol=1e-6,
        atol=1e-6,
    )

  def testBuiltinRoundOnTensor(self):
    # `around` is installed as `Tensor.__round__` at import time, so the
    # builtin `round()` reaches the same code path.
    tensor = constant_op.constant([1.234, 5.678], dtype='float32')
    np.testing.assert_allclose(
        np.asarray(round(tensor, 1)),
        np.around(np.array([1.234, 5.678], dtype=np.float32), 1),
        rtol=1e-6,
        atol=1e-6,
    )

  def testIndexUpdateHelpersPreserveDtype(self):
    # The `_with_index_*` helpers are attached to `Tensor` at import time,
    # rather than by `enable_numpy_methods_on_tensor()`, so they must not
    # require the opt-in either. They are attached as `functools.partial`
    # objects, which are not descriptors, so the tensor is passed explicitly
    # instead of being bound as `self`.
    tensor = constant_op.constant([1, 2, 3, 4], dtype='int32')
    updates = constant_op.constant([9, 9], dtype='int32')
    cases = [
        ('update', tensor._with_index_update, np.array([9, 9, 3, 4])),
        ('add', tensor._with_index_add, np.array([10, 11, 3, 4])),
    ]
    for name, helper, expected in cases:
      actual = helper(tensor, slice(0, 2), updates)
      self.assertEqual(actual.dtype, tensor.dtype, msg=name)
      np.testing.assert_array_equal(
          np.asarray(actual), expected, err_msg=name
      )


if __name__ == '__main__':
  ops.enable_eager_execution()
  # Intentionally not calling `np_math_ops.enable_numpy_methods_on_tensor()`.
  test.main()
