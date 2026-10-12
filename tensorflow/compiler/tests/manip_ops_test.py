# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Test cases for manip ops."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.platform import googletest


class ManipOpsTest(xla_test.XLATestCase):
  """Test cases for manip ops."""

  def _testRoll(self, a, shift, axis):
    with self.session() as session:
      with self.test_scope():
        p = array_ops.placeholder(dtypes.as_dtype(a.dtype), a.shape, name="a")
        output = manip_ops.roll(a, shift, axis)
      result = session.run(output, {p: a})
      self.assertAllEqual(result, np.roll(a, shift, axis))

  def testNumericTypes(self):
    for t in self.numeric_types:
      self._testRoll(np.random.randint(-100, 100, (5)).astype(t), 3, 0)
      self._testRoll(
          np.random.randint(-100, 100, (4, 4, 3)).astype(t), [1, -6, 6],
          [0, 1, 2])
      self._testRoll(
          np.random.randint(-100, 100, (4, 2, 1, 3)).astype(t), [0, 1, -2],
          [1, 2, 3])

  def testFloatTypes(self):
    for t in self.float_types:
      self._testRoll(np.random.rand(5).astype(t), 2, 0)
      self._testRoll(np.random.rand(3, 4).astype(t), [1, 2], [1, 0])
      self._testRoll(np.random.rand(1, 3, 4).astype(t), [1, 0, -3], [0, 1, 2])

  def testComplexTypes(self):
    for t in self.complex_types:
      x = np.random.rand(4, 4).astype(t)
      self._testRoll(x + 1j * x, 2, 0)
      x = np.random.rand(2, 5).astype(t)
      self._testRoll(x + 1j * x, [1, 2], [1, 0])
      x = np.random.rand(3, 2, 1, 1).astype(t)
      self._testRoll(x + 1j * x, [2, 1, 1, 0], [0, 3, 1, 2])

  def _rollSelected(self, x, masks, shift, axis):
    # Keeps the entries of x where masks[d] is true along each dimension d with
    # a mask, which gives that dimension a size only known at run time, and
    # rolls the result.
    dims = sorted(masks)

    @def_function.function(jit_compile=True)
    def f(x, *mask_args):
      for d, mask in zip(dims, mask_args):
        x = array_ops.gather(
            x, array_ops.reshape(array_ops.where(mask), [-1]), axis=d)
      return manip_ops.roll(x, shift, axis)

    x = np.array(x, dtype=np.int32)
    with self.session() as session:
      with self.device_scope():
        x_ph = array_ops.placeholder(dtypes.int32, x.shape)
        mask_phs = [
            array_ops.placeholder(dtypes.bool, [x.shape[d]]) for d in dims
        ]
        output = f(x_ph, *mask_phs)
      feed_dict = {x_ph: x}
      feed_dict.update({ph: masks[d] for ph, d in zip(mask_phs, dims)})
      return session.run(output, feed_dict)

  def testDynamicDimensions(self):
    # Regression test for GitHub issue 128951: a dynamic dimension was rolled
    # over its bound instead of its size, and the output had the bound.
    x = [1, 5, 2, 7, 3]
    masks = {0: [True, False, True, True, False]}  # Keeps [1, 2, 7].
    self.assertAllEqual(self._rollSelected(x, masks, 1, 0), [7, 1, 2])
    self.assertAllEqual(self._rollSelected(x, masks, -1, 0), [2, 7, 1])
    self.assertAllEqual(self._rollSelected(x, masks, 4, 0), [7, 1, 2])
    # Both dimensions dynamic, rolled together.
    x = np.arange(20).reshape(4, 5)
    masks = {0: [True, False, True, True], 1: [False, True, True, False, True]}
    self.assertAllEqual(
        self._rollSelected(x, masks, [1, 2], [0, 1]),
        [[17, 19, 16], [2, 4, 1], [12, 14, 11]])


if __name__ == "__main__":
  googletest.main()
