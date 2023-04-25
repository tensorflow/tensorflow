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
"""Tests for integer division by zero."""

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class ZeroDivisionTest(test.TestCase):

  def testZeros(self):
    with test_util.use_gpu():
      for dtype in dtypes.uint8, dtypes.int16, dtypes.int32, dtypes.int64:
        zero = constant_op.constant(0, dtype=dtype)
        one = constant_op.constant(1, dtype=dtype)
        bads = [lambda x, y: x // y]
        if dtype in (dtypes.int32, dtypes.int64):
          bads.append(lambda x, y: x % y)
        for bad in bads:
          try:
            result = self.evaluate(bad(one, zero))
          except (errors.OpError, errors.InvalidArgumentError) as e:
            # Ideally, we'd get a nice exception.  In theory, this should only
            # happen on CPU, but 32 bit integer GPU division is actually on
            # CPU due to a placer bug.
            # TODO(geoffreyi): Make stricter once the placer bug is fixed.
            self.assertIn('Integer division by zero', str(e))
          else:
            # On the GPU, integer division by zero produces all bits set.
            # But apparently on some GPUs "all bits set" for 64 bit division
            # means 32 bits set, so we allow 0xffffffff as well.  This isn't
            # very portable, so we may need to expand this list if other GPUs
            # do different things.
            #
            # XLA constant folds integer division by zero to 1.
            self.assertTrue(test.is_gpu_available())
            self.assertIn(result, (-1, 1, 2, 0xff, 0xffffffff))


if __name__ == '__main__':
  test.main()
