# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for quantized operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


class QuantizedOpsTest(xla_test.XLATestCase):

  # Verify that quantized types can be clustered by XLA.
  def testQuantizedTypeRoundtrip(self):
    with self.cached_session() as session:
      for dtype in self.quantized_tf_types:
        in_values = np.array([1, 2, 3, 4, 5, 6])
        expected = [[1, 2], [3, 4], [5, 6]]
        with self.test_scope():
          p = array_ops.placeholder(dtype=dtypes.int32)
          x = math_ops.cast(p, dtype)
          x = array_ops.reshape(x, [3, 2])

        value = session.run(x, {p: in_values})
        self.assertAllEqual(value, expected)


if __name__ == "__main__":
  googletest.main()
