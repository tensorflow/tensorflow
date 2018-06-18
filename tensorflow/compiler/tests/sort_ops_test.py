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
"""Tests for XlaSort."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class XlaSortOpTest(xla_test.XLATestCase):

  def _assertOpOutputMatchesExpected(self, op, args, expected):
    with self.test_session() as session:
      with self.test_scope():
        placeholders = [
            array_ops.placeholder(dtypes.as_dtype(arg.dtype), arg.shape)
            for arg in args
        ]
        feeds = {placeholders[i]: args[i] for i in range(0, len(args))}
        output = op(*placeholders)
      result = session.run(output, feeds)
      self.assertAllClose(result, expected, rtol=1e-3)

  def testSort(self):
    # TODO(b/26783907): The Sort HLO is not implemented on CPU or GPU.
    if self.device in ["XLA_CPU", "XLA_GPU"]:
      return
    supported_types = set([dtypes.bfloat16.as_numpy_dtype, np.float32])
    for dtype in supported_types.intersection(self.numeric_types):
      x = np.arange(101, dtype=dtype)
      np.random.shuffle(x)
      self._assertOpOutputMatchesExpected(
          xla.sort, [x], expected=np.arange(101, dtype=dtype))


if __name__ == "__main__":
  test.main()
