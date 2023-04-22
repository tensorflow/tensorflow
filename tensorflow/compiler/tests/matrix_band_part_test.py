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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class MatrixBandPartTest(xla_test.XLATestCase, parameterized.TestCase):

  @parameterized.parameters(
      {
          'batch_shape': [],
          'rows': 1,
          'cols': 1
      },
      {
          'batch_shape': [],
          'rows': 1,
          'cols': 2
      },
      {
          'batch_shape': [],
          'rows': 1,
          'cols': 7
      },
      {
          'batch_shape': [],
          'rows': 2,
          'cols': 1
      },
      {
          'batch_shape': [],
          'rows': 2,
          'cols': 2
      },
      {
          'batch_shape': [],
          'rows': 2,
          'cols': 7
      },
      {
          'batch_shape': [],
          'rows': 7,
          'cols': 1
      },
      {
          'batch_shape': [],
          'rows': 7,
          'cols': 2
      },
      {
          'batch_shape': [],
          'rows': 7,
          'cols': 7
      },
      {
          'batch_shape': [2,],
          'rows': 1,
          'cols': 1
      },
      {
          'batch_shape': [2,],
          'rows': 1,
          'cols': 2
      },
      {
          'batch_shape': [2,],
          'rows': 1,
          'cols': 7
      },
      {
          'batch_shape': [2,],
          'rows': 2,
          'cols': 1
      },
      {
          'batch_shape': [2,],
          'rows': 2,
          'cols': 2
      },
      {
          'batch_shape': [2,],
          'rows': 2,
          'cols': 7
      },
      {
          'batch_shape': [2,],
          'rows': 7,
          'cols': 1
      },
      {
          'batch_shape': [2,],
          'rows': 7,
          'cols': 2
      },
      {
          'batch_shape': [2,],
          'rows': 7,
          'cols': 7
      },
      {
          'batch_shape': [1, 3, 2],
          'rows': 1,
          'cols': 1
      },
      {
          'batch_shape': [1, 3, 2],
          'rows': 1,
          'cols': 2
      },
      {
          'batch_shape': [1, 3, 2],
          'rows': 1,
          'cols': 7
      },
      {
          'batch_shape': [1, 3, 2],
          'rows': 2,
          'cols': 1
      },
      {
          'batch_shape': [1, 3, 2],
          'rows': 2,
          'cols': 2
      },
      {
          'batch_shape': [1, 3, 2],
          'rows': 2,
          'cols': 7
      },
      {
          'batch_shape': [1, 3, 2],
          'rows': 7,
          'cols': 1
      },
      {
          'batch_shape': [1, 3, 2],
          'rows': 7,
          'cols': 2
      },
      {
          'batch_shape': [1, 3, 2],
          'rows': 7,
          'cols': 7
      },
  )
  def testMatrixBandPart(self, batch_shape, rows, cols):
    # TODO(b/125505881): Disabled due to LLVM backend crash.
    if self.device == 'XLA_CPU' and cols == 7 and rows == 1 and batch_shape == [
        1, 3, 2
    ]:
      pass
    for dtype in self.float_types:
      with self.session():
        mat = np.ones(batch_shape + [rows, cols]).astype(dtype)
        batch_mat = np.tile(mat, batch_shape + [1, 1])
        for lower in -1, 0, 1, rows - 1:
          for upper in -1, 0, 1, cols - 1:
            band_np = mat
            if lower >= 0:
              band_np = np.triu(band_np, -lower)
            if upper >= 0:
              band_np = np.tril(band_np, upper)
            if batch_shape:
              band_np = np.tile(band_np, batch_shape + [1, 1])

            placeholder = array_ops.placeholder(dtype)
            with self.test_scope():
              band = array_ops.matrix_band_part(
                  placeholder, constant_op.constant(lower, dtype=dtypes.int32),
                  constant_op.constant(upper, dtype=dtypes.int32))
              feed_dict = {placeholder: batch_mat}
              self.assertAllEqual(band_np, band.eval(feed_dict=feed_dict))


if __name__ == "__main__":
  test.main()
