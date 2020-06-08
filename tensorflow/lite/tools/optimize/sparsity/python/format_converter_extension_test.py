# Lint as: python3
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
"""Tests for the pybind11 bindings of format_converter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np

from tensorflow.lite.tools.optimize.sparsity.python import format_converter_extension as format_converter


class FormatConverterTest(absltest.TestCase):

  def test_bcsr_fp32(self):
    """Same as FormatConverterTest::BlockTestD0S1 but via pybind11."""
    # pyformat: disable
    dense_matrix = [1.0, 0.0, 2.0, 3.0,
                    0.0, 4.0, 0.0, 0.0,
                    0.0, 0.0, 5.0, 0.0,
                    0.0, 0.0, 0.0, 6.0]
    # pyformat: enable
    dense_shape = [4, 4]
    traversal_order = [0, 1, 2, 3]
    dim_types = [
        format_converter.TfLiteDimensionType.TF_LITE_DIM_DENSE,
        format_converter.TfLiteDimensionType.TF_LITE_DIM_SPARSE_CSR
    ]
    block_size = [2, 2]
    block_map = [0, 1]
    converter = format_converter.FormatConverterFp32(dense_shape,
                                                     traversal_order, dim_types,
                                                     block_size, block_map)

    converter.DenseToSparse(np.asarray(dense_matrix, dtype=np.float32).data)

    dim_metadata = converter.GetDimMetadata()
    self.assertEqual([2], dim_metadata[0])
    self.assertEmpty(dim_metadata[1])  # rows are dense.

    self.assertEqual([0, 2, 3], dim_metadata[2])  # array segments.
    self.assertEqual([0, 1, 1], dim_metadata[3])  # array indices.

    self.assertEqual([2], dim_metadata[4])
    self.assertEmpty(dim_metadata[5])  # sub block rows are dense.

    self.assertEqual([2], dim_metadata[6])
    self.assertEmpty(dim_metadata[7])  # sub block columns are dense.

    expected_data = [1.0, 0.0, 0.0, 4.0, 2.0, 3.0, 0.0, 0.0, 5.0, 0.0, 0.0, 6.0]
    sparse_data = converter.GetData()
    self.assertTrue(np.allclose(expected_data, sparse_data))

    converter.SparseToDense(np.asarray(sparse_data, dtype=np.float32).data)
    self.assertTrue(np.allclose(dense_matrix, converter.GetData()))


if __name__ == '__main__':
  absltest.main()
