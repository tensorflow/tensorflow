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
"""Large tests for tensorflow.ops.linalg_ops.qr."""

import unittest

from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.tests import qr_op_test
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


@test_util.run_all_without_tensor_float_32(
    "XLA QR op calls matmul. Also, matmul used for verification. Also with "
    'TensorFloat-32, mysterious "Unable to launch cuBLAS gemm" error '
    "occasionally occurs"
)
# TODO(b/165435566): Fix "Unable to launch cuBLAS gemm" error
class QrOpLargeTest(
    qr_op_test.QrOpTestBase, xla_test.XLATestCase, parameterized.TestCase
):

  def testLarge2000x2000(self):
    x_np = self._random_matrix(np.float32, (2000, 2000))
    self._test(x_np, full_matrices=True)

  @unittest.skip("Test times out on CI")
  def testLarge17500x128(self):
    x_np = self._random_matrix(np.float32, (17500, 128))
    self._test(x_np, full_matrices=True)


if __name__ == "__main__":
  test.main()
