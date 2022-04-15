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
"""Tests for querying registered kernels."""
from tensorflow.python.framework import kernels
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class GetAllRegisteredKernelsTest(test_util.TensorFlowTestCase):

  def testFindsAtLeastOneKernel(self):
    kernel_list = kernels.get_all_registered_kernels()
    self.assertGreater(len(kernel_list.kernel), 0)


class GetRegisteredKernelsForOp(test_util.TensorFlowTestCase):

  def testFindsAtLeastOneKernel(self):
    kernel_list = kernels.get_registered_kernels_for_op("KernelLabel")
    self.assertGreater(len(kernel_list.kernel), 0)
    self.assertEqual(kernel_list.kernel[0].op, "KernelLabel")


if __name__ == "__main__":
  googletest.main()
