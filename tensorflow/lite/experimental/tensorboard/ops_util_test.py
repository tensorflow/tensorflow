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
"""Tests for backend."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.lite.experimental.tensorboard import ops_util
from tensorflow.python.platform import test


class OpsUtilTest(test.TestCase):

  def testGetPotentiallySupportedOps(self):
    ops = ops_util.get_potentially_supported_ops()
    # See GetTensorFlowNodeConverterMap() in
    # tensorflow/lite/toco/import_tensorflow.cc
    self.assertIsInstance(ops, list)
    # Test partial ops that surely exist in the list.
    self.assertIn(ops_util.SupportedOp("Add"), ops)
    self.assertIn(ops_util.SupportedOp("Log"), ops)
    self.assertIn(ops_util.SupportedOp("Sigmoid"), ops)
    self.assertIn(ops_util.SupportedOp("Softmax"), ops)


if __name__ == "__main__":
  test.main()
