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
"""Tests for SavedModel utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.saved_model import utils


class UtilsTest(test.TestCase):

  def testBuildTensorInfo(self):
    x = array_ops.placeholder(dtypes.float32, 1, name="x")
    x_tensor_info = utils.build_tensor_info(x)
    self.assertEqual("x:0", x_tensor_info.name)
    self.assertEqual(types_pb2.DT_FLOAT, x_tensor_info.dtype)
    self.assertEqual(1, len(x_tensor_info.tensor_shape.dim))
    self.assertEqual(1, x_tensor_info.tensor_shape.dim[0].size)


if __name__ == "__main__":
  test.main()
