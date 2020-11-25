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
# ==============================================================================
"""Tests for test_util.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.lite.python import test_util as tflite_test_util
from tensorflow.python.framework import test_util
from tensorflow.python.platform import gfile
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test


class TestUtilTest(test_util.TensorFlowTestCase):

  def testBuiltinOp(self):
    model_path = resource_loader.get_path_to_datafile('../testdata/add.bin')
    op_set = tflite_test_util.get_ops_list(gfile.GFile(model_path, 'rb').read())
    self.assertCountEqual(op_set, ['ADD'])

  def testFlexOp(self):
    model_path = resource_loader.get_path_to_datafile(
        '../testdata/softplus_flex.bin')
    op_set = tflite_test_util.get_ops_list(gfile.GFile(model_path, 'rb').read())
    self.assertCountEqual(op_set, ['FlexSoftplus'])


if __name__ == '__main__':
  test.main()
