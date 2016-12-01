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

import tensorflow as tf

from tensorflow.core.framework import types_pb2
from tensorflow.python.saved_model import utils


class UtilsTest(tf.test.TestCase):

  def testBuildTensorInfo(self):
    x = tf.placeholder(tf.float32, 1, name="x")
    x_tensor_info = utils.build_tensor_info(x)
    self.assertEqual("x:0", x_tensor_info.name)
    self.assertEqual(types_pb2.DT_FLOAT, x_tensor_info.dtype)
    self.assertEqual(1, len(x_tensor_info.tensor_shape.dim))
    self.assertEqual(1, x_tensor_info.tensor_shape.dim[0].size)

  def testBuildSignatureDef(self):
    x = tf.placeholder(tf.float32, 1, name="x")
    x_tensor_info = utils.build_tensor_info(x)
    inputs = dict()
    inputs["foo-input"] = x_tensor_info

    y = tf.placeholder(tf.float32, name="y")
    y_tensor_info = utils.build_tensor_info(y)
    outputs = dict()
    outputs["foo-output"] = y_tensor_info

    signature_def = utils.build_signature_def(inputs, outputs,
                                              "foo-method-name")
    self.assertEqual("foo-method-name", signature_def.method_name)

    # Check inputs in signature def.
    self.assertEqual(1, len(signature_def.inputs))
    x_tensor_info_actual = signature_def.inputs["foo-input"]
    self.assertEqual("x:0", x_tensor_info_actual.name)
    self.assertEqual(types_pb2.DT_FLOAT, x_tensor_info_actual.dtype)
    self.assertEqual(1, len(x_tensor_info_actual.tensor_shape.dim))
    self.assertEqual(1, x_tensor_info_actual.tensor_shape.dim[0].size)

    # Check outputs in signature def.
    self.assertEqual(1, len(signature_def.outputs))
    y_tensor_info_actual = signature_def.outputs["foo-output"]
    self.assertEqual("y:0", y_tensor_info_actual.name)
    self.assertEqual(types_pb2.DT_FLOAT, y_tensor_info_actual.dtype)
    self.assertEqual(0, len(y_tensor_info_actual.tensor_shape.dim))


if __name__ == "__main__":
  tf.test.main()
