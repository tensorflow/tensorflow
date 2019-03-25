# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for util.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.lite.python import lite_constants
from tensorflow.lite.python import util
from tensorflow.lite.toco import types_pb2 as _types_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


# TODO(nupurgarg): Add test for Grappler and frozen graph related functions.
@test_util.run_v1_only("")
class UtilTest(test_util.TensorFlowTestCase):

  def testConvertDtype(self):
    self.assertEqual(
        util.convert_dtype_to_tflite_type(lite_constants.FLOAT),
        _types_pb2.FLOAT)
    self.assertEqual(
        util.convert_dtype_to_tflite_type(dtypes.float32), _types_pb2.FLOAT)
    self.assertEqual(
        util.convert_dtype_to_tflite_type(dtypes.int32), _types_pb2.INT32)
    self.assertEqual(
        util.convert_dtype_to_tflite_type(dtypes.int64), _types_pb2.INT64)
    self.assertEqual(
        util.convert_dtype_to_tflite_type(dtypes.string), _types_pb2.STRING)
    self.assertEqual(
        util.convert_dtype_to_tflite_type(dtypes.uint8),
        _types_pb2.QUANTIZED_UINT8)
    self.assertEqual(
        util.convert_dtype_to_tflite_type(dtypes.complex64),
        _types_pb2.COMPLEX64)
    with self.assertRaises(ValueError):
      util.convert_dtype_to_tflite_type(dtypes.bool)

  def testTensorName(self):
    in_tensor = array_ops.placeholder(shape=[4], dtype=dtypes.float32)
    # out_tensors should have names: "split:0", "split:1", "split:2", "split:3".
    out_tensors = array_ops.split(
        value=in_tensor, num_or_size_splits=[1, 1, 1, 1], axis=0)
    expect_names = ["split", "split:1", "split:2", "split:3"]

    for i in range(len(expect_names)):
      got_name = util.get_tensor_name(out_tensors[i])
      self.assertEqual(got_name, expect_names[i])


@test_util.run_v1_only("")
class TensorFunctionsTest(test_util.TensorFlowTestCase):

  @test_util.run_v1_only("b/120545219")
  def testGetTensorsValid(self):
    in_tensor = array_ops.placeholder(
        shape=[1, 16, 16, 3], dtype=dtypes.float32)
    _ = in_tensor + in_tensor
    sess = session.Session()

    tensors = util.get_tensors_from_tensor_names(sess.graph, ["Placeholder"])
    self.assertEqual("Placeholder:0", tensors[0].name)

  @test_util.run_v1_only("b/120545219")
  def testGetTensorsInvalid(self):
    in_tensor = array_ops.placeholder(
        shape=[1, 16, 16, 3], dtype=dtypes.float32)
    _ = in_tensor + in_tensor
    sess = session.Session()

    with self.assertRaises(ValueError) as error:
      util.get_tensors_from_tensor_names(sess.graph, ["invalid-input"])
    self.assertEqual("Invalid tensors 'invalid-input' were found.",
                     str(error.exception))

  @test_util.run_v1_only("b/120545219")
  def testSetTensorShapeValid(self):
    tensor = array_ops.placeholder(shape=[None, 3, 5], dtype=dtypes.float32)
    self.assertEqual([None, 3, 5], tensor.shape.as_list())

    util.set_tensor_shapes([tensor], {"Placeholder": [5, 3, 5]})
    self.assertEqual([5, 3, 5], tensor.shape.as_list())

  @test_util.run_v1_only("b/120545219")
  def testSetTensorShapeNoneValid(self):
    tensor = array_ops.placeholder(dtype=dtypes.float32)
    self.assertEqual(None, tensor.shape)

    util.set_tensor_shapes([tensor], {"Placeholder": [1, 3, 5]})
    self.assertEqual([1, 3, 5], tensor.shape.as_list())

  @test_util.run_v1_only("b/120545219")
  def testSetTensorShapeArrayInvalid(self):
    # Tests set_tensor_shape where the tensor name passed in doesn't exist.
    tensor = array_ops.placeholder(shape=[None, 3, 5], dtype=dtypes.float32)
    self.assertEqual([None, 3, 5], tensor.shape.as_list())

    with self.assertRaises(ValueError) as error:
      util.set_tensor_shapes([tensor], {"invalid-input": [5, 3, 5]})
    self.assertEqual(
        "Invalid tensor 'invalid-input' found in tensor shapes map.",
        str(error.exception))
    self.assertEqual([None, 3, 5], tensor.shape.as_list())

  @test_util.run_deprecated_v1
  def testSetTensorShapeDimensionInvalid(self):
    # Tests set_tensor_shape where the shape passed in is incompatiable.
    tensor = array_ops.placeholder(shape=[None, 3, 5], dtype=dtypes.float32)
    self.assertEqual([None, 3, 5], tensor.shape.as_list())

    with self.assertRaises(ValueError) as error:
      util.set_tensor_shapes([tensor], {"Placeholder": [1, 5, 5]})
    self.assertIn("The shape of tensor 'Placeholder' cannot be changed",
                  str(error.exception))
    self.assertEqual([None, 3, 5], tensor.shape.as_list())

  @test_util.run_v1_only("b/120545219")
  def testSetTensorShapeEmpty(self):
    tensor = array_ops.placeholder(shape=[None, 3, 5], dtype=dtypes.float32)
    self.assertEqual([None, 3, 5], tensor.shape.as_list())

    util.set_tensor_shapes([tensor], {})
    self.assertEqual([None, 3, 5], tensor.shape.as_list())


if __name__ == "__main__":
  test.main()
