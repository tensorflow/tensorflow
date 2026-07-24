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
"""Tests for parsing ops regression cases."""

import tensorflow as tf
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class ParseTensorVariantRegressionTest(test_util.TensorFlowTestCase):

  def testMalformedVariantTensorListDoesNotCrash(self):
    """Regression test: malformed TensorList Variant must raise clean error.

    A TensorProto with dtype=DT_VARIANT wrapping a tensorflow::TensorList
    with empty metadata previously caused a SIGSEGV in PyErr_Occurred() via
    pybind11 exception translation instead of raising InvalidArgumentError.
    Confirmed in TF 2.21.0 in fresh isolated containers.

    Root cause: FromProtoField<Variant> in tensor.cc called buf->Unref()
    with data[0..i] holding partially-decoded Variant objects after
    DecodeUnaryVariant returned false, corrupting Python thread state.
    """
    from tensorflow.core.framework import tensor_pb2, types_pb2
    outer = tensor_pb2.TensorProto()
    outer.dtype = types_pb2.DT_VARIANT
    outer.tensor_shape.dim.add().size = 1
    v = outer.variant_val.add()
    v.type_name = "tensorflow::TensorList"
    leaf = v.tensors.add()
    leaf.dtype = types_pb2.DT_FLOAT
    leaf.tensor_shape.dim.add().size = 1
    serialized = outer.SerializeToString()
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(tf.io.parse_tensor(serialized, out_type=tf.variant))


if __name__ == "__main__":
  test.main()
