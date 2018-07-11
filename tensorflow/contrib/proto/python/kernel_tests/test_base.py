# =============================================================================
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
# =============================================================================
"""Test case base for testing proto operations."""

# Python3 preparedness imports.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ctypes as ct
import os

from tensorflow.contrib.proto.python.kernel_tests import test_example_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python.platform import test


class ProtoOpTestBase(test.TestCase):
  """Base class for testing proto decoding and encoding ops."""

  def __init__(self, methodName="runTest"):  # pylint: disable=invalid-name
    super(ProtoOpTestBase, self).__init__(methodName)
    lib = os.path.join(os.path.dirname(__file__), "libtestexample.so")
    if os.path.isfile(lib):
      ct.cdll.LoadLibrary(lib)

  @staticmethod
  def named_parameters():
    return (
        ("defaults", ProtoOpTestBase.defaults_test_case()),
        ("minmax", ProtoOpTestBase.minmax_test_case()),
        ("nested", ProtoOpTestBase.nested_test_case()),
        ("optional", ProtoOpTestBase.optional_test_case()),
        ("promote_unsigned", ProtoOpTestBase.promote_unsigned_test_case()),
        ("ragged", ProtoOpTestBase.ragged_test_case()),
        ("shaped_batch", ProtoOpTestBase.shaped_batch_test_case()),
        ("simple", ProtoOpTestBase.simple_test_case()),
    )

  @staticmethod
  def defaults_test_case():
    test_case = test_example_pb2.TestCase()
    test_case.primitive.add()  # No fields specified, so we get all defaults.
    test_case.shape.append(1)
    test_case.sizes.append(0)
    field = test_case.field.add()
    field.name = "double_default"
    field.dtype = types_pb2.DT_DOUBLE
    field.expected.double_value.append(1.0)
    test_case.sizes.append(0)
    field = test_case.field.add()
    field.name = "float_default"
    field.dtype = types_pb2.DT_FLOAT
    field.expected.float_value.append(2.0)
    test_case.sizes.append(0)
    field = test_case.field.add()
    field.name = "int64_default"
    field.dtype = types_pb2.DT_INT64
    field.expected.int64_value.append(3)
    test_case.sizes.append(0)
    field = test_case.field.add()
    field.name = "sfixed64_default"
    field.dtype = types_pb2.DT_INT64
    field.expected.int64_value.append(11)
    test_case.sizes.append(0)
    field = test_case.field.add()
    field.name = "sint64_default"
    field.dtype = types_pb2.DT_INT64
    field.expected.int64_value.append(13)
    test_case.sizes.append(0)
    field = test_case.field.add()
    field.name = "uint64_default"
    field.dtype = types_pb2.DT_INT64
    field.expected.int64_value.append(4)
    test_case.sizes.append(0)
    field = test_case.field.add()
    field.name = "fixed64_default"
    field.dtype = types_pb2.DT_INT64
    field.expected.int64_value.append(6)
    test_case.sizes.append(0)
    field = test_case.field.add()
    field.name = "int32_default"
    field.dtype = types_pb2.DT_INT32
    field.expected.int32_value.append(5)
    test_case.sizes.append(0)
    field = test_case.field.add()
    field.name = "sfixed32_default"
    field.dtype = types_pb2.DT_INT32
    field.expected.int32_value.append(10)
    test_case.sizes.append(0)
    field = test_case.field.add()
    field.name = "sint32_default"
    field.dtype = types_pb2.DT_INT32
    field.expected.int32_value.append(12)
    test_case.sizes.append(0)
    field = test_case.field.add()
    field.name = "uint32_default"
    field.dtype = types_pb2.DT_INT32
    field.expected.int32_value.append(-1)
    test_case.sizes.append(0)
    field = test_case.field.add()
    field.name = "fixed32_default"
    field.dtype = types_pb2.DT_INT32
    field.expected.int32_value.append(7)
    test_case.sizes.append(0)
    field = test_case.field.add()
    field.name = "bool_default"
    field.dtype = types_pb2.DT_BOOL
    field.expected.bool_value.append(True)
    test_case.sizes.append(0)
    field = test_case.field.add()
    field.name = "string_default"
    field.dtype = types_pb2.DT_STRING
    field.expected.string_value.append("a")
    test_case.sizes.append(0)
    field = test_case.field.add()
    field.name = "bytes_default"
    field.dtype = types_pb2.DT_STRING
    field.expected.string_value.append("a longer default string")
    return test_case

  @staticmethod
  def minmax_test_case():
    test_case = test_example_pb2.TestCase()
    primitive = test_case.primitive.add()
    primitive.double_value.append(-1.7976931348623158e+308)
    primitive.double_value.append(2.2250738585072014e-308)
    primitive.double_value.append(1.7976931348623158e+308)
    primitive.float_value.append(-3.402823466e+38)
    primitive.float_value.append(1.175494351e-38)
    primitive.float_value.append(3.402823466e+38)
    primitive.int64_value.append(-9223372036854775808)
    primitive.int64_value.append(9223372036854775807)
    primitive.sfixed64_value.append(-9223372036854775808)
    primitive.sfixed64_value.append(9223372036854775807)
    primitive.sint64_value.append(-9223372036854775808)
    primitive.sint64_value.append(9223372036854775807)
    primitive.uint64_value.append(0)
    primitive.uint64_value.append(18446744073709551615)
    primitive.fixed64_value.append(0)
    primitive.fixed64_value.append(18446744073709551615)
    primitive.int32_value.append(-2147483648)
    primitive.int32_value.append(2147483647)
    primitive.sfixed32_value.append(-2147483648)
    primitive.sfixed32_value.append(2147483647)
    primitive.sint32_value.append(-2147483648)
    primitive.sint32_value.append(2147483647)
    primitive.uint32_value.append(0)
    primitive.uint32_value.append(4294967295)
    primitive.fixed32_value.append(0)
    primitive.fixed32_value.append(4294967295)
    primitive.bool_value.append(False)
    primitive.bool_value.append(True)
    primitive.string_value.append("")
    primitive.string_value.append("I refer to the infinite.")
    test_case.shape.append(1)
    test_case.sizes.append(3)
    field = test_case.field.add()
    field.name = "double_value"
    field.dtype = types_pb2.DT_DOUBLE
    field.expected.double_value.append(-1.7976931348623158e+308)
    field.expected.double_value.append(2.2250738585072014e-308)
    field.expected.double_value.append(1.7976931348623158e+308)
    test_case.sizes.append(3)
    field = test_case.field.add()
    field.name = "float_value"
    field.dtype = types_pb2.DT_FLOAT
    field.expected.float_value.append(-3.402823466e+38)
    field.expected.float_value.append(1.175494351e-38)
    field.expected.float_value.append(3.402823466e+38)
    test_case.sizes.append(2)
    field = test_case.field.add()
    field.name = "int64_value"
    field.dtype = types_pb2.DT_INT64
    field.expected.int64_value.append(-9223372036854775808)
    field.expected.int64_value.append(9223372036854775807)
    test_case.sizes.append(2)
    field = test_case.field.add()
    field.name = "sfixed64_value"
    field.dtype = types_pb2.DT_INT64
    field.expected.int64_value.append(-9223372036854775808)
    field.expected.int64_value.append(9223372036854775807)
    test_case.sizes.append(2)
    field = test_case.field.add()
    field.name = "sint64_value"
    field.dtype = types_pb2.DT_INT64
    field.expected.int64_value.append(-9223372036854775808)
    field.expected.int64_value.append(9223372036854775807)
    test_case.sizes.append(2)
    field = test_case.field.add()
    field.name = "uint64_value"
    field.dtype = types_pb2.DT_INT64
    field.expected.int64_value.append(0)
    field.expected.int64_value.append(-1)
    test_case.sizes.append(2)
    field = test_case.field.add()
    field.name = "fixed64_value"
    field.dtype = types_pb2.DT_INT64
    field.expected.int64_value.append(0)
    field.expected.int64_value.append(-1)
    test_case.sizes.append(2)
    field = test_case.field.add()
    field.name = "int32_value"
    field.dtype = types_pb2.DT_INT32
    field.expected.int32_value.append(-2147483648)
    field.expected.int32_value.append(2147483647)
    test_case.sizes.append(2)
    field = test_case.field.add()
    field.name = "sfixed32_value"
    field.dtype = types_pb2.DT_INT32
    field.expected.int32_value.append(-2147483648)
    field.expected.int32_value.append(2147483647)
    test_case.sizes.append(2)
    field = test_case.field.add()
    field.name = "sint32_value"
    field.dtype = types_pb2.DT_INT32
    field.expected.int32_value.append(-2147483648)
    field.expected.int32_value.append(2147483647)
    test_case.sizes.append(2)
    field = test_case.field.add()
    field.name = "uint32_value"
    field.dtype = types_pb2.DT_INT32
    field.expected.int32_value.append(0)
    field.expected.int32_value.append(-1)
    test_case.sizes.append(2)
    field = test_case.field.add()
    field.name = "fixed32_value"
    field.dtype = types_pb2.DT_INT32
    field.expected.int32_value.append(0)
    field.expected.int32_value.append(-1)
    test_case.sizes.append(2)
    field = test_case.field.add()
    field.name = "bool_value"
    field.dtype = types_pb2.DT_BOOL
    field.expected.bool_value.append(False)
    field.expected.bool_value.append(True)
    test_case.sizes.append(2)
    field = test_case.field.add()
    field.name = "string_value"
    field.dtype = types_pb2.DT_STRING
    field.expected.string_value.append("")
    field.expected.string_value.append("I refer to the infinite.")
    return test_case

  @staticmethod
  def nested_test_case():
    test_case = test_example_pb2.TestCase()
    primitive = test_case.primitive.add()
    message_value = primitive.message_value.add()
    message_value.double_value = 23.5
    test_case.shape.append(1)
    test_case.sizes.append(1)
    field = test_case.field.add()
    field.name = "message_value"
    field.dtype = types_pb2.DT_STRING
    message_value = field.expected.message_value.add()
    message_value.double_value = 23.5
    return test_case

  @staticmethod
  def optional_test_case():
    test_case = test_example_pb2.TestCase()
    primitive = test_case.primitive.add()
    primitive.bool_value.append(True)
    test_case.shape.append(1)
    test_case.sizes.append(1)
    field = test_case.field.add()
    field.name = "bool_value"
    field.dtype = types_pb2.DT_BOOL
    field.expected.bool_value.append(True)
    test_case.sizes.append(0)
    field = test_case.field.add()
    field.name = "double_value"
    field.dtype = types_pb2.DT_DOUBLE
    field.expected.double_value.append(0.0)
    return test_case

  @staticmethod
  def promote_unsigned_test_case():
    test_case = test_example_pb2.TestCase()
    primitive = test_case.primitive.add()
    primitive.fixed32_value.append(4294967295)
    primitive.uint32_value.append(4294967295)
    test_case.shape.append(1)
    test_case.sizes.append(1)
    field = test_case.field.add()
    field.name = "fixed32_value"
    field.dtype = types_pb2.DT_INT64
    field.expected.int64_value.append(4294967295)
    test_case.sizes.append(1)
    field = test_case.field.add()
    field.name = "uint32_value"
    field.dtype = types_pb2.DT_INT64
    field.expected.int64_value.append(4294967295)
    # Comes from an explicitly-specified default
    test_case.sizes.append(0)
    field = test_case.field.add()
    field.name = "uint32_default"
    field.dtype = types_pb2.DT_INT64
    field.expected.int64_value.append(4294967295)
    return test_case

  @staticmethod
  def ragged_test_case():
    test_case = test_example_pb2.TestCase()
    primitive = test_case.primitive.add()
    primitive.double_value.append(23.5)
    primitive.double_value.append(123.0)
    primitive.bool_value.append(True)
    primitive = test_case.primitive.add()
    primitive.double_value.append(3.1)
    primitive.bool_value.append(False)
    test_case.shape.append(2)
    test_case.sizes.append(2)
    test_case.sizes.append(1)
    test_case.sizes.append(1)
    test_case.sizes.append(1)
    field = test_case.field.add()
    field.name = "double_value"
    field.dtype = types_pb2.DT_DOUBLE
    field.expected.double_value.append(23.5)
    field.expected.double_value.append(123.0)
    field.expected.double_value.append(3.1)
    field.expected.double_value.append(0.0)
    field = test_case.field.add()
    field.name = "bool_value"
    field.dtype = types_pb2.DT_BOOL
    field.expected.bool_value.append(True)
    field.expected.bool_value.append(False)
    return test_case

  @staticmethod
  def shaped_batch_test_case():
    test_case = test_example_pb2.TestCase()
    primitive = test_case.primitive.add()
    primitive.double_value.append(23.5)
    primitive.bool_value.append(True)
    primitive = test_case.primitive.add()
    primitive.double_value.append(44.0)
    primitive.bool_value.append(False)
    primitive = test_case.primitive.add()
    primitive.double_value.append(3.14159)
    primitive.bool_value.append(True)
    primitive = test_case.primitive.add()
    primitive.double_value.append(1.414)
    primitive.bool_value.append(True)
    primitive = test_case.primitive.add()
    primitive.double_value.append(-32.2)
    primitive.bool_value.append(False)
    primitive = test_case.primitive.add()
    primitive.double_value.append(0.0001)
    primitive.bool_value.append(True)
    test_case.shape.append(3)
    test_case.shape.append(2)
    for _ in range(12):
      test_case.sizes.append(1)
    field = test_case.field.add()
    field.name = "double_value"
    field.dtype = types_pb2.DT_DOUBLE
    field.expected.double_value.append(23.5)
    field.expected.double_value.append(44.0)
    field.expected.double_value.append(3.14159)
    field.expected.double_value.append(1.414)
    field.expected.double_value.append(-32.2)
    field.expected.double_value.append(0.0001)
    field = test_case.field.add()
    field.name = "bool_value"
    field.dtype = types_pb2.DT_BOOL
    field.expected.bool_value.append(True)
    field.expected.bool_value.append(False)
    field.expected.bool_value.append(True)
    field.expected.bool_value.append(True)
    field.expected.bool_value.append(False)
    field.expected.bool_value.append(True)
    return test_case

  @staticmethod
  def simple_test_case():
    test_case = test_example_pb2.TestCase()
    primitive = test_case.primitive.add()
    primitive.double_value.append(23.5)
    primitive.bool_value.append(True)
    test_case.shape.append(1)
    test_case.sizes.append(1)
    field = test_case.field.add()
    field.name = "double_value"
    field.dtype = types_pb2.DT_DOUBLE
    field.expected.double_value.append(23.5)
    test_case.sizes.append(1)
    field = test_case.field.add()
    field.name = "bool_value"
    field.dtype = types_pb2.DT_BOOL
    field.expected.bool_value.append(True)
    return test_case
