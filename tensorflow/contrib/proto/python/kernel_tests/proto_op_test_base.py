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
    test_case.values.add()  # No fields specified, so we get all defaults.
    test_case.shapes.append(1)
    test_case.sizes.append(0)
    field = test_case.fields.add()
    field.name = "double_value_with_default"
    field.dtype = types_pb2.DT_DOUBLE
    field.value.double_value.append(1.0)
    test_case.sizes.append(0)
    field = test_case.fields.add()
    field.name = "float_value_with_default"
    field.dtype = types_pb2.DT_FLOAT
    field.value.float_value.append(2.0)
    test_case.sizes.append(0)
    field = test_case.fields.add()
    field.name = "int64_value_with_default"
    field.dtype = types_pb2.DT_INT64
    field.value.int64_value.append(3)
    test_case.sizes.append(0)
    field = test_case.fields.add()
    field.name = "sfixed64_value_with_default"
    field.dtype = types_pb2.DT_INT64
    field.value.int64_value.append(11)
    test_case.sizes.append(0)
    field = test_case.fields.add()
    field.name = "sint64_value_with_default"
    field.dtype = types_pb2.DT_INT64
    field.value.int64_value.append(13)
    test_case.sizes.append(0)
    field = test_case.fields.add()
    field.name = "uint64_value_with_default"
    field.dtype = types_pb2.DT_INT64
    field.value.int64_value.append(4)
    test_case.sizes.append(0)
    field = test_case.fields.add()
    field.name = "fixed64_value_with_default"
    field.dtype = types_pb2.DT_INT64
    field.value.int64_value.append(6)
    test_case.sizes.append(0)
    field = test_case.fields.add()
    field.name = "int32_value_with_default"
    field.dtype = types_pb2.DT_INT32
    field.value.int32_value.append(5)
    test_case.sizes.append(0)
    field = test_case.fields.add()
    field.name = "sfixed32_value_with_default"
    field.dtype = types_pb2.DT_INT32
    field.value.int32_value.append(10)
    test_case.sizes.append(0)
    field = test_case.fields.add()
    field.name = "sint32_value_with_default"
    field.dtype = types_pb2.DT_INT32
    field.value.int32_value.append(12)
    test_case.sizes.append(0)
    field = test_case.fields.add()
    field.name = "uint32_value_with_default"
    field.dtype = types_pb2.DT_INT32
    field.value.int32_value.append(9)
    test_case.sizes.append(0)
    field = test_case.fields.add()
    field.name = "fixed32_value_with_default"
    field.dtype = types_pb2.DT_INT32
    field.value.int32_value.append(7)
    test_case.sizes.append(0)
    field = test_case.fields.add()
    field.name = "bool_value_with_default"
    field.dtype = types_pb2.DT_BOOL
    field.value.bool_value.append(True)
    test_case.sizes.append(0)
    field = test_case.fields.add()
    field.name = "string_value_with_default"
    field.dtype = types_pb2.DT_STRING
    field.value.string_value.append("a")
    test_case.sizes.append(0)
    field = test_case.fields.add()
    field.name = "bytes_value_with_default"
    field.dtype = types_pb2.DT_STRING
    field.value.string_value.append("a longer default string")
    return test_case

  @staticmethod
  def minmax_test_case():
    test_case = test_example_pb2.TestCase()
    value = test_case.values.add()
    value.double_value.append(-1.7976931348623158e+308)
    value.double_value.append(2.2250738585072014e-308)
    value.double_value.append(1.7976931348623158e+308)
    value.float_value.append(-3.402823466e+38)
    value.float_value.append(1.175494351e-38)
    value.float_value.append(3.402823466e+38)
    value.int64_value.append(-9223372036854775808)
    value.int64_value.append(9223372036854775807)
    value.sfixed64_value.append(-9223372036854775808)
    value.sfixed64_value.append(9223372036854775807)
    value.sint64_value.append(-9223372036854775808)
    value.sint64_value.append(9223372036854775807)
    value.uint64_value.append(0)
    value.uint64_value.append(18446744073709551615)
    value.fixed64_value.append(0)
    value.fixed64_value.append(18446744073709551615)
    value.int32_value.append(-2147483648)
    value.int32_value.append(2147483647)
    value.sfixed32_value.append(-2147483648)
    value.sfixed32_value.append(2147483647)
    value.sint32_value.append(-2147483648)
    value.sint32_value.append(2147483647)
    value.uint32_value.append(0)
    value.uint32_value.append(4294967295)
    value.fixed32_value.append(0)
    value.fixed32_value.append(4294967295)
    value.bool_value.append(False)
    value.bool_value.append(True)
    value.string_value.append("")
    value.string_value.append("I refer to the infinite.")
    test_case.shapes.append(1)
    test_case.sizes.append(3)
    field = test_case.fields.add()
    field.name = "double_value"
    field.dtype = types_pb2.DT_DOUBLE
    field.value.double_value.append(-1.7976931348623158e+308)
    field.value.double_value.append(2.2250738585072014e-308)
    field.value.double_value.append(1.7976931348623158e+308)
    test_case.sizes.append(3)
    field = test_case.fields.add()
    field.name = "float_value"
    field.dtype = types_pb2.DT_FLOAT
    field.value.float_value.append(-3.402823466e+38)
    field.value.float_value.append(1.175494351e-38)
    field.value.float_value.append(3.402823466e+38)
    test_case.sizes.append(2)
    field = test_case.fields.add()
    field.name = "int64_value"
    field.dtype = types_pb2.DT_INT64
    field.value.int64_value.append(-9223372036854775808)
    field.value.int64_value.append(9223372036854775807)
    test_case.sizes.append(2)
    field = test_case.fields.add()
    field.name = "sfixed64_value"
    field.dtype = types_pb2.DT_INT64
    field.value.int64_value.append(-9223372036854775808)
    field.value.int64_value.append(9223372036854775807)
    test_case.sizes.append(2)
    field = test_case.fields.add()
    field.name = "sint64_value"
    field.dtype = types_pb2.DT_INT64
    field.value.int64_value.append(-9223372036854775808)
    field.value.int64_value.append(9223372036854775807)
    test_case.sizes.append(2)
    field = test_case.fields.add()
    field.name = "uint64_value"
    field.dtype = types_pb2.DT_INT64
    field.value.int64_value.append(0)
    field.value.int64_value.append(-1)
    test_case.sizes.append(2)
    field = test_case.fields.add()
    field.name = "fixed64_value"
    field.dtype = types_pb2.DT_INT64
    field.value.int64_value.append(0)
    field.value.int64_value.append(-1)
    test_case.sizes.append(2)
    field = test_case.fields.add()
    field.name = "int32_value"
    field.dtype = types_pb2.DT_INT32
    field.value.int32_value.append(-2147483648)
    field.value.int32_value.append(2147483647)
    test_case.sizes.append(2)
    field = test_case.fields.add()
    field.name = "sfixed32_value"
    field.dtype = types_pb2.DT_INT32
    field.value.int32_value.append(-2147483648)
    field.value.int32_value.append(2147483647)
    test_case.sizes.append(2)
    field = test_case.fields.add()
    field.name = "sint32_value"
    field.dtype = types_pb2.DT_INT32
    field.value.int32_value.append(-2147483648)
    field.value.int32_value.append(2147483647)
    test_case.sizes.append(2)
    field = test_case.fields.add()
    field.name = "uint32_value"
    field.dtype = types_pb2.DT_INT32
    field.value.int32_value.append(0)
    field.value.int32_value.append(-1)
    test_case.sizes.append(2)
    field = test_case.fields.add()
    field.name = "fixed32_value"
    field.dtype = types_pb2.DT_INT32
    field.value.int32_value.append(0)
    field.value.int32_value.append(-1)
    test_case.sizes.append(2)
    field = test_case.fields.add()
    field.name = "bool_value"
    field.dtype = types_pb2.DT_BOOL
    field.value.bool_value.append(False)
    field.value.bool_value.append(True)
    test_case.sizes.append(2)
    field = test_case.fields.add()
    field.name = "string_value"
    field.dtype = types_pb2.DT_STRING
    field.value.string_value.append("")
    field.value.string_value.append("I refer to the infinite.")
    return test_case

  @staticmethod
  def nested_test_case():
    test_case = test_example_pb2.TestCase()
    value = test_case.values.add()
    message_value = value.message_value.add()
    message_value.double_value = 23.5
    test_case.shapes.append(1)
    test_case.sizes.append(1)
    field = test_case.fields.add()
    field.name = "message_value"
    field.dtype = types_pb2.DT_STRING
    message_value = field.value.message_value.add()
    message_value.double_value = 23.5
    return test_case

  @staticmethod
  def optional_test_case():
    test_case = test_example_pb2.TestCase()
    value = test_case.values.add()
    value.bool_value.append(True)
    test_case.shapes.append(1)
    test_case.sizes.append(1)
    field = test_case.fields.add()
    field.name = "bool_value"
    field.dtype = types_pb2.DT_BOOL
    field.value.bool_value.append(True)
    test_case.sizes.append(0)
    field = test_case.fields.add()
    field.name = "double_value"
    field.dtype = types_pb2.DT_DOUBLE
    field.value.double_value.append(0.0)
    return test_case

  @staticmethod
  def promote_unsigned_test_case():
    test_case = test_example_pb2.TestCase()
    value = test_case.values.add()
    value.fixed32_value.append(4294967295)
    value.uint32_value.append(4294967295)
    test_case.shapes.append(1)
    test_case.sizes.append(1)
    field = test_case.fields.add()
    field.name = "fixed32_value"
    field.dtype = types_pb2.DT_INT64
    field.value.int64_value.append(4294967295)
    test_case.sizes.append(1)
    field = test_case.fields.add()
    field.name = "uint32_value"
    field.dtype = types_pb2.DT_INT64
    field.value.int64_value.append(4294967295)
    # Comes from an explicitly-specified default
    test_case.sizes.append(0)
    field = test_case.fields.add()
    field.name = "uint32_value_with_default"
    field.dtype = types_pb2.DT_INT64
    field.value.int64_value.append(9)
    return test_case

  @staticmethod
  def ragged_test_case():
    test_case = test_example_pb2.TestCase()
    value = test_case.values.add()
    value.double_value.append(23.5)
    value.double_value.append(123.0)
    value.bool_value.append(True)
    value = test_case.values.add()
    value.double_value.append(3.1)
    value.bool_value.append(False)
    test_case.shapes.append(2)
    test_case.sizes.append(2)
    test_case.sizes.append(1)
    test_case.sizes.append(1)
    test_case.sizes.append(1)
    field = test_case.fields.add()
    field.name = "double_value"
    field.dtype = types_pb2.DT_DOUBLE
    field.value.double_value.append(23.5)
    field.value.double_value.append(123.0)
    field.value.double_value.append(3.1)
    field.value.double_value.append(0.0)
    field = test_case.fields.add()
    field.name = "bool_value"
    field.dtype = types_pb2.DT_BOOL
    field.value.bool_value.append(True)
    field.value.bool_value.append(False)
    return test_case

  @staticmethod
  def shaped_batch_test_case():
    test_case = test_example_pb2.TestCase()
    value = test_case.values.add()
    value.double_value.append(23.5)
    value.bool_value.append(True)
    value = test_case.values.add()
    value.double_value.append(44.0)
    value.bool_value.append(False)
    value = test_case.values.add()
    value.double_value.append(3.14159)
    value.bool_value.append(True)
    value = test_case.values.add()
    value.double_value.append(1.414)
    value.bool_value.append(True)
    value = test_case.values.add()
    value.double_value.append(-32.2)
    value.bool_value.append(False)
    value = test_case.values.add()
    value.double_value.append(0.0001)
    value.bool_value.append(True)
    test_case.shapes.append(3)
    test_case.shapes.append(2)
    for _ in range(12):
      test_case.sizes.append(1)
    field = test_case.fields.add()
    field.name = "double_value"
    field.dtype = types_pb2.DT_DOUBLE
    field.value.double_value.append(23.5)
    field.value.double_value.append(44.0)
    field.value.double_value.append(3.14159)
    field.value.double_value.append(1.414)
    field.value.double_value.append(-32.2)
    field.value.double_value.append(0.0001)
    field = test_case.fields.add()
    field.name = "bool_value"
    field.dtype = types_pb2.DT_BOOL
    field.value.bool_value.append(True)
    field.value.bool_value.append(False)
    field.value.bool_value.append(True)
    field.value.bool_value.append(True)
    field.value.bool_value.append(False)
    field.value.bool_value.append(True)
    return test_case

  @staticmethod
  def simple_test_case():
    test_case = test_example_pb2.TestCase()
    value = test_case.values.add()
    value.double_value.append(23.5)
    value.bool_value.append(True)
    test_case.shapes.append(1)
    test_case.sizes.append(1)
    field = test_case.fields.add()
    field.name = "double_value"
    field.dtype = types_pb2.DT_DOUBLE
    field.value.double_value.append(23.5)
    test_case.sizes.append(1)
    field = test_case.fields.add()
    field.name = "bool_value"
    field.dtype = types_pb2.DT_BOOL
    field.value.bool_value.append(True)
    return test_case
