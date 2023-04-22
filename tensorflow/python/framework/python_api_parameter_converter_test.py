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
"""Tests for tensorflow.python.framework.python_api_parameter_converter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np

from tensorflow.core.framework import types_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import _pywrap_python_api_info
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.framework._pywrap_python_api_parameter_converter import Convert
from tensorflow.python.framework._pywrap_python_tensor_converter import PythonTensorConverter
from tensorflow.python.platform import googletest
from tensorflow.python.util import nest

# pylint: disable=g-long-lambda


# Helper function to make expected output in examples more compact:
def Const(x):
  return constant_op.constant(x)


@test_util.run_all_in_graph_and_eager_modes
class PythonAPIWrapperTest(test_util.TensorFlowTestCase,
                           parameterized.TestCase):

  def setUp(self):
    context.ensure_initialized()
    super(PythonAPIWrapperTest, self).setUp()

  def makeTensorConverter(self):
    """Returns a new PythonTensorConverter with the current context."""
    return PythonTensorConverter(context.context())

  def makeApiInfoForGenOp(self, op_name, op_func):
    """Returns a PythonAPIParameterConverter for the given gen_op."""
    api_info = _pywrap_python_api_info.PythonAPIInfo(op_name)
    api_info.InitializeFromRegisteredOp(op_name)
    return api_info

  def makeApiInfoFromParamSpecs(self,
                                api_name,
                                param_names,
                                input_specs,
                                attr_specs,
                                defaults=()):
    """Returns a PythonAPIParameterConverter built from the given specs."""
    api_info = _pywrap_python_api_info.PythonAPIInfo(api_name)
    api_info.InitializeFromParamSpecs(input_specs, attr_specs, param_names,
                                      defaults)
    return api_info

  def assertParamsEqual(self, actual_params, expected_params):
    """Asserts that converted parameters have the expected values & types."""
    self.assertLen(actual_params, len(expected_params))
    for actual, expected in zip(actual_params, expected_params):
      if isinstance(expected, list):
        self.assertIsInstance(actual, list)
        self.assertLen(actual, len(expected))
        for actual_item, expected_item in zip(actual, expected):
          self.assertParamEqual(actual_item, expected_item)
      else:
        self.assertParamEqual(actual, expected)

  def assertParamEqual(self, actual, expected):
    if isinstance(actual, ops.Tensor):
      self.assertAllEqual(actual, expected)
    else:
      self.assertEqual(actual, expected)
    self.assertIs(type(actual), type(expected))

  def assertInferredEqual(self, api_info, inferred, expected):
    """Asserts that inferred attributes have the expected values."""
    inferred_type_attrs = api_info.InferredTypeAttrs()
    inferred_type_list_attrs = api_info.InferredTypeListAttrs()
    inferred_length_attrs = api_info.InferredLengthAttrs()

    self.assertLen(inferred.types, len(inferred_type_attrs))
    self.assertLen(inferred.type_lists, len(inferred_type_list_attrs))
    self.assertLen(inferred.lengths, len(inferred_length_attrs))
    actual = {}
    for i, val in enumerate(inferred.types):
      if val._type_enum == types_pb2.DT_INVALID:
        val = types_pb2.DT_INVALID
      actual[inferred_type_attrs[i]] = val
    for i, val in enumerate(inferred.type_lists):
      actual[inferred_type_list_attrs[i]] = val
    for i, val in enumerate(inferred.lengths):
      actual[inferred_length_attrs[i]] = val
    self.assertEqual(actual, expected)

  # This test constructs a PythonAPIParameterConverter for an op that expects
  # a single argument, whose value is an attribute with a specified type; and
  # then uses that converter to convert parameters and checks that the result
  # is the expected value.
  @parameterized.named_parameters([
      ("FloatFromFloat", "float", 5.0, 5.0),
      ("FloatFromInt", "float", 5, 5.0),
      ("FloatFromNumpyScalar", "float", np.array(5.0), 5.0),
      ("IntFromInt", "int", 5, 5),
      ("IntFromFloat", "int", 5.0, 5),
      ("IntFromNumpyScalar", "int", np.array(5.0), 5),
      ("StringFromBytes", "string", b"foo", b"foo"),
      ("StringFromUnicode", "string", u"foo", "foo"),
      ("BoolFromBool", "bool", True, True),
      ("TypeFromInt", "type", 1, dtypes.float32),
      ("TypeFromDType", "type", dtypes.int32, dtypes.int32),
      ("TypeFromNumpyType", "type", np.int32, dtypes.int32),
      ("ShapeFromShape", "shape", tensor_shape.as_shape([1, 2]),
       tensor_shape.as_shape([1, 2])),
      ("ShapeFromInt", "shape", 1, tensor_shape.as_shape(1)),
      ("ShapeFromNone", "shape", None, tensor_shape.as_shape(None)),
      ("ShapeFromList", "shape", [1, 2, 3], tensor_shape.as_shape([1, 2, 3])),
      ("ListOfFloat", "list(float)", [1, 2.0, np.array(3)], [1.0, 2.0, 3.0]),
      ("ListOfInt", "list(int)", [1, 2.0, np.array(3)], [1, 2, 3]),
      ("ListOfString", "list(string)", [b"foo", u"bar"], [b"foo", u"bar"]),
      ("ListOfBool", "list(bool)", [True, False, True], [True, False, True]),
      ("ListOfType", "list(type)", [1, dtypes.int32, np.int64],
       [dtypes.float32, dtypes.int32, dtypes.int64]),
      ("ListOfShape", "list(shape)", [1, None, [2, 3]], [
          tensor_shape.as_shape(1),
          tensor_shape.as_shape(None),
          tensor_shape.as_shape([2, 3])
      ]),
  ])
  def testConvertAttribute(self, attr_type, attr_val, expected):
    api_info = self.makeApiInfoFromParamSpecs("ConvertAttributes", ["x"], {},
                                              {"x": attr_type})
    tensor_converter = self.makeTensorConverter()

    params = [attr_val]
    inferred = Convert(api_info, tensor_converter, params)
    self.assertEqual(inferred.types, [])
    self.assertEqual(inferred.type_lists, [])
    self.assertEqual(inferred.lengths, [])
    self.assertLen(params, 1)
    actual = params[0]
    self.assertEqual(actual, expected)

    # Check that we got the actual types we expected.  (Note that in Python,
    # two values may be equal even if they have different types.)
    self.assertIs(type(actual), type(expected))
    if isinstance(expected, list):
      self.assertLen(actual, len(expected))
      for (actual_item, expected_item) in zip(actual, expected):
        self.assertIs(type(actual_item), type(expected_item))

  def testConvertMultipleAttributes(self):
    attr_specs = {"x": "list(int)", "y": "shape", "z": "float"}
    api_info = self.makeApiInfoFromParamSpecs("ConvertAttributes",
                                              ["x", "y", "z"], {}, attr_specs)
    tensor_converter = self.makeTensorConverter()

    params = [[1, 2.0, np.array(3.0)], [1, 2], 10]
    inferred = Convert(api_info, tensor_converter, params)

    self.assertEqual(inferred.types, [])
    self.assertEqual(inferred.type_lists, [])
    self.assertEqual(inferred.lengths, [])
    self.assertLen(params, 3)
    self.assertEqual(params, [[1, 2, 3], tensor_shape.as_shape([1, 2]), 10.0])
    self.assertIsInstance(params[0][0], int)
    self.assertIsInstance(params[1], tensor_shape.TensorShape)
    self.assertIsInstance(params[2], float)

  @parameterized.named_parameters([
      ("StringFromInt", "string", 5, "Foo argument x: Failed to convert value "
       "of type 'int' to type 'string'."),
      ("IntFromNone", "int", None, "Foo argument x: Failed to convert value "
       "of type 'NoneType' to type 'int'."),
      ("BoolFromInt", "bool", 0,
       "Foo argument x: Failed to convert value of type 'int' to type 'bool'."),
  ])
  def testConvertAttributeError(self, attr_type, attr_val, message):
    api_info = self.makeApiInfoFromParamSpecs("Foo", ["x"], {},
                                              {"x": attr_type})
    tensor_converter = self.makeTensorConverter()
    with self.assertRaisesRegex(TypeError, message):
      Convert(api_info, tensor_converter, [attr_val])

  @parameterized.named_parameters([
      dict(
          testcase_name="FixedDTypeInputs",
          param_names=["x", "y"],
          input_specs=dict(x="int32", y="float32"),
          attr_specs={},
          inputs=lambda: [1, 2],
          outputs=lambda: [Const(1), Const(2.0)],
          inferred={}),
      dict(
          testcase_name="UnconstrainedTypeInput",
          param_names=["x"],
          input_specs=dict(x="T"),
          attr_specs=dict(T="type"),
          inputs=lambda: [np.array("foo")],
          outputs=lambda: [Const("foo")],
          inferred=dict(T=dtypes.string)),
      dict(
          testcase_name="ConstrainedTypeInput",
          param_names=["x"],
          input_specs=dict(x="T"),
          attr_specs=dict(T="{int32, float, string}"),
          inputs=lambda: [np.array("foo")],
          outputs=lambda: [Const("foo")],
          inferred=dict(T=dtypes.string)),
      dict(
          testcase_name="SharedTypeInputs",
          param_names=["x", "y"],
          input_specs=dict(x="T", y="T"),
          attr_specs=dict(T="{float, int32, int64}"),
          inputs=lambda: [1, np.array(2)],
          outputs=lambda: [Const(1), Const(2)],
          inferred=dict(T=dtypes.int32)),
      dict(
          testcase_name="SharedTypeInferredFromTensor",
          param_names=["x", "y"],
          input_specs=dict(x="T", y="T"),
          attr_specs=dict(T="{float, int32, int64}"),
          inputs=lambda: [1, Const(2.0)],
          outputs=lambda: [Const(1.0), Const(2.0)],
          inferred=dict(T=dtypes.float32)),
      dict(
          # If the native converted type for an input isn't in the ok_dtypes
          # list, then we try the default dtype instead.
          testcase_name="FallbackToDefaultDtype",
          param_names=["x"],
          input_specs=dict(x="T"),
          attr_specs=dict(T="{float, string} = DT_FLOAT"),
          inputs=lambda: [1],
          outputs=lambda: [Const(1.0)],
          inferred=dict(T=dtypes.float32)),
      dict(
          testcase_name="RepeatedInput",
          param_names=["x", "y"],
          input_specs=dict(x="N * T", y="T"),
          attr_specs=dict(T="{float, int32}", N="int"),
          inputs=lambda: [[1, 2, 3], 4],
          outputs=lambda: [[Const(1), Const(2), Const(3)],
                           Const(4)],
          inferred=dict(T=dtypes.int32, N=3)),
      dict(
          testcase_name="RepeatedInputInferDTypeFromRepeated",
          param_names=["x", "y"],
          input_specs=dict(x="N * T", y="T"),
          attr_specs=dict(T="{float, int32}", N="int"),
          inputs=lambda: [[1, 2, Const(3.0)], 4],
          outputs=lambda: [[Const(1.0), Const(2.0),
                            Const(3.0)],
                           Const(4.0)],
          inferred=dict(T=dtypes.float32, N=3)),
      dict(
          testcase_name="RepeatedInputInferDTypeFromSingleton",
          param_names=["x", "y"],
          input_specs=dict(x="N * T", y="T"),
          attr_specs=dict(T="{float, int32}", N="int"),
          inputs=lambda: [[1, 2, 3], Const(4.0)],
          outputs=lambda: [[Const(1.0), Const(2.0),
                            Const(3.0)],
                           Const(4.0)],
          inferred=dict(T=dtypes.float32, N=3)),
      dict(
          testcase_name="EmptyRepeatedInput",
          param_names=["x"],
          input_specs=dict(x="N * T"),
          attr_specs=dict(T="{float, int32} = DT_INT32", N="int"),
          inputs=lambda: [[]],
          outputs=lambda: [[]],
          inferred=dict(T=dtypes.int32, N=0)),
      dict(
          testcase_name="EmptyRepeatedInputWithNoDefaultDtype",
          param_names=["x"],
          input_specs=dict(x="N * T"),
          attr_specs=dict(T="{float, int32}", N="int"),
          inputs=lambda: [[]],
          outputs=lambda: [[]],
          inferred=dict(T=types_pb2.DT_INVALID, N=0)),
      dict(
          testcase_name="RepeatedInputWithExplicitCountAndType",
          param_names=["N", "T", "x", "y"],
          input_specs=dict(x="N * T", y="T"),
          attr_specs=dict(T="{float, int32}", N="int"),
          inputs=lambda: [3, np.float32, [1, 2, 3], 4],
          outputs=lambda:
          [3, dtypes.float32, [Const(1.0), Const(2.0),
                               Const(3.0)],
           Const(4.0)],
          inferred={}),
      dict(
          testcase_name="ListOfTypes",
          param_names=["x"],
          input_specs=dict(x="T"),
          attr_specs=dict(T="list({int32, float32})"),
          inputs=lambda: [[1, 2, Const(3.0)]],
          outputs=lambda: [[Const(1), Const(2), Const(3.0)]],
          inferred=dict(T=[dtypes.int32, dtypes.int32, dtypes.float32])),
      dict(
          testcase_name="EmptyListOfTypes",
          param_names=["x"],
          input_specs=dict(x="T"),
          attr_specs=dict(T="list({int32, float32}) >= 0"),
          inputs=lambda: [[]],
          outputs=lambda: [[]],
          inferred=dict(T=[])),
      dict(
          testcase_name="MatchingListsOfTypes",
          param_names=["x", "y", "z"],
          input_specs=dict(x="T", y="T", z="T"),
          attr_specs=dict(T="list({int32, float32})"),
          inputs=lambda: [
              [1, 2, constant_op.constant(3.0)],  # x
              [constant_op.constant(4.0), 5, 6],  # y
              [7, constant_op.constant(8), 9],  # z
          ],
          outputs=lambda: nest.map_structure(
              constant_op.constant,  #
              [[1.0, 2, 3.0], [4.0, 5, 6.0], [7.0, 8, 9.0]]),
          inferred=dict(T=[dtypes.float32, dtypes.int32, dtypes.float32])),
      dict(
          testcase_name="ExplicitListOfTypes",
          param_names=["x", "T"],
          input_specs=dict(x="T"),
          attr_specs=dict(T="list({int32, float32})"),
          inputs=lambda: [[1, 2, constant_op.constant(3.0)],
                          [dtypes.int32, dtypes.float32, dtypes.float32]],
          outputs=lambda: [[
              constant_op.constant(1, dtypes.int32),
              constant_op.constant(2, dtypes.float32),
              constant_op.constant(3.0, dtypes.float32)
          ], [dtypes.int32, dtypes.float32, dtypes.float32]],
          inferred={}),
      dict(
          testcase_name="NameParam",
          param_names=["x", "y", "name"],
          input_specs=dict(x="int32", y="float32"),
          attr_specs={},
          inputs=lambda: [1, 2, "bob"],
          outputs=lambda: [
              constant_op.constant(1, dtypes.int32),
              constant_op.constant(2, dtypes.float32), "bob"
          ],
          inferred={}),
      dict(
          testcase_name="NameParamInNonstandardPosition",
          param_names=["x", "name", "y"],
          input_specs=dict(x="int32", y="float32"),
          attr_specs={},
          inputs=lambda: [1, "bob", 2],
          outputs=lambda: [
              constant_op.constant(1, dtypes.int32), "bob",
              constant_op.constant(2, dtypes.float32)
          ],
          inferred={}),
      dict(
          testcase_name="NameParamIsNotConvertedOrModified",
          param_names=["x", "y", "name"],
          input_specs=dict(x="int32", y="float32"),
          attr_specs={},
          inputs=lambda: [1, 2, {
              "foo": ["bar", "baz"]
          }],
          outputs=lambda: [
              constant_op.constant(1, dtypes.int32),
              constant_op.constant(2, dtypes.float32), {
                  "foo": ["bar", "baz"]
              }
          ],
          inferred={}),
      dict(
          # Note: there don't appear to be any real-world ops that have a
          # type(list) attr whose default value is anything other than `[]`.
          # But we test this case anyway.
          testcase_name="ListOfTypesFallbackToDefault",
          param_names=["x"],
          input_specs=dict(x="T"),
          attr_specs=dict(T="list({string, float32}) = [DT_FLOAT, DT_FLOAT]"),
          inputs=lambda: [[1, 2.0]],
          outputs=lambda: [[
              constant_op.constant(1.0, dtypes.float32),
              constant_op.constant(2.0, dtypes.float32)
          ]],
          inferred=dict(T=[dtypes.float32, dtypes.float32])),
      dict(
          testcase_name="ComplexOp",
          param_names=["a", "b", "c", "d", "e", "f", "name"],
          input_specs=dict(a="X", b="N * X", e="Y", f="Y"),
          attr_specs=dict(
              c="list(int)",
              d="string",
              N="int",
              X="type",
              Y="list({int32, string})"),
          inputs=lambda: [
              [[1, 2, 3], [4, 5, 6]],  # a
              [[1, 2], [3, 4, 5], [6]],  # b
              [1, 2, 3],  # c
              "Foo",  # d
              [[1, 2], [["three"]], [4], "five"],  # e
              [1, "two", [[3, 4], [5, 6]], [["7"]]],  # f
          ],
          outputs=lambda: [
              Const([[1, 2, 3], [4, 5, 6]]),
              [Const([1, 2]), Const([3, 4, 5]),
               Const([6])],
              [1, 2, 3],
              "Foo",
              [Const([1, 2]),
               Const([["three"]]),
               Const([4]),
               Const("five")],
              [Const(1),
               Const("two"),
               Const([[3, 4], [5, 6]]),
               Const([["7"]])],
          ],
          inferred=dict(
              N=3,
              X=dtypes.int32,
              Y=[dtypes.int32, dtypes.string, dtypes.int32, dtypes.string])),
  ])
  def testConvert(self, param_names, input_specs, attr_specs, inputs, outputs,
                  inferred):
    api_info = self.makeApiInfoFromParamSpecs("TestFunc", param_names,
                                              input_specs, attr_specs)
    tensor_converter = self.makeTensorConverter()
    param_values = inputs()
    actual_inferred = Convert(api_info, tensor_converter, param_values)
    self.assertInferredEqual(api_info, actual_inferred, inferred)
    self.assertParamsEqual(param_values, outputs())

  @parameterized.named_parameters([
      dict(
          testcase_name="WrongDTypeForFixedDTypeInput",
          param_names=["x"],
          input_specs=dict(x="float"),
          attr_specs={},
          inputs=lambda: [constant_op.constant(1)],
          message="TestFunc argument x: Expected DT_FLOAT but got DT_INT32"),
      dict(
          testcase_name="AddIntTensorAndFloatTensor",
          param_names=["x", "y"],
          input_specs=dict(x="T", y="T"),
          attr_specs=dict(T="{float, int32, int64}"),
          inputs=lambda: [constant_op.constant(1),
                          constant_op.constant(2.0)],
          message="TestFunc argument y: Expected DT_INT32 but got DT_FLOAT"),
  ])
  def testConvertError(self,
                       param_names,
                       input_specs,
                       attr_specs,
                       inputs,
                       message,
                       exception=TypeError):
    api_info = self.makeApiInfoFromParamSpecs("TestFunc", param_names,
                                              input_specs, attr_specs)
    tensor_converter = self.makeTensorConverter()
    param_values = inputs()
    with self.assertRaisesRegex(exception, message):
      Convert(api_info, tensor_converter, param_values)


if __name__ == "__main__":
  googletest.main()
