/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op_def_builder.h"

#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

static void CanonicalizeAttrTypeListOrder(OpDef* def) {
  for (int i = 0; i < def->attr_size(); i++) {
    AttrValue* a = def->mutable_attr(i)->mutable_allowed_values();
    std::sort(a->mutable_list()->mutable_type()->begin(),
              a->mutable_list()->mutable_type()->end());
  }
}

class OpDefBuilderTest : public ::testing::Test {
 protected:
  OpDefBuilder b() { return OpDefBuilder("Test"); }

  void ExpectSuccess(const OpDefBuilder& builder, StringPiece proto) {
    OpDef op_def;
    Status status = builder.Finalize(&op_def);
    TF_EXPECT_OK(status);
    if (status.ok()) {
      OpDef expected;
      protobuf::TextFormat::ParseFromString(
          strings::StrCat("name: 'Test' ", proto), &expected);
      // Allow different orderings
      CanonicalizeAttrTypeListOrder(&op_def);
      CanonicalizeAttrTypeListOrder(&expected);
      EXPECT_EQ(op_def.ShortDebugString(), expected.ShortDebugString());
    }
  }

  void ExpectOrdered(const OpDefBuilder& builder, StringPiece proto) {
    OpDef op_def;
    Status status = builder.Finalize(&op_def);
    TF_EXPECT_OK(status);
    if (status.ok()) {
      OpDef expected;
      protobuf::TextFormat::ParseFromString(
          strings::StrCat("name: 'Test' ", proto), &expected);
      EXPECT_EQ(op_def.ShortDebugString(), expected.ShortDebugString());
    }
  }

  void ExpectFailure(const OpDefBuilder& builder, string error) {
    OpDef op_def;
    Status status = builder.Finalize(&op_def);
    EXPECT_FALSE(status.ok());
    if (!status.ok()) {
      EXPECT_EQ(status.error_message(), error);
    }
  }
};

TEST_F(OpDefBuilderTest, Attr) {
  ExpectSuccess(b().Attr("a:string"), "attr: { name: 'a' type: 'string' }");
  ExpectSuccess(b().Attr("A: int"), "attr: { name: 'A' type: 'int' }");
  ExpectSuccess(b().Attr("a1 :float"), "attr: { name: 'a1' type: 'float' }");
  ExpectSuccess(b().Attr("a_a : bool"), "attr: { name: 'a_a' type: 'bool' }");
  ExpectSuccess(b().Attr("aB  :  type"), "attr: { name: 'aB' type: 'type' }");
  ExpectSuccess(b().Attr("aB_3\t: shape"),
                "attr: { name: 'aB_3' type: 'shape' }");
  ExpectSuccess(b().Attr("t: tensor"), "attr: { name: 't' type: 'tensor' }");
  ExpectSuccess(b().Attr("XYZ\t:\tlist(type)"),
                "attr: { name: 'XYZ' type: 'list(type)' }");
  ExpectSuccess(b().Attr("f: func"), "attr { name: 'f' type: 'func'}");
}

TEST_F(OpDefBuilderTest, AttrFailure) {
  ExpectFailure(
      b().Attr("_:string"),
      "Trouble parsing '<name>:' from Attr(\"_:string\") for Op Test");
  ExpectFailure(
      b().Attr("9:string"),
      "Trouble parsing '<name>:' from Attr(\"9:string\") for Op Test");
  ExpectFailure(b().Attr(":string"),
                "Trouble parsing '<name>:' from Attr(\":string\") for Op Test");
  ExpectFailure(b().Attr("string"),
                "Trouble parsing '<name>:' from Attr(\"string\") for Op Test");
  ExpectFailure(b().Attr("a:invalid"),
                "Trouble parsing type string at 'invalid' from "
                "Attr(\"a:invalid\") for Op Test");
  ExpectFailure(
      b().Attr("b:"),
      "Trouble parsing type string at '' from Attr(\"b:\") for Op Test");
}

TEST_F(OpDefBuilderTest, AttrWithRestrictions) {
  // Types with restrictions.
  ExpectSuccess(b().Attr("a:numbertype"),
                "attr: { name: 'a' type: 'type' allowed_values { list { type: "
                "[DT_FLOAT, DT_DOUBLE, DT_INT64, DT_INT32, DT_UINT8, DT_INT16, "
                "DT_UINT16, DT_INT8, DT_COMPLEX64, DT_QINT8, DT_QUINT8, "
                "DT_QINT32] } } }");
  ExpectSuccess(b().Attr("a:realnumbertype"),
                "attr: { name: 'a' type: 'type' allowed_values { list { type: "
                "[DT_FLOAT, DT_DOUBLE, DT_INT64, DT_INT32, DT_UINT8, DT_INT16, "
                "DT_UINT16, DT_INT8] } } }");
  ExpectSuccess(b().Attr("a:quantizedtype"),
                "attr: { name: 'a' type: 'type' allowed_values { list { type: "
                "[DT_QINT8, DT_QUINT8, DT_QINT32, DT_QINT16, DT_QUINT16]} } }");
  ExpectSuccess(b().Attr("a:{string,int32}"),
                "attr: { name: 'a' type: 'type' allowed_values { list { type: "
                "[DT_STRING, DT_INT32] } } }");
  ExpectSuccess(b().Attr("a: { float , complex64 } "),
                "attr: { name: 'a' type: 'type' allowed_values { list { type: "
                "[DT_FLOAT, DT_COMPLEX64] } } }");
  ExpectSuccess(b().Attr("a: {float, complex64,} "),
                "attr: { name: 'a' type: 'type' allowed_values { list { type: "
                "[DT_FLOAT, DT_COMPLEX64] } }");
  ExpectSuccess(b().Attr(R"(a: { "X", "yz" })"),
                "attr: { name: 'a' type: 'string' allowed_values { list { s: "
                "['X', 'yz'] } } }");
  ExpectSuccess(b().Attr(R"(a: { "X", "yz", })"),
                "attr: { name: 'a' type: 'string' allowed_values { list { s: "
                "['X', 'yz'] } } }");
  ExpectSuccess(
      b().Attr("i: int >= -5"),
      "attr: { name: 'i' type: 'int' has_minimum: true minimum: -5 }");
  ExpectSuccess(b().Attr("i: int >= 9223372036854775807"),
                ("attr: { name: 'i' type: 'int' has_minimum: true "
                 "minimum: 9223372036854775807 }"));
  ExpectSuccess(b().Attr("i: int >= -9223372036854775808"),
                ("attr: { name: 'i' type: 'int' has_minimum: true "
                 "minimum: -9223372036854775808 }"));
}

TEST_F(OpDefBuilderTest, AttrRestrictionFailure) {
  ExpectFailure(
      b().Attr("a:{}"),
      "Trouble parsing type string at '}' from Attr(\"a:{}\") for Op Test");
  ExpectFailure(
      b().Attr("a:{,}"),
      "Trouble parsing type string at ',}' from Attr(\"a:{,}\") for Op Test");
  ExpectFailure(b().Attr("a:{invalid}"),
                "Unrecognized type string 'invalid' from Attr(\"a:{invalid}\") "
                "for Op Test");
  ExpectFailure(b().Attr("a:{\"str\", float}"),
                "Trouble parsing allowed string at 'float}' from "
                "Attr(\"a:{\"str\", float}\") for Op Test");
  ExpectFailure(b().Attr("a:{ float, \"str\" }"),
                "Trouble parsing type string at '\"str\" }' from Attr(\"a:{ "
                "float, \"str\" }\") for Op Test");
  ExpectFailure(b().Attr("a:{float,,string}"),
                "Trouble parsing type string at ',string}' from "
                "Attr(\"a:{float,,string}\") for Op Test");
  ExpectFailure(b().Attr("a:{float,,}"),
                "Trouble parsing type string at ',}' from "
                "Attr(\"a:{float,,}\") for Op Test");
  ExpectFailure(b().Attr("i: int >= a"),
                "Could not parse integer lower limit after '>=', "
                "found ' a' instead from Attr(\"i: int >= a\") for Op Test");
  ExpectFailure(b().Attr("i: int >= -a"),
                "Could not parse integer lower limit after '>=', found ' -a' "
                "instead from Attr(\"i: int >= -a\") for Op Test");
  ExpectFailure(b().Attr("i: int >= 9223372036854775808"),
                "Could not parse integer lower limit after '>=', found "
                "' 9223372036854775808' instead from "
                "Attr(\"i: int >= 9223372036854775808\") for Op Test");
  ExpectFailure(b().Attr("i: int >= -9223372036854775809"),
                "Could not parse integer lower limit after '>=', found "
                "' -9223372036854775809' instead from "
                "Attr(\"i: int >= -9223372036854775809\") for Op Test");
}

TEST_F(OpDefBuilderTest, AttrListOfRestricted) {
  ExpectSuccess(
      b().Attr("a:list(realnumbertype)"),
      "attr: { name: 'a' type: 'list(type)' allowed_values { list { type: "
      "[DT_FLOAT, DT_DOUBLE, DT_INT64, DT_INT32, DT_UINT8, DT_INT16, "
      "DT_UINT16, DT_INT8] } } }");
  ExpectSuccess(
      b().Attr("a:list(quantizedtype)"),
      "attr: { name: 'a' type: 'list(type)' allowed_values { list { type: "
      "[DT_QINT8, DT_QUINT8, DT_QINT32, DT_QINT16, DT_QUINT16] } } }");
  ExpectSuccess(
      b().Attr("a: list({float, string, bool})"),
      "attr: { name: 'a' type: 'list(type)' allowed_values { list { type: "
      "[DT_FLOAT, DT_STRING, DT_BOOL] } } }");
  ExpectSuccess(
      b().Attr(R"(a: list({ "one fish", "two fish" }))"),
      "attr: { name: 'a' type: 'list(string)' allowed_values { list { s: "
      "['one fish', 'two fish'] } } }");
  ExpectSuccess(
      b().Attr(R"(a: list({ 'red fish', 'blue fish' }))"),
      "attr: { name: 'a' type: 'list(string)' allowed_values { list { s: "
      "['red fish', 'blue fish'] } } }");
  ExpectSuccess(
      b().Attr(R"(a: list({ "single' ", 'double"' }))"),
      "attr: { name: 'a' type: 'list(string)' allowed_values { list { s: "
      "[\"single' \", 'double\"'] } } }");
  ExpectSuccess(
      b().Attr(R"(a: list({ 'escape\'\n', "from\\\"NY" }))"),
      "attr: { name: 'a' type: 'list(string)' allowed_values { list { s: "
      "[\"escape'\\n\", 'from\\\\\"NY'] } } }");
}

TEST_F(OpDefBuilderTest, AttrListWithMinLength) {
  ExpectSuccess(
      b().Attr("i: list(bool) >= 4"),
      "attr: { name: 'i' type: 'list(bool)' has_minimum: true minimum: 4 }");
}

TEST_F(OpDefBuilderTest, AttrWithDefaults) {
  ExpectSuccess(b().Attr(R"(a:string="foo")"),
                "attr: { name: 'a' type: 'string' default_value { s:'foo' } }");
  ExpectSuccess(b().Attr(R"(a:string='foo')"),
                "attr: { name: 'a' type: 'string' default_value { s:'foo' } }");
  ExpectSuccess(b().Attr("a:float = 1.25"),
                "attr: { name: 'a' type: 'float' default_value { f: 1.25 } }");
  ExpectSuccess(b().Attr("a:tensor = { dtype: DT_INT32 int_val: 5 }"),
                "attr: { name: 'a' type: 'tensor' default_value { tensor {"
                "    dtype: DT_INT32 int_val: 5 } } }");
  ExpectSuccess(b().Attr("a:shape = { dim { size: 3 } dim { size: 4 } }"),
                "attr: { name: 'a' type: 'shape' default_value { shape {"
                "    dim { size: 3 } dim { size: 4 } } } }");
  ExpectSuccess(b().Attr("a:shape = { dim { size: -1 } dim { size: 4 } }"),
                "attr: { name: 'a' type: 'shape' default_value { shape {"
                "    dim { size: -1 } dim { size: 4 } } } }");
}

TEST_F(OpDefBuilderTest, AttrFailedDefaults) {
  ExpectFailure(b().Attr(R"(a:int="foo")"),
                "Could not parse default value '\"foo\"' from "
                "Attr(\"a:int=\"foo\"\") for Op Test");
  ExpectFailure(b().Attr("a:float = [1.25]"),
                "Could not parse default value '[1.25]' from Attr(\"a:float = "
                "[1.25]\") for Op Test");
}

TEST_F(OpDefBuilderTest, AttrListWithDefaults) {
  ExpectSuccess(b().Attr(R"(a:list(string)=["foo", "bar"])"),
                "attr: { name: 'a' type: 'list(string)' "
                "default_value { list { s: ['foo', 'bar'] } } }");
  ExpectSuccess(b().Attr("a:list(bool)=[true, false, true]"),
                "attr: { name: 'a' type: 'list(bool)' "
                "default_value { list { b: [true, false, true] } } }");
  ExpectSuccess(b().Attr(R"(a:list(int)=[0, -1, 2, -4, 8])"),
                "attr: { name: 'a' type: 'list(int)' "
                "default_value { list { i: [0, -1, 2, -4, 8] } } }");
  ExpectSuccess(b().Attr(R"(a:list(int)=[  ])"),
                "attr: { name: 'a' type: 'list(int)' "
                "default_value { list { i: [] } } }");
}

TEST_F(OpDefBuilderTest, AttrFailedListDefaults) {
  ExpectFailure(b().Attr(R"(a:list(int)=["foo"])"),
                "Could not parse default value '[\"foo\"]' from "
                "Attr(\"a:list(int)=[\"foo\"]\") for Op Test");
  ExpectFailure(b().Attr(R"(a:list(int)=[7, "foo"])"),
                "Could not parse default value '[7, \"foo\"]' from "
                "Attr(\"a:list(int)=[7, \"foo\"]\") for Op Test");
  ExpectFailure(b().Attr("a:list(float) = [[1.25]]"),
                "Could not parse default value '[[1.25]]' from "
                "Attr(\"a:list(float) = [[1.25]]\") for Op Test");
  ExpectFailure(b().Attr("a:list(float) = 1.25"),
                "Could not parse default value '1.25' from "
                "Attr(\"a:list(float) = 1.25\") for Op Test");
  ExpectFailure(b().Attr(R"(a:list(string)='foo')"),
                "Could not parse default value ''foo'' from "
                "Attr(\"a:list(string)='foo'\") for Op Test");
  ExpectFailure(b().Attr("a:list(float) = ["),
                "Could not parse default value '[' from "
                "Attr(\"a:list(float) = [\") for Op Test");
  ExpectFailure(b().Attr("a:list(float) = "),
                "Could not parse default value '' from "
                "Attr(\"a:list(float) = \") for Op Test");
}

TEST_F(OpDefBuilderTest, InputOutput) {
  ExpectSuccess(b().Input("a: int32"),
                "input_arg: { name: 'a' type: DT_INT32 }");
  ExpectSuccess(b().Output("b: string"),
                "output_arg: { name: 'b' type: DT_STRING }");
  ExpectSuccess(b().Input("c: float  "),
                "input_arg: { name: 'c' type: DT_FLOAT }");
  ExpectSuccess(b().Output("d: Ref  (  bool ) "),
                "output_arg: { name: 'd' type: DT_BOOL is_ref: true }");
  ExpectOrdered(b().Input("a: bool")
                    .Output("c: complex64")
                    .Input("b: int64")
                    .Output("d: string"),
                "input_arg: { name: 'a' type: DT_BOOL } "
                "input_arg: { name: 'b' type: DT_INT64 } "
                "output_arg: { name: 'c' type: DT_COMPLEX64 } "
                "output_arg: { name: 'd' type: DT_STRING }");
}

TEST_F(OpDefBuilderTest, PolymorphicInputOutput) {
  ExpectSuccess(b().Input("a: foo").Attr("foo: type"),
                "input_arg: { name: 'a' type_attr: 'foo' } "
                "attr: { name: 'foo' type: 'type' }");
  ExpectSuccess(b().Output("a: foo").Attr("foo: { bool, int32 }"),
                "output_arg: { name: 'a' type_attr: 'foo' } "
                "attr: { name: 'foo' type: 'type' "
                "allowed_values: { list { type: [DT_BOOL, DT_INT32] } } }");
}

TEST_F(OpDefBuilderTest, InputOutputListSameType) {
  ExpectSuccess(b().Input("a: n * int32").Attr("n: int"),
                "input_arg: { name: 'a' number_attr: 'n' type: DT_INT32 } "
                "attr: { name: 'n' type: 'int' has_minimum: true minimum: 1 }");
  // Polymorphic case:
  ExpectSuccess(b().Output("b: n * foo").Attr("n: int").Attr("foo: type"),
                "output_arg: { name: 'b' number_attr: 'n' type_attr: 'foo' } "
                "attr: { name: 'n' type: 'int' has_minimum: true minimum: 1 } "
                "attr: { name: 'foo' type: 'type' }");
}

TEST_F(OpDefBuilderTest, InputOutputListAnyType) {
  ExpectSuccess(
      b().Input("c: foo").Attr("foo: list(type)"),
      "input_arg: { name: 'c' type_list_attr: 'foo' } "
      "attr: { name: 'foo' type: 'list(type)' has_minimum: true minimum: 1 }");
  ExpectSuccess(
      b().Output("c: foo").Attr("foo: list({string, float})"),
      "output_arg: { name: 'c' type_list_attr: 'foo' } "
      "attr: { name: 'foo' type: 'list(type)' has_minimum: true minimum: 1 "
      "allowed_values: { list { type: [DT_STRING, DT_FLOAT] } } }");
}

TEST_F(OpDefBuilderTest, InputOutputFailure) {
  ExpectFailure(b().Input("9: int32"),
                "Trouble parsing 'name:' from Input(\"9: int32\") for Op Test");
  ExpectFailure(
      b().Output("_: int32"),
      "Trouble parsing 'name:' from Output(\"_: int32\") for Op Test");
  ExpectFailure(b().Input(": int32"),
                "Trouble parsing 'name:' from Input(\": int32\") for Op Test");
  ExpectFailure(b().Output("int32"),
                "Trouble parsing 'name:' from Output(\"int32\") for Op Test");
  ExpectFailure(
      b().Input("CAPS: int32"),
      "Trouble parsing 'name:' from Input(\"CAPS: int32\") for Op Test");
  ExpectFailure(
      b().Input("_underscore: int32"),
      "Trouble parsing 'name:' from Input(\"_underscore: int32\") for Op Test");
  ExpectFailure(
      b().Input("0digit: int32"),
      "Trouble parsing 'name:' from Input(\"0digit: int32\") for Op Test");
  ExpectFailure(b().Input("a: _"),
                "Trouble parsing either a type or an attr name at '_' from "
                "Input(\"a: _\") for Op Test");
  ExpectFailure(b().Input("a: 9"),
                "Trouble parsing either a type or an attr name at '9' from "
                "Input(\"a: 9\") for Op Test");
  ExpectFailure(b().Input("a: 9 * int32"),
                "Trouble parsing either a type or an attr name at '9 * int32' "
                "from Input(\"a: 9 * int32\") for Op Test");
  ExpectFailure(
      b().Input("a: x * _").Attr("x: type"),
      "Extra '* _' unparsed at the end from Input(\"a: x * _\") for Op Test");
  ExpectFailure(b().Input("a: x * y extra").Attr("x: int").Attr("y: type"),
                "Extra 'extra' unparsed at the end from Input(\"a: x * y "
                "extra\") for Op Test");
  ExpectFailure(b().Input("a: Ref(int32"),
                "Did not find closing ')' for 'Ref(', instead found: '' from "
                "Input(\"a: Ref(int32\") for Op Test");
  ExpectFailure(
      b().Input("a: Ref"),
      "Reference to unknown attr 'Ref' from Input(\"a: Ref\") for Op Test");
  ExpectFailure(b().Input("a: Ref(x y").Attr("x: type"),
                "Did not find closing ')' for 'Ref(', instead found: 'y' from "
                "Input(\"a: Ref(x y\") for Op Test");
  ExpectFailure(
      b().Input("a: x"),
      "Reference to unknown attr 'x' from Input(\"a: x\") for Op Test");
  ExpectFailure(
      b().Input("a: x * y").Attr("x: int"),
      "Reference to unknown attr 'y' from Input(\"a: x * y\") for Op Test");
  ExpectFailure(b().Input("a: x").Attr("x: int"),
                "Reference to attr 'x' with type int that isn't type or "
                "list(type) from Input(\"a: x\") for Op Test");
}

TEST_F(OpDefBuilderTest, Set) {
  ExpectSuccess(b().SetIsStateful(), "is_stateful: true");
  ExpectSuccess(b().SetIsCommutative().SetIsAggregate(),
                "is_commutative: true is_aggregate: true");
}

TEST_F(OpDefBuilderTest, DocUnpackSparseFeatures) {
  ExpectOrdered(b().Input("sf: string")
                    .Output("indices: int32")
                    .Output("ids: int64")
                    .Output("weights: float")
                    .Doc(R"doc(
Converts a vector of strings with dist_belief::SparseFeatures to tensors.

Note that indices, ids and weights are vectors of the same size and have
one-to-one correspondence between their elements. ids and weights are each
obtained by sequentially concatenating sf[i].id and sf[i].weight, for i in
1...size(sf). Note that if sf[i].weight is not set, the default value for the
weight is assumed to be 1.0. Also for any j, if ids[j] and weights[j] were
extracted from sf[i], then index[j] is set to i.

sf: vector of string, where each element is the string encoding of
    SparseFeatures proto.
indices: vector of indices inside sf
ids: vector of id extracted from the SparseFeatures proto.
weights: vector of weight extracted from the SparseFeatures proto.
)doc"),
                R"proto(
input_arg {
  name: "sf"
  description: "vector of string, where each element is the string encoding of\nSparseFeatures proto."
  type: DT_STRING
}
output_arg {
  name: "indices"
  description: "vector of indices inside sf"
  type: DT_INT32
}
output_arg {
  name: "ids"
  description: "vector of id extracted from the SparseFeatures proto."
  type: DT_INT64
}
output_arg {
  name: "weights"
  description: "vector of weight extracted from the SparseFeatures proto."
  type: DT_FLOAT
}
summary: "Converts a vector of strings with dist_belief::SparseFeatures to tensors."
description: "Note that indices, ids and weights are vectors of the same size and have\none-to-one correspondence between their elements. ids and weights are each\nobtained by sequentially concatenating sf[i].id and sf[i].weight, for i in\n1...size(sf). Note that if sf[i].weight is not set, the default value for the\nweight is assumed to be 1.0. Also for any j, if ids[j] and weights[j] were\nextracted from sf[i], then index[j] is set to i."
)proto");
}

TEST_F(OpDefBuilderTest, DocConcat) {
  ExpectOrdered(b().Input("concat_dim: int32")
                    .Input("values: num_values * dtype")
                    .Output("output: dtype")
                    .Attr("dtype: type")
                    .Attr("num_values: int >= 2")
                    .Doc(R"doc(
Concatenate N Tensors along one dimension.

concat_dim: The (scalar) dimension along which to concatenate.  Must be
  in the range [0, rank(values...)).
values: The N Tensors to concatenate. Their ranks and types must match,
  and their sizes must match in all dimensions except concat_dim.
output: A Tensor with the concatenation of values stacked along the
  concat_dim dimension.  This Tensor's shape matches the Tensors in
  values, except in concat_dim where it has the sum of the sizes.
)doc"),
                R"proto(
input_arg {
  name: "concat_dim"
  description: "The (scalar) dimension along which to concatenate.  Must be\nin the range [0, rank(values...))."
  type: DT_INT32
}
input_arg {
  name: "values"
  description: "The N Tensors to concatenate. Their ranks and types must match,\nand their sizes must match in all dimensions except concat_dim."
  type_attr: "dtype"
  number_attr: "num_values"
}
output_arg {
  name: "output"
  description: "A Tensor with the concatenation of values stacked along the\nconcat_dim dimension.  This Tensor\'s shape matches the Tensors in\nvalues, except in concat_dim where it has the sum of the sizes."
  type_attr: "dtype"
}
summary: "Concatenate N Tensors along one dimension."
attr {
  name: "dtype"
  type: "type"
}
attr {
  name: "num_values"
  type: "int"
  has_minimum: true
  minimum: 2
}
)proto");
}

TEST_F(OpDefBuilderTest, DocAttr) {
  ExpectOrdered(b().Attr("i: int").Doc(R"doc(
Summary

i: How much to operate.
)doc"),
                R"proto(
summary: "Summary"
attr {
  name: "i"
  type: "int"
  description: "How much to operate."
}
)proto");
}

TEST_F(OpDefBuilderTest, DocCalledTwiceFailure) {
  ExpectFailure(b().Doc("What's").Doc("up, doc?"),
                "Extra call to Doc() for Op Test");
}

TEST_F(OpDefBuilderTest, DocFailureMissingName) {
  ExpectFailure(
      b().Input("a: int32").Doc(R"doc(
Summary

a: Something for a.
b: b is not defined.
)doc"),
      "No matching input/output/attr for name 'b' from Doc() for Op Test");

  ExpectFailure(
      b().Input("a: int32").Doc(R"doc(
Summary

b: b is not defined and by itself.
)doc"),
      "No matching input/output/attr for name 'b' from Doc() for Op Test");
}

TEST_F(OpDefBuilderTest, DefaultMinimum) {
  ExpectSuccess(b().Input("values: num_values * dtype")
                    .Output("output: anything")
                    .Attr("anything: list(type)")
                    .Attr("dtype: type")
                    .Attr("num_values: int"),
                R"proto(
input_arg {
  name: "values"
  type_attr: "dtype"
  number_attr: "num_values"
}
output_arg {
  name: "output"
  type_list_attr: "anything"
}
attr {
  name: "anything"
  type: "list(type)"
  has_minimum: true
  minimum: 1
}
attr {
  name: "dtype"
  type: "type"
}
attr {
  name: "num_values"
  type: "int"
  has_minimum: true
  minimum: 1
}
)proto");
}

}  // namespace
}  // namespace tensorflow
