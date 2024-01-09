/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/node_def_builder.h"

#include <memory>
#include <vector>

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_def_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class NodeDefBuilderTest : public ::testing::Test {
 protected:
  // Specify an OpDef via an OpDefBuilder.
  void Op(const OpDefBuilder& op_def_builder) {
    OpRegistrationData op_reg_data;
    TF_EXPECT_OK(op_def_builder.Finalize(&op_reg_data));
    op_def_ = op_reg_data.op_def;
  }

  // Resets builder_ with a new NodeDefBuilder using the Op from the last call
  // to Op() above.
  NodeDefBuilder& Builder() {
    EXPECT_FALSE(op_def_.name().empty()) << "Must call Op() before Builder()";
    builder_.reset(new NodeDefBuilder("n", &op_def_));
    return *builder_;
  }

  // Calls Finalize() and verifies it returns success and the result matches
  // expectations.
  void ExpectSuccess(NodeDefBuilder& builder,  // NOLINT
                     DataTypeSlice expected_in_types,
                     DataTypeSlice expected_out_types, StringPiece proto) {
    NodeDef node_def;
    Status status = builder.Finalize(&node_def);
    TF_EXPECT_OK(status);
    if (!status.ok()) return;
    NodeDef expected;
    protobuf::TextFormat::ParseFromString(strings::StrCat("name: 'n' ", proto),
                                          &expected);
    EXPECT_EQ(node_def.DebugString(), expected.DebugString());

    DataTypeVector in_types, out_types;
    status =
        InOutTypesForNode(node_def, builder.op_def(), &in_types, &out_types);
    TF_EXPECT_OK(status);
    if (!status.ok()) return;
    EXPECT_EQ(DataTypeSliceString(expected_in_types),
              DataTypeVectorString(in_types));
    EXPECT_EQ(DataTypeSliceString(expected_out_types),
              DataTypeVectorString(out_types));

    status = ValidateNodeDef(node_def, op_def_);
    TF_EXPECT_OK(status);
  }

  // Calls Finalize() and verifies it returns an error.
  // Each message must appear as a substring of the error.
  void ExpectFailures(NodeDefBuilder& builder,  // NOLINT
                      const std::vector<string>& messages) {
    NodeDef node_def;
    Status status = builder.Finalize(&node_def);
    EXPECT_FALSE(status.ok()) << SummarizeNodeDef(node_def);
    if (status.ok()) return;
    for (const string& message : messages) {
      EXPECT_TRUE(absl::StrContains(status.message(), message))
          << status << ", " << message;
    }
  }

  // Calls Finalize() and verifies it returns an error.
  // Message must appear as a substring of the error.
  void ExpectFailure(NodeDefBuilder& builder,  // NOLINT
                     const string& message) {
    ExpectFailures(builder, {message});
  }

  // Like ExpectFailure(), except that the error can come from
  // ValidateNodeDef().
  void ExpectInvalid(NodeDefBuilder& builder,  // NOLINT
                     const string& message) {
    NodeDef node_def;
    Status status = builder.Finalize(&node_def);
    if (status.ok()) {
      status = ValidateNodeDef(node_def, op_def_);
    }
    EXPECT_FALSE(status.ok()) << SummarizeNodeDef(node_def);
    if (status.ok()) return;
    EXPECT_TRUE(absl::StrContains(status.message(), message))
        << "Actual error: " << status.message()
        << "\nDoes not contain: " << message;
  }

  OpDef op_def_;
  std::unique_ptr<NodeDefBuilder> builder_;
};

TEST_F(NodeDefBuilderTest, Simple) {
  Op(OpDefBuilder("Simple").Input("a: int32").Output("out: float"));

  ExpectSuccess(Builder().Input("x", 0, DT_INT32), {DT_INT32}, {DT_FLOAT},
                R"proto(op: "Simple" input: "x")proto");

  // Port != 0
  ExpectSuccess(Builder().Input("y", 2, DT_INT32), {DT_INT32}, {DT_FLOAT},
                R"proto(op: "Simple" input: "y:2")proto");

  // FakeInput
  ExpectSuccess(Builder().Input(FakeInput()), {DT_INT32}, {DT_FLOAT}, R"proto(
    op: "Simple"
    input: "a")proto");

  ExpectSuccess(Builder().Input(FakeInput(DT_INT32)), {DT_INT32}, {DT_FLOAT},
                R"proto(op: "Simple" input: "a")proto");

  // Ref input
  ExpectSuccess(Builder().Input(FakeInput(DT_INT32_REF)), {DT_INT32},
                {DT_FLOAT}, R"proto(op: "Simple" input: "a")proto");

  // ControlInput
  ExpectSuccess(
      Builder().ControlInput("x").Input(FakeInput()).ControlInput("y"),
      {DT_INT32}, {DT_FLOAT}, R"proto(
        op: "Simple"
        input: [ "a", "^x", "^y" ])proto");

  // Device
  ExpectSuccess(Builder().Input(FakeInput()).Device("ddd"), {DT_INT32},
                {DT_FLOAT}, R"proto(
    op: "Simple" input: "a" device: "ddd")proto");

  // Extra input
  ExpectFailure(Builder().Input("x", 0, DT_INT32).Input("y", 0, DT_INT32),
                "More Input() calls than the 1 input_args while building "
                "NodeDef 'n' using Op<name=Simple; signature=a:int32 -> "
                "out:float>");

  // Missing input
  ExpectFailure(Builder(), "0 inputs specified of 1 inputs in Op while");

  {  // Finalize() twice.
    NodeDefBuilder& builder = Builder();
    // First call to Finalize()
    TF_EXPECT_OK(builder.Input(FakeInput()).Finalize(nullptr));
    // ExpectSuccess() also calls Finalize().
    ExpectSuccess(builder, {DT_INT32}, {DT_FLOAT}, R"proto(
      op: "Simple"
      input: "a")proto");
  }

  {  // Input() after Finalize()
    NodeDefBuilder& builder = Builder();
    // Calling Finalize() before enough inputs -> error.
    ExpectFailure(builder, "0 inputs specified of 1 inputs in Op while");
    builder.Input(FakeInput());
    // Calling Finalize() with enough inputs -> success
    ExpectSuccess(builder, {DT_INT32}, {DT_FLOAT}, R"proto(
      op: "Simple"
      input: "a")proto");
    // Calling Finalize() with too many inputs -> error.
    builder.Input(FakeInput(DT_INT32));
    ExpectFailure(builder, "More Input() calls than the 1 input_args while");
  }

  // Wrong input type
  ExpectFailure(Builder().Input("x", 0, DT_FLOAT),
                "Input 'a' passed float expected int32 ");

  ExpectFailure(Builder().Input("x", 0, DT_FLOAT_REF),
                "Input 'a' passed float_ref expected int32 ");

  // List input
  ExpectFailure(Builder().Input(FakeInput(3, DT_FLOAT)),
                "List provided to input 'a' when single Tensor expected while");

  ExpectFailure(Builder().Input(FakeInput(3)),
                "List provided to input 'a' when single Tensor expected while");

  // Bad ControlInput
  ExpectInvalid(Builder().Input(FakeInput()).ControlInput("z:2"),
                "Control input '^z:2' must not have ':' in NodeDef:");

  // Bad input name
  ExpectFailure(Builder().Input("", 0, DT_INT32),
                "Empty input node name while");

  ExpectFailure(Builder().Input("^x", 0, DT_INT32),
                "Non-control input starting with ^: ^x while");
}

TEST_F(NodeDefBuilderTest, OpDoesNotExist) {
  NodeDefBuilder builder("n", "Op Does Not Exist");
  builder.Input(FakeInput())
      .Input(FakeInput(12))
      .ControlInput("y")
      .Attr("foo", 12)
      .Device("device");
  ExpectFailures(builder, {"Op type not registered 'Op Does Not Exist'",
                           "while building NodeDef 'n'"});
}

TEST_F(NodeDefBuilderTest, Polymorphic) {
  Op(OpDefBuilder("Polymorphic")
         .Input("v: T")
         .Output("out: T")
         .Attr("T: type"));

  ExpectSuccess(Builder().Input(FakeInput(DT_INT32)), {DT_INT32}, {DT_INT32},
                R"proto(
                  op: "Polymorphic"
                  input: "a"
                  attr {
                    key: "T"
                    value { type: DT_INT32 }
                  })proto");

  ExpectSuccess(Builder().Input(FakeInput(DT_FLOAT)), {DT_FLOAT}, {DT_FLOAT},
                R"proto(
                  op: "Polymorphic"
                  input: "a"
                  attr {
                    key: "T"
                    value { type: DT_FLOAT }
                  })proto");

  // Redundant Attr()
  ExpectSuccess(Builder().Input(FakeInput(DT_BOOL)).Attr("T", DT_BOOL),
                {DT_BOOL}, {DT_BOOL}, R"proto(
    op: "Polymorphic"
    input: "a"
    attr {
      key: "T"
      value { type: DT_BOOL }
    })proto");

  // Conflicting Attr()
  ExpectFailure(Builder().Input(FakeInput(DT_BOOL)).Attr("T", DT_STRING),
                "Inconsistent values for attr 'T' DT_BOOL vs. DT_STRING while");

  ExpectFailure(Builder().Attr("T", DT_STRING).Input(FakeInput(DT_BOOL)),
                "Inconsistent values for attr 'T' DT_STRING vs. DT_BOOL while");

  ExpectFailure(Builder().Attr("T", 12).Input(FakeInput(DT_BOOL)),
                "Inconsistent values for attr 'T' 12 vs. DT_BOOL while");
}

TEST_F(NodeDefBuilderTest, PolymorphicOut) {
  Op(OpDefBuilder("PolymorphicOut").Output("out: T").Attr("T: type"));

  ExpectSuccess(Builder().Attr("T", DT_INT32), {}, {DT_INT32}, R"proto(
    op: "PolymorphicOut"
    attr {
      key: "T"
      value { type: DT_INT32 }
    })proto");

  ExpectSuccess(Builder().Attr("T", DT_FLOAT), {}, {DT_FLOAT}, R"proto(
    op: "PolymorphicOut"
    attr {
      key: "T"
      value { type: DT_FLOAT }
    })proto");

  // Redundant attr
  ExpectSuccess(Builder().Attr("T", DT_FLOAT).Attr("T", DT_FLOAT), {},
                {DT_FLOAT}, R"proto(
    op: "PolymorphicOut"
    attr {
      key: "T"
      value { type: DT_FLOAT }
    })proto");

  // Conflicting attr
  ExpectFailure(Builder().Attr("T", DT_BOOL).Attr("T", DT_FLOAT),
                "Inconsistent values for attr 'T' DT_BOOL vs. DT_FLOAT while");

  // Missing attr
  ExpectInvalid(Builder(), "NodeDef missing attr 'T' from");

  // Attr has the wrong type
  ExpectInvalid(
      Builder().Attr("T", {DT_INT32, DT_BOOL}),
      "AttrValue had value with type 'list(type)' when 'type' expected");

  ExpectInvalid(Builder().Attr("T", 12),
                "AttrValue had value with type 'int' when 'type' expected");
}

TEST_F(NodeDefBuilderTest, PolymorphicDefaultOut) {
  Op(OpDefBuilder("PolymorphicDefaultOut")
         .Output("out: T")
         .Attr("T: type = DT_STRING"));

  ExpectSuccess(Builder(), {}, {DT_STRING}, R"proto(
    op: "PolymorphicDefaultOut"
    attr {
      key: "T"
      value { type: DT_STRING }
    })proto");

  ExpectSuccess(Builder().Attr("T", DT_BOOL), {}, {DT_BOOL}, R"proto(
    op: "PolymorphicDefaultOut"
    attr {
      key: "T"
      value { type: DT_BOOL }
    })proto");
}

TEST_F(NodeDefBuilderTest, Binary) {
  Op(OpDefBuilder("Binary").Input("a: T").Input("b: T").Output("out: T").Attr(
      "T: type"));

  ExpectSuccess(Builder().Input(FakeInput(DT_INT32)).Input(FakeInput(DT_INT32)),
                {DT_INT32, DT_INT32}, {DT_INT32}, R"proto(
    op: "Binary"
    input: "a"
    input: "b"
    attr {
      key: "T"
      value { type: DT_INT32 }
    })proto");

  ExpectSuccess(Builder().Input(FakeInput(DT_STRING)).Input(FakeInput()),
                {DT_STRING, DT_STRING}, {DT_STRING}, R"proto(
    op: "Binary"
    input: "a"
    input: "b"
    attr {
      key: "T"
      value { type: DT_STRING }
    })proto");

  // Type mismatch
  ExpectFailure(Builder().Input(FakeInput(DT_BOOL)).Input(FakeInput(DT_STRING)),
                "Inconsistent values for attr 'T' DT_BOOL vs. DT_STRING while");
}

TEST_F(NodeDefBuilderTest, Restrict) {
  Op(OpDefBuilder("Restrict")
         .Input("a: T")
         .Output("out: T")
         .Attr("T: {string, bool}"));
  ExpectSuccess(Builder().Input(FakeInput(DT_STRING)), {DT_STRING}, {DT_STRING},
                R"proto(
                  op: "Restrict"
                  input: "a"
                  attr {
                    key: "T"
                    value { type: DT_STRING }
                  })proto");

  ExpectInvalid(Builder().Input(FakeInput(DT_INT32)),
                "Value for attr 'T' of int32 is not in the list of allowed "
                "values: string, bool");
}

TEST_F(NodeDefBuilderTest, TypeList) {
  Op(OpDefBuilder("TypeList").Input("a: T").Attr("T: list(type)"));

  ExpectSuccess(Builder().Input(FakeInput({DT_STRING, DT_INT32})),
                {DT_STRING, DT_INT32}, {}, R"proto(
    op: "TypeList"
    input: [ "a", "a:1" ]
    attr {
      key: "T"
      value { list { type: [ DT_STRING, DT_INT32 ] } }
    }
  )proto");

  ExpectSuccess(Builder().Input(FakeInput(3, DT_BOOL)),
                {DT_BOOL, DT_BOOL, DT_BOOL}, {}, R"proto(
    op: "TypeList"
    input: [ "a", "a:1", "a:2" ]
    attr {
      key: "T"
      value { list { type: [ DT_BOOL, DT_BOOL, DT_BOOL ] } }
    }
  )proto");

  ExpectInvalid(Builder().Input(FakeInput(0)),
                "Length for attr 'T' of 0 must be at least minimum 1");

  ExpectInvalid(Builder().Input(FakeInput({})),
                "Length for attr 'T' of 0 must be at least minimum 1");

  ExpectInvalid(Builder().Input(FakeInput(DT_BOOL)),
                "Single tensor passed to 'a', expected list while");

  ExpectFailures(Builder().Input(FakeInput()),
                 {"2 errors while building NodeDef",
                  "Could not infer list of types for input 'a': "
                  "No attr named 'T' in NodeDef:",
                  "0 inputs specified of 1 inputs in Op"});
}

TEST_F(NodeDefBuilderTest, TypeListNoMin) {
  Op(OpDefBuilder("TypeListNoMin").Input("a: T").Attr("T: list(type) >= 0"));

  ExpectSuccess(Builder().Input(FakeInput(0)), {}, {}, R"proto(
    op: "TypeListNoMin"
    attr {
      key: "T"
      value { list {} }
    })proto");

  ExpectSuccess(Builder().Input(FakeInput(DataTypeVector())), {}, {}, R"proto(
    op: "TypeListNoMin"
    attr {
      key: "T"
      value { list {} }
    })proto");

  ExpectSuccess(Builder().Input(FakeInput({})), {}, {}, R"proto(
    op: "TypeListNoMin"
    attr {
      key: "T"
      value { list {} }
    })proto");

  ExpectSuccess(Builder().Input(FakeInput({DT_BOOL})), {DT_BOOL}, {}, R"proto(
    op: "TypeListNoMin"
    input: "a"
    attr {
      key: "T"
      value { list { type: DT_BOOL } }
    })proto");
}

TEST_F(NodeDefBuilderTest, TypeListTwice) {
  Op(OpDefBuilder("TypeListTwice")
         .Input("a: T")
         .Input("b: T")
         .Attr("T: list(type) >= 0"));

  ExpectSuccess(Builder()
                    .Input(FakeInput({DT_INT32, DT_BOOL}))
                    .Input(FakeInput({DT_INT32, DT_BOOL})),
                {DT_INT32, DT_BOOL, DT_INT32, DT_BOOL}, {}, R"proto(
    op: "TypeListTwice"
    input: [ "a", "a:1", "b", "b:1" ]
    attr {
      key: "T"
      value { list { type: [ DT_INT32, DT_BOOL ] } }
    })proto");

  ExpectSuccess(
      Builder().Input(FakeInput({DT_INT32, DT_BOOL})).Input(FakeInput()),
      {DT_INT32, DT_BOOL, DT_INT32, DT_BOOL}, {}, R"proto(
        op: "TypeListTwice"
        input: [ "a", "a:1", "b", "b:1" ]
        attr {
          key: "T"
          value { list { type: [ DT_INT32, DT_BOOL ] } }
        })proto");

  ExpectSuccess(Builder().Input(FakeInput(0)).Input(FakeInput(0)), {}, {},
                R"proto(
                  op: "TypeListTwice"
                  attr {
                    key: "T"
                    value { list {} }
                  })proto");

  ExpectSuccess(Builder().Input(FakeInput(0)).Input(FakeInput()), {}, {},
                R"proto(
                  op: "TypeListTwice"
                  attr {
                    key: "T"
                    value { list {} }
                  })proto");

  ExpectFailure(Builder()
                    .Input(FakeInput({DT_INT32, DT_BOOL}))
                    .Input(FakeInput({DT_INT32, DT_STRING})),
                "Inconsistent values for attr 'T' [DT_INT32, DT_BOOL] vs. "
                "[DT_INT32, DT_STRING] while");
}

TEST_F(NodeDefBuilderTest, OutTypeList) {
  Op(OpDefBuilder("OutTypeList").Output("out: T").Attr("T: list(type) >= 0"));

  ExpectSuccess(Builder().Attr("T", {DT_FLOAT}), {}, {DT_FLOAT}, R"proto(
    op: "OutTypeList"
    attr {
      key: "T"
      value { list { type: DT_FLOAT } }
    })proto");

  ExpectSuccess(Builder().Attr("T", {DT_STRING, DT_BOOL}), {},
                {DT_STRING, DT_BOOL}, R"proto(
    op: "OutTypeList"
    attr {
      key: "T"
      value { list { type: [ DT_STRING, DT_BOOL ] } }
    })proto");

  ExpectSuccess(Builder().Attr("T", DataTypeVector()), {}, {}, R"proto(
    op: "OutTypeList"
    attr {
      key: "T"
      value { list {} }
    })proto");

  ExpectInvalid(
      Builder().Attr("T", DT_FLOAT),
      "AttrValue had value with type 'type' when 'list(type)' expected");
}

TEST_F(NodeDefBuilderTest, TypeListRestrict) {
  Op(OpDefBuilder("TypeListRestrict")
         .Input("a: T")
         .Attr("T: list({string, bool}) >= 0"));

  ExpectSuccess(Builder().Input(FakeInput({DT_STRING, DT_BOOL})),
                {DT_STRING, DT_BOOL}, {}, R"proto(
    op: "TypeListRestrict"
    input: [ "a", "a:1" ]
    attr {
      key: "T"
      value { list { type: [ DT_STRING, DT_BOOL ] } }
    })proto");

  ExpectInvalid(Builder().Input(FakeInput({DT_STRING, DT_INT32})),
                "Value for attr 'T' of int32 is not in the list of allowed "
                "values: string, bool");
}

TEST_F(NodeDefBuilderTest, OutTypeListRestrict) {
  Op(OpDefBuilder("OutTypeListRestrict")
         .Output("out: t")
         .Attr("t: list({string, bool}) >= 0"));

  ExpectSuccess(Builder().Attr("t", {DT_BOOL, DT_STRING}), {},
                {DT_BOOL, DT_STRING}, R"proto(
    op: "OutTypeListRestrict"
    attr {
      key: "t"
      value { list { type: [ DT_BOOL, DT_STRING ] } }
    })proto");

  ExpectInvalid(Builder().Attr("t", {DT_STRING, DT_INT32}),
                "Value for attr 't' of int32 is not in the list of allowed "
                "values: string, bool");
}

TEST_F(NodeDefBuilderTest, Attr) {
  Op(OpDefBuilder("Attr").Attr("a: int"));

  ExpectSuccess(Builder().Attr("a", 12), {}, {}, R"proto(
    op: "Attr"
    attr {
      key: "a"
      value { i: 12 }
    })proto");

  // Attr has wrong type
  ExpectInvalid(Builder().Attr("a", "bad"),
                "AttrValue had value with type 'string' when 'int' expected");

  ExpectInvalid(
      Builder().Attr("a", {12}),
      "AttrValue had value with type 'list(int)' when 'int' expected");

  // Missing attr
  ExpectInvalid(Builder(), "NodeDef missing attr 'a' from Op<");

  // Extra attribute should be ignored.
  ExpectSuccess(Builder().Attr("a", 10).Attr("b", 12), {}, {},
                R"proto(
                  op: "Attr"
                  attr {
                    key: "a"
                    value { i: 10 }
                  }
                  attr {
                    key: "b"
                    value { i: 12 }
                  }
                )proto");
}

TEST_F(NodeDefBuilderTest, AttrFloat) {
  Op(OpDefBuilder("AttrFloat").Attr("a: float"));

  ExpectSuccess(Builder().Attr("a", 1.2f /* float */), {}, {}, R"proto(
    op: "AttrFloat"
    attr {
      key: "a"
      value { f: 1.2 }
    }
  )proto");

  ExpectSuccess(Builder().Attr("a", 1.2 /* double */), {}, {}, R"proto(
    op: "AttrFloat"
    attr {
      key: "a"
      value { f: 1.2 }
    }
  )proto");

  // Won't automatically cast int to float
  ExpectInvalid(Builder().Attr("a", 12),
                "AttrValue had value with type 'int' when 'float' expected");
}

TEST_F(NodeDefBuilderTest, AttrBoolList) {
  Op(OpDefBuilder("AttrBoolList").Attr("a: list(bool)"));

  ExpectSuccess(Builder().Attr("a", {true, false, true}), {}, {}, R"proto(
    op: "AttrBoolList"
    attr {
      key: "a"
      value { list { b: [ true, false, true ] } }
    }
  )proto");

  ExpectSuccess(Builder().Attr("a", std::vector<bool>()), {}, {}, R"proto(
    op: "AttrBoolList"
    attr {
      key: "a"
      value { list {} }
    }
  )proto");

  // Won't cast int -> bool.
  ExpectInvalid(Builder().Attr("a", {0}),
                "AttrValue had value with type 'list(int)' when 'list(bool)' "
                "expected");
}

TEST_F(NodeDefBuilderTest, AttrMin) {
  Op(OpDefBuilder("AttrMin").Attr("a: int >= 5"));

  ExpectSuccess(Builder().Attr("a", 12), {}, {}, R"proto(
    op: "AttrMin"
    attr {
      key: "a"
      value { i: 12 }
    })proto");

  ExpectInvalid(Builder().Attr("a", 2),
                "Value for attr 'a' of 2 must be at least minimum 5");
}

TEST_F(NodeDefBuilderTest, AttrListMin) {
  Op(OpDefBuilder("AttrListMin").Attr("a: list(int) >= 2"));

  ExpectSuccess(Builder().Attr("a", {1, 2}), {}, {}, R"proto(
    op: "AttrListMin"
    attr {
      key: "a"
      value { list { i: [ 1, 2 ] } }
    })proto");

  ExpectInvalid(Builder().Attr("a", {17}),
                "Length for attr 'a' of 1 must be at least minimum 2");
}

TEST_F(NodeDefBuilderTest, AttrEnum) {
  Op(OpDefBuilder("AttrEnum").Attr("a: {'apples', 'oranges'}"));

  ExpectSuccess(Builder().Attr("a", "oranges"), {}, {}, R"proto(
    op: "AttrEnum"
    attr {
      key: "a"
      value { s: "oranges" }
    })proto");

  ExpectInvalid(
      Builder().Attr("a", "invalid"),
      "Value for attr 'a' of \"invalid\" is not in the list of allowed values: "
      "\"apples\", \"oranges\"");
}

TEST_F(NodeDefBuilderTest, AttrEnumList) {
  Op(OpDefBuilder("AttrEnumList").Attr("a: list({'apples', 'oranges'})"));

  ExpectSuccess(Builder().Attr("a", {"oranges", "apples"}), {}, {}, R"proto(
    op: "AttrEnumList"
    attr {
      key: "a"
      value { list { s: [ "oranges", "apples" ] } }
    })proto");

  ExpectInvalid(
      Builder().Attr("a", {"apples", "invalid", "oranges"}),
      "Value for attr 'a' of \"invalid\" is not in the list of allowed values: "
      "\"apples\", \"oranges\"");
}

TEST_F(NodeDefBuilderTest, AttrShape) {
  Op(OpDefBuilder("AttrShape").Attr("a: shape"));

  ExpectSuccess(Builder().Attr("a", TensorShape({5})), {}, {}, R"proto(
    op: "AttrShape"
    attr {
      key: "a"
      value { shape { dim { size: 5 } } }
    })proto");

  ExpectSuccess(Builder().Attr("a", TensorShape({4, 3, 2})), {}, {}, R"proto(
    op: "AttrShape"
    attr {
      key: "a"
      value {
        shape {
          dim { size: 4 }
          dim { size: 3 }
          dim { size: 2 }
        }
      }
    })proto");

  ExpectSuccess(Builder().Attr("a", TensorShape({3, 2})), {}, {},
                R"proto(
                  op: "AttrShape"
                  attr {
                    key: "a"
                    value {
                      shape {
                        dim { size: 3 }
                        dim { size: 2 }
                      }
                    }
                  })proto");

  ExpectSuccess(Builder().Attr("a", TensorShape()), {}, {}, R"proto(
    op: "AttrShape"
    attr {
      key: "a"
      value { shape {} }
    })proto");
}

TEST_F(NodeDefBuilderTest, AttrDefault) {
  Op(OpDefBuilder("AttrDefault").Attr("a: string = 'banana'"));

  ExpectSuccess(Builder(), {}, {}, R"proto(
    op: "AttrDefault"
    attr {
      key: "a"
      value { s: "banana" }
    })proto");

  ExpectSuccess(Builder().Attr("a", "kiwi"), {}, {}, R"proto(
    op: "AttrDefault"
    attr {
      key: "a"
      value { s: "kiwi" }
    })proto");
}

TEST_F(NodeDefBuilderTest, AttrManyDefault) {
  Op(OpDefBuilder("AttrManyDefault")
         .Attr("a: string = 'banana'")
         .Attr("b: string = 'kiwi'"));

  ExpectSuccess(Builder(), {}, {}, R"proto(
    op: "AttrManyDefault"
    attr {
      key: "a"
      value { s: "banana" }
    }
    attr {
      key: "b"
      value { s: "kiwi" }
    })proto");

  Op(OpDefBuilder("AttrManyDefaultWithMandatory")
         .Attr("a: string = 'banana'")
         .Attr("b: string = 'kiwi'")
         .Attr("c: string"));

  ExpectSuccess(Builder().Attr("c", "strawberry"), {}, {}, R"proto(
    op: "AttrManyDefaultWithMandatory"
    attr {
      key: "c"
      value { s: "strawberry" }
    }
    attr {
      key: "a"
      value { s: "banana" }
    }
    attr {
      key: "b"
      value { s: "kiwi" }
    })proto");

  Op(OpDefBuilder("AttrManyDefaultAndInferred")
         .Input("input: T")
         .Attr("T: {float, double}")
         .Attr("a: string")
         .Attr("b: list(string) >= 1")
         .Attr("c: bool = true")
         .Attr("d: float = 0.3")
         .Attr("e: string")
         .Attr("f: float = 0.25"));

  ExpectSuccess(Builder()
                    .Input(FakeInput(DT_FLOAT))
                    .Attr("a", "foo")
                    .Attr("e", "foo")
                    .Attr("b", std::vector<string>({"bar", "baz"}))
                    .Attr("f", 1.0f),
                {DT_FLOAT}, {}, R"proto(
    op: "AttrManyDefaultAndInferred"
    input: "a"
    attr {
      key: "T"
      value { type: DT_FLOAT }
    }
    attr {
      key: "a"
      value { s: "foo" }
    }
    attr {
      key: "e"
      value { s: "foo" }
    }
    attr {
      key: "b"
      value { list { s: "bar" s: "baz" } }
    }
    attr {
      key: "f"
      value { f: 1.0 }
    }
    attr {
      key: "c"
      value { b: true }
    }
    attr {
      key: "d"
      value { f: 0.3 }
    })proto");
}

TEST_F(NodeDefBuilderTest, AttrListDefault) {
  Op(OpDefBuilder("AttrListDefault").Attr("a: list(int) = [5, 15]"));

  ExpectSuccess(Builder(), {}, {}, R"proto(
    op: "AttrListDefault"
    attr {
      key: "a"
      value { list { i: [ 5, 15 ] } }
    })proto");

  ExpectSuccess(Builder().Attr("a", {3}), {}, {}, R"proto(
    op: "AttrListDefault"
    attr {
      key: "a"
      value { list { i: 3 } }
    })proto");

  ExpectSuccess(Builder().Attr("a", std::vector<int>()), {}, {}, R"proto(
    op: "AttrListDefault"
    attr {
      key: "a"
      value { list {} }
    })proto");
}

TEST_F(NodeDefBuilderTest, AttrEmptyListDefault) {
  Op(OpDefBuilder("AttrEmptyListDefault").Attr("a: list(int) = []"));

  ExpectSuccess(Builder(), {}, {}, R"proto(
    op: "AttrEmptyListDefault"
    attr {
      key: "a"
      value { list {} }
    })proto");

  ExpectSuccess(Builder().Attr("a", {3}), {}, {}, R"proto(
    op: "AttrEmptyListDefault"
    attr {
      key: "a"
      value { list { i: 3 } }
    })proto");

  ExpectSuccess(Builder().Attr("a", std::vector<int>()), {}, {}, R"proto(
    op: "AttrEmptyListDefault"
    attr {
      key: "a"
      value { list {} }
    })proto");
}

TEST_F(NodeDefBuilderTest, NIntsIn) {
  Op(OpDefBuilder("NIntsIn").Input("a: N*int32").Attr("N: int >= 2"));

  ExpectSuccess(Builder().Input(FakeInput(2)), {DT_INT32, DT_INT32}, {},
                R"proto(
                  op: "NIntsIn"
                  input: [ "a", "a:1" ]
                  attr {
                    key: "N"
                    value { i: 2 }
                  })proto");

  ExpectSuccess(Builder().Input(FakeInput(5, DT_INT32)),
                {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32}, {}, R"proto(
    op: "NIntsIn"
    input: [ "a", "a:1", "a:2", "a:3", "a:4" ]
    attr {
      key: "N"
      value { i: 5 }
    })proto");

  ExpectFailures(Builder().Input(FakeInput(2, DT_STRING)),
                 {"2 errors while building NodeDef",
                  "Input 'a' passed string expected int32"});

  ExpectInvalid(Builder().Input(FakeInput(1)),
                "Value for attr 'N' of 1 must be at least minimum 2");

  ExpectFailures(
      Builder().Input(FakeInput(DT_INT32)),
      {"2 errors while building NodeDef",
       "Could not infer length of input 'a': No attr named 'N' in NodeDef:",
       "0 inputs specified of 1 inputs in Op"});

  ExpectFailure(Builder().Input({{"in", 0, DT_INT32}, {"in", 1, DT_STRING}}),
                "Input 'a' passed string expected int32 while");

  ExpectFailures(
      Builder().Input(FakeInput()),
      {"2 errors while building NodeDef",
       "Could not infer length of input 'a': No attr named 'N' in NodeDef:",
       "0 inputs specified of 1 inputs in Op"});
}

TEST_F(NodeDefBuilderTest, NPolymorphicIn) {
  Op(OpDefBuilder("NPolymorphicIn")
         .Input("a: N*T")
         .Attr("T: type")
         .Attr("N: int >= 2"));

  ExpectSuccess(Builder().Input(FakeInput(2, DT_INT32)), {DT_INT32, DT_INT32},
                {}, R"proto(
    op: "NPolymorphicIn"
    input: [ "a", "a:1" ]
    attr {
      key: "N"
      value { i: 2 }
    }
    attr {
      key: "T"
      value { type: DT_INT32 }
    })proto");

  ExpectSuccess(Builder().Input(FakeInput(3, DT_STRING)),
                {DT_STRING, DT_STRING, DT_STRING}, {}, R"proto(
    op: "NPolymorphicIn"
    input: [ "a", "a:1", "a:2" ]
    attr {
      key: "N"
      value { i: 3 }
    }
    attr {
      key: "T"
      value { type: DT_STRING }
    })proto");

  ExpectFailures(
      Builder().Input(FakeInput(2)),
      {"2 errors while building NodeDef",
       "Could not infer type for input 'a': No attr named 'T' in NodeDef:",
       "0 inputs specified of 1 inputs in Op"});

  ExpectFailure(Builder().Input(FakeInput({DT_INT32, DT_STRING})),
                "Input 'a' passed string expected int32 while");

  ExpectFailure(Builder().Input({{"in", 0, DT_INT32}, {"in", 1, DT_STRING}}),
                "Input 'a' passed string expected int32 while");

  ExpectInvalid(Builder().Input(FakeInput(1, DT_INT32)),
                "Value for attr 'N' of 1 must be at least minimum 2");

  ExpectFailure(Builder().Input("in", 0, DT_INT32),
                "Single tensor passed to 'a', expected list while");
}

TEST_F(NodeDefBuilderTest, NPolymorphicRestrictIn) {
  Op(OpDefBuilder("NPolymorphicRestrictIn")
         .Input("a: N*T")
         .Attr("T: {string, bool}")
         .Attr("N: int >= 2"));

  ExpectSuccess(Builder().Input(FakeInput(2, DT_BOOL)), {DT_BOOL, DT_BOOL}, {},
                R"proto(
                  op: "NPolymorphicRestrictIn"
                  input: [ "a", "a:1" ]
                  attr {
                    key: "N"
                    value { i: 2 }
                  }
                  attr {
                    key: "T"
                    value { type: DT_BOOL }
                  })proto");

  ExpectSuccess(Builder().Input(FakeInput(3, DT_STRING)),
                {DT_STRING, DT_STRING, DT_STRING}, {}, R"proto(
    op: "NPolymorphicRestrictIn"
    input: [ "a", "a:1", "a:2" ]
    attr {
      key: "N"
      value { i: 3 }
    }
    attr {
      key: "T"
      value { type: DT_STRING }
    })proto");

  ExpectInvalid(Builder().Input(FakeInput(2, DT_INT32)),
                "Value for attr 'T' of int32 is not in the list of allowed "
                "values: string, bool");
}

TEST_F(NodeDefBuilderTest, NInTwice) {
  Op(OpDefBuilder("NInTwice")
         .Input("a: N*int32")
         .Input("b: N*string")
         .Attr("N: int >= 0"));

  ExpectSuccess(Builder().Input(FakeInput(2)).Input(FakeInput(2)),
                {DT_INT32, DT_INT32, DT_STRING, DT_STRING}, {}, R"proto(
    op: "NInTwice"
    input: [ "a", "a:1", "b", "b:1" ]
    attr {
      key: "N"
      value { i: 2 }
    })proto");

  ExpectSuccess(Builder().Input(FakeInput(0)).Input(FakeInput()), {}, {},
                R"proto(
                  op: "NInTwice"
                  attr {
                    key: "N"
                    value { i: 0 }
                  })proto");

  ExpectFailure(Builder().Input(FakeInput(3)).Input(FakeInput(1)),
                "Inconsistent values for attr 'N' 3 vs. 1 while");
}

TEST_F(NodeDefBuilderTest, NInPolymorphicTwice) {
  Op(OpDefBuilder("NInPolymorphicTwice")
         .Input("a: N*T")
         .Input("b: N*T")
         .Attr("T: type")
         .Attr("N: int >= 0"));

  ExpectSuccess(Builder().Input(FakeInput(2, DT_INT32)).Input(FakeInput()),
                {DT_INT32, DT_INT32, DT_INT32, DT_INT32}, {}, R"proto(
    op: "NInPolymorphicTwice"
    input: [ "a", "a:1", "b", "b:1" ]
    attr {
      key: "N"
      value { i: 2 }
    }
    attr {
      key: "T"
      value { type: DT_INT32 }
    })proto");

  ExpectFailure(
      Builder().Input(FakeInput(3, DT_INT32)).Input(FakeInput(1, DT_INT32)),
      "Inconsistent values for attr 'N' 3 vs. 1 while");

  ExpectFailure(Builder().Input(FakeInput(3, DT_INT32)).Input(FakeInput(1)),
                "Inconsistent values for attr 'N' 3 vs. 1 while");

  ExpectFailure(
      Builder().Input(FakeInput(2, DT_INT32)).Input(FakeInput(2, DT_STRING)),
      "Inconsistent values for attr 'T' DT_INT32 vs. DT_STRING while");

  ExpectFailure(
      Builder().Input(FakeInput(2, DT_INT32)).Input(FakeInput(DT_STRING)),
      "Inconsistent values for attr 'T' DT_INT32 vs. DT_STRING while");
}

TEST_F(NodeDefBuilderTest, NInTwoTypeVariables) {
  Op(OpDefBuilder("NInTwoTypeVariables")
         .Input("a: N*S")
         .Input("b: N*T")
         .Attr("S: type")
         .Attr("T: type")
         .Attr("N: int >= 0"));

  ExpectSuccess(
      Builder().Input(FakeInput(2, DT_INT32)).Input(FakeInput(2, DT_BOOL)),
      {DT_INT32, DT_INT32, DT_BOOL, DT_BOOL}, {}, R"proto(
        op: "NInTwoTypeVariables"
        input: [ "a", "a:1", "b", "b:1" ]
        attr {
          key: "N"
          value { i: 2 }
        }
        attr {
          key: "S"
          value { type: DT_INT32 }
        }
        attr {
          key: "T"
          value { type: DT_BOOL }
        })proto");

  ExpectSuccess(
      Builder().Input(FakeInput(2, DT_INT32)).Input(FakeInput(DT_BOOL)),
      {DT_INT32, DT_INT32, DT_BOOL, DT_BOOL}, {}, R"proto(
        op: "NInTwoTypeVariables"
        input: [ "a", "a:1", "b", "b:1" ]
        attr {
          key: "N"
          value { i: 2 }
        }
        attr {
          key: "S"
          value { type: DT_INT32 }
        }
        attr {
          key: "T"
          value { type: DT_BOOL }
        })proto");

  ExpectFailure(
      Builder().Input(FakeInput(3, DT_INT32)).Input(FakeInput(1, DT_STRING)),
      "Inconsistent values for attr 'N' 3 vs. 1 while");
}

TEST_F(NodeDefBuilderTest, InPolymorphicTwice) {
  Op(OpDefBuilder("InPolymorphicTwice")
         .Input("a: N*T")
         .Input("b: M*T")
         .Attr("T: type")
         .Attr("N: int >= 0")
         .Attr("M: int >= 0"));

  ExpectSuccess(
      Builder().Input(FakeInput(1, DT_INT32)).Input(FakeInput(3, DT_INT32)),
      {DT_INT32, DT_INT32, DT_INT32, DT_INT32}, {}, R"proto(
        op: "InPolymorphicTwice"
        input: [ "a", "b", "b:1", "b:2" ]
        attr {
          key: "N"
          value { i: 1 }
        }
        attr {
          key: "T"
          value { type: DT_INT32 }
        }
        attr {
          key: "M"
          value { i: 3 }
        })proto");

  ExpectSuccess(Builder().Input(FakeInput(1, DT_BOOL)).Input(FakeInput(0)),
                {DT_BOOL}, {}, R"proto(
    op: "InPolymorphicTwice"
    input: "a"
    attr {
      key: "N"
      value { i: 1 }
    }
    attr {
      key: "T"
      value { type: DT_BOOL }
    }
    attr {
      key: "M"
      value { i: 0 }
    })proto");

  ExpectSuccess(Builder().Input(FakeInput(0)).Input(FakeInput(1, DT_BOOL)),
                {DT_BOOL}, {}, R"proto(
    op: "InPolymorphicTwice"
    input: "b"
    attr {
      key: "N"
      value { i: 0 }
    }
    attr {
      key: "M"
      value { i: 1 }
    }
    attr {
      key: "T"
      value { type: DT_BOOL }
    })proto");

  ExpectFailure(
      Builder().Input(FakeInput(2, DT_INT32)).Input(FakeInput(2, DT_STRING)),
      "Inconsistent values for attr 'T' DT_INT32 vs. DT_STRING while");
}

TEST_F(NodeDefBuilderTest, NIntsOut) {
  Op(OpDefBuilder("NIntsOut").Output("a: N*int32").Attr("N: int >= 2"));

  ExpectSuccess(Builder().Attr("N", 2), {}, {DT_INT32, DT_INT32}, R"proto(
    op: "NIntsOut"
    attr {
      key: "N"
      value { i: 2 }
    })proto");

  ExpectSuccess(Builder().Attr("N", 3), {}, {DT_INT32, DT_INT32, DT_INT32},
                R"proto(
                  op: "NIntsOut"
                  attr {
                    key: "N"
                    value { i: 3 }
                  })proto");

  ExpectInvalid(Builder().Attr("N", 1),
                "Value for attr 'N' of 1 must be at least minimum 2");

  ExpectInvalid(
      Builder().Attr("N", {3}),
      "AttrValue had value with type 'list(int)' when 'int' expected");

  ExpectInvalid(Builder(), "NodeDef missing attr 'N' from");
}

TEST_F(NodeDefBuilderTest, NIntsOutDefault) {
  Op(OpDefBuilder("NIntsOutDefault")
         .Output("a: N*int32")
         .Attr("N: int >= 2 = 3"));

  ExpectSuccess(Builder(), {}, {DT_INT32, DT_INT32, DT_INT32}, R"proto(
    op: "NIntsOutDefault"
    attr {
      key: "N"
      value { i: 3 }
    })proto");

  ExpectSuccess(Builder().Attr("N", 2), {}, {DT_INT32, DT_INT32}, R"proto(
    op: "NIntsOutDefault"
    attr {
      key: "N"
      value { i: 2 }
    })proto");
}

TEST_F(NodeDefBuilderTest, NPolymorphicOut) {
  Op(OpDefBuilder("NPolymorphicOut")
         .Output("a: N*T")
         .Attr("T: type")
         .Attr("N: int >= 2"));

  ExpectSuccess(Builder().Attr("T", DT_INT32).Attr("N", 2), {},
                {DT_INT32, DT_INT32}, R"proto(
    op: "NPolymorphicOut"
    attr {
      key: "T"
      value { type: DT_INT32 }
    }
    attr {
      key: "N"
      value { i: 2 }
    })proto");

  ExpectSuccess(Builder().Attr("N", 3).Attr("T", DT_STRING), {},
                {DT_STRING, DT_STRING, DT_STRING}, R"proto(
    op: "NPolymorphicOut"
    attr {
      key: "N"
      value { i: 3 }
    }
    attr {
      key: "T"
      value { type: DT_STRING }
    })proto");

  ExpectInvalid(Builder().Attr("N", 1).Attr("T", DT_STRING),
                "Value for attr 'N' of 1 must be at least minimum 2");

  ExpectInvalid(
      Builder().Attr("N", 3).Attr("T", {DT_STRING}),
      "AttrValue had value with type 'list(type)' when 'type' expected");
}

TEST_F(NodeDefBuilderTest, NPolymorphicOutDefault) {
  Op(OpDefBuilder("NPolymorphicOutDefault")
         .Output("a: N*T")
         .Attr("T: type = DT_BOOL")
         .Attr("N: int >= 2 = 2"));

  ExpectSuccess(Builder(), {}, {DT_BOOL, DT_BOOL}, R"proto(
    op: "NPolymorphicOutDefault"
    attr {
      key: "T"
      value { type: DT_BOOL }
    }
    attr {
      key: "N"
      value { i: 2 }
    })proto");

  ExpectSuccess(Builder().Attr("N", 3), {}, {DT_BOOL, DT_BOOL, DT_BOOL},
                R"proto(
                  op: "NPolymorphicOutDefault"
                  attr {
                    key: "N"
                    value { i: 3 }
                  }
                  attr {
                    key: "T"
                    value { type: DT_BOOL }
                  })proto");

  ExpectSuccess(Builder().Attr("T", DT_INT32), {}, {DT_INT32, DT_INT32},
                R"proto(
                  op: "NPolymorphicOutDefault"
                  attr {
                    key: "T"
                    value { type: DT_INT32 }
                  }
                  attr {
                    key: "N"
                    value { i: 2 }
                  })proto");

  ExpectSuccess(Builder().Attr("N", 3).Attr("T", DT_INT32), {},
                {DT_INT32, DT_INT32, DT_INT32}, R"proto(
    op: "NPolymorphicOutDefault"
    attr {
      key: "N"
      value { i: 3 }
    }
    attr {
      key: "T"
      value { type: DT_INT32 }
    })proto");
}

TEST_F(NodeDefBuilderTest, NPolymorphicRestrictOut) {
  Op(OpDefBuilder("NPolymorphicRestrictOut")
         .Output("a: N*T")
         .Attr("T: {string, bool}")
         .Attr("N: int >= 2"));

  ExpectSuccess(Builder().Attr("N", 3).Attr("T", DT_BOOL), {},
                {DT_BOOL, DT_BOOL, DT_BOOL}, R"proto(
    op: "NPolymorphicRestrictOut"
    attr {
      key: "N"
      value { i: 3 }
    }
    attr {
      key: "T"
      value { type: DT_BOOL }
    })proto");

  ExpectInvalid(Builder().Attr("N", 3).Attr("T", DT_INT32),
                "Value for attr 'T' of int32 is not in the list of allowed "
                "values: string, bool");
}

TEST_F(NodeDefBuilderTest, RefIn) {
  Op(OpDefBuilder("RefIn").Input("a: Ref(int32)"));

  ExpectSuccess(Builder().Input(FakeInput(DT_INT32_REF)), {DT_INT32_REF}, {},
                R"proto(
                  op: "RefIn" input: "a")proto");

  ExpectFailure(Builder().Input(FakeInput(DT_BOOL_REF)),
                "Input 'a' passed bool_ref expected int32_ref while");

  ExpectFailure(Builder().Input(FakeInput(DT_INT32)),
                "Input 'a' passed int32 expected int32_ref while");
}

TEST_F(NodeDefBuilderTest, PolymorphicRefIn) {
  Op(OpDefBuilder("PolymorphicRefIn").Input("a: Ref(T)").Attr("T: type"));

  ExpectSuccess(Builder().Input(FakeInput(DT_BOOL_REF)), {DT_BOOL_REF}, {},
                R"proto(
                  op: "PolymorphicRefIn"
                  input: "a"
                  attr {
                    key: "T"
                    value { type: DT_BOOL }
                  })proto");

  ExpectFailure(Builder().Input(FakeInput(DT_BOOL)),
                "Input 'a' passed bool expected ref type while");
}

TEST_F(NodeDefBuilderTest, RefOut) {
  Op(OpDefBuilder("RefOut").Output("a: Ref(string)"));

  ExpectSuccess(Builder(), {}, {DT_STRING_REF}, R"proto(
    op: "RefOut")proto");
}

TEST_F(NodeDefBuilderTest, PolymorphicRefOut) {
  Op(OpDefBuilder("PolymorphicRefOut").Output("a: Ref(t)").Attr("t: type"));

  ExpectSuccess(Builder().Attr("t", DT_BOOL), {}, {DT_BOOL_REF}, R"proto(
    op: "PolymorphicRefOut"
    attr {
      key: "t"
      value { type: DT_BOOL }
    })proto");
}

TEST_F(NodeDefBuilderTest, SpecifyDevice) {
  Op(OpDefBuilder("SpecifyDevice"));

  ExpectSuccess(Builder().Device("ADevice"), {}, {}, R"proto(
    op: "SpecifyDevice"
    device: "ADevice")proto");
}

}  // namespace
}  // namespace tensorflow
