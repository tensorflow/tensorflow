#include "tensorflow/core/framework/op_def_util.h"

#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include <gtest/gtest.h>

namespace tensorflow {
namespace {

OpDef FromText(const string& text) {
  OpDef op_def;
  EXPECT_TRUE(protobuf::TextFormat::MergeFromString(text, &op_def));
  return op_def;
}

class ValidateOpDefTest : public ::testing::Test {
 protected:
  Status TestProto(const string& text) {
    return ValidateOpDef(FromText(text));
  }

  Status TestBuilder(const OpDefBuilder& builder) {
    OpDef op_def;
    Status status = builder.Finalize(&op_def);
    EXPECT_OK(status);
    if (!status.ok()) {
      return status;
    } else {
      return ValidateOpDef(op_def);
    }
  }

  void ExpectFailure(const Status& status, const string& message) {
    EXPECT_FALSE(status.ok()) << "Did not see error with: " << message;
    if (!status.ok()) {
      LOG(INFO) << "message: " << status;
      EXPECT_TRUE(StringPiece(status.ToString()).contains(message))
          << "Actual: " << status << "\nExpected to contain: " << message;
    }
  }
};

TEST_F(ValidateOpDefTest, OpDefValid) {
  EXPECT_OK(TestBuilder(OpDefBuilder("X").Attr("a: int")));
  EXPECT_OK(TestBuilder(OpDefBuilder("X").Input("a: int32")));
  EXPECT_OK(TestBuilder(OpDefBuilder("X").Output("a: bool")));
  EXPECT_OK(TestBuilder(OpDefBuilder("X").Attr("t: type").Input("a: t")));
  EXPECT_OK(TestBuilder(OpDefBuilder("X").Attr("a: int = 3")));
  EXPECT_OK(TestBuilder(OpDefBuilder("X").Attr("a: int >= -5")));
  EXPECT_OK(TestBuilder(OpDefBuilder("X").Attr("a: int >= -5")));
  EXPECT_OK(TestBuilder(OpDefBuilder("X").Attr("a: int >= -5 = 3")));
  EXPECT_OK(TestBuilder(OpDefBuilder("X").Attr("a: numbertype")));
  EXPECT_OK(TestBuilder(OpDefBuilder("Uppercase")));
}

TEST_F(ValidateOpDefTest, InvalidName) {
  ExpectFailure(TestBuilder(OpDefBuilder("lower").Attr("a: int")),
                "Invalid name");
  ExpectFailure(TestBuilder(OpDefBuilder("BadSuffix 7%")), "Invalid name");
}

TEST_F(ValidateOpDefTest, DuplicateName) {
  ExpectFailure(
      TestBuilder(OpDefBuilder("DupeName").Input("a: int32").Input("a: float")),
      "Duplicate name: a");
  ExpectFailure(
      TestBuilder(
          OpDefBuilder("DupeName").Input("a: int32").Output("a: float")),
      "Duplicate name: a");
  ExpectFailure(
      TestBuilder(
          OpDefBuilder("DupeName").Output("a: int32").Output("a: float")),
      "Duplicate name: a");
  ExpectFailure(
      TestBuilder(OpDefBuilder("DupeName").Input("a: int32").Attr("a: float")),
      "Duplicate name: a");
  ExpectFailure(
      TestBuilder(OpDefBuilder("DupeName").Output("a: int32").Attr("a: float")),
      "Duplicate name: a");
  ExpectFailure(
      TestBuilder(OpDefBuilder("DupeName").Attr("a: int").Attr("a: float")),
      "Duplicate name: a");
}

TEST_F(ValidateOpDefTest, BadAttrName) {
  ExpectFailure(TestBuilder(OpDefBuilder("BadAttrtude").Attr("int32: int")),
                "Attr can't have name int32 that matches a data type");
  ExpectFailure(TestBuilder(OpDefBuilder("BadAttrtude").Attr("float: string")),
                "Attr can't have name float that matches a data type");
}

TEST_F(ValidateOpDefTest, BadAttrType) {
  ExpectFailure(
      TestProto("name: 'BadAttrType' attr { name: 'a' type: 'illegal' }"),
      "Unrecognized type");
  ExpectFailure(
      TestProto("name: 'BadAttrType' attr { name: 'a' type: 'list(illegal)' }"),
      "Unrecognized type");
  ExpectFailure(
      TestProto("name: 'BadAttrType' attr { name: 'a' type: 'int extra' }"),
      "Extra ' extra' at the end");
  ExpectFailure(
      TestProto(
          "name: 'BadAttrType' attr { name: 'a' type: 'list(int extra)' }"),
      "'list(' is missing ')' in attr");
  ExpectFailure(
      TestProto(
          "name: 'BadAttrType' attr { name: 'a' type: 'list(int) extra' }"),
      "Extra ' extra' at the end");
}

TEST_F(ValidateOpDefTest, BadAttrDefault) {
  ExpectFailure(
      TestProto("name: 'BadAttrDef' attr { name: 'a' "
                "type: 'int' default_value { s: 'x' } }"),
      "AttrValue had value with type string when int expected\n\t for "
      "attr 'a'\n\t in Op 'BadAttrDef'");
  ExpectFailure(TestProto("name: 'BadAttrDef' attr { name: 'a' "
                          "type: 'int' default_value { f: 0.5 } }"),
                "AttrValue had value with type float when int expected\n\t for "
                "attr 'a'\n\t in Op 'BadAttrDef'");
  ExpectFailure(
      TestProto("name: 'BadAttrDef' attr { name: 'a' type: 'int' "
                "default_value { i: 5 list { i: [2] } } }"),
      "AttrValue had value with type list(int) when int expected\n\t for "
      "attr 'a'\n\t in Op 'BadAttrDef'");
  ExpectFailure(
      TestProto("name: 'BadAttrDef' attr { name: 'a' "
                "type: 'list(int)' default_value { f: 0.5 } }"),
      "AttrValue had value with type float when list(int) expected\n\t "
      "for attr 'a'\n\t in Op 'BadAttrDef'");
  ExpectFailure(
      TestProto("name: 'BadAttrDef' attr { name: 'a' type: 'list(int)' "
                "default_value { list { i: [5] f: [0.5] } } }"),
      "AttrValue had value with type list(float) when list(int) "
      "expected\n\t for attr 'a'\n\t in Op 'BadAttrDef'");

  ExpectFailure(TestProto("name: 'BadAttrDef' attr { name: 'a' "
                          "type: 'type' default_value { } }"),
                "AttrValue missing value with expected type type\n\t for "
                "attr 'a'\n\t in Op 'BadAttrDef'");
  ExpectFailure(TestProto("name: 'BadAttrDef' attr { name: 'a' "
                          "type: 'shape' default_value { } }"),
                "AttrValue missing value with expected type shape\n\t for "
                "attr 'a'\n\t in Op 'BadAttrDef'");
  ExpectFailure(TestProto("name: 'BadAttrDef' attr { name: 'a' "
                          "type: 'tensor' default_value { } }"),
                "AttrValue missing value with expected type tensor\n\t for "
                "attr 'a'\n\t in Op 'BadAttrDef'");

  // default_value {} is indistinguishable from default_value{ list{} } (one
  // with an empty list) in proto3 semantics.
  EXPECT_OK(
      TestProto("name: 'GoodAttrDef' attr { name: 'a' "
                "type: 'list(int)' default_value { } }"));

  // Empty lists are allowed:
  EXPECT_OK(
      TestProto("name: 'GoodAttrDef' attr { name: 'a' "
                "type: 'list(int)' default_value { list { } } }"));
  // Builder should make the same proto:
  EXPECT_OK(TestBuilder(OpDefBuilder("GoodAttrDef").Attr("a: list(int) = []")));

  // Unless there is a minimum length specified:
  ExpectFailure(TestProto("name: 'BadAttrDef' attr { name: 'a' "
                          "type: 'list(int)' has_minimum: true minimum: 2 "
                          "default_value { list { } } }"),
                "Length for attr 'a' of 0 must be at least minimum 2\n\t in Op "
                "'BadAttrDef'");
  ExpectFailure(
      TestBuilder(OpDefBuilder("GoodAttrDef").Attr("a: list(bool) >=2 = []")),
      "Length for attr 'a' of 0 must be at least minimum 2\n\t in Op "
      "'GoodAttrDef'");
  ExpectFailure(TestProto("name: 'BadAttrDef' attr { name: 'a' type: "
                          "'list(string)' has_minimum: true minimum: 2 "
                          "default_value { list { s: ['foo'] } } }"),
                "Length for attr 'a' of 1 must be at least minimum 2\n\t in Op "
                "'BadAttrDef'");
  ExpectFailure(TestBuilder(OpDefBuilder("GoodAttrDef")
                                .Attr("a: list(type) >=2 = [DT_STRING]")),
                "Length for attr 'a' of 1 must be at least minimum 2\n\t in Op "
                "'GoodAttrDef'");
}

TEST_F(ValidateOpDefTest, NoRefTypes) {
  ExpectFailure(TestBuilder(OpDefBuilder("BadAttrDef").Input("i: float_ref")),
                "Illegal use of ref type 'float_ref'. "
                "Use 'Ref(type)' instead for input 'i'");
  ExpectFailure(
      TestBuilder(OpDefBuilder("BadAttrDef").Attr("T: type = DT_INT32_REF")),
      "AttrValue must not have reference type value of int32_ref");
  ExpectFailure(TestBuilder(OpDefBuilder("BadAttrDef")
                                .Attr("T: list(type) = [DT_STRING_REF]")),
                "AttrValue must not have reference type value of string_ref");
}

TEST_F(ValidateOpDefTest, BadAttrMin) {
  ExpectFailure(TestProto("name: 'BadAttrMin' attr { name: 'a' type: 'string' "
                          "has_minimum: true minimum: 0 }"),
                "minimum for unsupported type string");
  ExpectFailure(
      TestProto("name: 'BadAttrMin' attr { name: 'a' type: 'int' default_value "
                "{ i: 2 } has_minimum: true minimum: 7 }"),
      "Value for attr 'a' of 2 must be at least minimum 7\n\t in Op "
      "'BadAttrMin'");
  ExpectFailure(
      TestProto("name: 'BadAttrMin' attr { name: 'a' "
                "type: 'list(string)' has_minimum: true minimum: -5 }"),
      "list type must have a non-negative minimum, not -5");
  EXPECT_OK(
      TestProto("name: 'GoodAttrMin' attr { name: 'a' type: 'list(string)' "
                "has_minimum: true minimum: 1 }"));
  ExpectFailure(TestProto("name: 'NoHasMin' attr { name: 'a' "
                          "type: 'list(string)' minimum: 3 }"),
                "Attr 'a' with has_minimum = false but minimum 3 not equal to "
                "default of 0");
}

TEST_F(ValidateOpDefTest, BadAttrAllowed) {
  // Is in list of allowed types.
  EXPECT_OK(TestBuilder(
      OpDefBuilder("GoodAttrtude").Attr("x: numbertype = DT_INT32")));
  // Not in list of allowed types.
  ExpectFailure(TestBuilder(OpDefBuilder("BadAttrtude")
                                .Attr("x: numbertype = DT_STRING")),
                "attr 'x' of string is not in the list of allowed values");
  ExpectFailure(
      TestBuilder(OpDefBuilder("BadAttrtude")
                      .Attr("x: list(realnumbertype) = [DT_COMPLEX64]")),
      "attr 'x' of complex64 is not in the list of allowed values");
  // Is in list of allowed strings.
  EXPECT_OK(TestBuilder(
      OpDefBuilder("GoodAttrtude").Attr("x: {'foo', 'bar'} = 'bar'")));
  // Not in list of allowed strings.
  ExpectFailure(TestBuilder(OpDefBuilder("BadAttrtude")
                                .Attr("x: {'foo', 'bar'} = 'baz'")),
                "attr 'x' of \"baz\" is not in the list of allowed values");
  ExpectFailure(TestBuilder(OpDefBuilder("BadAttrtude")
                                .Attr("x: list({'foo', 'bar'}) = ['baz']")),
                "attr 'x' of \"baz\" is not in the list of allowed values");
  ExpectFailure(TestProto(
                    "name: 'BadAttrtude' attr { name: 'a' "
                    "type: 'string' allowed_values { s: 'not list' } }"),
                "with type string when list(string) expected");
  ExpectFailure(
      TestProto("name: 'BadAttrtude' attr { name: 'a' "
                "type: 'string' allowed_values { list { i: [6] } } }"),
      "with type list(int) when list(string) expected");
}

TEST_F(ValidateOpDefTest, BadArgType) {
  ExpectFailure(TestProto("name: 'BadArg' input_arg { name: 'a' "
                          "type: DT_INT32 } input_arg { name: 'b' }"),
                "Missing type for input 'b'");
  ExpectFailure(TestProto("name: 'BadArg' input_arg { name: 'a' "
                          "type: DT_INT32 } output_arg { name: 'b' }"),
                "Missing type for output 'b'");
  ExpectFailure(
      TestProto("name: 'BadArg' input_arg { name: 'a' type: "
                "DT_INT32 type_attr: 'x' } attr { name: 'x' type: 'type' }"),
      "Exactly one of type, type_attr, type_list_attr must be set for input "
      "'a'");
  ExpectFailure(TestProto("name: 'BadArg' input_arg { name: 'a' "
                          "type_attr: 'x' } attr { name: 'x' type: 'int' }"),
                "Attr 'x' used as type_attr for input 'a' has type int");
  ExpectFailure(
      TestProto("name: 'BadArg' input_arg { name: 'a' "
                "type_attr: 'x' } attr { name: 'x' type: 'list(type)' }"),
      "Attr 'x' used as type_attr for input 'a' has type list(type)");
  ExpectFailure(
      TestProto("name: 'BadArg' input_arg { name: 'a' "
                "type_list_attr: 'x' } attr { name: 'x' type: 'int' }"),
      "Attr 'x' used as type_list_attr for input 'a' has type int");
  ExpectFailure(
      TestProto("name: 'BadArg' input_arg { name: 'a' "
                "type_list_attr: 'x' } attr { name: 'x' type: 'type' }"),
      "Attr 'x' used as type_list_attr for input 'a' has type type");
  ExpectFailure(TestProto("name: 'BadArg' input_arg { name: 'a' "
                          "type_attr: 'x' }"),
                "No attr with name 'x' for input 'a'");
  ExpectFailure(
      TestProto("name: 'BadArg' input_arg { name: 'a' number_attr: 'n' "
                "type_attr: 'x' } attr { name: 'x' type: 'list(type)' } "
                "attr { name: 'n' type: 'int' has_minimum: true minimum: 1 }"),
      "Attr 'x' used as type_attr for input 'a' has type list(type)");
  // But list(type) is fine as the type of an arg without a number_attr:
  EXPECT_OK(TestProto(
      "name: 'Arg' input_arg { name: 'a' type_list_attr: 'x' } "
      "attr { name: 'x' type: 'list(type)' } attr { name: 'n' type: 'int' "
      "has_minimum: true minimum: 1 }"));

  // number_attr
  EXPECT_OK(TestProto(
      "name: 'Arg' input_arg { name: 'a' type: DT_INT32 number_attr: 'n' } "
      "attr { name: 'n' type: 'int' has_minimum: true minimum: 0 }"));

  ExpectFailure(TestProto("name: 'Arg' input_arg { name: 'a' type: DT_INT32 "
                          "number_attr: 'n' }"),
                "No attr with name 'n'");
  ExpectFailure(
      TestProto(
          "name: 'Arg' input_arg { name: 'a' type: "
          "DT_INT32 number_attr: 'n' } attr { name: 'n' type: 'string' }"),
      "Attr 'n' used as length for input 'a' has type string");
  ExpectFailure(
      TestProto("name: 'Arg' input_arg { name: 'a' type: "
                "DT_INT32 number_attr: 'n' } attr { name: 'n' type: 'int' }"),
      "Attr 'n' used as length for input 'a' must have minimum;");
  ExpectFailure(
      TestProto("name: 'Arg' input_arg { name: 'a' type: DT_INT32 number_attr: "
                "'n' } attr { name: 'n' type: 'int' has_minimum: true minimum: "
                "-5 }"),
      "Attr 'n' used as length for input 'a' must have minimum >= 0;");
  ExpectFailure(
      TestProto("name: 'Arg' input_arg { name: 'a' number_attr: 'n' } attr { "
                "name: 'n' type: 'int' has_minimum: true minimum: 2 }"),
      "Missing type for input 'a'; in OpDef:");
  ExpectFailure(TestProto("name: 'BadArg' input_arg { name: 'a' number_attr: "
                          "'n' type_list_attr: 'x' } attr { name: 'n' type: "
                          "'int' has_minimum: true minimum: 1 } attr { name: "
                          "'x' type: 'list(type)' }"),
                "Can't have both number_attr and type_list_attr for input 'a'");
}

}  // namespace
}  // namespace tensorflow
