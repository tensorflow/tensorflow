/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/tools/parser/hlo_parser.h"

#include <string>
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace tools {
namespace {

using tensorflow::StringPiece;

struct TestData {
  string test_name;
  string module_string;
};

string TestDataToString(const ::testing::TestParamInfo<TestData>& data) {
  return data.param.test_name;
}

std::vector<TestData> CreateTestCases() {
  // clang-format off
  return std::vector<TestData>({
// ax + y
{
"AxpyParam",
R"(HloModule axpy_module:

ENTRY %axpy.v5 (alpha: f32[2,4], x: f32[2,4], y: f32[2,4]) -> f32[2,4] {
  %alpha = f32[2,4]{1,0} parameter(0)
  %x = f32[2,4]{1,0} parameter(1)
  %multiply = f32[2,4]{1,0} multiply(f32[2,4]{1,0} %alpha, f32[2,4]{1,0} %x)
  %y = f32[2,4]{1,0} parameter(2)
  ROOT %add = f32[2,4]{1,0} add(f32[2,4]{1,0} %multiply, f32[2,4]{1,0} %y)
}

)"
},
// pred constant
{
"ConstantPred",
R"(HloModule constant_pred_module:

ENTRY %constant_pred () -> pred[] {
  ROOT %constant = pred[] constant(true)
}

)"
},
// s32 constant
{
"ConstantS32",
R"(HloModule constant_s32_module:

ENTRY %constant_s32 () -> s32[] {
  ROOT %constant = s32[] constant(-42)
}

)"
},
// f32 constant, but the value is not a decimal
{
"ConstantF32", R"(HloModule ConstantF32_module:

ENTRY %ConstantF32.v4 () -> f32[] {
  ROOT %constant = f32[] constant(42)
}

)"
},
// constant 4D
{
"Constant4D",
R"(HloModule Small_3x2x1x1_module:

ENTRY %Small_3x2x1x1.v1 () -> f32[3,2,1,1] {
  ROOT %constant = f32[3,2,1,1]{3,2,1,0} constant(f32[3,2,1,1] { { /*i0=0*/ { /*i1=0*/ {-1} }, { /*i1=1*/ {4.1} } }, { /*i0=1*/ { /*i1=0*/ {2} }, { /*i1=1*/ {4.1} } }, { /*i0=2*/ { /*i1=0*/ {5} }, { /*i1=1*/ {4.4} } } })
}

)"
},
// non-finite constants: nan, inf, -inf
{
"ConstantNonFinite",
R"(HloModule IsFiniteR1F32s_module:

ENTRY %IsFiniteR1F32s.v2 () -> pred[6] {
  %constant = f32[6]{0} constant({nan, 7, nan, -1, inf, -inf})
  ROOT %is-finite = pred[6]{0} is-finite(f32[6]{0} %constant)
}

)"
},
// constant f16
{
"ConstantF16",
R"(HloModule ConstantF16_module:

ENTRY %ConstantF16.v4 () -> f16[] {
  ROOT %constant = f16[] constant(500)
}

)"
},
// constant + constant
{
"AddConstants",
R"(HloModule add_constants_module:

ENTRY %add_constants () -> f32[] {
  %constant = f32[] constant(3.14)
  ROOT %add = f32[] add(f32[] %constant, f32[] %constant)
}

)"
},
// tuple constant
{
"TupleConstant",
R"(HloModule TupleConstant_module:

ENTRY %TupleConstant.v1 () -> (f32[2,1], f32[2]) {
  ROOT %constant = (f32[2,1]{1,0}, f32[2]{0}) constant((f32[2,1], f32[2]) ( f32[2,1] { { 1 }, { 2 } }, {2, 42} ))
}

)"
},
// v1 > v2 ? v1 : v2
{
"SelectR1F32",
R"(HloModule SelectR1F32WithCmpR1F32sFromParamsSmall_module:

ENTRY %SelectR1F32WithCmpR1F32sFromParamsSmall.v4 (v1: f32[4], v2: f32[4]) -> f32[4] {
  %v1 = f32[4]{0} parameter(0), sharding={maximal device=1}
  %v2 = f32[4]{0} parameter(1), sharding={maximal device=1}
  %greater-than = pred[4]{0} greater-than(f32[4]{0} %v1, f32[4]{0} %v2), sharding={replicated}
  ROOT %select = f32[4]{0} select(pred[4]{0} %greater-than, f32[4]{0} %v1, f32[4]{0} %v2)
}

)"
},
// empty tuple
{
"EmptyTupleCreate",
R"(HloModule EmptyTupleCreate_module:

ENTRY %EmptyTupleCreate.v1 () -> () {
  ROOT %tuple = () tuple()
}

)"
},
// tuple
{
"TupleCreate",
R"(HloModule TupleCreate_module:

ENTRY %TupleCreate.v4 (v1: f32[], v2: f32[3], v3: f32[2,3]) -> (f32[], f32[3], f32[2,3]) {
  %v1 = f32[] parameter(0)
  %v2 = f32[3]{0} parameter(1)
  %v3 = f32[2,3]{1,0} parameter(2)
  ROOT %tuple = (f32[], f32[3]{0}, f32[2,3]{1,0}) tuple(f32[] %v1, f32[3]{0} %v2, f32[2,3]{1,0} %v3)
}

)"
},
// int32 result = 0;
// while (result < 5) { result = result + 1; }
{
"WhileWithScalarS32Result",
R"(HloModule WhileWithScalarS32Result_module:

%body.v3 (prev.1: s32[]) -> s32[] {
  %constant = s32[] constant(1)
  %prev.1 = s32[] parameter(0)
  ROOT %add = s32[] add(s32[] %constant, s32[] %prev.1)
}

%condition.v3 (prev.2: s32[]) -> pred[] {
  %constant.1 = s32[] constant(5)
  %prev.2 = s32[] parameter(0)
  ROOT %greater-than = pred[] greater-than(s32[] %constant.1, s32[] %prev.2)
}

ENTRY %WhileWithScalarS32Result.v2 () -> s32[] {
  %constant.2 = s32[] constant(0)
  ROOT %while = s32[] while(s32[] %constant.2), condition=%condition.v3, body=%body.v3
}

)"
},
// send and recv
{
"SendRecv",
R"(HloModule TwoSendRecvBothWayRecvFist_module:

ENTRY %TwoSendRecvBothWayRecvFist.v3 () -> f32[] {
  %recv = f32[] recv(), channel_id=15, sharding={maximal device=1}
  ROOT %constant = f32[] constant(2.1), sharding={maximal device=0}
  %send = () send(f32[] %constant), channel_id=16, sharding={maximal device=0}
}

)"
},
// get-tuple-element
{
"GetTupleElement",
R"(HloModule GetTupleElement_module:

ENTRY %GetTupleElement.v4 () -> s32[2,3] {
  %constant = f32[3]{0} constant({1, 2, 3})
  %constant.1 = s32[2,3]{1,0} constant(s32[2,3] { { 1, 2, 3 }, { 4, 5, 6 } })
  %tuple = (f32[3]{0}, s32[2,3]{1,0}) tuple(f32[3]{0} %constant, s32[2,3]{1,0} %constant.1)
  ROOT %get-tuple-element = s32[2,3]{1,0} get-tuple-element((f32[3]{0}, s32[2,3]{1,0}) %tuple), index=1, sharding={maximal device=0}
}

)"
},
// call
{
"Call",
R"(HloModule CallR0F32IdentityScalar_module:

%Identity.v1 (x: f32[]) -> f32[] {
  ROOT %x = f32[] parameter(0)
}

ENTRY %CallR0F32IdentityScalar.v2 () -> f32[] {
  %constant = f32[] constant(42)
  ROOT %call = f32[] call(f32[] %constant), to_apply=%Identity.v1
}

)"
}
  });
  // clang-format on
}

class HloParserTest : public ::testing::Test,
                      public ::testing::WithParamInterface<TestData> {
 protected:
  static void ExpectHasSubstr(StringPiece s, StringPiece expected) {
    EXPECT_TRUE(StringPiece(s).contains(expected))
        << "'" << s << "' does not contain '" << expected << "'";
  }

  void ExpectSuccess() {
    const string& original = GetParam().module_string;
    auto result = Parse(original);
    TF_EXPECT_OK(result.status());
    EXPECT_EQ(original,
              result.ValueOrDie()->ToString(/*include_large_constants=*/true));
  }
};

TEST_P(HloParserTest, Run) { ExpectSuccess(); }

INSTANTIATE_TEST_CASE_P(HloParserTestSuccessInstantiation, HloParserTest,
                        ::testing::ValuesIn(CreateTestCases()),
                        TestDataToString);

TEST_F(HloParserTest, Empty) {
  const string original = "";
  auto result = Parse(original);
  EXPECT_NE(tensorflow::Status::OK(), result.status());
}

TEST_F(HloParserTest, Garbage) {
  const string original = "HloModule thi$ str1ng makes# N0 sen$e @all!*&^%$";
  auto result = Parse(original);
  EXPECT_NE(tensorflow::Status::OK(), result.status());
}

TEST_F(HloParserTest, WrongOpcode) {
  const string original = R"(HloModule wrong_opcode:

ENTRY %blabla (x: f32[], y: f32[]) -> f32[] {
  %x = f32[]{} parameter(0)
  %y = f32[]{} parameter(1)
  %le = pred[]{} le(f32[]{} %x, f32[]{} %y)
}

)";
  auto result = Parse(original);
  EXPECT_NE(tensorflow::Status::OK(), result.status());
}

TEST_F(HloParserTest, WrongShape) {
  const string original = R"(HloModule wrong_opcode:

ENTRY %blabla (x: g32[]) -> g32[] {
  %x = g32[]{} parameter(0)
}

)";
  auto result = Parse(original);
  EXPECT_NE(tensorflow::Status::OK(), result.status());
}

TEST_F(HloParserTest, WrongOperandsSize) {
  const string original = R"(HloModule wrong_opcode:

ENTRY %blabla (x: f32[]) -> pred[] {
  %x = f32[]{} parameter(0)
  %eq = pred[]{} equal-to(f32[]{} %x)
}

)";
  auto result = Parse(original);
  EXPECT_NE(tensorflow::Status::OK(), result.status());
}

TEST_F(HloParserTest, OperandNotFound) {
  const string original = R"(HloModule operand_not_found:
ENTRY %blabla (x: f32[]) -> pred[] {
  %x = f32[]{} parameter(0)
  %eq = pred[]{} equal-to(f32[]{} %x, f32[]{} %y)
}
)";
  auto result = Parse(original);
  EXPECT_NE(tensorflow::Status::OK(), result.status());
}

TEST_F(HloParserTest, MoreConstants) {
  const string original = R"(HloModule SelectScalarS32True_module:

ENTRY %SelectScalarS32True.v4 () -> s32[] {
  %constant.2 = pred[] constant(true)
  %constant.1 = s32[] constant(-42), sharding={s32[5,6] devices=[2,3]1,2,3,4}
  %constant = s32[] constant(42)
  %select = s32[] select(pred[] %constant.2, s32[] %constant.1, s32[] %constant)
}

)";
  auto result = Parse(original);
  TF_EXPECT_OK(result.status());
  // Constant instructions have no name. The string will be parsed successfully
  // but the constant names will not be exactly the same.
}

TEST_F(HloParserTest, LiteralDimensionsMismatch_1) {
  const string original = R"(HloModule some_2_module:

ENTRY %some_2 () -> f32[2] {
  ROOT %constant = f32[2]{0} constant({1,{2}})
}

)";
  auto result = Parse(original);
  EXPECT_NE(tensorflow::Status::OK(), result.status());
  ExpectHasSubstr(result.status().error_message(),
                  "expects nested array in rank 1, but sees larger");
}

TEST_F(HloParserTest, LiteralDimensionsMismatch_2) {
  const string original = R"(HloModule some_2x3_module:

ENTRY %some_2x3 () -> f32[2,3] {
  ROOT %constant = f32[2,3]{1,0} constant(f32[2,3] {1, 2, 3, 4, 5, 6})
}

)";
  auto result = Parse(original);
  EXPECT_NE(tensorflow::Status::OK(), result.status());
  ExpectHasSubstr(result.status().error_message(),
                  "expects nested array in rank 2, but sees 1");
}

TEST_F(HloParserTest, LiteralDimensionsMismatch_3) {
  const string original = R"(HloModule some_2x3x2_module:

ENTRY %some_2x3x2 () -> f32[2,3,2] {
  ROOT %constant = f32[2,3,2]{2,1,0} constant(f32[2,3,2] {{{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}, {11, 12}}})
}

)";
  auto result = Parse(original);
  EXPECT_NE(tensorflow::Status::OK(), result.status());
  ExpectHasSubstr(result.status().error_message(),
                  "expects 3 elements in the [0]th element");
}

TEST_F(HloParserTest, ConstantF16Overflow) {
  const string original =
      R"(HloModule ConstantF16Overflow_module:

ENTRY %ConstantF16Overflow.v4 () -> f16[] {
  ROOT %constant = f16[] constant(-65505)
}

)";
  auto result = Parse(original);
  EXPECT_NE(tensorflow::Status::OK(), result.status());
  ExpectHasSubstr(result.status().error_message(),
                  "is out of range for literal's primitive type F16");
}

TEST_F(HloParserTest, ConstantWithExp) {
  const string original = R"(HloModule ConstantWithExp_module:

ENTRY %ConstantWithExp.v4 () -> f32[] {
  %constant.1 = f32[] constant(3e+2)
}

)";
  auto result = Parse(original);
  TF_EXPECT_OK(result.status());
  // The string will be parsed successfully but the output strings are not
  // exactly the same, because "3e2" is parsed into value 300 and will be
  // printed as "300".
}

}  // namespace
}  // namespace tools
}  // namespace xla
