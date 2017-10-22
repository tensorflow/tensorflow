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
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace tools {
namespace {

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
  %add = f32[2,4]{1,0} add(f32[2,4]{1,0} %multiply, f32[2,4]{1,0} %y)
}

)"
},
// pred constant
{
"ConstantPred",
R"(HloModule constant_pred_module:

ENTRY %constant_pred () -> pred[] {
  %constant = pred[] constant(true)
}

)"
},
// s32 constant
{
"ConstantS32",
R"(HloModule constant_s32_module:

ENTRY %constant_s32 () -> s32[] {
  %constant = s32[] constant(-42)
}

)"
},
// f32 constant, but the value is not a decimal
{
"ConstantF32", R"(HloModule ConstantF32_module:

ENTRY %ConstantF32.v4 () -> f32[] {
  %constant = f32[] constant(42)
}

)"
},
// constant + constant
{
"AddConstants",
R"(HloModule add_constants_module:

ENTRY %add_constants () -> f32[] {
  %constant = f32[] constant(3.14)
  %add = f32[] add(f32[] %constant, f32[] %constant)
}

)"
},
// v1 > v2 ? v1 : v2
{
"SelectR1F32",
R"(HloModule SelectR1F32WithCmpR1F32sFromParamsSmall_module:

ENTRY %SelectR1F32WithCmpR1F32sFromParamsSmall.v4 (v1: f32[4], v2: f32[4]) -> f32[4] {
  %v1 = f32[4]{0} parameter(0)
  %v2 = f32[4]{0} parameter(1)
  %greater-than = pred[4]{0} greater-than(f32[4]{0} %v1, f32[4]{0} %v2)
  %select = f32[4]{0} select(pred[4]{0} %greater-than, f32[4]{0} %v1, f32[4]{0} %v2)
}

)"
}
  });
  // clang-format on
}

class HloParserTest : public ::testing::Test,
                      public ::testing::WithParamInterface<TestData> {
 protected:
  void ExpectSuccess() {
    const string& original = GetParam().module_string;
    auto result = Parse(original);
    TF_EXPECT_OK(result.status());
    EXPECT_EQ(original, result.ValueOrDie()->ToString());
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
  %constant.1 = s32[] constant(-42)
  %constant = s32[] constant(42)
  %select = s32[] select(pred[] %constant.2, s32[] %constant.1, s32[] %constant)
}

)";
  auto result = Parse(original);
  TF_EXPECT_OK(result.status());
  // Constant instructions have no name. The string will be parsed successfully
  // but the constant names will not be exactly the same.
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

TEST_F(HloParserTest, Tuple) {
  const string original = R"(HloModule EmptyTupleCreate_module:

ENTRY %EmptyTupleCreate.v1 () -> () {
  %tuple = () tuple()
}

)";
  auto result = Parse(original);
  EXPECT_NE(tensorflow::Status::OK(), result.status());
}

}  // namespace
}  // namespace tools
}  // namespace xla
