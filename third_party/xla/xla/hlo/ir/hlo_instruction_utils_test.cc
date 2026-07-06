/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/hlo/ir/hlo_instruction_utils.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/shape_util.h"

namespace xla {

namespace hlo_instruction_utils {

namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

class HloInstructionUtilsTest : public HloHardwareIndependentTestBase {};

TEST_F(HloInstructionUtilsTest, TestIsUnstridedSlice) {
  const char* hlo_text = R"(
    HloModule test
    ENTRY main {
      param = f32[2,8] parameter(0)
      strided_slice = f32[2,2] slice(param), slice={[0:2:1], [4:8:2]}
      unstrided_slice = f32[2,4] slice(param), slice={[0:2:1], [4:8:1]}
      ROOT tuple = (f32[2,2], f32[2,4]) tuple(strided_slice, unstrided_slice)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_text));
  HloInstruction* unstrided_slice =
      hlo_query::FindInstruction(m->entry_computation(), "unstrided_slice");
  HloInstruction* strided_slice =
      hlo_query::FindInstruction(m->entry_computation(), "strided_slice");
  EXPECT_NE(unstrided_slice, nullptr);
  EXPECT_NE(strided_slice, nullptr);
  EXPECT_TRUE(IsUnstridedSlice(unstrided_slice));
  EXPECT_FALSE(IsUnstridedSlice(strided_slice));
}

TEST_F(HloInstructionUtilsTest, KeepsBitwidth) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(R"(
e {
  a = s8[2] parameter(0)
  b = s16[] bitcast(a)
  c = s16[] add(b, b)
})"));
  const HloInstruction& root = *m->entry_computation()->root_instruction();
  EXPECT_TRUE(KeepsBitwidth(root));
  EXPECT_FALSE(KeepsBitwidth(*root.operand(0)));
  EXPECT_TRUE(KeepsBitwidth(*root.operand(0)->operand(0)));
}

TEST_F(HloInstructionUtilsTest, TestAddOrUpdateVectorOfPairsAsAttribute) {
  const char* hlo = R"(
    HloModule test
    ENTRY main {
      ROOT param = s32[] parameter(0), frontend_attributes={foo="bar", baz="qux"}
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo));
  HloInstruction* param = m->entry_computation()->root_instruction();
  EXPECT_EQ(param->frontend_attributes().map().size(), 2);
  EXPECT_EQ(param->frontend_attributes().map().at("foo"), "bar");
  EXPECT_EQ(param->frontend_attributes().map().at("baz"), "qux");

  std::string new_key = "quux";
  std::vector<std::pair<int64_t, int64_t>> value = {{1, 2}, {3, 4}};
  AddOrUpdateVectorOfPairsAsAttribute(param, new_key, value);
  EXPECT_EQ(param->frontend_attributes().map().size(), 3);
  EXPECT_EQ(param->frontend_attributes().map().at("foo"), "bar");
  EXPECT_EQ(param->frontend_attributes().map().at("baz"), "qux");
  EXPECT_EQ(param->frontend_attributes().map().at("quux"), "{{1,2},{3,4}}");

  std::vector<std::pair<int64_t, int64_t>> new_value = {{5, 6}, {7, 8}};
  AddOrUpdateVectorOfPairsAsAttribute(param, new_key, new_value);
  EXPECT_EQ(param->frontend_attributes().map().size(), 3);
  EXPECT_EQ(param->frontend_attributes().map().at("foo"), "bar");
  EXPECT_EQ(param->frontend_attributes().map().at("baz"), "qux");
  EXPECT_EQ(param->frontend_attributes().map().at("quux"), "{{5,6},{7,8}}");
}

TEST_F(HloInstructionUtilsTest,
       AreOperandsAndOutputFullyBound_FullyBoundStart) {
  const char* const hlo = R"(
HloModule test

async_computation {
  p0 = f32[2,3] parameter(0)
  ROOT abs = f32[2,3] abs(p0)
}

ENTRY main {
  p0 = f32[2,3] parameter(0)
  sc-start = ((f32[2,3]), f32[2,3], s32[]) call-start(p0), to_apply=async_computation
  ROOT done = f32[2,3] call-done(sc-start)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  HloInstruction* start = FindInstruction(module.get(), "sc-start");
  EXPECT_THAT(async::AreOperandsAndOutputFullyBound(start, {1}),
              IsOkAndHolds(true));
}

TEST_F(HloInstructionUtilsTest,
       AreOperandsAndOutputFullyBound_FullyBoundUpdate) {
  const char* const hlo = R"(
HloModule test

async_computation {
  p0 = f32[2,3] parameter(0)
  ROOT abs = f32[2,3] abs(p0)
}

ENTRY main {
  p0 = f32[2,3] parameter(0)
  start = ((), (), s32[]) call-start(), to_apply=async_computation
  update = ((f32[2,3]), f32[2,3], ()) call-update(start, p0)
  ROOT done = f32[2,3] call-done(update)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  HloAsyncInstruction* start =
      Cast<HloAsyncInstruction>(FindInstruction(module.get(), "start"));
  ASSERT_NE(start, nullptr);
  EXPECT_THAT(async::AreOperandsAndOutputFullyBound(start, {0}),
              IsOkAndHolds(false));
  EXPECT_THAT(async::AreOperandsAndOutputFullyBound(start, {1}),
              IsOkAndHolds(false));
  EXPECT_THAT(async::AreOperandsAndOutputFullyBound(start, {0, 0}),
              IsOkAndHolds(false));
  HloAsyncInstruction* update =
      Cast<HloAsyncInstruction>(FindInstruction(module.get(), "update"));
  ASSERT_NE(update, nullptr);
  EXPECT_THAT(async::AreOperandsAndOutputFullyBound(update, {1}),
              IsOkAndHolds(true));
}

TEST_F(HloInstructionUtilsTest,
       AreOperandsAndOutputFullyBound_PartiallyBoundStart) {
  const char* const hlo = R"(
HloModule test

async_computation {
  p0 = f32[2,3] parameter(0)
  ROOT abs = f32[2,3] abs(p0)
}

ENTRY main {
  // Partially bound since no parameters are explicitly provided
  ROOT start = ((), (), s32[]) call-start(), to_apply=async_computation
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));

  HloAsyncInstruction* start =
      Cast<HloAsyncInstruction>(FindInstruction(module.get(), "start"));
  EXPECT_THAT(async::AreOperandsAndOutputFullyBound(start, {0}),
              IsOkAndHolds(false));
}

TEST_F(HloInstructionUtilsTest, AreOperandsAndOutputFullyBound_FullyBoundDone) {
  const char* const hlo = R"(
HloModule test

async_computation {
  p0 = f32[2,3] parameter(0)
  ROOT abs = f32[2,3] abs(p0)
}

ENTRY main {
  p0 = f32[2,3] parameter(0)
  start = ((f32[2,3]), f32[2,3], s32[]) call-start(p0), to_apply=async_computation
  ROOT done = f32[2,3] call-done(start)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  HloInstruction* done = FindInstruction(module.get(), "done");
  EXPECT_THAT(async::AreOperandsAndOutputFullyBound(done, {1}),
              IsOkAndHolds(true));
}

TEST_F(HloInstructionUtilsTest,
       AreOperandsAndOutputFullyBound_ErrorOnNonEmptyIndex) {
  const char* const hlo = R"(
HloModule test

async_computation {
  p0 = f32[2,3] parameter(0)
  ROOT abs = f32[2,3] abs(p0)
}

ENTRY main {
  p0 = f32[2,3] parameter(0)
  start = ((f32[2,3]), f32[2,3], s32[]) call-start(p0), to_apply=async_computation
  ROOT done = f32[2,3] call-done(start)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  HloInstruction* start = FindInstruction(module.get(), "start");
  EXPECT_THAT(async::AreOperandsAndOutputFullyBound(start, {1, 0}),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(async::AreOperandsAndOutputFullyBound(start, {2}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(HloInstructionUtilsTest,
       AreOperandsAndOutputFullyBound_EmptyIndexFullyBound2Tuple) {
  const char* const hlo = R"(
HloModule test

async_computation {
  p0 = f32[2,3] parameter(0)
  ROOT abs = f32[2,3] abs(p0)
}

ENTRY main {
  p0 = f32[2,3] parameter(0)
  sc-start = ((f32[2,3]), f32[2,3]) call-start(p0), to_apply=async_computation
  ROOT done = f32[2,3] call-done(sc-start)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  HloInstruction* start = FindInstruction(module.get(), "sc-start");
  EXPECT_THAT(async::AreOperandsAndOutputFullyBound(start), IsOkAndHolds(true));
}

TEST_F(HloInstructionUtilsTest,
       AreOperandsAndOutputFullyBound_EmptyIndexFullyBound3Tuple) {
  const char* const hlo = R"(
HloModule test

async_computation {
  p0 = f32[2,3] parameter(0)
  ROOT abs = f32[2,3] abs(p0)
}

ENTRY main {
  p0 = f32[2,3] parameter(0)
  sc-start = ((f32[2,3]), f32[2,3], s32[]) call-start(p0), to_apply=async_computation
  ROOT done = f32[2,3] call-done(sc-start)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  HloInstruction* start = FindInstruction(module.get(), "sc-start");
  EXPECT_THAT(async::AreOperandsAndOutputFullyBound(start), IsOkAndHolds(true));
}

TEST_F(HloInstructionUtilsTest,
       AreOperandsAndOutputFullyBound_EmptyIndexPartiallyBound) {
  const char* const hlo = R"(
HloModule test

async_computation {
  p0 = f32[2,3] parameter(0)
  ROOT abs = f32[2,3] abs(p0)
}

ENTRY main {
  ROOT start = ((), (), s32[]) call-start(), to_apply=async_computation
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  HloInstruction* start = FindInstruction(module.get(), "start");
  EXPECT_THAT(async::AreOperandsAndOutputFullyBound(start),
              IsOkAndHolds(false));
}

TEST_F(HloInstructionUtilsTest,
       AreOperandsAndOutputFullyBound_InvalidSubIndex) {
  const char* const hlo = R"(
HloModule test

async_computation {
  p0 = f32[2,3] parameter(0)
  p1 = f32[2,3] parameter(1)
  ROOT abs = f32[2,3] abs(p0)
}

ENTRY main {
  p0 = f32[2,3] parameter(0)
  start = ((), (), s32[]) call-start(), to_apply=async_computation
  // Partially bound, only binds p0 at index 0. Actual shape at {0} is (f32[2,3]).
  update = ((f32[2,3]), (), s32[]) call-update(start, p0)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  HloInstruction* update = FindInstruction(module.get(), "update");
  ASSERT_NE(update, nullptr);

  // Expected shape has parameters tuple (f32[2,3], f32[2,3]), so index {0, 1}
  // is valid in expected_shape. However, in `update`, only 1 parameter is
  // bound, so index {0, 1} is invalid in the actual shape.
  // Verify that the function returns false safely.
  ASSERT_OK_AND_ASSIGN(bool is_bound,
                       async::AreOperandsAndOutputFullyBound(update, {0, 1}));
  EXPECT_FALSE(is_bound);
}

TEST_F(HloInstructionUtilsTest,
       AreOperandsAndOutputFullyBound_InvalidOutputSubIndex) {
  const char* const hlo = R"(
HloModule test

async_computation {
  p0 = f32[2,3] parameter(0)
  abs = f32[2,3] abs(p0)
  ROOT tuple = (f32[2,3], f32[2,3]) tuple(abs, abs)
}

ENTRY main {
  p0 = f32[2,3] parameter(0)
  start = ((), (), s32[]) call-start(), to_apply=async_computation
  // Output not bound, shape at {1} is ().
  update = ((f32[2,3]), (), s32[]) call-update(start, p0)
  ROOT done = (f32[2,3], f32[2,3]) call-done(update)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  HloInstruction* update = FindInstruction(module.get(), "update");
  ASSERT_NE(update, nullptr);

  // Expected output has tuple shape (f32[2,3], f32[2,3]), so index {1, 1} is
  // valid in expected_shape. However, in `update`, the output is not bound, so
  // index {1, 1} is invalid in output_shape.
  // Verify that the function returns false safely.
  ASSERT_OK_AND_ASSIGN(bool is_bound,
                       async::AreOperandsAndOutputFullyBound(update, {1, 1}));
  EXPECT_FALSE(is_bound);
}

TEST_F(HloInstructionUtilsTest,
       AreOperandsAndOutputFullyBound_DoneBoundOutput) {
  const char* const hlo = R"(
HloModule test

async_computation {
  p0 = f32[2,3] parameter(0)
  ROOT abs = f32[2,3] abs(p0)
}

ENTRY main {
  p0 = f32[2,3] parameter(0)
  start = ((), (), s32[]) call-start(), to_apply=async_computation
  // Output is not bound here (index 1 is ()).
  update = ((f32[2,3]), (), s32[]) call-update(start, p0)
  ROOT done = f32[2,3] call-done(update)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  HloInstruction* done = FindInstruction(module.get(), "done");
  ASSERT_NE(done, nullptr);

  // Verify that even though update doesn't bind the output, done does, making
  // it fully bound.
  EXPECT_THAT(async::AreOperandsAndOutputFullyBound(done), IsOkAndHolds(true));
}

TEST_F(HloInstructionUtilsTest,
       AreOperandsAndOutputFullyBound_NonAsyncInstruction) {
  const char* const hlo = R"(
HloModule test

ENTRY main {
  p0 = f32[2,3] parameter(0)
  ROOT add = f32[2,3] add(p0, p0)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  HloInstruction* add = FindInstruction(module.get(), "add");
  EXPECT_THAT(
      async::AreOperandsAndOutputFullyBound(add),
      StatusIs(absl::StatusCode::kInvalidArgument,
               ::testing::HasSubstr("Instruction is not asynchronous")));
}

TEST_F(HloInstructionUtilsTest,
       AreOperandsAndOutputFullyBound_NonTupleAsyncShape) {
  const char* const hlo = R"(
HloModule test

async_computation {
  p0 = f32[2,3] parameter(0)
  ROOT abs = f32[2,3] abs(p0)
}

ENTRY main {
  p0 = f32[2,3] parameter(0)
  ROOT start = ((f32[2,3]), f32[2,3]) call-start(p0), to_apply=async_computation
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  HloInstruction* start = FindInstruction(module.get(), "start");
  ASSERT_NE(start, nullptr);
  *start->mutable_shape() = ShapeUtil::MakeShape(F32, {2, 3});
  EXPECT_THAT(async::AreOperandsAndOutputFullyBound(start),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       ::testing::HasSubstr(
                           "Expected async tuple shape to be a tuple")));
}

TEST_F(HloInstructionUtilsTest,
       AreOperandsAndOutputFullyBound_TooSmallTupleAsyncShape) {
  const char* const hlo = R"(
HloModule test

async_computation {
  p0 = f32[2,3] parameter(0)
  ROOT abs = f32[2,3] abs(p0)
}

ENTRY main {
  p0 = f32[2,3] parameter(0)
  ROOT start = ((f32[2,3]), f32[2,3]) call-start(p0), to_apply=async_computation
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  HloInstruction* start = FindInstruction(module.get(), "start");
  ASSERT_NE(start, nullptr);
  *start->mutable_shape() =
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {2, 3})});
  EXPECT_THAT(async::AreOperandsAndOutputFullyBound(start),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       ::testing::HasSubstr(
                           "Expected async tuple shape to be a tuple")));
}

TEST_F(HloInstructionUtilsTest,
       AreOperandsAndOutputFullyBound_AsyncDoneNoOperands) {
  const char* const hlo = R"(
HloModule test

async_computation {
  p0 = f32[2,3] parameter(0)
  ROOT abs = f32[2,3] abs(p0)
}

ENTRY main {
  p0 = f32[2,3] parameter(0)
  start = ((f32[2,3]), f32[2,3]) call-start(p0), to_apply=async_computation
  ROOT done = f32[2,3] call-done(start)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  HloInstruction* done = FindInstruction(module.get(), "done");
  ASSERT_NE(done, nullptr);
  HloInstruction* start = FindInstruction(module.get(), "start");
  ASSERT_NE(start, nullptr);
  start->DetachFromOperandsAndUsers();
  EXPECT_THAT(
      async::AreOperandsAndOutputFullyBound(done),
      StatusIs(absl::StatusCode::kInvalidArgument,
               ::testing::HasSubstr("is not part of a valid async chain")));
}

TEST_F(HloInstructionUtilsTest,
       AreOperandsAndOutputFullyBound_NoWrappedComputation) {
  const char* const hlo = R"(
HloModule test

async_computation {
  p0 = f32[2,3] parameter(0)
  ROOT abs = f32[2,3] abs(p0)
}

ENTRY main {
  p0 = f32[2,3] parameter(0)
  ROOT start = ((f32[2,3]), f32[2,3]) call-start(p0), to_apply=async_computation
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  HloInstruction* start = FindInstruction(module.get(), "start");
  ASSERT_NE(start, nullptr);
  start->ClearCalledComputations();
  EXPECT_THAT(
      async::AreOperandsAndOutputFullyBound(start),
      StatusIs(absl::StatusCode::kInvalidArgument,
               ::testing::HasSubstr("has no valid wrapped computation")));
}

TEST_F(HloInstructionUtilsTest, GetAsyncBoundOperandsTest) {
  const char* const hlo = R"(
HloModule test

async_computation {
  p0 = f32[2,3] parameter(0)
  p1 = f32[2,3] parameter(1)
  ROOT abs = f32[2,3] abs(p0)
}

ENTRY main {
  p0 = f32[2,3] parameter(0)
  p1 = f32[2,3] parameter(1)
  start = ((f32[2,3]), f32[2,3], s32[]) call-start(p0), to_apply=async_computation
  update = ((f32[2,3], f32[2,3]), f32[2,3], ()) call-update(start, p1)
  ROOT done = f32[2,3] call-done(update)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  HloAsyncInstruction* start =
      Cast<HloAsyncInstruction>(FindInstruction(module.get(), "start"));
  HloAsyncInstruction* update =
      Cast<HloAsyncInstruction>(FindInstruction(module.get(), "update"));
  HloAsyncInstruction* done =
      Cast<HloAsyncInstruction>(FindInstruction(module.get(), "done"));
  HloInstruction* p0 = module->entry_computation()->parameter_instruction(0);
  HloInstruction* p1 = module->entry_computation()->parameter_instruction(1);

  EXPECT_THAT(async::GetAsyncBoundOperands(start), ::testing::ElementsAre(p0));
  EXPECT_THAT(async::GetAsyncBoundOperands(update),
              ::testing::ElementsAre(p0, p1));
  EXPECT_THAT(async::GetAsyncBoundOperands(done),
              ::testing::ElementsAre(p0, p1));
}

TEST_F(HloInstructionUtilsTest, IsFirstFullyBound_LateBinding) {
  const char* const hlo = R"(
HloModule test

async_computation {
  p0 = f32[2,3] parameter(0)
  ROOT abs = f32[2,3] abs(p0)
}

ENTRY main {
  p0 = f32[2,3] parameter(0)
  start = ((), (), s32[]) call-start(), to_apply=async_computation
  update = ((f32[2,3]), f32[2,3], ()) call-update(start, p0)
  ROOT done = f32[2,3] call-done(update)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  HloInstruction* start = FindInstruction(module.get(), "start");
  HloInstruction* update = FindInstruction(module.get(), "update");
  HloInstruction* done = FindInstruction(module.get(), "done");

  EXPECT_THAT(async::IsFirstFullyBound(start), IsOkAndHolds(false));
  EXPECT_THAT(async::IsFirstFullyBound(update), IsOkAndHolds(true));
  EXPECT_THAT(async::IsFirstFullyBound(done), IsOkAndHolds(false));
}

TEST_F(HloInstructionUtilsTest, IsFirstFullyBound_EarlyBinding) {
  const char* const hlo = R"(
HloModule test

async_computation {
  p0 = f32[2,3] parameter(0)
  ROOT abs = f32[2,3] abs(p0)
}

ENTRY main {
  p0 = f32[2,3] parameter(0)
  start = ((f32[2,3]), f32[2,3], s32[]) call-start(p0), to_apply=async_computation
  ROOT done = f32[2,3] call-done(start)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  HloInstruction* start = FindInstruction(module.get(), "start");
  HloInstruction* done = FindInstruction(module.get(), "done");

  EXPECT_THAT(async::IsFirstFullyBound(start), IsOkAndHolds(true));
  EXPECT_THAT(async::IsFirstFullyBound(done), IsOkAndHolds(false));
}
TEST_F(HloInstructionUtilsTest, IsFirstFullyBound_Parameterless) {
  const char* const hlo = R"(
HloModule test

async_computation {
  constant = f32[2,3] constant(1.0)
  ROOT abs = f32[2,3] abs(constant)
}

ENTRY main {
  start = ((), f32[2,3], s32[]) call-start(), to_apply=async_computation
  ROOT done = f32[2,3] call-done(start)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  HloInstruction* start = FindInstruction(module.get(), "start");
  HloInstruction* done = FindInstruction(module.get(), "done");

  EXPECT_THAT(async::IsFirstFullyBound(start), IsOkAndHolds(true));
  EXPECT_THAT(async::IsFirstFullyBound(done), IsOkAndHolds(false));
}

TEST_F(HloInstructionUtilsTest, IsFirstFullyBound_LateBinding_MultipleUpdates) {
  const char* const hlo = R"(
HloModule test

async_computation {
  p0 = f32[2,3] parameter(0)
  p1 = f32[2,3] parameter(1)
  ROOT abs = f32[2,3] abs(p0)
}

ENTRY main {
  p0 = f32[2,3] parameter(0)
  p1 = f32[2,3] parameter(1)
  start = ((), (), s32[]) call-start(), to_apply=async_computation
  update1 = ((f32[2,3]), (), s32[]) call-update(start, p0)
  update2 = ((f32[2,3], f32[2,3]), f32[2,3], ()) call-update(update1, p1)
  ROOT done = f32[2,3] call-done(update2)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  HloInstruction* start = FindInstruction(module.get(), "start");
  HloInstruction* update1 = FindInstruction(module.get(), "update1");
  HloInstruction* update2 = FindInstruction(module.get(), "update2");
  HloInstruction* done = FindInstruction(module.get(), "done");

  EXPECT_THAT(async::IsFirstFullyBound(start), IsOkAndHolds(false));
  EXPECT_THAT(async::IsFirstFullyBound(update1), IsOkAndHolds(false));
  EXPECT_THAT(async::IsFirstFullyBound(update2), IsOkAndHolds(true));
  EXPECT_THAT(async::IsFirstFullyBound(done), IsOkAndHolds(false));
}

TEST_F(HloInstructionUtilsTest, IsFirstFullyBound_NonAsyncInstruction) {
  const char* const hlo = R"(
HloModule test

ENTRY main {
  p0 = f32[2,3] parameter(0)
  ROOT add = f32[2,3] add(p0, p0)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  HloInstruction* add = FindInstruction(module.get(), "add");
  EXPECT_THAT(
      async::IsFirstFullyBound(add),
      StatusIs(absl::StatusCode::kInvalidArgument,
               ::testing::HasSubstr("Instruction is not asynchronous")));
}
}  // namespace

}  // namespace hlo_instruction_utils

}  // namespace xla
