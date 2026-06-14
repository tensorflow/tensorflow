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

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

namespace hlo_instruction_utils {

namespace {

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
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  HloInstruction* start = FindInstruction(module.get(), "sc-start");
  EXPECT_TRUE(async::AreOperandsAndOutputFullyBound(start, {1}).value());
}

TEST_F(HloInstructionUtilsTest,
       AreOperandsAndOutputFullyBound_FullyBoundUpdate) {
  const char* const hlo = R"(
HloModule test

async_computation {
  p0 = f32[2,3] parameter(0)
  abs = f32[2,3] abs(p0)
  ROOT tuple = (f32[2,3]) tuple(abs)
}

ENTRY main {
  p0 = f32[2,3] parameter(0)
  start = ((), (), s32[]) call-start(), to_apply=async_computation
  update = ((f32[2,3]), f32[2,3], ()) call-update(start, p0)
  ROOT done = f32[2,3] call-done(update)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  HloAsyncInstruction* start =
      Cast<HloAsyncInstruction>(FindInstruction(module.get(), "start"));
  ASSERT_NE(start, nullptr);
  EXPECT_FALSE(async::AreOperandsAndOutputFullyBound(start, {0}).value());
  EXPECT_FALSE(async::AreOperandsAndOutputFullyBound(start, {1}).value());
  EXPECT_FALSE(async::AreOperandsAndOutputFullyBound(start, {0, 0}).value());
  HloAsyncInstruction* update =
      Cast<HloAsyncInstruction>(FindInstruction(module.get(), "update"));
  ASSERT_NE(update, nullptr);
  EXPECT_TRUE(async::AreOperandsAndOutputFullyBound(update, {1}).value());
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
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));

  HloAsyncInstruction* start =
      Cast<HloAsyncInstruction>(FindInstruction(module.get(), "start"));
  EXPECT_FALSE(async::AreOperandsAndOutputFullyBound(start, {0}).value());
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
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  HloInstruction* done = FindInstruction(module.get(), "done");
  EXPECT_TRUE(async::AreOperandsAndOutputFullyBound(done, {1}).value());
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
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  HloInstruction* start = FindInstruction(module.get(), "start");
  // TODO: update this test
  EXPECT_FALSE(async::AreOperandsAndOutputFullyBound(start, {1, 0}).ok());
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
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  HloInstruction* start = FindInstruction(module.get(), "sc-start");
  EXPECT_TRUE(async::AreOperandsAndOutputFullyBound(start).value());
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
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  HloInstruction* start = FindInstruction(module.get(), "sc-start");
  EXPECT_TRUE(async::AreOperandsAndOutputFullyBound(start).value());
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
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  HloInstruction* start = FindInstruction(module.get(), "start");
  EXPECT_FALSE(async::AreOperandsAndOutputFullyBound(start).value());
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
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  HloInstruction* start = FindInstruction(module.get(), "start");
  HloInstruction* update = FindInstruction(module.get(), "update");
  HloInstruction* p0 = module->entry_computation()->parameter_instruction(0);
  HloInstruction* p1 = module->entry_computation()->parameter_instruction(1);

  EXPECT_EQ(async::GetAsyncBoundOperands(start).size(), 1);
  EXPECT_EQ(async::GetAsyncBoundOperands(start)[0], p0);

  EXPECT_EQ(async::GetAsyncBoundOperands(update).size(), 2);
  EXPECT_EQ(async::GetAsyncBoundOperands(update)[0], p0);
  EXPECT_EQ(async::GetAsyncBoundOperands(update)[1], p1);
}

}  // namespace

}  // namespace hlo_instruction_utils

}  // namespace xla
