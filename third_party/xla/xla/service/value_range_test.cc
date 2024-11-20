/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/value_range.h"

#include <utility>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/constant_value.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

class ValueRangeTest : public HloTestBase {};

TEST_F(ValueRangeTest, AddedValue) {
  constexpr absl::string_view hlo_string = R"(
  HloModule module

  ENTRY entry {
    c0 = s32[] constant(124)
    p0 = s32[] parameter(0)
    ROOT %a = s32[] add(p0, c0)
  }
  )";
  auto module =
      ParseAndReturnUnverifiedModule(hlo_string, HloModuleConfig{}).value();
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* p0 = root->operand(0);
  absl::flat_hash_map<const HloInstruction*, Range> fs;
  fs.insert(
      std::make_pair(p0, Range{ConstantValue::GetZero(32, /*is_signed=*/true),
                               ConstantValue::GetSigned(5, 32),
                               ConstantValue::GetOne(32, /*is_signed=*/false),
                               /*is_linear=*/true}));
  auto range = RecursivelyIdentifyRange(root, fs);
  EXPECT_FALSE(range.IsEmpty());
  EXPECT_FALSE(range.IsSingleValue());
  EXPECT_TRUE(range.IsLinear());
  EXPECT_EQ(range.min().GetSignedValue(), 124);
  EXPECT_EQ(range.max().GetSignedValue(), 129);
  EXPECT_EQ(range.step().GetSignedValue(), 1);
}

TEST_F(ValueRangeTest, MultiplyValue) {
  constexpr absl::string_view hlo_string = R"(
  HloModule module

  ENTRY entry {
    c0 = s32[] constant(1024)
    p0 = s32[] parameter(0)
    ROOT %a = s32[] multiply(p0, c0)
  }
  )";
  auto module =
      ParseAndReturnUnverifiedModule(hlo_string, HloModuleConfig{}).value();
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* p0 = root->operand(0);
  absl::flat_hash_map<const HloInstruction*, Range> fs;
  fs.insert(
      std::make_pair(p0, Range{ConstantValue::GetZero(32, /*is_signed=*/true),
                               ConstantValue::GetSigned(5, 32),
                               ConstantValue::GetOne(32, /*is_signed=*/false),
                               /*is_linear=*/true}));
  auto range = RecursivelyIdentifyRange(root, fs);
  EXPECT_FALSE(range.IsEmpty());
  EXPECT_FALSE(range.IsSingleValue());
  EXPECT_TRUE(range.IsLinear());
  EXPECT_EQ(range.min().GetSignedValue(), 0);
  EXPECT_EQ(range.max().GetSignedValue(), 5120);
  EXPECT_EQ(range.step().GetSignedValue(), 1);
}

TEST_F(ValueRangeTest, ConstantValueWithConditional) {
  constexpr absl::string_view hlo_string = R"(
  HloModule module
  region1 {
    region1_param = s32[] parameter(0)
    region1_c0 = s32[] constant(1024)
    %add = s32[] add(region1_param, region1_c0)
    ROOT out = (s32[], s32[]) tuple(%add, %add)
  }
  region2 {
    region2_param = s32[] parameter(0)
    region2_c0 = s32[] constant(1024)
    %mult = s32[] multiply(region2_param, region2_c0)
    ROOT out = (s32[], s32[]) tuple(%mult, %mult)
  }
  ENTRY entry {
    p0 = s32[] parameter(0)
    branch_index = s32[] parameter(1)
    ROOT conditional.1 = (s32[], s32[]) conditional(branch_index, p0, p0), branch_computations={region1, region2}
  }
  )";
  auto module =
      ParseAndReturnUnverifiedModule(hlo_string, HloModuleConfig{}).value();
  TF_ASSERT_OK_AND_ASSIGN(auto alias_analysis,
                          HloAliasAnalysis::Run(module.get()));
  HloComputation* region1 = module->GetComputationWithName("region1");
  HloComputation* region2 = module->GetComputationWithName("region2");
  HloInstruction* add = region1->GetInstructionWithName("add");
  HloInstruction* mult = region2->GetInstructionWithName("mult");
  const HloInstruction* p0 =
      module->entry_computation()->parameter_instruction(0);
  absl::flat_hash_map<const HloInstruction*, Range> fs;
  fs.insert(
      std::make_pair(p0, Range{ConstantValue::GetZero(32, /*is_signed=*/true),
                               ConstantValue::GetSigned(5, 32),
                               ConstantValue::GetOne(32, /*is_signed=*/false),
                               /*is_linear=*/true}));

  auto add_range = RecursivelyIdentifyRange(add, fs, alias_analysis.get());
  EXPECT_FALSE(add_range.IsEmpty());
  EXPECT_FALSE(add_range.IsSingleValue());
  EXPECT_TRUE(add_range.IsLinear());
  EXPECT_EQ(add_range.min().GetSignedValue(), 1024);
  EXPECT_EQ(add_range.max().GetSignedValue(), 1029);
  EXPECT_EQ(add_range.step().GetSignedValue(), 1);

  auto mult_range = RecursivelyIdentifyRange(mult, fs, alias_analysis.get());
  EXPECT_FALSE(mult_range.IsEmpty());
  EXPECT_FALSE(mult_range.IsSingleValue());
  EXPECT_TRUE(mult_range.IsLinear());
  EXPECT_EQ(mult_range.min().GetSignedValue(), 0);
  EXPECT_EQ(mult_range.max().GetSignedValue(), 5120);
  EXPECT_EQ(mult_range.step().GetSignedValue(), 1);
}

TEST_F(ValueRangeTest, AddedValueUnsigned) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  c0 = u16[] constant(32768)
  p0 = u16[] parameter(0)
  ROOT %a = u16[] add(p0, c0)
}
)";
  auto module =
      ParseAndReturnUnverifiedModule(hlo_string, HloModuleConfig{}).value();
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* p0 = root->operand(0);
  absl::flat_hash_map<const HloInstruction*, Range> fs;
  fs.insert(std::make_pair(
      p0, Range{ConstantValue::GetZero(32, /*is_signed=*/false),
                ConstantValue::GetUnsigned(5, 32), /*is_linear=*/true}));
  auto range = RecursivelyIdentifyRange(root, fs);
  EXPECT_FALSE(range.IsEmpty());
  EXPECT_FALSE(range.IsSingleValue());
  EXPECT_TRUE(range.IsLinear());
  EXPECT_EQ(range.min().GetUnsignedValue(), 32768);
  EXPECT_EQ(range.max().GetUnsignedValue(), 32773);
}

TEST_F(ValueRangeTest, SubtractValue) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  c0 = s32[] constant(124)
  p0 = s32[] parameter(0)
  ROOT %a = s32[] subtract(p0, c0)
}
)";
  auto module =
      ParseAndReturnUnverifiedModule(hlo_string, HloModuleConfig{}).value();
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* p0 = root->operand(0);
  absl::flat_hash_map<const HloInstruction*, Range> fs;
  fs.insert(std::make_pair(
      p0, Range{ConstantValue::GetZero(32, /*is_signed=*/true),
                ConstantValue::GetSigned(5, 32), /*is_linear=*/true}));
  auto range = RecursivelyIdentifyRange(root, fs);
  EXPECT_FALSE(range.IsEmpty());
  EXPECT_FALSE(range.IsSingleValue());
  EXPECT_TRUE(range.IsLinear());
  EXPECT_EQ(range.min().GetSignedValue(), -124);
  EXPECT_EQ(range.max().GetSignedValue(), -119);
}

TEST_F(ValueRangeTest, SelectValue) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  c0 = s32[] constant(124)
  p0 = s32[] parameter(0)
  c = pred[] compare(p0, c0), direction=LT
  %s = s32[] subtract(p0, c0)
  %a = s32[] add(c0, p0)
  ROOT slct = s32[] select(c, s, a)
}
)";
  auto module =
      ParseAndReturnUnverifiedModule(hlo_string, HloModuleConfig{}).value();
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* p0 = root->operand(0)->operand(0);
  absl::flat_hash_map<const HloInstruction*, Range> fs;
  fs.insert(std::make_pair(
      p0, Range{ConstantValue::GetZero(32, /*is_signed=*/true),
                ConstantValue::GetSigned(5, 32), /*is_linear=*/true}));
  auto range = RecursivelyIdentifyRange(root, fs);
  EXPECT_FALSE(range.IsEmpty());
  EXPECT_FALSE(range.IsSingleValue());
  EXPECT_TRUE(range.IsLinear());
  EXPECT_EQ(range.max().GetSignedValue(), -119);
  EXPECT_EQ(range.min().GetSignedValue(), -124);
}

TEST_F(ValueRangeTest, SelectValue2) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  c0 = s32[] constant(124)
  p0 = s32[] parameter(0)
  c = pred[] compare(c0, p0), direction=LT
  %s = s32[] subtract(p0, c0)
  %a = s32[] add(c0, p0)
  ROOT slct = s32[] select(c, s, a)
}
)";
  auto module =
      ParseAndReturnUnverifiedModule(hlo_string, HloModuleConfig{}).value();
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* p0 = root->operand(0)->operand(1);
  absl::flat_hash_map<const HloInstruction*, Range> fs;
  fs.insert(std::make_pair(
      p0, Range{ConstantValue::GetZero(32, /*is_signed=*/true),
                ConstantValue::GetSigned(5, 32), /*is_linear=*/true}));
  auto range = RecursivelyIdentifyRange(root, fs);
  EXPECT_FALSE(range.IsEmpty());
  EXPECT_FALSE(range.IsSingleValue());
  EXPECT_TRUE(range.IsLinear());
  EXPECT_EQ(range.max().GetSignedValue(), 129);
  EXPECT_EQ(range.min().GetSignedValue(), 124);
}

TEST_F(ValueRangeTest, AddSubtractValue) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  c0 = s32[] constant(124)
  c1 = s32[] constant(12)
  c2 = s32[] constant(5)
  p0 = s32[] parameter(0)
  sub = s32[] subtract(p0, c0)
  a = s32[] add(sub, c1)
  sub2 = s32[] subtract(c2, a)
}
)";
  auto module =
      ParseAndReturnUnverifiedModule(hlo_string, HloModuleConfig{}).value();
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* p0 = root->operand(1)->operand(0)->operand(0);
  absl::flat_hash_map<const HloInstruction*, Range> fs;
  fs.insert(std::make_pair(
      p0, Range{ConstantValue::GetZero(32, /*is_signed=*/true),
                ConstantValue::GetSigned(5, 32), /*is_linear=*/true}));
  auto range = RecursivelyIdentifyRange(root, fs);
  EXPECT_FALSE(range.IsEmpty());
  EXPECT_FALSE(range.IsSingleValue());
  EXPECT_TRUE(range.IsLinear());
  EXPECT_EQ(range.min().GetSignedValue(), 112);
  EXPECT_EQ(range.max().GetSignedValue(), 117);
}

TEST_F(ValueRangeTest, SubtractWrapAroundValue) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  c0 = s16[] constant(124)
  p0 = s16[] parameter(0)
  ROOT %a = s16[] subtract(p0, c0)
}
)";
  auto module =
      ParseAndReturnUnverifiedModule(hlo_string, HloModuleConfig{}).value();
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* p0 = root->operand(0);
  absl::flat_hash_map<const HloInstruction*, Range> fs;
  fs.insert(
      std::make_pair(p0, Range{ConstantValue::GetSigned(-32768, 16),
                               ConstantValue::GetZero(16, /*is_signed=*/true),
                               /*is_linear=*/true}));
  auto range = RecursivelyIdentifyRange(root, fs);
  EXPECT_TRUE(range.IsEmpty());
  EXPECT_FALSE(range.IsSingleValue());
  EXPECT_FALSE(range.IsLinear());
}

TEST_F(ValueRangeTest, AddWrapAroundValue) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  c0 = s16[] constant(124)
  p0 = s16[] parameter(0)
  ROOT %a = s16[] add(p0, c0)
}
)";
  auto module =
      ParseAndReturnUnverifiedModule(hlo_string, HloModuleConfig{}).value();
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* p0 = root->operand(0);
  absl::flat_hash_map<const HloInstruction*, Range> fs;
  fs.insert(
      std::make_pair(p0, Range{ConstantValue::GetZero(16, /*is_signed=*/true),
                               ConstantValue::GetSigned(32760, 16),
                               /*is_linear=*/true}));
  auto range = RecursivelyIdentifyRange(root, fs);
  EXPECT_TRUE(range.IsEmpty());
  EXPECT_FALSE(range.IsSingleValue());
  EXPECT_FALSE(range.IsLinear());
}

}  // namespace
}  // namespace xla
