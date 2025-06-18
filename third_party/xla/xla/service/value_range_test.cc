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

#include <optional>
#include <utility>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/constant_value.h"
#include "xla/service/hlo_module_config.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

class ValueRangeTest : public HloHardwareIndependentTestBase {};

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
  EXPECT_EQ(range.max()->GetSignedValue(), 124 + 5);
  EXPECT_EQ(range.step()->GetSignedValue(), 1);
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
  // p0 has range min = 0, max = 32, step = 2.
  fs.insert(std::make_pair(
      p0, Range{/*min=*/ConstantValue::GetSigned(0, /*bitwidth=*/32),
                /*max=*/ConstantValue::GetSigned(32, /*bitwidth=*/32),
                /*step=*/ConstantValue::GetUnsigned(2, /*bitwidth=*/32),
                /*is_linear=*/true}));
  auto range = RecursivelyIdentifyRange(root, fs);
  EXPECT_FALSE(range.IsEmpty());
  EXPECT_FALSE(range.IsSingleValue());
  EXPECT_TRUE(range.IsLinear());
  EXPECT_EQ(range.min().GetSignedValue(), 0);
  EXPECT_EQ(range.max()->GetSignedValue(), 32 * 1024);
  EXPECT_EQ(range.step()->GetSignedValue(), 2 * 1024);
}

TEST_F(ValueRangeTest, MultiplyValuePassedToLoop) {
  constexpr absl::string_view hlo_string = R"(
  HloModule module
  body.comp {
    p0 = (s32[], s32[]) parameter(0)
    gte = s32[] get-tuple-element(p0), index=0
    ROOT tuple = (s32[], s32[]) tuple(gte, gte)
  }
  cond.comp {
    p0 = (s32[], s32[]) parameter(0)
    ROOT out = pred[] constant(true)
  }
  ENTRY entry {
    c0 = s32[] constant(1024)
    p0 = s32[] parameter(0)
    %mul = s32[] multiply(p0, c0)
    tuple = (s32[], s32[]) tuple(%mul, %mul)
    ROOT out = (s32[], s32[]) while(tuple), condition=cond.comp,
    body=body.comp
  }
  )";
  auto module =
      ParseAndReturnUnverifiedModule(hlo_string, HloModuleConfig{}).value();
  TF_ASSERT_OK_AND_ASSIGN(auto alias_analysis,
                          HloAliasAnalysis::Run(module.get()));
  const HloInstruction* p0 =
      module->entry_computation()->parameter_instruction(0);
  absl::flat_hash_map<const HloInstruction*, Range> fs;
  // p0 has range min = 0, max = 32, step = 2.
  fs.insert(std::make_pair(
      p0, Range{/*min=*/ConstantValue::GetSigned(0, /*bitwidth=*/32),
                /*max=*/ConstantValue::GetSigned(32, /*bitwidth=*/32),
                /*step=*/ConstantValue::GetUnsigned(2, /*bitwidth=*/32),
                /*is_linear=*/true}));
  HloComputation* body = module->GetComputationWithName("body.comp");
  HloInstruction* gte = body->GetInstructionWithName("gte");
  auto range = RecursivelyIdentifyRange(gte, fs, alias_analysis.get());
  EXPECT_FALSE(range.IsEmpty());
  EXPECT_FALSE(range.IsSingleValue());
  EXPECT_TRUE(range.IsLinear());
  EXPECT_EQ(range.min().GetSignedValue(), 0);
  EXPECT_EQ(range.max()->GetSignedValue(), 32 * 1024);
  EXPECT_EQ(range.step()->GetSignedValue(), 2 * 1024);
}

TEST_F(ValueRangeTest, ConstantValuePred) {
  constexpr absl::string_view hlo_string = R"(
  HloModule module
  ENTRY entry {
    false_pred = pred[] constant(false)
    true_pred = pred[] constant(true)
    ROOT out = tuple(false_pred, true_pred)
  }
  )";
  auto module =
      ParseAndReturnUnverifiedModule(hlo_string, HloModuleConfig{}).value();
  const HloInstruction* tuple = module->entry_computation()->root_instruction();
  absl::flat_hash_map<const HloInstruction*, Range> known_ranges;
  auto false_range = RecursivelyIdentifyRange(tuple->operand(0), known_ranges);
  VLOG(3) << "false_range: " << false_range.ToString();
  EXPECT_FALSE(false_range.IsEmpty());
  EXPECT_TRUE(false_range.IsSingleValue());
  EXPECT_TRUE(false_range.IsLinear());
  EXPECT_EQ(false_range.min().GetUnsignedValue(), 0);

  auto true_range = RecursivelyIdentifyRange(tuple->operand(1), known_ranges);
  VLOG(3) << "true_range: " << true_range.ToString();
  EXPECT_FALSE(true_range.IsEmpty());
  EXPECT_TRUE(true_range.IsSingleValue());
  EXPECT_TRUE(true_range.IsLinear());
  EXPECT_EQ(true_range.min().GetUnsignedValue(), 1);
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
    ROOT conditional.1 = (s32[], s32[]) conditional(branch_index, p0, p0),
    branch_computations={region1, region2}
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
  // p0 has range min = 0, max = 32, step = 2.
  fs.insert(std::make_pair(
      p0, Range{/*min=*/ConstantValue::GetSigned(0, /*bitwidth=*/32),
                /*max=*/ConstantValue::GetSigned(32, /*bitwidth=*/32),
                /*step=*/ConstantValue::GetUnsigned(2, /*bitwidth=*/32),
                /*is_linear=*/true}));

  auto add_range = RecursivelyIdentifyRange(add, fs, alias_analysis.get());
  EXPECT_FALSE(add_range.IsEmpty());
  EXPECT_FALSE(add_range.IsSingleValue());
  EXPECT_TRUE(add_range.IsLinear());
  EXPECT_EQ(add_range.min().GetSignedValue(), 1024);
  EXPECT_EQ(add_range.max()->GetSignedValue(), 1024 + 32);
  EXPECT_EQ(add_range.step()->GetSignedValue(), 2);

  auto mult_range = RecursivelyIdentifyRange(mult, fs, alias_analysis.get());
  EXPECT_FALSE(mult_range.IsEmpty());
  EXPECT_FALSE(mult_range.IsSingleValue());
  EXPECT_TRUE(mult_range.IsLinear());
  EXPECT_EQ(mult_range.min().GetSignedValue(), 0);
  EXPECT_EQ(mult_range.max()->GetSignedValue(), 32 * 1024);
  EXPECT_EQ(mult_range.step()->GetSignedValue(), 2 * 1024);
}

TEST_F(ValueRangeTest, SelectValueWithCompareInConditional) {
  constexpr absl::string_view hlo_string = R"(
  HloModule module
  region1 {
    region1_param = s32[] parameter(0)
    region1_c0 = s32[] constant(1024)
    %add = s32[] add(region1_param, region1_c0)

    compare_const = s32[] constant(1030)
    compare1 = pred[] compare(%add, compare_const), direction=LT
    select1 = s32[] select(compare1, region1_param, %add)

    ROOT out = (s32[], s32[]) tuple(%add, %add)
  }
  region2 {
    region2_param = s32[] parameter(0)
    region2_c0 = s32[] constant(1024)
    %mult = s32[] multiply(region2_param, region2_c0)

    compare_const = s32[] constant(5121)
    compare2 = pred[] compare(%mult, compare_const), direction=LT
    select2 = s32[] select(compare2, region2_param, %mult)

    ROOT out = (s32[], s32[]) tuple(%mult, %mult)
  }
  ENTRY entry {
    p0 = s32[] parameter(0)
    branch_index = s32[] parameter(1)
    ROOT conditional.1 = (s32[], s32[]) conditional(branch_index, p0, p0),
    branch_computations={region1, region2}
  }
  )";
  auto module =
      ParseAndReturnUnverifiedModule(hlo_string, HloModuleConfig{}).value();
  TF_ASSERT_OK_AND_ASSIGN(auto alias_analysis,
                          HloAliasAnalysis::Run(module.get()));
  HloComputation* region1 = module->GetComputationWithName("region1");
  HloComputation* region2 = module->GetComputationWithName("region2");
  HloInstruction* select1 = region1->GetInstructionWithName("select1");
  HloInstruction* select2 = region2->GetInstructionWithName("select2");
  const HloInstruction* p0 =
      module->entry_computation()->parameter_instruction(0);
  absl::flat_hash_map<const HloInstruction*, Range> fs;
  // p0 has range min = 0, max = 32, step = 2.
  fs.insert(std::make_pair(
      p0, Range{/*min=*/ConstantValue::GetSigned(0, /*bitwidth=*/32),
                /*max=*/ConstantValue::GetSigned(32, /*bitwidth=*/32),
                /*step=*/ConstantValue::GetUnsigned(2, /*bitwidth=*/32),
                /*is_linear=*/true}));

  auto select1_range =
      RecursivelyIdentifyRange(select1, fs, alias_analysis.get());
  auto select2_range =
      RecursivelyIdentifyRange(select2, fs, alias_analysis.get());
  // We expect the select ranges to be the same as the parameter range since
  // both selects return true values.
  EXPECT_EQ(select1_range, select2_range);
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
  EXPECT_EQ(range.max()->GetUnsignedValue(), 32773);
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
  EXPECT_EQ(range.max()->GetSignedValue(), -119);
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
  EXPECT_EQ(range.max()->GetSignedValue(), -119);
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
  EXPECT_EQ(range.max()->GetSignedValue(), 129);
  EXPECT_EQ(range.min().GetSignedValue(), 124);
}

TEST_F(ValueRangeTest, SelectBoundedFromUnboundedRange) {
  constexpr absl::string_view hlo_string = R"(
HloModule module

ENTRY entry {
  p0 = s32[] parameter(0)
  p1 = s32[] parameter(1)
  ROOT %s = s32[] subtract(p0, p1)
}
)";
  auto module =
      ParseAndReturnUnverifiedModule(hlo_string, HloModuleConfig{}).value();
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* p0 =
      module->entry_computation()->parameter_instruction(0);
  const HloInstruction* p1 =
      module->entry_computation()->parameter_instruction(1);
  absl::flat_hash_map<const HloInstruction*, Range> fs;
  // p0 has range min = 1, max = Unknown, step = 2
  fs.insert(std::make_pair(
      p0, Range{/*min=*/ConstantValue::GetSigned(1, 32),
                /*max=*/std::nullopt,
                /*step=*/ConstantValue::GetUnsigned(2, /*bitwidth=*/32),
                /*is_linear=*/true}));
  // p1 has range min = 0, max = 10, step = 2
  fs.insert(std::make_pair(
      p1, Range{/*min=*/ConstantValue::GetZero(32, /*is_signed=*/true),
                /*max=*/ConstantValue::GetSigned(10, 32),
                /*step=*/ConstantValue::GetUnsigned(2, /*bitwidth=*/32),
                /*is_linear=*/true}));
  auto range = RecursivelyIdentifyRange(root, fs);
  EXPECT_FALSE(range.IsSingleValue());
  EXPECT_TRUE(range.IsLinear());
  EXPECT_FALSE(range.IsBounded());
  EXPECT_EQ(range.min().GetSignedValue(), 1 - 10);
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
  EXPECT_EQ(range.max()->GetSignedValue(), 117);
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
  fs.insert(std::make_pair(p0, Range{ConstantValue::GetSigned(-32768, 16),
                                     ConstantValue::GetZero(16,
                                                            /*is_signed=*/true),
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
  fs.insert(std::make_pair(p0, Range{ConstantValue::GetZero(16,
                                                            /*is_signed=*/true),
                                     ConstantValue::GetSigned(32760, 16),
                                     /*is_linear=*/true}));
  auto range = RecursivelyIdentifyRange(root, fs);
  EXPECT_TRUE(range.IsEmpty());
  EXPECT_FALSE(range.IsSingleValue());
  EXPECT_FALSE(range.IsLinear());
}

}  // namespace
}  // namespace xla
