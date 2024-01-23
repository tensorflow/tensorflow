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
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_parser.h"
#include "xla/tests/hlo_test_base.h"

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
  fs.insert(std::make_pair(
      p0, Range{ConstantValue::GetZero(32, /*is_signed=*/true),
                ConstantValue::GetSigned(5, 32), /*is_linear=*/true}));
  auto range = RecursivelyIdentifyRange(root, fs);
  EXPECT_FALSE(range.IsEmpty());
  EXPECT_FALSE(range.IsSingleValue());
  EXPECT_TRUE(range.IsLinear());
  EXPECT_EQ(range.min().GetSignedValue(), 124);
  EXPECT_EQ(range.max().GetSignedValue(), 129);
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
  s = s32[] subtract(p0, c0)
  a = s32[] add(s, c1)
  s2 = s32[] subtract(c2, a)
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
