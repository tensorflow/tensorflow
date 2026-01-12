/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/core/host_offloading/host_offloading_transforms.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using testing::IsEmpty;
using testing::UnorderedElementsAre;

class RewriteToDestinationPassingStyleTest
    : public HloHardwareIndependentTestBase {
 public:
  // Helper to create ProgramShape from an HloModule's entry computation layout
  static ProgramShape GetProgramShape(const HloModule& module) {
    return module.entry_computation_layout().ComputeProgramShape();
  }

  // Helper to create HloInputOutputAliasConfig from an HloModule
  static HloInputOutputAliasConfig GetAliasConfig(const HloModule& module) {
    return module.input_output_alias_config();
  }
};

TEST_F(RewriteToDestinationPassingStyleTest, AlreadyPreparedNoChange) {
  const char* const hlo_string = R"(
HloModule test_module

ENTRY entry {
  param0 = f32[10] parameter(0)
  ROOT result = f32[10] add(param0, param0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  // Set up the initial state: flattened params, aliased output
  ProgramShape program_shape = GetProgramShape(*module);
  HloInputOutputAliasConfig alias_config = GetAliasConfig(*module);
  ASSERT_OK(alias_config.SetUpAlias(
      /*output_index=*/{}, /*param_number=*/0, /*param_index=*/{},
      HloInputOutputAliasConfig::kMustAlias));
  module->set_input_output_alias_config(alias_config);

  EXPECT_OK(RewriteToDestinationPassingStyle(module.get(), program_shape,
                                             alias_config));

  // Verify no changes
  EXPECT_EQ(module->entry_computation()->num_parameters(), 1);
  EXPECT_EQ(module->entry_computation()->parameter_instruction(0)->shape(),
            ShapeUtil::MakeShape(F32, {10}));
  const auto& final_alias_config = module->input_output_alias_config();
  auto alias = final_alias_config.GetAliasedParameter({});
  ASSERT_TRUE(alias.has_value());
  EXPECT_EQ(alias->parameter_number, 0);
  EXPECT_THAT(alias->parameter_index, IsEmpty());
  EXPECT_EQ(alias->kind, HloInputOutputAliasConfig::kMustAlias);
}

TEST_F(RewriteToDestinationPassingStyleTest, FlatteningOnly) {
  const char* const hlo_string = R"(
HloModule test_module

ENTRY entry {
  param0 = (f32[10], s32[5]) parameter(0)
  gte0 = f32[10] get-tuple-element(param0), index=0
  gte1 = s32[5] get-tuple-element(param0), index=1
  add0 = f32[10] add(gte0, gte0)
  add1 = s32[5] add(gte1, gte1)
  ROOT result = (f32[10], s32[5]) tuple(add0, add1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ProgramShape program_shape = GetProgramShape(*module);
  HloInputOutputAliasConfig alias_config = GetAliasConfig(*module);
  // Alias output leaves to corresponding input leaves
  ASSERT_OK(alias_config.SetUpAlias(
      /*output_index=*/{0}, /*param_number=*/0, /*param_index=*/{0},
      HloInputOutputAliasConfig::kMustAlias));
  ASSERT_OK(alias_config.SetUpAlias(
      /*output_index=*/{1}, /*param_number=*/0, /*param_index=*/{1},
      HloInputOutputAliasConfig::kMustAlias));
  module->set_input_output_alias_config(alias_config);

  ASSERT_OK(RewriteToDestinationPassingStyle(module.get(), program_shape,
                                             alias_config));

  // Verify parameters flattened
  HloComputation* entry = module->entry_computation();
  ASSERT_EQ(entry->num_parameters(), 2);
  HloInstruction* param0 = entry->parameter_instruction(0);
  HloInstruction* param1 = entry->parameter_instruction(1);
  EXPECT_EQ(param0->shape(), ShapeUtil::MakeShape(F32, {10}));
  EXPECT_EQ(param1->shape(), ShapeUtil::MakeShape(S32, {5}));
  EXPECT_EQ(param0->user_count(), 1);
  EXPECT_EQ(param1->user_count(), 1);
  HloInstruction* reconstructed_tuple = param0->users()[0];
  EXPECT_EQ(
      reconstructed_tuple->shape(),
      ShapeUtil::MakeTupleShape(absl::Span<const Shape>{
          ShapeUtil::MakeShape(F32, {10}), ShapeUtil::MakeShape(S32, {5})}));
  EXPECT_THAT(reconstructed_tuple->users(),
              UnorderedElementsAre(entry->GetInstructionWithName("gte0"),
                                   entry->GetInstructionWithName("gte1")));
  // Verify alias config updated for new parameters
  const auto& final_alias_config = module->input_output_alias_config();
  auto alias0 = final_alias_config.GetAliasedParameter({0});
  ASSERT_TRUE(alias0.has_value());
  EXPECT_EQ(alias0->parameter_number, 0);
  EXPECT_THAT(alias0->parameter_index, IsEmpty());
  EXPECT_EQ(alias0->kind, HloInputOutputAliasConfig::kMustAlias);
  auto alias1 = final_alias_config.GetAliasedParameter({1});
  ASSERT_TRUE(alias1.has_value());
  EXPECT_EQ(alias1->parameter_number, 1);
  EXPECT_THAT(alias1->parameter_index, IsEmpty());
  EXPECT_EQ(alias1->kind, HloInputOutputAliasConfig::kMustAlias);
}

TEST_F(RewriteToDestinationPassingStyleTest, AliasingOnly) {
  const char* const hlo_string = R"(
HloModule test_module

ENTRY entry {
  param0 = f32[10] parameter(0)
  param1 = s32[5] parameter(1)
  add0 = f32[10] add(param0, param0)
  add1 = s32[5] add(param1, param1)
  ROOT result = (f32[10], s32[5]) tuple(add0, add1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ProgramShape program_shape = GetProgramShape(*module);
  HloInputOutputAliasConfig alias_config = GetAliasConfig(*module);
  // Alias only the first output leaf
  ASSERT_OK(alias_config.SetUpAlias(
      /*output_index=*/{0}, /*param_number=*/0, /*param_index=*/{},
      HloInputOutputAliasConfig::kMustAlias));
  module->set_input_output_alias_config(alias_config);

  ASSERT_OK(RewriteToDestinationPassingStyle(module.get(), program_shape,
                                             alias_config));

  // Verify destination parameter added
  HloComputation* entry = module->entry_computation();
  ASSERT_EQ(entry->num_parameters(), 3);
  EXPECT_EQ(entry->parameter_instruction(0)->shape(),
            ShapeUtil::MakeShape(F32, {10}));
  EXPECT_EQ(entry->parameter_instruction(1)->shape(),
            ShapeUtil::MakeShape(S32, {5}));
  EXPECT_EQ(entry->parameter_instruction(2)->shape(),
            ShapeUtil::MakeShape(S32, {5}));
  EXPECT_EQ(entry->parameter_instruction(2)->name(), "output_param");
  // Verify Alias config updated
  const auto& final_alias_config = module->input_output_alias_config();
  // Original alias should persist
  auto alias0 = final_alias_config.GetAliasedParameter({0});
  ASSERT_TRUE(alias0.has_value());
  EXPECT_EQ(alias0->parameter_number, 0);
  EXPECT_THAT(alias0->parameter_index, IsEmpty());
  EXPECT_EQ(alias0->kind, HloInputOutputAliasConfig::kMustAlias);
  // New alias for the previously unaliased output
  auto alias1 = final_alias_config.GetAliasedParameter({1});
  ASSERT_TRUE(alias1.has_value());
  EXPECT_EQ(alias1->parameter_number,
            2);  // Aliased to the new destination param
  EXPECT_THAT(alias1->parameter_index, IsEmpty());
  EXPECT_EQ(alias1->kind, HloInputOutputAliasConfig::kMustAlias);
}

TEST_F(RewriteToDestinationPassingStyleTest, MixedParamsFlatten) {
  const char* const hlo_string = R"(
HloModule test_module

ENTRY entry {
  param0 = f32[10] parameter(0)
  param1 = (s32[5], u32[2]) parameter(1)
  gte1_0 = s32[5] get-tuple-element(param1), index=0
  gte1_1 = u32[2] get-tuple-element(param1), index=1
  add0 = f32[10] add(param0, param0)
  add1_0 = s32[5] add(gte1_0, gte1_0)
  add1_1 = u32[2] add(gte1_1, gte1_1)
  ROOT result = (f32[10], s32[5], u32[2]) tuple(add0, add1_0, add1_1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ProgramShape program_shape = GetProgramShape(*module);
  HloInputOutputAliasConfig alias_config = GetAliasConfig(*module);
  // Alias all outputs to corresponding inputs/input elements
  ASSERT_OK(alias_config.SetUpAlias(
      /*output_index=*/{0}, /*param_number=*/0, /*param_index=*/{},
      HloInputOutputAliasConfig::kMustAlias));
  ASSERT_OK(alias_config.SetUpAlias(
      /*output_index=*/{1}, /*param_number=*/1, /*param_index=*/{0},
      HloInputOutputAliasConfig::kMustAlias));
  ASSERT_OK(alias_config.SetUpAlias(
      /*output_index=*/{2}, /*param_number=*/1, /*param_index=*/{1},
      HloInputOutputAliasConfig::kMustAlias));
  module->set_input_output_alias_config(alias_config);

  ASSERT_OK(RewriteToDestinationPassingStyle(module.get(), program_shape,
                                             alias_config));

  // Verify parameters flattened/reordered
  HloComputation* entry = module->entry_computation();
  ASSERT_EQ(entry->num_parameters(), 3);
  EXPECT_EQ(entry->parameter_instruction(0)->shape(),
            ShapeUtil::MakeShape(F32, {10}));
  EXPECT_EQ(entry->parameter_instruction(1)->shape(),
            ShapeUtil::MakeShape(S32, {5}));
  EXPECT_EQ(entry->parameter_instruction(2)->shape(),
            ShapeUtil::MakeShape(U32, {2}));

  // Verify alias config updated for new parameter numbers/indices
  const auto& final_alias_config = module->input_output_alias_config();
  auto alias0 = final_alias_config.GetAliasedParameter({0});
  ASSERT_TRUE(alias0.has_value());
  EXPECT_EQ(alias0->parameter_number, 0);
  EXPECT_THAT(alias0->parameter_index, IsEmpty());
  EXPECT_EQ(alias0->kind, HloInputOutputAliasConfig::kMustAlias);
  auto alias1 = final_alias_config.GetAliasedParameter({1});
  ASSERT_TRUE(alias1.has_value());
  EXPECT_EQ(alias1->parameter_number, 1);
  EXPECT_THAT(alias1->parameter_index, IsEmpty());
  EXPECT_EQ(alias1->kind, HloInputOutputAliasConfig::kMustAlias);
  auto alias2 = final_alias_config.GetAliasedParameter({2});
  ASSERT_TRUE(alias2.has_value());
  EXPECT_EQ(alias2->parameter_number, 2);
  EXPECT_THAT(alias2->parameter_index, IsEmpty());
  EXPECT_EQ(alias2->kind, HloInputOutputAliasConfig::kMustAlias);
}

}  // namespace
}  // namespace xla
