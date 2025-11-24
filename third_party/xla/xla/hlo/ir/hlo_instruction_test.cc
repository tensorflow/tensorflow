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

#include "xla/hlo/ir/hlo_instruction.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/printer.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/side_effect_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::testing::ElementsAre;

using HloInstructionTest = HloHardwareIndependentTestBase;

TEST_F(HloInstructionTest, SetFrontendAttribute) {
  HloConstantInstruction instr(ShapeUtil::MakeShape(U32, {3, 2}));
  instr.set_frontend_attribute("key1", "value1");
  EXPECT_EQ(instr.get_frontend_attribute("key1").value(), "value1");
  instr.set_frontend_attribute("key1", "value2");
  EXPECT_EQ(instr.get_frontend_attribute("key1").value(), "value2");
}

TEST_F(HloInstructionTest, AddFrontendAttribute) {
  HloConstantInstruction instr(ShapeUtil::MakeShape(U32, {3, 2}));
  EXPECT_TRUE(instr.add_frontend_attribute("key1", "value1"));
  EXPECT_EQ(instr.get_frontend_attribute("key1").value(), "value1");
  EXPECT_FALSE(instr.add_frontend_attribute("key1", "value2"));
  EXPECT_EQ(instr.get_frontend_attribute("key1").value(), "value1");
}

TEST_F(HloInstructionTest, SetFrontendAttributes) {
  HloConstantInstruction instr(ShapeUtil::MakeShape(U32, {3, 2}));
  instr.add_frontend_attribute("key1", "value1");
  FrontendAttributes attributes;
  attributes.mutable_map()->insert({"key1", "value2"});
  attributes.mutable_map()->insert({"key2", "value2"});
  instr.set_frontend_attributes(attributes);
  EXPECT_EQ(instr.get_frontend_attribute("key1").value(), "value2")
      << "key1 should be overwritten";
  EXPECT_EQ(instr.get_frontend_attribute("key2").value(), "value2");
}

TEST_F(HloInstructionTest, AddFrontendAttributes) {
  HloConstantInstruction instr(ShapeUtil::MakeShape(U32, {3, 2}));
  instr.add_frontend_attribute("key1", "value1");
  FrontendAttributes attributes;
  attributes.mutable_map()->insert({"key1", "value2"});
  attributes.mutable_map()->insert({"key2", "value2"});
  instr.add_frontend_attributes(attributes);
  EXPECT_EQ(instr.get_frontend_attribute("key1").value(), "value1")
      << "key1 should not be overwritten";
  EXPECT_EQ(instr.get_frontend_attribute("key2").value(), "value2");
}

TEST_F(HloInstructionTest, CustomCallInstructionStorage) {
  HloCustomCallInstruction instr(ShapeUtil::MakeShape(U32, {3, 2}),
                                 /*operands=*/{}, "custom_call_target",
                                 /*opaque=*/"",
                                 CustomCallApiVersion::API_VERSION_ORIGINAL);
  EXPECT_EQ(instr.GetPerInstructionStorage(), nullptr);
  auto* storage1 = new HloCustomCallInstruction::PerInstructionStorage();
  auto* storage2 = new HloCustomCallInstruction::PerInstructionStorage();

  instr.SetPerInstructionStorage(
      std::unique_ptr<HloCustomCallInstruction::PerInstructionStorage>(
          storage1));
  instr.SetPerInstructionStorage(
      std::unique_ptr<HloCustomCallInstruction::PerInstructionStorage>(
          storage2));

  EXPECT_EQ(instr.GetPerInstructionStorage(), storage1);
}

TEST_F(HloInstructionTest, DeriveComputeTypeAttribute) {
  HloConstantInstruction instr0(ShapeUtil::MakeShape(U32, {3, 2}));
  instr0.add_frontend_attribute(kXlaComputeTypeAttr, kXlaComputeTypeHost);
  HloConstantInstruction instr1(ShapeUtil::MakeShape(U32, {3, 2}));
  instr0.SetupDerivedInstruction(&instr1);
  EXPECT_FALSE(instr1.has_frontend_attributes());
}

TEST_F(HloInstructionTest, CloneImplScheduledAsyncOp) {
  constexpr absl::string_view kHlo = R"(
HloModule main, is_scheduled=true

ENTRY main {
  arg.0 = s32[] parameter(0)
  call-start.0 = ((s32[]), s32[], s32[]) call-start(arg.0), to_apply={
    arg.0 = s32[] parameter(0)
    ROOT abs.0 = abs(arg.0)
  }, async_execution_thread="thread"
  ROOT call-done.0 = s32[] call-done(call-start.0)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  ASSERT_TRUE(module->has_schedule());
  TF_ASSERT_OK(module->schedule().Verify());

  HloInstruction* async_done = module->entry_computation()->root_instruction();
  ASSERT_EQ(async_done->opcode(), HloOpcode::kAsyncDone);
  HloInstruction* async_start = async_done->async_chain_start();
  HloInstruction* clone = module->entry_computation()->AddInstruction(
      async_start->CloneWithNewOperands(async_start->shape(),
                                        {async_start->mutable_operand(0)}));
  TF_ASSERT_OK(async_start->ReplaceAllUsesWith(clone));

  // Cleanup the main thread.
  TF_ASSERT_OK(HloDCE()
                   .Run(module.get(), {HloInstruction::kMainExecutionThread})
                   .status());
  TF_ASSERT_OK(
      module->schedule().Update({HloInstruction::kMainExecutionThread}));

  // The schedule for the entire module should still be valid.
  TF_EXPECT_OK(module->schedule().Verify());
}

TEST_F(HloInstructionTest, CloneImplCollectivePermuteOp) {
  constexpr absl::string_view kHlo = R"(
HloModule main

ENTRY main {
  arg.0 = f32[32,32]{1,0} parameter(0)
  ROOT collective-permute.0 = (f32[32,32]{1,0}, f32[32,32]{1,0}) collective-permute(arg.0, arg.0), channel_id=388, source_target_pairs={{0,0},{4,1}}
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));

  HloInstruction* cp = module->entry_computation()->root_instruction();
  ASSERT_EQ(cp->opcode(), HloOpcode::kCollectivePermute);
  auto clone = cp->CloneWithNewOperands(cp->shape(), cp->operands());
  EXPECT_EQ(clone->operand_count(), 2);
}

TEST_F(HloInstructionTest, PrintCompareOpWorksIfDead) {
  const char* const kModuleStr = R"(
    HloModule m
    ENTRY main {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT result = pred[] compare(p0, p1), direction=GT, type=TOTALORDER
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(
      root->ToString(),
      "%result = pred[] compare(%p0, %p1), direction=GT, type=TOTALORDER");
  module->entry_computation()->set_root_instruction(
      root->mutable_operand(0), /*accept_different_shape=*/true);
  root->DetachFromOperandsAndUsers();
  EXPECT_EQ(
      root->ToString(),
      "%result = pred[] compare(null , null ), direction=GT, type=TOTALORDER");
  TF_ASSERT_OK(module->entry_computation()->RemoveInstruction(root));
  EXPECT_EQ(root->ToString(),
            "%result = pred[] compare(), direction=GT, type=TOTALORDER");
  *module->mutable_entry_computation_layout() =
      module->compute_computation_layout();
}

TEST_F(HloInstructionTest, CanonicalPrintingSupportsInt64) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           R"(
    HloModule m
    ENTRY main {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT result = pred[] compare(p0, p1), direction=GT, type=TOTALORDER
    }
  )"));

  xla::HloPrintOptions hlo_print_options =
      xla::HloPrintOptions(xla::HloPrintOptions::Canonical());
  hlo_print_options.set_is_in_nested_computation(true);

  xla::CanonicalNameMap new_map;
  xla::StringPrinter printer;
  // Param 0
  module->entry_computation()
      ->parameter_instruction(0)
      ->PrintWithCanonicalNameMap(&printer, hlo_print_options, &new_map);
  std::string param1_to_string = std::move(printer).ToString();

  printer = StringPrinter();
  // Param 1
  module->entry_computation()
      ->parameter_instruction(1)
      ->PrintWithCanonicalNameMap(&printer, hlo_print_options, &new_map);
  std::string param2_to_string = std::move(printer).ToString();

  printer = StringPrinter();
  // Result Root Instruction
  module->entry_computation()->root_instruction()->PrintWithCanonicalNameMap(
      &printer, hlo_print_options, &new_map);
  std::string param3_to_string = std::move(printer).ToString();

  EXPECT_EQ(param1_to_string, "tmp_0 = f32[] parameter(0)");
  EXPECT_EQ(param2_to_string, "tmp_1 = f32[] parameter(1)");
  EXPECT_EQ(param3_to_string,
            "tmp_2 = pred[] compare(f32[] tmp_0, f32[] tmp_1), direction=GT, "
            "type=TOTALORDER");
}

}  // namespace
}  // namespace xla
