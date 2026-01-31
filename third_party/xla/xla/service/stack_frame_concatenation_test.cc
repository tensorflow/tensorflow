/* Copyright 2026 The OpenXLA Authors.

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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal_util.h"
#include "xla/service/call_inliner.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

using ::testing::AnyOf;
using ::testing::Not;

using StackFrameConcatenationTest = HloHardwareIndependentTestBase;

struct ModuleWithCall {
  std::unique_ptr<HloModule> module;
  HloInstruction* call;
  HloInstruction* neg;
};

ModuleWithCall CreateModuleWithCall(const std::string& module_name) {
  auto module = std::make_unique<HloModule>(module_name, HloModuleConfig());
  HloComputation::Builder inner_builder("inner");
  auto* p0 = inner_builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {}), "p0"));
  auto* neg = inner_builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(F32, {}), HloOpcode::kNegate, p0));
  HloComputation* inner = module->AddEmbeddedComputation(inner_builder.Build());

  HloComputation::Builder outer_builder("outer");
  auto* constant = outer_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0f)));
  auto* call = outer_builder.AddInstruction(HloInstruction::CreateCall(
      ShapeUtil::MakeShape(F32, {}), {constant}, inner));
  module->AddEntryComputation(outer_builder.Build());

  return {std::move(module), call, neg};
}

void SetStackFrames(StackFrameIndexProto* index, int frame2_parent_id) {
  index->add_file_names("file1.py");
  index->add_file_names("file2.py");
  index->add_function_names("func1");
  index->add_function_names("func2");

  auto* loc1 = index->add_file_locations();
  loc1->set_file_name_id(1);
  loc1->set_function_name_id(1);
  loc1->set_line(10);

  auto* loc2 = index->add_file_locations();
  loc2->set_file_name_id(2);
  loc2->set_function_name_id(2);
  loc2->set_line(20);

  // Frame 1: func1 (caller).
  auto* frame1 = index->add_stack_frames();
  frame1->set_file_location_id(1);
  frame1->set_parent_frame_id(0);

  // Frame 2: func2 (callee).
  auto* frame2 = index->add_stack_frames();
  frame2->set_file_location_id(2);
  frame2->set_parent_frame_id(frame2_parent_id);
}

TEST_F(StackFrameConcatenationTest, InlinedStackFrameConcatenation) {
  auto [module, call_orig, neg] = CreateModuleWithCall(TestName());

  OpMetadata neg_metadata;
  neg_metadata.set_stack_frame_id(2);
  neg->set_metadata(neg_metadata);

  OpMetadata call_metadata;
  call_metadata.set_stack_frame_id(1);
  call_orig->set_metadata(call_metadata);

  auto module_proto = module->ToProto();
  SetStackFrames(module_proto.mutable_stack_frame_index(), 0);

  TF_ASSERT_OK_AND_ASSIGN(
      auto reconstructed_module,
      HloModule::CreateFromProto(module_proto, module->config()));
  module = std::move(reconstructed_module);

  HloInstruction* call = nullptr;
  for (auto* inst : module->entry_computation()->instructions()) {
    if (inst->opcode() == HloOpcode::kCall) {
      call = inst;
      break;
    }
  }
  ASSERT_NE(call, nullptr);

  TF_ASSERT_OK(CallInliner::Inline(call).status());

  HloInstruction* inlined_neg = nullptr;
  for (auto* inst : module->entry_computation()->instructions()) {
    if (inst->opcode() == HloOpcode::kNegate) {
      inlined_neg = inst;
      break;
    }
  }
  ASSERT_NE(inlined_neg, nullptr);

  int32_t new_frame_id = inlined_neg->metadata().stack_frame_id();
  EXPECT_THAT(new_frame_id, Not(AnyOf(0, 1, 2)));

  const auto& frames = module->stack_frame_index()->stack_frames();
  ASSERT_GE(frames.size(), 3);
  const auto& new_frame = frames[new_frame_id - 1];
  EXPECT_EQ(new_frame.file_location_id(), 2);
  EXPECT_EQ(new_frame.parent_frame_id(), 1);
}

TEST_F(StackFrameConcatenationTest,
       InlinedStackFrameRedundantPrefixSkipsConcatenation) {
  auto [module, call_orig, neg] = CreateModuleWithCall(TestName());

  OpMetadata call_metadata;
  call_metadata.set_stack_frame_id(1);
  call_orig->set_metadata(call_metadata);

  OpMetadata neg_metadata;
  neg_metadata.set_stack_frame_id(2);
  neg->set_metadata(neg_metadata);

  auto module_proto = module->ToProto();
  SetStackFrames(module_proto.mutable_stack_frame_index(), 1);

  TF_ASSERT_OK_AND_ASSIGN(
      auto reconstructed_module,
      HloModule::CreateFromProto(module_proto, module->config()));
  module = std::move(reconstructed_module);

  HloInstruction* call = nullptr;
  for (auto* inst : module->entry_computation()->instructions()) {
    if (inst->opcode() == HloOpcode::kCall) {
      call = inst;
      break;
    }
  }
  ASSERT_NE(call, nullptr);

  TF_ASSERT_OK(CallInliner::Inline(call).status());

  HloInstruction* inlined_neg = nullptr;
  for (auto* inst : module->entry_computation()->instructions()) {
    if (inst->opcode() == HloOpcode::kNegate) {
      inlined_neg = inst;
      break;
    }
  }
  ASSERT_NE(inlined_neg, nullptr);
  EXPECT_EQ(inlined_neg->metadata().stack_frame_id(), 2);
}

}  // namespace
}  // namespace xla
