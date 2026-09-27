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

#include "xla/hlo/transforms/propagate_call_metadata.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module_metadata.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

class PropagateCallMetadataTest : public HloHardwareIndependentTestBase {};

TEST_F(PropagateCallMetadataTest, PropagatesOpNameRecursivelyIntoWhile) {
  const char* hlo = R"(

cond {
  input = f32[128,32] parameter(0)
  ROOT c0 = pred[] constant(0), metadata={op_name="while/cond"}
}

body {
  input = f32[128,32] parameter(0)
  ROOT convert = f32[128,32] convert(input), metadata={op_name="while/body"}
}

callee {
  input = f32[128,32] parameter(0)
  ROOT while = f32[128,32] while(input), metadata={op_name="while"},
    condition=cond, body=body
}

ENTRY main {
  input = f32[128,32] parameter(0)
  ROOT result = f32[128,32] call(input), to_apply=callee, metadata={op_name="x"}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          PropagateCallMetadata().Run(module.get()));
  EXPECT_TRUE(changed);

  auto* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kCall);

  HloComputation* callee = root->called_computations()[0];
  HloInstruction* while_instr = callee->root_instruction();
  EXPECT_EQ(while_instr->metadata().op_name(), "x/while");
  EXPECT_EQ(
      while_instr->while_condition()->root_instruction()->metadata().op_name(),
      "x/while/cond");
  EXPECT_EQ(while_instr->while_body()->root_instruction()->metadata().op_name(),
            "x/while/body");
}

TEST_F(PropagateCallMetadataTest, NoMetadataInEmbeddedComputations) {
  const char* hlo = R"(

reducer {
  x = f32[] parameter(0)
  y = f32[] parameter(1)
  ROOT add = f32[] add(x, y)
}

callee {
  input = f32[128,32] parameter(0)
  const = f32[] constant(0)
  ROOT reduce = f32[128] reduce(input, const), dimensions={1}, to_apply=reducer, metadata={op_name="reduce"}
}

ENTRY main {
  input = f32[128,32] parameter(0)
  ROOT result = f32[128] call(input), to_apply=callee, metadata={op_name="x"}
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          PropagateCallMetadata().Run(module.get()));
  EXPECT_TRUE(changed);

  HloComputation* callee = module->GetComputationWithName("callee");
  HloInstruction* reduce = callee->root_instruction();
  EXPECT_EQ(reduce->metadata().op_name(), "x/reduce");
  HloComputation* reducer = module->GetComputationWithName("reducer");
  EXPECT_EQ(reducer->root_instruction()->metadata().op_name(), "");
}

TEST_F(PropagateCallMetadataTest, SkipsLongOpNames) {
  const char* hlo = R"(
callee {
  input = f32[128,32] parameter(0)
  ROOT y = f32[128,32] negate(input), metadata={op_name="y"}
}

ENTRY main {
  input = f32[128,32] parameter(0)
  ROOT result = f32[128,32] call(input), to_apply=callee
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  auto* call_instr = module->entry_computation()->root_instruction();
  OpMetadata metadata = call_instr->metadata();
  std::string long_prefix(1024, 'x');
  metadata.set_op_name(long_prefix);
  call_instr->set_metadata(metadata);

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          PropagateCallMetadata().Run(module.get()));
  // Combined would exceed kMaxOpNameSize, so op_name is left unchanged.
  EXPECT_FALSE(changed);

  HloComputation* callee = module->GetComputationWithName("callee");
  HloInstruction* neg = callee->root_instruction();
  EXPECT_EQ(neg->metadata().op_name(), "y");
}

TEST_F(PropagateCallMetadataTest, StackFrameConcatenation) {
  const char* hlo = R"(
callee {
  input = f32[4] parameter(0)
  ROOT neg = f32[4] negate(input)
}

ENTRY main {
  input = f32[4] parameter(0)
  ROOT result = f32[4] call(input), to_apply=callee
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));

  HloStackFrame frame1;
  frame1.file_name = "file1.py";
  frame1.function_name = "func1";
  frame1.line = 10;
  frame1.parent_frame_id = StackFrameId{0};
  StackFrameId id1 = module->mutable_stack_frames().AddStackFrame(frame1);

  HloStackFrame frame2;
  frame2.file_name = "file2.py";
  frame2.function_name = "func2";
  frame2.line = 20;
  frame2.parent_frame_id = StackFrameId{0};
  StackFrameId id2 = module->mutable_stack_frames().AddStackFrame(frame2);

  HloComputation* callee = module->GetComputationWithName("callee");
  HloInstruction* neg = callee->root_instruction();
  OpMetadata neg_metadata;
  neg_metadata.set_stack_frame_id(id2.value);
  neg->set_metadata(neg_metadata);

  auto* call_instr = module->entry_computation()->root_instruction();
  OpMetadata call_metadata;
  call_metadata.set_stack_frame_id(id1.value);
  call_instr->set_metadata(call_metadata);

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          PropagateCallMetadata().Run(module.get()));
  EXPECT_TRUE(changed);

  StackFrameId new_frame_id{neg->metadata().stack_frame_id()};
  EXPECT_NE(new_frame_id, id1);
  EXPECT_NE(new_frame_id, id2);
  EXPECT_TRUE(new_frame_id.valid());

  HloStackFrame new_frame = module->stack_frames().GetStackFrame(new_frame_id);
  EXPECT_EQ(new_frame.file_name, "file2.py");
  EXPECT_EQ(new_frame.function_name, "func2");
  EXPECT_EQ(new_frame.parent_frame_id, id1);
}

TEST_F(PropagateCallMetadataTest, StackFrameRedundantPrefixSkipsConcatenation) {
  const char* hlo = R"(
callee {
  input = f32[4] parameter(0)
  ROOT neg = f32[4] negate(input)
}

ENTRY main {
  input = f32[4] parameter(0)
  ROOT result = f32[4] call(input), to_apply=callee
})";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));

  HloStackFrame frame1;
  frame1.file_name = "file1.py";
  frame1.function_name = "func1";
  frame1.line = 10;
  frame1.parent_frame_id = StackFrameId{0};
  StackFrameId id1 = module->mutable_stack_frames().AddStackFrame(frame1);

  HloStackFrame frame2;
  frame2.file_name = "file2.py";
  frame2.function_name = "func2";
  frame2.line = 20;
  frame2.parent_frame_id = id1;
  StackFrameId id2 = module->mutable_stack_frames().AddStackFrame(frame2);

  auto* call_instr = module->entry_computation()->root_instruction();
  OpMetadata call_metadata;
  call_metadata.set_stack_frame_id(id1.value);
  call_instr->set_metadata(call_metadata);

  HloComputation* callee = module->GetComputationWithName("callee");
  HloInstruction* neg = callee->root_instruction();
  OpMetadata neg_metadata;
  neg_metadata.set_stack_frame_id(id2.value);
  neg->set_metadata(neg_metadata);

  TF_ASSERT_OK(PropagateCallMetadata().Run(module.get()).status());
  EXPECT_EQ(neg->metadata().stack_frame_id(), id2.value);
}

TEST_F(PropagateCallMetadataTest, NoChangeWhenCallHasNoMetadata) {
  const char* hlo = R"(
callee {
  p0 = f32[4] parameter(0)
  ROOT add = f32[4] add(p0, p0), metadata={op_name="inner"}
}

ENTRY main {
  p0 = f32[4] parameter(0)
  ROOT call = f32[4] call(p0), to_apply=callee
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          PropagateCallMetadata().Run(module.get()));
  EXPECT_FALSE(changed);

  HloComputation* callee = module->GetComputationWithName("callee");
  HloInstruction* add = callee->GetInstructionWithName("add");
  EXPECT_EQ(add->metadata().op_name(), "inner");
}

TEST_F(PropagateCallMetadataTest, DoesNotDuplicateExistingPrefix) {
  const char* hlo = R"(
callee {
  p0 = f32[4] parameter(0)
  ROOT add = f32[4] add(p0, p0), metadata={op_name="outer/inner"}
}

ENTRY main {
  p0 = f32[4] parameter(0)
  ROOT call = f32[4] call(p0), to_apply=callee, metadata={op_name="outer"}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK(PropagateCallMetadata().Run(module.get()).status());

  HloComputation* callee = module->GetComputationWithName("callee");
  HloInstruction* add = callee->GetInstructionWithName("add");
  EXPECT_EQ(add->metadata().op_name(), "outer/inner");
}

TEST_F(PropagateCallMetadataTest, PropagatesIntoNestedCalls) {
  const char* hlo = R"(
deep_callee {
  p0 = f32[4] parameter(0)
  ROOT add = f32[4] add(p0, p0), metadata={op_name="deep_add"}
}

inner_callee {
  p0 = f32[4] parameter(0)
  ROOT call = f32[4] call(p0), to_apply=deep_callee, metadata={op_name="inner"}
}

ENTRY main {
  p0 = f32[4] parameter(0)
  ROOT call = f32[4] call(p0), to_apply=inner_callee, metadata={op_name="outer"}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          PropagateCallMetadata().Run(module.get()));
  EXPECT_TRUE(changed);

  HloComputation* inner = module->GetComputationWithName("inner_callee");
  EXPECT_EQ(inner->root_instruction()->metadata().op_name(), "outer/inner");

  HloComputation* deep = module->GetComputationWithName("deep_callee");
  HloInstruction* add = deep->GetInstructionWithName("add");
  EXPECT_EQ(add->metadata().op_name(), "outer/inner/deep_add");
}

TEST_F(PropagateCallMetadataTest, IdempotentOnSecondRun) {
  const char* hlo = R"(
callee {
  p0 = f32[4] parameter(0)
  ROOT add = f32[4] add(p0, p0), metadata={op_name="inner"}
}

ENTRY main {
  p0 = f32[4] parameter(0)
  ROOT call = f32[4] call(p0), to_apply=callee, metadata={op_name="outer"}
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(bool changed1,
                          PropagateCallMetadata().Run(module.get()));
  EXPECT_TRUE(changed1);

  HloComputation* callee = module->GetComputationWithName("callee");
  HloInstruction* add = callee->GetInstructionWithName("add");
  EXPECT_EQ(add->metadata().op_name(), "outer/inner");

  TF_ASSERT_OK_AND_ASSIGN(bool changed2,
                          PropagateCallMetadata().Run(module.get()));
  EXPECT_FALSE(changed2);
  EXPECT_EQ(add->metadata().op_name(), "outer/inner");
}

}  // namespace
}  // namespace xla
