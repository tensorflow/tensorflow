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
#include "xla/hlo/transforms/strip_memory_placement_annotations.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

using StripMemoryPlacementAnnotationsTest = HloHardwareIndependentTestBase;

TEST_F(StripMemoryPlacementAnnotationsTest, StripAnnotateDevicePlacement) {
  const std::string hlo_string = R"(
HloModule main

ENTRY main {
  p0 = f32[4]{0} parameter(0)
  annotate = f32[4]{0} custom-call(p0), custom_call_target="annotate_device_placement"
  ROOT add = f32[4]{0} add(annotate, annotate)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  StripMemoryPlacementAnnotations pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  // Verify custom-call is gone and add uses parameter p0 directly.
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAdd);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kParameter);
}

TEST_F(StripMemoryPlacementAnnotationsTest, StripMoveToHostAndDevice) {
  const std::string hlo_string = R"(
HloModule main

ENTRY main {
  p0 = f32[4]{0} parameter(0)
  to_host = f32[4]{0} custom-call(p0), custom_call_target="MoveToHost"
  to_device = f32[4]{0} custom-call(to_host), custom_call_target="MoveToDevice"
  ROOT add = f32[4]{0} add(to_device, to_device)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  StripMemoryPlacementAnnotations pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAdd);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kParameter);
}

TEST_F(StripMemoryPlacementAnnotationsTest, NoStripUnrelatedCustomCall) {
  const std::string hlo_string = R"(
HloModule main

ENTRY main {
  p0 = f32[4]{0} parameter(0)
  unrelated = f32[4]{0} custom-call(p0), custom_call_target="SomeUnrelatedTarget"
  ROOT add = f32[4]{0} add(unrelated, unrelated)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  StripMemoryPlacementAnnotations pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(StripMemoryPlacementAnnotationsTest, StripAnnotationAtRoot) {
  const std::string hlo_string = R"(
HloModule main

ENTRY main {
  p0 = f32[4]{0} parameter(0)
  ROOT annotate = f32[4]{0} custom-call(p0), custom_call_target="annotate_device_placement"
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  StripMemoryPlacementAnnotations pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kParameter);
}

}  // namespace
}  // namespace xla
