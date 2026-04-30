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

#include "xla/service/cpu/cpu_multi_output_fusion.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/tsl/platform/statusor.h"

namespace op = xla::testing::opcode_matchers;

namespace xla::cpu {
namespace {

class MultiOutputFusionTest : public HloHardwareIndependentTestBase {
 protected:
  AliasInfo alias_info_;
};

TEST_F(MultiOutputFusionTest, TrivialReusedInput) {
  // The current implementation of the multi-output fusion pass only fuses when
  // one of the instructions is a fusion.
  // TODO(willfroom): Fix this to enable fusing of two un-fused instructions.
  static constexpr absl::string_view kTrivialReusedInput = R"(
    HloModule module

    %add_fn {
      %arg0 = f32[100] parameter(0)
      %arg1 = f32[100] parameter(1)
      ROOT %add = f32[100] add(%arg0, %arg1)
    }


    ENTRY %main (arg0: f32[100], arg1: f32[100]) -> (f32[100], f32[100]) {
      %arg0 = f32[100] parameter(0)
      %arg1 = f32[100] parameter(1)
      %double = f32[100] fusion(%arg0, %arg1), kind=kLoop, calls=%add_fn
      %square = f32[100] multiply(%arg0, %arg1)
      ROOT %result = (f32[100], f32[100]) tuple(%double, %square)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kTrivialReusedInput));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, CpuMultiOutputFusion(&alias_info_).Run(hlo_module.get()));
  EXPECT_TRUE(changed);
  HloComputation* entry_computation = hlo_module->entry_computation();
  EXPECT_THAT(entry_computation->instructions(),
              ::testing::Contains(op::Fusion()).Times(1));
  auto fusion_match = op::Fusion(op::Parameter(0), op::Parameter(1));
  EXPECT_THAT(entry_computation->root_instruction(),
              op::Tuple(op::GetTupleElement(fusion_match),
                        op::GetTupleElement(fusion_match)));
}

// Regression test for the scan-body fusion gate in
// InstructionFusion::ShouldFuseIntoMultiOutput.
TEST_F(MultiOutputFusionTest, DoesNotFuseInsideScanBody) {
  static constexpr absl::string_view kHloModule = R"(
    HloModule module

    %scan_body {
      %carry = f32[] parameter(0)
      %input = f32[] parameter(1)
      %abs1 = f32[] abs(%input)
      %mul1 = f32[] multiply(%abs1, %abs1)
      %mul2 = f32[] multiply(%abs1, %carry)
      %next = f32[] add(%mul1, %mul2)
      ROOT %t = (f32[], f32[]) tuple(%next, %next)
    }

    ENTRY %main (input: f32[128], init: f32[]) -> (f32[128], f32[]) {
      %input = f32[128]{0} parameter(0)
      %init = f32[] parameter(1)
      ROOT %scan = (f32[128]{0}, f32[]) scan(%input, %init), dimensions={0},
          num_carries=1, is_reverse=false, to_apply=%scan_body,
          is_associative=true
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kHloModule));
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, CpuMultiOutputFusion(&alias_info_).Run(hlo_module.get()));
  EXPECT_FALSE(changed) << hlo_module->ToString();

  // Locate the scan body and verify it stays flat (no kFusion instructions).
  HloComputation* body = nullptr;
  for (HloComputation* c : hlo_module->MakeNonfusionComputations()) {
    if (c->name() == "scan_body") {
      body = c;
      break;
    }
  }
  ASSERT_NE(body, nullptr) << "Scan body computation not found";
  for (const HloInstruction* instr : body->instructions()) {
    EXPECT_NE(instr->opcode(), HloOpcode::kFusion)
        << "Found a kFusion inside a scan body: " << instr->ToString();
  }
}

}  // namespace
}  // namespace xla::cpu
