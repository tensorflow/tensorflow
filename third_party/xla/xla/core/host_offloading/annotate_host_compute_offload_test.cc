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

#include "xla/core/host_offloading/annotate_host_compute_offload.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/host_offload_utils.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

class AnnotateHostComputeOffloadTest : public HloHardwareIndependentTestBase {
 protected:
  static absl::StatusOr<bool> RunPasses(HloModule* module) {
    HloPassPipeline pipeline("AnnotateHostComputeOffloadTest");
    pipeline.AddPass<AnnotateHostComputeOffload>();
    return pipeline.Run(module);
  }

  static void ExpectAllHostInstructions(HloComputation& computation) {
    for (HloInstruction* instruction : computation.instructions()) {
      if (instruction->opcode() == HloOpcode::kParameter) {
        continue;
      }
      EXPECT_TRUE(host_offload_utils::ComputeTypeIsHost(instruction));
    }
  }
};

TEST_F(AnnotateHostComputeOffloadTest, TestUnmodifiedModule) {
  std::string hlo_string = R"(
  HloModule TestUnmodifiedModule

  ENTRY main {
    c = s32[] constant(1)
    b = s32[32] broadcast(c)
    ROOT copy = s32[32] copy(b)
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> verified_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool modified, RunPasses(verified_module.get()));
  EXPECT_FALSE(modified);
}

// Test that the host compute offload instructions are annotated for a host
// call.
TEST_F(AnnotateHostComputeOffloadTest, AnnotatesHostComputeOffloadForHostCall) {
  std::string hlo_string = R"(
  HloModule AnnotatesHostComputeOffloadForHostCall
  
  host_compute_offload {
    p0 = s32[32]{0} parameter(0)
    p1 = s32[32]{0} parameter(1)
    ROOT add = add(p0, p1)
  }

  ENTRY main {
    p0 = s32[32]{0} parameter(0)
    p1 = s32[32]{0} parameter(1)
    ROOT call = s32[32] call(p0, p1), to_apply=host_compute_offload, frontend_attributes={_xla_compute_type="host"}
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> verified_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool modified, RunPasses(verified_module.get()));
  EXPECT_TRUE(modified);

  HloComputation* host_compute_offload =
      verified_module->GetComputationWithName("host_compute_offload");
  ASSERT_NE(host_compute_offload, nullptr);
  ExpectAllHostInstructions(*host_compute_offload);
}

// Test that the host compute offload instructions are annotated for a nested
// host compute offload.
TEST_F(AnnotateHostComputeOffloadTest,
       AnnotatesNestedHostComputeOffloadForHostCall) {
  std::string hlo_string = R"(
  HloModule AnnotatesNestedHostComputeOffloadForHostCall
  
  nested_host_compute_offload {
    p0 = s32[32]{0} parameter(0)
    p1 = s32[32]{0} parameter(1)
    ROOT add = add(p0, p1)
  }
  
  host_compute_offload {
    p0 = s32[32]{0} parameter(0)
    p1 = s32[32]{0} parameter(1)
    ROOT call = s32[32] call(p0, p1), to_apply=nested_host_compute_offload
  }

  ENTRY main {
    p0 = s32[32]{0} parameter(0)
    p1 = s32[32]{0} parameter(1)
    ROOT call = s32[32] call(p0, p1), to_apply=host_compute_offload, frontend_attributes={_xla_compute_type="host"}
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> verified_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool modified, RunPasses(verified_module.get()));
  EXPECT_TRUE(modified);

  HloComputation* host_compute_offload =
      verified_module->GetComputationWithName("host_compute_offload");
  ASSERT_NE(host_compute_offload, nullptr);
  ExpectAllHostInstructions(*host_compute_offload);

  HloComputation* nested_host_compute_offload =
      verified_module->GetComputationWithName("nested_host_compute_offload");
  ASSERT_NE(nested_host_compute_offload, nullptr);
  ExpectAllHostInstructions(*nested_host_compute_offload);
}

}  // namespace
}  // namespace xla
