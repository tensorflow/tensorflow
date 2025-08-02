/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/hlo_identity_computation_remover.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

class HloIdentityComputationRemoverTest
    : public HloHardwareIndependentTestBase {};

TEST_F(HloIdentityComputationRemoverTest, ValidIdentityComputation) {
  const absl::string_view kHlo = R"(
HloModule test
Identity {
  ROOT %param = f32[] parameter(0)
}
ENTRY main {
  ROOT %param = f32[] parameter(0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  HloComputation* computation = module->GetComputationWithName("Identity");
  EXPECT_TRUE(
      HloIdentityComputationRemover::IsIdentityComputation(computation));
}

TEST_F(HloIdentityComputationRemoverTest, MultipleInstructions) {
  const absl::string_view kHlo = R"(
HloModule test
NotIdentity_MultipleInstructions {
  %param = f32[] parameter(0)
  ROOT %negate = f32[] negate(%param)
}
ENTRY main {
  ROOT %param = f32[] parameter(0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  HloComputation* computation =
      module->GetComputationWithName("NotIdentity_MultipleInstructions");
  EXPECT_FALSE(
      HloIdentityComputationRemover::IsIdentityComputation(computation));
}

TEST_F(HloIdentityComputationRemoverTest, EntryComputation) {
  const absl::string_view kHlo = R"(
HloModule test
ENTRY main {
  ROOT %param = f32[] parameter(0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  HloComputation* computation = module->GetComputationWithName("main");
  EXPECT_FALSE(
      HloIdentityComputationRemover::IsIdentityComputation(computation));
}

TEST_F(HloIdentityComputationRemoverTest, CallAndAsyncStartAndAyncDone) {
  constexpr absl::string_view kHlo = R"(
HloModule HloTest, entry_computation_layout={(f32[128]{0})->f32[128]{0}}

%called_computation.1 (param: f32[128]) -> f32[128] {
  ROOT %param = f32[128]{0} parameter(0)
}, execution_thread="other"

%async_computation.1 (param: f32[128]) -> f32[128] {
  %param = f32[128]{0} parameter(0)
  ROOT %call = f32[128]{0} call(%param), to_apply=%called_computation.1
}, execution_thread="other"

%called_computation (param: f32[128]) -> f32[128] {
  %param = f32[128]{0} parameter(0)
  %async-start = ((f32[128]{0}), f32[128]{0}, u32[]) async-start(%param), async_execution_thread="other", calls=%async_computation.1
  ROOT %async-done = f32[128]{0} async-done(%async-start)
}, execution_thread="other"

%async_computation (param: f32[128]) -> f32[128] {
  %param = f32[128]{0} parameter(0)
  ROOT %call = f32[128]{0} call(%param), to_apply=%called_computation
}, execution_thread="other"

ENTRY %main (param: f32[128]) -> f32[128] {
  %param = f32[128]{0} parameter(0)
  %async-start = ((f32[128]{0}), f32[128]{0}, u32[]) async-start(%param), async_execution_thread="other", calls=%async_computation
  %async-done = f32[128]{0} async-done(%async-start)
  ROOT %copy = f32[128]{0} copy(%async-done)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  HloIdentityComputationRemover remover(/*should_run_dce=*/true);
  ASSERT_TRUE(
      remover.Run(module.get(), /*execution_threads=*/{"other"}).value());
  EXPECT_EQ(module->computation_count(), 1);
  EXPECT_EQ(module->entry_computation()->instruction_count(), 2);
  EXPECT_EQ(
      module->entry_computation()->root_instruction()->operand(0)->opcode(),
      HloOpcode::kParameter);
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kCopy);
}

}  // namespace
}  // namespace xla
