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

TEST_F(HloIdentityComputationRemoverTest, BasicValidIdentityComputation) {
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

TEST_F(HloIdentityComputationRemoverTest,
       ValidIdentityComputationWithDeadOperations) {
  const absl::string_view kHlo = R"(
HloModule test
Identity {
  ROOT %param = f32[] parameter(0)
  %copy = f32[] copy(%param)
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

TEST_F(HloIdentityComputationRemoverTest, ValidIdentityComputationTuple) {
  const absl::string_view kHlo = R"(
HloModule test
Identity {
  %param0 = f32[] parameter(0)
  %param1 = f32[] parameter(1)
  ROOT %tuple = (f32[], f32[]) tuple(%param0, %param1)
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

TEST_F(HloIdentityComputationRemoverTest,
       ValidIdentityComputationWithShuffledTuple) {
  const absl::string_view kHlo = R"(
HloModule test
Identity {
  %param0 = f32[] parameter(0)
  %param1 = f32[] parameter(1)
  ROOT %tuple = (f32[], f32[]) tuple(%param1, %param0)
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

TEST_F(HloIdentityComputationRemoverTest,
       ValidIdentityComputationWithNestedTuple) {
  const absl::string_view kHlo = R"(
HloModule test
Identity {
  %param0 = f32[] parameter(0)
  %param1 = f32[] parameter(1)
  %inner_tuple = (f32[]) tuple(%param1)
  ROOT %tuple = ((f32[]), f32[]) tuple(%inner_tuple, %param0)
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

TEST_F(HloIdentityComputationRemoverTest, RootIsNotParameter) {
  const absl::string_view kHlo = R"(
HloModule test
NotIdentity {
  %param = f32[] parameter(0)
  ROOT %negate = f32[] negate(%param)
}
ENTRY main {
  ROOT %param = f32[] parameter(0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  HloComputation* computation = module->GetComputationWithName("NotIdentity");
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

// HloIdentityComputationRemover does not intend to cover scenarios beyond
// identity values from parameters and tuples. Other forms of identity values
// (e.g., copy) should have been handled by other passes.
TEST_F(HloIdentityComputationRemoverTest, IsIdentityComputationWithCopy) {
  const absl::string_view kHlo = R"(
HloModule test
IdentityWithCopy {
  %param0 = f32[] parameter(0)
  %copy_of_param = f32[] copy(%param0)
  ROOT %tuple = (f32[]) tuple(%copy_of_param)
}
ENTRY main {
  ROOT %param = f32[] parameter(0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  HloComputation* computation =
      module->GetComputationWithName("IdentityWithCopy");
  EXPECT_FALSE(
      HloIdentityComputationRemover::IsIdentityComputation(computation));
}

// HloIdentityComputationRemover does not intend to cover scenarios beyond
// identity values from parameters and tuples. Other forms of identity values
// (e.g., bitcast) should have been handled by other passes.
TEST_F(HloIdentityComputationRemoverTest, IsIdentityComputationWithBitcast) {
  const absl::string_view kHlo = R"(
HloModule test
IdentityWithBitcast {
  %param0 = f32[] parameter(0)
  ROOT %bitcast = s32[] bitcast(%param0)
}
ENTRY main {
  ROOT %param = f32[] parameter(0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  HloComputation* computation =
      module->GetComputationWithName("IdentityWithBitcast");
  EXPECT_FALSE(
      HloIdentityComputationRemover::IsIdentityComputation(computation));
}

TEST_F(HloIdentityComputationRemoverTest,
       CallShuffledTupleIdentityComputation) {
  constexpr absl::string_view kHlo = R"(
HloModule HloTest, entry_computation_layout={(f32[], f32[])->(f32[], f32[])}

%tuple_identity_shuffled (p0: f32[], p1: f32[]) -> (f32[], f32[]) {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  ROOT %tuple = (f32[], f32[]) tuple(%p1, %p0)
}, execution_thread="other"

ENTRY %main (p0: f32[], p1: f32[]) -> (f32[], f32[]) {
  %param0 = f32[] parameter(0)
  %param1 = f32[] parameter(1)
  %call = (f32[], f32[]) call(%param0, %param1), to_apply=%tuple_identity_shuffled
  ROOT %copy = (f32[], f32[]) copy(%call)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  HloIdentityComputationRemover remover(/*run_cleanup=*/true);
  ASSERT_TRUE(
      remover.Run(module.get(), /*execution_threads=*/{"other"}).value());
  EXPECT_EQ(module->computation_count(), 1);
  EXPECT_EQ(module->entry_computation()->instruction_count(), 4);
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kCopy);
  const HloInstruction* tuple = root->operand(0);
  EXPECT_EQ(tuple->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(tuple->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(tuple->operand(0)->parameter_number(), 1);
  EXPECT_EQ(tuple->operand(1)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(tuple->operand(1)->parameter_number(), 0);
}

TEST_F(HloIdentityComputationRemoverTest,
       CallShuffledTupleIdentityComputationAsync) {
  constexpr absl::string_view kHlo = R"(
HloModule HloTest, entry_computation_layout={(f32[], f32[])->(f32[], f32[])}

%tuple_identity_shuffled (p0: f32[], p1: f32[]) -> (f32[], f32[]) {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  ROOT %tuple = (f32[], f32[]) tuple(%p1, %p0)
}, execution_thread="other"

%async_computation (p0: f32[], p1: f32[]) -> (f32[], f32[]) {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  ROOT %call = (f32[], f32[]) call(%p0, %p1), to_apply=%tuple_identity_shuffled
}, execution_thread="other"

ENTRY %main (p0: f32[], p1: f32[]) -> (f32[], f32[]) {
  %param0 = f32[] parameter(0)
  %param1 = f32[] parameter(1)
  %async-start = ((f32[], f32[]), (f32[], f32[]), u32[]) async-start(%param0, %param1), async_execution_thread="other", calls=%async_computation
  %async-done = (f32[], f32[]) async-done(%async-start)
  ROOT %copy = (f32[], f32[]) copy(%async-done)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  HloIdentityComputationRemover remover(/*run_cleanup=*/true);
  ASSERT_TRUE(
      remover.Run(module.get(), /*execution_threads=*/{"other"}).value());
  EXPECT_EQ(module->computation_count(), 1);
  EXPECT_EQ(module->entry_computation()->instruction_count(), 4);
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kCopy);
  const HloInstruction* tuple = root->operand(0);
  EXPECT_EQ(tuple->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(tuple->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(tuple->operand(0)->parameter_number(), 1);
  EXPECT_EQ(tuple->operand(1)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(tuple->operand(1)->parameter_number(), 0);
}

TEST_F(HloIdentityComputationRemoverTest, CallTupleIdentityComputation) {
  constexpr absl::string_view kHlo = R"(
HloModule HloTest, entry_computation_layout={(f32[], f32[])->(f32[], f32[])}

%tuple_identity (p0: f32[], p1: f32[]) -> (f32[], f32[]) {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  ROOT %tuple = (f32[], f32[]) tuple(%p0, %p1)
}, execution_thread="other"

ENTRY %main (p0: f32[], p1: f32[]) -> (f32[], f32[]) {
  %param0 = f32[] parameter(0)
  %param1 = f32[] parameter(1)
  %call = (f32[], f32[]) call(%param0, %param1), to_apply=%tuple_identity
  ROOT %copy = (f32[], f32[]) copy(%call)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  HloIdentityComputationRemover remover(/*run_cleanup=*/true);
  ASSERT_TRUE(
      remover.Run(module.get(), /*execution_threads=*/{"other"}).value());
  EXPECT_EQ(module->computation_count(), 1);
  EXPECT_EQ(module->entry_computation()->instruction_count(), 4);
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kCopy);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(root->operand(0)->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(root->operand(0)->operand(1)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(root->operand(0)->operand(0)->parameter_number(), 0);
  EXPECT_EQ(root->operand(0)->operand(1)->parameter_number(), 1);
}

TEST_F(HloIdentityComputationRemoverTest, CallTupleIdentityComputationAsync) {
  constexpr absl::string_view kHlo = R"(
HloModule HloTest, entry_computation_layout={(f32[], f32[])->(f32[], f32[])}

%tuple_identity (p0: f32[], p1: f32[]) -> (f32[], f32[]) {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  ROOT %tuple = (f32[], f32[]) tuple(%p0, %p1)
}, execution_thread="other"

%async_computation (p0: f32[], p1: f32[]) -> (f32[], f32[]) {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  ROOT %call = (f32[], f32[]) call(%p0, %p1), to_apply=%tuple_identity
}, execution_thread="other"

ENTRY %main (p0: f32[], p1: f32[]) -> (f32[], f32[]) {
  %param0 = f32[] parameter(0)
  %param1 = f32[] parameter(1)
  %async-start = ((f32[], f32[]), (f32[], f32[]), u32[]) async-start(%param0, %param1), async_execution_thread="other", calls=%async_computation
  %async-done = (f32[], f32[]) async-done(%async-start)
  ROOT %copy = (f32[], f32[]) copy(%async-done)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  HloIdentityComputationRemover remover(/*run_cleanup=*/true);
  ASSERT_TRUE(
      remover.Run(module.get(), /*execution_threads=*/{"other"}).value());
  EXPECT_EQ(module->computation_count(), 1);
  EXPECT_EQ(module->entry_computation()->instruction_count(), 4);
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kCopy);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(root->operand(0)->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(root->operand(0)->operand(1)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(root->operand(0)->operand(0)->parameter_number(), 0);
  EXPECT_EQ(root->operand(0)->operand(1)->parameter_number(), 1);
}

TEST_F(HloIdentityComputationRemoverTest, RemoveCallAndAsyncStartAndAsyncDone) {
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
  HloIdentityComputationRemover remover(/*run_cleanup=*/true);
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

TEST_F(HloIdentityComputationRemoverTest,
       CallShuffledTupleIdentityComputationNoCopy) {
  constexpr absl::string_view kHlo = R"(
HloModule HloTest, entry_computation_layout={(f32[], f32[])->(f32[], f32[])}

%tuple_identity_shuffled (p0: f32[], p1: f32[]) -> (f32[], f32[]) {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  ROOT %tuple = (f32[], f32[]) tuple(%p1, %p0)
}, execution_thread="other"

ENTRY %main (p0: f32[], p1: f32[]) -> (f32[], f32[]) {
  %param0 = f32[] parameter(0)
  %param1 = f32[] parameter(1)
  ROOT %call = (f32[], f32[]) call(%param0, %param1), to_apply=%tuple_identity_shuffled
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  HloIdentityComputationRemover remover(/*run_cleanup=*/true);
  ASSERT_TRUE(
      remover.Run(module.get(), /*execution_threads=*/{"other"}).value());
  EXPECT_EQ(module->computation_count(), 1);
  EXPECT_EQ(module->entry_computation()->instruction_count(), 3);
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(root->operand(0)->parameter_number(), 1);
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(root->operand(1)->parameter_number(), 0);
}

TEST_F(HloIdentityComputationRemoverTest,
       CallShuffledTupleIdentityComputationAsyncNoCopy) {
  constexpr absl::string_view kHlo = R"(
HloModule HloTest, entry_computation_layout={(f32[], f32[])->(f32[], f32[])}

%tuple_identity_shuffled (p0: f32[], p1: f32[]) -> (f32[], f32[]) {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  ROOT %tuple = (f32[], f32[]) tuple(%p1, %p0)
}, execution_thread="other"

%async_computation (p0: f32[], p1: f32[]) -> (f32[], f32[]) {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  ROOT %call = (f32[], f32[]) call(%p0, %p1), to_apply=%tuple_identity_shuffled
}, execution_thread="other"

ENTRY %main (p0: f32[], p1: f32[]) -> (f32[], f32[]) {
  %param0 = f32[] parameter(0)
  %param1 = f32[] parameter(1)
  %async-start = ((f32[], f32[]), (f32[], f32[]), u32[]) async-start(%param0, %param1), async_execution_thread="other", calls=%async_computation
  ROOT %async-done = (f32[], f32[]) async-done(%async-start)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  HloIdentityComputationRemover remover(/*run_cleanup=*/true);
  ASSERT_TRUE(
      remover.Run(module.get(), /*execution_threads=*/{"other"}).value());
  EXPECT_EQ(module->computation_count(), 1);
  EXPECT_EQ(module->entry_computation()->instruction_count(), 3);
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(root->operand(0)->parameter_number(), 1);
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(root->operand(1)->parameter_number(), 0);
}

TEST_F(HloIdentityComputationRemoverTest,
       CallUnshuffledTupleIdentityComputation) {
  constexpr absl::string_view kHlo = R"(
HloModule HloTest, entry_computation_layout={(f32[], f32[])->(f32[], f32[])}

%tuple_identity_unshuffled (p0: f32[], p1: f32[]) -> (f32[], f32[]) {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  ROOT %tuple = (f32[], f32[]) tuple(%p0, %p1)
}, execution_thread="other"

ENTRY %main (p0: f32[], p1: f32[]) -> (f32[], f32[]) {
  %param0 = f32[] parameter(0)
  %param1 = f32[] parameter(1)
  %call = (f32[], f32[]) call(%param0, %param1), to_apply=%tuple_identity_unshuffled
  ROOT %copy = (f32[], f32[]) copy(%call)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  HloIdentityComputationRemover remover(/*run_cleanup=*/true);
  ASSERT_TRUE(
      remover.Run(module.get(), /*execution_threads=*/{"other"}).value());
  EXPECT_EQ(module->computation_count(), 1);
  EXPECT_EQ(module->entry_computation()->instruction_count(), 4);
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kCopy);
  const HloInstruction* tuple = root->operand(0);
  EXPECT_EQ(tuple->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(tuple->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(tuple->operand(0)->parameter_number(), 0);
  EXPECT_EQ(tuple->operand(1)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(tuple->operand(1)->parameter_number(), 1);
}

TEST_F(HloIdentityComputationRemoverTest,
       CallUnshuffledTupleIdentityComputationAsync) {
  constexpr absl::string_view kHlo = R"(
HloModule HloTest, entry_computation_layout={(f32[], f32[])->(f32[], f32[])}

%tuple_identity_unshuffled (p0: f32[], p1: f32[]) -> (f32[], f32[]) {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  ROOT %tuple = (f32[], f32[]) tuple(%p0, %p1)
}, execution_thread="other"

%async_computation (p0: f32[], p1: f32[]) -> (f32[], f32[]) {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  ROOT %call = (f32[], f32[]) call(%p0, %p1), to_apply=%tuple_identity_unshuffled
}, execution_thread="other"

ENTRY %main (p0: f32[], p1: f32[]) -> (f32[], f32[]) {
  %param0 = f32[] parameter(0)
  %param1 = f32[] parameter(1)
  %async-start = ((f32[], f32[]), (f32[], f32[]), u32[]) async-start(%param0, %param1), async_execution_thread="other", calls=%async_computation
  %async-done = (f32[], f32[]) async-done(%async-start)
  ROOT %copy = (f32[], f32[]) copy(%async-done)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  HloIdentityComputationRemover remover(/*run_cleanup=*/true);
  ASSERT_TRUE(
      remover.Run(module.get(), /*execution_threads=*/{"other"}).value());
  EXPECT_EQ(module->computation_count(), 1);
  EXPECT_EQ(module->entry_computation()->instruction_count(), 4);
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kCopy);
  const HloInstruction* tuple = root->operand(0);
  EXPECT_EQ(tuple->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(tuple->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(tuple->operand(0)->parameter_number(), 0);
  EXPECT_EQ(tuple->operand(1)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(tuple->operand(1)->parameter_number(), 1);
}

TEST_F(HloIdentityComputationRemoverTest,
       CallUnshuffledTupleIdentityComputationNoCopy) {
  constexpr absl::string_view kHlo = R"(
HloModule HloTest, entry_computation_layout={(f32[], f32[])->(f32[], f32[])}

%tuple_identity_unshuffled (p0: f32[], p1: f32[]) -> (f32[], f32[]) {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  ROOT %tuple = (f32[], f32[]) tuple(%p0, %p1)
}, execution_thread="other"

ENTRY %main (p0: f32[], p1: f32[]) -> (f32[], f32[]) {
  %param0 = f32[] parameter(0)
  %param1 = f32[] parameter(1)
  ROOT %call = (f32[], f32[]) call(%param0, %param1), to_apply=%tuple_identity_unshuffled
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  HloIdentityComputationRemover remover(/*run_cleanup=*/true);
  ASSERT_TRUE(
      remover.Run(module.get(), /*execution_threads=*/{"other"}).value());
  EXPECT_EQ(module->computation_count(), 1);
  EXPECT_EQ(module->entry_computation()->instruction_count(), 3);
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(root->operand(0)->parameter_number(), 0);
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(root->operand(1)->parameter_number(), 1);
}

TEST_F(HloIdentityComputationRemoverTest,
       CallUnshuffledTupleIdentityComputationAsyncNoCopy) {
  constexpr absl::string_view kHlo = R"(
HloModule HloTest, entry_computation_layout={(f32[], f32[])->(f32[], f32[])}

%tuple_identity_unshuffled (p0: f32[], p1: f32[]) -> (f32[], f32[]) {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  ROOT %tuple = (f32[], f32[]) tuple(%p0, %p1)
}, execution_thread="other"

%async_computation (p0: f32[], p1: f32[]) -> (f32[], f32[]) {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  ROOT %call = (f32[], f32[]) call(%p0, %p1), to_apply=%tuple_identity_unshuffled
}, execution_thread="other"

ENTRY %main (p0: f32[], p1: f32[]) -> (f32[], f32[]) {
  %param0 = f32[] parameter(0)
  %param1 = f32[] parameter(1)
  %async-start = ((f32[], f32[]), (f32[], f32[]), u32[]) async-start(%param0, %param1), async_execution_thread="other", calls=%async_computation
  ROOT %async-done = (f32[], f32[]) async-done(%async-start)
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  HloIdentityComputationRemover remover(/*run_cleanup=*/true);
  ASSERT_TRUE(
      remover.Run(module.get(), /*execution_threads=*/{"other"}).value());
  EXPECT_EQ(module->computation_count(), 1);
  EXPECT_EQ(module->entry_computation()->instruction_count(), 3);
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(root->operand(0)->parameter_number(), 0);
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(root->operand(1)->parameter_number(), 1);
}

TEST_F(HloIdentityComputationRemoverTest,
       RemoveIdentityComputationWithNestedTuple) {
  const absl::string_view kHlo = R"(
HloModule test, entry_computation_layout={(f32[], f32[])->((f32[]), f32[])}
IdentityWithNestedTuple {
  %param0 = f32[] parameter(0)
  %param1 = f32[] parameter(1)
  %inner_tuple = (f32[]) tuple(%param1)
  ROOT %tuple = ((f32[]), f32[]) tuple(%inner_tuple, %param0)
}, execution_thread="other"
ENTRY main {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  ROOT %call = ((f32[]), f32[]) call(%p0, %p1), to_apply=IdentityWithNestedTuple
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  HloIdentityComputationRemover remover(/*run_cleanup=*/true);
  ASSERT_TRUE(
      remover.Run(module.get(), /*execution_threads=*/{"other"}).value());
  EXPECT_EQ(module->computation_count(), 1);
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(root->operand(0)->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(root->operand(0)->operand(0)->parameter_number(), 1);
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(root->operand(1)->parameter_number(), 0);
}

TEST_F(HloIdentityComputationRemoverTest,
       RemoveIdentityComputationWithDeeplyNestedShuffledTuple) {
  const absl::string_view kHlo = R"(
HloModule test, entry_computation_layout={(f32[], f32[])->((f32[], (f32[])), f32[])}
IdentityWithDeeplyNestedTuple {
  %param0 = f32[] parameter(0)
  %param1 = f32[] parameter(1)
  %inner_tuple = (f32[]) tuple(%param1)
  %middle_tuple = (f32[], (f32[])) tuple(%param0, %inner_tuple)
  ROOT %tuple = ((f32[], (f32[])), f32[]) tuple(%middle_tuple, %param0)
}, execution_thread="other"
ENTRY main {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  ROOT %call = ((f32[], (f32[])), f32[]) call(%p0, %p1), to_apply=IdentityWithDeeplyNestedTuple
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  HloIdentityComputationRemover remover(/*run_cleanup=*/true);
  ASSERT_TRUE(
      remover.Run(module.get(), /*execution_threads=*/{"other"}).value());
  EXPECT_EQ(module->computation_count(), 1);
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kTuple);
  const HloInstruction* middle_tuple = root->operand(0);
  EXPECT_EQ(middle_tuple->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(middle_tuple->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(middle_tuple->operand(0)->parameter_number(), 0);
  const HloInstruction* inner_tuple = middle_tuple->operand(1);
  EXPECT_EQ(inner_tuple->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(inner_tuple->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(inner_tuple->operand(0)->parameter_number(), 1);
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(root->operand(1)->parameter_number(), 0);
}

TEST_F(HloIdentityComputationRemoverTest,
       RemoveIdentityComputationWithDeeplyNestedUnshuffledTuple) {
  const absl::string_view kHlo = R"(
HloModule test, entry_computation_layout={(f32[], f32[])->((f32[], (f32[])), f32[])}
IdentityWithDeeplyNestedTuple {
  %param0 = f32[] parameter(0)
  %param1 = f32[] parameter(1)
  %inner_tuple = (f32[]) tuple(%param1)
  %middle_tuple = (f32[], (f32[])) tuple(%param0, %inner_tuple)
  ROOT %tuple = ((f32[], (f32[])), f32[]) tuple(%middle_tuple, %param1)
}, execution_thread="other"
ENTRY main {
  %p0 = f32[] parameter(0)
  %p1 = f32[] parameter(1)
  ROOT %call = ((f32[], (f32[])), f32[]) call(%p0, %p1), to_apply=IdentityWithDeeplyNestedTuple
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  HloIdentityComputationRemover remover(/*run_cleanup=*/true);
  ASSERT_TRUE(
      remover.Run(module.get(), /*execution_threads=*/{"other"}).value());
  EXPECT_EQ(module->computation_count(), 1);
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kTuple);
  const HloInstruction* middle_tuple = root->operand(0);
  EXPECT_EQ(middle_tuple->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(middle_tuple->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(middle_tuple->operand(0)->parameter_number(), 0);
  const HloInstruction* inner_tuple = middle_tuple->operand(1);
  EXPECT_EQ(inner_tuple->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(inner_tuple->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(inner_tuple->operand(0)->parameter_number(), 1);
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(root->operand(1)->parameter_number(), 1);
}

TEST_F(HloIdentityComputationRemoverTest,
       RemoveIdentityComputationWithDuplicateParameterUse) {
  const absl::string_view kHlo = R"(
HloModule test, entry_computation_layout={(f32[])->(f32[], f32[])}
IdentityWithDuplicateParams {
  %param0 = f32[] parameter(0)
  ROOT %tuple = (f32[], f32[]) tuple(%param0, %param0)
}, execution_thread="other"
ENTRY main {
  %p0 = f32[] parameter(0)
  ROOT %call = (f32[], f32[]) call(%p0), to_apply=IdentityWithDuplicateParams
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  HloIdentityComputationRemover remover(/*run_cleanup=*/true);
  ASSERT_TRUE(
      remover.Run(module.get(), /*execution_threads=*/{"other"}).value());
  EXPECT_EQ(module->computation_count(), 1);
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(root->operand(0)->parameter_number(), 0);
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(root->operand(1)->parameter_number(), 0);
}

TEST_F(HloIdentityComputationRemoverTest,
       RemoveIdentityComputationWithNestedDuplicateParameterUse) {
  const absl::string_view kHlo = R"(
HloModule test, entry_computation_layout={(f32[])->((f32[]), f32[])}
IdentityWithNestedDuplicateParams {
  %param0 = f32[] parameter(0)
  %inner_tuple = (f32[]) tuple(%param0)
  ROOT %tuple = ((f32[]), f32[]) tuple(%inner_tuple, %param0)
}, execution_thread="other"
ENTRY main {
  %p0 = f32[] parameter(0)
  ROOT %call = ((f32[]), f32[]) call(%p0), to_apply=IdentityWithNestedDuplicateParams
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  HloIdentityComputationRemover remover(/*run_cleanup=*/true);
  ASSERT_TRUE(
      remover.Run(module.get(), /*execution_threads=*/{"other"}).value());
  EXPECT_EQ(module->computation_count(), 1);
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kTuple);
  const HloInstruction* inner_tuple = root->operand(0);
  EXPECT_EQ(inner_tuple->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(inner_tuple->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(inner_tuple->operand(0)->parameter_number(), 0);
  EXPECT_EQ(root->operand(1)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(root->operand(1)->parameter_number(), 0);
}

TEST_F(HloIdentityComputationRemoverTest,
       RemoveIdentityComputationWithControlDependencies) {
  const absl::string_view kHlo = R"(
HloModule test_module, entry_computation_layout={(f32[])->f32[]}

%identity (p: f32[]) -> f32[] {
  ROOT %p = f32[] parameter(0)
}, execution_thread="other"

%async_computation (p: f32[]) -> f32[] {
  %p = f32[] parameter(0)
  ROOT %call = f32[] call(%p), to_apply=%identity
}, execution_thread="other"

%fused_computation (p1: f32[], p2: f32[]) -> f32[] {
  %p1 = f32[] parameter(0)
  %p2 = f32[] parameter(1)
  ROOT %add = f32[] add(%p1, %p2)
}

ENTRY %main (p: f32[]) -> f32[] {
  %p = f32[] parameter(0)
  %async-start = ((f32[]), f32[], u32[]) async-start(%p), async_execution_thread="other", calls=%async_computation
  %async-done = f32[] async-done(%async-start)
  %fusion = f32[] fusion(%p, %async-done), kind=kLoop, calls=%fused_computation, control-predecessors={%async-start, %async-done}
  ROOT %copy = f32[] copy(%fusion)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  HloIdentityComputationRemover remover(/*run_cleanup=*/true);
  ASSERT_TRUE(
      remover.Run(module.get(), /*execution_threads=*/{"other"}).value());
  EXPECT_EQ(module->computation_count(), 2);
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kCopy);
  HloInstruction* fusion = root->mutable_operand(0);
  EXPECT_EQ(fusion->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(fusion->control_predecessors().size(), 1);
  EXPECT_EQ(fusion->control_predecessors()[0], fusion->operand(0));
}

}  // namespace
}  // namespace xla
