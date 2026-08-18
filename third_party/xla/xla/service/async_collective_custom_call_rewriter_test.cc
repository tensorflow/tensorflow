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

#include "xla/service/async_collective_custom_call_rewriter.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instruction_utils.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/hlo_verifier.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

class AsyncCollectiveCustomCallRewriterTest
    : public HloHardwareIndependentTestBase {};

TEST_F(AsyncCollectiveCustomCallRewriterTest, DirectRewriteLegacyAndGeneric) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY main {
      p0 = f32[16,10] parameter(0)
      start = f32[32,10] custom-call(p0), custom_call_target="all-gather-start",
          frontend_attributes={async_collective_config="{\"all_gather_dimension\":0,\"replica_groups\":[[0,1]]}"}
      ROOT done = f32[32,10] custom-call(start), custom_call_target="all-gather-done"
    }
  )";

  // Legacy collectives mode.
  {
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnUnverifiedModule(hlo_string));
    AsyncCollectiveCustomCallRewriter rewriter(/*use_legacy_collectives=*/true);
    TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
    EXPECT_TRUE(changed);
    EXPECT_TRUE(HloVerifier(HloVerifierOpts{}).Run(module.get()).status().ok());

    HloComputation* entry = module->entry_computation();
    EXPECT_EQ(entry->root_instruction()->opcode(), HloOpcode::kAllGatherDone);
    EXPECT_EQ(entry->root_instruction()->operand(0)->opcode(),
              HloOpcode::kAllGatherStart);
  }

  // Generic async mode.
  {
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnUnverifiedModule(hlo_string));
    AsyncCollectiveCustomCallRewriter rewriter(
        /*use_legacy_collectives=*/false);
    TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
    EXPECT_TRUE(changed);
    EXPECT_TRUE(HloVerifier(HloVerifierOpts{}).Run(module.get()).status().ok());

    HloComputation* entry = module->entry_computation();
    EXPECT_EQ(entry->root_instruction()->opcode(), HloOpcode::kAsyncDone);
    EXPECT_EQ(entry->root_instruction()->operand(0)->opcode(),
              HloOpcode::kAsyncStart);
  }
}

TEST_F(AsyncCollectiveCustomCallRewriterTest, RewriteThroughIntermediaries) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY main {
      p0 = f32[16,10] parameter(0)
      tok = token[] after-all()
      start = f32[32,10] custom-call(p0), custom_call_target="all-gather-start",
          frontend_attributes={async_collective_config="{\"all_gather_dimension\":0,\"replica_groups\":[[0,1]]}"}
      sharding = f32[32,10] custom-call(start), custom_call_target="Sharding"
      tup = (token[], f32[32,10]) tuple(tok, sharding)
      barrier = (token[], f32[32,10]) opt-barrier(tup)
      gte_tok = token[] get-tuple-element(barrier), index=0
      gte_start = f32[32,10] get-tuple-element(barrier), index=1
      done = f32[32,10] custom-call(gte_start), custom_call_target="all-gather-done"
      ROOT res = (token[], f32[32,10]) tuple(gte_tok, done)
    }
  )";

  // Legacy mode.
  {
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnUnverifiedModule(hlo_string));
    AsyncCollectiveCustomCallRewriter rewriter(/*use_legacy_collectives=*/true);
    TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
    EXPECT_TRUE(changed);
    EXPECT_TRUE(HloVerifier(HloVerifierOpts{}).Run(module.get()).status().ok());

    HloInstruction* done =
        module->entry_computation()->root_instruction()->mutable_operand(1);
    EXPECT_EQ(done->opcode(), HloOpcode::kAllGatherDone);
    const HloInstruction* producer =
        hlo_instruction_utils::async::FindAsyncProducer(done->operand(0));
    ASSERT_NE(producer, nullptr);
    EXPECT_EQ(producer->opcode(), HloOpcode::kAllGatherStart);
  }

  // Generic async mode.
  {
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnUnverifiedModule(hlo_string));
    AsyncCollectiveCustomCallRewriter rewriter(
        /*use_legacy_collectives=*/false);
    TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
    EXPECT_TRUE(changed);
    EXPECT_TRUE(HloVerifier(HloVerifierOpts{}).Run(module.get()).status().ok());

    HloInstruction* done =
        module->entry_computation()->root_instruction()->mutable_operand(1);
    EXPECT_EQ(done->opcode(), HloOpcode::kAsyncDone);
    const HloInstruction* producer =
        hlo_instruction_utils::async::FindAsyncProducer(done->operand(0));
    ASSERT_NE(producer, nullptr);
    EXPECT_EQ(producer->opcode(), HloOpcode::kAsyncStart);
  }
}

TEST_F(AsyncCollectiveCustomCallRewriterTest,
       ControlDependenciesPreservedAndRemoved) {
  constexpr absl::string_view hlo_string = R"(
    HloModule test
    ENTRY main {
      p0 = f32[16,10] parameter(0)
      pred_op = f32[16,10] negate(p0)
      start = f32[32,10] custom-call(p0), custom_call_target="all-gather-start",
          frontend_attributes={async_collective_config="{\"all_gather_dimension\":0,\"replica_groups\":[[0,1]]}"},
          control-predecessors={pred_op}
      succ_op = f32[16,10] negate(p0), control-predecessors={start}
      done = f32[32,10] custom-call(start), custom_call_target="all-gather-done",
          control-predecessors={succ_op}
      final_succ = f32[16,10] negate(p0), control-predecessors={done}
      ROOT res = (f32[32,10], f32[16,10], f32[16,10], f32[16,10]) tuple(done, pred_op, succ_op, final_succ)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  AsyncCollectiveCustomCallRewriter rewriter(/*use_legacy_collectives=*/false);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(HloVerifier(HloVerifierOpts{}).Run(module.get()).status().ok());

  HloComputation* entry = module->entry_computation();
  HloInstruction* root = entry->root_instruction();
  HloInstruction* done = root->mutable_operand(0);
  EXPECT_EQ(done->opcode(), HloOpcode::kAsyncDone);
  HloInstruction* start = done->mutable_operand(0);
  EXPECT_EQ(start->opcode(), HloOpcode::kAsyncStart);

  HloInstruction* pred_op = root->mutable_operand(1);
  HloInstruction* succ_op = root->mutable_operand(2);
  HloInstruction* final_succ = root->mutable_operand(3);

  EXPECT_THAT(start->control_predecessors(), ::testing::Contains(pred_op));
  EXPECT_THAT(start->control_successors(), ::testing::Contains(succ_op));
  EXPECT_THAT(done->control_predecessors(), ::testing::Contains(succ_op));
  EXPECT_THAT(done->control_successors(), ::testing::Contains(final_succ));
}

TEST_F(AsyncCollectiveCustomCallRewriterTest,
       UnmatchedOrInvalidConfigHandling) {
  // Non-transparent op between start and done leaves module unchanged.
  {
    constexpr absl::string_view unrewritable_hlo = R"(
      HloModule test
      ENTRY main {
        p0 = f32[16,10] parameter(0)
        start = f32[32,10] custom-call(p0), custom_call_target="all-gather-start",
            frontend_attributes={async_collective_config="{\"all_gather_dimension\":0,\"replica_groups\":[[0,1]]}"}
        neg = f32[32,10] negate(start)
        ROOT done = f32[32,10] custom-call(neg), custom_call_target="all-gather-done"
      }
    )";
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnUnverifiedModule(unrewritable_hlo));
    AsyncCollectiveCustomCallRewriter rewriter(
        /*use_legacy_collectives=*/false);
    TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
    EXPECT_FALSE(changed);
  }

  // Invalid JSON config returns an error status.
  {
    constexpr absl::string_view invalid_config_hlo = R"(
      HloModule test
      ENTRY main {
        p0 = f32[16,10] parameter(0)
        start = f32[32,10] custom-call(p0), custom_call_target="all-gather-start",
            frontend_attributes={async_collective_config="invalid_json"}
        ROOT done = f32[32,10] custom-call(start), custom_call_target="all-gather-done"
      }
    )";
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnUnverifiedModule(invalid_config_hlo));
    AsyncCollectiveCustomCallRewriter rewriter(
        /*use_legacy_collectives=*/false);
    EXPECT_FALSE(rewriter.Run(module.get()).ok());
  }
}

}  // namespace
}  // namespace xla
