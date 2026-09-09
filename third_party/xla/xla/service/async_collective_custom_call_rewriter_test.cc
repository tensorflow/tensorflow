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
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instruction_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/hlo_verifier.h"

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
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnUnverifiedModule(hlo_string));
    AsyncCollectiveCustomCallRewriter rewriter(/*use_legacy_collectives=*/true);
    ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
    EXPECT_TRUE(changed);
    EXPECT_TRUE(HloVerifier(HloVerifierOpts{}).Run(module.get()).status().ok());

    HloComputation* entry = module->entry_computation();
    EXPECT_EQ(entry->root_instruction()->opcode(), HloOpcode::kAllGatherDone);
    EXPECT_EQ(entry->root_instruction()->operand(0)->opcode(),
              HloOpcode::kAllGatherStart);
  }

  // Generic async mode.
  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnUnverifiedModule(hlo_string));
    AsyncCollectiveCustomCallRewriter rewriter(
        /*use_legacy_collectives=*/false);
    ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
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
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnUnverifiedModule(hlo_string));
    AsyncCollectiveCustomCallRewriter rewriter(/*use_legacy_collectives=*/true);
    ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
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
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnUnverifiedModule(hlo_string));
    AsyncCollectiveCustomCallRewriter rewriter(
        /*use_legacy_collectives=*/false);
    ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(hlo_string));
  AsyncCollectiveCustomCallRewriter rewriter(/*use_legacy_collectives=*/false);
  ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
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
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnUnverifiedModule(unrewritable_hlo));
    AsyncCollectiveCustomCallRewriter rewriter(
        /*use_legacy_collectives=*/false);
    ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
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
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnUnverifiedModule(invalid_config_hlo));
    AsyncCollectiveCustomCallRewriter rewriter(
        /*use_legacy_collectives=*/false);
    EXPECT_FALSE(rewriter.Run(module.get()).ok());
  }
}

TEST_F(AsyncCollectiveCustomCallRewriterTest, ComputeOnStart) {
  constexpr absl::string_view hlo_string = R"(
      HloModule test
      %compute_on.2.2.clone (new_param: f32[256,256]) -> f32[256,2048] {
  %new_param = f32[256,256]{1,0} parameter(0)
  ROOT %all_gather_reduced.2 = f32[256,2048]{1,0} all-gather(%new_param), channel_id=1, replica_groups=mesh['x'=8] {'x'}, dimensions={1}, use_global_device_ids=true, frontend_attributes={_xla_compute_type="sparseoffload"}, metadata={op_name="shard_map/all_gather_reduced" stack_frame_id=14}
}

ENTRY %main.0_spmd (param.1: f32[128,2048], param.2: f32[256,256], param: f32[256,256]) -> f32[128,2048] {
  %param = f32[256,256]{1,0} parameter(2), sharding={devices=[1,8]<=[8]}, metadata={op_name="w2"}
  %compute_on.2 = f32[256,2048]{1,0} custom-call(%param), custom_call_target="compute-on-start", called_computations={%compute_on.2.2.clone}, frontend_attributes={backend_config={"sparse_core_config": {"core_ids": [0], "core_id_mutability": false}}}, metadata={op_name="jit(f)/compute_on" stack_frame_id=13}
  %param.1 = f32[128,2048]{1,0} parameter(0), sharding={devices=[8,1]<=[8]}, metadata={op_name="x"}
  %param.2 = f32[256,256]{1,0} parameter(1), sharding={devices=[8,1]<=[8]}, metadata={op_name="w1"}
  %tuple.2 = (f32[256,2048]{1,0}, f32[128,2048]{1,0}, f32[256,256]{1,0}) tuple(%compute_on.2, %param.1, %param.2), metadata={op_name="jit(f)/optimization_barrier" stack_frame_id=15}
  %optimization_barrier.8 = (f32[256,2048]{1,0}, f32[128,2048]{1,0}, f32[256,256]{1,0}) opt-barrier(%tuple.2), metadata={op_name="jit(f)/optimization_barrier" stack_frame_id=15}
  %all-gather.2 = f32[2048,256]{1,0} all-gather(%optimization_barrier.8#2), channel_id=4, replica_groups=mesh['axis_0'=8] {'axis_0'}, dimensions={0}, use_global_device_ids=true, frontend_attributes={is_spmd_generated="true"}, metadata={op_name="jit(f)/optimization_barrier" stack_frame_id=15}
  %dot.1 = f32[128,256]{1,0} dot(%optimization_barrier.8#1, %all-gather.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="jit(f)/dot_general" stack_frame_id=15}
  %tuple.4 = (f32[128,256]{1,0}, f32[256,2048]{1,0}) tuple(%dot.1, %optimization_barrier.8#0), metadata={op_name="jit(f)/optimization_barrier" stack_frame_id=16}
  %optimization_barrier.9 = (f32[128,256]{1,0}, f32[256,2048]{1,0}) opt-barrier(%tuple.4), metadata={op_name="jit(f)/optimization_barrier" stack_frame_id=16}
  %compute_on_done.2 = f32[256,2048]{1,0} custom-call(%optimization_barrier.9#1), custom_call_target="compute-on-done", metadata={op_name="jit(f)/compute_on_done" stack_frame_id=16}
  %tuple.6 = (f32[128,256]{1,0}, f32[256,2048]{1,0}) tuple(%optimization_barrier.9#0, %compute_on_done.2), metadata={op_name="jit(f)/optimization_barrier" stack_frame_id=17}
  %optimization_barrier.10 = (f32[128,256]{1,0}, f32[256,2048]{1,0}) opt-barrier(%tuple.6), metadata={op_name="jit(f)/optimization_barrier" stack_frame_id=17}
  ROOT %dot.3 = f32[128,2048]{1,0} dot(%optimization_barrier.10#0, %optimization_barrier.10#1), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="jit(f)/dot_general" stack_frame_id=17}
}

    )";

  // The rewritten done op only reaches the root through %tuple.6 and
  // %optimization_barrier.10, so trace back through those intermediaries.
  auto find_async_done = [](HloModule* module) {
    return hlo_instruction_utils::async::TraceAsyncDataflow(
        module->entry_computation()->root_instruction()->mutable_operand(1),
        [](const HloInstruction* instr) { return instr->IsAsyncDone(); });
  };

  // Legacy collectives mode: the compute-on pair becomes an all-gather
  // start/done pair configured from the all-gather inside the compute-on
  // computation.
  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnUnverifiedModule(hlo_string));
    AsyncCollectiveCustomCallRewriter rewriter(/*use_legacy_collectives=*/true);
    ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
    EXPECT_TRUE(changed);
    EXPECT_TRUE(HloVerifier(HloVerifierOpts{}).Run(module.get()).status().ok());

    HloInstruction* done = find_async_done(module.get());
    ASSERT_NE(done, nullptr);
    EXPECT_EQ(done->opcode(), HloOpcode::kAllGatherDone);
    const HloInstruction* producer =
        hlo_instruction_utils::async::FindAsyncProducer(done->operand(0));
    ASSERT_NE(producer, nullptr);
    ASSERT_EQ(producer->opcode(), HloOpcode::kAllGatherStart);

    const auto* all_gather_start = Cast<HloAllGatherInstruction>(producer);
    EXPECT_EQ(all_gather_start->all_gather_dimension(), 1);
    EXPECT_EQ(all_gather_start->channel_id(), 1);
    EXPECT_TRUE(all_gather_start->use_global_device_ids());
  }

  // Generic async mode: the compute-on pair becomes an async start/done pair
  // wrapping a call to the compute-on computation.
  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnUnverifiedModule(hlo_string));
    AsyncCollectiveCustomCallRewriter rewriter(
        /*use_legacy_collectives=*/false);
    ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
    EXPECT_TRUE(changed);
    EXPECT_TRUE(HloVerifier(HloVerifierOpts{}).Run(module.get()).status().ok());

    HloInstruction* done = find_async_done(module.get());
    ASSERT_NE(done, nullptr);
    EXPECT_EQ(done->opcode(), HloOpcode::kAsyncDone);
    const HloInstruction* producer =
        hlo_instruction_utils::async::FindAsyncProducer(done->operand(0));
    ASSERT_NE(producer, nullptr);
    ASSERT_EQ(producer->opcode(), HloOpcode::kAsyncStart);

    const HloInstruction* wrapped = producer->async_wrapped_instruction();
    EXPECT_EQ(wrapped->opcode(), HloOpcode::kCall);
    ASSERT_EQ(wrapped->called_computations().size(), 1);
    EXPECT_EQ(wrapped->called_computations()[0]->name(),
              "compute_on.2.2.clone");
  }
}

TEST_F(AsyncCollectiveCustomCallRewriterTest,
       ComputeOnWithUnsupportedComputationIsUnimplemented) {
  // The compute-on computation contains an instruction that is neither an
  // all-gather nor a parameter.
  constexpr absl::string_view unsupported_instruction_hlo = R"(
    HloModule test

    %compute_on_computation (new_param: f32[256,256]) -> f32[256,256] {
      %new_param = f32[256,256]{1,0} parameter(0)
      ROOT %negate = f32[256,256]{1,0} negate(%new_param)
    }

    ENTRY %main (param: f32[256,256]) -> f32[256,256] {
      %param = f32[256,256]{1,0} parameter(0)
      %compute_on = f32[256,256]{1,0} custom-call(%param),
          custom_call_target="compute-on-start",
          called_computations={%compute_on_computation}
      ROOT %compute_on_done = f32[256,256]{1,0} custom-call(%compute_on),
          custom_call_target="compute-on-done"
    }
  )";

  // The compute-on computation only contains supported instructions, but no
  // all-gather to derive the rewrite from.
  constexpr absl::string_view no_all_gather_hlo = R"(
    HloModule test

    %compute_on_computation (new_param: f32[256,256]) -> f32[256,256] {
      ROOT %new_param = f32[256,256]{1,0} parameter(0)
    }

    ENTRY %main (param: f32[256,256]) -> f32[256,256] {
      %param = f32[256,256]{1,0} parameter(0)
      %compute_on = f32[256,256]{1,0} custom-call(%param),
          custom_call_target="compute-on-start",
          called_computations={%compute_on_computation}
      ROOT %compute_on_done = f32[256,256]{1,0} custom-call(%compute_on),
          custom_call_target="compute-on-done"
    }
  )";

  // The compute-on computation contains an all-gather with more than one
  // operand, which the rewrite does not know how to handle.
  constexpr absl::string_view variadic_all_gather_hlo = R"(
    HloModule test

    %compute_on_computation (p0: f32[256,256], p1: f32[256,256]) -> (f32[256,2048], f32[256,2048]) {
      %p0 = f32[256,256]{1,0} parameter(0)
      %p1 = f32[256,256]{1,0} parameter(1)
      ROOT %all_gather = (f32[256,2048]{1,0}, f32[256,2048]{1,0}) all-gather(%p0, %p1), channel_id=1, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={1}, use_global_device_ids=true
    }

    ENTRY %main (param: f32[256,256]) -> (f32[256,2048], f32[256,2048]) {
      %param = f32[256,256]{1,0} parameter(0)
      %compute_on = (f32[256,2048]{1,0}, f32[256,2048]{1,0}) custom-call(%param, %param),
          custom_call_target="compute-on-start",
          called_computations={%compute_on_computation}
      ROOT %compute_on_done = (f32[256,2048]{1,0}, f32[256,2048]{1,0}) custom-call(%compute_on),
          custom_call_target="compute-on-done"
    }
  )";

  // Both modes only wrap computations made up of all-gather and parameter
  // instructions, so the negate is rejected in either mode.
  for (bool use_legacy_collectives : {true, false}) {
    SCOPED_TRACE(use_legacy_collectives ? "use_legacy_collectives=true"
                                        : "use_legacy_collectives=false");
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<HloModule> module,
        ParseAndReturnUnverifiedModule(unsupported_instruction_hlo));
    AsyncCollectiveCustomCallRewriter rewriter(use_legacy_collectives);
    EXPECT_THAT(rewriter.Run(module.get()),
                ::testing::status::StatusIs(
                    absl::StatusCode::kUnimplemented,
                    ::testing::HasSubstr("contains a negate instruction")));
  }

  // Both modes require an all-gather to be present in the compute-on
  // computation.
  for (bool use_legacy_collectives : {true, false}) {
    SCOPED_TRACE(use_legacy_collectives ? "use_legacy_collectives=true"
                                        : "use_legacy_collectives=false");
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnUnverifiedModule(no_all_gather_hlo));
    AsyncCollectiveCustomCallRewriter rewriter(use_legacy_collectives);
    EXPECT_THAT(
        rewriter.Run(module.get()),
        ::testing::status::StatusIs(absl::StatusCode::kUnimplemented,
                                    ::testing::HasSubstr("none was found")));
  }

  // Both modes require the all-gather to have exactly one operand.
  for (bool use_legacy_collectives : {true, false}) {
    SCOPED_TRACE(use_legacy_collectives ? "use_legacy_collectives=true"
                                        : "use_legacy_collectives=false");
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<HloModule> module,
        ParseAndReturnUnverifiedModule(variadic_all_gather_hlo));
    AsyncCollectiveCustomCallRewriter rewriter(use_legacy_collectives);
    EXPECT_THAT(
        rewriter.Run(module.get()),
        ::testing::status::StatusIs(absl::StatusCode::kUnimplemented,
                                    ::testing::HasSubstr("has 2 operands")));
  }
}

}  // namespace
}  // namespace xla
