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

#include "xla/service/async_compute_on_rewriter.h"

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
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

using ::absl_testing::StatusIs;

class AsyncComputeOnRewriterTest : public HloHardwareIndependentTestBase {};

TEST_F(AsyncComputeOnRewriterTest, ComputeOnStart) {
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
    AsyncComputeOnRewriter rewriter(/*use_legacy_collectives=*/true);
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
    AsyncComputeOnRewriter rewriter(
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

TEST_F(AsyncComputeOnRewriterTest, ComputeOnStartToSimplify) {
  constexpr absl::string_view hlo_string = R"(
HloModule test
  %compute_on.2.2.clone {
    ROOT %new_param = f32[256,256]{1,0} parameter(0)
  }

ENTRY %main.0_spmd {
  %param = f32[256,256]{1,0} parameter(2), sharding={devices=[1,8]<=[8]}, metadata={op_name="w2"}
  %compute_on.2 = f32[256,256]{1,0} custom-call(%param), custom_call_target="compute-on-start", called_computations={%compute_on.2.2.clone}, frontend_attributes={backend_config={"sparse_core_config": {"core_ids": [0], "core_id_mutability": false}}}, metadata={op_name="jit(f)/compute_on" stack_frame_id=13}
  %param.1 = f32[128,2048]{1,0} parameter(0), sharding={devices=[8,1]<=[8]}, metadata={op_name="x"}
  %param.2 = f32[256,256]{1,0} parameter(1), sharding={devices=[8,1]<=[8]}, metadata={op_name="w1"}
  %tuple.2 = (f32[256,256]{1,0}, f32[128,2048]{1,0}, f32[256,256]{1,0}) tuple(%compute_on.2, %param.1, %param.2), metadata={op_name="jit(f)/optimization_barrier" stack_frame_id=15}
  %optimization_barrier.8 = (f32[256,256]{1,0}, f32[128,2048]{1,0}, f32[256,256]{1,0}) opt-barrier(%tuple.2), metadata={op_name="jit(f)/optimization_barrier" stack_frame_id=15}
  %all-gather.2 = f32[2048,256]{1,0} all-gather(%optimization_barrier.8#2), channel_id=4, replica_groups=mesh['axis_0'=8] {'axis_0'}, dimensions={0}, use_global_device_ids=true, frontend_attributes={is_spmd_generated="true"}, metadata={op_name="jit(f)/optimization_barrier" stack_frame_id=15}
  %dot.1 = f32[128,256]{1,0} dot(%optimization_barrier.8#1, %all-gather.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="jit(f)/dot_general" stack_frame_id=15}
  %tuple.4 = (f32[128,256]{1,0}, f32[256,256]{1,0}) tuple(%dot.1, %optimization_barrier.8#0), metadata={op_name="jit(f)/optimization_barrier" stack_frame_id=16}
  %optimization_barrier.9 = (f32[128,256]{1,0}, f32[256,256]{1,0}) opt-barrier(%tuple.4), metadata={op_name="jit(f)/optimization_barrier" stack_frame_id=16}
  %compute_on_done.2 = f32[256,256]{1,0} custom-call(%optimization_barrier.9#1), custom_call_target="compute-on-done", metadata={op_name="jit(f)/compute_on_done" stack_frame_id=16}
  %tuple.6 = (f32[128,256]{1,0}, f32[256,256]{1,0}) tuple(%optimization_barrier.9#0, %compute_on_done.2), metadata={op_name="jit(f)/optimization_barrier" stack_frame_id=17}
  %optimization_barrier.10 = (f32[128,256]{1,0}, f32[256,2048]{1,0}) opt-barrier(%tuple.6), metadata={op_name="jit(f)/optimization_barrier" stack_frame_id=17}
  ROOT %dot.3 = f32[128,256]{1,0} dot(%optimization_barrier.10#0, %optimization_barrier.10#1), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="jit(f)/dot_general" stack_frame_id=17}
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(hlo_string));
  AsyncComputeOnRewriter rewriter(/*use_legacy_collectives=*/true);
  ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_TRUE(changed);
  auto instructions = module->entry_computation()->instructions();
  ASSERT_TRUE(absl::c_none_of(instructions, [](const HloInstruction* instr) {
    return instr->opcode() == HloOpcode::kCustomCall;
  }));
}

TEST_F(AsyncComputeOnRewriterTest,
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

  constexpr absl::string_view two_all_gathers_hlo = R"(
    HloModule test

    %compute_on_computation (p0: f32[256,256]) -> f32[256,16384] {
      %p0 = f32[256,256]{1,0} parameter(0)
      %all_gather.0 = f32[256,2048]{1,0} all-gather(%p0), channel_id=1, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={1}, use_global_device_ids=true
      ROOT %all_gather.1 = f32[256,16384]{1,0} all-gather(%all_gather.0), channel_id=2, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={1}, use_global_device_ids=true
    }

    ENTRY %main (param: f32[256,256]) -> f32[256,16384] {
      %param = f32[256,256]{1,0} parameter(0)
      %compute_on = f32[256,16384]{1,0} custom-call(%param),
          custom_call_target="compute-on-start",
          called_computations={%compute_on_computation}
      ROOT %compute_on_done = f32[256,16384]{1,0} custom-call(%compute_on),
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
    AsyncComputeOnRewriter rewriter(use_legacy_collectives);
    EXPECT_THAT(
        rewriter.Run(module.get()),
        StatusIs(absl::StatusCode::kUnimplemented,
                 ::testing::HasSubstr("contains a negate instruction")));
  }

  // Both modes require the all-gather to have exactly one operand.
  for (bool use_legacy_collectives : {true, false}) {
    SCOPED_TRACE(use_legacy_collectives ? "use_legacy_collectives=true"
                                        : "use_legacy_collectives=false");
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<HloModule> module,
        ParseAndReturnUnverifiedModule(variadic_all_gather_hlo));
    AsyncComputeOnRewriter rewriter(use_legacy_collectives);
    EXPECT_THAT(rewriter.Run(module.get()),
                StatusIs(absl::StatusCode::kUnimplemented,
                         ::testing::HasSubstr("has 2 operands")));
  }

  // Both modes require the compute-on computation to contain exactly one
  // all-gather.
  for (bool use_legacy_collectives : {true, false}) {
    SCOPED_TRACE(use_legacy_collectives ? "use_legacy_collectives=true"
                                        : "use_legacy_collectives=false");
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnUnverifiedModule(two_all_gathers_hlo));
    AsyncComputeOnRewriter rewriter(use_legacy_collectives);
    EXPECT_THAT(rewriter.Run(module.get()),
                StatusIs(absl::StatusCode::kUnimplemented,
                         ::testing::HasSubstr("contains at least two")));
  }
}

class RecordingComputeOnHelper : public AsyncComputeOnHelper {
 public:
  explicit RecordingComputeOnHelper(absl::Status status = absl::OkStatus())
      : status_(std::move(status)) {}

  absl::Status AddBackendSpecializations(HloInstruction* custom_call_start,
                                         HloInstruction* async_start) override {
    ++call_count_;
    has_frontend_attributes_ = custom_call_start->has_frontend_attributes();
    recorded_async_start_opcode_ = async_start->opcode();
    return status_;
  }

  int call_count() const { return call_count_; }
  bool has_frontend_attributes() const { return has_frontend_attributes_; }
  HloOpcode recorded_async_start_opcode() const {
    return recorded_async_start_opcode_;
  }

 private:
  int call_count_ = 0;
  bool has_frontend_attributes_ = false;
  HloOpcode recorded_async_start_opcode_ = HloOpcode::kParameter;
  absl::Status status_;
};

TEST_F(AsyncComputeOnRewriterTest, InvokesComputeOnHelperForAsyncRewriteOnly) {
  constexpr absl::string_view all_gather_hlo = R"(
    HloModule test
    %compute_on_comp (p: f32[256,256]) -> f32[256,2048] {
      %p = f32[256,256]{1,0} parameter(0)
      ROOT %ag = f32[256,2048]{1,0} all-gather(%p), channel_id=1, replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={1}, use_global_device_ids=true
    }
    ENTRY %main (param: f32[256,256]) -> f32[256,2048] {
      %param = f32[256,256]{1,0} parameter(0)
      %start = f32[256,2048]{1,0} custom-call(%param), custom_call_target="compute-on-start", called_computations={%compute_on_comp}, frontend_attributes={backend_config="test_cfg"}
      ROOT %done = f32[256,2048]{1,0} custom-call(%start), custom_call_target="compute-on-done"
    }
  )";

  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnUnverifiedModule(all_gather_hlo));
    RecordingComputeOnHelper helper;
    AsyncComputeOnRewriter rewriter(/*use_legacy_collectives=*/true, &helper);
    ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
    EXPECT_TRUE(changed);
    EXPECT_EQ(helper.call_count(), 1);
    EXPECT_TRUE(helper.has_frontend_attributes());
    EXPECT_EQ(helper.recorded_async_start_opcode(), HloOpcode::kAllGatherStart);
  }

  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnUnverifiedModule(all_gather_hlo));
    RecordingComputeOnHelper failing_helper(
        absl::InvalidArgumentError("bad backend config"));
    AsyncComputeOnRewriter rewriter(/*use_legacy_collectives=*/false,
                                    &failing_helper);
    EXPECT_THAT(rewriter.Run(module.get()),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         ::testing::HasSubstr("bad backend config")));
    EXPECT_EQ(failing_helper.call_count(), 1);
  }

  constexpr absl::string_view empty_compute_on_hlo = R"(
    HloModule test
    %empty_comp (p: f32[256,256]) -> f32[256,256] {
      ROOT %p = f32[256,256]{1,0} parameter(0)
    }
    ENTRY %main (param: f32[256,256]) -> f32[256,256] {
      %param = f32[256,256]{1,0} parameter(0)
      %start = f32[256,256]{1,0} custom-call(%param), custom_call_target="compute-on-start", called_computations={%empty_comp}
      ROOT %done = f32[256,256]{1,0} custom-call(%start), custom_call_target="compute-on-done"
    }
  )";

  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnUnverifiedModule(empty_compute_on_hlo));
    RecordingComputeOnHelper helper;
    AsyncComputeOnRewriter rewriter(/*use_legacy_collectives=*/true, &helper);
    ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
    EXPECT_TRUE(changed);
    EXPECT_EQ(helper.call_count(), 0);
  }
}

TEST_F(AsyncComputeOnRewriterTest, PreservesComputeOnFrontendAttributes) {
  constexpr absl::string_view compute_on_hlo = R"(
    HloModule PreservesCollectiveFrontendAttributesComputeOn
    gather {
      p0 = f32[16,10] parameter(0)
      ROOT ag = f32[32,10] all-gather(p0), replica_groups={{0,1}}, dimensions={0}
    }
    ENTRY main {
      p0 = f32[16,10] parameter(0)
      start = f32[32,10] custom-call(p0), custom_call_target="compute-on-start",
          called_computations={gather},
          frontend_attributes={_xla_stream_annotation="1",start_tag="gather"}
      ROOT done = f32[32,10] custom-call(start), custom_call_target="compute-on-done",
          frontend_attributes={_xla_stream_annotation="1",done_tag="wait"}
    }
  )";

  for (bool use_legacy_collectives : {true, false}) {
    SCOPED_TRACE(use_legacy_collectives ? "use_legacy_collectives=true"
                                        : "use_legacy_collectives=false");

    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnUnverifiedModule(compute_on_hlo));
    const HloInstruction* original_done =
        module->entry_computation()->root_instruction();
    const auto done_attrs = original_done->frontend_attributes();
    const auto start_attrs = original_done->operand(0)->frontend_attributes();

    // Run the rewriter.
    AsyncComputeOnRewriter rewriter(use_legacy_collectives);
    ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
    EXPECT_TRUE(changed);

    // Get the updated start/done instructions.
    const HloInstruction* done =
        module->entry_computation()->root_instruction();
    const HloInstruction* start = done->operand(0);

    // Updated start/done have the expected opcodes.
    EXPECT_EQ(start->opcode(), use_legacy_collectives
                                   ? HloOpcode::kAllGatherStart
                                   : HloOpcode::kAsyncStart);
    EXPECT_EQ(done->opcode(), use_legacy_collectives ? HloOpcode::kAllGatherDone
                                                     : HloOpcode::kAsyncDone);

    //  Frontend attributes are still present.
    EXPECT_THAT(start->frontend_attributes().map(),
                ::testing::UnorderedElementsAreArray(start_attrs.map()));
    EXPECT_THAT(done->frontend_attributes().map(),
                ::testing::UnorderedElementsAreArray(done_attrs.map()));
  }
}

TEST_F(AsyncComputeOnRewriterTest,
       SimplifiedComputeOnDoesNotPropagateAnnotations) {
  constexpr absl::string_view compute_on_hlo = R"(
    HloModule SimplifiedComputeOnDoesNotPropagateAnnotations
    simplified {
      ROOT p0 = f32[16,10] parameter(0)
    }
    ENTRY main {
      p0 = f32[16,10] parameter(0),
          frontend_attributes={input_tag="original"},
          metadata={op_name="input" source_file="input.py" source_line=10}
      start = f32[16,10] custom-call(p0), custom_call_target="compute-on-start",
          called_computations={simplified},
          frontend_attributes={_xla_stream_annotation="1",start_tag="identity"},
          metadata={op_name="compute_on_start" source_file="compute_on.py" source_line=20}
      ROOT done = f32[16,10] custom-call(start), custom_call_target="compute-on-done",
          frontend_attributes={_xla_stream_annotation="1",done_tag="wait"},
          metadata={op_name="compute_on_done" source_file="compute_on.py" source_line=30}
    }
  )";

  for (bool use_legacy_collectives : {true, false}) {
    SCOPED_TRACE(use_legacy_collectives ? "use_legacy_collectives=true"
                                        : "use_legacy_collectives=false");

    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnUnverifiedModule(compute_on_hlo));
    const HloInstruction* original_input =
        module->entry_computation()->parameter_instruction(0);
    const auto input_attrs = original_input->frontend_attributes();
    const auto input_metadata = original_input->metadata();

    // Run the rewriter.
    AsyncComputeOnRewriter rewriter(use_legacy_collectives);
    ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
    EXPECT_TRUE(changed);
    ASSERT_OK(HloVerifier(HloVerifierOpts{}).Run(module.get()).status());

    const HloInstruction* root =
        module->entry_computation()->root_instruction();
    ASSERT_EQ(root->opcode(), HloOpcode::kCall);
    ASSERT_EQ(root->operand_count(), 1);
    EXPECT_EQ(root->operand(0), original_input);

    // The existing input keeps its own annotations.
    EXPECT_THAT(original_input->frontend_attributes().map(),
                ::testing::UnorderedElementsAreArray(input_attrs.map()));
    EXPECT_EQ(original_input->metadata().SerializeAsString(),
              input_metadata.SerializeAsString());

    // The replacement identity call does not inherit start/done annotations.
    EXPECT_TRUE(root->frontend_attributes().map().empty());
    EXPECT_EQ(root->metadata().ByteSizeLong(), 0);
  }
}

}  // namespace
}  // namespace xla
