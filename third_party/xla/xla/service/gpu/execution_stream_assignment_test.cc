/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/execution_stream_assignment.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/execution_stream_id.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

class ExecutionStreamAssignmentTest : public HloHardwareIndependentTestBase {};

TEST_F(ExecutionStreamAssignmentTest, AsyncFusion) {
  const char* kModuleStr = R"(
    HloModule m

    // Leaf computations.
    leaf1 {
      p0 = f32[2,2] parameter(0)
      ROOT add = f32[2,2] add(p0, p0)
    }
    leaf2 {
      p0 = f32[2,2] parameter(0)
      ROOT add = f32[2,2] add(p0, p0)
    }
    leaf3 {
      p0 = f32[2,2] parameter(0)
      ROOT add = f32[2,2] add(p0, p0)
    }

    // Entry computation that calls each of the leaves asynchronously.
    ENTRY entry {
      p0 = f32[2,2] parameter(0)
      start1 = ((f32[2,2]), f32[2,2], s32[]) fusion-start(p0),
          kind=kLoop, calls=leaf1
      start2 = ((f32[2,2]), f32[2,2], s32[]) fusion-start(p0),
          kind=kLoop, calls=leaf2
      start3 = ((f32[2,2]), f32[2,2], s32[]) fusion-start(p0),
          kind=kLoop, calls=leaf3
      update1 = ((f32[2,2]), f32[2,2], s32[]) fusion-update(start1)
      update2 = ((f32[2,2]), f32[2,2], s32[]) fusion-update(start2)
      update3 = ((f32[2,2]), f32[2,2], s32[]) fusion-update(start3)
      done1 = f32[2,2] fusion-done(update1)
      done2 = f32[2,2] fusion-done(update2)
      done3 = f32[2,2] fusion-done(update3)
      ROOT done = f32[2,2] custom-call(done1, done2, done3),
          custom_call_target="target"
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  ExecutionStreamAssignment assignment(
      module.get(),
      ExecutionStreamAssignment::Options{/*number_of_execution_streams=*/2});

  // Only start operations get execution stream IDs. With 2 compute streams,
  // start1 gets ComputationStreamId(0), start2 gets ComputationStreamId(1),
  // start3 wraps around to ComputationStreamId(0).
  EXPECT_THAT(
      assignment.GetExecutionStreamId(FindInstruction(module.get(), "start1")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(ComputationStreamId(0))));
  EXPECT_THAT(
      assignment.GetExecutionStreamId(FindInstruction(module.get(), "start2")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(ComputationStreamId(1))));
  EXPECT_THAT(
      assignment.GetExecutionStreamId(FindInstruction(module.get(), "start3")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(ComputationStreamId(0))));

  // Update and done operations do not get assigned stream IDs (they inherit
  // from the parent scope at run time via structured concurrency).
  for (absl::string_view instruction :
       {"update1", "update2", "update3", "done1", "done2", "done3"}) {
    EXPECT_THAT(assignment.GetExecutionStreamId(
                    FindInstruction(module.get(), instruction)),
                absl_testing::StatusIs(absl::StatusCode::kNotFound));
  }
}

TEST_F(ExecutionStreamAssignmentTest, CopyStartStreamIdTest) {
  const char* const hlo_copy_start_string = R"(
  HloModule Module

  ENTRY CopyStartAndCopyDone {
    p0 = f32[2,3]{1,0:S(1)} parameter(0)
    copy-start = (f32[2,3]{1,0:S(2)}, f32[2,3]{1,0:S(1)}, u32[]) copy-start(p0)
    ROOT copy-done = f32[2,3]{1,0:S(2)} copy-done(copy-start)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_copy_start_string));

  ExecutionStreamAssignment assignment(module.get());

  // copy-start is a compute scope start, gets ComputationStreamId(0).
  EXPECT_THAT(
      assignment.GetExecutionStreamId(
          FindInstruction(module.get(), "copy-start")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(ComputationStreamId(0))));
}

TEST_F(ExecutionStreamAssignmentTest, FusionComputations) {
  const char* kModuleStr = R"(
    HloModule m

    reduce {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }
    fusion {
      p0 = f32[4] parameter(0)
      c0 = f32[] constant(0)
      ROOT reduce = f32[] reduce(p0, c0), dimensions={0}, to_apply=reduce
    }

    // Entry computation with a synchronous fusion.
    ENTRY entry {
      p0 = f32[4] parameter(0)
      ROOT done = f32[] fusion(p0), kind=kLoop, calls=fusion
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  ExecutionStreamAssignment assignment(module.get());

  // Synchronous instructions (including fusions) do not get execution stream
  // IDs. They run on the default stream or inherit from a parent scope.
  for (const HloInstruction* instruction :
       FindComputation(module.get(), "entry")->instructions()) {
    EXPECT_THAT(assignment.GetExecutionStreamId(instruction),
                absl_testing::StatusIs(absl::StatusCode::kNotFound));
  }
}

TEST_F(ExecutionStreamAssignmentTest, UnreachableComputation) {
  const char* kModuleStr = R"(
    HloModule m

    // Unreachable computation.
    unreachable {
      p0 = f32[2,2] parameter(0)
      ROOT add = f32[2,2] add(p0, p0)
    }

    ENTRY entry {
      p0 = f32[2,2] parameter(0)
      ROOT add = f32[2,2] add(p0, p0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  ExecutionStreamAssignment assignment(module.get());

  // No scope-start operations exist, so nothing gets assigned.
  for (const HloInstruction* instruction :
       FindComputation(module.get(), "entry")->instructions()) {
    EXPECT_THAT(assignment.GetExecutionStreamId(instruction),
                absl_testing::StatusIs(absl::StatusCode::kNotFound));
  }
}

TEST_F(ExecutionStreamAssignmentTest, ExplicitStreams) {
  const char* kModuleStr = R"(
  HloModule m
  %gemm1 (x: f32[2048,2048], y: f32[2048,2048]) -> f32[2048,2048] {
    %y = f32[2048,2048]{1,0} parameter(1)
    %x = f32[2048,2048]{1,0} parameter(0)
    %custom-call.1 = (f32[2048,2048]{1,0}, s8[33554432]{0}) custom-call(f32[2048,2048]{1,0} %x, f32[2048,2048]{1,0} %y), custom_call_target="__cublas$gemm", backend_config={"gemm_backend_config":{"alpha_real":1,"alpha_imag":0,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"precision_config":{"operand_precision":["DEFAULT","DEFAULT"],"algorithm":"ALG_UNSET"},"epilogue":"DEFAULT","damax_output":false,"lhs_stride":"4194304","rhs_stride":"4194304","grad_x":false,"grad_y":false}}
    ROOT %get-tuple-element = f32[2048,2048]{1,0} get-tuple-element((f32[2048,2048]{1,0}, s8[33554432]{0}) %custom-call.1), index=0
  }

  %gemm2 (x: f32[2048,2048], y: f32[2048,2048]) -> f32[2048,2048] {
    %y = f32[2048,2048]{1,0} parameter(1)
    %x = f32[2048,2048]{1,0} parameter(0)
    %custom-call.2 = (f32[2048,2048]{1,0}, s8[33554432]{0}) custom-call(f32[2048,2048]{1,0} %x, f32[2048,2048]{1,0} %y), custom_call_target="__cublas$gemm", backend_config={"gemm_backend_config":{"alpha_real":1,"alpha_imag":0,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"precision_config":{"operand_precision":["DEFAULT","DEFAULT"],"algorithm":"ALG_UNSET"},"epilogue":"DEFAULT","damax_output":false,"lhs_stride":"4194304","rhs_stride":"4194304","grad_x":false,"grad_y":false}}
    ROOT %get-tuple-element = f32[2048,2048]{1,0} get-tuple-element((f32[2048,2048]{1,0}, s8[33554432]{0}) %custom-call.2), index=0
  }

  ENTRY %entry (y: f32[2048,2048], x: f32[2048,2048]) -> f32[2048,2048] {
    %x = f32[2048,2048]{1,0} parameter(1), metadata={op_name="b" scheduling_name="x"}
    %y = f32[2048,2048]{1,0} parameter(0), metadata={op_name="a" scheduling_name="y"}
    %call-start = ((f32[2048,2048]{1,0}, f32[2048,2048]{1,0}), f32[2048,2048]{1,0}) call-start(f32[2048,2048]{1,0} %y, f32[2048,2048]{1,0} %x), to_apply=%gemm1, frontend_attributes={_xla_stream_annotation="1"}
    %call-done = f32[2048,2048]{1,0} call-done(((f32[2048,2048]{1,0}, f32[2048,2048]{1,0}), f32[2048,2048]{1,0}) %call-start)
    %call-start.2 = ((f32[2048,2048]{1,0}, f32[2048,2048]{1,0}), f32[2048,2048]{1,0}) call-start(f32[2048,2048]{1,0} %y, f32[2048,2048]{1,0} %call-done), to_apply=%gemm2, frontend_attributes={_xla_stream_annotation="2"}
    ROOT %call-done-2 = f32[2048,2048]{1,0} call-done(((f32[2048,2048]{1,0}, f32[2048,2048]{1,0}), f32[2048,2048]{1,0}) %call-start.2)
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  ExecutionStreamAssignment assignment(
      module.get(),
      ExecutionStreamAssignment::Options{/*number_of_execution_streams=*/4});

  // call-start has explicit annotation "1" -> ComputationStreamId(1).
  EXPECT_THAT(
      assignment.GetExecutionStreamId(
          FindInstruction(module.get(), "call-start")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(ComputationStreamId(1))));

  // call-start.2 has explicit annotation "2" -> ComputationStreamId(2).
  EXPECT_THAT(
      assignment.GetExecutionStreamId(
          FindInstruction(module.get(), "call-start.2")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(ComputationStreamId(2))));

  // Done operations don't get stream IDs.
  EXPECT_THAT(assignment.GetExecutionStreamId(
                  FindInstruction(module.get(), "call-done")),
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(ExecutionStreamAssignmentTest, AsyncCollectiveTest) {
  const char* const hlo_string = R"(
  HloModule m, is_scheduled=true
    reduce {
      x = f32[] parameter(0)
      y = f32[] parameter(1)
      ROOT _ = f32[] add(x, y)
    }
    ENTRY main {
      p0 = f32[] parameter(0)
      p1 = f32[2] parameter(1)
      p2 = f32[2] parameter(2)
      ar-start = f32[] all-reduce-start(p0), to_apply=reduce
      rs-start = ((f32[2]), f32[1]) reduce-scatter-start(p1), to_apply=reduce, dimensions={0}
      add.0 = f32[2] add(p1, p2)
      ar-done = f32[] all-reduce-done(ar-start)
      rs-done = f32[1] reduce-scatter-done(rs-start)
      ROOT _ = (f32[], f32[1], f32[2]) tuple(ar-done, rs-done, add.0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  // With 4 compute streams and 2 collective streams, ar-start and rs-start
  // get CommunicationStreamId(0) and CommunicationStreamId(1).
  ExecutionStreamAssignment assignment(
      module.get(), {/*number_of_compute_execution_streams=*/4,
                     /*number_of_communication_execution_streams=*/2});
  EXPECT_THAT(
      assignment.GetExecutionStreamId(
          FindInstruction(module.get(), "ar-start")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(CommunicationStreamId(0))));
  EXPECT_THAT(
      assignment.GetExecutionStreamId(
          FindInstruction(module.get(), "rs-start")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(CommunicationStreamId(1))));

  // Redo with 1 collective stream: both should get CommunicationStreamId(0).
  assignment = ExecutionStreamAssignment(
      module.get(), {/*number_of_compute_execution_streams=*/4,
                     /*number_of_communication_execution_streams=*/1});
  EXPECT_THAT(
      assignment.GetExecutionStreamId(
          FindInstruction(module.get(), "ar-start")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(CommunicationStreamId(0))));
  EXPECT_THAT(
      assignment.GetExecutionStreamId(
          FindInstruction(module.get(), "rs-start")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(CommunicationStreamId(0))));

  // Done operations don't get stream IDs.
  EXPECT_THAT(
      assignment.GetExecutionStreamId(FindInstruction(module.get(), "ar-done")),
      absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(ExecutionStreamAssignmentTest, PipelinedSendRecv) {
  const char* kModuleStr = R"(
    HloModule test, num_partitions=2

    cond {
      param = (u32[], (f32[2,2], u32[], token[]), (f32[2,2], u32[], token[]))
          parameter(0)
      i = u32[] get-tuple-element(param), index=0
      c2 = u32[] constant(2)
      ROOT cmp = pred[] compare(i, c2), direction=LT
    }

    body {
      param = (u32[], (f32[2,2], u32[], token[]), (f32[2,2], u32[], token[]))
          parameter(0)
      i = u32[] get-tuple-element(param), index=0
      send_ctx = get-tuple-element(param), index=1
      recv_ctx = get-tuple-element(param), index=2
      send_done_inner = token[] send-done(send_ctx), channel_id=1
      recv_done_inner = (f32[2,2], token[]) recv-done(recv_ctx), channel_id=2
      data = get-tuple-element(recv_done_inner), index=0
      after_all = token[] after-all()
      send_ctx_inner = (f32[2,2], u32[], token[]) send(data, after_all),
          frontend_attributes={_xla_send_recv_source_target_pairs={{0,1}}},
          channel_id=1
      recv_ctx_inner = (f32[2,2], u32[], token[]) recv(after_all),
          frontend_attributes={_xla_send_recv_source_target_pairs={{0,1}}},
          channel_id=2
      c1 = u32[] constant(1)
      i_ = u32[] add(i, c1)
      ROOT result = (u32[], (f32[2,2], u32[], token[]),
          (f32[2,2], u32[], token[])) tuple(i_, send_ctx_inner, recv_ctx_inner)
    }

    ENTRY test_computation {
      data = f32[2,2] parameter(0)
      i = u32[] constant(0)
      after_all = token[] after-all()
      send_ctx_ = (f32[2,2], u32[], token[]) send(data, after_all),
          frontend_attributes={_xla_send_recv_source_target_pairs={{0,1}}},
          channel_id=1
      recv_ctx_ = (f32[2,2], u32[], token[]) recv(after_all),
          frontend_attributes={_xla_send_recv_source_target_pairs={{0,1}}},
          channel_id=2
      init = (u32[], (f32[2,2], u32[], token[]), (f32[2,2], u32[], token[]))
          tuple(i, send_ctx_, recv_ctx_)
      while = (u32[], (f32[2,2], u32[], token[]), (f32[2,2], u32[], token[]))
          while(init), condition=cond, body=body
      send_ctx = get-tuple-element(while), index=1
      recv_ctx = get-tuple-element(while), index=2
      send_done = token[] send-done(send_ctx), channel_id=1
      recv_done = (f32[2,2], token[]) recv-done(recv_ctx), channel_id=2
      ROOT data_ = get-tuple-element(recv_done), index=0
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  ExecutionStreamAssignment assignment(
      module.get(), ExecutionStreamAssignment::Options{4, 4});

  // Only the canonical send/recv starts get stream IDs.
  EXPECT_THAT(
      assignment.GetExecutionStreamId(
          FindInstruction(module.get(), "send_ctx_")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(CommunicationStreamId(0))));
  EXPECT_THAT(
      assignment.GetExecutionStreamId(
          FindInstruction(module.get(), "recv_ctx_")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(CommunicationStreamId(1))));

  // Non-canonical send/recv and done operations don't get stream IDs.
  for (absl::string_view instruction :
       {"send_ctx_inner", "recv_ctx_inner", "send_done_inner",
        "recv_done_inner", "send_done", "recv_done"}) {
    EXPECT_THAT(assignment.GetExecutionStreamId(
                    FindInstruction(module.get(), instruction)),
                absl_testing::StatusIs(absl::StatusCode::kNotFound));
  }
}

}  // namespace
}  // namespace xla::gpu
