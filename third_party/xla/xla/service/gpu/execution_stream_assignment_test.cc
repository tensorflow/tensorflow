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
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;

using AsyncExecutionStreamIds =
    ::xla::gpu::ExecutionStreamAssignment::AsyncExecutionStreamIds;

namespace xla::gpu {
namespace {

class ExecutionStreamAssignmentTest : public HloTestBase {
 protected:
  // Adds expectations for the `ExecutionStreamId` for all synchronous
  // `HloInstructions` in the given `HloComputation`.
  void ExpectExecutionStreamForSyncInstructions(
      const ExecutionStreamAssignment& assignment, HloComputation* computation,
      ExecutionStreamId stream) const {
    for (const HloInstruction* instruction : computation->instructions()) {
      if (instruction->IsAsynchronous()) continue;
      EXPECT_THAT(assignment.GetSyncExecutionStreamId(instruction),
                  IsOkAndHolds(stream));
    }
  }
};

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
      ExecutionStreamAssignmentOptions{/*number_of_execution_streams=*/2});

  // The outermost computation should run on `ExecutionStreamId(0)`. The two
  // asynchronous branches should be launched on `ExecutionStreamId(1)` and
  // `ExecutionStreamId(2)`, respectively. The third asynchronous branch should
  // reuse `ExecutionStreamId(1)` because we set `number_of_execution_streams`
  // to `2`.
  ExpectExecutionStreamForSyncInstructions(
      assignment, FindComputation(module.get(), "entry"), ExecutionStreamId(0));
  for (absl::string_view instruction : {"start1", "update1", "done1"}) {
    EXPECT_THAT(assignment.GetAsyncExecutionStreamIds(Cast<HloAsyncInstruction>(
                    FindInstruction(module.get(), instruction))),
                IsOkAndHolds(AsyncExecutionStreamIds{
                    /*source_stream_id=*/ExecutionStreamId(0),
                    /*destination_stream_id=*/ExecutionStreamId(1)}));
  }
  for (absl::string_view instruction : {"start2", "update2", "done2"}) {
    EXPECT_THAT(assignment.GetAsyncExecutionStreamIds(Cast<HloAsyncInstruction>(
                    FindInstruction(module.get(), instruction))),
                IsOkAndHolds(AsyncExecutionStreamIds{
                    /*source_stream_id=*/ExecutionStreamId(0),
                    /*destination_stream_id=*/ExecutionStreamId(2)}));
  }
  for (absl::string_view instruction : {"start3", "update3", "done3"}) {
    EXPECT_THAT(assignment.GetAsyncExecutionStreamIds(Cast<HloAsyncInstruction>(
                    FindInstruction(module.get(), instruction))),
                IsOkAndHolds(AsyncExecutionStreamIds{
                    /*source_stream_id=*/ExecutionStreamId(0),
                    /*destination_stream_id=*/ExecutionStreamId(1)}));
  }

  // Leaf computations should run on the respective asynchronous
  // `ExecutionStreamIds`.
  ExpectExecutionStreamForSyncInstructions(
      assignment,
      Cast<HloAsyncInstruction>(FindInstruction(module.get(), "start1"))
          ->async_wrapped_computation(),
      ExecutionStreamId(1));
  ExpectExecutionStreamForSyncInstructions(
      assignment,
      Cast<HloAsyncInstruction>(FindInstruction(module.get(), "start2"))
          ->async_wrapped_computation(),
      ExecutionStreamId(2));
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

  for (absl::string_view instruction : {"copy-start"}) {
    EXPECT_THAT(
        assignment.GetAsyncExecutionStreamIds(Cast<HloCopyStartInstruction>(
            FindInstruction(module.get(), instruction))),
        IsOkAndHolds(AsyncExecutionStreamIds{
            /*source_stream_id=*/ExecutionStreamId(0),
            /*destination_stream_id=*/ExecutionStreamId(1)}));
  }
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

    // Entry computation that calls each of the leaves asynchronously.
    ENTRY entry {
      p0 = f32[4] parameter(0)
      ROOT done = f32[] fusion(p0), kind=kLoop, calls=fusion
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  ExecutionStreamAssignment assignment(module.get());

  // The outermost computation should run on `ExecutionStreamId(0)`.
  ExpectExecutionStreamForSyncInstructions(
      assignment, FindComputation(module.get(), "entry"), ExecutionStreamId(0));

  // Computations only reachable through fusion nodes should have no assigned
  // `ExecutionStreamId`.
  for (absl::string_view computation : {"reduce", "fusion"}) {
    for (const HloInstruction* instruction :
         FindComputation(module.get(), computation)->instructions()) {
      EXPECT_THAT(assignment.GetSyncExecutionStreamId(instruction),
                  StatusIs(absl::StatusCode::kNotFound));
    }
  }
}

TEST_F(ExecutionStreamAssignmentTest, UnreachableComputation) {
  // We'll create an `HloModule` with two computations: the ENTRY computation,
  // and another unreachable embedded computation.
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
  ExpectExecutionStreamForSyncInstructions(
      assignment, FindComputation(module.get(), "entry"), ExecutionStreamId(0));

  // Unreachable instructions should have no assigned `ExecutionStreamId`.
  for (const HloInstruction* instruction :
       FindComputation(module.get(), "unreachable")->instructions()) {
    EXPECT_THAT(assignment.GetSyncExecutionStreamId(instruction),
                StatusIs(absl::StatusCode::kNotFound));
  }
}

}  // namespace
}  // namespace xla::gpu
