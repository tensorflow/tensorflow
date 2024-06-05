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
#include "xla/shape.h"
#include "xla/shape_util.h"
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

  // Adds expectations for the `ExecutionStreamId` for all asynchronous
  // `HloInstructions` in the given `HloComputation`.
  void ExpectExecutionStreamForAsyncInstructions(
      const ExecutionStreamAssignment& assignment, HloComputation* computation,
      ExecutionStreamId source_stream,
      ExecutionStreamId destination_stream) const {
    for (const HloInstruction* instruction : computation->instructions()) {
      if (!instruction->IsAsynchronous()) continue;
      AsyncExecutionStreamIds expected_stream_ids;
      expected_stream_ids.source_stream_id = source_stream;
      expected_stream_ids.destination_stream_id = destination_stream;
      EXPECT_THAT(assignment.GetAsyncExecutionStreamIds(
                      Cast<HloAsyncInstruction>(instruction)),
                  IsOkAndHolds(expected_stream_ids));
    }
  }

  const Shape kTensorShape = ShapeUtil::MakeShape(F32, {2, 2});
};

TEST_F(ExecutionStreamAssignmentTest, AsyncFusion) {
  // We'll create an `HloModule` with two nested `async-fusions`.
  // ENTRY -> ASYNC-FUSION -> ASYNC-FUSION -> BINARY_OP
  const char* kModuleStr = R"(
    HloModule m

    // Leaf computation.
    leaf {
      p0 = f32[2,2] parameter(0)
      ROOT add = f32[2,2] add(p0, p0)
    }

    // Innermost `async-fusion`.
    fusion {
      p0 = f32[2,2] parameter(0)
      start = ((f32[2,2]), f32[2,2], s32[]) fusion-start(p0),
          kind=kLoop, calls=leaf
      update = ((f32[2,2]), f32[2,2], s32[]) fusion-update(start)
      ROOT done = f32[2,2] fusion-done(update)
    }

    // Outermost `async-fusion` and entrypoint for the module.
    ENTRY entry {
      p0 = f32[2,2] parameter(0)
      start = ((f32[2,2]), f32[2,2], s32[]) fusion-start(p0),
          kind=kLoop, calls=fusion
      update = ((f32[2,2]), f32[2,2], s32[]) fusion-update(start)
      ROOT done = f32[2,2] fusion-done(update)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  ExecutionStreamAssignment assignment(module.get());

  // The outermost computation should run on `ExecutionStreamId(0)` and launch
  // asynchronous work on `ExecutionStreamId(1)`.
  ExpectExecutionStreamForSyncInstructions(
      assignment, FindComputation(module.get(), "entry"), ExecutionStreamId(0));
  ExpectExecutionStreamForAsyncInstructions(
      assignment, FindComputation(module.get(), "entry"), ExecutionStreamId(0),
      ExecutionStreamId(1));

  // The nested computation should run on `ExecutionStreamId(1)` and launch
  // asynchronous work on `ExecutionStreamId(2)`.
  ExpectExecutionStreamForSyncInstructions(
      assignment, FindComputation(module.get(), "fusion"),
      ExecutionStreamId(1));
  ExpectExecutionStreamForAsyncInstructions(
      assignment, FindComputation(module.get(), "fusion"), ExecutionStreamId(1),
      ExecutionStreamId(2));

  // The innermost computation should run on `ExecutionStreamId(2)`
  ExpectExecutionStreamForSyncInstructions(
      assignment, FindComputation(module.get(), "leaf"), ExecutionStreamId(2));
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
