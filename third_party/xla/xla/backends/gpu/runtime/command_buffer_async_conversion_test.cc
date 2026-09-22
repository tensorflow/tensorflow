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

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/async_thunk.h"
#include "xla/backends/gpu/runtime/command_buffer_conversion_pass.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/execution_stream_id.h"
#include "xla/backends/gpu/runtime/memset_thunk.h"
#include "xla/backends/gpu/runtime/replica_id_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/backends/gpu/runtime/thunk_pass_pipeline.h"
#include "xla/backends/gpu/runtime/while_thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

using ::testing::ElementsAre;
using ::testing::Pointee;

MATCHER_P(ThunkKindIs, kind, "") { return arg.kind() == kind; }

template <typename... Kinds>
auto ThunkKindsAre(Kinds... kinds) {
  return ElementsAre(Pointee(ThunkKindIs(kinds))...);
}

class RejectingAllocator : public ThunkPassBufferAllocator {
 public:
  absl::StatusOr<BufferAllocation*> NewEmptyAllocation(int64_t size) override {
    return absl::InternalError("Unexpected allocation during conversion");
  }
};

class CommandBufferAsyncConversionTest : public testing::Test {
 protected:
  Thunk::ThunkInfo Info() {
    Thunk::ThunkInfo info;
    info.thunk_id = ids_.GetNextThunkId();
    return info;
  }

  void AddCommand(ThunkSequence& thunks) {
    thunks.Emplace<ReplicaIdThunk>(Info(), slice_);
  }

  AsyncStartThunk* Start(ThunkSequence& thunks, bool convertible = true,
                         ExecutionStreamId stream = ComputationStreamId(0)) {
    ThunkSequence body;
    if (convertible) {
      AddCommand(body);
    } else {
      // Memset thunks are not eligible for conversion in this pass.
      body.Emplace<Memset32BitValueThunk>(Info(), 0, slice_);
    }
    auto start =
        std::make_unique<AsyncStartThunk>(Info(), stream, std::move(body));
    auto* result = start.get();
    thunks.push_back(std::move(start));
    return result;
  }

  void Done(ThunkSequence& thunks, AsyncStartThunk* start) {
    thunks.Emplace<AsyncDoneThunk>(Info(), start->async_execution());
  }

  WhileThunk* Loop(ThunkSequence& thunks, ThunkSequence body) {
    auto loop = std::make_unique<WhileThunk>(
        Info(), BufferAllocation::Slice(&allocation_, 0, 1), ThunkSequence{},
        std::move(body), /*trip_count=*/1);
    auto* result = loop.get();
    thunks.push_back(std::move(loop));
    return result;
  }

  const ThunkSequence& CommandBufferThunks(const Thunk& thunk) {
    return static_cast<const CommandBufferThunk&>(thunk).thunks()->thunks();
  }

  absl::StatusOr<bool> Convert(ThunkSequence& thunks,
                               bool enable_while = false) {
    DebugOptions options;
    options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);
    if (enable_while) {
      options.add_xla_gpu_enable_command_buffer(DebugOptions::WHILE);
    }
    options.set_xla_gpu_graph_min_graph_size(1);
    options.set_xla_gpu_command_buffer_scheduling_mode(DebugOptions::LHS);
    se::DeviceDescription device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
    RejectingAllocator allocator;
    return CommandBufferConversionPass("test").Run(
        &thunks, options, /*hlo_module=*/nullptr, device, allocator);
  }

  BufferAllocation allocation_{0, sizeof(int32_t), 0};
  BufferAllocation::Slice slice_{&allocation_, 0, sizeof(int32_t)};
  ThunkIdGenerator ids_;
};

TEST_F(CommandBufferAsyncConversionTest, KeepsUnsupportedRegionsIntact) {
  for (bool crossed : {false, true}) {
    SCOPED_TRACE(crossed);
    ThunkSequence thunks;
    AddCommand(thunks);
    auto* outer = Start(thunks, /*convertible=*/false);
    auto* inner = Start(thunks);
    Done(thunks, crossed ? outer : inner);
    Done(thunks, crossed ? inner : outer);
    AddCommand(thunks);

    ASSERT_OK_AND_ASSIGN(bool changed, Convert(thunks));
    EXPECT_TRUE(changed);
    // Capturing the inner start/done alone would drop its stream ordering
    // against the unsupported outer operation. Capture resumes after both
    // joins, and ordinary commands before the region can still be captured.
    EXPECT_THAT(thunks,
                ThunkKindsAre(Thunk::kCommandBuffer, Thunk::kAsyncStart,
                              Thunk::kAsyncStart, Thunk::kAsyncDone,
                              Thunk::kAsyncDone, Thunk::kCommandBuffer));
  }
}

TEST_F(CommandBufferAsyncConversionTest, ConvertsPlainCommandsInOpenRegion) {
  ThunkSequence thunks;
  auto* start = Start(thunks, /*convertible=*/false);
  AddCommand(thunks);
  AddCommand(thunks);
  Done(thunks, start);
  AddCommand(thunks);

  ASSERT_OK_AND_ASSIGN(bool changed, Convert(thunks));
  EXPECT_TRUE(changed);
  // Commands between the unsupported start and its done execute on the main
  // stream, exactly as their thunks would, so they can still be captured.
  EXPECT_THAT(thunks, ThunkKindsAre(Thunk::kAsyncStart, Thunk::kCommandBuffer,
                                    Thunk::kAsyncDone, Thunk::kCommandBuffer));
  EXPECT_THAT(CommandBufferThunks(*thunks[1]),
              ThunkKindsAre(Thunk::kReplicaId, Thunk::kReplicaId));
}

TEST_F(CommandBufferAsyncConversionTest, KeepsNestedRegionsInOpenRegion) {
  ThunkSequence thunks;
  auto* outer = Start(thunks, /*convertible=*/false);
  auto* inner = Start(thunks);
  AddCommand(thunks);
  Done(thunks, inner);
  AddCommand(thunks);
  Done(thunks, outer);

  ASSERT_OK_AND_ASSIGN(bool changed, Convert(thunks));
  EXPECT_TRUE(changed);
  // Capturing the inner pair would run its body in a graph on the main stream
  // and lose stream order with the outstanding outer operation. Its start and
  // done stay thunks while the plain commands around them are captured.
  EXPECT_THAT(thunks, ThunkKindsAre(Thunk::kAsyncStart, Thunk::kAsyncStart,
                                    Thunk::kCommandBuffer, Thunk::kAsyncDone,
                                    Thunk::kCommandBuffer, Thunk::kAsyncDone));
}

TEST_F(CommandBufferAsyncConversionTest, ConvertsLoopBodiesInOpenRegion) {
  for (bool body_has_async : {false, true}) {
    SCOPED_TRACE(body_has_async);
    ThunkSequence thunks;
    auto* start = Start(thunks);
    ThunkSequence body;
    if (body_has_async) {
      auto* inner = Start(body);
      Done(body, inner);
    }
    AddCommand(body);
    // WHILE conversion is disabled, so the loop keeps the region open and its
    // body is converted recursively under the open-region rules.
    WhileThunk* loop = Loop(thunks, std::move(body));
    Done(thunks, start);
    AddCommand(thunks);

    ASSERT_OK_AND_ASSIGN(bool changed, Convert(thunks));
    EXPECT_TRUE(changed);
    EXPECT_THAT(thunks,
                ThunkKindsAre(Thunk::kAsyncStart, Thunk::kWhile,
                              Thunk::kAsyncDone, Thunk::kCommandBuffer));
    if (body_has_async) {
      EXPECT_THAT(loop->body_executor().thunks(),
                  ThunkKindsAre(Thunk::kAsyncStart, Thunk::kAsyncDone,
                                Thunk::kCommandBuffer));
    } else {
      EXPECT_THAT(loop->body_executor().thunks(),
                  ThunkKindsAre(Thunk::kCommandBuffer));
    }
  }
}

TEST_F(CommandBufferAsyncConversionTest,
       DoesNotCaptureLoopWithAsyncInOpenRegion) {
  for (bool open_region : {false, true}) {
    SCOPED_TRACE(open_region);
    ThunkSequence thunks;
    AsyncStartThunk* start = nullptr;
    if (open_region) {
      start = Start(thunks, /*convertible=*/false);
    }
    ThunkSequence body;
    auto* inner = Start(body);
    Done(body, inner);
    WhileThunk* loop = Loop(thunks, std::move(body));
    if (open_region) {
      Done(thunks, start);
    }

    ASSERT_OK_AND_ASSIGN(bool changed, Convert(thunks, /*enable_while=*/true));
    if (open_region) {
      // A convertible loop whose body starts async work on a stream with an
      // outstanding operation must not become a graph on the main stream.
      EXPECT_FALSE(changed);
      EXPECT_THAT(thunks, ThunkKindsAre(Thunk::kAsyncStart, Thunk::kWhile,
                                        Thunk::kAsyncDone));
      EXPECT_THAT(loop->body_executor().thunks(),
                  ThunkKindsAre(Thunk::kAsyncStart, Thunk::kAsyncDone));
    } else {
      EXPECT_TRUE(changed);
      EXPECT_THAT(thunks, ThunkKindsAre(Thunk::kCommandBuffer));
    }
  }
}

TEST_F(CommandBufferAsyncConversionTest, ConvertsTailAfterUnmatchedStart) {
  ThunkSequence thunks;
  AddCommand(thunks);
  Start(thunks);  // Its done can belong to a different pipelined computation.
  AddCommand(thunks);
  auto* inner = Start(thunks);
  Done(thunks, inner);
  AddCommand(thunks);

  ASSERT_OK_AND_ASSIGN(bool changed, Convert(thunks));
  EXPECT_TRUE(changed);
  // The stream stays open until the end of the sequence: later pairs on the
  // same stream remain thunks while plain commands are still captured.
  EXPECT_THAT(thunks, ThunkKindsAre(Thunk::kCommandBuffer, Thunk::kAsyncStart,
                                    Thunk::kCommandBuffer, Thunk::kAsyncStart,
                                    Thunk::kAsyncDone, Thunk::kCommandBuffer));
}

TEST_F(CommandBufferAsyncConversionTest,
       ConvertsTailAfterUnmatchedStartOnOtherStream) {
  ThunkSequence thunks;
  AddCommand(thunks);
  Start(thunks, /*convertible=*/true, ComputationStreamId(0));
  AddCommand(thunks);
  auto* inner = Start(thunks, /*convertible=*/true, ComputationStreamId(1));
  Done(thunks, inner);
  AddCommand(thunks);

  ASSERT_OK_AND_ASSIGN(bool changed, Convert(thunks));
  EXPECT_TRUE(changed);
  // The unmatched start only keeps its own stream open. A pair on another
  // stream never observed that stream in thunk mode either, so capturing it
  // loses no ordering.
  EXPECT_THAT(thunks, ThunkKindsAre(Thunk::kCommandBuffer, Thunk::kAsyncStart,
                                    Thunk::kCommandBuffer));
  EXPECT_THAT(CommandBufferThunks(*thunks[2]),
              ThunkKindsAre(Thunk::kReplicaId, Thunk::kAsyncStart,
                            Thunk::kAsyncDone, Thunk::kReplicaId));
}

TEST_F(CommandBufferAsyncConversionTest, ConvertsNestedRegionOnOtherStream) {
  ThunkSequence thunks;
  auto* outer = Start(thunks, /*convertible=*/false, ComputationStreamId(0));
  auto* inner = Start(thunks, /*convertible=*/true, ComputationStreamId(1));
  AddCommand(thunks);
  Done(thunks, inner);
  AddCommand(thunks);
  Done(thunks, outer);

  ASSERT_OK_AND_ASSIGN(bool changed, Convert(thunks));
  EXPECT_TRUE(changed);
  EXPECT_THAT(thunks, ThunkKindsAre(Thunk::kAsyncStart, Thunk::kCommandBuffer,
                                    Thunk::kAsyncDone));
  EXPECT_THAT(CommandBufferThunks(*thunks[1]),
              ThunkKindsAre(Thunk::kAsyncStart, Thunk::kReplicaId,
                            Thunk::kAsyncDone, Thunk::kReplicaId));
}

TEST_F(CommandBufferAsyncConversionTest, ClosesStreamAtMatchingDone) {
  ThunkSequence thunks;
  auto* a = Start(thunks, /*convertible=*/false, ComputationStreamId(0));
  auto* b = Start(thunks, /*convertible=*/false, ComputationStreamId(1));
  Done(thunks, a);
  auto* c = Start(thunks, /*convertible=*/true, ComputationStreamId(0));
  Done(thunks, c);
  Done(thunks, b);
  AddCommand(thunks);

  ASSERT_OK_AND_ASSIGN(bool changed, Convert(thunks));
  EXPECT_TRUE(changed);
  // Stream 0 is joined at `a`'s done, so `c` can be captured even though
  // stream 1 is still open.
  EXPECT_THAT(thunks, ThunkKindsAre(Thunk::kAsyncStart, Thunk::kAsyncStart,
                                    Thunk::kAsyncDone, Thunk::kCommandBuffer,
                                    Thunk::kAsyncDone, Thunk::kCommandBuffer));
  EXPECT_THAT(CommandBufferThunks(*thunks[3]),
              ThunkKindsAre(Thunk::kAsyncStart, Thunk::kAsyncDone));
}

TEST_F(CommandBufferAsyncConversionTest, KeepsRegionWithStartOnOpenStream) {
  ThunkSequence thunks;
  auto* a = Start(thunks, /*convertible=*/false, ComputationStreamId(0));
  auto* b = Start(thunks, /*convertible=*/true, ComputationStreamId(1));
  auto* c = Start(thunks, /*convertible=*/true, ComputationStreamId(0));
  Done(thunks, c);
  Done(thunks, b);
  Done(thunks, a);

  ASSERT_OK_AND_ASSIGN(bool changed, Convert(thunks));
  // `b`'s region is closed and convertible, but it contains `c`, which starts
  // work on the stream still occupied by `a`. Capturing the region would run
  // `c`'s body in a graph on the main stream.
  EXPECT_FALSE(changed);
  EXPECT_THAT(thunks, ThunkKindsAre(Thunk::kAsyncStart, Thunk::kAsyncStart,
                                    Thunk::kAsyncStart, Thunk::kAsyncDone,
                                    Thunk::kAsyncDone, Thunk::kAsyncDone));
}

TEST_F(CommandBufferAsyncConversionTest, ConvertsLoopBodyRegionOnOtherStream) {
  ThunkSequence thunks;
  auto* start = Start(thunks, /*convertible=*/false, ComputationStreamId(0));
  ThunkSequence body;
  auto* inner = Start(body, /*convertible=*/true, ComputationStreamId(1));
  Done(body, inner);
  AddCommand(body);
  WhileThunk* loop = Loop(thunks, std::move(body));
  Done(thunks, start);

  ASSERT_OK_AND_ASSIGN(bool changed, Convert(thunks));
  EXPECT_TRUE(changed);
  EXPECT_THAT(thunks, ThunkKindsAre(Thunk::kAsyncStart, Thunk::kWhile,
                                    Thunk::kAsyncDone));
  // Only stream 0 is open inside the loop body; the pair on stream 1 is
  // captured together with the plain command.
  EXPECT_THAT(loop->body_executor().thunks(),
              ThunkKindsAre(Thunk::kCommandBuffer));
  EXPECT_THAT(
      CommandBufferThunks(*loop->body_executor().thunks()[0]),
      ThunkKindsAre(Thunk::kAsyncStart, Thunk::kAsyncDone, Thunk::kReplicaId));
}

TEST_F(CommandBufferAsyncConversionTest, DoesNotCaptureLoopWithLoneDone) {
  ThunkSequence thunks;
  auto* start = Start(thunks, /*convertible=*/false);
  ThunkSequence body;
  Done(body, start);
  AddCommand(body);
  WhileThunk* loop = Loop(thunks, std::move(body));

  ASSERT_OK_AND_ASSIGN(bool changed, Convert(thunks, /*enable_while=*/true));
  EXPECT_TRUE(changed);
  // The loop body joins an operation started outside the loop. Capturing the
  // loop whole would drop that join, so only the plain command in the body is
  // captured.
  EXPECT_THAT(thunks, ThunkKindsAre(Thunk::kAsyncStart, Thunk::kWhile));
  EXPECT_THAT(loop->body_executor().thunks(),
              ThunkKindsAre(Thunk::kAsyncDone, Thunk::kCommandBuffer));
}

TEST_F(CommandBufferAsyncConversionTest, MatchesCanonicalAsyncExecution) {
  ThunkSequence thunks;
  auto* canonical = Start(thunks);
  Done(thunks, canonical);
  ThunkSequence body;
  AddCommand(body);
  thunks.push_back(std::make_unique<AsyncStartThunk>(
      Info(), ComputationStreamId(0), std::move(body),
      canonical->async_execution()));
  Done(thunks, canonical);

  ASSERT_OK_AND_ASSIGN(bool changed, Convert(thunks));
  EXPECT_TRUE(changed);
  EXPECT_THAT(thunks, ThunkKindsAre(Thunk::kCommandBuffer));
}

TEST_F(CommandBufferAsyncConversionTest, DuplicateOutstandingStart) {
  ThunkSequence thunks;
  auto* canonical = Start(thunks);
  ThunkSequence body;
  AddCommand(body);
  thunks.push_back(std::make_unique<AsyncStartThunk>(
      Info(), ComputationStreamId(0), std::move(body),
      canonical->async_execution()));
  Done(thunks, canonical);
  Done(thunks, canonical);
  AddCommand(thunks);

  // The runtime permits only one outstanding start per AsyncExecution, so this
  // sequence can never execute. Debug builds catch it in the pass; optimized
  // builds leave the region as thunks and still capture the trailing command.
#ifndef NDEBUG
  EXPECT_DEATH(Convert(thunks).IgnoreError(), "started twice");
#else
  ASSERT_OK_AND_ASSIGN(bool changed, Convert(thunks));
  EXPECT_TRUE(changed);
  EXPECT_THAT(thunks, ThunkKindsAre(Thunk::kAsyncStart, Thunk::kAsyncStart,
                                    Thunk::kAsyncDone, Thunk::kAsyncDone,
                                    Thunk::kCommandBuffer));
#endif
}

}  // namespace
}  // namespace xla::gpu
