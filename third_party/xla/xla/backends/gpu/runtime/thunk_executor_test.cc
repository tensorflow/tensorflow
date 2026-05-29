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

#include "xla/backends/gpu/runtime/thunk_executor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/memset_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/while_loop.h"
#include "xla/backends/gpu/runtime/while_thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

static absl::StatusOr<se::StreamExecutor*> GpuExecutor() {
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  auto* platform = se::PlatformManager::PlatformWithName(name).value();
  return platform->ExecutorForDevice(0).value();
}

static Thunk::ThunkInfo ThunkInfo(absl::string_view name) {
  Thunk::ThunkInfo info;
  info.profile_annotation = name;
  return info;
};

TEST(SequentialThunkProgressTrackerTest, TrackProgress) {
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * stream_executor, GpuExecutor());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                       stream_executor->CreateStream());

  static constexpr int64_t kLength = 4;
  static constexpr int64_t kByteLength = sizeof(int32_t) * kLength;

  // Create a buffer slice for each thunk added to a sequence.
  BufferAllocation alloc(/*index=*/0, kByteLength, /*color=*/0);
  std::vector<BufferAllocation::Slice> slices(kLength);
  for (int64_t i = 0; i < kLength; ++i) {
    slices[i] =
        BufferAllocation::Slice(&alloc, i * sizeof(int32_t), sizeof(int32_t));
  }

  // Prepare execute params for running thunk.
  ServiceExecutableRunOptions run_options;
  se::StreamExecutorAddressAllocator allocator(stream_executor);

  se::DeviceAddress<int32_t> storage =
      stream_executor->AllocateArray<int32_t>(kLength, 0);
  BufferAllocations allocations({storage}, 0, &allocator);

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), nullptr, nullptr, nullptr);

  // Create a trivial thunk sequence that runs memzero on each slice.
  Shape shape = ShapeUtil::MakeShape(S32, {kLength});

  // Create two thunk sequences to check that progress tracker is available to
  // both of them at run time.
  ThunkSequence outer_sequence(kLength), nested_sequence(kLength);
  for (int64_t i = 0; i < kLength; ++i) {
    outer_sequence[i] = std::make_unique<MemzeroThunk>(
        ThunkInfo(absl::StrCat("outer_memzero#", i)),
        ShapedSlice{slices[i], shape});
    nested_sequence[i] = std::make_unique<MemzeroThunk>(
        ThunkInfo(absl::StrCat("nested_memzero#", i)),
        ShapedSlice{slices[i], shape});
  }

  // Create a nested sequential thunk in the outer thunk sequence.
  outer_sequence.push_back(std::make_unique<SequentialThunk>(
      ThunkInfo("nested"), std::move(nested_sequence)));

  ThunkExecutor executor(std::move(outer_sequence));
  ASSERT_OK_AND_ASSIGN(ThunkExecutor::ScopedProgressTracker tracker,
                       InstallProgressTracker(stream_executor, executor));

  static constexpr size_t kTotal = 1 + kLength * 2;  // 1 nested sequences
  ASSERT_EQ(tracker.num_thunks(), kTotal);

  // Before execution, no thunks have been launched so both queries return
  // empty results (thunks with InfinitePast executed time are filtered out).
  EXPECT_THAT(tracker.LastCompletedThunks(5), testing::IsEmpty());
  EXPECT_THAT(tracker.FirstPendingThunks(5), testing::IsEmpty());

  ASSERT_OK(executor.ExecuteOnStream(params));

  // After synchronization all executed thunks must be completed.
  ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_THAT(tracker.FirstPendingThunks(5), testing::IsEmpty());

  // CUDA API calls between thunks are slow enough to produce unique
  // timestamps, so we can check strict ordering. LastCompletedThunks returns
  // thunks sorted by executed time descending (most recent first).
  auto completed = tracker.LastCompletedThunks(5);
  ASSERT_EQ(completed.size(), 5);
  EXPECT_EQ(completed[0].name, "nested");
  EXPECT_EQ(completed[1].name, "nested_memzero#3");
  EXPECT_EQ(completed[2].name, "nested_memzero#2");
  EXPECT_EQ(completed[3].name, "nested_memzero#1");
  EXPECT_EQ(completed[4].name, "nested_memzero#0");

  // Execute the same thunk sequence a second time to verify that execution
  // events accumulate and the progress tracker reports 2x completed thunks.
  ASSERT_OK(executor.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());

  static constexpr size_t kThunksPerExecution = kLength + 1 + kLength;
  EXPECT_EQ(tracker.NumCompletedThunks(), 2 * kThunksPerExecution);
  EXPECT_EQ(tracker.NumPendingThunks(), 0);

  // LastCompletedThunks should return thunks from the second execution first
  // (most recent), then from the first execution.
  auto completed2 = tracker.LastCompletedThunks(5);
  ASSERT_EQ(completed2.size(), 5);
  EXPECT_EQ(completed2[0].name, "nested");
  EXPECT_EQ(completed2[1].name, "nested_memzero#3");
  EXPECT_EQ(completed2[2].name, "nested_memzero#2");
  EXPECT_EQ(completed2[3].name, "nested_memzero#1");
  EXPECT_EQ(completed2[4].name, "nested_memzero#0");

  // Second execution thunks must have later timestamps than first execution.
  for (size_t i = 0; i < completed2.size(); ++i) {
    EXPECT_GT(completed2[i].executed, completed[i].executed);
  }
}

TEST(SequentialThunkProgressTrackerTest, TrackWhileLoopNest) {
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * stream_executor, GpuExecutor());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                       stream_executor->CreateStream());

  static constexpr int64_t kLength = 4;
  static constexpr int64_t kByteLength = sizeof(int32_t) * kLength;
  static constexpr int64_t kTripCount = 3;

  BufferAllocation alloc(/*index=*/0, kByteLength, /*color=*/0);
  BufferAllocation::Slice slice(&alloc, 0, sizeof(int32_t));

  ServiceExecutableRunOptions run_options;
  se::StreamExecutorAddressAllocator allocator(stream_executor);

  se::DeviceAddress<int32_t> storage =
      stream_executor->AllocateArray<int32_t>(kLength, 0);
  BufferAllocations allocations({storage}, 0, &allocator);

  Thunk::ExecuteParams params =
      Thunk::ExecuteParams::Create(run_options, allocations, stream.get(),
                                   stream.get(), nullptr, nullptr, nullptr);

  Shape shape = ShapeUtil::MakeShape(S32, {1});

  // Build a while loop with a known trip count containing a single body thunk.
  ThunkSequence condition_thunks;  // empty, not used with trip_count
  ThunkSequence body_thunks;
  body_thunks.push_back(std::make_unique<MemzeroThunk>(
      ThunkInfo("loop_body_memzero"), ShapedSlice{slice, shape}));

  ThunkSequence outer_sequence;
  outer_sequence.push_back(std::make_unique<WhileThunk>(
      ThunkInfo("while_loop"), slice, std::move(condition_thunks),
      std::move(body_thunks), /*trip_count=*/kTripCount));

  ThunkExecutor executor(std::move(outer_sequence));
  ASSERT_OK_AND_ASSIGN(ThunkExecutor::ScopedProgressTracker tracker,
                       InstallProgressTracker(stream_executor, executor));

  ASSERT_OK(executor.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());

  // The while loop body executes kTripCount times, plus one event for the
  // WhileThunk itself. LastCompletedThunks returns most recent first.
  auto completed = tracker.LastCompletedThunks(kTripCount + 1);
  ASSERT_EQ(completed.size(), kTripCount + 1);

  // The most recent event is the WhileThunk itself (recorded after all body
  // iterations complete). It has an empty loop nest since it runs at top level.
  EXPECT_EQ(completed[0].name, "while_loop");
  EXPECT_THAT(completed[0].loop_nest, testing::IsEmpty());

  // The remaining events are the body thunk executions in reverse iteration
  // order: iteration kTripCount-1 first, then kTripCount-2, etc.
  for (size_t i = 0; i < kTripCount; ++i) {
    const auto& thunk = completed[1 + i];
    EXPECT_EQ(thunk.name, "loop_body_memzero");
    ASSERT_EQ(thunk.loop_nest.size(), 1);
    EXPECT_EQ(thunk.loop_nest[0], WhileLoopState({"while_loop", kTripCount, 0,
                                                  kTripCount - 1 - i}));
  }
}

}  // namespace xla::gpu
