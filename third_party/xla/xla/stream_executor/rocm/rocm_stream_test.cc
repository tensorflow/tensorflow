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

#include "xla/stream_executor/rocm/rocm_stream.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_test_kernels.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/rocm/rocm_event.h"
#include "xla/stream_executor/rocm/rocm_executor.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace stream_executor {
namespace gpu {
namespace {

using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::UnorderedElementsAreArray;

class RocmStreamTest : public ::testing::Test {
 public:
  std::optional<RocmExecutor> executor_;

 private:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                            stream_executor::PlatformManager::PlatformWithId(
                                stream_executor::rocm::kROCmPlatformId));
    executor_.emplace(platform, 0);
    ASSERT_THAT(executor_->Init(), absl_testing::IsOk());
  }
};

TEST_F(RocmStreamTest, Memset32) {
  constexpr int kBufferNumElements = 42;
  DeviceAddress<uint32_t> buffer =
      executor_->AllocateArray<uint32_t>(kBufferNumElements, 0);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RocmStream> stream,
                          RocmStream::Create(&executor_.value(),
                                             /*priority=*/std::nullopt));

  // Should fail due to the invalid size parameter.
  EXPECT_THAT(stream->Memset32(&buffer, 0xDEADBEEF,
                               kBufferNumElements * sizeof(uint32_t) + 1),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));

  // Should fail due to the non-4-byte-aligned pointer.
  DeviceAddressBase unaligned_pointer =
      buffer.GetByteSlice(/*offset_bytes=*/1, /*size_bytes=*/0);
  EXPECT_THAT(stream->Memset32(&unaligned_pointer, 0xDEADBEEF,
                               kBufferNumElements * sizeof(uint32_t) + 1),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));

  // Correct call. Should succeed.
  EXPECT_THAT(stream->Memset32(&buffer, 0xDEADBEEF,
                               kBufferNumElements * sizeof(uint32_t)),
              absl_testing::IsOk());

  std::array<uint32_t, kBufferNumElements> host_buffer;
  EXPECT_THAT(stream->MemcpyD2H(buffer, absl::MakeSpan(host_buffer)),
              absl_testing::IsOk());

  EXPECT_THAT(stream->BlockHostUntilDone(), absl_testing::IsOk());
  EXPECT_THAT(host_buffer, Each(0xDEADBEEF));
}

TEST_F(RocmStreamTest, MemZero) {
  constexpr int kBufferNumElements = 42;
  DeviceAddress<uint32_t> buffer =
      executor_->AllocateArray<uint32_t>(kBufferNumElements, 0);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RocmStream> stream,
                          RocmStream::Create(&executor_.value(),
                                             /*priority=*/std::nullopt));

  EXPECT_THAT(stream->Memset32(&buffer, 0xDEADBEEF,
                               kBufferNumElements * sizeof(uint32_t)),
              absl_testing::IsOk());

  // We overwrite half the buffer with zeros.
  EXPECT_THAT(
      stream->MemZero(&buffer, kBufferNumElements / 2 * sizeof(uint32_t)),
      absl_testing::IsOk());

  std::array<uint32_t, kBufferNumElements> host_buffer;
  EXPECT_THAT(stream->MemcpyD2H(buffer, absl::MakeSpan(host_buffer)),
              absl_testing::IsOk());

  EXPECT_THAT(stream->BlockHostUntilDone(), absl_testing::IsOk());
  // We expect the first half of the buffer to be zeros.
  EXPECT_THAT(
      absl::MakeConstSpan(host_buffer).subspan(0, kBufferNumElements / 2),
      Each(0x0));

  // And it shouldn't have touched the second half.
  EXPECT_THAT(absl::MakeConstSpan(host_buffer).subspan(kBufferNumElements / 2),
              Each(0xDEADBEEF));
}

TEST_F(RocmStreamTest, MemcpyHostToDeviceAndBack) {
  constexpr int kBufferNumElements = 42;
  DeviceAddress<uint32_t> buffer =
      executor_->AllocateArray<uint32_t>(kBufferNumElements, 0);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RocmStream> stream,
                          RocmStream::Create(&executor_.value(),
                                             /*priority=*/std::nullopt));

  std::array<uint32_t, kBufferNumElements> src_buffer;
  std::generate(src_buffer.begin(), src_buffer.end(),
                [i = 0]() mutable { return i++; });

  EXPECT_THAT(stream->MemcpyH2D(absl::MakeConstSpan(src_buffer), &buffer),
              absl_testing::IsOk());

  std::array<uint32_t, kBufferNumElements> host_buffer;
  EXPECT_THAT(stream->MemcpyD2H(buffer, absl::MakeSpan(host_buffer)),
              absl_testing::IsOk());

  EXPECT_THAT(stream->BlockHostUntilDone(), absl_testing::IsOk());
  EXPECT_THAT(host_buffer, ElementsAreArray(src_buffer));
}

TEST_F(RocmStreamTest, MemcpyDeviceToDevice) {
  constexpr int kBufferNumElements = 42;
  DeviceAddress<uint32_t> buffer1 =
      executor_->AllocateArray<uint32_t>(kBufferNumElements, 0);
  DeviceAddress<uint32_t> buffer2 =
      executor_->AllocateArray<uint32_t>(kBufferNumElements, 0);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RocmStream> stream,
                          RocmStream::Create(&executor_.value(),
                                             /*priority=*/std::nullopt));

  EXPECT_THAT(stream->Memset32(&buffer1, 0xDEADBEEF,
                               kBufferNumElements * sizeof(uint32_t)),
              absl_testing::IsOk());

  EXPECT_THAT(stream->MemcpyD2D(&buffer2, buffer1,
                                kBufferNumElements * sizeof(uint32_t)),
              absl_testing::IsOk());

  std::array<uint32_t, kBufferNumElements> host_buffer;
  EXPECT_THAT(stream->MemcpyD2H(buffer2, absl::MakeSpan(host_buffer)),
              absl_testing::IsOk());

  EXPECT_THAT(stream->BlockHostUntilDone(), absl_testing::IsOk());
  EXPECT_THAT(host_buffer, Each(0xDEADBEEF));
}

TEST_F(RocmStreamTest, DoHostCallback) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RocmStream> stream,
                          RocmStream::Create(&executor_.value(),
                                             /*priority=*/std::nullopt));

  bool callback_called = false;
  EXPECT_THAT(
      stream->DoHostCallback([&callback_called]() { callback_called = true; }),
      absl_testing::IsOk());

  EXPECT_THAT(stream->BlockHostUntilDone(), absl_testing::IsOk());
  EXPECT_TRUE(callback_called);
}

TEST_F(RocmStreamTest, LaunchKernel) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RocmStream> stream,
                          RocmStream::Create(&executor_.value(),
                                             /*priority=*/std::nullopt));

  TF_ASSERT_OK_AND_ASSIGN(auto add, LoadAddI32TestKernel(&executor_.value()));

  constexpr int64_t kLength = 4;
  constexpr int64_t kByteLength = sizeof(int32_t) * kLength;

  // Prepare arguments: a=1, b=2, c=0
  DeviceAddress<int32_t> a = executor_->AllocateArray<int32_t>(kLength, 0);
  DeviceAddress<int32_t> b = executor_->AllocateArray<int32_t>(kLength, 0);
  DeviceAddress<int32_t> c = executor_->AllocateArray<int32_t>(kLength, 0);

  EXPECT_THAT(stream->Memset32(&a, 1, kByteLength), absl_testing::IsOk());
  EXPECT_THAT(stream->Memset32(&b, 2, kByteLength), absl_testing::IsOk());
  EXPECT_THAT(stream->MemZero(&c, kByteLength), absl_testing::IsOk());
  EXPECT_THAT(add.Launch(ThreadDim(), BlockDim(kLength), stream.get(), a, b, c),
              absl_testing::IsOk());

  EXPECT_THAT(stream->BlockHostUntilDone(), absl_testing::IsOk());

  std::array<int32_t, kLength> host_buffer;
  EXPECT_THAT(stream->MemcpyD2H(c, absl::MakeSpan(host_buffer)),
              absl_testing::IsOk());
  EXPECT_THAT(host_buffer, Each(3));
}

TEST_F(RocmStreamTest, SetName) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RocmStream> stream,
                          RocmStream::Create(&executor_.value(),
                                             /*priority=*/std::nullopt));

  constexpr absl::string_view kStreamName = "Test stream";
  stream->SetName(std::string(kStreamName));
  EXPECT_EQ(stream->GetName(), kStreamName);
}

TEST_F(RocmStreamTest, WaitForEvent) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RocmStream> stream,
                          RocmStream::Create(&executor_.value(),
                                             /*priority=*/std::nullopt));

  TF_ASSERT_OK_AND_ASSIGN(
      RocmEvent event,
      RocmEvent::Create(&executor_.value(), /*allow_timing=*/false));

  EXPECT_THAT(stream->WaitFor(&event), absl_testing::IsOk());

  bool callback_called = false;
  EXPECT_THAT(
      stream->DoHostCallback([&callback_called]() { callback_called = true; }),
      absl_testing::IsOk());

  EXPECT_THAT(stream->RecordEvent(&event), absl_testing::IsOk());
  EXPECT_THAT(stream->BlockHostUntilDone(), absl_testing::IsOk());
  EXPECT_TRUE(callback_called);
}

TEST_F(RocmStreamTest, WaitForOtherStream) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RocmStream> stream1,
                          RocmStream::Create(&executor_.value(),
                                             /*priority=*/std::nullopt));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<RocmStream> stream2,
                          RocmStream::Create(&executor_.value(),
                                             /*priority=*/std::nullopt));

  TF_ASSERT_OK_AND_ASSIGN(
      RocmEvent event,
      RocmEvent::Create(&executor_.value(), /*allow_timing=*/false));

  enum class ExecutionStage {
    kBeforeWaitForEvent,
    kAfterWaitForEvent,
    kAfterWaitForStream
  };

  std::vector<ExecutionStage> execution_order;

  // - stream1 waits for the event to be recorded and
  // - stream2 waits for stream1 to be done.
  // - Afterwards stream2 invokes the host callback.
  EXPECT_THAT(stream1->DoHostCallback([&execution_order]() {
    execution_order.push_back(ExecutionStage::kBeforeWaitForEvent);
  }),
              absl_testing::IsOk());
  EXPECT_THAT(stream1->WaitFor(&event), absl_testing::IsOk());
  EXPECT_THAT(stream1->DoHostCallback([&execution_order]() {
    execution_order.push_back(ExecutionStage::kAfterWaitForEvent);
  }),
              absl_testing::IsOk());
  EXPECT_THAT(stream2->WaitFor(stream1.get()), absl_testing::IsOk());
  EXPECT_THAT(stream2->DoHostCallback([&execution_order]() {
    execution_order.push_back(ExecutionStage::kAfterWaitForStream);
  }),
              absl_testing::IsOk());

  EXPECT_THAT(stream1->RecordEvent(&event), absl_testing::IsOk());
  EXPECT_THAT(stream2->BlockHostUntilDone(), absl_testing::IsOk());
  EXPECT_THAT(execution_order,
              ElementsAre(ExecutionStage::kBeforeWaitForEvent,
                          ExecutionStage::kAfterWaitForEvent,
                          ExecutionStage::kAfterWaitForStream));
}

// ---------------------------------------------------------------------------
// HIP stream handle cache tests
// ---------------------------------------------------------------------------

// Core invariant: after a RocmStream is destroyed its underlying hipStream_t
// is placed in the cache and returned by the very next Create() call that uses
// the same (device, flags, priority) key.
TEST_F(RocmStreamTest, StreamHandleIsReusedAfterDestruction) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<RocmStream> stream,
      RocmStream::Create(&executor_.value(), /*priority=*/std::nullopt));
  hipStream_t original_handle = stream->stream_handle();

  // Destroying the stream should deposit the handle into the cache.
  stream.reset();

  // The very next Create() with identical parameters must return the cached
  // handle rather than allocating a new one.
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<RocmStream> new_stream,
      RocmStream::Create(&executor_.value(), /*priority=*/std::nullopt));

  EXPECT_EQ(new_stream->stream_handle(), original_handle)
      << "hipStream_t handle was not reused from the cache";
}

// The cache stores one vector of handles per key; verify that N handles can be
// deposited and then all retrieved (in LIFO order, so the set must match).
TEST_F(RocmStreamTest, MultipleStreamHandlesAreCachedAndReused) {
  constexpr int kNumStreams = 4;
  std::vector<hipStream_t> original_handles;
  original_handles.reserve(kNumStreams);

  {
    std::vector<std::unique_ptr<RocmStream>> streams;
    streams.reserve(kNumStreams);
    for (int i = 0; i < kNumStreams; ++i) {
      TF_ASSERT_OK_AND_ASSIGN(
          auto s,
          RocmStream::Create(&executor_.value(), /*priority=*/std::nullopt));
      original_handles.push_back(s->stream_handle());
      streams.push_back(std::move(s));
    }
    // All kNumStreams handles are deposited into the cache here.
  }

  // Every new stream must come from the cache (no fresh hipStreamCreate calls).
  std::vector<hipStream_t> reused_handles;
  reused_handles.reserve(kNumStreams);
  {
    std::vector<std::unique_ptr<RocmStream>> streams;
    streams.reserve(kNumStreams);
    for (int i = 0; i < kNumStreams; ++i) {
      TF_ASSERT_OK_AND_ASSIGN(
          auto s,
          RocmStream::Create(&executor_.value(), /*priority=*/std::nullopt));
      reused_handles.push_back(s->stream_handle());
      streams.push_back(std::move(s));
    }
  }

  EXPECT_THAT(reused_handles, UnorderedElementsAreArray(original_handles))
      << "Not all hipStream_t handles were reused from the cache";
}

// A stream recycled from the cache must still be able to enqueue work and
// produce correct results — i.e. the underlying HIP stream is truly idle and
// in a valid state when it is returned from the cache.
TEST_F(RocmStreamTest, ReusedStreamIsFullyFunctional) {
  // Deposit one default-priority handle into the cache.
  {
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<RocmStream> stream,
        RocmStream::Create(&executor_.value(), /*priority=*/std::nullopt));
  }

  // This stream is obtained from the cache (not freshly created).
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<RocmStream> stream,
      RocmStream::Create(&executor_.value(), /*priority=*/std::nullopt));

  constexpr int kNumElements = 16;
  DeviceAddress<uint32_t> buffer =
      executor_->AllocateArray<uint32_t>(kNumElements, /*memory_space=*/0);

  constexpr uint32_t kPattern = 0xCAFEBABE;
  EXPECT_THAT(
      stream->Memset32(&buffer, kPattern, kNumElements * sizeof(uint32_t)),
      absl_testing::IsOk());

  std::array<uint32_t, kNumElements> host_buffer;
  EXPECT_THAT(stream->MemcpyD2H(buffer, absl::MakeSpan(host_buffer)),
              absl_testing::IsOk());
  EXPECT_THAT(stream->BlockHostUntilDone(), absl_testing::IsOk());
  EXPECT_THAT(host_buffer, Each(kPattern));
}

// The cache key includes the stream priority.  A default-priority handle must
// not be handed out in place of a highest-priority handle and vice versa.
//
// Destruction order and creation order are chosen so that the LIFO property
// produces deterministic results whether or not Default and Highest map to the
// same integer priority on this device:
//
//   Destroy order : default (Hd) first, highest (Hh) second.
//   Create order  : highest first, default second.
//
// If priorities differ → separate buckets:
//   - new_highest pops Hh from bucket_highest ✓
//   - new_default  pops Hd from bucket_default ✓
//
// If priorities are the same → shared bucket [Hd, Hh] (Hh on top):
//   - new_highest pops Hh (top of shared bucket) ✓
//   - new_default  pops Hd (next item)            ✓
TEST_F(RocmStreamTest, CacheIsolatesStreamsByPriority) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<RocmStream> default_stream,
      RocmStream::Create(&executor_.value(), StreamPriority::Default));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<RocmStream> highest_stream,
      RocmStream::Create(&executor_.value(), StreamPriority::Highest));

  hipStream_t default_handle = default_stream->stream_handle();
  hipStream_t highest_handle = highest_stream->stream_handle();

  // Destroy default first, highest second → highest handle is on top.
  default_stream.reset();
  highest_stream.reset();

  // Create in reverse order: highest first, default second.
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<RocmStream> new_highest,
      RocmStream::Create(&executor_.value(), StreamPriority::Highest));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<RocmStream> new_default,
      RocmStream::Create(&executor_.value(), StreamPriority::Default));

  EXPECT_EQ(new_highest->stream_handle(), highest_handle)
      << "Highest-priority stream did not reuse its cached hipStream_t handle";
  EXPECT_EQ(new_default->stream_handle(), default_handle)
      << "Default-priority stream did not reuse its cached hipStream_t handle";
}

}  // namespace
}  // namespace gpu
}  // namespace stream_executor
