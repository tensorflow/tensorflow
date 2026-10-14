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

#include "xla/stream_executor/cuda/green_context.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/stream_executor/cuda/cuda_executor.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_test_kernels.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {
namespace gpu {
namespace {

using ::absl_testing::IsOk;
using ::testing::Each;

class GreenContextTest : public ::testing::Test {
 public:
  CudaExecutor* executor_;

 private:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(Platform * platform,
                         PlatformManager::PlatformWithId(
                             stream_executor::cuda::kCudaPlatformId));
    ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                         platform->ExecutorForDevice(0));
    executor_ = reinterpret_cast<CudaExecutor*>(executor);
  }
};

// A general green context created from a raw SM count can launch a kernel on
// its SM partition.
TEST_F(GreenContextTest, GeneralGreenContext) {
  const int total_sms = executor_->GetDeviceDescription().core_count();
  ASSERT_GT(total_sms, 0);

  absl::StatusOr<std::unique_ptr<GreenContext>> green_context =
      executor_->CreateGreenContext(/*sm_count=*/std::max(1, total_sms / 2));
  if (!green_context.ok()) {
    GTEST_SKIP() << "Green contexts not supported on this device: "
                 << green_context.status();
  }
  EXPECT_GT((*green_context)->sm_count(), 0);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream,
                       executor_->CreateStreamInGreenContext(**green_context));

  ASSERT_OK_AND_ASSIGN(auto add, LoadAddI32TestKernel(executor_));

  constexpr int64_t kLength = 4;
  constexpr int64_t kByteLength = sizeof(int32_t) * kLength;
  DeviceAddress<int32_t> a = executor_->AllocateArray<int32_t>(kLength, 0);
  DeviceAddress<int32_t> b = executor_->AllocateArray<int32_t>(kLength, 0);
  DeviceAddress<int32_t> c = executor_->AllocateArray<int32_t>(kLength, 0);

  EXPECT_THAT(stream->Memset32(&a, 1, kByteLength), IsOk());
  EXPECT_THAT(stream->Memset32(&b, 2, kByteLength), IsOk());
  EXPECT_THAT(stream->MemZero(&c, kByteLength), IsOk());
  EXPECT_THAT(add.Launch(ThreadDim(), BlockDim(kLength), stream.get(), a, b, c),
              IsOk());

  std::array<int32_t, kLength> host;
  EXPECT_THAT(stream->MemcpyD2H(c, absl::MakeSpan(host)), IsOk());
  EXPECT_THAT(stream->BlockHostUntilDone(), IsOk());
  EXPECT_THAT(host, Each(3));
}

// A green-context (partition) compute stream and a primary-context stream can
// be ordered with a cross-stream event handshake (the interaction exercised by
// async collectives), and produce correct results.
TEST_F(GreenContextTest, GreenToPrimaryEventHandshake) {
  const int total_sms = executor_->GetDeviceDescription().core_count();
  ASSERT_GT(total_sms, 0);

  absl::StatusOr<std::unique_ptr<GreenContext>> green_context =
      executor_->CreateGreenContext(/*sm_count=*/std::max(1, total_sms / 2));
  if (!green_context.ok()) {
    GTEST_SKIP() << "Green contexts not supported on this device: "
                 << green_context.status();
  }

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> green_stream,
                       executor_->CreateStreamInGreenContext(**green_context));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> primary_stream,
                       executor_->CreateStream(/*priority=*/std::nullopt));

  ASSERT_OK_AND_ASSIGN(auto add, LoadAddI32TestKernel(executor_));

  constexpr int64_t kLength = 4;
  constexpr int64_t kByteLength = sizeof(int32_t) * kLength;
  DeviceAddress<int32_t> a = executor_->AllocateArray<int32_t>(kLength, 0);
  DeviceAddress<int32_t> b = executor_->AllocateArray<int32_t>(kLength, 0);
  DeviceAddress<int32_t> c = executor_->AllocateArray<int32_t>(kLength, 0);
  DeviceAddress<int32_t> d = executor_->AllocateArray<int32_t>(kLength, 0);

  // Green stream computes c = a + b = 3.
  EXPECT_THAT(green_stream->Memset32(&a, 1, kByteLength), IsOk());
  EXPECT_THAT(green_stream->Memset32(&b, 2, kByteLength), IsOk());
  EXPECT_THAT(green_stream->MemZero(&c, kByteLength), IsOk());
  EXPECT_THAT(
      add.Launch(ThreadDim(), BlockDim(kLength), green_stream.get(), a, b, c),
      IsOk());

  // Primary stream waits for the green stream, then computes d = c + c = 6.
  EXPECT_THAT(primary_stream->WaitFor(green_stream.get()), IsOk());
  EXPECT_THAT(
      add.Launch(ThreadDim(), BlockDim(kLength), primary_stream.get(), c, c, d),
      IsOk());

  std::array<int32_t, kLength> host;
  EXPECT_THAT(primary_stream->MemcpyD2H(d, absl::MakeSpan(host)), IsOk());
  EXPECT_THAT(primary_stream->BlockHostUntilDone(), IsOk());
  EXPECT_THAT(host, Each(6));
}

}  // namespace
}  // namespace gpu
}  // namespace stream_executor
