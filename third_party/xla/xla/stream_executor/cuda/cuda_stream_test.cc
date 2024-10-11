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

#include "xla/stream_executor/cuda/cuda_stream.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/stream_executor/cuda/cuda_executor.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace stream_executor {
namespace gpu {
namespace {

using ::testing::Each;
using ::tsl::testing::IsOk;

class CudaStreamTest : public ::testing::Test {
 public:
  std::optional<CudaExecutor> executor_;
  std::unique_ptr<Stream> stream_;
  GpuStream* gpu_stream_;

 private:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                            stream_executor::PlatformManager::PlatformWithId(
                                stream_executor::cuda::kCudaPlatformId));
    executor_.emplace(platform, 0);
    ASSERT_THAT(executor_->Init(), IsOk());
    TF_ASSERT_OK_AND_ASSIGN(stream_, executor_->CreateStream(std::nullopt));
    gpu_stream_ = AsGpuStream(stream_.get());
  }
};

TEST_F(CudaStreamTest, Memset32) {
  constexpr int kBufferNumElements = 42;
  DeviceMemory<uint32_t> buffer =
      executor_->AllocateArray<uint32_t>(kBufferNumElements, 0);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<CudaStream> stream,
                          CudaStream::Create(&executor_.value(),
                                             /*priority=*/std::nullopt));

  // Should fail due to the invalid size parameter.
  EXPECT_THAT(stream->Memset32(&buffer, 0xDEADBEEF,
                               kBufferNumElements * sizeof(uint32_t) + 1),
              ::tsl::testing::StatusIs(absl::StatusCode::kInvalidArgument));

  // Should fail due to the non-4-byte-aligned pointer.
  DeviceMemoryBase unaligned_pointer =
      buffer.GetByteSlice(/*offset_bytes=*/1, /*size_bytes=*/0);
  EXPECT_THAT(stream->Memset32(&unaligned_pointer, 0xDEADBEEF,
                               kBufferNumElements * sizeof(uint32_t) + 1),
              ::tsl::testing::StatusIs(absl::StatusCode::kInvalidArgument));

  // Correct call. Should succeed.
  EXPECT_THAT(stream->Memset32(&buffer, 0xDEADBEEF,
                               kBufferNumElements * sizeof(uint32_t)),
              IsOk());

  std::array<uint32_t, kBufferNumElements> host_buffer;
  EXPECT_THAT(stream->MemcpyD2H(buffer, absl::MakeSpan(host_buffer)), IsOk());

  EXPECT_THAT(stream->BlockHostUntilDone(), IsOk());
  EXPECT_THAT(host_buffer, Each(0xDEADBEEF));
}

}  // namespace
}  // namespace gpu
}  // namespace stream_executor
