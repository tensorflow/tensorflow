/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/sdc_log.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include <gtest/gtest.h>
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tsl/platform/statusor.h"

namespace se = stream_executor;

namespace {

class SdcLogTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(platform_,
                            se::PlatformManager::PlatformWithName("CUDA"));
    TF_ASSERT_OK_AND_ASSIGN(executor_, platform_->ExecutorForDevice(0));
    TF_ASSERT_OK_AND_ASSIGN(stream_, executor_->CreateStream(std::nullopt));
    allocator_ =
        std::make_unique<se::StreamExecutorMemoryAllocator>(stream_->parent());
  }

  se::Platform* platform_;
  se::StreamExecutor* executor_;
  std::unique_ptr<se::Stream> stream_;
  std::unique_ptr<se::StreamExecutorMemoryAllocator> allocator_;
};

TEST_F(SdcLogTest, CreateSdcLogOnDevice_AllocatesEmptyLog) {
  TF_ASSERT_OK_AND_ASSIGN(se::cuda::SdcLog device_log,
                          se::cuda::SdcLog::CreateOnDevice(
                              /*max_entries=*/10, *stream_, *allocator_));
  TF_ASSERT_OK_AND_ASSIGN(auto host_log, device_log.ReadFromDevice(*stream_));

  EXPECT_EQ(host_log.size(), 0);
}

TEST_F(SdcLogTest, CreateSdcLogOnDevice_AllocatesEnoughSpace) {
  constexpr uint32_t kMaxEntries = 10;
  constexpr size_t kExpectedHeaderSize = sizeof(se::cuda::SdcLogHeader);
  constexpr size_t kExpectedEntriesSize =
      sizeof(se::cuda::SdcLogEntry) * kMaxEntries;

  TF_ASSERT_OK_AND_ASSIGN(
      se::cuda::SdcLog device_log,
      se::cuda::SdcLog::CreateOnDevice(kMaxEntries, *stream_, *allocator_));
  EXPECT_EQ(device_log.GetDeviceHeader().size(), kExpectedHeaderSize);
  EXPECT_EQ(device_log.GetDeviceEntries().size(), kExpectedEntriesSize);
}

TEST_F(SdcLogTest, CreateSdcLogOnDevice_InitializesHeader) {
  constexpr uint32_t kMaxEntries = 10;

  TF_ASSERT_OK_AND_ASSIGN(
      se::cuda::SdcLog device_log,
      se::cuda::SdcLog::CreateOnDevice(kMaxEntries, *stream_, *allocator_));

  TF_ASSERT_OK_AND_ASSIGN(se::cuda::SdcLogHeader header,
                          device_log.ReadHeaderFromDevice(*stream_));
  EXPECT_EQ(header.write_idx, 0);
  EXPECT_EQ(header.capacity, kMaxEntries);
}

}  // namespace
