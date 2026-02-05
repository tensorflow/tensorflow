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

#include "xla/stream_executor/gpu/buffer_debug_log.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/ascii.h"
#include "xla/backends/gpu/runtime/buffer_debug_log.pb.h"
#include "xla/backends/gpu/runtime/buffer_debug_log_structs.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::gpu {
namespace {

using ::xla::gpu::BufferDebugLogEntry;
using ::xla::gpu::BufferDebugLogHeader;
using ::xla::gpu::ThunkId;

class BufferDebugLogTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto name = absl::AsciiStrToUpper(
        xla::PlatformUtil::CanonicalPlatformName("gpu").value());
    TF_ASSERT_OK_AND_ASSIGN(platform_, PlatformManager::PlatformWithName(name));
    TF_ASSERT_OK_AND_ASSIGN(executor_, platform_->ExecutorForDevice(0));
    TF_ASSERT_OK_AND_ASSIGN(stream_, executor_->CreateStream(std::nullopt));
    allocator_ =
        std::make_unique<StreamExecutorAddressAllocator>(stream_->parent());
  }

  Platform* platform_;
  StreamExecutor* executor_;
  std::unique_ptr<Stream> stream_;
  std::unique_ptr<StreamExecutorAddressAllocator> allocator_;
};

TEST_F(BufferDebugLogTest, CreateBufferDebugLogOnDevice_InitializesEmptyLog) {
  DeviceAddress<uint8_t> log_buffer = executor_->AllocateArray<uint8_t>(1024);

  TF_ASSERT_OK_AND_ASSIGN(auto device_log,
                          BufferDebugLog<BufferDebugLogEntry>::CreateOnDevice(
                              *stream_, log_buffer));
  TF_ASSERT_OK_AND_ASSIGN(auto host_log, device_log.ReadFromDevice(*stream_));

  EXPECT_EQ(host_log.size(), 0);
}

TEST_F(BufferDebugLogTest,
       CreateBufferDebugLogOnDevice_InitializesLogWithCorrectCapacity) {
  constexpr size_t kMaxEntries = 10;
  constexpr size_t kExpectedHeaderSize = sizeof(BufferDebugLogHeader);
  constexpr size_t kExpectedEntriesSize =
      sizeof(BufferDebugLogEntry) * kMaxEntries;
  DeviceAddress<uint8_t> log_buffer = executor_->AllocateArray<uint8_t>(
      kExpectedHeaderSize + kExpectedEntriesSize);

  TF_ASSERT_OK_AND_ASSIGN(auto device_log,
                          BufferDebugLog<BufferDebugLogEntry>::CreateOnDevice(
                              *stream_, log_buffer));

  EXPECT_EQ(device_log.GetDeviceHeader().size(), kExpectedHeaderSize);
  EXPECT_EQ(device_log.GetDeviceEntries().size(), kExpectedEntriesSize);
}

TEST_F(BufferDebugLogTest, CreateBufferDebugLogOnDevice_InitializesHeader) {
  constexpr size_t kMaxEntries = 123;
  DeviceAddress<uint8_t> log_buffer = executor_->AllocateArray<uint8_t>(
      BufferDebugLog<BufferDebugLogEntry>::RequiredSizeForEntries(kMaxEntries));

  TF_ASSERT_OK_AND_ASSIGN(auto device_log,
                          BufferDebugLog<BufferDebugLogEntry>::CreateOnDevice(
                              *stream_, log_buffer));
  TF_ASSERT_OK_AND_ASSIGN(BufferDebugLogHeader header,
                          device_log.ReadHeaderFromDevice(*stream_));

  EXPECT_EQ(header.write_idx, 0);
  EXPECT_EQ(header.capacity, kMaxEntries);
}

TEST_F(BufferDebugLogTest, CreateBufferDebugLogOnDevice_FailsForNullBuffer) {
  EXPECT_THAT(BufferDebugLog<BufferDebugLogEntry>::CreateOnDevice(
                  *stream_, DeviceAddress<uint8_t>()),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(BufferDebugLogTest,
       CreateBufferDebugLogOnDevice_FailsForTooSmallBuffer) {
  DeviceAddress<uint8_t> log_buffer = executor_->AllocateArray<uint8_t>(
      BufferDebugLog<BufferDebugLogEntry>::RequiredSizeForEntries(1) - 1);

  EXPECT_THAT(
      BufferDebugLog<BufferDebugLogEntry>::CreateOnDevice(*stream_, log_buffer),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace stream_executor::gpu
