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

#include "xla/stream_executor/cuda/checksum_kernel_cuda.h"

#include <array>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>

#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace se = stream_executor;

namespace {

const std::array<uint8_t, 16> kIotaInput16 = {0, 1, 2,  3,  4,  5,  6,  7,
                                              8, 9, 10, 11, 12, 13, 14, 15};
const std::array<uint8_t, 1023> kIotaInput1023 = []() {
  std::array<uint8_t, 1023> result;
  std::iota(result.begin(), result.end(), 0);
  return result;
}();

class ChecksumKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    platform_ = se::PlatformManager::PlatformWithName("CUDA").value();
    executor_ = platform_->ExecutorForDevice(0).value();
    TF_ASSERT_OK_AND_ASSIGN(stream_, executor_->CreateStream(std::nullopt));
  }

  absl::StatusOr<uint64_t> ComputeChecksumOnDevice(
      absl::Span<const uint8_t> input, uint64_t key) {
    se::DeviceMemory<uint8_t> device_input =
        executor_->AllocateArray<uint8_t>(input.size());
    if (device_input.is_null()) {
      return absl::InternalError(absl::StrFormat(
          "Failed to allocate device input (%zu B)", input.size()));
    }
    auto input_cleanup =
        absl::MakeCleanup([&]() { executor_->Deallocate(&device_input); });

    se::DeviceMemory<uint64_t> device_output =
        executor_->AllocateScalar<uint64_t>();
    if (device_output.is_null()) {
      return absl::InternalError(absl::StrFormat(
          "Failed to allocate device output (%zu B)", sizeof(uint64_t)));
    }
    auto output_cleanup =
        absl::MakeCleanup([&]() { executor_->Deallocate(&device_output); });

    TF_RETURN_IF_ERROR(
        stream_->Memcpy(&device_input, input.data(), input.size()));

    TF_RETURN_IF_ERROR(LaunchHalfSipHash13Kernel(stream_.get(), &device_input,
                                                 key, &device_output));

    uint64_t gpu_result = 0;
    TF_RETURN_IF_ERROR(
        stream_->Memcpy(&gpu_result, device_output, sizeof(gpu_result)));
    TF_RETURN_IF_ERROR(stream_->BlockHostUntilDone());

    return gpu_result;
  }

  se::Platform* platform_;
  se::StreamExecutor* executor_;
  std::unique_ptr<se::Stream> stream_;
};

TEST_F(ChecksumKernelTest, ComputesCorrectChecksum) {
  uint64_t result;

  TF_ASSERT_OK_AND_ASSIGN(result,
                          ComputeChecksumOnDevice(kIotaInput16, /*key=*/0));
  EXPECT_EQ(0xda0e34756a97b8cd, result);

  TF_ASSERT_OK_AND_ASSIGN(result,
                          ComputeChecksumOnDevice(kIotaInput1023, /*key=*/0));
  EXPECT_EQ(0x7b1042caf86d8f9d, result);
}

TEST_F(ChecksumKernelTest, ComputesCorrectChecksumWithKey) {
  const uint64_t kTestKey = 0x1234567890abcdef;

  uint64_t result;

  TF_ASSERT_OK_AND_ASSIGN(result,
                          ComputeChecksumOnDevice(kIotaInput16, kTestKey));
  EXPECT_EQ(0x1c199a44d1e8a192, result);

  TF_ASSERT_OK_AND_ASSIGN(result,
                          ComputeChecksumOnDevice(kIotaInput1023, kTestKey));
  EXPECT_EQ(0xe26e222700b89641, result);
}

}  // namespace
