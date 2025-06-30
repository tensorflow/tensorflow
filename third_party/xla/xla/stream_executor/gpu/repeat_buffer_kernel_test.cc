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

#include "xla/stream_executor/gpu/repeat_buffer_kernel.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/ascii.h"
#include "absl/types/span.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::gpu {
namespace {
using ::testing::ElementsAreArray;
using ::tsl::testing::IsOk;

class RepeatBufferKernelTest : public testing::Test {
 public:
  RepeatBufferKernelTest() {
    std::string name = absl::AsciiStrToUpper(
        xla::PlatformUtil::CanonicalPlatformName("gpu").value());
    Platform* platform =
        stream_executor::PlatformManager::PlatformWithName(name).value();
    executor_ = platform->ExecutorForDevice(0).value();
  }

  StreamExecutor* executor_;
};

TEST_F(RepeatBufferKernelTest, CreateRepeatedBufferAndTestResult) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream,
                          executor_->CreateStream());

  constexpr int kNumberOfRepeatedElements = 4;
  constexpr std::array<float, kNumberOfRepeatedElements> kInitialBuf = {42, 24,
                                                                        33, 66};

  // We use a non-dividing number of elements here to ensure that also the last
  // non-complete section is handled correctly.
  constexpr int kNumberOfTotalElements = 129;
  DeviceMemory<float> buffer =
      executor_->AllocateArray<float>(kNumberOfTotalElements);

  TF_CHECK_OK(stream->MemcpyH2D(absl::MakeConstSpan(kInitialBuf), &buffer));

  TF_ASSERT_OK_AND_ASSIGN(
      RepeatBufferKernel::KernelType kernel,
      GpuKernelRegistry::GetGlobalRegistry().LoadKernel<RepeatBufferKernel>(
          executor_));

  EXPECT_THAT(
      kernel.Launch(
          ThreadDim{kNumberOfRepeatedElements * sizeof(float), 1, 1},
          BlockDim{1, 1, 1}, stream.get(),
          static_cast<const DeviceMemoryBase&>(buffer),
          static_cast<int64_t>(kNumberOfRepeatedElements * sizeof(float)),
          static_cast<int64_t>(kNumberOfTotalElements * sizeof(float))),
      IsOk());

  std::array<float, kNumberOfTotalElements> result_buffer{};
  absl::Span<const float> result = absl::MakeConstSpan(result_buffer);

  TF_CHECK_OK(stream->MemcpyD2H(buffer, absl::MakeSpan(result_buffer)));
  TF_CHECK_OK(stream->BlockHostUntilDone());

  for (int offset = 0; offset < kNumberOfTotalElements;
       offset += kNumberOfRepeatedElements) {
    absl::Span<const float> repeated_section =
        result.subspan(offset, kNumberOfRepeatedElements);

    // This subspan ensures we take into account that the last section only has
    // one element.
    absl::Span<const float> reference =
        absl::MakeConstSpan(kInitialBuf).subspan(0, repeated_section.size());

    EXPECT_THAT(repeated_section, ElementsAreArray(reference));
  }
}

}  // namespace
}  // namespace stream_executor::gpu
