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

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/buffer_debug_log_structs.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/buffer_debug_float_check_kernel.h"
#include "xla/stream_executor/gpu/buffer_debug_log.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/stream_executor/typed_kernel_factory.h"  // IWYU pragma: keep, required for KernelType::FactoryType::Create
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"

namespace se = stream_executor;

namespace stream_executor::cuda {
namespace {

using xla::gpu::BufferDebugLogEntryId;
using xla::gpu::ThunkId;

class FloatCheckKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(platform_,
                            se::PlatformManager::PlatformWithName("CUDA"));
    TF_ASSERT_OK_AND_ASSIGN(executor_, platform_->ExecutorForDevice(0));
    TF_ASSERT_OK_AND_ASSIGN(stream_, executor_->CreateStream(std::nullopt));
    allocator_ =
        std::make_unique<se::StreamExecutorMemoryAllocator>(stream_->parent());

    if (!executor_->GetDeviceDescription()
             .cuda_compute_capability()
             .IsAtLeastPascal()) {
      GTEST_SKIP()
          << "Buffer checking is not supported on CUDA architectures older "
             "than Pascal due to missing atomic fetch_add with system scope";
    }
  }

  template <typename T>
  absl::StatusOr<se::DeviceMemory<T>> CheckNotNull(
      se::DeviceMemory<T> device_memory, absl::string_view name) {
    if (device_memory.is_null()) {
      return absl::InternalError(
          absl::StrFormat("Device memory for %s is null", name));
    }
    return device_memory;
  }

  template <typename Kernel, typename T>
  absl::Status AppendFloatCheckOnDevice(
      BufferDebugLogEntryId entry_id, const std::vector<T>& input,
      se::gpu::BufferDebugLog& buffer_debug_log,
      stream_executor::ThreadDim dim = stream_executor::ThreadDim(1, 1, 1)) {
    // Load kernel
    gpu::GpuKernelRegistry registry =
        gpu::GpuKernelRegistry::GetGlobalRegistry();
    TF_ASSIGN_OR_RETURN(auto kernel, registry.LoadKernel<Kernel>(executor_));

    // Setup device buffers
    TF_ASSIGN_OR_RETURN(
        se::DeviceMemory<T> device_input,
        CheckNotNull(executor_->AllocateArray<T>(input.size()), "input"));
    auto cleanup_input =
        absl::MakeCleanup([&]() { executor_->Deallocate(&device_input); });

    // Call kernel
    TF_RETURN_IF_ERROR(stream_->Memcpy(&device_input, input.data(),
                                       input.size() * sizeof(input[0])));
    TF_RETURN_IF_ERROR(kernel.Launch(dim, stream_executor::BlockDim(1, 1, 1),
                                     stream_.get(), entry_id, device_input,
                                     device_input.ElementCount() * sizeof(T),
                                     buffer_debug_log.GetDeviceHeader(),
                                     buffer_debug_log.GetDeviceEntries()));
    TF_RETURN_IF_ERROR(stream_->BlockHostUntilDone());

    // The result gets stored in `buffer_debug_log`.
    return absl::OkStatus();
  }

  se::Platform* platform_;
  se::StreamExecutor* executor_;
  std::unique_ptr<se::Stream> stream_;
  std::unique_ptr<se::StreamExecutorMemoryAllocator> allocator_;
};

TEST_F(FloatCheckKernelTest, ChecksFloatsForF32) {
  se::DeviceMemory<uint8_t> mem = executor_->AllocateArray<uint8_t>(1024);
  std::vector<float> input = {1.0f, std::numeric_limits<float>::quiet_NaN(),
                              2.0f, std::numeric_limits<float>::quiet_NaN()};
  TF_ASSERT_OK_AND_ASSIGN(
      se::gpu::BufferDebugLog device_log,
      se::gpu::BufferDebugLog::CreateOnDevice(*stream_, mem));

  TF_EXPECT_OK(AppendFloatCheckOnDevice<gpu::BufferDebugFloatCheckF32Kernel>(
      BufferDebugLogEntryId{123}, input, device_log));

  TF_ASSERT_OK_AND_ASSIGN(auto host_log, device_log.ReadFromDevice(*stream_));
  ASSERT_GE(host_log.size(), 1);
  EXPECT_EQ(host_log[0].value, 2);
}

TEST_F(FloatCheckKernelTest, ChecksFloatsForBf16) {
  se::DeviceMemory<uint8_t> mem = executor_->AllocateArray<uint8_t>(1024);
  std::vector<xla::bfloat16> input = {
      xla::bfloat16(1.0f),
      xla::bfloat16(std::numeric_limits<float>::quiet_NaN()),
      xla::bfloat16(2.0f),
      xla::bfloat16(std::numeric_limits<float>::quiet_NaN())};
  TF_ASSERT_OK_AND_ASSIGN(
      se::gpu::BufferDebugLog device_log,
      se::gpu::BufferDebugLog::CreateOnDevice(*stream_, mem));

  TF_EXPECT_OK(AppendFloatCheckOnDevice<gpu::BufferDebugFloatCheckBf16Kernel>(
      BufferDebugLogEntryId{0}, input, device_log));

  TF_ASSERT_OK_AND_ASSIGN(auto host_log, device_log.ReadFromDevice(*stream_));
  ASSERT_GE(host_log.size(), 1);
  EXPECT_EQ(host_log[0].value, 2);
}

TEST_F(FloatCheckKernelTest, ChecksFloatsInParallel) {
  se::DeviceMemory<uint8_t> mem = executor_->AllocateArray<uint8_t>(1024);
  std::vector<float> input(1024, 1.0f);
  input[100] = std::numeric_limits<float>::quiet_NaN();
  input[200] = std::numeric_limits<float>::quiet_NaN();
  input[300] = std::numeric_limits<float>::quiet_NaN();

  TF_ASSERT_OK_AND_ASSIGN(
      se::gpu::BufferDebugLog device_log,
      se::gpu::BufferDebugLog::CreateOnDevice(*stream_, mem));

  TF_EXPECT_OK(AppendFloatCheckOnDevice<gpu::BufferDebugFloatCheckF32Kernel>(
      BufferDebugLogEntryId{0}, input, device_log, se::ThreadDim(2, 4, 8)));
  TF_EXPECT_OK(AppendFloatCheckOnDevice<gpu::BufferDebugFloatCheckF32Kernel>(
      BufferDebugLogEntryId{0}, input, device_log, se::ThreadDim(2, 4, 8)));

  TF_ASSERT_OK_AND_ASSIGN(auto host_log, device_log.ReadFromDevice(*stream_));
  ASSERT_GE(host_log.size(), 2);
  EXPECT_EQ(host_log[0].value, 3);
  EXPECT_EQ(host_log[1].value, 3);
}

}  // namespace
}  // namespace stream_executor::cuda
