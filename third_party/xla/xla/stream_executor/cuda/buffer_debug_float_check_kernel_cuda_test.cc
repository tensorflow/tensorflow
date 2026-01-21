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

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/buffer_debug_log_structs.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
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

using xla::gpu::BufferDebugFloatCheckEntry;
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
        std::make_unique<StreamExecutorAddressAllocator>(stream_->parent());

    if (!executor_->GetDeviceDescription()
             .cuda_compute_capability()
             .IsAtLeastPascal()) {
      GTEST_SKIP()
          << "Buffer checking is not supported on CUDA architectures older "
             "than Pascal due to missing atomic fetch_add with system scope";
    }
  }

  template <typename T>
  absl::StatusOr<se::DeviceAddress<T>> CheckNotNull(
      se::DeviceAddress<T> device_memory, absl::string_view name) {
    if (device_memory.is_null()) {
      return absl::InternalError(
          absl::StrFormat("Device memory for %s is null", name));
    }
    return device_memory;
  }

  template <typename Kernel, typename InputType, typename BufferType>
  absl::Status AppendFloatCheckOnDevice(
      BufferDebugLogEntryId entry_id, const std::vector<InputType>& input,
      se::gpu::BufferDebugLog<BufferType>& buffer_debug_log,
      stream_executor::BlockDim block_dim = stream_executor::BlockDim(1, 1, 1),
      size_t temp_buffer_size_elements = 1024) {
    // Load kernel
    gpu::GpuKernelRegistry registry =
        gpu::GpuKernelRegistry::GetGlobalRegistry();
    TF_ASSIGN_OR_RETURN(auto kernel, registry.LoadKernel<Kernel>(executor_));
    TF_ASSIGN_OR_RETURN(
        auto reduce_kernel,
        registry
            .LoadKernel<gpu::BufferDebugAppendReducedFloatCheckResultsKernel>(
                executor_));

    // Setup device buffers
    TF_ASSIGN_OR_RETURN(
        se::DeviceAddress<InputType> device_input,
        CheckNotNull(executor_->AllocateArray<InputType>(input.size()),
                     "input"));
    auto cleanup_input =
        absl::MakeCleanup([&]() { executor_->Deallocate(&device_input); });

    TF_ASSIGN_OR_RETURN(
        se::DeviceAddress<xla::gpu::FloatCheckResult> device_tmp,
        CheckNotNull(executor_->AllocateArray<xla::gpu::FloatCheckResult>(
                         temp_buffer_size_elements),
                     "tmp"));
    auto cleanup_tmp =
        absl::MakeCleanup([&]() { executor_->Deallocate(&device_tmp); });

    const se::ThreadDim thread_dim(1024, 1, 1);

    // Call kernel
    TF_RETURN_IF_ERROR(stream_->Memcpy(&device_input, input.data(),
                                       input.size() * sizeof(input[0])));
    TF_RETURN_IF_ERROR(kernel.Launch(thread_dim, block_dim, stream_.get(),
                                     device_input, device_input.ElementCount(),
                                     device_tmp, device_tmp.ElementCount()));
    TF_RETURN_IF_ERROR(reduce_kernel.Launch(
        thread_dim, se::BlockDim(1, 1, 1), stream_.get(), device_tmp,
        std::min(device_tmp.ElementCount(),
                 block_dim.x * block_dim.y * block_dim.z),
        entry_id, buffer_debug_log.GetDeviceHeader(),
        buffer_debug_log.GetDeviceEntries()));
    TF_RETURN_IF_ERROR(stream_->BlockHostUntilDone());

    // The result gets stored in `buffer_debug_log`.
    return absl::OkStatus();
  }

  se::Platform* platform_;
  se::StreamExecutor* executor_;
  std::unique_ptr<se::Stream> stream_;
  std::unique_ptr<StreamExecutorAddressAllocator> allocator_;
};

TEST_F(FloatCheckKernelTest, ChecksFloatsForF32) {
  se::DeviceAddress<uint8_t> mem = executor_->AllocateArray<uint8_t>(1024);
  std::vector<float> input(1024, 1.0f);
  input[100] = std::numeric_limits<float>::quiet_NaN();
  input[200] = std::numeric_limits<float>::quiet_NaN();
  input[300] = 0.0f;
  input[400] = std::numeric_limits<float>::infinity();
  input[500] = std::numeric_limits<float>::infinity();
  input[600] = std::numeric_limits<float>::infinity();
  TF_ASSERT_OK_AND_ASSIGN(
      auto device_log,
      se::gpu::BufferDebugLog<BufferDebugFloatCheckEntry>::CreateOnDevice(
          *stream_, mem));

  EXPECT_OK(AppendFloatCheckOnDevice<gpu::BufferDebugFloatCheckF32Kernel>(
      BufferDebugLogEntryId{123}, input, device_log));

  TF_ASSERT_OK_AND_ASSIGN(auto host_log, device_log.ReadFromDevice(*stream_));
  ASSERT_GE(host_log.size(), 1);
  EXPECT_EQ(host_log[0].nan_count, 2);
  EXPECT_EQ(host_log[0].inf_count, 3);
  EXPECT_EQ(host_log[0].zero_count, 1);
}

TEST_F(FloatCheckKernelTest, ChecksFloatsForBf16) {
  std::vector<xla::bfloat16> input(1024, xla::bfloat16(1.0f));
  input[10] = xla::bfloat16(std::numeric_limits<float>::quiet_NaN());
  input[20] = xla::bfloat16(std::numeric_limits<float>::quiet_NaN());
  input[30] = xla::bfloat16(0.0f),
  input[40] = xla::bfloat16(std::numeric_limits<float>::infinity());
  input[50] = xla::bfloat16(std::numeric_limits<float>::infinity());
  input[60] = xla::bfloat16(std::numeric_limits<float>::infinity());

  se::DeviceAddress<uint8_t> mem = executor_->AllocateArray<uint8_t>(1024);
  TF_ASSERT_OK_AND_ASSIGN(
      auto device_log,
      se::gpu::BufferDebugLog<BufferDebugFloatCheckEntry>::CreateOnDevice(
          *stream_, mem));

  EXPECT_OK(AppendFloatCheckOnDevice<gpu::BufferDebugFloatCheckBf16Kernel>(
      BufferDebugLogEntryId{0}, input, device_log));

  TF_ASSERT_OK_AND_ASSIGN(auto host_log, device_log.ReadFromDevice(*stream_));
  ASSERT_GE(host_log.size(), 1);
  EXPECT_EQ(host_log[0].nan_count, 2);
  EXPECT_EQ(host_log[0].inf_count, 3);
  EXPECT_EQ(host_log[0].zero_count, 1);
}

TEST_F(FloatCheckKernelTest, ChecksFloatsInParallel) {
  static constexpr size_t kNumNaNs = 100;
  static constexpr size_t kNumInfs = 200;
  static constexpr size_t kNumZeros = 300;
  static constexpr size_t kMaxTestValues =
      std::max(std::max(kNumNaNs, kNumInfs), kNumZeros);

  const se::DeviceDescription& device_desc = executor_->GetDeviceDescription();
  const size_t threads_per_core = device_desc.threads_per_core_limit();
  const size_t num_cores = device_desc.core_count();
  const size_t input_size = num_cores * threads_per_core * 3 / 2;
  const size_t test_value_stride = input_size / (kMaxTestValues + 1);
  ASSERT_GT(input_size, kMaxTestValues);
  ASSERT_GT(test_value_stride, 2);

  std::vector<float> input(input_size, 1.0f);
  for (size_t i = 0; i < kNumNaNs; ++i) {
    input[i * test_value_stride] = std::numeric_limits<float>::quiet_NaN();
  }
  for (size_t i = 0; i < kNumInfs; ++i) {
    input[i * test_value_stride + 1] = std::numeric_limits<float>::infinity();
  }
  for (size_t i = 0; i < kNumZeros; ++i) {
    input[i * test_value_stride + 2] = 0.0f;
  }

  se::DeviceAddress<uint8_t> log_mem = executor_->AllocateArray<uint8_t>(1024);
  TF_ASSERT_OK_AND_ASSIGN(
      auto device_log,
      se::gpu::BufferDebugLog<BufferDebugFloatCheckEntry>::CreateOnDevice(
          *stream_, log_mem));

  int64_t threads_per_block;
  int64_t num_blocks;
  CalculateDimensionality(executor_->GetDeviceDescription(), input.size(),
                          &threads_per_block, &num_blocks);
  const se::BlockDim block_dim(num_blocks);
  EXPECT_OK(AppendFloatCheckOnDevice<gpu::BufferDebugFloatCheckF32Kernel>(
      BufferDebugLogEntryId{0}, input, device_log, block_dim));
  EXPECT_OK(AppendFloatCheckOnDevice<gpu::BufferDebugFloatCheckF32Kernel>(
      BufferDebugLogEntryId{0}, input, device_log, block_dim));

  TF_ASSERT_OK_AND_ASSIGN(auto host_log, device_log.ReadFromDevice(*stream_));
  ASSERT_GE(host_log.size(), 2);
  EXPECT_EQ(host_log[0].nan_count, kNumNaNs);
  EXPECT_EQ(host_log[0].inf_count, kNumInfs);
  EXPECT_EQ(host_log[0].zero_count, kNumZeros);
  EXPECT_EQ(host_log[1].nan_count, kNumNaNs);
  EXPECT_EQ(host_log[1].inf_count, kNumInfs);
  EXPECT_EQ(host_log[1].zero_count, kNumZeros);
}

TEST_F(FloatCheckKernelTest, ReduceFloatCheckResults) {
  static constexpr size_t kNumNaNs = 100;
  static constexpr size_t kNumInfs = 200;
  static constexpr size_t kNumZeros = 300;
  static constexpr size_t kIntermediateResults = 16 * 1024;

  std::vector<xla::gpu::FloatCheckResult> results(kIntermediateResults);
  for (size_t i = 0; i < kIntermediateResults; ++i) {
    results[i].nan_count = i < kNumNaNs ? 1 : 0;
    results[i].inf_count = i < kNumInfs ? 1 : 0;
    results[i].zero_count = i < kNumZeros ? 1 : 0;
  }

  gpu::GpuKernelRegistry registry = gpu::GpuKernelRegistry::GetGlobalRegistry();
  TF_ASSERT_OK_AND_ASSIGN(
      auto reduce_kernel,
      registry.LoadKernel<gpu::BufferDebugAppendReducedFloatCheckResultsKernel>(
          executor_));

  se::DeviceAddress<uint8_t> log_mem = executor_->AllocateArray<uint8_t>(1024);
  TF_ASSERT_OK_AND_ASSIGN(
      auto device_log,
      se::gpu::BufferDebugLog<BufferDebugFloatCheckEntry>::CreateOnDevice(
          *stream_, log_mem));
  TF_ASSERT_OK_AND_ASSIGN(
      se::DeviceAddress<xla::gpu::FloatCheckResult> device_results,
      CheckNotNull(executor_->AllocateArray<xla::gpu::FloatCheckResult>(
                       kIntermediateResults),
                   "results"));
  auto cleanup_results =
      absl::MakeCleanup([&]() { executor_->Deallocate(&device_results); });

  ASSERT_OK(stream_->Memcpy(&device_results, results.data(),
                            results.size() * sizeof(results[0])));
  ASSERT_OK(reduce_kernel.Launch(
      se::ThreadDim(1024, 1, 1), se::BlockDim(1, 1, 1), stream_.get(),
      device_results, device_results.ElementCount(), BufferDebugLogEntryId{0},
      device_log.GetDeviceHeader(), device_log.GetDeviceEntries()));
  TF_ASSERT_OK_AND_ASSIGN(auto host_log, device_log.ReadFromDevice(*stream_));

  ASSERT_GE(host_log.size(), 1);
  EXPECT_EQ(host_log[0].nan_count, kNumNaNs);
  EXPECT_EQ(host_log[0].inf_count, kNumInfs);
  EXPECT_EQ(host_log[0].zero_count, kNumZeros);
}

}  // namespace
}  // namespace stream_executor::cuda
