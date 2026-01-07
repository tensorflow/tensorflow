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

#include <gtest/gtest.h>
#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
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

  template <typename InputType, typename BufferType>
  absl::Status AppendFloatCheckOnDevice(
      const BufferDebugLogEntryId entry_id, const std::vector<InputType>& input,
      se::gpu::BufferDebugLog<BufferType>& buffer_debug_log) {
    return AppendFloatChecksOnDevice<InputType, BufferType>(
        absl::MakeConstSpan(&entry_id, 1), absl::MakeConstSpan(&input, 1),
        buffer_debug_log);
  }

  template <typename InputType, typename BufferType>
  absl::Status AppendFloatChecksOnDevice(
      absl::Span<const BufferDebugLogEntryId> entry_ids,
      absl::Span<const std::vector<InputType>> inputs,
      se::gpu::BufferDebugLog<BufferType>& buffer_debug_log) {
    EXPECT_EQ(inputs.size(), entry_ids.size());

    // Load kernel
    gpu::GpuKernelRegistry registry =
        gpu::GpuKernelRegistry::GetGlobalRegistry();
    TF_ASSIGN_OR_RETURN(
        auto append_kernel,
        registry.LoadKernel<gpu::BufferDebugAppendFloatCheckResultsKernel>(
            executor_));

    TF_ASSIGN_OR_RETURN(
        se::DeviceAddress<xla::gpu::FloatCheckResult> device_results,
        CheckNotNull(
            executor_->AllocateArray<xla::gpu::FloatCheckResult>(inputs.size()),
            "results"));
    std::vector<se::DeviceAddressBase> temp_allocs;
    temp_allocs.push_back(device_results);
    auto cleanup_results = absl::MakeCleanup([&]() {
      for (auto alloc : temp_allocs) {
        executor_->Deallocate(&alloc);
      }
    });

    for (size_t i = 0; i < inputs.size(); ++i) {
      const auto& input = inputs[i];
      se::DeviceAddress<xla::gpu::FloatCheckResult> device_result =
          device_results.GetSlice(i, 1);

      // Setup device buffers
      TF_ASSIGN_OR_RETURN(
          se::DeviceAddress<InputType> device_input,
          CheckNotNull(executor_->AllocateArray<InputType>(input.size()),
                       "input"));
      temp_allocs.push_back(device_input);

      // Call kernel
      TF_RETURN_IF_ERROR(stream_->Memcpy(&device_input, input.data(),
                                         input.size() * sizeof(input[0])));
      TF_RETURN_IF_ERROR(se::gpu::CheckFloats<InputType>(
          device_input, device_result, stream_.get()));
    }

    TF_ASSIGN_OR_RETURN(
        se::DeviceAddress<xla::gpu::BufferDebugLogEntryId> device_ids,
        CheckNotNull(executor_->AllocateArray<xla::gpu::BufferDebugLogEntryId>(
                         entry_ids.size()),
                     "ids"));
    temp_allocs.push_back(device_ids);
    TF_RETURN_IF_ERROR(
        stream_->Memcpy(&device_ids, entry_ids.data(),
                        entry_ids.size() * sizeof(entry_ids[0])));

    TF_RETURN_IF_ERROR(
        append_kernel.Launch(se::ThreadDim(1, 1, 1), se::BlockDim(1, 1, 1),
                             stream_.get(), device_results, device_ids,
                             inputs.size(), buffer_debug_log.GetDeviceHeader(),
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

  TF_EXPECT_OK(
      AppendFloatCheckOnDevice(BufferDebugLogEntryId{123}, input, device_log));

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

  TF_EXPECT_OK(
      AppendFloatCheckOnDevice(BufferDebugLogEntryId{0}, input, device_log));

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

  TF_EXPECT_OK(
      AppendFloatCheckOnDevice(BufferDebugLogEntryId{0}, input, device_log));
  TF_EXPECT_OK(
      AppendFloatCheckOnDevice(BufferDebugLogEntryId{0}, input, device_log));

  TF_ASSERT_OK_AND_ASSIGN(auto host_log, device_log.ReadFromDevice(*stream_));
  ASSERT_GE(host_log.size(), 2);
  EXPECT_EQ(host_log[0].nan_count, kNumNaNs);
  EXPECT_EQ(host_log[0].inf_count, kNumInfs);
  EXPECT_EQ(host_log[0].zero_count, kNumZeros);
  EXPECT_EQ(host_log[1].nan_count, kNumNaNs);
  EXPECT_EQ(host_log[1].inf_count, kNumInfs);
  EXPECT_EQ(host_log[1].zero_count, kNumZeros);
}

TEST_F(FloatCheckKernelTest, AppendsMultipleDistinctBuffersToLog) {
  std::vector<float> input1 = {1.0f, std::numeric_limits<float>::quiet_NaN()};
  std::vector<float> input2 = {std::numeric_limits<float>::infinity(), 0.0f,
                               std::numeric_limits<float>::infinity()};

  se::DeviceAddress<uint8_t> log_mem = executor_->AllocateArray<uint8_t>(1024);
  TF_ASSERT_OK_AND_ASSIGN(
      auto device_log,
      se::gpu::BufferDebugLog<BufferDebugFloatCheckEntry>::CreateOnDevice(
          *stream_, log_mem));

  TF_EXPECT_OK(
      AppendFloatCheckOnDevice(BufferDebugLogEntryId{101}, input1, device_log));
  TF_EXPECT_OK(
      AppendFloatCheckOnDevice(BufferDebugLogEntryId{102}, input2, device_log));

  TF_ASSERT_OK_AND_ASSIGN(auto host_log, device_log.ReadFromDevice(*stream_));
  ASSERT_GE(host_log.size(), 2);

  EXPECT_EQ(host_log[0].entry_id, 101);
  EXPECT_EQ(host_log[0].nan_count, 1);
  EXPECT_EQ(host_log[0].inf_count, 0);
  EXPECT_EQ(host_log[0].zero_count, 0);

  EXPECT_EQ(host_log[1].entry_id, 102);
  EXPECT_EQ(host_log[1].nan_count, 0);
  EXPECT_EQ(host_log[1].inf_count, 2);
  EXPECT_EQ(host_log[1].zero_count, 1);
}

TEST_F(FloatCheckKernelTest, AppendsBatchedFloatChecksToLog) {
  std::vector<float> input1 = {1.0f, std::numeric_limits<float>::quiet_NaN()};
  std::vector<float> input2 = {std::numeric_limits<float>::infinity(), 0.0f,
                               std::numeric_limits<float>::infinity()};
  std::vector<std::vector<float>> inputs = {input1, input2};
  std::vector<BufferDebugLogEntryId> ids = {BufferDebugLogEntryId{201},
                                            BufferDebugLogEntryId{202}};

  se::DeviceAddress<uint8_t> log_mem = executor_->AllocateArray<uint8_t>(1024);
  TF_ASSERT_OK_AND_ASSIGN(
      auto device_log,
      se::gpu::BufferDebugLog<BufferDebugFloatCheckEntry>::CreateOnDevice(
          *stream_, log_mem));

  TF_EXPECT_OK((AppendFloatChecksOnDevice<float, BufferDebugFloatCheckEntry>(
      ids, inputs, device_log)));

  TF_ASSERT_OK_AND_ASSIGN(auto host_log, device_log.ReadFromDevice(*stream_));
  ASSERT_GE(host_log.size(), 2);

  EXPECT_EQ(host_log[0].entry_id, 201);
  EXPECT_EQ(host_log[0].nan_count, 1);

  EXPECT_EQ(host_log[1].entry_id, 202);
  EXPECT_EQ(host_log[1].inf_count, 2);
}

TEST_F(FloatCheckKernelTest, AppendFloatCheckResult) {
  const xla::gpu::FloatCheckResult result{
      /*nan_count=*/111,
      /*inf_count=*/222,
      /*zero_count=*/333,
  };
  static constexpr size_t kCheckedBuffersCount = 1;

  gpu::GpuKernelRegistry registry = gpu::GpuKernelRegistry::GetGlobalRegistry();
  TF_ASSERT_OK_AND_ASSIGN(
      auto append_kernel,
      registry.LoadKernel<gpu::BufferDebugAppendFloatCheckResultsKernel>(
          executor_));

  se::DeviceAddress<uint8_t> log_mem = executor_->AllocateArray<uint8_t>(1024);
  TF_ASSERT_OK_AND_ASSIGN(
      auto device_log,
      se::gpu::BufferDebugLog<BufferDebugFloatCheckEntry>::CreateOnDevice(
          *stream_, log_mem));
  TF_ASSERT_OK_AND_ASSIGN(
      se::DeviceAddress<xla::gpu::FloatCheckResult> device_results,
      CheckNotNull(executor_->AllocateArray<xla::gpu::FloatCheckResult>(
                       kCheckedBuffersCount),
                   "results"));
  TF_ASSERT_OK_AND_ASSIGN(
      se::DeviceAddress<xla::gpu::BufferDebugLogEntryId> device_ids,
      CheckNotNull(executor_->AllocateArray<xla::gpu::BufferDebugLogEntryId>(
                       kCheckedBuffersCount),
                   "ids"));
  auto cleanup_results =
      absl::MakeCleanup([&]() { executor_->Deallocate(&device_results); });

  TF_ASSERT_OK(stream_->Memcpy(&device_results, &result, sizeof(result)));
  TF_ASSERT_OK(append_kernel.Launch(
      se::ThreadDim(1, 1, 1), se::BlockDim(1, 1, 1), stream_.get(),
      device_results, device_ids, kCheckedBuffersCount,
      device_log.GetDeviceHeader(), device_log.GetDeviceEntries()));
  TF_ASSERT_OK_AND_ASSIGN(auto host_log, device_log.ReadFromDevice(*stream_));

  ASSERT_GE(host_log.size(), 1);
  EXPECT_EQ(host_log[0].nan_count, result.nan_count);
  EXPECT_EQ(host_log[0].inf_count, result.inf_count);
  EXPECT_EQ(host_log[0].zero_count, result.zero_count);
}

}  // namespace
}  // namespace stream_executor::cuda
