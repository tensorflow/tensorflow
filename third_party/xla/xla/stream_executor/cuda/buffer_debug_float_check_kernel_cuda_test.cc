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
#include "xla/backends/gpu/runtime/buffer_debug_log_structs_test_matchers.h"
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
#include "xla/stream_executor/stream_executor_address_allocator.h"
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
using xla::gpu::InfCountIs;
using xla::gpu::MaxValueIs;
using xla::gpu::MinValueIs;
using xla::gpu::NanCountIs;
using xla::gpu::ThunkId;
using xla::gpu::ZeroCountIs;

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

  template <typename CheckKernel, typename InputType, typename BufferType>
  absl::Status AppendFloatCheckOnDevice(
      BufferDebugLogEntryId entry_id, const std::vector<InputType>& input,
      se::gpu::BufferDebugLog<BufferType>& buffer_debug_log,
      stream_executor::BlockDim block_dim = stream_executor::BlockDim(1, 1, 1),
      size_t temp_buffer_size_elements = 1024) {
    // Load kernel
    gpu::GpuKernelRegistry registry =
        gpu::GpuKernelRegistry::GetGlobalRegistry();
    TF_ASSIGN_OR_RETURN(auto check_kernel,
                        registry.LoadKernel<CheckKernel>(executor_));
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
    TF_RETURN_IF_ERROR(check_kernel.Launch(
        thread_dim, block_dim, stream_.get(), device_input,
        device_input.ElementCount(), device_tmp, device_tmp.ElementCount()));
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

template <typename TInput, typename TCheckKernel>
struct TestConfig {
  using InputType = TInput;
  using CheckKernel = TCheckKernel;
};

using FloatTypes = ::testing::Types<
    TestConfig<float, gpu::BufferDebugFloatCheckF32Kernel>,
    TestConfig<Eigen::bfloat16, gpu::BufferDebugFloatCheckBf16Kernel>,
    TestConfig<double, gpu::BufferDebugFloatCheckF64Kernel>>;

template <typename T>
class FloatCheckKernelTypedTest : public FloatCheckKernelTest {};
TYPED_TEST_SUITE(FloatCheckKernelTypedTest, FloatTypes);

TYPED_TEST(FloatCheckKernelTypedTest, ChecksFloats) {
  using InputType = typename TypeParam::InputType;
  using CheckKernel = typename TypeParam::CheckKernel;

  std::vector<InputType> input(1024, InputType(1.0));
  input[10] = std::numeric_limits<InputType>::quiet_NaN();
  input[20] = std::numeric_limits<InputType>::quiet_NaN();
  input[30] = InputType(0.0);
  input[40] = std::numeric_limits<InputType>::infinity();
  input[50] = std::numeric_limits<InputType>::infinity();
  input[60] = std::numeric_limits<InputType>::infinity();
  input[70] = InputType(-2.0);
  input[80] = InputType(3.0);

  se::DeviceAddress<uint8_t> mem =
      this->executor_->template AllocateArray<uint8_t>(1024);
  TF_ASSERT_OK_AND_ASSIGN(
      auto device_log,
      se::gpu::BufferDebugLog<BufferDebugFloatCheckEntry>::CreateOnDevice(
          *this->stream_, mem));

  TF_EXPECT_OK((this->template AppendFloatCheckOnDevice<CheckKernel>(
      BufferDebugLogEntryId{0}, input, device_log)));

  TF_ASSERT_OK_AND_ASSIGN(auto host_log,
                          device_log.ReadFromDevice(*this->stream_));
  ASSERT_GE(host_log.size(), 1);
  EXPECT_THAT(host_log[0], NanCountIs(2));
  EXPECT_THAT(host_log[0], InfCountIs(3));
  EXPECT_THAT(host_log[0], ZeroCountIs(1));
  EXPECT_THAT(host_log[0], MinValueIs(InputType(-2.0)));
  EXPECT_THAT(host_log[0], MaxValueIs(InputType(3.0)));
}

TYPED_TEST(FloatCheckKernelTypedTest, ChecksFloatsWithNoFiniteValues) {
  using InputType = typename TypeParam::InputType;
  using CheckKernel = typename TypeParam::CheckKernel;

  std::vector<InputType> input(1024,
                               std::numeric_limits<InputType>::quiet_NaN());
  input[10] = std::numeric_limits<InputType>::infinity();
  input[20] = -std::numeric_limits<InputType>::infinity();

  se::DeviceAddress<uint8_t> mem =
      this->executor_->template AllocateArray<uint8_t>(1024);
  TF_ASSERT_OK_AND_ASSIGN(
      auto device_log,
      se::gpu::BufferDebugLog<BufferDebugFloatCheckEntry>::CreateOnDevice(
          *this->stream_, mem));

  TF_EXPECT_OK((this->template AppendFloatCheckOnDevice<CheckKernel>(
      BufferDebugLogEntryId{0}, input, device_log)));

  TF_ASSERT_OK_AND_ASSIGN(auto host_log,
                          device_log.ReadFromDevice(*this->stream_));
  ASSERT_GE(host_log.size(), 1);
  // If the input buffer contains no finite values, the min/max values are
  // undefined. The implementation returns +/- infinity for min/max in this
  // case.
  //
  // nan_count and inf_count can be used to determine how many finite values
  // were in the input buffer.
  EXPECT_EQ(host_log[0].result.nan_count + host_log[0].result.inf_count,
            input.size());
  EXPECT_THAT(host_log[0],
              MinValueIs(std::numeric_limits<InputType>::infinity()));
  EXPECT_THAT(host_log[0],
              MaxValueIs(-std::numeric_limits<InputType>::infinity()));
}

TYPED_TEST(FloatCheckKernelTypedTest, ChecksFloatsInParallel) {
  using InputType = typename TypeParam::InputType;
  using CheckKernel = typename TypeParam::CheckKernel;

  static constexpr size_t kNumNaNs = 100;
  static constexpr size_t kNumInfs = 200;
  static constexpr size_t kNumZeros = 300;
  static constexpr size_t kMaxTestValues =
      std::max(std::max(kNumNaNs, kNumInfs), kNumZeros);

  const se::DeviceDescription& device_desc =
      this->executor_->GetDeviceDescription();
  const size_t threads_per_core = device_desc.threads_per_core_limit();
  const size_t num_cores = device_desc.core_count();
  const size_t input_size = num_cores * threads_per_core * 3 / 2;
  const size_t test_value_stride = input_size / (kMaxTestValues + 1);
  ASSERT_GT(input_size, kMaxTestValues);
  ASSERT_GT(test_value_stride, 2);

  std::vector<InputType> input(input_size, InputType(1.0));
  for (size_t i = 0; i < kNumNaNs; ++i) {
    input[i * test_value_stride] = std::numeric_limits<InputType>::quiet_NaN();
  }
  for (size_t i = 0; i < kNumInfs; ++i) {
    input[i * test_value_stride + 1] =
        std::numeric_limits<InputType>::infinity();
  }
  for (size_t i = 0; i < kNumZeros; ++i) {
    input[i * test_value_stride + 2] = InputType(0.0);
  }

  se::DeviceAddress<uint8_t> log_mem =
      this->executor_->template AllocateArray<uint8_t>(1024);
  TF_ASSERT_OK_AND_ASSIGN(
      auto device_log,
      se::gpu::BufferDebugLog<BufferDebugFloatCheckEntry>::CreateOnDevice(
          *this->stream_, log_mem));

  int64_t threads_per_block;
  int64_t num_blocks;
  CalculateDimensionality(this->executor_->GetDeviceDescription(), input.size(),
                          &threads_per_block, &num_blocks);
  const se::BlockDim block_dim(num_blocks);
  TF_EXPECT_OK((this->template AppendFloatCheckOnDevice<CheckKernel>(
      BufferDebugLogEntryId{0}, input, device_log, block_dim)));
  TF_EXPECT_OK((this->template AppendFloatCheckOnDevice<CheckKernel>(
      BufferDebugLogEntryId{0}, input, device_log, block_dim)));

  TF_ASSERT_OK_AND_ASSIGN(auto host_log,
                          device_log.ReadFromDevice(*this->stream_));
  ASSERT_GE(host_log.size(), 2);
  EXPECT_THAT(host_log[0], NanCountIs(kNumNaNs));
  EXPECT_THAT(host_log[0], InfCountIs(kNumInfs));
  EXPECT_THAT(host_log[0], ZeroCountIs(kNumZeros));
  EXPECT_THAT(host_log[0], MinValueIs(InputType(0.0)));
  EXPECT_THAT(host_log[0], MaxValueIs(InputType(1.0)));
  EXPECT_THAT(host_log[1], NanCountIs(kNumNaNs));
  EXPECT_THAT(host_log[1], InfCountIs(kNumInfs));
  EXPECT_THAT(host_log[1], ZeroCountIs(kNumZeros));
  EXPECT_THAT(host_log[1], MinValueIs(InputType(0.0)));
  EXPECT_THAT(host_log[1], MaxValueIs(InputType(1.0)));
}

TYPED_TEST(FloatCheckKernelTypedTest, ReduceFloatCheckResults) {
  using InputType = typename TypeParam::InputType;

  static constexpr size_t kNumNaNs = 100;
  static constexpr size_t kNumInfs = 200;
  static constexpr size_t kNumZeros = 300;
  static constexpr size_t kIntermediateResults = 16 * 1024;

  std::vector<xla::gpu::FloatCheckResult> results(kIntermediateResults);
  for (size_t i = 0; i < kIntermediateResults; ++i) {
    results[i].nan_count = i < kNumNaNs ? 1 : 0;
    results[i].inf_count = i < kNumInfs ? 1 : 0;
    results[i].zero_count = i < kNumZeros ? 1 : 0;
    results[i].min_value = 1.0;
    results[i].max_value = 2.0;
  }
  results[10].min_value = -1.0;
  results[20].max_value = 10.0;

  gpu::GpuKernelRegistry registry = gpu::GpuKernelRegistry::GetGlobalRegistry();
  TF_ASSERT_OK_AND_ASSIGN(
      auto reduce_kernel,
      registry.template LoadKernel<
          gpu::BufferDebugAppendReducedFloatCheckResultsKernel>(
          this->executor_));

  se::DeviceAddress<uint8_t> log_mem =
      this->executor_->template AllocateArray<uint8_t>(1024);
  TF_ASSERT_OK_AND_ASSIGN(
      auto device_log,
      se::gpu::BufferDebugLog<BufferDebugFloatCheckEntry>::CreateOnDevice(
          *this->stream_, log_mem));
  TF_ASSERT_OK_AND_ASSIGN(
      se::DeviceAddress<xla::gpu::FloatCheckResult> device_results,
      this->CheckNotNull(
          this->executor_->template AllocateArray<xla::gpu::FloatCheckResult>(
              kIntermediateResults),
          "results"));
  auto cleanup_results = absl::MakeCleanup(
      [&]() { this->executor_->Deallocate(&device_results); });

  TF_ASSERT_OK(this->stream_->Memcpy(&device_results, results.data(),
                                     results.size() * sizeof(results[0])));
  TF_ASSERT_OK(reduce_kernel.Launch(
      se::ThreadDim(1024, 1, 1), se::BlockDim(1, 1, 1), this->stream_.get(),
      device_results, device_results.ElementCount(), BufferDebugLogEntryId{0},
      device_log.GetDeviceHeader(), device_log.GetDeviceEntries()));
  TF_ASSERT_OK_AND_ASSIGN(auto host_log,
                          device_log.ReadFromDevice(*this->stream_));

  ASSERT_GE(host_log.size(), 1);
  EXPECT_THAT(host_log[0], NanCountIs(kNumNaNs));
  EXPECT_THAT(host_log[0], InfCountIs(kNumInfs));
  EXPECT_THAT(host_log[0], ZeroCountIs(kNumZeros));
  EXPECT_THAT(host_log[0], MinValueIs(InputType(-1.0)));
  EXPECT_THAT(host_log[0], MaxValueIs(InputType(10.0)));
}

}  // namespace
}  // namespace stream_executor::cuda
