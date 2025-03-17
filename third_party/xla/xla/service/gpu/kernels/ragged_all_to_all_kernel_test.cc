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

#include "xla/service/gpu/kernels/ragged_all_to_all_kernel.h"

#include <cstdint>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_handle.h"
#include "xla/stream_executor/gpu/gpu_init.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

se::StreamExecutor* GetGpuExecutor() {
  auto* platform =
      se::PlatformManager::PlatformWithName(se::GpuPlatformName()).value();
  return platform->ExecutorForDevice(0).value();
}

template <typename T>
std::vector<std::vector<T>> GetExpectedOutputResults(
    absl::Span<const T> input_data, absl::Span<const int64_t> input_offsets,
    absl::Span<const int64_t> send_sizes,
    absl::Span<const int64_t> output_offsets, int64_t num_ranks,
    int64_t num_updates_per_rank, int64_t num_input_rows,
    int64_t num_row_elements) {
  std::vector<std::vector<T>> expected_output(
      num_ranks, std::vector<T>(num_input_rows * num_row_elements, 0));

  for (int64_t i = 0; i < num_ranks; ++i) {
    for (int64_t j = 0; j < num_updates_per_rank; ++j) {
      int64_t update_idx = i * num_updates_per_rank + j;
      int64_t input_offset = input_offsets[update_idx];
      int64_t send_size = send_sizes[update_idx];
      int64_t output_offset = output_offsets[update_idx];

      for (int k = 0; k < send_size * num_row_elements; ++k) {
        expected_output[i][output_offset * num_row_elements + k] =
            input_data[input_offset * num_row_elements + k];
      }
    }
  }
  return expected_output;
}

using RaggedAllToAllKernelTest = ::testing::Test;

TEST_F(RaggedAllToAllKernelTest, SimpleKernelTest) {
  using T = float;

  auto* executor = GetGpuExecutor();
  auto stream = executor->CreateStream().value();

  constexpr int64_t num_outputs = 2;
  constexpr int64_t num_update_per_output = 2;
  constexpr int64_t num_input_rows = 8;
  constexpr int64_t num_row_elements = 2;
  constexpr int64_t n = num_input_rows * num_row_elements;

  stream_executor::DeviceMemoryHandle input_buffer(
      executor, executor->AllocateArray<T>(n));

  std::vector<stream_executor::DeviceMemoryHandle> output_buffers;
  for (int64_t i = 0; i < num_outputs; ++i) {
    output_buffers.emplace_back(executor, executor->AllocateArray<T>(n));
    ASSERT_TRUE(!output_buffers[i].memory().is_null());
  }

  stream_executor::DeviceMemoryHandle input_offsets_buffer(
      executor,
      executor->AllocateArray<int64_t>(num_outputs * num_update_per_output));
  stream_executor::DeviceMemoryHandle send_sizes_buffer(
      executor,
      executor->AllocateArray<int64_t>(num_outputs * num_update_per_output));
  stream_executor::DeviceMemoryHandle output_offsets_buffer(
      executor,
      executor->AllocateArray<int64_t>(num_outputs * num_update_per_output));

  ASSERT_TRUE(!(input_offsets_buffer.memory().is_null() ||
                input_offsets_buffer.memory().is_null() ||
                output_offsets_buffer.memory().is_null()));

  std::vector<T> input_data(n);
  std::iota(input_data.begin(), input_data.end(), 0);
  TF_ASSERT_OK(stream->Memcpy(input_buffer.memory_ptr(), input_data.data(),
                              n * sizeof(T)));

  std::vector<int64_t> input_offsets = {1, 4, 0, 3};
  std::vector<int64_t> send_sizes = {2, 3, 1, 2};
  std::vector<int64_t> output_offsets = {0, 4, 1, 5};

  TF_ASSERT_OK(stream->Memcpy(input_offsets_buffer.memory_ptr(),
                              input_offsets.data(),
                              input_offsets.size() * sizeof(int64_t)));
  TF_ASSERT_OK(stream->Memcpy(send_sizes_buffer.memory_ptr(), send_sizes.data(),
                              send_sizes.size() * sizeof(int64_t)));
  TF_ASSERT_OK(stream->Memcpy(output_offsets_buffer.memory_ptr(),
                              output_offsets.data(),
                              output_offsets.size() * sizeof(int64_t)));

  std::vector<se::DeviceMemoryBase> output_buffers_span;
  for (auto& output_buffer : output_buffers) {
    output_buffers_span.push_back(output_buffer.memory());
  }

  TF_ASSERT_OK(RunRaggedAllToAllKernel(
      stream.get(), primitive_util::NativeToPrimitiveType<T>(),
      input_buffer.memory(), output_buffers_span, input_offsets_buffer.memory(),
      send_sizes_buffer.memory(), output_offsets_buffer.memory(), num_outputs,
      num_update_per_output, num_input_rows, num_row_elements));

  std::vector<std::vector<T>> output_results(num_outputs);

  for (int64_t i = 0; i < num_outputs; ++i) {
    output_results[i].resize(n);
    TF_ASSERT_OK(stream->Memcpy(output_results[i].data(),
                                output_buffers[i].memory(), n * sizeof(T)));
  }

  std::vector<std::vector<T>> expected_output_results =
      GetExpectedOutputResults<T>(
          input_data, input_offsets, send_sizes, output_offsets, num_outputs,
          num_update_per_output, num_input_rows, num_row_elements);

  ASSERT_EQ(output_results.size(), expected_output_results.size());
  EXPECT_EQ(output_results, expected_output_results);
}

}  // namespace
}  // namespace xla::gpu
