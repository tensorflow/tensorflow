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

#include "xla/backends/gpu/runtime/ragged_all_to_all.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "third_party/nccl/nccl.h"
#include "xla/backends/gpu/collectives/nccl_symmetric_memory.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_handle.h"
#include "xla/stream_executor/gpu/gpu_init.h"
#include "xla/stream_executor/gpu/ragged_all_to_all_kernel.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"
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

template <typename T>
stream_executor::DeviceAddressHandle CreateDeviceBuffer(
    se::StreamExecutor* executor, const std::vector<T>& data) {
  stream_executor::DeviceAddressHandle device_buffer(
      executor, executor->AllocateArray<int64_t>(data.size()));
  CHECK(!device_buffer.address().is_null());
  CHECK_OK(executor->SynchronousMemcpy(device_buffer.address_ptr(), data.data(),
                                       data.size() * sizeof(T)));
  return device_buffer;
}

template <typename T>
std::vector<std::vector<T>> CopyDeviceToHost2D(
    se::StreamExecutor* executor,
    const std::vector<stream_executor::DeviceAddressHandle>& device_buffers,
    int64_t num_elements) {
  std::vector<std::vector<T>> host_buffers;
  host_buffers.reserve(device_buffers.size());
  for (auto& device_buffer : device_buffers) {
    std::vector<T> host_buffer(num_elements);
    CHECK_OK(executor->SynchronousMemcpy(
        host_buffer.data(), device_buffer.address(), num_elements * sizeof(T)));
    host_buffers.push_back(std::move(host_buffer));
  }
  return host_buffers;
}

template <typename T>
std::vector<std::vector<T>> CopyDeviceToHost2D(
    const std::vector<std::unique_ptr<se::Stream>>& streams,
    const std::vector<std::unique_ptr<se::MemoryAllocation>>& device_buffers,
    size_t offset_bytes, int64_t num_elements) {
  const size_t copy_size = num_elements * sizeof(T);
  std::vector<std::vector<T>> host_buffers;
  host_buffers.reserve(device_buffers.size());
  CHECK_EQ(streams.size(), device_buffers.size());
  for (int i = 0; i < device_buffers.size(); ++i) {
    se::DeviceAddressBase full_buffer = device_buffers[i]->address();
    se::DeviceAddressBase sliced_address =
        full_buffer.GetByteSlice(offset_bytes, copy_size);
    std::vector<T> host_buffer(num_elements);
    CHECK_OK(streams[i]->Memcpy(host_buffer.data(), sliced_address, copy_size));
    CHECK_OK(streams[i]->BlockHostUntilDone());
    host_buffers.push_back(std::move(host_buffer));
  }
  return host_buffers;
}

absl::StatusOr<std::vector<std::unique_ptr<xla::SymmetricMemory>>>
CreateSymmetricMemory(
    tsl::Executor& exec, const std::vector<ncclComm_t>& comms,
    const std::vector<std::unique_ptr<se::MemoryAllocation>>& buffers) {
  int64_t num_devices = comms.size();
  std::vector<tsl::Future<std::unique_ptr<NcclSymmetricMemory>>>
      symmetric_memory_futures(num_devices);
  for (int i = 0; i < num_devices; ++i) {
    symmetric_memory_futures[i] = tsl::MakeFutureOn(exec, [&, i]() {
      return NcclSymmetricMemory::Create(comms[i], buffers[i]->address());
    });
  }

  std::vector<std::unique_ptr<xla::SymmetricMemory>> symmetric_memory;
  for (int i = 0; i < num_devices; ++i) {
    ASSIGN_OR_RETURN(auto mem, std::move(symmetric_memory_futures[i]).Await());
    symmetric_memory.push_back(std::move(mem));
  }
  return symmetric_memory;
}

using RaggedAllToAllKernelTest = ::testing::Test;

TEST_F(RaggedAllToAllKernelTest, KernelWithArrayOfOutputPointers) {
  using T = float;

  auto* executor = GetGpuExecutor();
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  constexpr int64_t num_outputs = 2;
  constexpr int64_t num_update_per_output = 2;
  constexpr int64_t num_input_rows = 8;
  constexpr int64_t num_row_elements = 2;
  constexpr int64_t n = num_input_rows * num_row_elements;

  stream_executor::DeviceAddressHandle input_buffer(
      executor, executor->AllocateArray<T>(n));

  std::vector<stream_executor::DeviceAddressHandle> output_buffers;
  for (int64_t i = 0; i < num_outputs; ++i) {
    output_buffers.emplace_back(executor, executor->AllocateArray<T>(n));
    ASSERT_TRUE(!output_buffers[i].address().is_null());
    TF_ASSERT_OK(
        stream->MemZero(output_buffers[i].address_ptr(), n * sizeof(T)));
  }

  std::vector<T> input_data(n);
  absl::c_iota(input_data, 0);
  TF_ASSERT_OK(stream->Memcpy(input_buffer.address_ptr(), input_data.data(),
                              n * sizeof(T)));

  std::vector<int64_t> input_offsets = {1, 4, 0, 3};
  std::vector<int64_t> send_sizes = {2, 3, 1, 2};
  std::vector<int64_t> output_offsets = {0, 4, 1, 5};

  stream_executor::DeviceAddressHandle input_offsets_buffer =
      CreateDeviceBuffer(executor, input_offsets);
  stream_executor::DeviceAddressHandle send_sizes_buffer =
      CreateDeviceBuffer(executor, send_sizes);
  stream_executor::DeviceAddressHandle output_offsets_buffer =
      CreateDeviceBuffer(executor, output_offsets);

  stream_executor::gpu::RaggedAllToAllOutputPtrs output_buffers_array;
  for (int64_t i = 0; i < num_outputs; ++i) {
    output_buffers_array[i] = output_buffers[i].address().opaque();
  }

  TF_ASSERT_OK(RunRaggedAllToAllKernel(
      stream.get(), primitive_util::NativeToPrimitiveType<T>(),
      input_buffer.address(), output_buffers_array,
      input_offsets_buffer.address(), send_sizes_buffer.address(),
      output_offsets_buffer.address(), num_outputs, num_update_per_output,
      num_input_rows, num_row_elements));

  std::vector<std::vector<T>> output_results =
      CopyDeviceToHost2D<T>(executor, output_buffers, n);

  std::vector<std::vector<T>> expected_output_results =
      GetExpectedOutputResults<T>(
          input_data, input_offsets, send_sizes, output_offsets, num_outputs,
          num_update_per_output, num_input_rows, num_row_elements);

  ASSERT_EQ(output_results.size(), expected_output_results.size());
  EXPECT_EQ(output_results, expected_output_results);
}

TEST_F(RaggedAllToAllKernelTest, KernelWithOutputPtrsInDeviceMemory) {
  using T = float;

  auto* executor = GetGpuExecutor();
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  constexpr int64_t num_outputs = 2;
  constexpr int64_t num_update_per_output = 2;
  constexpr int64_t num_input_rows = 8;
  constexpr int64_t num_row_elements = 2;
  constexpr int64_t n = num_input_rows * num_row_elements;

  stream_executor::DeviceAddressHandle input_buffer(
      executor, executor->AllocateArray<T>(n));

  std::vector<stream_executor::DeviceAddressHandle> output_buffers;
  for (int64_t i = 0; i < num_outputs; ++i) {
    output_buffers.emplace_back(executor, executor->AllocateArray<T>(n));
    ASSERT_TRUE(!output_buffers[i].address().is_null());
    TF_ASSERT_OK(
        stream->MemZero(output_buffers[i].address_ptr(), n * sizeof(T)));
  }

  std::vector<T> input_data(n);
  absl::c_iota(input_data, 0);
  TF_ASSERT_OK(stream->Memcpy(input_buffer.address_ptr(), input_data.data(),
                              n * sizeof(T)));

  std::vector<int64_t> input_offsets = {1, 4, 0, 3};
  std::vector<int64_t> send_sizes = {2, 3, 1, 2};
  std::vector<int64_t> output_offsets = {0, 4, 1, 5};

  stream_executor::DeviceAddressHandle input_offsets_buffer =
      CreateDeviceBuffer(executor, input_offsets);
  stream_executor::DeviceAddressHandle send_sizes_buffer =
      CreateDeviceBuffer(executor, send_sizes);
  stream_executor::DeviceAddressHandle output_offsets_buffer =
      CreateDeviceBuffer(executor, output_offsets);

  std::vector<void*> output_buffers_span;
  for (auto& output_buffer : output_buffers) {
    output_buffers_span.push_back(output_buffer.address().opaque());
  }

  stream_executor::DeviceAddressHandle output_buffers_ptr_buffer =
      CreateDeviceBuffer(executor, output_buffers_span);

  TF_ASSERT_OK(RunRaggedAllToAllKernel(
      stream.get(), primitive_util::NativeToPrimitiveType<T>(),
      input_buffer.address(), output_buffers_ptr_buffer.address(),
      input_offsets_buffer.address(), send_sizes_buffer.address(),
      output_offsets_buffer.address(), num_outputs, num_update_per_output,
      num_input_rows, num_row_elements));

  std::vector<std::vector<T>> output_results =
      CopyDeviceToHost2D<T>(executor, output_buffers, n);

  std::vector<std::vector<T>> expected_output_results =
      GetExpectedOutputResults<T>(
          input_data, input_offsets, send_sizes, output_offsets, num_outputs,
          num_update_per_output, num_input_rows, num_row_elements);

  ASSERT_EQ(output_results.size(), expected_output_results.size());
  EXPECT_EQ(output_results, expected_output_results);
}

TEST_F(RaggedAllToAllKernelTest, KernelWithSymmetricMemory) {
  using T = float;

  constexpr int64_t num_outputs = 2;
  constexpr int64_t num_update_per_output = 2;
  constexpr int64_t num_input_rows = 8;
  constexpr int64_t num_row_elements = 2;
  constexpr size_t output_sym_offset = 1024;  // 1KB padding for output buffers
  constexpr int64_t n = num_input_rows * num_row_elements;

  ASSERT_OK_AND_ASSIGN(
      se::Platform * platform,
      se::PlatformManager::PlatformWithId(se::cuda::kCudaPlatformId));
  int visible_device_count =
      std::min<int>(platform->VisibleDeviceCount(), num_outputs);

  if (visible_device_count < num_outputs) {
    GTEST_SKIP() << "Skipping test because there are not enough visible "
                    "devices.";
  }

  std::vector<se::StreamExecutor*> executors(visible_device_count);
  std::vector<std::unique_ptr<se::Stream>> streams(visible_device_count);

  for (int i = 0; i < visible_device_count; ++i) {
    ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor,
                         platform->ExecutorForDevice(i));
    executors[i] = executor;

    ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
    streams[i] = std::move(stream);
  }

  if (!executors[0]
           ->GetDeviceDescription()
           .cuda_compute_capability()
           .IsAtLeastHopper()) {
    GTEST_SKIP() << "Test requires at least Hopper architecture";
  }

  std::vector<std::unique_ptr<se::MemoryAllocator>> collective_allocators;
  for (int i = 0; i < visible_device_count; ++i) {
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<se::MemoryAllocator> allocator,
        executors[i]->CreateMemoryAllocator(se::MemorySpace::kCollective));
    collective_allocators.push_back(std::move(allocator));
  }

  stream_executor::DeviceAddressHandle input_buffer(
      executors[0], executors[0]->AllocateArray<T>(n));

  std::vector<std::unique_ptr<se::MemoryAllocation>> output_buffers;
  for (int64_t i = 0; i < num_outputs; ++i) {
    size_t total_bytes = output_sym_offset + (n * sizeof(T));
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::MemoryAllocation> output_buffer,
                         collective_allocators[i]->Allocate(total_bytes));
    se::DeviceAddressBase output_buffer_address = output_buffer->address();
    ASSERT_TRUE(!output_buffer_address.is_null());

    TF_ASSERT_OK(streams[i]->MemZero(&output_buffer_address, total_bytes));

    output_buffers.push_back(std::move(output_buffer));
  }

  std::vector<T> input_data(n);
  absl::c_iota(input_data, 0);
  TF_ASSERT_OK(streams[0]->Memcpy(input_buffer.address_ptr(), input_data.data(),
                                  n * sizeof(T)));

  std::vector<int64_t> input_offsets = {1, 4, 0, 3};
  std::vector<int64_t> send_sizes = {2, 3, 1, 2};
  std::vector<int64_t> output_offsets = {0, 4, 1, 5};

  stream_executor::DeviceAddressHandle input_offsets_buffer =
      CreateDeviceBuffer(executors[0], input_offsets);
  stream_executor::DeviceAddressHandle send_sizes_buffer =
      CreateDeviceBuffer(executors[0], send_sizes);
  stream_executor::DeviceAddressHandle output_offsets_buffer =
      CreateDeviceBuffer(executors[0], output_offsets);

  std::vector<ncclComm_t> comms(visible_device_count);
  ncclResult_t result =
      ncclCommInitAll(comms.data(), visible_device_count, /*devlist=*/nullptr);
  ASSERT_EQ(result, ncclSuccess);

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "nccl",
                               visible_device_count);
  tsl::Executor& exec = *pool.AsExecutor();

  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<xla::SymmetricMemory>>
                           output_buffers_symmetric_memory,
                       CreateSymmetricMemory(exec, comms, output_buffers));

  TF_ASSERT_OK(RunRaggedAllToAllWithSymmetricMemoryKernel(
      streams[0].get(), primitive_util::NativeToPrimitiveType<T>(),
      input_buffer.address(), output_buffers_symmetric_memory[0].get(),
      output_sym_offset, input_offsets_buffer.address(),
      send_sizes_buffer.address(), output_offsets_buffer.address(), num_outputs,
      num_update_per_output, num_input_rows, num_row_elements));

  std::vector<std::vector<T>> output_results =
      CopyDeviceToHost2D<T>(streams, output_buffers, output_sym_offset, n);

  std::vector<std::vector<T>> expected_output_results =
      GetExpectedOutputResults<T>(
          input_data, input_offsets, send_sizes, output_offsets, num_outputs,
          num_update_per_output, num_input_rows, num_row_elements);

  ASSERT_EQ(output_results.size(), expected_output_results.size());
  EXPECT_EQ(output_results, expected_output_results);
}

}  // namespace
}  // namespace xla::gpu
