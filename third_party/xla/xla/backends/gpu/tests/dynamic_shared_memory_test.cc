/* Copyright 2022 The OpenXLA Authors.

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
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_init.h"
#include "xla/stream_executor/gpu/gpu_test_kernels.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {
namespace {

namespace se = stream_executor;

TEST(SharedMemoryUseTest, ArrayReversalWorks) {
  // Test that shared memory is fully available to kernels requesting it.
  // Create an array with a 2D pattern of numbers, fill the requested shared
  // memory with it, read it back inverting both axes,
  // copy the result back to the host and verify it.
  ASSERT_OK_AND_ASSIGN(
      se::Platform * platform,
      se::PlatformManager::PlatformWithName(se::GpuPlatformName()));
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  // Use 90% of the available shared memory to verify that a fractional
  // amount works as well, not only the full size.
  const unsigned n_cols =
      executor->GetDeviceDescription().threads_per_block_limit();
  const unsigned n_rows =
      0.9 * executor->GetDeviceDescription().shared_memory_per_block_optin() /
      n_cols;
  const int n_elements = n_cols * n_rows;
  using data_type = uint8_t;
  constexpr int max_value = UINT8_MAX;
  const int buffer_size_bytes = n_elements * sizeof(data_type);
  VLOG(1) << "Using " << buffer_size_bytes << " bytes of shared memory";

  ASSERT_OK_AND_ASSIGN(auto kernel, se::gpu::LoadDynShmemTestKernel(executor));

  se::DeviceAddress<data_type> device_buffer =
      executor->AllocateArray<data_type>(n_elements);
  std::vector<data_type> host_buffer(n_elements);
  for (int row = 0; row < n_rows; ++row) {
    for (int col = 0; col < n_cols; ++col) {
      // Fill the buffer with a reasonably non-uniform pattern, multiples of
      // 3 and 5 make it non-symmetric with respect to the main diagonal.
      host_buffer[row * n_cols + col] = (3 * col + 5 * row) % max_value;
    }
  }

  CHECK_OK(
      stream->Memcpy(&device_buffer, host_buffer.data(), buffer_size_bytes));
  se::DeviceAddress<uint32_t> dev_n_cols = executor->AllocateScalar<uint32_t>();
  CHECK_OK(stream->Memcpy(&dev_n_cols, &n_cols, sizeof(uint32_t)));
  se::DeviceAddress<uint32_t> dev_n_rows = executor->AllocateScalar<uint32_t>();
  CHECK_OK(stream->Memcpy(&dev_n_rows, &n_rows, sizeof(uint32_t)));
  CHECK_OK(stream->BlockHostUntilDone());

  CHECK_OK(kernel.Launch(se::ThreadDim(n_cols, 1, 1), se::BlockDim(1, 1, 1),
                         buffer_size_bytes, stream.get(), device_buffer,
                         dev_n_cols, dev_n_rows));
  CHECK_OK(stream->BlockHostUntilDone());
  CHECK_OK(
      stream->Memcpy(host_buffer.data(), device_buffer, buffer_size_bytes));
  CHECK_OK(stream->BlockHostUntilDone());

  for (int row = 0; row < n_rows; ++row) {
    for (int col = 0; col < n_cols; ++col) {
      EXPECT_EQ(host_buffer[(n_rows - row - 1) * n_cols + (n_cols - col - 1)],
                (3 * col + 5 * row) % max_value)
          << row << " " << col;
    }
  }
}

}  // namespace
}  // namespace gpu
}  // namespace xla
