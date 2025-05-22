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

#include "xla/backends/gpu/runtime/all_reduce.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_handle.h"
#include "xla/stream_executor/gpu/gpu_init.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

se::StreamExecutor* GetGpuExecutor(int64_t device_ordinal) {
  auto* platform =
      se::PlatformManager::PlatformWithName(se::GpuPlatformName()).value();
  return platform->ExecutorForDevice(device_ordinal).value();
}

template <typename T>
class AllReduceKernelTest : public ::testing::Test {};

using AllReduceKernelTestTypes = ::testing::Types<float, bfloat16>;

class AllReduceKernelTestNameGenerator {
 public:
  template <typename T>
  static std::string GetName(int) {
    if constexpr (std::is_same_v<T, float>) {
      return "f32";
    }
    if constexpr (std::is_same_v<T, bfloat16>) {
      return "bf16";
    }
  }
};

TYPED_TEST_SUITE(AllReduceKernelTest, AllReduceKernelTestTypes,
                 AllReduceKernelTestNameGenerator);

TYPED_TEST(AllReduceKernelTest, SimpleKernelTest) {
  using T = TypeParam;

  constexpr int64_t kNumRanks = 2;
  constexpr int64_t kNumElements = 128000;

  LaunchDimensions launch_dimensions(/*block_x_count=*/8,
                                     /*thread_x_count_per_block=*/512);

  std::vector<se::StreamExecutor*> executors;
  std::vector<std::unique_ptr<se::Stream>> streams;
  std::vector<se::DeviceMemoryHandle> local_input_buffers;
  std::vector<se::DeviceMemoryHandle> data_buffers;
  std::vector<se::DeviceMemoryHandle> signal_flags_buffers;
  std::vector<T> output_data(kNumElements);

  std::vector<se::DeviceMemoryBase> remote_input_buffers_span;
  std::vector<se::DeviceMemoryBase> signal_flags_buffers_span;

  for (int i = 0; i < kNumRanks; ++i) {
    auto* executor = GetGpuExecutor(i);
    executors.push_back(executor);
    streams.push_back(executor->CreateStream().value());

    local_input_buffers.emplace_back(executor,
                                     executor->AllocateArray<T>(kNumElements));
    ASSERT_TRUE(!local_input_buffers[i].memory().is_null());

    data_buffers.emplace_back(executor,
                              executor->AllocateArray<T>(kNumElements));
    ASSERT_TRUE(!data_buffers[i].memory().is_null());

    signal_flags_buffers.emplace_back(
        executor, executor->AllocateArray<uint32_t>(
                      kNumRanks * launch_dimensions.num_blocks()));
    ASSERT_TRUE(!signal_flags_buffers[i].memory().is_null());

    std::vector<T> input_data(kNumElements);
    std::iota(input_data.begin(), input_data.end(), static_cast<T>(0));
    std::transform(input_data.begin(), input_data.end(), output_data.begin(),
                   output_data.begin(), std::plus<T>());

    TF_ASSERT_OK(streams[i]->Memcpy(local_input_buffers[i].memory_ptr(),
                                    input_data.data(),
                                    kNumElements * sizeof(T)));

    remote_input_buffers_span.push_back(data_buffers[i].memory());
    signal_flags_buffers_span.push_back(signal_flags_buffers[i].memory());
  }

  for (int i = 0; i < kNumRanks; ++i) {
    TF_ASSERT_OK(streams[i]->BlockHostUntilDone());
  }

  if (!executors[0]->CanEnablePeerAccessTo(executors[1])) {
    GTEST_SKIP() << "Test requires direct peer memory access between devices.";
  }

  TF_ASSERT_OK(executors[0]->EnablePeerAccessTo(executors[1]));
  TF_ASSERT_OK(executors[1]->EnablePeerAccessTo(executors[0]));

  for (int i = 0; i < kNumRanks; ++i) {
    auto active_context = executors[i]->Activate();
    TF_ASSERT_OK(RunAllReduceKernel(
        streams[i].get(), launch_dimensions,
        primitive_util::NativeToPrimitiveType<T>(), remote_input_buffers_span,
        // Memory is aliased for both input and output (similar to what nccl
        // would do).
        /*local_input_buffer=*/local_input_buffers[i].memory(),
        /*output_buffer=*/local_input_buffers[i].memory(),
        /*rank=*/RankId(i), /*num_ranks=*/kNumRanks, kNumElements,
        signal_flags_buffers_span));
  }

  for (int i = 0; i < kNumRanks; ++i) {
    TF_ASSERT_OK(streams[i]->BlockHostUntilDone());
  }

  for (int i = 0; i < kNumRanks; ++i) {
    std::vector<T> output_results(kNumElements);
    TF_ASSERT_OK(streams[i]->Memcpy(output_results.data(),
                                    local_input_buffers[i].memory(),
                                    kNumElements * sizeof(T)));

    EXPECT_EQ(output_results, output_data);
  }
}

}  // namespace
}  // namespace xla::gpu
