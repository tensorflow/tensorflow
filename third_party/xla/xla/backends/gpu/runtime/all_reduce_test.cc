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
#include <numeric>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/primitive_util.h"
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

se::StreamExecutor* GetGpuExecutor() {
  auto* platform =
      se::PlatformManager::PlatformWithName(se::GpuPlatformName()).value();
  return platform->ExecutorForDevice(0).value();
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

  auto* executor = GetGpuExecutor();
  auto stream = executor->CreateStream().value();

  constexpr int64_t num_inputs = 2;
  constexpr int64_t num_elements = 128000;

  std::vector<se::DeviceMemoryHandle> input_buffers;
  for (int64_t i = 0; i < num_inputs; ++i) {
    input_buffers.emplace_back(executor,
                               executor->AllocateArray<T>(num_elements));
    ASSERT_TRUE(!input_buffers[i].memory().is_null());
  }

  se::DeviceMemoryHandle output_buffer(
      executor, executor->AllocateArray<T>(num_elements));
  ASSERT_TRUE(!output_buffer.memory().is_null());

  std::vector<T> output_data(num_elements);
  for (int i = 0; i < num_inputs; ++i) {
    std::vector<T> input_data(num_elements);
    std::iota(input_data.begin(), input_data.end(), static_cast<T>(0));

    TF_ASSERT_OK(stream->Memcpy(input_buffers[i].memory_ptr(),
                                input_data.data(), num_elements * sizeof(T)));

    std::transform(input_data.begin(), input_data.end(), output_data.begin(),
                   output_data.begin(), std::plus<T>());
  }

  TF_ASSERT_OK(stream->BlockHostUntilDone());

  std::vector<se::DeviceMemoryBase> input_buffers_span;
  for (auto& input_buffer : input_buffers) {
    input_buffers_span.push_back(input_buffer.memory());
  }

  TF_ASSERT_OK(RunAllReduceKernel(
      stream.get(), primitive_util::NativeToPrimitiveType<T>(),
      input_buffers_span, output_buffer.memory(), num_inputs, num_elements));

  TF_ASSERT_OK(stream->BlockHostUntilDone());

  std::vector<T> output_results(num_elements);
  TF_ASSERT_OK(stream->Memcpy(output_results.data(), output_buffer.memory(),
                              num_elements * sizeof(T)));

  EXPECT_EQ(output_results, output_data);
}

}  // namespace
}  // namespace xla::gpu
