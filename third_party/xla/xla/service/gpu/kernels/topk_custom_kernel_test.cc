/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/kernels/topk_custom_kernel.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include "absl/random/random.h"
#include "absl/strings/ascii.h"
#include "absl/strings/substitute.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla::gpu::kernel::topk {

using ::testing::Combine;
using ::testing::Values;

template <typename T>
std::vector<T> RandomVecRange(int num_elements, T start, T end) {
  std::vector<T> local;
  local.reserve(num_elements);
  thread_local absl::BitGen gen;
  for (int i = 0; i < num_elements; ++i) {
    local.push_back(absl::Uniform<T>(gen, start, end));
  }
  return local;
}

template <typename T>
std::vector<T> RandomVec(int num_elements) {
  return RandomVecRange(num_elements, static_cast<T>(0),
                        static_cast<T>(num_elements));
}

template <typename T>
std::vector<T> RandomVecNegative(int num_elements) {
  return RandomVecRange(num_elements, -static_cast<T>(num_elements),
                        static_cast<T>(0));
}

PrimitiveType Get(float) { return PrimitiveType::F32; }

PrimitiveType Get(bfloat16) { return PrimitiveType::BF16; }

// Params:
//  - n_kb: number of elements in kilobytes.
//  - k: number of elements to return.
//  - batch_size
//  - offset
using TopKKernelTest = ::testing::TestWithParam<std::tuple<int, int, int, int>>;

// In this test we only check that the TopK logic works with float. For the full
// dtype coverage suite, please add them to topk_test.cc, where we can use XLA
// utilities to simplify the test logic.
TEST_P(TopKKernelTest, TopKFloat) {
  using T = float;

  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  se::Platform* platform = se::PlatformManager::PlatformWithName(name).value();
  se::StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  auto stream = executor->CreateStream().value();

  const auto [n_kb, k, batch_size, offset] = GetParam();
  const size_t n = n_kb * 1024 + offset;

  se::DeviceMemory<T> input_buffer =
      executor->AllocateArray<T>(n * batch_size, 0);
  se::DeviceMemory<T> output_values =
      executor->AllocateArray<T>(k * batch_size, 0);
  se::DeviceMemory<uint32_t> output_indices =
      executor->AllocateArray<uint32_t>(k * batch_size, 0);

  auto source = RandomVec<T>(n * batch_size);
  TF_ASSERT_OK(
      stream->Memcpy(&input_buffer, source.data(), n * batch_size * sizeof(T)));
  TF_ASSERT_OK(stream->MemZero(&output_values, k * batch_size * sizeof(T)));
  TF_ASSERT_OK(
      stream->MemZero(&output_indices, k * batch_size * sizeof(uint32_t)));

  auto custom_kernel =
      GetTopKKernel("topk", PrimitiveType::F32, n, k, batch_size);

  TF_ASSERT_OK_AND_ASSIGN(
      auto kernel, se::Kernel::Create(executor, custom_kernel->kernel_spec()));

  // Launch topk kernel with device memory arguments.
  se::KernelArgsDeviceMemoryArray arr(
      std::vector<se::DeviceMemoryBase>(
          {input_buffer, output_values, output_indices}),
      custom_kernel->shared_memory_bytes());
  TF_ASSERT_OK(executor->Launch(stream.get(), custom_kernel->thread_dims(),
                                custom_kernel->block_dims(), *kernel, arr));

  std::vector<T> got(k);
  ASSERT_TRUE(stream->BlockHostUntilDone().ok());
  for (int i = 0; i < batch_size; i++) {
    TF_ASSERT_OK(stream->Memcpy(got.data(), output_values.GetSlice(k * i, k),
                                k * sizeof(T)));
    std::vector<T> slice(source.data() + n * i, source.data() + n * (i + 1));
    std::sort(slice.begin(), slice.end(), std::greater<T>());
    slice.resize(k);
    EXPECT_THAT(got, ::testing::ElementsAreArray(slice))
        << " k=" << k << ", batch_size=" << batch_size << " i=" << i;
  }
}

TEST_P(TopKKernelTest, TopKPackedNegative) {
  using T = float;

  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  se::Platform* platform = se::PlatformManager::PlatformWithName(name).value();
  se::StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  auto stream = executor->CreateStream().value();

  const auto [n_kb, k, batch_size, offset] = GetParam();
  const size_t n = n_kb * 1024 + offset;

  se::DeviceMemory<T> input_buffer =
      executor->AllocateArray<T>(n * batch_size, 0);
  se::DeviceMemory<T> output_values =
      executor->AllocateArray<T>(k * batch_size, 0);
  se::DeviceMemory<uint32_t> output_indices =
      executor->AllocateArray<uint32_t>(k * batch_size, 0);

  auto source = RandomVecNegative<T>(n * batch_size);
  TF_ASSERT_OK(
      stream->Memcpy(&input_buffer, source.data(), n * batch_size * sizeof(T)));
  TF_ASSERT_OK(stream->MemZero(&output_values, k * batch_size * sizeof(T)));
  TF_ASSERT_OK(
      stream->MemZero(&output_indices, k * batch_size * sizeof(uint32_t)));

  auto custom_kernel =
      GetTopKKernel("topk", PrimitiveType::F32, n, k, batch_size);

  TF_ASSERT_OK_AND_ASSIGN(
      auto kernel, se::Kernel::Create(executor, custom_kernel->kernel_spec()));

  // Launch topk kernel with device memory arguments.
  se::KernelArgsDeviceMemoryArray arr(
      std::vector<se::DeviceMemoryBase>(
          {input_buffer, output_values, output_indices}),
      custom_kernel->shared_memory_bytes());
  TF_ASSERT_OK(executor->Launch(stream.get(), custom_kernel->thread_dims(),
                                custom_kernel->block_dims(), *kernel, arr));

  std::vector<T> got(k);
  ASSERT_TRUE(stream->BlockHostUntilDone().ok());
  for (int i = 0; i < batch_size; i++) {
    TF_ASSERT_OK(stream->Memcpy(got.data(), output_values.GetSlice(k * i, k),
                                k * sizeof(T)));
    std::vector<T> slice(source.data() + n * i, source.data() + n * (i + 1));
    std::sort(slice.begin(), slice.end(), std::greater<T>());
    slice.resize(k);
    EXPECT_THAT(got, ::testing::ElementsAreArray(slice))
        << " k=" << k << ", batch_size=" << batch_size << " i=" << i;
  }
}

INSTANTIATE_TEST_SUITE_P(TopKTests, TopKKernelTest,
                         Combine(
                             /*n_kb=*/Values(1, 8, 12, 64, 128),
                             /*k=*/Values(1, 2, 8, 16, 7, 12),
                             /*batch_size=*/Values(1, 16, 64, 128),
                             /*offset=*/Values(0, 7, 4)),
                         [](const auto& info) {
                           return absl::Substitute(
                               "n$0KiB_k$1_batch_size$2_offset$3",
                               std::get<0>(info.param), std::get<1>(info.param),
                               std::get<2>(info.param),
                               std::get<3>(info.param));
                         });

}  // namespace xla::gpu::kernel::topk
