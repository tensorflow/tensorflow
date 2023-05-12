/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/runtime/topk_kernel.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <functional>
#include <tuple>
#include <vector>

#include "absl/random/random.h"
#include "absl/strings/substitute.h"
#include "third_party/eigen3/Eigen/Core"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/test.h"
#include "tensorflow/tsl/platform/test_benchmark.h"

namespace xla::gpu {
namespace {

using ::stream_executor::gpu::GpuStreamHandle;
using ::testing::Combine;
using ::testing::Values;

#define CUDA_CHECK(s)                                  \
  do {                                                 \
    CHECK_EQ(s, cudaSuccess) << cudaGetErrorString(s); \
  } while (0)

template <typename T>
T* AllocateGpuBuffer(int num_elements) {
  void* buffer;
  CUDA_CHECK(cudaMalloc(&buffer, num_elements * sizeof(T)));
  return static_cast<T*>(buffer);
}

template <typename T>
std::vector<T> RandomFillRange(void* buffer, int num_elements, T start, T end) {
  std::vector<T> local;
  local.reserve(num_elements);
  thread_local absl::BitGen gen;
  for (int i = 0; i < num_elements; ++i) {
    local.push_back(absl::Uniform<T>(gen, start, end));
  }
  CUDA_CHECK(cudaMemcpy(buffer, local.data(), num_elements * sizeof(T),
                        cudaMemcpyHostToDevice));
  return local;
}

template <typename T>
std::vector<T> RandomFill(void* buffer, int num_elements) {
  return RandomFillRange(buffer, num_elements, static_cast<T>(0),
                         static_cast<T>(num_elements));
}

template <typename T>
std::vector<T> RandomFillNegative(void* buffer, int num_elements) {
  return RandomFillRange(buffer, num_elements, -static_cast<T>(num_elements),
                         static_cast<T>(0));
}

PrimitiveType Get(float) { return PrimitiveType::F32; }
PrimitiveType Get(Eigen::bfloat16) { return PrimitiveType::BF16; }

// Params:
//  - n_kb: number of elements in kilobytes.
//  - k: number of elements to return.
//  - batch_size
//  - offset
using TopkTest = ::testing::TestWithParam<std::tuple<int, int, int, int>>;

// In this test we only check that the TopK logic works with float. For the full
// dtype coverage suite, please add them to topk_test.cc, where we can use XLA
// utilities to simplify the test logic.
TEST_P(TopkTest, TopKFloat) {
  using T = float;
  const auto [n_kb, k, batch_size, offset] = GetParam();
  const size_t n = n_kb * 1024 + offset;
  T* input_buffer = AllocateGpuBuffer<T>(n * batch_size);
  auto source = RandomFill<T>(input_buffer, n * batch_size);
  T* output_values = AllocateGpuBuffer<T>(k * batch_size);
  auto* output_indices =
      static_cast<uint32_t*>(AllocateGpuBuffer<uint32_t>(k * batch_size));
  GpuStreamHandle stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  ASSERT_TRUE(RunTopk(stream, Get(T()), input_buffer, n, output_values,
                      output_indices, k, batch_size)
                  .ok());
  std::vector<T> got(k);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (int i = 0; i < batch_size; i++) {
    CUDA_CHECK(cudaMemcpy(got.data(), &output_values[k * i], k * sizeof(T),
                          cudaMemcpyDeviceToHost));
    std::vector<T> slice(source.data() + n * i, source.data() + n * (i + 1));
    std::sort(slice.begin(), slice.end(), std::greater<T>());
    slice.resize(k);
    EXPECT_THAT(got, ::testing::ElementsAreArray(slice))
        << " k=" << k << ", batch_size=" << batch_size;
  }
  CUDA_CHECK(cudaFree(input_buffer));
  CUDA_CHECK(cudaFree(output_indices));
  CUDA_CHECK(cudaFree(output_values));
}

TEST_P(TopkTest, TopKPackedNegative) {
  using T = float;
  const auto [n_kb, k, batch_size, offset] = GetParam();
  const size_t n = n_kb * 1024 + offset;
  T* input_buffer = AllocateGpuBuffer<T>(n * batch_size);
  auto source = RandomFillNegative<T>(input_buffer, n * batch_size);
  T* output_values = AllocateGpuBuffer<T>(k * batch_size);
  auto* output_indices =
      static_cast<uint32_t*>(AllocateGpuBuffer<uint32_t>(k * batch_size));
  GpuStreamHandle stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  ASSERT_TRUE(RunTopk(stream, Get(T()), input_buffer, n, output_values,
                      output_indices, k, batch_size)
                  .ok());
  std::vector<T> got(k);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (int i = 0; i < batch_size; i++) {
    CUDA_CHECK(cudaMemcpy(got.data(), &output_values[k * i], k * sizeof(T),
                          cudaMemcpyDeviceToHost));
    std::vector<T> slice(source.data() + n * i, source.data() + n * (i + 1));
    std::sort(slice.begin(), slice.end(), std::greater<T>());
    slice.resize(k);
    EXPECT_THAT(got, ::testing::ElementsAreArray(slice))
        << " k=" << k << ", batch_size=" << batch_size;
  }
  CUDA_CHECK(cudaFree(input_buffer));
  CUDA_CHECK(cudaFree(output_indices));
  CUDA_CHECK(cudaFree(output_values));
}

INSTANTIATE_TEST_SUITE_P(TopkTests, TopkTest,
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

template <size_t K>
void BM_SmallTopk(benchmark::State& state) {
  using T = float;
  size_t k = K;
  size_t batch_size = state.range(0);
  size_t n = state.range(1) * 1024;
  state.SetLabel(
      absl::Substitute("n=$0Ki k=$1 batch_size=$2", n / 1024, k, batch_size));
  void* input_buffer = AllocateGpuBuffer<T>(n * batch_size);
  auto source = RandomFill<T>(input_buffer, n);
  void* output_values = AllocateGpuBuffer<T>(k);
  auto* output_indices = static_cast<uint32_t*>(AllocateGpuBuffer<uint32_t>(k));
  GpuStreamHandle stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  for (auto _ : state) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));
    CHECK(RunTopk(stream, Get(T()), input_buffer, n, output_values,
                  output_indices, k, batch_size)
              .ok());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    state.SetIterationTime(static_cast<double>(milliseconds) / 1000);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
  }
  size_t items_processed = batch_size * n * state.iterations();
  state.SetItemsProcessed(items_processed);
  state.SetBytesProcessed(items_processed * sizeof(T));
  CUDA_CHECK(cudaFree(input_buffer));
  CUDA_CHECK(cudaFree(output_values));
  CUDA_CHECK(cudaFree(output_indices));
}

BENCHMARK(BM_SmallTopk<1>)->RangePair(1, 512, 16, 1024)->UseManualTime();
BENCHMARK(BM_SmallTopk<2>)->RangePair(1, 512, 16, 1024)->UseManualTime();
BENCHMARK(BM_SmallTopk<4>)->RangePair(1, 512, 16, 1024)->UseManualTime();
BENCHMARK(BM_SmallTopk<8>)->RangePair(1, 512, 16, 1024)->UseManualTime();
BENCHMARK(BM_SmallTopk<16>)->RangePair(1, 512, 16, 1024)->UseManualTime();

}  // namespace
}  // namespace xla::gpu
