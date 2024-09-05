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
#if GOOGLE_CUDA

#include <memory>

#include <gtest/gtest.h>
#include "xla/experiments/sm_bandwidth_benchmark/sm_bw_kernels.h"
#include "xla/experiments/sm_bandwidth_benchmark/sm_bw_utils.h"

namespace experiments {
namespace benchmark {
namespace {

constexpr int kNumSM = 108;
constexpr int kNum32BitRegisters = 64 * 1024;
constexpr int kMaxBlockSize = 1024;

template <typename T>
struct DeviceMemoryDeleter {
  void operator()(T* ptr) { cudaFree(ptr); }
};
template <typename T>
using DeviceMemory = std::unique_ptr<T, DeviceMemoryDeleter<T>>;

template <typename T>
DeviceMemory<T> MakeDeviceMemory(size_t size) {
  T* gpu_ptr = nullptr;
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&gpu_ptr), size * sizeof(T)));
  return DeviceMemory<T>(gpu_ptr);
}

template <typename T>
struct HostMemoryDeleter {
  void operator()(T* ptr) { free(ptr); }
};
template <typename T>
using HostMemory = std::unique_ptr<T, HostMemoryDeleter<T>>;

template <typename T>
HostMemory<T> MakeHostMemory(size_t size) {
  T* h_in = (T*)malloc(size * sizeof(T));
  return HostMemory<T>(h_in);
}

struct EventDeleter {
  using pointer = cudaEvent_t;
  void operator()(pointer event) { cudaEventDestroy(event); }
};
using Event = std::unique_ptr<cudaEvent_t, EventDeleter>;
Event MakeEvent() {
  cudaEvent_t event = nullptr;
  CHECK_CUDA(cudaEventCreate(&event));
  return Event(event);
}

bool CheckOutputAndClean(float* h_in, float* h_out, float* d_out, size_t size) {
  cudaMemcpy(h_out, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < size; i++) {
    if ((h_in[i] - h_out[i]) > 1e-6) {
      LOG(ERROR) << "mismatch :(, i = " << i << " , values are " << h_in[i]
                 << ", " << h_out[i];
      return false;
    }
    h_out[i] = 0;
  }
  return true;
}

template <int chunks>
float BenchmarkCustomDeviceCopy(int kReps, float* d_in, float* d_out,
                                size_t size, int num_blocks = kNumSM,
                                int num_threads = 64) {
  Event start = MakeEvent();
  Event stop = MakeEvent();
  CHECK_CUDA(cudaEventRecord(start.get()));
  for (int i = 0; i < kReps; i++) {
    BenchmarkDeviceCopy<chunks>(d_in, d_out, size, num_blocks, num_threads);
  }
  CHECK_CUDA(cudaEventRecord(stop.get()));
  CHECK_CUDA(cudaEventSynchronize(stop.get()));
  float time_diff = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&time_diff, start.get(), stop.get()));
  return time_diff / kReps;
}

float BenchmarkDev2DevCopy(int kReps, float* d_in, float* d_out, size_t size) {
  Event start = MakeEvent();
  Event stop = MakeEvent();
  CHECK_CUDA(cudaEventRecord(start.get()));
  for (int i = 0; i < kReps; i++) {
    CHECK_CUDA(cudaMemcpy(d_out, d_in, size * sizeof(float),
                          cudaMemcpyDeviceToDevice));
  }
  CHECK_CUDA(cudaEventRecord(stop.get()));
  CHECK_CUDA(cudaEventSynchronize(stop.get()));
  float time_diff = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&time_diff, start.get(), stop.get()));
  return time_diff / kReps;
}

// B/ms -> TB/s
float TbPerSec(size_t size, float time_diff) {
  return 2 * sizeof(float) * size / (1e9 * time_diff);
}

TEST(SMBandwidthTest, IncreasingMemorySize) {
  constexpr int64_t kOneM = 1024 * 1024;
  constexpr int64_t kOneG = 1024 * 1024 * 1024;
  constexpr int64_t kMaxSize = kOneG;

  DeviceMemory<float> d_in = MakeDeviceMemory<float>(kMaxSize);
  DeviceMemory<float> d_out = MakeDeviceMemory<float>(kMaxSize);

  HostMemory<float> h_in = MakeHostMemory<float>(kMaxSize);
  HostMemory<float> h_out = MakeHostMemory<float>(kMaxSize);

  for (size_t i = 0; i < kMaxSize; i++) {
    h_in.get()[i] = i;
  }
  CHECK_CUDA(cudaMemcpy(d_in.get(), h_in.get(), kMaxSize * sizeof(float),
                        cudaMemcpyHostToDevice));

  constexpr int kReps = 10;
  LOG(ERROR) << "size,custom TB/s,devTodev TB/s";
  for (size_t size = kOneM; size <= kMaxSize; size *= 2) {
    float time_diff_c =
        BenchmarkCustomDeviceCopy<1>(kReps, d_in.get(), d_out.get(), size);
    EXPECT_TRUE(
        CheckOutputAndClean(h_in.get(), h_out.get(), d_out.get(), size));

    float time_diff_d2d =
        BenchmarkDev2DevCopy(kReps, d_in.get(), d_out.get(), size);
    EXPECT_TRUE(
        CheckOutputAndClean(h_in.get(), h_out.get(), d_out.get(), size));

    LOG(ERROR) << size << "," << TbPerSec(size, time_diff_c) << ","
               << TbPerSec(size, time_diff_d2d);
  }
}

TEST(SMBandwidthTest, IncreasingNumBlocks) {
  constexpr size_t kSize = 1 << 28;
  constexpr int kReps = 10;
  constexpr int kNumThreads = 64;

  DeviceMemory<float> d_in = MakeDeviceMemory<float>(kSize);
  DeviceMemory<float> d_out = MakeDeviceMemory<float>(kSize);

  HostMemory<float> h_in = MakeHostMemory<float>(kSize);
  HostMemory<float> h_out = MakeHostMemory<float>(kSize);

  for (size_t i = 0; i < kSize; i++) {
    h_in.get()[i] = i;
  }
  CHECK_CUDA(cudaMemcpy(d_in.get(), h_in.get(), kSize * sizeof(float),
                        cudaMemcpyHostToDevice));

  LOG(ERROR) << "num_blocks,TB/s";
  for (int64_t num_blocks = kNumSM; num_blocks <= kNumSM * 32;
       num_blocks += kNumSM) {
    Event start = MakeEvent();
    Event stop = MakeEvent();
    CHECK_CUDA(cudaEventRecord(start.get()));
    for (int i = 0; i < kReps; i++) {
      BenchmarkDeviceCopy<1>(d_in.get(), d_out.get(), kSize, num_blocks,
                             kNumThreads);
    }
    CHECK_CUDA(cudaEventRecord(stop.get()));
    CHECK_CUDA(cudaEventSynchronize(stop.get()));
    float time_diff = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&time_diff, start.get(), stop.get()));
    time_diff /= kReps;
    LOG(ERROR) << num_blocks << "," << TbPerSec(kSize, time_diff);

    CHECK_CUDA(cudaMemcpy(h_out.get(), d_out.get(), kSize * sizeof(float),
                          cudaMemcpyDeviceToHost));
    EXPECT_TRUE(
        CheckOutputAndClean(h_in.get(), h_out.get(), d_out.get(), kSize));
  }
}

template <int chunks_log>
struct ForLoop {
  template <template <int> class Func>
  static void iterate() {
    Func<chunks_log>()();
    ForLoop<chunks_log - 1>::template iterate<Func>();
  }
};

template <>
struct ForLoop<0> {
  template <template <int> class Func>
  static void iterate() {
    Func<0>()();
  }
};

template <int chunks_log>
struct IterateOverChunkSizeImpl {
  void operator()() {
    constexpr size_t kSize = 1 << 28;
    constexpr int kReps = 10;

    DeviceMemory<float> d_in = MakeDeviceMemory<float>(kSize);
    DeviceMemory<float> d_out = MakeDeviceMemory<float>(kSize);

    HostMemory<float> h_in = MakeHostMemory<float>(kSize);
    HostMemory<float> h_out = MakeHostMemory<float>(kSize);

    for (size_t i = 0; i < kSize; i++) {
      h_in.get()[i] = i;
    }
    CHECK_CUDA(cudaMemcpy(d_in.get(), h_in.get(), kSize * sizeof(float),
                          cudaMemcpyHostToDevice));

    float time_diff = BenchmarkCustomDeviceCopy<1 << chunks_log>(
        kReps, d_in.get(), d_out.get(), kSize);
    EXPECT_TRUE(
        CheckOutputAndClean(h_in.get(), h_out.get(), d_out.get(), kSize));

    LOG(ERROR) << (1 << chunks_log) << "," << TbPerSec(kSize, time_diff);
  }
};

TEST(SMBandwidthTest, IterateOverChunkSize) {
  LOG(ERROR) << "chunks,TB/s";
  ForLoop<10>::iterate<IterateOverChunkSizeImpl>();
}

TEST(SMBandwidthTest, BestParameters) {
  constexpr size_t kSize = 1 << 28;
  constexpr int kReps = 10;
  constexpr int kNumThreads = 1024;
  constexpr int kChunkSize = 32;

  DeviceMemory<float> d_in = MakeDeviceMemory<float>(kSize);
  DeviceMemory<float> d_out = MakeDeviceMemory<float>(kSize);

  HostMemory<float> h_in = MakeHostMemory<float>(kSize);
  HostMemory<float> h_out = MakeHostMemory<float>(kSize);

  for (size_t i = 0; i < kSize; i++) {
    h_in.get()[i] = i;
  }
  CHECK_CUDA(cudaMemcpy(d_in.get(), h_in.get(), kSize * sizeof(float),
                        cudaMemcpyHostToDevice));

  for (int num_blocks = 1; num_blocks <= kNumSM; num_blocks++) {
    float time_diff = BenchmarkCustomDeviceCopy<kChunkSize>(
        kReps, d_in.get(), d_out.get(), kSize, num_blocks, kNumThreads);
    EXPECT_TRUE(
        CheckOutputAndClean(h_in.get(), h_out.get(), d_out.get(), kSize));
    LOG(ERROR) << "num_blocks: " << num_blocks
               << ", num_threads: " << kNumThreads
               << ", TB/sec: " << TbPerSec(kSize, time_diff);
  }
}

template <int chunks_log>
struct UseMaxNumberOfRegistersPerSmImpl {
  void operator()() {
    constexpr size_t kSize = 1 << 28;
    constexpr int kReps = 10;
    constexpr int kNumBlocks = kNumSM;

    DeviceMemory<float> d_in = MakeDeviceMemory<float>(kSize);
    DeviceMemory<float> d_out = MakeDeviceMemory<float>(kSize);

    HostMemory<float> h_in = MakeHostMemory<float>(kSize);
    HostMemory<float> h_out = MakeHostMemory<float>(kSize);

    for (size_t i = 0; i < kSize; i++) {
      h_in.get()[i] = i;
    }
    CHECK_CUDA(cudaMemcpy(d_in.get(), h_in.get(), kSize * sizeof(float),
                          cudaMemcpyHostToDevice));

    for (int coeff = 1; coeff <= 3; coeff++) {
      int num_threads = kNum32BitRegisters / ((1 << chunks_log) * coeff);
      if (num_threads > kMaxBlockSize) continue;
      float time_diff = BenchmarkCustomDeviceCopy<1 << chunks_log>(
          kReps, d_in.get(), d_out.get(), kSize, kNumBlocks, num_threads);
      EXPECT_TRUE(
          CheckOutputAndClean(h_in.get(), h_out.get(), d_out.get(), kSize));
      LOG(ERROR) << "num_threads: " << num_threads
                 << ", chunks: " << (1 << chunks_log)
                 << ", TB/sec: " << TbPerSec(kSize, time_diff);
    }
  }
};

TEST(SMBandwidthTest, UseMaxNumberOfRegistersPerSm) {
  ForLoop<10>::iterate<UseMaxNumberOfRegistersPerSmImpl>();
}

}  // namespace
}  // namespace benchmark
}  // namespace experiments

#endif  // GOOGLE_CUDA
