/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_GPU_GPU_CUDAMALLOCASYNC_ALLOCATOR_H_
#define TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_GPU_GPU_CUDAMALLOCASYNC_ALLOCATOR_H_

#include <memory>
#include <optional>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/tsl/framework/allocator.h"
#include "tensorflow/tsl/framework/device_id.h"
#include "tensorflow/tsl/platform/macros.h"
#include "tensorflow/tsl/platform/mutex.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"

#define TF_CUDA_MALLOC_ASYNC_SUPPORTED CUDA_VERSION >= 11020
#endif  // GOOGLE_CUDA


namespace stream_executor {

// An allocator that wraps cudaMallocAsync. It has fewer fragmentation
// issues then the BFC memory allocator.  The compute-sanitizer tool
// helps to detect OOB memory errors when using cudaMallocAsync. Use
// the environment variable `TF_GPU_ALLOCATOR=cuda_malloc_async` to
// enable it.
//
// It needs CUDA 11.2+. When using a container, this only needs the
// container driver to be 11.2. It has a WAR again a driver bug in
// multi-GPU setup with CUDA 11.2. The WAR creates an extra context on
// GPU 0.
//
// We configure cudaMallocAsync to grow when more memory is needed
// instead of preallocating everything up front and to keep a local
// pool up to pool_size bytes that is never released to other processes.
// So no other process will "steal" the GPU memory already used by the
// current process. This is to speed up execution and prevent crashes
// of long-running jobs. Use `reserve_memory=true` if you want to
// preallocate the full pool_size. You can also use the environment
// variable `TF_CUDA_MALLOC_ASYNC_SUPPORTED_PREALLOC=nb_bytes` to preallocate
// that amount of memory. `TF_CUDA_MALLOC_ASYNC_SUPPORTED_PREALLOC=-1` is a
// special value that preallocate all what the BFC memory allocator
// would have allocated. This is useful when benchmarking as it doesn't
// change when driver allocations are done.
//
// Here, the pool_size isn't the absolute max as for [Gpu]BFCAllocator.
// The pool can grow above that up to the total GPU memory.  But the
// driver can return the excess memory to other processes.
class GpuCudaMallocAsyncAllocator : public tsl::Allocator {
 public:
  explicit GpuCudaMallocAsyncAllocator(tsl::PlatformDeviceId platform_device_id,
                                       size_t pool_size,
                                       bool reserve_memory = false,
                                       bool compute_stats = true);
  ~GpuCudaMallocAsyncAllocator() override;
  std::string Name() override { return name_; }
  void* AllocateRaw(size_t alignment,
                    size_t num_bytes) override ABSL_NO_THREAD_SAFETY_ANALYSIS;
  void DeallocateRaw(void* ptr) override ABSL_NO_THREAD_SAFETY_ANALYSIS;

  bool TracksAllocationSizes() const override;

  size_t RequestedSize(const void* ptr) const override;

  size_t AllocatedSize(const void* ptr) const override;

  std::optional<tsl::AllocatorStats> GetStats() override;

  bool ClearStats() override;

  void SetStreamAndPreallocateMemory(void* stream) override;

  // With the right VLOG set, it prints:
  // - the number of ptr currently allocated per size (histogram).
  // - each ptr value and its size.
  // - If CUDA_VERSION >= 11030, print cudaMallocAsync statistics.
  void PrintAllocatorStatistics();

  static int GetInstantiatedCountTestOnly() { return number_instantiated_; }

  tsl::AllocatorMemoryType GetMemoryType() const override {
    return tsl::AllocatorMemoryType::kDevice;
  }

 private:
  void PrintAllocatorStatisticsNoLock() ABSL_EXCLUSIVE_LOCKS_REQUIRED(lock_);

#if TF_CUDA_MALLOC_ASYNC_SUPPORTED
  StreamExecutor* stream_exec_;  // Not owned.

  // cudaMallocAsync is stream aware. But TF StreamExecutor use only 1
  // compute stream and already synchronize with the h2d, d2h and d2d
  // stream. So we do not need to ask cudaMallocAsync to add extra
  // synchronization.
  // Not owned.
  CUstream cuda_stream_;

  // Not owned. The default pool of the associated GPU.
  // If null, then the instanciation failed and the first allocation
  // will return an error.
  CUmemoryPool pool_;
#endif  // TF_CUDA_MALLOC_ASYNC_SUPPORTED

  // Just a counter for the number of time this class is instantiated.
  // Only useful for tests.
  static std::atomic<int> number_instantiated_;

  std::string name_;

  bool reserve_memory_;

  TF_DISALLOW_COPY_AND_ASSIGN(GpuCudaMallocAsyncAllocator);

  // Stats.
  // Structures mutable after construction
  mutable tsl::mutex lock_;
  std::unique_ptr<tsl::AllocatorStats> stats_ ABSL_PT_GUARDED_BY(lock_);
  absl::flat_hash_map<const void*, size_t> size_map_ ABSL_GUARDED_BY(lock_);
};

}  // namespace stream_executor

#endif  // TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_GPU_GPU_CUDAMALLOCASYNC_ALLOCATOR_H_
