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

#include <__clang_cuda_runtime_wrapper.h>

#include <cstdint>

#include "third_party/gpus/cuda/include/cuda/std/__cuda/atomic.h"
#include "xla/stream_executor/gpu/collective_kernel_util.cu.h"

namespace stream_executor::gpu {

template <>
__device__ __forceinline__ void PutSignalFlag<PlatformType::CUDA>(
    uint32_t* addr, uint32_t val) {
  ::cuda::atomic_ref<uint32_t, ::cuda::thread_scope_system> ref(*addr);
  // During signaling release semantics are used to ensure that writes
  // by the current thread are visible to the waiting thread.
  ref.store(val, ::cuda::memory_order_release);
}

template <>
__device__ __forceinline__ void WaitSignalFlag<PlatformType::CUDA>(
    uint32_t* addr, uint32_t expected) {
  ::cuda::atomic_ref<uint32_t, ::cuda::thread_scope_system> ref(*addr);
  // During waiting we use acquire semantics to ensure all memory writes by the
  // remote thread are visible to the current thread.
  // If the flag is greater it means that the other GPU has already signaled
  // the next sync point.
  while (ref.load(::cuda::memory_order_acquire) < expected) {
  }
}

}  // namespace stream_executor::gpu
