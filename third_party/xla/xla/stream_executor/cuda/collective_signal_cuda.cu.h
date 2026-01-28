/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_CUDA_COLLECTIVE_SIGNAL_CUDA_CU_H_
#define XLA_STREAM_EXECUTOR_CUDA_COLLECTIVE_SIGNAL_CUDA_CU_H_

#include <cstdint>

#include "third_party/gpus/cuda/include/cuda/atomic"
#include "third_party/gpus/cuda/include/cuda_bf16.h"
#include "xla/stream_executor/gpu/collective_signal.cu.h"
#include "xla/stream_executor/kernel_spec.h"

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

#endif  // XLA_STREAM_EXECUTOR_CUDA_COLLECTIVE_SIGNAL_CUDA_CU_H_
