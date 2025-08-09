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

#include <cstddef>
#include <cstdint>

#include "absl/base/casts.h"
#include "third_party/gpus/cuda/include/cuda/atomic"
#include "third_party/gpus/cuda/include/cuda_bf16.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/gpu/all_reduce_kernel.h"
#include "xla/stream_executor/gpu/all_reduce_kernel_lib.cu.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/types.h"

namespace stream_executor::gpu {

template <>
union alignas(8) Vec<__nv_bfloat16> {
  using PackedType = int2;

  __nv_bfloat16 data[4];
  PackedType packed;
};

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

// C++ macros don't like commas in template arguments, so we need to use
// __VA_ARGS__ to get around this.
#define SINGLE_ARG(...) __VA_ARGS__

#define REGISTER_ALL_REDUCE_KERNEL_IMPL(SUFFIX, XLA_TYPE, NV_TYPE,             \
                                        REDUCTION_KIND, STRATEGY)              \
  GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(                              \
      AllReduceKernelCuda##SUFFIX##STRATEGY,                                   \
      SINGLE_ARG(stream_executor::gpu::AllReduceKernel<                        \
                 XLA_TYPE, xla::ReductionKind::REDUCTION_KIND,                 \
                 xla::se::gpu::AllReduceStrategy::STRATEGY>),                  \
      stream_executor::cuda::kCudaPlatformId, ([](size_t arity) {              \
        return stream_executor::KernelLoaderSpec::CreateInProcessSymbolSpec(   \
            absl::bit_cast<void*>(&stream_executor::gpu::AllReduceKernelImpl<  \
                                  NV_TYPE, xla::ReductionKind::REDUCTION_KIND, \
                                  xla::se::gpu::AllReduceStrategy::STRATEGY,   \
                                  stream_executor::gpu::PlatformType::CUDA>),  \
            "all_reduce_" #SUFFIX #STRATEGY, arity);                           \
      }));

// Create instantiations for all all-reduce strategies.
#define REGISTER_ALL_REDUCE_KERNEL(SUFFIX, XLA_TYPE, NV_TYPE, REDUCTION_KIND) \
  REGISTER_ALL_REDUCE_KERNEL_IMPL(SUFFIX, XLA_TYPE, NV_TYPE, REDUCTION_KIND,  \
                                  kOneShot)                                   \
  REGISTER_ALL_REDUCE_KERNEL_IMPL(SUFFIX, XLA_TYPE, NV_TYPE, REDUCTION_KIND,  \
                                  kTwoShot)

// Register the kernel for different types using the macro
REGISTER_ALL_REDUCE_KERNEL(AddBF16, xla::bfloat16, __nv_bfloat16, SUM);
REGISTER_ALL_REDUCE_KERNEL(AddF32, float, float, SUM);

// AllReduce doesn't have a corresponding reduction kind for logical operations.
// NCCL uses MAX and MIN on uint8_t for logical operations.
REGISTER_ALL_REDUCE_KERNEL(OrPRED, bool, uint8_t, MAX);
