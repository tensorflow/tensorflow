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
#ifndef XLA_STREAM_EXECUTOR_GPU_ALL_REDUCE_KERNEL_LIB_CU_H_
#define XLA_STREAM_EXECUTOR_GPU_ALL_REDUCE_KERNEL_LIB_CU_H_

#include <array>
#include <cassert>
#include <cstdint>
#include <type_traits>

#include "third_party/gpus/cuda/include/cuda/atomic"
#include "third_party/gpus/cuda/include/cuda_bf16.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/gpu/all_reduce_kernel.h"

namespace stream_executor::gpu {

template <typename T>
union Vec;

template <>
union alignas(16) Vec<float> {
  using PackedType = int4;

  float data[4];
  PackedType packed;
};

template <>
union alignas(8) Vec<__nv_bfloat16> {
  using PackedType = int2;

  __nv_bfloat16 data[4];
  PackedType packed;
};

template <>
union alignas(4) Vec<uint8_t> {
  using PackedType = int32_t;

  uint8_t data[4];
  PackedType packed;
};

template <typename T, xla::ReductionKind ReductionKindT>
__device__ __forceinline__
    typename std::enable_if<ReductionKindT == xla::ReductionKind::SUM, T>::type
    ApplyBinaryOp(T a, T b) {
  return a + b;
}

template <typename T, xla::ReductionKind ReductionKindT>
__device__ __forceinline__
    typename std::enable_if<ReductionKindT == xla::ReductionKind::MAX, T>::type
    ApplyBinaryOp(T a, T b) {
  return max(a, b);
}

template <typename T>
__device__ __forceinline__ Vec<T> VecLoad(T* addr) {
  Vec<T> vec;
  vec.packed = *(reinterpret_cast<typename Vec<T>::PackedType*>(addr));
  return vec;
}

template <typename T>
__device__ __forceinline__ void VecStore(T* addr, const Vec<T>& vec) {
  *(reinterpret_cast<typename Vec<T>::PackedType*>(addr)) = vec.packed;
}

template <typename T, xla::ReductionKind ReductionKindT>
__device__ __forceinline__ void VecOp(Vec<T>& res, const Vec<T>& vec) {
  res.data[0] = ApplyBinaryOp<T, ReductionKindT>(res.data[0], vec.data[0]);
  res.data[1] = ApplyBinaryOp<T, ReductionKindT>(res.data[1], vec.data[1]);
  res.data[2] = ApplyBinaryOp<T, ReductionKindT>(res.data[2], vec.data[2]);
  res.data[3] = ApplyBinaryOp<T, ReductionKindT>(res.data[3], vec.data[3]);
}

__device__ __forceinline__ void PutSignalFlag(uint32_t* addr, uint32_t val) {
  ::cuda::atomic_ref<uint32_t, ::cuda::thread_scope_system> ref(*addr);
  // During signaling release semantics are used to ensure that writes
  // by the current thread are visible to the waiting thread.
  ref.store(val, ::cuda::memory_order_release);
}

__device__ __forceinline__ void WaitSignalFlag(uint32_t* addr,
                                               uint32_t expected) {
  ::cuda::atomic_ref<uint32_t, ::cuda::thread_scope_system> ref(*addr);
  // During waiting we use acquire semantics to ensure all memory writes by the
  // remote thread are visible to the current thread.
  // If the flag is greater it means that the other GPU has already signaled
  // the next sync point.
  while (ref.load(::cuda::memory_order_acquire) < expected) {
  }
}

__device__ __forceinline__ void SyncRemoteBlocks(
    std::array<RestrictedPtr<uint32_t>, kMaxNumAllReduceInputPtrs>
        signal_pad_ptrs,
    int64_t rank, int64_t num_ranks, uint32_t signal_value) {
  if (threadIdx.x < num_ranks) {
    auto target_rank = threadIdx.x;
    PutSignalFlag(signal_pad_ptrs[target_rank] + blockIdx.x * num_ranks + rank,
                  signal_value);
    WaitSignalFlag(signal_pad_ptrs[rank] + blockIdx.x * num_ranks + target_rank,
                   signal_value);
  }
}

template <typename T, xla::ReductionKind ReductionKindT>
__device__ __forceinline__ void OneShotAllReduceKernelImpl(
    const AllReduceKernelParams<T>& args) {
  int64_t offset =
      kNumElementsPerThread * (blockIdx.x * blockDim.x + threadIdx.x);
  int64_t stride = kNumElementsPerThread * blockDim.x * gridDim.x;

  // Copy data from local input buffer to remote input buffer.
  for (int i = offset; i < args.num_elements; i += stride) {
    VecStore(args.remote_input_buffers[args.rank] + i,
             VecLoad(args.input_buffer + i));
  }

  SyncRemoteBlocks(args.signal_flags_buffers, args.rank, args.num_ranks,
                   args.signal_value);
  __syncthreads();

  for (int i = offset; i < args.num_elements; i += stride) {
    Vec<T> acc = VecLoad(args.remote_input_buffers[0] + i);

    // Since `remote_input_buffers` are provided in rank order, we get stable
    // reduction results on all devices.
#pragma unroll
    for (int j = 1; j < kMaxNumAllReduceInputPtrs; ++j) {
      if (j < args.num_ranks) {
        VecOp<T, ReductionKindT>(acc,
                                 VecLoad(args.remote_input_buffers[j] + i));
      }
    }

    VecStore(args.output_buffer + i, acc);
  }
}

template <typename T, xla::ReductionKind ReductionKindT>
__device__ __forceinline__ void TwoShotAllReduceKernelImpl(
    const AllReduceKernelParams<T>& args) {
  const int64_t offset = blockIdx.x * args.num_elements_per_block +
                         threadIdx.x * kNumElementsPerThread;
  const int64_t offset_end = (blockIdx.x + 1) * args.num_elements_per_block;

  const int64_t block_stride = kNumElementsPerThread * blockDim.x;
  // Responsibility for accumulation for this rank.
  const int64_t rank_offset = args.rank * args.num_elements_per_rank;

  // Step1: Copy data from input buffer to the local shared buffer.
  // Each GPU will copy data from its local input buffer to its own local shared
  // buffer from where it will be read by participating devices (PULLed).
  // We use a grid stride loop for simplicity.
  {
    const int64_t grid_offset =
        kNumElementsPerThread * (blockIdx.x * blockDim.x + threadIdx.x);
    const int64_t grid_stride = kNumElementsPerThread * blockDim.x * gridDim.x;
    for (int i = grid_offset; i < args.num_elements; i += grid_stride) {
      VecStore(args.remote_input_buffers[args.rank] + i,
               VecLoad(args.input_buffer + i));
    }
  }

  // Shot1: Wait for all participating devices to finish copying data to their
  // shared buffer.
  SyncRemoteBlocks(args.signal_flags_buffers, args.rank, args.num_ranks,
                   args.signal_value);
  __syncthreads();

  // Step2: Accumulate data for the responsible indices in the shared buffers.
  for (int i = offset; i < offset_end; i += block_stride) {
    // Each rank is only responsible for accumulating num_elements_per_rank
    // elements.
    const int64_t offset_i = rank_offset + i;
    Vec<T> acc = VecLoad(args.remote_input_buffers[0] + offset_i);

    // Since `remote_input_ptrs` are provided in rank order, we get stable
    // reduction results on all devices.
#pragma unroll
    for (int r = 1; r < kMaxNumAllReduceInputPtrs; ++r) {
      if (r >= args.num_ranks) {
        continue;
      }
      VecOp<T, ReductionKindT>(
          acc, VecLoad(args.remote_input_buffers[r] + offset_i));
    }
    VecStore(args.remote_input_buffers[args.rank] + offset_i, acc);
  }

  // Shot2: Wait for all participating devices to finish accumulating data in
  // the shared buffer. Note that signal_value + 1 is used to ensure that the
  // synchronization is different from the one used above.
  SyncRemoteBlocks(args.signal_flags_buffers, args.rank, args.num_ranks,
                   args.signal_value + 1);
  __syncthreads();

  // Step3: Copy data from the shared buffers to the output buffer.
  for (int i = offset; i < offset_end; i += block_stride) {
#pragma unroll
    for (int r = 0; r < kMaxNumAllReduceInputPtrs; ++r) {
      if (r >= args.num_ranks) {
        continue;
      }
      // Rotate ranks to circumvent all GPUs reading from the same location
      // simultaneously.
      const int64_t remote_rank = (args.rank + r) % args.num_ranks;
      const int64_t offset_i = remote_rank * args.num_elements_per_rank + i;
      if (offset_i >= args.num_elements) {
        continue;
      }
      VecStore(args.output_buffer + offset_i,
               VecLoad(args.remote_input_buffers[remote_rank] + offset_i));
    }
  }
}

template <typename T, xla::ReductionKind ReductionKindT,
          AllReduceStrategy kAllReduceStrategy>
__global__ void AllReduceKernelImpl(AllReduceKernelParams<T> args) {
  if constexpr (kAllReduceStrategy == AllReduceStrategy::kOneShot) {
    OneShotAllReduceKernelImpl<T, ReductionKindT>(args);
  } else if constexpr (kAllReduceStrategy == AllReduceStrategy::kTwoShot) {
    TwoShotAllReduceKernelImpl<T, ReductionKindT>(args);
  } else {
    assert(false && "Unsupported all-reduce strategy");
  }
}

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_ALL_REDUCE_KERNEL_LIB_CU_H_
