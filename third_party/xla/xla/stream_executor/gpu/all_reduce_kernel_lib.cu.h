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

#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/gpu/all_reduce_kernel.h"
#include "xla/stream_executor/gpu/collective_signal.cu.h"

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

template <typename T>
__device__ __forceinline__ RestrictedPtr<T> GetPeerPtr(
    void* ptr, int64_t peer_rank, int64_t argument_index, int num_ranks,
    const CollectiveKernelMetadata& metadata) {
  uint64_t argument_offset = num_ranks * argument_index;
  uint64_t current_base =
      (uint64_t)metadata.param_to_peers[argument_offset + metadata.rank];
  uint64_t peer_base =
      (uint64_t)metadata.param_to_peers[argument_offset + peer_rank];
  uint64_t offset = (uint64_t)ptr - current_base;

  return (RestrictedPtr<T>)(peer_base + offset);
}

template <typename T>
__device__ __forceinline__ RestrictedPtr<T> GetMultimemPtr(
    void* ptr, int64_t argument_index, int num_ranks,
    const CollectiveKernelMetadata& metadata) {
  uint64_t argument_offset = num_ranks * argument_index;
  uint64_t current_base =
      (uint64_t)metadata.param_to_peers[argument_offset + metadata.rank];
  uint64_t offset = (uint64_t)ptr - current_base;

  return (RestrictedPtr<T>)((uint64_t)metadata
                                .param_to_multimem_addresses[argument_index] +
                            offset);
}

template <typename T, xla::ReductionKind ReductionKindT, PlatformType PlatformT>
__device__ __forceinline__ void OneShotAllReduceKernelImpl(
    const AllReduceKernelParams<T>& args) {
  __shared__ std::array<RestrictedPtr<uint32_t>, kMaxNumAllReduceInputPtrs>
      signal_flags_buffers;
  __shared__ std::array<RestrictedPtr<T>, kMaxNumAllReduceInputPtrs>
      remote_input_buffers;

  if (threadIdx.x < args.num_ranks) {
    remote_input_buffers[threadIdx.x] =
        GetPeerPtr<T>(args.symmetric_input_ptrs, threadIdx.x,
                      /*argument_index=*/0, args.num_ranks, *args.metadata);
    signal_flags_buffers[threadIdx.x] = GetPeerPtr<uint32_t>(
        args.symmetric_signal_ptrs, threadIdx.x, /*argument_index=*/1,
        args.num_ranks, *args.metadata);
  }

  __syncthreads();

  int64_t offset =
      kNumElementsPerThread * (blockIdx.x * blockDim.x + threadIdx.x);
  int64_t stride = kNumElementsPerThread * blockDim.x * gridDim.x;

  // Copy data from local input buffer to remote input buffer.
  for (int i = offset; i < args.num_elements; i += stride) {
    VecStore(args.symmetric_input_ptrs + i, VecLoad(args.input_buffer + i));
  }

  SyncRemoteBlocks<PlatformT, kMaxNumAllReduceInputPtrs>(
      signal_flags_buffers, args.rank, args.num_ranks, args.signal_value);
  __syncthreads();

  for (int i = offset; i < args.num_elements; i += stride) {
    Vec<T> acc = VecLoad(remote_input_buffers[0] + i);

    // Since `remote_input_buffers` are provided in rank order, we get stable
    // reduction results on all devices.
#pragma unroll
    for (int j = 1; j < kMaxNumAllReduceInputPtrs; ++j) {
      if (j < args.num_ranks) {
        VecOp<T, ReductionKindT>(acc, VecLoad(remote_input_buffers[j] + i));
      }
    }

    VecStore(args.output_buffer + i, acc);
  }
}

#if __CUDA_ARCH__ >= 900

// This is the simplest implementation of all-reduce with multimem instructions.
// Right now all devices are copying their data to the remote buffer after
// which, the first device performs the reduce and broadcast operations using
// multimem instructions.
template <typename T, xla::ReductionKind ReductionKindT, PlatformType PlatformT>
__device__ __forceinline__ void MultimemAllReduceKernelImpl(
    const AllReduceKernelParams<T>& args) {
  if (!std::is_same_v<T, float>) {
    assert(false &&
           "Multimem all-reduce strategy is only supported for float.");
  }

  __shared__ std::array<RestrictedPtr<uint32_t>, kMaxNumAllReduceInputPtrs>
      signal_flags_buffers;

  if (threadIdx.x < args.num_ranks) {
    signal_flags_buffers[threadIdx.x] = GetPeerPtr<uint32_t>(
        args.symmetric_signal_ptrs, threadIdx.x, /*argument_index=*/1,
        args.num_ranks, *args.metadata);
  }

  int64_t offset =
      kNumElementsPerThread * (blockIdx.x * blockDim.x + threadIdx.x);
  int64_t stride = kNumElementsPerThread * blockDim.x * gridDim.x;

  __syncthreads();
  SyncRemoteBlocks<PlatformT, kMaxNumAllReduceInputPtrs>(
      signal_flags_buffers, args.rank, args.num_ranks, args.signal_value);
  __syncthreads();

  RestrictedPtr<T> src_multimem =
      GetMultimemPtr<T>(args.input_buffer, 0, args.num_ranks, *args.metadata);
  RestrictedPtr<T> dst_multimem =
      GetMultimemPtr<T>(args.output_buffer, 2, args.num_ranks, *args.metadata);
  if (args.metadata->rank == 0) {
    for (int i = offset; i < args.num_elements; i += stride) {
      T* src_multimem_element_ptr = src_multimem + i;
      T* dst_multimem_element_ptr = dst_multimem + i;

      // Reduce
      Vec<T> vec;
      asm volatile(
          "multimem.ld_reduce.relaxed.sys.global.add.v4.f32 {%0,%1,%2,%3}, "
          "[%4];"
          : "=f"(vec.data[0]), "=f"(vec.data[1]), "=f"(vec.data[2]),
            "=f"(vec.data[3])
          : "l"(src_multimem_element_ptr)
          : "memory");

      // Broadcast
      asm volatile(
          "multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(
              dst_multimem_element_ptr),
          "f"(vec.data[0]), "f"(vec.data[1]), "f"(vec.data[2]), "f"(vec.data[3])
          : "memory");
    }
  }

  __syncthreads();
  // Wait for all participants to receive the data.
  SyncRemoteBlocks<PlatformT, kMaxNumAllReduceInputPtrs>(
      signal_flags_buffers, args.rank, args.num_ranks, args.signal_value + 1);
  __syncthreads();
}
#endif  // __CUDA_ARCH__ >= 900

template <typename T, xla::ReductionKind ReductionKindT, PlatformType PlatformT>
__device__ __forceinline__ void TwoShotAllReduceKernelImpl(
    const AllReduceKernelParams<T>& args) {
  __shared__ std::array<RestrictedPtr<uint32_t>, kMaxNumAllReduceInputPtrs>
      signal_flags_buffers;
  __shared__ std::array<RestrictedPtr<T>, kMaxNumAllReduceInputPtrs>
      remote_input_buffers;

  if (threadIdx.x < args.num_ranks) {
    remote_input_buffers[threadIdx.x] =
        GetPeerPtr<T>(args.symmetric_input_ptrs, threadIdx.x,
                      /*argument_index=*/0, args.num_ranks, *args.metadata);
    signal_flags_buffers[threadIdx.x] = GetPeerPtr<uint32_t>(
        args.symmetric_signal_ptrs, threadIdx.x, /*argument_index=*/1,
        args.num_ranks, *args.metadata);
  }

  __syncthreads();

  const int64_t offset = blockIdx.x * args.num_elements_per_block +
                         threadIdx.x * kNumElementsPerThread;
  const int64_t offset_end =
      std::min((blockIdx.x + 1) * args.num_elements_per_block,
               args.num_elements_per_rank);

  const int64_t block_stride = kNumElementsPerThread * blockDim.x;

  // Step1: Copy data from input buffer to the local shared buffer.
  // Each GPU will copy data from its local input buffer to its own local shared
  // buffer from where it will be read by participating devices (PULLed).
  for (int i = offset; i < offset_end; i += block_stride) {
#pragma unroll
    for (int j = 0; j < kMaxNumAllReduceInputPtrs; ++j) {
      if (j >= args.num_ranks) {
        continue;
      }
      const int64_t offset_i = j * args.num_elements_per_rank + i;
      if (offset_i >= args.num_elements) {
        continue;
      }
      VecStore(remote_input_buffers[args.rank] + offset_i,
               VecLoad(args.input_buffer + offset_i));
    }
  }

  // Shot1: Wait for all participating devices to finish copying data to their
  // shared buffer.
  __syncthreads();  // Make sure all writes to shared buffers are complete.
  SyncRemoteBlocks<PlatformT, kMaxNumAllReduceInputPtrs>(
      signal_flags_buffers, args.rank, args.num_ranks, args.signal_value);
  __syncthreads();  // Block must wait here until remote signals are updated.

  // Step2: Accumulate data for the responsible indices in the shared buffers.
  for (int i = offset; i < offset_end; i += block_stride) {
    // Each rank is only responsible for accumulating num_elements_per_rank
    // elements.
    const int64_t offset_i = args.rank_offset + i;
    if (offset_i >= args.num_elements) {
      continue;
    }
    std::array<Vec<T>, kMaxNumAllReduceInputPtrs> accs;
#pragma unroll
    for (int r = 0; r < kMaxNumAllReduceInputPtrs; ++r) {
      if (r >= args.num_ranks) {
        continue;
      }
      accs[args.rotated_ranks[r]] =
          VecLoad(remote_input_buffers[args.rotated_ranks[r]] + offset_i);
    }

    Vec<T> acc = accs[0];
    // Since `remote_input_ptrs` are provided in rank order, we get stable
    // reduction results on all devices.
#pragma unroll
    for (int r = 1; r < kMaxNumAllReduceInputPtrs; ++r) {
      if (r >= args.num_ranks) {
        continue;
      }
      VecOp<T, ReductionKindT>(acc, accs[r]);
    }
    VecStore(remote_input_buffers[args.rank] + offset_i, acc);
  }

  // Shot2: Wait for all participating devices to finish accumulating data in
  // the shared buffer. Note that signal_value + 1 is used to ensure that the
  // synchronization is different from the one used above.
  __syncthreads();  // Wait for all accumulations to shared buffer.
  SyncRemoteBlocks<PlatformT, kMaxNumAllReduceInputPtrs>(
      signal_flags_buffers, args.rank, args.num_ranks, args.signal_value + 1);
  __syncthreads();  // Block must wait here until remote signals are updated.

  // Step3: Copy data from the shared buffers to the output buffer.
  for (int i = offset; i < offset_end; i += block_stride) {
#pragma unroll
    for (int r = 0; r < kMaxNumAllReduceInputPtrs; ++r) {
      if (r >= args.num_ranks) {
        continue;
      }
      const int64_t offset_i =
          args.rotated_ranks[r] * args.num_elements_per_rank + i;
      if (offset_i >= args.num_elements) {
        continue;
      }
      VecStore(args.output_buffer + offset_i,
               VecLoad(remote_input_buffers[args.rotated_ranks[r]] + offset_i));
    }
  }
}

template <typename T, xla::ReductionKind ReductionKindT,
          AllReduceStrategy kAllReduceStrategy, PlatformType PlatformT>
__global__ void AllReduceKernelImpl(AllReduceKernelParams<T> args) {
  if constexpr (kAllReduceStrategy == AllReduceStrategy::kOneShot) {
    OneShotAllReduceKernelImpl<T, ReductionKindT, PlatformT>(args);
  } else if constexpr (kAllReduceStrategy == AllReduceStrategy::kTwoShot) {
    TwoShotAllReduceKernelImpl<T, ReductionKindT, PlatformT>(args);
  } else if constexpr (kAllReduceStrategy == AllReduceStrategy::kMultimem) {
#if __CUDA_ARCH__ >= 900
    MultimemAllReduceKernelImpl<T, ReductionKindT, PlatformT>(args);
#else
    assert(false &&
           "Multimem all-reduce strategy is not supported on this "
           "architecture.");
#endif
  } else {
    assert(false && "Unsupported all-reduce strategy");
  }
}

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_ALL_REDUCE_KERNEL_LIB_CU_H_
