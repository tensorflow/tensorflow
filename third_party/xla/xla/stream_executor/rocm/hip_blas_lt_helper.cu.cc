/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/stream_executor/rocm/hip_blas_lt_helper.h"

#include <cstdint>
#include <utility>

#include "absl/log/log.h"
#include "rocm/include/hip/hip_runtime.h"
#include "rocm/include/hipblaslt/hipblaslt-ext.hpp"
#include "rocm/include/hipcub/hipcub.hpp"

namespace stream_executor {

namespace rocm {

constexpr int BLOCK_SIZE = 256;

// Inline device function for copying data from shared memory to global memory
// Uses vectorized uint4 copy for efficiency, with fallback for remaining bytes
__device__ __forceinline__ void copy_shared_to_global(const void* shared_src,
                                                      void* global_dest,
                                                      size_t total_bytes) {
  size_t count_uint4 = total_bytes / sizeof(uint4);

  // Vectorized copy using uint4 (16 bytes per iteration)
  const uint4* src_ptr = reinterpret_cast<const uint4*>(shared_src);
  uint4* dest_ptr = reinterpret_cast<uint4*>(global_dest);

  for (size_t i = threadIdx.x; i < count_uint4; i += blockDim.x) {
    dest_ptr[i] = src_ptr[i];
  }

  // Handle remaining bytes (if total_bytes is not a multiple of 16)
  const uint8_t* src8 = reinterpret_cast<const uint8_t*>(shared_src);
  uint8_t* dest8 = reinterpret_cast<uint8_t*>(global_dest);
  size_t ofs = count_uint4 * sizeof(uint4);
  for (size_t i = ofs + threadIdx.x; i < total_bytes; i += blockDim.x) {
    dest8[i] = src8[i];
  }
}

// Plain-old-data view of the parameters a UserArguments-update kernel needs.
// Built on the host from GroupGemmUpdateArgsParams and passed by value, which
// keeps the kernel signature to a single argument.
struct GroupGemmKernelArgs {
  hipblaslt_ext::UserArguments* dest_args;
  const void *a, *b, *c;
  void* d;
  const void *bias, *group_sizes;
  // Byte strides (element stride premultiplied by the operand's element width),
  // with stride_a/stride_b already swapped on the host when must_swap_operands.
  uint64_t stride_a, stride_b, c_stride_ragged_dim, output_stride_ragged_dim;
  bool must_swap_operands;
  uint32_t m, n, k, batch;
  uint32_t strideA1, strideA2, strideB1, strideB2, strideC1, strideC2, strideD1,
      strideD2;
  uint32_t num_gemms;
  int32_t activation_type, bias_type;
  bool has_matrix_bias;
};

// Returns base + index * stride_bytes as a mutable pointer. stride_bytes is a
// byte stride (the element stride already premultiplied by the element width on
// the host), so no per-element width shift is needed here.
__device__ __forceinline__ void* OffsetPtr(const void* base, uint64_t index,
                                           uint64_t stride_bytes) {
  return const_cast<void*>(static_cast<const void*>(
      static_cast<const uint8_t*>(base) + index * stride_bytes));
}

// Fills the fields of a UserArguments entry that are identical across all
// ragged-dot modes (strides, alpha/beta, scales, bias, activation).
// The SetUserArgsKernel kernels below build hipblaslt_ext::UserArguments in
// uninitialized LDS and copy whole structs to global memory, so every field
// must be written.
__device__ __forceinline__ void FillCommonUserArgs(
    hipblaslt_ext::UserArguments& arg, const GroupGemmKernelArgs& p) {
  arg.strideA1 = p.strideA1;
  arg.strideA2 = p.strideA2;
  arg.strideB1 = p.strideB1;
  arg.strideB2 = p.strideB2;
  arg.strideC1 = p.strideC1;
  arg.strideC2 = p.strideC2;
  arg.strideD1 = p.strideD1;
  arg.strideD2 = p.strideD2;
  arg.strideE1 = 0;
  arg.strideE2 = 0;
  // Set alpha and beta from hipBLASLt defaults
  for (int8_t i = 0; i < 16; i++) {
    arg.alpha[i] = 0;
    arg.beta[i] = 0;
  }
  // Alpha is always 1.0 (represented as -128, 63 in packed format)
  arg.alpha[2] = -128;
  arg.alpha[3] = 63;
  // Beta is 0.0 (0, 0) for no matrix bias or 1.0 (-128, 63) for matrix bias
  arg.beta[2] = p.has_matrix_bias ? -128 : 0;
  arg.beta[3] = p.has_matrix_bias ? 63 : 0;
  arg.scaleA = nullptr;
  arg.scaleB = nullptr;
  arg.scaleC = nullptr;
  arg.scaleD = nullptr;
  arg.scaleAlphaVec = nullptr;
  arg.bias = const_cast<void*>(p.bias);
  arg.biasType = p.bias_type;
  arg.reserved = 0;
  arg.e = nullptr;
  // Use activation parameters (always 0.0 from hipBLASLt defaults)
  arg.act0 = 0.0f;
  arg.act1 = 0.0f;
  arg.activationType = p.activation_type;
}

// Populates the device-side hipBLASLt UserArguments array for a grouped/
// ragged-dot GEMM. The three ragged modes share the block-scan prefix-sum
// loop and the common UserArguments fill; only the m/n/k/batch assignment and
// the index (per-gemm idx vs. running prefix-sum offset) that drives each
// pointer offset differ, which is handled with `if constexpr (Mode)`.
template <typename T, gpu::RaggedDotMode Mode>
__launch_bounds__(BLOCK_SIZE) __global__
    void SetUserArgsKernel(GroupGemmKernelArgs p) {
  __builtin_assume(p.num_gemms != 0);
  const T* typed_group_sizes = static_cast<const T*>(p.group_sizes);

  // Static shared memory for BlockScan temporary storage
  __shared__
      typename hipcub::BlockScan<uint32_t, BLOCK_SIZE>::TempStorage scan_temp;
  // Static shared memory for the running carry of the global prefix sum.
  // 64-bit so the total ragged-dimension extent can exceed 2^32 elements.
  __shared__ uint64_t cumulative_offset;

  // Dynamic shared memory for UserArguments array
  extern __shared__ uint8_t shared_mem[];
  auto* sharedUserArgs =
      reinterpret_cast<hipblaslt_ext::UserArguments*>(shared_mem);

  // Last thread initialize cumulative offset
  // No need to syncthread here as this variable will be
  // updated later by the last thread before
  // being read by other threads.
  if (threadIdx.x == BLOCK_SIZE - 1) {
    cumulative_offset = 0;
  }

  // Process all elements in batches of BLOCK_SIZE
  for (uint32_t batch_start = 0; batch_start < p.num_gemms;
       batch_start += BLOCK_SIZE) {
    uint32_t idx = batch_start + threadIdx.x;
    // Load group size for this thread (0 if out of bounds)
    uint32_t group_size = (idx < p.num_gemms) ? typed_group_sizes[idx] : 0;

    // Compute exclusive prefix sum for this batch
    uint32_t offset_in_batch;
    hipcub::BlockScan<uint32_t, BLOCK_SIZE>(scan_temp).ExclusiveSum(
        group_size, offset_in_batch);

    // Global exclusive offset = carry from previous chunks + in-chunk offset.
    // cumulative_offset is initialized to 0 and made visible by the barrier
    // inside the ExclusiveSum above, so no special case is needed for the first
    // chunk.
    uint64_t offset_group = cumulative_offset + offset_in_batch;

    // Determine the last active thread in this batch
    uint32_t batch_size = min(BLOCK_SIZE, p.num_gemms - batch_start);

    if (idx < p.num_gemms) {
      auto& arg = sharedUserArgs[threadIdx.x];

      if constexpr (Mode == gpu::RaggedDotMode::kRaggedNonContracting) {
        if (p.must_swap_operands) {
          // The ragged matrix has been set as operand B.
          arg.n = typed_group_sizes[idx];
          arg.m = p.m;
          arg.a = OffsetPtr(p.a, idx, p.stride_a);
          arg.b = OffsetPtr(p.b, offset_group, p.stride_b);
        } else {
          arg.m = typed_group_sizes[idx];
          arg.n = p.n;
          arg.a = OffsetPtr(p.a, offset_group, p.stride_a);
          arg.b = OffsetPtr(p.b, idx, p.stride_b);
        }
        arg.c = OffsetPtr(p.c, offset_group, p.c_stride_ragged_dim);
        arg.d = OffsetPtr(p.d, offset_group, p.output_stride_ragged_dim);
        arg.k = p.k;
        arg.batch = p.batch;
      } else if constexpr (Mode == gpu::RaggedDotMode::kRaggedContracting) {
        arg.m = p.m;
        arg.n = p.n;
        arg.a = OffsetPtr(p.a, offset_group, p.stride_a);
        arg.b = OffsetPtr(p.b, offset_group, p.stride_b);
        arg.c = OffsetPtr(p.c, idx, p.c_stride_ragged_dim);
        arg.d = OffsetPtr(p.d, idx, p.output_stride_ragged_dim);
        arg.k = typed_group_sizes[idx];
        arg.batch = p.batch;
      } else {  // gpu::RaggedDotMode::kRaggedBatch
        arg.m = p.m;
        arg.n = p.n;
        arg.a = OffsetPtr(p.a, offset_group, p.stride_a);
        arg.b = OffsetPtr(p.b, offset_group, p.stride_b);
        arg.c = OffsetPtr(p.c, offset_group, p.c_stride_ragged_dim);
        arg.d = OffsetPtr(p.d, offset_group, p.output_stride_ragged_dim);
        arg.k = p.k;
        arg.batch = typed_group_sizes[idx];
      }

      FillCommonUserArgs(arg, p);
    }  // if (idx < p.num_gemms)

    __barrier(__CLK_LOCAL_MEM_FENCE);

    // Last thread updates cumulative offset for next batch
    if (threadIdx.x == batch_size - 1) {
      cumulative_offset += offset_in_batch + group_size;
    }

    // Copy from shared memory to global memory
    size_t total_bytes = batch_size * sizeof(hipblaslt_ext::UserArguments);
    copy_shared_to_global(sharedUserArgs, &p.dest_args[batch_start],
                          total_bytes);
    // Synchronize before next iteration to ensure copy is complete
    __barrier(__CLK_LOCAL_MEM_FENCE | __CLK_GLOBAL_MEM_FENCE);
  }  // for
}

void GroupGemmUpdateArgs(hipStream_t stream,
                         const GroupGemmUpdateArgsParams& p) {
#define DISPATCH_KERNEL(Mode)                                         \
  case gpu::RaggedDotMode::Mode:                                      \
    if (p.group_size_bytewidth == 4) {                                \
      kernel = SetUserArgsKernel<uint32_t, gpu::RaggedDotMode::Mode>; \
    } else {                                                          \
      kernel = SetUserArgsKernel<uint64_t, gpu::RaggedDotMode::Mode>; \
    }                                                                 \
    break;

  using KernelFn = void (*)(GroupGemmKernelArgs);
  KernelFn kernel = nullptr;
  switch (p.ragged_mode) {
    DISPATCH_KERNEL(kRaggedNonContracting)
    DISPATCH_KERNEL(kRaggedContracting)
    DISPATCH_KERNEL(kRaggedBatch)
    default:
      LOG(FATAL) << "Invalid ragged mode: " << static_cast<int>(p.ragged_mode);
  }
#undef DISPATCH_KERNEL
  // clang-format off
  GroupGemmKernelArgs kargs{
      static_cast<hipblaslt_ext::UserArguments*>(p.args.opaque()),
      p.a.opaque(), p.b.opaque(),
      p.c.opaque(), p.d.opaque(),
      p.bias.opaque(), p.group_sizes.opaque(),
      p.stride_a_bytes, p.stride_b_bytes,
      p.c_stride_ragged_dim_bytes,
      p.output_stride_ragged_dim_bytes,
      p.must_swap_operands,
      p.m, p.n, p.k, p.batch,
      p.strideA1, p.strideA2,
      p.strideB1, p.strideB2,
      p.strideC1, p.strideC2,
      p.strideD1, p.strideD2,
      p.num_gemms,
      p.activation_type,
      p.bias_type,
      p.has_matrix_bias,
  };
  // clang-format on
  // Calculate dynamic shared memory size for UserArguments array
  const uint32_t block_sz = BLOCK_SIZE;
  size_t shared_mem_size =
      min(block_sz, p.num_gemms) * sizeof(hipblaslt_ext::UserArguments);

  hipLaunchKernelGGL(kernel, dim3(1), dim3(block_sz), shared_mem_size, stream,
                     kargs);
}

}  // namespace rocm
}  // namespace stream_executor
