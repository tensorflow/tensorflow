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

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <hipcub/hipcub.hpp>  // NOLINT

#include "rocm/include/hipblaslt/hipblaslt-ext.hpp"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"

namespace stream_executor {
namespace gpu {

__global__ void rocm_Broadcast_fp32Kernel(float* dst, int dst_stride,
                                          int batches, float* src, int size) {
  dst += blockIdx.y * 4 * dst_stride + blockIdx.z * dst_stride * batches;
  src += blockIdx.z * size;
  float* dst2 = dst + dst_stride;
  float* dst3 = dst + dst_stride * 2;
  float* dst4 = dst + dst_stride * 3;
  bool b2 = (blockIdx.y * 4 + 1 < batches);
  bool b3 = (blockIdx.y * 4 + 2 < batches);
  bool b4 = (blockIdx.y * 4 + 3 < batches);
  for (int i = threadIdx.x + blockIdx.x * 256; i < size;
       i += blockDim.x * gridDim.x) {
    dst[i] = src[i];
    if (b2) {
      dst2[i] = src[i];
    }
    if (b3) {
      dst3[i] = src[i];
    }
    if (b4) {
      dst4[i] = src[i];
    }
  }
}

void rocm_Broadcast_fp32(void* stream, float* dst, int dst_stride, int batches,
                         int src_batches, float* src, int size) {
  int x_blocks = (size + 255) / 256;
  hipLaunchKernelGGL(rocm_Broadcast_fp32Kernel,
                     dim3(x_blocks, (batches + 3) / 4, src_batches),
                     min(256, (int)size), 0, (hipStream_t)stream, dst,
                     dst_stride, batches, src, size);
}

__device__ float sigmoid(float x) {
  if (x > 0)
    return 1. / (1. + __expf(-x));
  else
    return __expf(x) / (__expf(x) + 1.);
}

template <typename T, typename Tbias, int act_mode>
__global__ void launchInplaceBiasActivation_kernel(
    T* c_data, const Tbias* bias_data, const T* side_input_data,
    float side_input_scale, uint64_t m, uint64_t n, int64_t ldc, float param,
    int transpose) {
  uint64_t x = threadIdx.x + blockIdx.x * blockDim.x;
  uint64_t y = threadIdx.y + blockIdx.y * blockDim.y;
  uint64_t z = blockIdx.z;
  if (x >= n || y >= m) return;
  float v;
  uint64_t addr = x + y * ldc + z * m * n;
  if (!transpose)
    v = static_cast<float>(c_data[addr]) + static_cast<float>(bias_data[x]);
  else
    v = static_cast<float>(c_data[addr]) + static_cast<float>(bias_data[y]);
  if (side_input_data != 0)
    v += static_cast<float>(side_input_data[addr]) * side_input_scale;
  if (act_mode == 1)
    v = sigmoid(v);
  else if (act_mode == 2)
    v = v > 0.0f ? v : 0.0f;
  else if (act_mode == 3)
    v = v > 0.0f ? (v > 6.0f ? 6.0f : v) : 0.0f;
  else if (act_mode == 4)
    v = v > 0.0f ? (v > param ? param : v) : 0.0f;
  else if (act_mode == 5)
    v = tanh(v);
  else if (act_mode == 6)
    v = v > -param ? (v > param ? param : v) : -param;
  else if (act_mode == 7)
    v = v > 0.0f ? v : __expf(v) - 1;
  else if (act_mode == 8)
    v = v > 0.0f ? v : param * v;
  else if (act_mode == 9)
    v = 0.5 * v * (1 + erf(v / sqrt(2.0f)));
  c_data[addr] = (T)v;
}

template <typename T, typename Tbias>
void launchInplaceBiasActivation(hipStream_t stream, void* c_data,
                                 const void* bias_data,
                                 const void* side_input_data,
                                 float side_input_scale, int activation_mode,
                                 uint64_t batch, uint64_t m, uint64_t n,
                                 int64_t ldc, float param) {
  uint64_t bx = min(n, static_cast<uint64_t>(256));
  uint64_t by = min(m, static_cast<uint64_t>(256) / bx);
  uint64_t gx = (n + bx - 1) / bx;
  uint64_t gy = (m + by - 1) / by;
  int transpose = (activation_mode >= 10);
  activation_mode %= 10;
  auto kernel = launchInplaceBiasActivation_kernel<T, Tbias, 0>;
  if (activation_mode == 1)
    kernel = launchInplaceBiasActivation_kernel<T, Tbias, 1>;
  else if (activation_mode == 2)
    kernel = launchInplaceBiasActivation_kernel<T, Tbias, 2>;
  else if (activation_mode == 3)
    kernel = launchInplaceBiasActivation_kernel<T, Tbias, 3>;
  else if (activation_mode == 4)
    kernel = launchInplaceBiasActivation_kernel<T, Tbias, 4>;
  else if (activation_mode == 5)
    kernel = launchInplaceBiasActivation_kernel<T, Tbias, 5>;
  else if (activation_mode == 6)
    kernel = launchInplaceBiasActivation_kernel<T, Tbias, 6>;
  else if (activation_mode == 7)
    kernel = launchInplaceBiasActivation_kernel<T, Tbias, 7>;
  else if (activation_mode == 8)
    kernel = launchInplaceBiasActivation_kernel<T, Tbias, 8>;
  else if (activation_mode == 9)
    kernel = launchInplaceBiasActivation_kernel<T, Tbias, 9>;

  hipLaunchKernelGGL(kernel, dim3(gx, gy, batch), dim3(bx, by, 1), 0, stream,
                     static_cast<T*>(c_data),
                     static_cast<const Tbias*>(bias_data),
                     static_cast<const T*>(side_input_data), side_input_scale,
                     m, n, ldc, param, transpose);
}

#define INSTANTIATE_BIAS_ACTIVATION(X, Y)                          \
  template void launchInplaceBiasActivation<X, Y>(                 \
      hipStream_t stream, void* c_data, const void* bias_data,     \
      const void* side_input_data, float side_input_scale,         \
      int activation_mode, uint64_t batch, uint64_t m, uint64_t n, \
      int64_t ldc, float param);

INSTANTIATE_BIAS_ACTIVATION(__half, __half)
INSTANTIATE_BIAS_ACTIVATION(__half, float)
INSTANTIATE_BIAS_ACTIVATION(hip_bfloat16, hip_bfloat16)
INSTANTIATE_BIAS_ACTIVATION(hip_bfloat16, float)
INSTANTIATE_BIAS_ACTIVATION(float, float)
INSTANTIATE_BIAS_ACTIVATION(double, double)

};  // namespace gpu

namespace rocm {

constexpr int BLOCK_SIZE = 256;

// Inline device function for copying data from shared memory to global memory
// Uses vectorized uint4 copy for efficiency, with fallback for remaining bytes
__device__ __forceinline__ void copy_shared_to_global(void* shared_src,
                                                      void* global_dest,
                                                      size_t total_bytes) {
  size_t count_uint4 = total_bytes / sizeof(uint4);

  // Vectorized copy using uint4 (16 bytes per iteration)
  uint4* src_ptr = reinterpret_cast<uint4*>(shared_src);
  uint4* dest_ptr = reinterpret_cast<uint4*>(global_dest);

  for (size_t i = threadIdx.x; i < count_uint4; i += blockDim.x) {
    dest_ptr[i] = src_ptr[i];
  }

  // Handle remaining bytes (if total_bytes is not a multiple of 16)
  size_t remaining_bytes = total_bytes % sizeof(uint4);
  if (remaining_bytes > 0) {
    uint8_t* src_ptr = reinterpret_cast<uint8_t*>(shared_src);
    uint8_t* dest_ptr = reinterpret_cast<uint8_t*>(global_dest);
    size_t offset = count_uint4 * sizeof(uint4);

    for (size_t i = threadIdx.x; i < remaining_bytes; i += blockDim.x) {
      dest_ptr[offset + i] = src_ptr[offset + i];
    }
  }
}

template <typename T>
__launch_bounds__(BLOCK_SIZE) __global__
    void SetUserArgsKernelRaggedInNonContractingDim(
        hipblaslt_ext::UserArguments* dest_args, const void* a, const void* b,
        const void* c, void* d, const void* bias, const void* group_sizes,
        uint8_t log2_byte_width_elem_a, uint8_t log2_byte_width_elem_b,
        uint8_t log2_byte_width_elem_c, uint8_t log2_byte_width_elem_d,
        uint32_t stride_a, uint32_t stride_b, uint32_t c_stride_ragged_dim,
        uint32_t output_stride_ragged_dim, bool must_swap_operands, uint32_t m,
        uint32_t n, uint32_t k, uint32_t batch, uint32_t strideA1,
        uint32_t strideA2, uint32_t strideB1, uint32_t strideB2,
        uint32_t strideC1, uint32_t strideC2, uint32_t strideD1,
        uint32_t strideD2, uint32_t num_gemms, int32_t activation_type,
        int8_t bias_type, bool has_matrix_bias) {
  __builtin_assume(num_gemms != 0);
  const T* typed_group_sizes = static_cast<const T*>(group_sizes);

  // Static shared memory for BlockScan temporary storage
  __shared__
      typename hipcub::BlockScan<uint32_t, BLOCK_SIZE>::TempStorage scan_temp;
  // Static shared memory for cumulative offset
  __shared__ uint32_t cumulative_offset;

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
  for (uint64_t batch_start = 0; batch_start < num_gemms;
       batch_start += BLOCK_SIZE) {
    uint32_t idx = batch_start + threadIdx.x;
    // Load group size for this thread (0 if out of bounds)
    uint32_t group_size = (idx < num_gemms) ? typed_group_sizes[idx] : 0;

    // Compute exclusive prefix sum for this batch
    uint32_t offset_in_batch;
    hipcub::BlockScan<uint32_t, BLOCK_SIZE>(scan_temp).ExclusiveSum(
        group_size, offset_in_batch);

    // Add cumulative offset to get global offset
    // On first iteration, cumulative_offset is 0
    uint32_t offset_group = (batch_start == 0)
                                ? offset_in_batch
                                : (cumulative_offset + offset_in_batch);

    // Determine the last active thread in this batch
    uint32_t batch_size =
        min(BLOCK_SIZE, static_cast<uint32_t>(num_gemms - batch_start));

    if (idx < num_gemms) {
      auto& arg = sharedUserArgs[threadIdx.x];
      if (must_swap_operands) {
        // The ragged matrix has been set as operand B.
        arg.n = typed_group_sizes[idx];
        arg.m = m;

        arg.a = const_cast<void*>(static_cast<const void*>(
            static_cast<const uint8_t*>(a) +
            (static_cast<intptr_t>(idx * stride_a) << log2_byte_width_elem_a)));
        arg.b = const_cast<void*>(static_cast<const void*>(
            static_cast<const uint8_t*>(b) +
            (static_cast<intptr_t>(offset_group * stride_b)
             << log2_byte_width_elem_b)));
      } else {
        arg.m = typed_group_sizes[idx];
        arg.n = n;

        arg.a = const_cast<void*>(static_cast<const void*>(
            static_cast<const uint8_t*>(a) +
            (static_cast<intptr_t>(offset_group * stride_a)
             << log2_byte_width_elem_a)));
        arg.b = const_cast<void*>(static_cast<const void*>(
            static_cast<const uint8_t*>(b) +
            (static_cast<intptr_t>(idx * stride_b) << log2_byte_width_elem_b)));
      }

      arg.c = const_cast<void*>(static_cast<const void*>(
          static_cast<const uint8_t*>(c) +
          (static_cast<intptr_t>(offset_group * c_stride_ragged_dim)
           << log2_byte_width_elem_c)));
      arg.d = static_cast<void*>(
          static_cast<uint8_t*>(d) +
          (static_cast<intptr_t>(offset_group * output_stride_ragged_dim)
           << log2_byte_width_elem_d));
      arg.k = k;
      arg.batch = batch;
      arg.strideA1 = strideA1;
      arg.strideA2 = strideA2;
      arg.strideB1 = strideB1;
      arg.strideB2 = strideB2;
      arg.strideC1 = strideC1;
      arg.strideC2 = strideC2;
      arg.strideD1 = strideD1;
      arg.strideD2 = strideD2;
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
      arg.beta[2] = has_matrix_bias ? -128 : 0;
      arg.beta[3] = has_matrix_bias ? 63 : 0;
      arg.scaleA = nullptr;
      arg.scaleB = nullptr;
      arg.scaleC = nullptr;
      arg.scaleD = nullptr;
      arg.scaleAlphaVec = nullptr;
      arg.bias = const_cast<void*>(bias);
      arg.biasType = bias_type;
      arg.e = nullptr;
      // Use activation parameters (always 0.0 from hipBLASLt defaults)
      arg.act0 = 0.0f;
      arg.act1 = 0.0f;
      arg.activationType = activation_type;
    }

    __barrier(__CLK_LOCAL_MEM_FENCE);

    // Last thread updates cumulative offset for next batch
    if (threadIdx.x == batch_size - 1) {
      cumulative_offset += offset_in_batch + group_size;
    }

    // Copy from shared memory to global memory
    size_t total_bytes = batch_size * sizeof(hipblaslt_ext::UserArguments);
    copy_shared_to_global(sharedUserArgs, &dest_args[batch_start], total_bytes);
    // Synchronize before next iteration to ensure copy is complete
    __barrier(__CLK_LOCAL_MEM_FENCE | __CLK_GLOBAL_MEM_FENCE);
  }
}

template <typename T>
__launch_bounds__(BLOCK_SIZE) __global__
    void SetUserArgsKernelRaggedInContractingDim(
        hipblaslt_ext::UserArguments* dest_args, const void* a, const void* b,
        const void* c, void* d, const void* bias, const void* group_sizes,
        uint8_t log2_byte_width_elem_a, uint8_t log2_byte_width_elem_b,
        uint8_t log2_byte_width_elem_c, uint8_t log2_byte_width_elem_d,
        uint32_t stride_a, uint32_t stride_b, uint32_t c_stride_ragged_dim,
        uint32_t output_stride_ragged_dim, bool must_swap_operands, uint32_t m,
        uint32_t n, uint32_t k, uint32_t batch, uint32_t strideA1,
        uint32_t strideA2, uint32_t strideB1, uint32_t strideB2,
        uint32_t strideC1, uint32_t strideC2, uint32_t strideD1,
        uint32_t strideD2, uint32_t num_gemms, int32_t activation_type,
        int8_t bias_type, bool has_matrix_bias) {
  __builtin_assume(num_gemms != 0);
  const T* typed_group_sizes = static_cast<const T*>(group_sizes);

  // Static shared memory for BlockScan temporary storage
  __shared__
      typename hipcub::BlockScan<uint32_t, BLOCK_SIZE>::TempStorage scan_temp;
  // Static shared memory for cumulative offset
  __shared__ uint32_t cumulative_offset;

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
  for (uint64_t batch_start = 0; batch_start < num_gemms;
       batch_start += BLOCK_SIZE) {
    uint32_t idx = batch_start + threadIdx.x;

    // Load group size for this thread (0 if out of bounds)
    uint32_t group_size = (idx < num_gemms) ? typed_group_sizes[idx] : 0;

    // Compute exclusive prefix sum for this batch
    uint32_t offset_in_batch;
    hipcub::BlockScan<uint32_t, BLOCK_SIZE>(scan_temp).ExclusiveSum(
        group_size, offset_in_batch);

    // Add cumulative offset to get global offset
    // On first iteration, cumulative_offset is 0
    uint32_t offset_group = (batch_start == 0)
                                ? offset_in_batch
                                : (cumulative_offset + offset_in_batch);

    // Determine the last active thread in this batch
    uint32_t batch_size =
        min(BLOCK_SIZE, static_cast<uint32_t>(num_gemms - batch_start));

    if (idx < num_gemms) {
      auto& arg = sharedUserArgs[threadIdx.x];

      arg.m = m;
      arg.n = n;
      arg.a = const_cast<void*>(static_cast<const void*>(
          static_cast<const uint8_t*>(a) +
          (static_cast<intptr_t>(offset_group * stride_a)
           << log2_byte_width_elem_a)));
      arg.b = const_cast<void*>(static_cast<const void*>(
          static_cast<const uint8_t*>(b) +
          (static_cast<intptr_t>(offset_group * stride_b)
           << log2_byte_width_elem_b)));
      arg.c = const_cast<void*>(static_cast<const void*>(
          static_cast<const uint8_t*>(c) +
          (static_cast<intptr_t>(idx * c_stride_ragged_dim)
           << log2_byte_width_elem_c)));
      arg.d = const_cast<void*>(static_cast<void*>(
          static_cast<uint8_t*>(d) +
          (static_cast<intptr_t>(idx * output_stride_ragged_dim)
           << log2_byte_width_elem_d)));
      arg.k = typed_group_sizes[idx];
      arg.batch = batch;
      arg.strideA1 = strideA1;
      arg.strideA2 = strideA2;
      arg.strideB1 = strideB1;
      arg.strideB2 = strideB2;
      arg.strideC1 = strideC1;
      arg.strideC2 = strideC2;
      arg.strideD1 = strideD1;
      arg.strideD2 = strideD2;
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
      arg.beta[2] = has_matrix_bias ? -128 : 0;
      arg.beta[3] = has_matrix_bias ? 63 : 0;
      arg.scaleA = nullptr;
      arg.scaleB = nullptr;
      arg.scaleC = nullptr;
      arg.scaleD = nullptr;
      arg.scaleAlphaVec = nullptr;
      arg.bias = const_cast<void*>(bias);
      arg.biasType = bias_type;
      arg.e = nullptr;
      // Use activation parameters (always 0.0 from hipBLASLt defaults)
      arg.act0 = 0.0f;
      arg.act1 = 0.0f;
      arg.activationType = activation_type;
    }

    __barrier(__CLK_LOCAL_MEM_FENCE);

    // Last thread updates cumulative offset for next batch
    if (threadIdx.x == batch_size - 1) {
      cumulative_offset += offset_in_batch + group_size;
    }

    // Copy from shared memory to global memory
    size_t total_bytes = batch_size * sizeof(hipblaslt_ext::UserArguments);
    copy_shared_to_global(sharedUserArgs, &dest_args[batch_start], total_bytes);
    // Synchronize before next iteration to ensure copy is complete
    __barrier(__CLK_LOCAL_MEM_FENCE | __CLK_GLOBAL_MEM_FENCE);
  }
}

template <typename T>
__launch_bounds__(BLOCK_SIZE) __global__ void SetUserArgsKernelRaggedInBatchDim(
    hipblaslt_ext::UserArguments* dest_args, const void* a, const void* b,
    const void* c, void* d, const void* bias, const void* group_sizes,
    uint8_t log2_byte_width_elem_a, uint8_t log2_byte_width_elem_b,
    uint8_t log2_byte_width_elem_c, uint8_t log2_byte_width_elem_d,
    uint32_t stride_a, uint32_t stride_b, uint32_t c_stride_ragged_dim,
    uint32_t output_stride_ragged_dim, bool must_swap_operands, uint32_t m,
    uint32_t n, uint32_t k, uint32_t batch, uint32_t strideA1,
    uint32_t strideA2, uint32_t strideB1, uint32_t strideB2, uint32_t strideC1,
    uint32_t strideC2, uint32_t strideD1, uint32_t strideD2, uint32_t num_gemms,
    int32_t activation_type, int8_t bias_type, bool has_matrix_bias) {
  __builtin_assume(num_gemms != 0);
  const T* typed_group_sizes = static_cast<const T*>(group_sizes);

  // Static shared memory for BlockScan temporary storage
  __shared__
      typename hipcub::BlockScan<uint32_t, BLOCK_SIZE>::TempStorage scan_temp;
  // Static shared memory for cumulative offset
  __shared__ uint32_t cumulative_offset;

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
  for (uint64_t batch_start = 0; batch_start < num_gemms;
       batch_start += BLOCK_SIZE) {
    uint32_t idx = batch_start + threadIdx.x;

    // Load group size for this thread (0 if out of bounds)
    uint32_t group_size = (idx < num_gemms) ? typed_group_sizes[idx] : 0;

    // Compute exclusive prefix sum for this batch
    uint32_t offset_in_batch;
    hipcub::BlockScan<uint32_t, BLOCK_SIZE>(scan_temp).ExclusiveSum(
        group_size, offset_in_batch);

    // Add cumulative offset to get global offset
    // On first iteration, cumulative_offset is 0
    uint32_t offset_group = (batch_start == 0)
                                ? offset_in_batch
                                : (cumulative_offset + offset_in_batch);

    // Determine the last active thread in this batch
    uint32_t batch_size =
        min(BLOCK_SIZE, static_cast<uint32_t>(num_gemms - batch_start));

    if (idx < num_gemms) {
      auto& arg = sharedUserArgs[threadIdx.x];

      arg.m = m;
      arg.n = n;
      arg.a = const_cast<void*>(static_cast<const void*>(
          static_cast<const uint8_t*>(a) +
          (static_cast<intptr_t>(offset_group * stride_a)
           << log2_byte_width_elem_a)));
      arg.b = const_cast<void*>(static_cast<const void*>(
          static_cast<const uint8_t*>(b) +
          (static_cast<intptr_t>(offset_group * stride_b)
           << log2_byte_width_elem_b)));
      arg.c = const_cast<void*>(static_cast<const void*>(
          static_cast<const uint8_t*>(c) +
          (static_cast<intptr_t>(offset_group * c_stride_ragged_dim)
           << log2_byte_width_elem_c)));
      arg.d = static_cast<void*>(
          static_cast<uint8_t*>(d) +
          (static_cast<intptr_t>(offset_group * output_stride_ragged_dim)
           << log2_byte_width_elem_d));
      arg.k = k;
      arg.batch = typed_group_sizes[idx];
      arg.strideA1 = strideA1;
      arg.strideA2 = strideA2;
      arg.strideB1 = strideB1;
      arg.strideB2 = strideB2;
      arg.strideC1 = strideC1;
      arg.strideC2 = strideC2;
      arg.strideD1 = strideD1;
      arg.strideD2 = strideD2;
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
      arg.beta[2] = has_matrix_bias ? -128 : 0;
      arg.beta[3] = has_matrix_bias ? 63 : 0;
      arg.scaleA = nullptr;
      arg.scaleB = nullptr;
      arg.scaleC = nullptr;
      arg.scaleD = nullptr;
      arg.scaleAlphaVec = nullptr;
      arg.bias = const_cast<void*>(bias);
      arg.biasType = bias_type;
      arg.e = nullptr;
      // Use activation parameters (always 0.0 from hipBLASLt defaults)
      arg.act0 = 0.0f;
      arg.act1 = 0.0f;
      arg.activationType = activation_type;
    }

    __barrier(__CLK_LOCAL_MEM_FENCE);

    // Last thread updates cumulative offset for next batch
    if (threadIdx.x == batch_size - 1) {
      cumulative_offset += offset_in_batch + group_size;
    }

    // Copy from shared memory to global memory
    size_t total_bytes = batch_size * sizeof(hipblaslt_ext::UserArguments);
    copy_shared_to_global(sharedUserArgs, &dest_args[batch_start], total_bytes);
    // Synchronize before next iteration to ensure copy is complete
    __barrier(__CLK_LOCAL_MEM_FENCE | __CLK_GLOBAL_MEM_FENCE);
  }
}

void GroupGemmUpdateArgs(
    hipStream_t stream, DeviceAddressBase args, DeviceAddressBase a,
    DeviceAddressBase b, DeviceAddressBase c, DeviceAddressBase d,
    DeviceAddressBase bias, DeviceAddressBase group_sizes,
    uint8_t group_size_bytewidth, uint8_t log2_byte_width_elem_a,
    uint8_t log2_byte_width_elem_b, uint8_t log2_byte_width_elem_c,
    uint8_t log2_byte_width_elem_d, uint32_t stride_ragged_dim,
    uint32_t stride_group_dim, uint32_t c_stride_ragged_dim,
    uint32_t output_stride_ragged_dim, bool must_swap_operands, uint32_t m,
    uint32_t n, uint32_t k, uint32_t batch, uint32_t strideA1,
    uint32_t strideA2, uint32_t strideB1, uint32_t strideB2, uint32_t strideC1,
    uint32_t strideC2, uint32_t strideD1, uint32_t strideD2,
    gpu::RaggedDotMode ragged_mode, uint32_t num_gemms, int32_t activation_type,
    int8_t bias_type, bool has_matrix_bias) {
  const uint32_t block_sz = BLOCK_SIZE;
  auto kernel = SetUserArgsKernelRaggedInNonContractingDim<uint64_t>;
  switch (ragged_mode) {
    case gpu::RaggedDotMode::kRaggedNonContracting: {
      if (group_size_bytewidth == 4) {
        kernel = SetUserArgsKernelRaggedInNonContractingDim<uint32_t>;
      }
      break;
    }
    case gpu::RaggedDotMode::kRaggedContracting: {
      kernel = SetUserArgsKernelRaggedInContractingDim<uint64_t>;
      if (group_size_bytewidth == 4) {
        kernel = SetUserArgsKernelRaggedInContractingDim<uint32_t>;
      }
      break;
    }
    case gpu::RaggedDotMode::kRaggedBatch: {
      kernel = SetUserArgsKernelRaggedInBatchDim<uint64_t>;
      if (group_size_bytewidth == 4) {
        kernel = SetUserArgsKernelRaggedInBatchDim<uint32_t>;
      }
      break;
    }
  }
  auto stride_a = stride_ragged_dim;
  auto stride_b = stride_group_dim;
  if (must_swap_operands) {
    std::swap(stride_a, stride_b);
  }

  // Calculate dynamic shared memory size for UserArguments array
  size_t shared_mem_size =
      min(block_sz, num_gemms) * sizeof(hipblaslt_ext::UserArguments);

  hipLaunchKernelGGL(
      kernel, dim3(1), dim3(block_sz), shared_mem_size, stream,
      static_cast<hipblaslt_ext::UserArguments*>(args.opaque()), a.opaque(),
      b.opaque(), c.opaque(), d.opaque(), bias.opaque(), group_sizes.opaque(),
      log2_byte_width_elem_a, log2_byte_width_elem_b, log2_byte_width_elem_c,
      log2_byte_width_elem_d, stride_a, stride_b, c_stride_ragged_dim,
      output_stride_ragged_dim, must_swap_operands, m, n, k, batch, strideA1,
      strideA2, strideB1, strideB2, strideC1, strideC2, strideD1, strideD2,
      num_gemms, activation_type, bias_type, has_matrix_bias);
}
};  // namespace rocm

};  // namespace stream_executor
