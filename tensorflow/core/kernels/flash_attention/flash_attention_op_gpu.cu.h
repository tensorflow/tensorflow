/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_FLASH_ATTENTION_FLASH_ATTENTION_OP_GPU_CU_H_
#define TENSORFLOW_CORE_KERNELS_FLASH_ATTENTION_FLASH_ATTENTION_OP_GPU_CU_H_
#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <cstddef>
#include <initializer_list>
#include <utility>

#include "absl/status/status.h"
#include "tensorflow/core/kernels/flash_attention/flash_attention_op.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow::functor {

inline constexpr float EPSF = 1e-6f;
inline constexpr float ZEROF = 0.0f;
inline constexpr float NEGINF = -1e9f;

// Zeroes each (pointer, bytes) buffer on the device's stream. Used for empty
// inputs, where the kernels are skipped but the outputs still need values.
inline absl::Status ZeroOutputs(
    const Eigen::GpuDevice &d,
    std::initializer_list<std::pair<void *, size_t>> buffers) {
  for (const auto &[ptr, bytes] : buffers) {
    if (bytes == 0) continue;
    const cudaError_t err = cudaMemsetAsync(ptr, 0, bytes, d.stream());
    if (err != cudaSuccess) {
      return errors::Internal("FlashAttention: zeroing outputs failed: ",
                              cudaGetErrorString(err));
    }
  }
  return absl::OkStatus();
}

__device__ __forceinline__ float get_random_float_v2(uint64_t global_idx,
                                                     uint64_t j) {
  uint64_t state = global_idx + j;
  state = state * 6364136223846793005ULL + 1442695040888963407ULL;
  uint32_t val = (state >> 32) & 0xFFFFFFFF;
  return static_cast<float>(val) / 4294967296.0f;
}

__device__ __forceinline__ uint64_t
ResolveSeed(const functor::FlashAttentionRng &rng) {
  return rng.seed_dev != nullptr ? *rng.seed_dev : rng.seed;
}

__device__ __forceinline__ uint64_t
ResolveOffset(const functor::FlashAttentionRng &rng) {
  return rng.offset_dev != nullptr ? *rng.offset_dev : rng.offset;
}

// Total elements in the shared memory tile allocated for this feature
template <typename T, int D_MAX>
__device__ __forceinline__ void load_tile(
    T *__restrict__ s_mem,        // Shared memory destination
    const T *__restrict__ g_mem,  // Global memory source
    int tid,                      // threadIdx.x
    int block_dim,                // blockDim.x
    int b_size,                   // Br or Bc
    int d_size,                   // D_qk or D_v
    int s_size,                   // S_q or S_kv
    int tile_offset,              // tile_offset (in S dimension)
    size_t batch_offset)          // batch index offset
{
  const int total_elements = b_size * D_MAX;
  for (int i = tid; i < total_elements; i += block_dim) {
    const int s = i / D_MAX;
    const int d = i - s * D_MAX;
    const int s_offset = tile_offset + s;
    if (s_offset < s_size && d < d_size) {
      s_mem[i] = g_mem[(batch_offset + s_offset) * d_size + d];
    } else {
      s_mem[i] = T(0.0f);
    }
  }
}

namespace traits {

/*
forward pass:
- (2Br+Bc) * size(float) * D_MAX <= 48KB
- Br <= Th
*/
template <int D_MAX>
struct FlashTraits;
template <>
struct FlashTraits<8> {
  static constexpr int Br = 256;  // Larger tile for Q since registers are free
  static constexpr int Bc = 256;  // Larger tile for K/V
  static constexpr int Th = 256;
};
template <>
struct FlashTraits<16> {
  static constexpr int Br = 256;
  static constexpr int Bc = 256;
  static constexpr int Th = 256;
};
template <>
struct FlashTraits<32> {
  static constexpr int Br = 128;
  static constexpr int Bc = 128;
  static constexpr int Th = 128;
};
template <>
struct FlashTraits<64> {
  static constexpr int Br = 128;
  static constexpr int Bc = 32;
  static constexpr int Th = 128;
};
template <>
struct FlashTraits<128> {
  static constexpr int Br = 64;
  static constexpr int Bc = 16;
  static constexpr int Th = 64;
};
/*template <> struct FlashTraits<256>
{
    static constexpr int Br = 64;
    static constexpr int Bc = 32;
    static constexpr int Th = 64;
};
template <> struct FlashTraits<512>
{
    static constexpr int Br = 64;
    static constexpr int Bc = 64;
    static constexpr int Th = 64;
};
template <> struct FlashTraits<1024>
{
    static constexpr int Br = 64;
    static constexpr int Bc = 64;
    static constexpr int Th = 128;
};*/

/*
backward KV pass:
- ((2Br + 2Bc) * D_MAX +2Br) * size(float) <= 48KB
- Bc <= Th
*/
template <int D_MAX>
struct FlashTraitsKV;
template <>
struct FlashTraitsKV<8> {
  static constexpr int Br = 224;
  static constexpr int Bc = 256;
  static constexpr int Th = 256;
};
template <>
struct FlashTraitsKV<16> {
  static constexpr int Br = 96;
  static constexpr int Bc = 256;
  static constexpr int Th = 256;
};
template <>
struct FlashTraitsKV<32> {
  static constexpr int Br = 32;
  static constexpr int Bc = 128;
  static constexpr int Th = 128;
};
template <>
struct FlashTraitsKV<64> {
  static constexpr int Br = 16;
  static constexpr int Bc = 64;
  static constexpr int Th = 64;
};
template <>
struct FlashTraitsKV<128> {
  static constexpr int Br = 8;
  static constexpr int Bc = 32;
  static constexpr int Th = 64;
};
/*template <> struct FlashTraitsKV<256>
{
    static constexpr int Br = 16;
    static constexpr int Bc = 32;
    static constexpr int Th = 64;
};
template <> struct FlashTraitsKV<512>
{
    static constexpr int Br = 64;
    static constexpr int Bc = 32;
    static constexpr int Th = 64;
};
template <> struct FlashTraitsKV<1024>
{
    static constexpr int Br = 64;
    static constexpr int Bc = 64;
    static constexpr int Th = 64;
};*/

/*
backward Q pass:
- ((2Br + 2Bc) * D_MAX) * size(float) <= 48KB
- Br <= Th
*/
template <int D_MAX>
struct FlashTraitsQ;
template <>
struct FlashTraitsQ<8> {
  static constexpr int Br = 128;
  static constexpr int Bc = 128;
  static constexpr int Th = 128;
};
template <>
struct FlashTraitsQ<16> {
  static constexpr int Br = 128;
  static constexpr int Bc = 64;
  static constexpr int Th = 128;
};
template <>
struct FlashTraitsQ<32> {
  static constexpr int Br = 128;
  static constexpr int Bc = 64;
  static constexpr int Th = 128;
};
template <>
struct FlashTraitsQ<64> {
  static constexpr int Br = 64;
  static constexpr int Bc = 32;
  static constexpr int Th = 64;
};
template <>
struct FlashTraitsQ<128> {
  static constexpr int Br = 32;
  static constexpr int Bc = 16;
  static constexpr int Th = 64;
};
/*template <> struct FlashTraitsQ<256>
{
    static constexpr int Br = 64;
    static constexpr int Bc = 16;
    static constexpr int Th = 64;
};
template <> struct FlashTraitsQ<512>
{
    static constexpr int Br = 64;
    static constexpr int Bc = 32;
    static constexpr int Th = 64;
};
template <> struct FlashTraitsQ<1024>
{
    static constexpr int Br = 64;
    static constexpr int Bc = 64;
    static constexpr int Th = 64;
};*/
}  // namespace traits
}  // namespace tensorflow::functor

#endif

#endif  // TENSORFLOW_CORE_KERNELS_FLASH_ATTENTION_FLASH_ATTENTION_OP_GPU_CU_H_
