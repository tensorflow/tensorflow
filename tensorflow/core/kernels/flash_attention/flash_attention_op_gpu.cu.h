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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <utility>

#include "absl/status/status.h"
#include "tensorflow/core/kernels/flash_attention/flash_attention_op.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow::functor {

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

template <typename T>
__global__ void FillKernel(T *__restrict__ ptr, size_t n, T value) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
       i += size_t(blockDim.x) * gridDim.x) {
    ptr[i] = value;
  }
}

// Sets n floats at ptr to value on the device's stream (cudaMemset can only
// set bytes).
inline absl::Status FillFloats(const Eigen::GpuDevice &d, float *ptr, size_t n,
                               float value) {
  if (n == 0) return absl::OkStatus();
  constexpr int kThreads = 256;
  const int blocks =
      static_cast<int>(std::min<size_t>((n + kThreads - 1) / kThreads, 4096));
  return GpuLaunchKernel(FillKernel<float>, blocks, kThreads, 0, d.stream(),
                         ptr, n, value);
}

__device__ __forceinline__ float get_random_float_v2(uint64_t global_idx,
                                                     uint64_t j) {
  uint64_t state = global_idx + j;
  state = state * 6364136223846793005ULL + 1442695040888963407ULL;
  uint32_t val = (state >> 32) & 0xFFFFFFFF;
  return static_cast<float>(val >> 8) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ uint64_t
ResolveSeed(const functor::FlashAttentionRng &rng) {
  return rng.seed_dev != nullptr ? *rng.seed_dev : rng.seed;
}

__device__ __forceinline__ uint64_t
ResolveOffset(const functor::FlashAttentionRng &rng) {
  return rng.offset_dev != nullptr ? *rng.offset_dev : rng.offset;
}

// Tensors are [B, S, H, D], so position s of the (b, h) sequence is row
// RowBase(b, h, S, H) + s * H of the [B * S * H, D] matrix.
__device__ __forceinline__ size_t RowBase(int b, int h, int S, int H) {
  return static_cast<size_t>(b) * S * H + h;
}

// Layout and accessors for the [rows][D_MAX] tiles kept in shared memory.
//
// Rows are stored as 16-byte chunks and read a whole chunk at a time (128-bit
// loads), which shared memory serves 8 threads at a time. When all threads read
// the same row this is a broadcast. When each thread reads its own row (the
// same chunk of 8 consecutive rows), an unrotated layout puts all 8 in the same
// bank group whenever a row is a multiple of 128 bytes, serializing the read.
// With kShift > 0, chunk c of row r is kept at chunk position
// (c + kShift * (r / kRowsPerShift)) % kChunks, which spreads those 8 reads
// over all 8 bank groups without the extra shared memory that padding would
// need. kShift = 2 is for two threads per row that read alternate chunks (see
// FlashAttnBwdKernel): the 8 threads then cover 4 rows, which a shift of 2
// chunks per row still keeps apart. Tiles only read by broadcast use
// kShift = 0, saving the index arithmetic.
template <typename T, int D_MAX, int kShift>
struct Tile {
  static constexpr int kChunk = 16 / sizeof(T);  // elements per chunk
  static_assert(D_MAX % kChunk == 0, "rows must be whole 16-byte chunks");
  static constexpr int kChunks = D_MAX / kChunk;
  // 8 chunks span all 32 banks; narrower rows share a rotation.
  static constexpr int kRowsPerShift = kChunks >= 8 ? 1 : 8 / kChunks;

  // Position of element d of row `row`.
  __device__ __forceinline__ static int Index(int row, int d) {
    if constexpr (kShift == 0) return row * D_MAX + d;
    const int chunk = (d / kChunk + kShift * (row / kRowsPerShift)) % kChunks;
    return row * D_MAX + chunk * kChunk + d % kChunk;
  }

  // Loads chunk c (elements c * kChunk ...) of `row` with one 128-bit load.
  __device__ __forceinline__ static void LoadChunk(const T *tile, int row,
                                                   int c, T (&out)[kChunk]) {
    // ::uint4 is CUDA's 128-bit vector; plain uint4 here is TF's 4-bit int.
    const ::uint4 v =
        *reinterpret_cast<const ::uint4 *>(tile + Index(row, c * kChunk));
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int e = 0; e < kChunk; ++e) {
      if constexpr (sizeof(T) == 4) {
        out[e] = __uint_as_float(w[e]);
      } else {
        out[e] = Eigen::numext::bit_cast<T>(
            static_cast<uint16_t>(w[e / 2] >> (16 * (e % 2))));
      }
    }
  }

  // out[n] = Dot(a, tile[row + n]) for n < N. The N sums are independent, so
  // their multiply-adds overlap instead of each waiting for the previous one.
  template <int N>
  __device__ __forceinline__ static void DotN(const float (&a)[D_MAX],
                                              const T *tile, int row,
                                              float (&out)[N]) {
#pragma unroll
    for (int n = 0; n < N; ++n) out[n] = 0.0f;
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
#pragma unroll
      for (int n = 0; n < N; ++n) {
        T y[kChunk];
        LoadChunk(tile, row + n, c, y);
#pragma unroll
        for (int e = 0; e < kChunk; ++e) {
          out[n] += a[c * kChunk + e] * static_cast<float>(y[e]);
        }
      }
    }
  }

  // Dot product of a (a row already in registers) and tile[row].
  __device__ __forceinline__ static float Dot(const float (&a)[D_MAX],
                                              const T *tile, int row) {
    float sum = 0.0f;
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      T y[kChunk];
      LoadChunk(tile, row, c, y);
#pragma unroll
      for (int e = 0; e < kChunk; ++e) {
        sum += a[c * kChunk + e] * static_cast<float>(y[e]);
      }
    }
    return sum;
  }

  // Copies tile[row] into registers.
  __device__ __forceinline__ static void LoadRow(const T *tile, int row,
                                                 float (&out)[D_MAX]) {
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      T x[kChunk];
      LoadChunk(tile, row, c, x);
#pragma unroll
      for (int e = 0; e < kChunk; ++e) {
        out[c * kChunk + e] = static_cast<float>(x[e]);
      }
    }
  }

  // acc += alpha * tile[row]. With kSplit > 1, a row is shared by kSplit
  // threads: `part` handles chunks part, part + kSplit, ..., and acc holds just
  // those, in order (acc[c' * kChunk + e] is element (c' * kSplit + part) *
  // kChunk + e).
  template <int kSplit = 1>
  __device__ __forceinline__ static void Axpy(float (&acc)[D_MAX / kSplit],
                                              float alpha, const T *tile,
                                              int row, int part = 0) {
#pragma unroll
    for (int c = 0; c < kChunks / kSplit; ++c) {
      T x[kChunk];
      LoadChunk(tile, row, c * kSplit + part, x);
#pragma unroll
      for (int e = 0; e < kChunk; ++e) {
        acc[c * kChunk + e] += alpha * static_cast<float>(x[e]);
      }
    }
  }

  // acc += sum_n alpha[n] * tile[row + n] for n < N, with kSplit as in Axpy.
  template <int N, int kSplit = 1>
  __device__ __forceinline__ static void AxpyN(float (&acc)[D_MAX / kSplit],
                                               const float (&alpha)[N],
                                               const T *tile, int row,
                                               int part = 0) {
#pragma unroll
    for (int c = 0; c < kChunks / kSplit; ++c) {
#pragma unroll
      for (int n = 0; n < N; ++n) {
        T x[kChunk];
        LoadChunk(tile, row + n, c * kSplit + part, x);
#pragma unroll
        for (int e = 0; e < kChunk; ++e) {
          acc[c * kChunk + e] += alpha[n] * static_cast<float>(x[e]);
        }
      }
    }
  }
};

// Dot product of a[a_row] and b[b_row] over the zero-padded rows, for tiles
// with layouts TileA and TileB. With kSplit > 1, only over the chunks of
// `part` (see Tile::Axpy); the kSplit partial sums add up to the full one.
template <typename TileA, typename TileB, int kSplit = 1, typename T>
__device__ __forceinline__ float TileDot(const T *a, int a_row, const T *b,
                                         int b_row, int part = 0) {
  static_assert(TileA::kChunks == TileB::kChunks);
  float sum = 0.0f;
#pragma unroll
  for (int c = 0; c < TileA::kChunks / kSplit; ++c) {
    T x[TileA::kChunk], y[TileA::kChunk];
    TileA::LoadChunk(a, a_row, c * kSplit + part, x);
    TileB::LoadChunk(b, b_row, c * kSplit + part, y);
#pragma unroll
    for (int e = 0; e < TileA::kChunk; ++e) {
      sum += static_cast<float>(x[e]) * static_cast<float>(y[e]);
    }
  }
  return sum;
}

// out[n] = TileDot(a[a_row + n], b[b_row]) for n < N, as N independent sums.
// Each chunk of b is loaded once for all N rows of a.
template <typename TileA, typename TileB, int N, int kSplit = 1, typename T>
__device__ __forceinline__ void TileDotN(const T *a, int a_row, const T *b,
                                         int b_row, float (&out)[N],
                                         int part = 0) {
  static_assert(TileA::kChunks == TileB::kChunks);
#pragma unroll
  for (int n = 0; n < N; ++n) out[n] = 0.0f;
#pragma unroll
  for (int c = 0; c < TileA::kChunks / kSplit; ++c) {
    T y[TileA::kChunk];
    TileB::LoadChunk(b, b_row, c * kSplit + part, y);
#pragma unroll
    for (int n = 0; n < N; ++n) {
      T x[TileA::kChunk];
      TileA::LoadChunk(a, a_row + n, c * kSplit + part, x);
#pragma unroll
      for (int e = 0; e < TileA::kChunk; ++e) {
        out[n] += static_cast<float>(x[e]) * static_cast<float>(y[e]);
      }
    }
  }
}

// Asynchronous global -> shared copies (cp.async, sm80+). They complete in the
// background, so a tile load can overlap computing on another tile: copies
// issued since the last CpAsyncCommit() form a group, and CpAsyncWait<n>()
// waits until at most n groups are still in flight (each thread for its own
// copies; a __syncthreads() after it makes the data visible to the block).
// Before sm80 the copies are ordinary loads and stores, done on return.
__device__ __forceinline__ uint32_t SharedAddress(const void *p) {
  uint32_t addr;
  asm("{ .reg .u64 a; cvta.to.shared.u64 a, %1; cvt.u32.u64 %0, a; }"
      : "=r"(addr)
      : "l"(p));
  return addr;
}

// Copies 16 bytes, or writes 16 zero bytes if !valid (src is then not read).
__device__ __forceinline__ void CpAsync16(void *dst, const void *src,
                                          bool valid) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(
                   SharedAddress(dst)),
               "l"(src), "r"(valid ? 16 : 0));
#else
  *reinterpret_cast<::uint4 *>(dst) =
      valid ? *reinterpret_cast<const ::uint4 *>(src) : make_uint4(0, 0, 0, 0);
#endif
}

// Copies one float, or writes 0 if !valid (src is then not read).
__device__ __forceinline__ void CpAsync4(float *dst, const float *src,
                                         bool valid) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(
                   SharedAddress(dst)),
               "l"(src), "r"(valid ? 4 : 0));
#else
  *dst = valid ? *src : 0.0f;
#endif
}

__device__ __forceinline__ void CpAsyncCommit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int N>
__device__ __forceinline__ void CpAsyncWait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

// Whether tiles can be loaded 16 bytes at a time: every row of the [.., D]
// tensors starts on a 16-byte boundary.
template <typename T>
bool CanLoadChunks(std::initializer_list<const void *> ptrs,
                   std::initializer_list<int> dims) {
  constexpr int kChunk = 16 / sizeof(T);
  for (int d : dims) {
    if (d % kChunk != 0) return false;
  }
  for (const void *p : ptrs) {
    if (reinterpret_cast<uintptr_t>(p) % 16 != 0) return false;
  }
  return true;
}

// Loads rows [tile_offset, tile_offset + b_size) of one (batch, head) sequence
// into a Tile, zero-padding rows past s_size and columns past d_size.
template <typename T, int D_MAX, int kShift>
__device__ __forceinline__ void load_tile(
    T *__restrict__ s_mem,        // Shared memory destination
    const T *__restrict__ g_mem,  // Global memory source
    int tid,                      // threadIdx.x
    int block_dim,                // blockDim.x
    int b_size,                   // Br or Bc
    int d_size,                   // D_qk or D_v
    int s_size,                   // S_q or S_kv
    int tile_offset,              // tile_offset (in S dimension)
    size_t row_base,              // RowBase() of this (batch, head)
    int row_step)                 // H: rows between sequence positions
{
  const int total_elements = b_size * D_MAX;
  for (int i = tid; i < total_elements; i += block_dim) {
    const int s = i / D_MAX;
    const int d = i - s * D_MAX;
    const int s_offset = tile_offset + s;
    const int idx = Tile<T, D_MAX, kShift>::Index(s, d);
    if (s_offset < s_size && d < d_size) {
      s_mem[idx] =
          g_mem[(row_base + static_cast<size_t>(s_offset) * row_step) * d_size +
                d];
    } else {
      s_mem[idx] = T(0.0f);
    }
  }
}

// Starts loading a tile as load_tile does. With chunks (see CanLoadChunks),
// it copies whole 16-byte chunks with cp.async, so the caller must
// CpAsyncCommit() and CpAsyncWait() before the __syncthreads() that precedes
// reading the tile; otherwise it falls back to load_tile, which is done on
// return.
template <typename T, int D_MAX, int kShift>
__device__ __forceinline__ void LoadTile(bool chunks, T *__restrict__ s_mem,
                                         const T *__restrict__ g_mem, int tid,
                                         int block_dim, int b_size,
                                         int d_size, int s_size,
                                         int tile_offset, size_t row_base,
                                         int row_step) {
  if (!chunks) {
    load_tile<T, D_MAX, kShift>(s_mem, g_mem, tid, block_dim, b_size, d_size,
                                s_size, tile_offset, row_base, row_step);
    return;
  }
  using TileT = Tile<T, D_MAX, kShift>;
  constexpr int kChunk = TileT::kChunk;
  constexpr int kChunks = TileT::kChunks;
  const int d_chunks = d_size / kChunk;
  for (int i = tid; i < b_size * kChunks; i += block_dim) {
    const int s = i / kChunks;
    const int c = i - s * kChunks;
    const int s_offset = tile_offset + s;
    const bool valid = s_offset < s_size && c < d_chunks;
    const T *src =
        valid ? g_mem +
                    (row_base + static_cast<size_t>(s_offset) * row_step) *
                        d_size +
                    c * kChunk
              : g_mem;
    CpAsync16(s_mem + TileT::Index(s, c * kChunk), src, valid);
  }
}

namespace traits {

// The fallback kernels' dynamic shared memory stays within the 48KB that needs
// no opt-in. Tiles are double-buffered (kStages = 2, loading the next tile
// while computing on this one) when two stages fit, single-buffered otherwise.
inline constexpr int kMaxSmemBytes = 48 * 1024;

/*
forward pass:
- Br <= Th; Bc % kKeys == 0
- Q is read into registers for D_MAX <= 64, else kept as a [Br] tile
*/
template <int D_MAX>
struct FlashTraits;
template <>
struct FlashTraits<8> {
  static constexpr int Br = 256;  // Larger tile for Q since registers are free
  static constexpr int Bc = 256;  // Larger tile for K/V
  static constexpr int Th = 256;
  static constexpr int kKeys = 4;  // keys per step of the inner loop
  static constexpr int kMinBlocks = 1;  // __launch_bounds__ blocks per SM
};
template <>
struct FlashTraits<16> {
  static constexpr int Br = 256;
  static constexpr int Bc = 128;
  static constexpr int Th = 256;
  static constexpr int kKeys = 4;
  static constexpr int kMinBlocks = 1;
};
template <>
struct FlashTraits<32> {
  static constexpr int Br = 128;
  static constexpr int Bc = 64;
  static constexpr int Th = 128;
  static constexpr int kKeys = 4;
  static constexpr int kMinBlocks = 1;
};
template <>
struct FlashTraits<64> {
  static constexpr int Br = 128;
  static constexpr int Bc = 32;
  static constexpr int Th = 128;
  static constexpr int kKeys = 4;
  // Keeps registers <= 168 so that 3 blocks fit per SM (2 at the 172 ptxas
  // picks otherwise, too few warps to hide the shared memory latency).
  static constexpr int kMinBlocks = 3;
};
template <>
struct FlashTraits<128> {
  static constexpr int Br = 64;
  static constexpr int Bc = 16;
  static constexpr int Th = 64;
  static constexpr int kKeys = 2;
  static constexpr int kMinBlocks = 1;
};

template <typename T, int D_MAX>
constexpr int FwdQTileBytes() {
  return D_MAX <= 64 ? 0 : FlashTraits<D_MAX>::Br * D_MAX * sizeof(T);
}
template <typename T, int D_MAX>
constexpr int FwdStageBytes() {  // K and V tiles
  return 2 * FlashTraits<D_MAX>::Bc * D_MAX * sizeof(T);
}
template <typename T, int D_MAX>
constexpr int FwdStages() {
  return FwdQTileBytes<T, D_MAX>() + 2 * FwdStageBytes<T, D_MAX>() <=
                 kMaxSmemBytes
             ? 2
             : 1;
}
template <typename T, int D_MAX>
constexpr int FwdSmemBytes() {
  constexpr int bytes = FwdQTileBytes<T, D_MAX>() +
                        FwdStages<T, D_MAX>() * FwdStageBytes<T, D_MAX>();
  static_assert(bytes <= kMaxSmemBytes);
  return bytes;
}

/*
backward KV pass (FlashAttnBwdKernel):
- K, V tiles [Bc] + kStages x (Q, dO tiles [Br] + Stats, dOsum [Br]) + the
  float dS tile [Br][Bc + 1] when accumulating dQ, within kMaxSmemBytes
- Bc * kSplit <= Th, with kSplit threads per key; Br even
*/
template <int D_MAX>
struct FlashTraitsKV;
template <>
struct FlashTraitsKV<8> {
  static constexpr int Br = 16;
  static constexpr int Bc = 256;
  static constexpr int Th = 256;
  static constexpr int kSplit = 1;
};
template <>
struct FlashTraitsKV<16> {
  static constexpr int Br = 32;
  static constexpr int Bc = 128;
  static constexpr int Th = 128;
  static constexpr int kSplit = 1;
};
template <>
struct FlashTraitsKV<32> {
  static constexpr int Br = 16;
  static constexpr int Bc = 64;
  static constexpr int Th = 128;
  static constexpr int kSplit = 2;
};
template <>
struct FlashTraitsKV<64> {
  static constexpr int Br = 8;
  static constexpr int Bc = 64;
  static constexpr int Th = 128;
  static constexpr int kSplit = 2;
};
template <>
struct FlashTraitsKV<128> {
  static constexpr int Br = 8;
  static constexpr int Bc = 32;
  static constexpr int Th = 64;
  static constexpr int kSplit = 2;
};

template <typename T, int D_MAX>
constexpr int BwdKVTileBytes() {  // K and V tiles
  return 2 * FlashTraitsKV<D_MAX>::Bc * D_MAX * sizeof(T);
}
template <typename T, int D_MAX>
constexpr int BwdQStageBytes() {  // Q, dO tiles, Stats, dOsum
  constexpr int Br = FlashTraitsKV<D_MAX>::Br;
  return 2 * Br * D_MAX * sizeof(T) + 2 * Br * sizeof(float);
}
template <int D_MAX, bool kAccumulateDQ>
constexpr int BwdDSBytes() {
  return kAccumulateDQ ? FlashTraitsKV<D_MAX>::Br *
                             (FlashTraitsKV<D_MAX>::Bc + 1) * sizeof(float)
                       : 0;
}
template <typename T, int D_MAX, bool kAccumulateDQ>
constexpr int BwdKVStages() {
  return BwdKVTileBytes<T, D_MAX>() + 2 * BwdQStageBytes<T, D_MAX>() +
                     BwdDSBytes<D_MAX, kAccumulateDQ>() <=
                 kMaxSmemBytes
             ? 2
             : 1;
}
template <typename T, int D_MAX, bool kAccumulateDQ>
constexpr int BwdKVSmemBytes() {
  constexpr int bytes =
      BwdKVTileBytes<T, D_MAX>() +
      BwdKVStages<T, D_MAX, kAccumulateDQ>() * BwdQStageBytes<T, D_MAX>() +
      BwdDSBytes<D_MAX, kAccumulateDQ>();
  static_assert(bytes <= kMaxSmemBytes);
  return bytes;
}

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
}  // namespace traits
}  // namespace tensorflow::functor

#endif

#endif  // TENSORFLOW_CORE_KERNELS_FLASH_ATTENTION_FLASH_ATTENTION_OP_GPU_CU_H_
