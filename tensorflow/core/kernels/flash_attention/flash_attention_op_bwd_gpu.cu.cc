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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <algorithm>
#include <type_traits>

#include "tensorflow/core/kernels/flash_attention/fa2_api.h"
#include "tensorflow/core/kernels/flash_attention/flash_attention_op.h"
#include "tensorflow/core/kernels/flash_attention/flash_attention_op_gpu.cu.h"

// Backward pass of the fallback kernels:
//  1. FlashAttnBwdDotDoOKernel: dOsum_i = sum_d dO_id * Out_id, once per row.
//  2. FlashAttnBwdKernel: one block per KV tile, one thread per key j. It
//     computes dS_ij once per (i, j) and accumulates dK_j and dV_j in
//     registers. With kAccumulateDQ it also adds dQ_i += sum_j dS_ij K_j to a
//     float buffer with atomicAdd, and FlashAttnBwdConvertDQKernel converts
//     that to T.
//  3. Float atomics make dQ nondeterministic, so in deterministic mode
//     FlashAttnBwdKernel_Q recomputes dS per query row and writes dQ
//     directly instead.

namespace tensorflow::functor {

template <typename T>
__global__ void FlashAttnBwdDotDoOKernel(const T *__restrict__ Out,
                                         const T *__restrict__ dO,
                                         float *__restrict__ dOsum, int BH,
                                         int H, int S_q, int D_v) {
  const size_t rows = static_cast<size_t>(BH) * S_q;
  for (size_t r = blockIdx.x * size_t(blockDim.x) + threadIdx.x; r < rows;
       r += size_t(blockDim.x) * gridDim.x) {
    const int bh = r / S_q;
    const int i = r - static_cast<size_t>(bh) * S_q;
    const int b = bh / H;
    const int h = bh - b * H;
    const size_t base = (RowBase(b, h, S_q, H) + static_cast<size_t>(i) * H) *
                        static_cast<size_t>(D_v);
    float sum = ZEROF;
    for (int d = 0; d < D_v; ++d) {
      sum +=
          static_cast<float>(dO[base + d]) * static_cast<float>(Out[base + d]);
    }
    dOsum[r] = sum;
  }
}

// dQ[b, i, h, d] = dQ_accum[(bh * S_q + i) * D_qk + d].
template <typename T>
__global__ void FlashAttnBwdConvertDQKernel(const float *__restrict__ dQ_accum,
                                            T *__restrict__ dQ, int BH, int H,
                                            int S_q, int D_qk) {
  const size_t n = static_cast<size_t>(BH) * S_q * D_qk;
  for (size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x; idx < n;
       idx += size_t(blockDim.x) * gridDim.x) {
    const size_t r = idx / D_qk;
    const int d = idx - r * D_qk;
    const int bh = r / S_q;
    const int i = r - static_cast<size_t>(bh) * S_q;
    const int b = bh / H;
    const int h = bh - b * H;
    dQ[(RowBase(b, h, S_q, H) + static_cast<size_t>(i) * H) * D_qk + d] =
        static_cast<T>(dQ_accum[idx]);
  }
}

// See FlashAttnFwdKernel for why the kernels have __launch_bounds__.
template <typename T, int D_MAX, bool kAccumulateDQ>
__global__ void __launch_bounds__(traits::FlashTraitsKV<D_MAX>::Th)
    FlashAttnBwdKernel(const T *__restrict__ Q, const T *__restrict__ K,
                       const T *__restrict__ V, const float *__restrict__ Stats,
                       const float *__restrict__ dOsum,
                       const T *__restrict__ dO, T *__restrict__ dK,
                       T *__restrict__ dV, float *__restrict__ dQ_accum, int BH,
                       int H, int S_q, int S_kv, int D_qk, int D_v,
                       bool causal_mask, float dropout_rate, float scale,
                       FlashAttentionRng rng, bool load_chunks) {
  const uint64_t seed = ResolveSeed(rng);
  const uint64_t offset = ResolveOffset(rng);
  constexpr int Br = traits::FlashTraitsKV<D_MAX>::Br;
  constexpr int Bc = traits::FlashTraitsKV<D_MAX>::Bc;
  constexpr int kStages = traits::BwdKVStages<T, D_MAX, kAccumulateDQ>();
  // Query rows per step of the inner loop: each K/V chunk a thread loads
  // serves both rows, and the rows' dot products overlap.
  constexpr int kRows = 2;
  static_assert(Br % kRows == 0);
  // Row stride of s_dS; the +1 keeps rows i and i + 1 in different banks when
  // the dQ loop has several rows per warp.
  constexpr int kDSStride = Bc + 1;
  // r_dK and r_dV hold 2 * D_MAX / kSplit floats per thread. From D_MAX = 32,
  // two threads share each key, each taking alternate 16-byte chunks of the
  // row (traits::FlashTraitsKV has Th = 2 * Bc for this): that halves the
  // registers per thread, so twice as many warps fit on an SM to hide latency,
  // and for D_MAX = 128 it keeps the accumulators within 255 registers.
  constexpr int kSplit = traits::FlashTraitsKV<D_MAX>::kSplit;
  constexpr int kPart = D_MAX / kSplit;
  static_assert(traits::FlashTraitsKV<D_MAX>::Th >= Bc * kSplit);
  // Each thread reads its own K/V row (rotated); Q/dO rows are broadcast.
  using TileKV = Tile<T, D_MAX, kSplit>;
  using TileQ = Tile<T, D_MAX, 0>;
  const float dp_inverse =
      (dropout_rate > 0.0f) ? (1.0f / (1.0f - dropout_rate)) : 1.0f;

  // Shared Memory Layout:
  // K_tile[Bc] | V_tile[Bc] | kStages x (Q_tile[Br] | dO_tile[Br] | Stats[Br]
  // | dOsum[Br]) | dS[Br][kDSStride] (with kAccumulateDQ)
  extern __shared__ __align__(16) char smem_[];
  T *s_K = reinterpret_cast<T *>(smem_);
  T *s_V = s_K + Bc * D_MAX;
  char *s_stages = reinterpret_cast<char *>(s_V + Bc * D_MAX);
  constexpr int kStageBytes = traits::BwdQStageBytes<T, D_MAX>();
  float *s_dS = reinterpret_cast<float *>(s_stages + kStages * kStageBytes);

  const int tid = threadIdx.x;
  const int key = tid / kSplit;  // key j = tile_offset_kv + key
  const int part = tid % kSplit;
  // The kSplit threads of a key are adjacent lanes of one warp.
  const unsigned pair_mask =
      kSplit == 1 ? 0u : (0x3u << ((threadIdx.x % 32) & ~1));

  const int tiles_per_batch = (S_kv + Bc - 1) / Bc;
  const bool block_valid = blockIdx.x < tiles_per_batch * BH;
  const int bh = block_valid ? blockIdx.x / tiles_per_batch : 0;
  const int b = bh / H;
  const int h = bh - b * H;

  const int tile_idx_kv = blockIdx.x % tiles_per_batch;
  const int tile_offset_kv = tile_idx_kv * Bc;
  const int kv_offset = tile_offset_kv + key;

  const bool active = block_valid && (key < Bc) && (kv_offset < S_kv);
  // Stats, dOsum, dQ_accum and the dropout RNG index (batch, head) as bh.
  const size_t batch_offset_q = static_cast<size_t>(bh) * S_q;
  const size_t row_base_q = RowBase(b, h, S_q, H);
  const size_t row_base_kv = RowBase(b, h, S_kv, H);
  const uint64_t global_rng_idx =
      seed + offset + static_cast<uint64_t>(kv_offset);

  float r_dK[kPart];
  float r_dV[kPart];
#pragma unroll
  for (int d = 0; d < kPart; ++d) {
    r_dK[d] = ZEROF;
    r_dV[d] = ZEROF;
  }

  // Arrive with the first Q tile.
  if (block_valid) {
    LoadTile<T, D_MAX, kSplit>(load_chunks, s_K, K, tid, blockDim.x, Bc, D_qk,
                               S_kv, tile_offset_kv, row_base_kv, H);
    LoadTile<T, D_MAX, kSplit>(load_chunks, s_V, V, tid, blockDim.x, Bc, D_v,
                               S_kv, tile_offset_kv, row_base_kv, H);
  }

  // Q tiles [t * Br, t * Br + Br). Causal: tiles ending before this KV tile's
  // first key are fully masked, so start at the one containing it. The range
  // is the same for the whole block, so the __syncthreads() stay matched.
  const int q_tiles = (S_q + Br - 1) / Br;
  const int q_begin =
      !block_valid ? q_tiles : (causal_mask ? tile_offset_kv / Br : 0);
  auto stage = [&](int t) { return s_stages + (t % kStages) * kStageBytes; };
  auto load_q = [&](int t) {
    T *sQ = reinterpret_cast<T *>(stage(t));
    T *sdO = sQ + Br * D_MAX;
    float *sStats = reinterpret_cast<float *>(sdO + Br * D_MAX);
    float *sdOsum = sStats + Br;
    const int tile_offset_q = t * Br;
    LoadTile<T, D_MAX, 0>(load_chunks, sQ, Q, tid, blockDim.x, Br, D_qk, S_q,
                          tile_offset_q, row_base_q, H);
    LoadTile<T, D_MAX, 0>(load_chunks, sdO, dO, tid, blockDim.x, Br, D_v, S_q,
                          tile_offset_q, row_base_q, H);
    for (int r = tid; r < Br; r += blockDim.x) {
      const bool valid = tile_offset_q + r < S_q;
      const size_t idx = valid ? batch_offset_q + tile_offset_q + r : 0;
      CpAsync4(sStats + r, Stats + idx, valid);
      CpAsync4(sdOsum + r, dOsum + idx, valid);
    }
  };

  if constexpr (kStages > 1) {
    if (q_begin < q_tiles) load_q(q_begin);
    CpAsyncCommit();
  }
  for (int t = q_begin; t < q_tiles; ++t) {
    if constexpr (kStages == 1) {
      load_q(t);
      CpAsyncCommit();
    }
    CpAsyncWait<0>();
    // Makes tile t visible to the block. With two stages, it also means every
    // thread is done with tile t - 1 (including the dQ loop's reads of s_dS),
    // so its buffer can take tile t + 1, which then loads while this tile is
    // processed.
    __syncthreads();
    if constexpr (kStages > 1) {
      if (t + 1 < q_tiles) load_q(t + 1);
      CpAsyncCommit();
    }
    const T *s_Q = reinterpret_cast<const T *>(stage(t));
    const T *s_dO = s_Q + Br * D_MAX;
    const float *s_Stats = reinterpret_cast<const float *>(s_dO + Br * D_MAX);
    const float *s_dOsum = s_Stats + Br;
    const int tile_offset_q = t * Br;
    const int rows = std::min(Br, S_q - tile_offset_q);

    if (key < Bc) {
      for (int i = 0; i < rows; i += kRows) {
        // Rows past `rows` (zero-padded) and causally masked pairs get
        // p = dS = 0.
        float p[kRows], dS[kRows];
        if (active) {
          // Both threads of a key take this branch together, so the
          // shuffles see their partner.
          float score[kRows], dP[kRows];
          TileDotN<TileQ, TileKV, kRows, kSplit>(s_Q, i, s_K, key, score,
                                                 part);
          TileDotN<TileQ, TileKV, kRows, kSplit>(s_dO, i, s_V, key, dP, part);
#pragma unroll
          for (int n = 0; n < kRows; ++n) {
            if constexpr (kSplit > 1) {
              score[n] += __shfl_xor_sync(pair_mask, score[n], 1);
              dP[n] += __shfl_xor_sync(pair_mask, dP[n], 1);
            }
            const int q_offset = tile_offset_q + i + n;
            const bool valid =
                (i + n < rows) && !(causal_mask && q_offset < kv_offset);
            float p_ij =
                valid ? Eigen::numext::exp(score[n] * scale - s_Stats[i + n])
                      : ZEROF;
            float ds = scale;
            bool is_dropped = false;
            if (dropout_rate > 0.0f && valid) {
              const size_t q_idx = batch_offset_q + q_offset;
              if (get_random_float_v2(global_rng_idx, q_idx * S_kv) <
                  dropout_rate) {
                is_dropped = true;
                ds *= -p_ij * s_dOsum[i + n];
                p_ij = ZEROF;
              } else {
                p_ij *= dp_inverse;
              }
            }
            if (!is_dropped) {
              ds *= p_ij * (dP[n] - s_dOsum[i + n] / dp_inverse);
            }
            p[n] = p_ij;
            dS[n] = valid ? ds : ZEROF;
          }
          TileQ::template AxpyN<kRows, kSplit>(r_dV, p, s_dO, i, part);
          TileQ::template AxpyN<kRows, kSplit>(r_dK, dS, s_Q, i, part);
        } else {
#pragma unroll
          for (int n = 0; n < kRows; ++n) dS[n] = ZEROF;
        }
        if constexpr (kAccumulateDQ) {
          if (part == 0) {
#pragma unroll
            for (int n = 0; n < kRows; ++n) {
              s_dS[(i + n) * kDSStride + key] = dS[n];
            }
          }
        }
      }
    }

    if constexpr (kAccumulateDQ) {
      __syncthreads();
      // dQ_i += sum_j dS_ij K_j for this tile's rows. Keys past S_kv have
      // dS = 0 and zero-padded K rows. Each work item is one 16-byte chunk
      // of a dQ row, so a key costs one dS load and one 128-bit K load for
      // kChunk multiply-adds. Neighbouring threads take neighbouring chunks
      // of a row: different bank groups for K, consecutive addresses for the
      // atomics.
      constexpr int kChunk = TileKV::kChunk;
      const int chunks = (D_qk + kChunk - 1) / kChunk;
      for (int o = tid; o < rows * chunks; o += blockDim.x) {
        const int i = o / chunks;
        const int c = o - i * chunks;
        const float *dS_row = s_dS + i * kDSStride;
        float sum[kChunk];
#pragma unroll
        for (int e = 0; e < kChunk; ++e) sum[e] = ZEROF;
#pragma unroll 4
        for (int j = 0; j < Bc; ++j) {
          T k[kChunk];
          TileKV::LoadChunk(s_K, j, c, k);
          const float ds = dS_row[j];
#pragma unroll
          for (int e = 0; e < kChunk; ++e) {
            sum[e] += ds * static_cast<float>(k[e]);
          }
        }
        float *out =
            dQ_accum + (batch_offset_q + tile_offset_q + i) * D_qk + c * kChunk;
#pragma unroll
        for (int e = 0; e < kChunk; ++e) {
          if (c * kChunk + e < D_qk) atomicAdd(out + e, sum[e]);
        }
      }
    }
    if constexpr (kStages == 1) {
      // The next iteration overwrites this tile and s_dS.
      __syncthreads();
    }
  }
  // K/V copies are still in flight if no Q tile was processed.
  CpAsyncWait<0>();

  if (active) {
    // r_dK[c * kChunk + e] is element (c * kSplit + part) * kChunk + e.
    constexpr int kChunk = TileKV::kChunk;
    const size_t kv_idx = row_base_kv + static_cast<size_t>(kv_offset) * H;
#pragma unroll
    for (int c = 0; c < kPart / kChunk; ++c) {
#pragma unroll
      for (int e = 0; e < kChunk; ++e) {
        const int d = (c * kSplit + part) * kChunk + e;
        if (d < D_qk) {
          dK[kv_idx * D_qk + d] = static_cast<T>(r_dK[c * kChunk + e]);
        }
        if (d < D_v) {
          dV[kv_idx * D_v + d] = static_cast<T>(r_dV[c * kChunk + e]);
        }
      }
    }
  }
}

// Deterministic dQ: one thread per query row (two for D_MAX = 128, sharing it
// as in FlashAttnBwdKernel), looping over all keys.
template <typename T, int D_MAX>
__global__ void __launch_bounds__(traits::FlashTraitsQ<D_MAX>::Th)
    FlashAttnBwdKernel_Q(const T *__restrict__ Q, const T *__restrict__ K,
                         const T *__restrict__ V,
                         const float *__restrict__ Stats,
                         const float *__restrict__ dOsum,
                         const T *__restrict__ dO, T *__restrict__ dQ, int BH,
                         int H, int S_q, int S_kv, int D_qk, int D_v,
                         bool causal_mask, float dropout_rate, float scale,
                         FlashAttentionRng rng, bool load_chunks) {
  const uint64_t seed = ResolveSeed(rng);
  const uint64_t offset = ResolveOffset(rng);
  constexpr int Br = traits::FlashTraitsQ<D_MAX>::Br;
  constexpr int Bc = traits::FlashTraitsQ<D_MAX>::Bc;
  // r_dQ holds D_MAX / kSplit floats. For D_MAX = 128, two threads share each
  // query row so that it and the rest of the kernel's state fit in registers
  // (traits::FlashTraitsQ<128> has Th = 2 * Br for this).
  constexpr int kSplit = D_MAX >= 128 ? 2 : 1;
  constexpr int kPart = D_MAX / kSplit;
  static_assert(traits::FlashTraitsQ<D_MAX>::Th >= Br * kSplit);
  // Each thread reads its own Q/dO row (rotated); K/V rows are broadcast.
  using TileQ = Tile<T, D_MAX, kSplit>;
  using TileKV = Tile<T, D_MAX, 0>;
  const float dp_inverse =
      (dropout_rate > 0.0f) ? (1.0f / (1.0f - dropout_rate)) : 1.0f;

  // Shared Memory Layout:
  // Q_tile[Br][D_qk] | dO_tile[Br][D_v] | K_tile[Bc][D_qk] | V_tile[Bc][D_v]
  extern __shared__ __align__(16) char smem_[];
  T *s_Q = reinterpret_cast<T *>(smem_);
  T *s_dO = s_Q + Br * D_MAX;
  T *s_K = s_dO + Br * D_MAX;
  T *s_V = s_K + Bc * D_MAX;

  const int tid = threadIdx.x;
  const int row = tid / kSplit;  // query row within the tile
  const int part = tid % kSplit;
  // The kSplit threads of a row are adjacent lanes of one warp.
  const unsigned pair_mask =
      kSplit == 1 ? 0u : (0x3u << ((threadIdx.x % 32) & ~1));
  const int tiles_per_batch = (S_q + Br - 1) / Br;
  const bool block_valid = blockIdx.x < tiles_per_batch * BH;
  const int bh = block_valid ? blockIdx.x / tiles_per_batch : 0;
  const int b = bh / H;
  const int h = bh - b * H;
  const int tile_idx_q = blockIdx.x % tiles_per_batch;
  const int tile_offset_q = tile_idx_q * Br;
  const int q_offset = tile_offset_q + row;
  const bool active = block_valid && (row < Br) && (q_offset < S_q);
  const size_t row_base_q = RowBase(b, h, S_q, H);
  const size_t row_base_kv = RowBase(b, h, S_kv, H);
  // Index into Stats, dOsum and the dropout RNG, which use bh for (batch,
  // head).
  const size_t q_idx = static_cast<size_t>(bh) * S_q + q_offset;
  // Row of this query position in Q/dO/dQ.
  const size_t q_row = row_base_q + static_cast<size_t>(q_offset) * H;
  const uint64_t global_rng_idx =
      seed + offset + static_cast<uint64_t>(q_idx * S_kv);

  float r_dQ[kPart];
#pragma unroll
  for (int d = 0; d < kPart; ++d) {
    r_dQ[d] = ZEROF;
  }

  // Arrive with the first K/V tile.
  if (block_valid) {
    LoadTile<T, D_MAX, kSplit>(load_chunks, s_Q, Q, tid, blockDim.x, Br, D_qk,
                               S_q, tile_offset_q, row_base_q, H);
    LoadTile<T, D_MAX, kSplit>(load_chunks, s_dO, dO, tid, blockDim.x, Br, D_v,
                               S_q, tile_offset_q, row_base_q, H);
  }
  const float row_dOsum = active ? dOsum[q_idx] : ZEROF;
  const float stats_val = active ? Stats[q_idx] : ZEROF;

  for (int tile_offset_kv = 0; tile_offset_kv < S_kv; tile_offset_kv += Bc) {
    const bool tile_is_masked =
        (!block_valid) ||
        (causal_mask && (tile_offset_kv > tile_offset_q + Br - 1));

    __syncthreads();
    if (!tile_is_masked) {
      LoadTile<T, D_MAX, 0>(load_chunks, s_K, K, tid, blockDim.x, Bc, D_qk,
                            S_kv, tile_offset_kv, row_base_kv, H);
      LoadTile<T, D_MAX, 0>(load_chunks, s_V, V, tid, blockDim.x, Bc, D_v,
                            S_kv, tile_offset_kv, row_base_kv, H);
    }
    CpAsyncCommit();
    CpAsyncWait<0>();
    __syncthreads();

    if (active && !tile_is_masked) {
      for (int j = 0; j < Bc && (tile_offset_kv + j) < S_kv; ++j) {
        const int kv_offset = tile_offset_kv + j;
        if (causal_mask && q_offset < kv_offset) continue;

        // Both threads of a row take the same path through this loop, so the
        // shuffles see their partner.
        float score = TileDot<TileQ, TileKV, kSplit>(s_Q, row, s_K, j, part);
        if constexpr (kSplit > 1) {
          score += __shfl_xor_sync(pair_mask, score, 1);
        }
        float p_ij = Eigen::numext::exp(score * scale - stats_val);
        float dS = scale;
        bool is_dropped = false;
        if (dropout_rate > 0.0f) {
          if (get_random_float_v2(global_rng_idx, kv_offset) < dropout_rate) {
            is_dropped = true;
            dS *= -p_ij * row_dOsum;
            p_ij = ZEROF;
          } else {
            p_ij *= dp_inverse;
          }
        }
        if (!is_dropped) {
          float dP = TileDot<TileQ, TileKV, kSplit>(s_dO, row, s_V, j, part);
          if constexpr (kSplit > 1) {
            dP += __shfl_xor_sync(pair_mask, dP, 1);
          }
          dS *= p_ij * (dP - row_dOsum / dp_inverse);
        }
        TileKV::template Axpy<kSplit>(r_dQ, dS, s_K, j, part);
      }
    }
  }

  if (active) {
    // r_dQ[c * kChunk + e] is element (c * kSplit + part) * kChunk + e.
    constexpr int kChunk = TileKV::kChunk;
    const size_t out_offset = q_row * D_qk;
#pragma unroll
    for (int c = 0; c < kPart / kChunk; ++c) {
#pragma unroll
      for (int e = 0; e < kChunk; ++e) {
        const int d = (c * kSplit + part) * kChunk + e;
        if (d < D_qk) {
          dQ[out_offset + d] = static_cast<T>(r_dQ[c * kChunk + e]);
        }
      }
    }
  }
}

// Grid for the elementwise kernels: grid-stride loops over n items.
inline int ElementwiseBlocks(size_t n, int threads) {
  return static_cast<int>(
      std::max<size_t>(1, std::min<size_t>((n + threads - 1) / threads, 4096)));
}

template <typename T>
struct FlashAttentionGradFunctor<GPUDevice, T> {
  static constexpr bool kIsBf16 = std::is_same_v<T, Eigen::bfloat16>;
  static constexpr bool kIsFa2Type = kIsBf16 || std::is_same_v<T, Eigen::half>;

  // The fallback kernels' scratch memory: dOsum (float [B, H, S_q]), then,
  // unless deterministic, the dQ accumulator (float [B, H, S_q, D_qk]).
  static size_t DOsumBytes(int B, int H, int S_q) {
    return (static_cast<size_t>(B) * H * S_q * sizeof(float) + 255) / 256 * 256;
  }
  static size_t FallbackWorkspaceBytes(int B, int H, int S_q, int D_qk,
                                       bool deterministic) {
    return DOsumBytes(B, H, S_q) +
           (deterministic
                ? 0
                : static_cast<size_t>(B) * H * S_q * D_qk * sizeof(float));
  }

  // Shape-only (no device query) so XLA can size the buffer at compile time.
  // Big enough for whichever path the device ends up taking.
  static int64_t WorkspaceBytes(int B, int H, int S_q, int S_kv, int D_qk,
                                int D_v, bool deterministic) {
    size_t bytes = 0;
    if (std::max(D_qk, D_v) <= 128) {
      bytes = FallbackWorkspaceBytes(B, H, S_q, D_qk, deterministic);
    }
    if constexpr (kIsFa2Type) {
      if (flash_attention_fa2::ShapeSupported(D_qk, D_v) &&
          flash_attention_fa2::GridSupported(B, H)) {
        bytes = std::max(bytes, flash_attention_fa2::BackwardWorkspaceBytes(
                                    B, H, S_q, D_qk, deterministic));
      }
    }
    return static_cast<int64_t>(bytes);
  }

  static absl::Status CheckFeatureSize(int D_qk, int D_v, float dropout_rate) {
    if (std::max(D_qk, D_v) <= 128) return absl::OkStatus();
    if constexpr (kIsFa2Type) {
      if (flash_attention_fa2::BackwardSupported(D_qk, D_v, dropout_rate))
        return absl::OkStatus();
    }
    return FeatureSizeError(D_qk, D_v);
  }

  absl::Status operator()(
      const GPUDevice &d, typename TTypes<T, 4>::ConstTensor Q,
      typename TTypes<T, 4>::ConstTensor K,
      typename TTypes<T, 4>::ConstTensor V,
      typename TTypes<T, 4>::ConstTensor Out,
      typename TTypes<float, 3>::ConstTensor Stats,
      typename TTypes<T, 4>::ConstTensor dO, typename TTypes<T, 4>::Tensor dQ,
      typename TTypes<T, 4>::Tensor dK, typename TTypes<T, 4>::Tensor dV,
      bool causal_mask, float dropout_rate, float scale, FlashAttentionRng rng,
      bool deterministic, void *workspace) const {
    const int B = Q.dimension(0);
    const int S_q = Q.dimension(1);
    const int H = Q.dimension(2);
    const int D_qk = Q.dimension(3);
    const int S_kv = K.dimension(1);
    const int D_v = V.dimension(3);

    if (B == 0 || H == 0) return absl::OkStatus();
    if (S_q == 0 || S_kv == 0) {
      // Nothing attends to anything, so all gradients are zero.
      return ZeroOutputs(d, {{dQ.data(), dQ.size() * sizeof(T)},
                             {dK.data(), dK.size() * sizeof(T)},
                             {dV.data(), dV.size() * sizeof(T)}});
    }

    if constexpr (kIsFa2Type) {
      if (flash_attention_fa2::BackwardSupported(D_qk, D_v, dropout_rate) &&
          flash_attention_fa2::GridSupported(B, H)) {
        const cudaError_t err = flash_attention_fa2::Backward(
            d.stream(), kIsBf16, Q.data(), K.data(), V.data(), Out.data(),
            Stats.data(), dO.data(), dQ.data(), dK.data(), dV.data(), workspace,
            B, H, S_q, S_kv, D_qk, scale, dropout_rate, causal_mask,
            deterministic, rng.seed, rng.offset, rng.seed_dev, rng.offset_dev);
        if (err != cudaSuccess) {
          return errors::Internal("FlashAttention-2 backward failed: ",
                                  cudaGetErrorString(err));
        }
        return absl::OkStatus();
      }
    }

    const Args args{Q.data(),     K.data(),     V.data(),  Out.data(),
                    dO.data(),    Stats.data(), dQ.data(), dK.data(),
                    dV.data(),    workspace,    B * H,     H,
                    S_q,          S_kv,         D_qk,      D_v,
                    causal_mask,  dropout_rate, scale,     rng,
                    deterministic};
    const int max_D = std::max(D_qk, D_v);
    if (max_D <= 8) return Launch<8>(d, args);
    if (max_D <= 16) return Launch<16>(d, args);
    if (max_D <= 32) return Launch<32>(d, args);
    if (max_D <= 64) return Launch<64>(d, args);
    if (max_D <= 128) return Launch<128>(d, args);
    return FeatureSizeError(D_qk, D_v);
  }

 private:
  // Everything the fallback kernels need. BH = B * H fits in int, see
  // CheckDimsFitInt32.
  struct Args {
    const T *Q, *K, *V, *Out, *dO;
    const float *Stats;
    T *dQ, *dK, *dV;
    void *workspace;
    int BH, H, S_q, S_kv, D_qk, D_v;
    bool causal_mask;
    float dropout_rate, scale;
    FlashAttentionRng rng;
    bool deterministic;
  };

  static bool LoadChunks(const Args &a) {
    return CanLoadChunks<T>({a.Q, a.K, a.V, a.dO}, {a.D_qk, a.D_v});
  }

  static absl::Status CheckGrid(uint64_t total_blocks) {
    if (total_blocks > 2147483647ULL) {
      return errors::InvalidArgument(
          "FlashAttention: total blocks exceed 1D grid limit");
    }
    return absl::OkStatus();
  }

  template <int D_MAX>
  absl::Status Launch(const GPUDevice &d, const Args &a) const {
    if (a.workspace == nullptr) {
      return errors::Internal("FlashAttention backward: missing workspace");
    }
    const int B = a.BH / a.H;
    float *dOsum = static_cast<float *>(a.workspace);
    float *dQ_accum = reinterpret_cast<float *>(
        static_cast<char *>(a.workspace) + DOsumBytes(B, a.H, a.S_q));
    const size_t rows_q = static_cast<size_t>(a.BH) * a.S_q;

    constexpr int kThreads = 256;
    TF_RETURN_IF_ERROR(GpuLaunchKernel(
        FlashAttnBwdDotDoOKernel<T>, ElementwiseBlocks(rows_q, kThreads),
        kThreads, 0, d.stream(), a.Out, a.dO, dOsum, a.BH, a.H, a.S_q, a.D_v));

    if (a.deterministic) {
      TF_RETURN_IF_ERROR(LaunchKV<D_MAX, false>(d, a, dOsum, nullptr));
      return LaunchQ<D_MAX>(d, a, dOsum);
    }
    const size_t dq_elems = rows_q * a.D_qk;
    if (cudaError_t err =
            cudaMemsetAsync(dQ_accum, 0, dq_elems * sizeof(float), d.stream());
        err != cudaSuccess) {
      return errors::Internal("FlashAttention: zeroing dQ failed: ",
                              cudaGetErrorString(err));
    }
    TF_RETURN_IF_ERROR(LaunchKV<D_MAX, true>(d, a, dOsum, dQ_accum));
    return GpuLaunchKernel(
        FlashAttnBwdConvertDQKernel<T>, ElementwiseBlocks(dq_elems, kThreads),
        kThreads, 0, d.stream(), dQ_accum, a.dQ, a.BH, a.H, a.S_q, a.D_qk);
  }

  template <int D_MAX, bool kAccumulateDQ>
  absl::Status LaunchKV(const GPUDevice &d, const Args &a, const float *dOsum,
                        float *dQ_accum) const {
    constexpr int Bc = traits::FlashTraitsKV<D_MAX>::Bc;
    constexpr int threads_per_block = traits::FlashTraitsKV<D_MAX>::Th;
    constexpr int smem_size =
        traits::BwdKVSmemBytes<T, D_MAX, kAccumulateDQ>();
    const uint64_t total_blocks =
        static_cast<uint64_t>(a.BH) * ((a.S_kv + Bc - 1) / Bc);
    TF_RETURN_IF_ERROR(CheckGrid(total_blocks));
    return GpuLaunchKernel(FlashAttnBwdKernel<T, D_MAX, kAccumulateDQ>,
                           static_cast<int>(total_blocks), threads_per_block,
                           smem_size, d.stream(), a.Q, a.K, a.V, a.Stats, dOsum,
                           a.dO, a.dK, a.dV, dQ_accum, a.BH, a.H, a.S_q, a.S_kv,
                           a.D_qk, a.D_v, a.causal_mask, a.dropout_rate,
                           a.scale, a.rng, LoadChunks(a));
  }

  template <int D_MAX>
  absl::Status LaunchQ(const GPUDevice &d, const Args &a,
                       const float *dOsum) const {
    constexpr int Br = traits::FlashTraitsQ<D_MAX>::Br;
    constexpr int Bc = traits::FlashTraitsQ<D_MAX>::Bc;
    constexpr int threads_per_block = traits::FlashTraitsQ<D_MAX>::Th;
    // Q, dO (Br rows) + K, V (Bc rows).
    const int smem_size = (2 * Br + 2 * Bc) * D_MAX * sizeof(T);
    const uint64_t total_blocks =
        static_cast<uint64_t>(a.BH) * ((a.S_q + Br - 1) / Br);
    TF_RETURN_IF_ERROR(CheckGrid(total_blocks));
    return GpuLaunchKernel(FlashAttnBwdKernel_Q<T, D_MAX>,
                           static_cast<int>(total_blocks), threads_per_block,
                           smem_size, d.stream(), a.Q, a.K, a.V, a.Stats, dOsum,
                           a.dO, a.dQ, a.BH, a.H, a.S_q, a.S_kv, a.D_qk, a.D_v,
                           a.causal_mask, a.dropout_rate, a.scale, a.rng,
                           LoadChunks(a));
  }
};

template struct FlashAttentionGradFunctor<GPUDevice, float>;
template struct FlashAttentionGradFunctor<GPUDevice, Eigen::half>;
template struct FlashAttentionGradFunctor<GPUDevice, Eigen::bfloat16>;

}  // namespace tensorflow::functor

#endif
