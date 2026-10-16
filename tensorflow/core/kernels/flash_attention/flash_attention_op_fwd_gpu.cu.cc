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
#include <cmath>
#include <type_traits>

#include "tensorflow/core/kernels/flash_attention/fa2_api.h"
#include "tensorflow/core/kernels/flash_attention/flash_attention_op.h"
#include "tensorflow/core/kernels/flash_attention/flash_attention_op_gpu.cu.h"

namespace tensorflow::functor {

template <typename T, int D_MAX>
__global__ void FlashAttnFwdKernel(const T *__restrict__ Q,
                                   const T *__restrict__ K,
                                   const T *__restrict__ V, T *__restrict__ Out,
                                   float *__restrict__ stats_out, int BH, int H,
                                   int S_q, int S_kv, int D_qk, int D_v,
                                   bool causal_mask, float dropout_rate,
                                   float scale, FlashAttentionRng rng) {
  const uint64_t seed = ResolveSeed(rng);
  const uint64_t offset = ResolveOffset(rng);
  constexpr int Br = traits::FlashTraits<D_MAX>::Br;
  constexpr int Bc = traits::FlashTraits<D_MAX>::Bc;
  const float dp_inverse =
      (dropout_rate > 0.0f) ? (1.0f / (1.0f - dropout_rate)) : 1.0f;

  // Shared Memory Layout:
  // Q_tile[Br][D_qk] | K_tile[Bc][D_qk] | V_tile[Bc][D_v]
  extern __shared__ char smem_[];
  T *s_Q = reinterpret_cast<T *>(smem_);
  T *s_K = s_Q + Br * D_MAX;
  T *s_V = s_K + Bc * D_MAX;

  // Grid Mapping
  // blockIdx.x covers all tiles: (Batch * Heads * S_q) / Br
  const int tiles_per_batch = (S_q + Br - 1) / Br;
  const bool block_valid =
      blockIdx.x <
      tiles_per_batch * BH;  // in case current block exceeds total blocks
  const int bh = block_valid ? blockIdx.x / tiles_per_batch : 0;
  const int b = bh / H;
  const int h = bh - b * H;
  // Stats and the dropout RNG index (batch, head) pairs as bh.
  const size_t batch_offset_q = static_cast<size_t>(bh) * S_q;
  const size_t row_base_q = RowBase(b, h, S_q, H);
  const size_t row_base_kv = RowBase(b, h, S_kv, H);

  const int tile_idx_q = blockIdx.x % tiles_per_batch;
  const int tile_offset_q = tile_idx_q * Br;
  const int tid = threadIdx.x;
  const int q_offset = tile_offset_q + tid;
  const bool active = block_valid && (tid < Br) && (q_offset < S_q);

  // Registers for Output Accumulation
  float acc_o[D_MAX];
#pragma unroll
  for (int d = 0; d < D_MAX; ++d) {
    acc_o[d] = ZEROF;
  }
  float m_i = NEGINF;
  float l_i = ZEROF;  // Keeps track of sum(exp(S - m))

  float score_tmp[Bc];
  for (int d = 0; d < Bc; ++d) {
    score_tmp[d] = ZEROF;
  }

  const uint64_t global_rng_idx =
      seed + offset + static_cast<uint64_t>((batch_offset_q + q_offset) * S_kv);

  // Load Q Tile
  if (block_valid) {
    load_tile<T, D_MAX>(s_Q, Q, tid, blockDim.x, Br, D_qk, S_q, tile_offset_q,
                        row_base_q, H);
  }

  T *q_ptr = s_Q + tid * D_MAX;

  for (int tile_offset_kv = 0; tile_offset_kv < S_kv; tile_offset_kv += Bc) {
    bool tile_is_masked =
        (!block_valid) ||
        (causal_mask && (tile_offset_kv > tile_offset_q + Br - 1));

    __syncthreads();

    // Load KV Tile
    if (!tile_is_masked) {
      load_tile<T, D_MAX>(s_K, K, tid, blockDim.x, Bc, D_qk, S_kv,
                          tile_offset_kv, row_base_kv, H);
      load_tile<T, D_MAX>(s_V, V, tid, blockDim.x, Bc, D_v, S_kv,
                          tile_offset_kv, row_base_kv, H);
    }

    __syncthreads();

    // Compute attention inside this tile
    if (active && !tile_is_masked) {
      // find valid tile range
      int valid_start = 0;
      int valid_end =
          (tile_offset_kv + Bc > S_kv) ? (S_kv - tile_offset_kv) : Bc;
      if (causal_mask && (tile_offset_kv + valid_end > q_offset)) {
        valid_end = q_offset - tile_offset_kv + 1;
      }
      if (valid_end > valid_start) {
        // find score max within this kv tile
        float score_max = NEGINF;
        for (int j = valid_start; j < valid_end; ++j) {
          float score = ZEROF;
          T *k_ptr = s_K + j * D_MAX;
#pragma unroll
          for (int d = 0; d < D_MAX; ++d) {
            score += static_cast<float>(q_ptr[d] * k_ptr[d]);
          }
          score *= scale;
          score_tmp[j] = score;
          if (score > score_max) {
            score_max = score;
          }
        }

        const float m_prev = m_i;
        m_i = (score_max > m_i) ? score_max : m_i;
        const float ratio = Eigen::numext::exp(m_prev - m_i);

        // scaling l_i and acc_o
        l_i *= ratio;
#pragma unroll
        for (int d = 0; d < D_MAX; ++d) {
          acc_o[d] *= ratio;
        }

        // adding l_i and acc_o
        for (int j = valid_start; j < valid_end; ++j) {
          const int kv_offset = tile_offset_kv + j;
          float p_ij = Eigen::numext::exp(score_tmp[j] - m_i);
          l_i += p_ij;
          if (dropout_rate > 0.0f) {
            if (get_random_float_v2(global_rng_idx, kv_offset) < dropout_rate) {
              p_ij = ZEROF;
            } else {
              p_ij *= dp_inverse;
            }
          }
          T *v_ptr = s_V + j * D_MAX;
#pragma unroll
          for (int d = 0; d < D_MAX; ++d) {
            acc_o[d] += p_ij * static_cast<float>(v_ptr[d]);
          }
        }
      }
    }
  }

  if (active) {
    // l_i >= 1 once any key is attended (the max term contributes exp(0)), so
    // the LSE is exact. A row without keys gets -inf, the log-sum-exp of the
    // empty set, as for S_kv == 0.
    const bool has_keys = l_i > ZEROF;
    stats_out[batch_offset_q + q_offset] =
        has_keys ? m_i + Eigen::numext::log(l_i) : -INFINITY;
    float inv_l = has_keys ? (1.0f / l_i) : ZEROF;
    const size_t out_base =
        (row_base_q + static_cast<size_t>(q_offset) * H) * D_v;
#pragma unroll
    for (int d = 0; d < D_MAX; ++d) {
      if (d < D_v) {
        Out[out_base + d] = static_cast<T>(acc_o[d] * inv_l);
      }
    }
  }
}

template <typename T>
struct FlashAttentionFunctor<GPUDevice, T> {
  static constexpr bool kIsBf16 = std::is_same_v<T, Eigen::bfloat16>;
  static constexpr bool kIsFa2Type = kIsBf16 || std::is_same_v<T, Eigen::half>;

  static absl::Status CheckFeatureSize(int D_qk, int D_v) {
    if (std::max(D_qk, D_v) <= 128) return absl::OkStatus();
    if constexpr (kIsFa2Type) {
      if (flash_attention_fa2::Supported(D_qk, D_v)) return absl::OkStatus();
    }
    return FeatureSizeError(D_qk, D_v);
  }

  absl::Status operator()(const GPUDevice &d,
                          typename TTypes<T, 4>::ConstTensor Q,
                          typename TTypes<T, 4>::ConstTensor K,
                          typename TTypes<T, 4>::ConstTensor V,
                          typename TTypes<T, 4>::Tensor Out,
                          typename TTypes<float, 3>::Tensor Stats,
                          bool causal_mask, float dropout_rate, float scale,
                          FlashAttentionRng rng) const {
    const int B = Q.dimension(0);
    const int S_q = Q.dimension(1);
    const int H = Q.dimension(2);
    const int D_qk = Q.dimension(3);
    const int S_kv = K.dimension(1);
    const int D_v = V.dimension(3);
    int max_D = std::max(D_qk, D_v);

    if (B == 0 || H == 0 || S_q == 0) return absl::OkStatus();
    if (S_kv == 0) {
      // No keys to attend to: the output is zero, and the log-sum-exp of the
      // empty set is -inf.
      TF_RETURN_IF_ERROR(
          ZeroOutputs(d, {{Out.data(), Out.size() * sizeof(T)}}));
      return FillFloats(d, Stats.data(), Stats.size(), -INFINITY);
    }

    if constexpr (kIsFa2Type) {
      if (flash_attention_fa2::Supported(D_qk, D_v) &&
          flash_attention_fa2::GridSupported(B, H)) {
        const cudaError_t err = flash_attention_fa2::Forward(
            d.stream(), kIsBf16, Q.data(), K.data(), V.data(), Out.data(),
            Stats.data(), B, H, S_q, S_kv, D_qk, scale, dropout_rate,
            causal_mask, rng.seed, rng.offset, rng.seed_dev, rng.offset_dev);
        if (err != cudaSuccess) {
          return errors::Internal("FlashAttention-2 forward failed: ",
                                  cudaGetErrorString(err));
        }
        return absl::OkStatus();
      }
    }

    if (max_D <= 8) {
      return LaunchKernel<8>(d, Q, K, V, Out, Stats, B, H, S_q, S_kv, D_qk, D_v,
                             causal_mask, dropout_rate, scale, rng);
    } else if (max_D <= 16) {
      return LaunchKernel<16>(d, Q, K, V, Out, Stats, B, H, S_q, S_kv, D_qk,
                              D_v, causal_mask, dropout_rate, scale, rng);
    } else if (max_D <= 32) {
      return LaunchKernel<32>(d, Q, K, V, Out, Stats, B, H, S_q, S_kv, D_qk,
                              D_v, causal_mask, dropout_rate, scale, rng);
    } else if (max_D <= 64) {
      return LaunchKernel<64>(d, Q, K, V, Out, Stats, B, H, S_q, S_kv, D_qk,
                              D_v, causal_mask, dropout_rate, scale, rng);
    } else if (max_D <= 128) {
      return LaunchKernel<128>(d, Q, K, V, Out, Stats, B, H, S_q, S_kv, D_qk,
                               D_v, causal_mask, dropout_rate, scale, rng);
    } else {
      return FeatureSizeError(D_qk, D_v);
    }
  }

  template <int D_MAX>
  absl::Status LaunchKernel(const GPUDevice &d,
                            typename TTypes<T, 4>::ConstTensor Q,
                            typename TTypes<T, 4>::ConstTensor K,
                            typename TTypes<T, 4>::ConstTensor V,
                            typename TTypes<T, 4>::Tensor Out,
                            typename TTypes<float, 3>::Tensor Stats, int B,
                            int H, int S_q, int S_kv, int D_qk, int D_v,
                            bool causal_mask, float dropout_rate, float scale,
                            FlashAttentionRng rng) const {
    constexpr int Br = traits::FlashTraits<D_MAX>::Br;
    constexpr int Bc = traits::FlashTraits<D_MAX>::Bc;
    constexpr int threads_per_block = traits::FlashTraits<D_MAX>::Th;
    int smem_size = (Br + Bc + Bc) * D_MAX * sizeof(T);
    uint64_t blocks_per_batch = (S_q + Br - 1) / Br;
    const int BH = B * H;  // fits in int, see CheckDimsFitInt32
    uint64_t total_blocks = static_cast<uint64_t>(BH) * blocks_per_batch;
    if (total_blocks > 2147483647ULL) {
      return errors::InvalidArgument(
          "FlashAttention: total blocks exceed 1D grid limit");
    }
    return GpuLaunchKernel(FlashAttnFwdKernel<T, D_MAX>, total_blocks,
                           threads_per_block, smem_size, d.stream(), Q.data(),
                           K.data(), V.data(), Out.data(), Stats.data(), BH, H,
                           S_q, S_kv, D_qk, D_v, causal_mask, dropout_rate,
                           scale, rng);
  }
};

template struct FlashAttentionFunctor<GPUDevice, float>;
template struct FlashAttentionFunctor<GPUDevice, Eigen::half>;
template struct FlashAttentionFunctor<GPUDevice, Eigen::bfloat16>;

}  // namespace tensorflow::functor

#endif
