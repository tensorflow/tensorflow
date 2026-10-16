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

// use both kv loop & q loop to avoid atomic add

namespace tensorflow::functor {

template <typename T, int D_MAX>
__global__ void FlashAttnBwdKernel_KV(
    const T *__restrict__ Q, const T *__restrict__ K, const T *__restrict__ V,
    const T *__restrict__ Out, const float *__restrict__ Stats,
    const T *__restrict__ dO, T *__restrict__ dK, T *__restrict__ dV, int BH,
    int H, int S_q, int S_kv, int D_qk, int D_v, bool causal_mask,
    float dropout_rate, float scale, FlashAttentionRng rng) {
  const uint64_t seed = ResolveSeed(rng);
  const uint64_t offset = ResolveOffset(rng);
  constexpr int Br = traits::FlashTraitsKV<D_MAX>::Br;
  constexpr int Bc = traits::FlashTraitsKV<D_MAX>::Bc;
  const float dp_inverse =
      (dropout_rate > 0.0f) ? (1.0f / (1.0f - dropout_rate)) : 1.0f;

  extern __shared__ char smem_[];
  T *s_K = reinterpret_cast<T *>(smem_);
  T *s_V = s_K + Bc * D_MAX;
  T *s_Q = s_V + Bc * D_MAX;
  T *s_dO = s_Q + Br * D_MAX;
  float *s_Stats = reinterpret_cast<float *>(s_dO + Br * D_MAX);  // Br * 1
  float *s_dOsum = s_Stats + Br;                                  // Br * 1

  const int tid = threadIdx.x;  // kv id in column (0 ~ Bc)

  const int tiles_per_batch = (S_kv + Bc - 1) / Bc;
  const bool block_valid = blockIdx.x < tiles_per_batch * BH;
  const int bh = block_valid ? blockIdx.x / tiles_per_batch : 0;
  const int b = bh / H;
  const int h = bh - b * H;

  const int tile_idx_kv = blockIdx.x % tiles_per_batch;
  const int tile_offset_kv = tile_idx_kv * Bc;
  const int kv_offset = tile_offset_kv + tid;

  const bool active = block_valid && (tid < Bc) && (kv_offset < S_kv);
  // Stats and the dropout RNG index (batch, head) pairs as bh.
  const size_t batch_offset_q = static_cast<size_t>(bh) * S_q;
  const size_t row_base_q = RowBase(b, h, S_q, H);
  const size_t row_base_kv = RowBase(b, h, S_kv, H);
  const uint64_t global_rng_idx =
      seed + offset + static_cast<uint64_t>(kv_offset);

  // Local accumulators for the specific column(s) this thread handles
  float r_dK[D_MAX];
  float r_dV[D_MAX];
#pragma unroll
  for (int d = 0; d < D_MAX; ++d) {
    r_dK[d] = ZEROF;
    r_dV[d] = ZEROF;
  }

  // 1. Load K_j and V_j tile into Shared Memory
  if (block_valid) {
    load_tile<T, D_MAX>(s_K, K, tid, blockDim.x, Bc, D_qk, S_kv, tile_offset_kv,
                        row_base_kv, H);
    load_tile<T, D_MAX>(s_V, V, tid, blockDim.x, Bc, D_v, S_kv, tile_offset_kv,
                        row_base_kv, H);
  }

  T *k_ptr = (tid < Bc) ? s_K + tid * D_MAX : nullptr;
  T *v_ptr = (tid < Bc) ? s_V + tid * D_MAX : nullptr;

  // 2. Inner Loop: Iterate over Query blocks (FlashAttention-2 style)
  for (int tile_offset_q = 0; tile_offset_q < S_q; tile_offset_q += Br) {
    // Check if this entire query block is masked for this KV tile
    const bool tile_is_masked =
        (!block_valid) ||
        (causal_mask && ((tile_offset_q + Br - 1) < tile_offset_kv));

    __syncthreads();

    if (!tile_is_masked) {
      // Load Q_i and dO_i into Shared Memory
      load_tile<T, D_MAX>(s_Q, Q, tid, blockDim.x, Br, D_qk, S_q, tile_offset_q,
                          row_base_q, H);
      load_tile<T, D_MAX>(s_dO, dO, tid, blockDim.x, Br, D_v, S_q,
                          tile_offset_q, row_base_q, H);
    }

    __syncthreads();

    if (!tile_is_masked) {
      // load stats & dOsum
      for (int r = tid; r < Br; r += blockDim.x) {
        const int q_offset = tile_offset_q + r;
        float dOsum_val = ZEROF;
        float stats_val = ZEROF;
        if (q_offset < S_q) {
          const size_t out_row = row_base_q + static_cast<size_t>(q_offset) * H;
          for (int d = 0; d < D_v; ++d) {
            dOsum_val += static_cast<float>(s_dO[r * D_MAX + d] *
                                            Out[out_row * D_v + d]);
          }
          stats_val = Stats[batch_offset_q + q_offset];
        }
        s_dOsum[r] = dOsum_val;
        s_Stats[r] = stats_val;
      }
    }

    __syncthreads();

    if (active && !tile_is_masked) {
      // Process current Q-block row by row
      for (int i = 0; i < Br && (tile_offset_q + i) < S_q; ++i) {
        const int q_offset = tile_offset_q + i;
        const size_t q_idx = batch_offset_q + q_offset;

        // Current thread calculates contribution for column j = kv_offset
        const bool not_masked = !(causal_mask && (q_offset < kv_offset));

        if (not_masked) {
          T *q_ptr = s_Q + i * D_MAX;
          T *dO_ptr = s_dO + i * D_MAX;

          float score = ZEROF;
          for (int d = 0; d < D_qk; ++d) {
            score += static_cast<float>(q_ptr[d] * k_ptr[d]);
          }
          float p_ij = Eigen::numext::exp(score * scale - s_Stats[i]);
          float dS = scale;
          bool is_dropped = false;
          if (dropout_rate > 0.0f) {
            if (get_random_float_v2(global_rng_idx, q_idx * S_kv) <
                dropout_rate) {
              is_dropped = true;
              dS *= -p_ij * s_dOsum[i];
              p_ij = ZEROF;
            } else {
              p_ij *= dp_inverse;
            }
          }

          if (!is_dropped) {
            float dP = ZEROF;
            for (int d = 0; d < D_v; ++d) {
              dP += static_cast<float>(dO_ptr[d] * v_ptr[d]);
            }
            dS *= p_ij * (dP - s_dOsum[i] / dp_inverse);
          }

#pragma unroll
          for (int d = 0; d < D_MAX; ++d) {
            if (d < D_v) {
              r_dV[d] += p_ij * static_cast<float>(dO_ptr[d]);
            }
            if (d < D_qk) {
              r_dK[d] += dS * static_cast<float>(q_ptr[d]);
            }
          }
        }
      }
    }
  }

  // 3. Final global write for dK and dV (One write per sequence element)
  if (active) {
    const size_t kv_idx = row_base_kv + static_cast<size_t>(kv_offset) * H;
#pragma unroll
    for (int d = 0; d < D_MAX; ++d) {
      if (d < D_qk) {
        dK[kv_idx * D_qk + d] = static_cast<T>(r_dK[d]);
      }
      if (d < D_v) {
        dV[kv_idx * D_v + d] = static_cast<T>(r_dV[d]);
      }
    }
  }
}

template <typename T, int D_MAX>
__global__ void FlashAttnBwdKernel_Q(
    const T *__restrict__ Q, const T *__restrict__ K, const T *__restrict__ V,
    const T *__restrict__ Out, const float *__restrict__ Stats,
    const T *__restrict__ dO, T *__restrict__ dQ, int BH, int H, int S_q,
    int S_kv, int D_qk, int D_v, bool causal_mask, float dropout_rate,
    float scale, FlashAttentionRng rng) {
  const uint64_t seed = ResolveSeed(rng);
  const uint64_t offset = ResolveOffset(rng);
  constexpr int Br = traits::FlashTraitsQ<D_MAX>::Br;
  constexpr int Bc = traits::FlashTraitsQ<D_MAX>::Bc;
  const float dp_inverse =
      (dropout_rate > 0.0f) ? (1.0f / (1.0f - dropout_rate)) : 1.0f;

  // Shared Memory Layout:
  // Q_tile[Br][D_qk] | dO_tile[Br][D_v] | K_tile[Bc][D_qk] | V_tile[Bc][D_v]
  extern __shared__ char smem_[];
  T *s_Q = reinterpret_cast<T *>(smem_);
  T *s_dO = s_Q + Br * D_MAX;
  T *s_K = s_dO + Br * D_MAX;
  T *s_V = s_K + Bc * D_MAX;

  const int tid = threadIdx.x;
  const int tiles_per_batch = (S_q + Br - 1) / Br;
  const bool block_valid = blockIdx.x < tiles_per_batch * BH;
  const int bh = block_valid ? blockIdx.x / tiles_per_batch : 0;
  const int b = bh / H;
  const int h = bh - b * H;
  const int tile_idx_q = blockIdx.x % tiles_per_batch;
  const int tile_offset_q = tile_idx_q * Br;
  const int q_offset = tile_offset_q + tid;
  const bool active = block_valid && (tid < Br) && (q_offset < S_q);
  const size_t row_base_q = RowBase(b, h, S_q, H);
  const size_t row_base_kv = RowBase(b, h, S_kv, H);
  // Index into Stats and the dropout RNG, which use bh for (batch, head).
  const size_t q_idx = static_cast<size_t>(bh) * S_q + q_offset;
  // Row of this query position in Q/Out/dO/dQ.
  const size_t q_row = row_base_q + static_cast<size_t>(q_offset) * H;
  const uint64_t global_rng_idx =
      seed + offset + static_cast<uint64_t>(q_idx * S_kv);

  // dQ accumulator
  float r_dQ[D_MAX];
#pragma unroll
  for (int d = 0; d < D_MAX; ++d) {
    r_dQ[d] = ZEROF;
  }

  // 1. Load Q_i and dO_i tile into Shared Memory
  if (block_valid) {
    load_tile<T, D_MAX>(s_Q, Q, tid, blockDim.x, Br, D_qk, S_q, tile_offset_q,
                        row_base_q, H);
    load_tile<T, D_MAX>(s_dO, dO, tid, blockDim.x, Br, D_v, S_q, tile_offset_q,
                        row_base_q, H);
  }
  __syncthreads();
  T *q_ptr = (tid < Br) ? s_Q + tid * D_MAX : nullptr;
  T *dO_ptr = (tid < Br) ? s_dO + tid * D_MAX : nullptr;

  // 2. pre sum dOsum
  float dOsum = ZEROF;
  float stats_val = ZEROF;
  if (active) {
    stats_val = Stats[q_idx];
    const size_t out_offset = q_row * D_v;
    for (int d = 0; d < D_v; ++d) {
      dOsum += static_cast<float>(dO_ptr[d] * Out[out_offset + d]);
    }
  }

  // 3. Inner Loop
  for (int tile_offset_kv = 0; tile_offset_kv < S_kv; tile_offset_kv += Bc) {
    const bool tile_is_masked =
        (!block_valid) ||
        (causal_mask && (tile_offset_kv > tile_offset_q + Br - 1));

    __syncthreads();
    if (!tile_is_masked) {
      // Load K_j and V_j into Shared Memory
      load_tile<T, D_MAX>(s_K, K, tid, blockDim.x, Bc, D_qk, S_kv,
                          tile_offset_kv, row_base_kv, H);
      load_tile<T, D_MAX>(s_V, V, tid, blockDim.x, Bc, D_v, S_kv,
                          tile_offset_kv, row_base_kv, H);
    }

    __syncthreads();

    if (active && !tile_is_masked) {
      for (int j = 0; j < Bc && (tile_offset_kv + j) < S_kv; ++j) {
        const int kv_offset = tile_offset_kv + j;
        const bool not_masked = !(causal_mask && (q_offset < kv_offset));

        if (not_masked) {
          T *k_ptr = s_K + j * D_MAX;

          float score = ZEROF;
          for (int d = 0; d < D_qk; ++d) {
            score += static_cast<float>(q_ptr[d] * k_ptr[d]);
          }
          float p_ij = Eigen::numext::exp(score * scale - stats_val);
          float dS = scale;
          bool is_dropped = false;
          if (dropout_rate > 0.0f) {
            if (get_random_float_v2(global_rng_idx, kv_offset) < dropout_rate) {
              is_dropped = true;
              dS *= -p_ij * dOsum;
              p_ij = ZEROF;
            } else {
              p_ij *= dp_inverse;
            }
          }

          if (!is_dropped) {
            T *v_ptr = s_V + j * D_MAX;
            float dP = ZEROF;
            for (int d = 0; d < D_v; ++d) {
              dP += static_cast<float>(dO_ptr[d] * v_ptr[d]);
            }
            dS *= p_ij * (dP - dOsum / dp_inverse);
          }

#pragma unroll
          for (int d = 0; d < D_MAX; ++d) {
            if (d < D_qk) r_dQ[d] += dS * static_cast<float>(k_ptr[d]);
          }
        }
      }
    }
  }

  if (active) {
    const size_t out_offset = q_row * D_qk;
#pragma unroll
    for (int d = 0; d < D_MAX; ++d) {
      if (d < D_qk) {
        dQ[out_offset + d] = static_cast<T>(r_dQ[d]);
      }
    }
  }
}

template <typename T>
struct FlashAttentionGradFunctor<GPUDevice, T> {
  static constexpr bool kIsBf16 = std::is_same_v<T, Eigen::bfloat16>;
  static constexpr bool kIsFa2Type = kIsBf16 || std::is_same_v<T, Eigen::half>;

  // Shape-only (no device query) so XLA can size the buffer at compile time.
  static int64_t WorkspaceBytes(int B, int H, int S_q, int S_kv, int D_qk,
                                int D_v) {
    if constexpr (kIsFa2Type) {
      if (flash_attention_fa2::ShapeSupported(D_qk, D_v) &&
          flash_attention_fa2::GridSupported(B, H)) {
        return flash_attention_fa2::BackwardWorkspaceBytes(B, H, S_q, D_qk);
      }
    }
    return 0;
  }

  static absl::Status CheckFeatureSize(int D_qk, int D_v, float dropout_rate) {
    if (std::max(D_qk, D_v) <= 128) return absl::OkStatus();
    if constexpr (kIsFa2Type) {
      if (flash_attention_fa2::BackwardSupported(D_qk, D_v, dropout_rate))
        return absl::OkStatus();
    }
    return FeatureSizeError(D_qk, D_v);
  }

  absl::Status operator()(const GPUDevice &d,
                          typename TTypes<T, 4>::ConstTensor Q,
                          typename TTypes<T, 4>::ConstTensor K,
                          typename TTypes<T, 4>::ConstTensor V,
                          typename TTypes<T, 4>::ConstTensor Out,
                          typename TTypes<float, 3>::ConstTensor Stats,
                          typename TTypes<T, 4>::ConstTensor dO,
                          typename TTypes<T, 4>::Tensor dQ,
                          typename TTypes<T, 4>::Tensor dK,
                          typename TTypes<T, 4>::Tensor dV, bool causal_mask,
                          float dropout_rate, float scale,
                          FlashAttentionRng rng, void *workspace) const {
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
            B, H, S_q, S_kv, D_qk, scale, dropout_rate, causal_mask, rng.seed,
            rng.offset, rng.seed_dev, rng.offset_dev);
        if (err != cudaSuccess) {
          return errors::Internal("FlashAttention-2 backward failed: ",
                                  cudaGetErrorString(err));
        }
        return absl::OkStatus();
      }
    }

    int max_D = std::max(D_qk, D_v);
    if (max_D <= 8) {
      return LaunchKernel<8>(d, Q, K, V, Out, Stats, dO, dQ, dK, dV, B, H, S_q,
                             S_kv, D_qk, D_v, causal_mask, dropout_rate, scale,
                             rng);
    } else if (max_D <= 16) {
      return LaunchKernel<16>(d, Q, K, V, Out, Stats, dO, dQ, dK, dV, B, H, S_q,
                              S_kv, D_qk, D_v, causal_mask, dropout_rate, scale,
                              rng);
    } else if (max_D <= 32) {
      return LaunchKernel<32>(d, Q, K, V, Out, Stats, dO, dQ, dK, dV, B, H, S_q,
                              S_kv, D_qk, D_v, causal_mask, dropout_rate, scale,
                              rng);
    } else if (max_D <= 64) {
      return LaunchKernel<64>(d, Q, K, V, Out, Stats, dO, dQ, dK, dV, B, H, S_q,
                              S_kv, D_qk, D_v, causal_mask, dropout_rate, scale,
                              rng);
    } else if (max_D <= 128) {
      return LaunchKernel<128>(d, Q, K, V, Out, Stats, dO, dQ, dK, dV, B, H,
                               S_q, S_kv, D_qk, D_v, causal_mask, dropout_rate,
                               scale, rng);
    } else {
      return FeatureSizeError(D_qk, D_v);
    }
  }

  template <int D_MAX>
  absl::Status LaunchKernel(
      const GPUDevice &d, typename TTypes<T, 4>::ConstTensor Q,
      typename TTypes<T, 4>::ConstTensor K,
      typename TTypes<T, 4>::ConstTensor V,
      typename TTypes<T, 4>::ConstTensor Out,
      typename TTypes<float, 3>::ConstTensor Stats,
      typename TTypes<T, 4>::ConstTensor dO, typename TTypes<T, 4>::Tensor dQ,
      typename TTypes<T, 4>::Tensor dK, typename TTypes<T, 4>::Tensor dV, int B,
      int H, int S_q, int S_kv, int D_qk, int D_v, bool causal_mask,
      float dropout_rate, float scale, FlashAttentionRng rng) const {
    const int BH = B * H;  // fits in int, see CheckDimsFitInt32
    TF_RETURN_IF_ERROR(LaunchKernelKV<D_MAX>(
        d, Q, K, V, Out, Stats, dO, dK, dV, BH, H, S_q, S_kv, D_qk, D_v,
        causal_mask, dropout_rate, scale, rng));
    return LaunchKernelQ<D_MAX>(d, Q, K, V, Out, Stats, dO, dQ, BH, H, S_q,
                                S_kv, D_qk, D_v, causal_mask, dropout_rate,
                                scale, rng);
  }

  template <int D_MAX>
  absl::Status LaunchKernelKV(const GPUDevice &d,
                              typename TTypes<T, 4>::ConstTensor Q,
                              typename TTypes<T, 4>::ConstTensor K,
                              typename TTypes<T, 4>::ConstTensor V,
                              typename TTypes<T, 4>::ConstTensor Out,
                              typename TTypes<float, 3>::ConstTensor Stats,
                              typename TTypes<T, 4>::ConstTensor dO,
                              typename TTypes<T, 4>::Tensor dK,
                              typename TTypes<T, 4>::Tensor dV, int BH, int H,
                              int S_q, int S_kv, int D_qk, int D_v,
                              bool causal_mask, float dropout_rate, float scale,
                              FlashAttentionRng rng) const {
    constexpr int Br = traits::FlashTraitsKV<D_MAX>::Br;
    constexpr int Bc = traits::FlashTraitsKV<D_MAX>::Bc;
    constexpr int threads_per_block = traits::FlashTraitsKV<D_MAX>::Th;
    // Shared Memory size: Q, dO (Br rows) + K, V (Bc rows) + Stats, dOsum (Br
    // floats)
    int smem_size =
        (2 * Br + 2 * Bc) * D_MAX * sizeof(T) + 2 * Br * sizeof(float);
    uint64_t blocks_per_batch = (S_kv + Bc - 1) / Bc;
    uint64_t total_blocks = static_cast<uint64_t>(BH) * blocks_per_batch;
    if (total_blocks > 2147483647ULL) {
      return errors::InvalidArgument(
          "FlashAttention: total blocks exceed 1D grid limit");
    }
    return GpuLaunchKernel(FlashAttnBwdKernel_KV<T, D_MAX>, total_blocks,
                           threads_per_block, smem_size, d.stream(), Q.data(),
                           K.data(), V.data(), Out.data(), Stats.data(),
                           dO.data(), dK.data(), dV.data(), BH, H, S_q, S_kv,
                           D_qk, D_v, causal_mask, dropout_rate, scale, rng);
  }

  template <int D_MAX>
  absl::Status LaunchKernelQ(const GPUDevice &d,
                             typename TTypes<T, 4>::ConstTensor Q,
                             typename TTypes<T, 4>::ConstTensor K,
                             typename TTypes<T, 4>::ConstTensor V,
                             typename TTypes<T, 4>::ConstTensor Out,
                             typename TTypes<float, 3>::ConstTensor Stats,
                             typename TTypes<T, 4>::ConstTensor dO,
                             typename TTypes<T, 4>::Tensor dQ, int BH, int H,
                             int S_q, int S_kv, int D_qk, int D_v,
                             bool causal_mask, float dropout_rate, float scale,
                             FlashAttentionRng rng) const {
    constexpr int Br = traits::FlashTraitsQ<D_MAX>::Br;
    constexpr int Bc = traits::FlashTraitsQ<D_MAX>::Bc;
    constexpr int threads_per_block = traits::FlashTraitsQ<D_MAX>::Th;

    // Shared Memory size: Q, dO (Br rows) + K, V (Bc rows) + Stats, dOsum (Br
    // floats)
    int smem_size = (2 * Br + 2 * Bc) * D_MAX * sizeof(T);
    uint64_t blocks_per_batch = (S_q + Br - 1) / Br;
    uint64_t total_blocks = static_cast<uint64_t>(BH) * blocks_per_batch;
    if (total_blocks > 2147483647ULL) {
      return errors::InvalidArgument(
          "FlashAttention: total blocks exceed 1D grid limit");
    }
    return GpuLaunchKernel(FlashAttnBwdKernel_Q<T, D_MAX>, total_blocks,
                           threads_per_block, smem_size, d.stream(), Q.data(),
                           K.data(), V.data(), Out.data(), Stats.data(),
                           dO.data(), dQ.data(), BH, H, S_q, S_kv, D_qk, D_v,
                           causal_mask, dropout_rate, scale, rng);
  }
};

template struct FlashAttentionGradFunctor<GPUDevice, float>;
template struct FlashAttentionGradFunctor<GPUDevice, Eigen::half>;
template struct FlashAttentionGradFunctor<GPUDevice, Eigen::bfloat16>;

}  // namespace tensorflow::functor

#endif
