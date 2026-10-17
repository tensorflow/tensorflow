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

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/flash_attention/flash_attention_op.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/determinism.h"

namespace tensorflow {

// Checks key/value against query's batch, heads and feature size. Both are
// [batch, seq_kv, heads, dim].
static absl::Status CheckKV(const Tensor &k, const Tensor &v, int64_t B,
                            int64_t H, int64_t D_qk) {
  if (k.dim_size(0) != B || v.dim_size(0) != B) {
    return errors::InvalidArgument("K/V batch size mismatch");
  }
  if (k.dim_size(2) != H || v.dim_size(2) != H) {
    return errors::InvalidArgument("K/V heads mismatch");
  }
  if (k.dim_size(3) != D_qk) {
    return errors::InvalidArgument("K feature size mismatch");
  }
  if (v.dim_size(1) != k.dim_size(1)) {
    return errors::InvalidArgument("V seq size mismatch");
  }
  return absl::OkStatus();
}

static absl::Status ExpectShape(absl::string_view name, const Tensor &t,
                                const TensorShape &want) {
  if (t.shape() != want) {
    return errors::InvalidArgument(name, " has shape ", t.shape().DebugString(),
                                   ", expected ", want.DebugString());
  }
  return absl::OkStatus();
}

template <typename Device, typename T>
class FlashAttentionOp : public OpKernel {
 private:
  float dropout_rate_;
  float scale_;
  bool causal_mask_;

 public:
  explicit FlashAttentionOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dropout_rate", &dropout_rate_));
    OP_REQUIRES(context, dropout_rate_ >= 0.0 && dropout_rate_ < 1.0,
                errors::InvalidArgument("dropout_rate must be in [0, 1)"));
    OP_REQUIRES_OK(context, context->GetAttr("scale", &scale_));
    OP_REQUIRES_OK(context, context->GetAttr("is_causal", &causal_mask_));
  }

  void Compute(OpKernelContext *context) override {
    const Tensor &q_tensor = context->input(0);
    const Tensor &k_tensor = context->input(1);
    const Tensor &v_tensor = context->input(2);
    const Tensor &seed_tensor = context->input(3);
    const Tensor &offset_tensor = context->input(4);

    OP_REQUIRES(context, TensorShapeUtils::IsScalar(seed_tensor.shape()),
                errors::InvalidArgument("seed must be scalar"));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(offset_tensor.shape()),
                errors::InvalidArgument("offset must be scalar"));

    uint64_t seed_val = seed_tensor.scalar<uint64_t>()();
    uint64_t offset_val = offset_tensor.scalar<uint64_t>()();

    for (const Tensor *t : {&q_tensor, &k_tensor, &v_tensor}) {
      OP_REQUIRES(context, t->dims() == 4,
                  errors::InvalidArgument(
                      "query, key, and value must be rank 4 ([batch, seq, "
                      "heads, dim]), got shape ",
                      t->shape().DebugString()));
    }

    const int64_t B = q_tensor.dim_size(0);
    const int64_t S_q = q_tensor.dim_size(1);
    const int64_t H = q_tensor.dim_size(2);
    const int64_t D_qk = q_tensor.dim_size(3);
    const int64_t S_kv = k_tensor.dim_size(1);
    const int64_t D_v = v_tensor.dim_size(3);
    OP_REQUIRES_OK(context,
                   functor::CheckDimsFitInt32(B, H, S_q, S_kv, D_qk, D_v));

    OP_REQUIRES_OK(context, CheckKV(k_tensor, v_tensor, B, H, D_qk));

    if (causal_mask_) {
      OP_REQUIRES(
          context, S_q == S_kv,
          errors::InvalidArgument("seq size Q != K while is_causal = True"));
    }

    OP_REQUIRES_OK(
        context,
        functor::FlashAttentionFunctor<Device, T>::CheckFeatureSize(D_qk, D_v));

    // Each dimension fits in int32, but the output's element count can still
    // overflow (e.g. a large D_v with S_kv = 0); BuildTensorShape reports that
    // instead of CHECK-failing.
    TensorShape output_shape, stats_shape;
    OP_REQUIRES_OK(context, TensorShape::BuildTensorShape({B, S_q, H, D_v},
                                                          &output_shape));
    OP_REQUIRES_OK(context,
                   TensorShape::BuildTensorShape({B, H, S_q}, &stats_shape));

    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_tensor));

    Tensor *stats_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, stats_shape, &stats_tensor));

    auto Q = q_tensor.template tensor<T, 4>();
    auto K = k_tensor.template tensor<T, 4>();
    auto V = v_tensor.template tensor<T, 4>();
    auto Out = output_tensor->template tensor<T, 4>();
    auto Stats = stats_tensor->template tensor<float, 3>();

    const Device &d = context->template eigen_device<Device>();
    OP_REQUIRES_OK(
        context, functor::FlashAttentionFunctor<Device, T>()(
                     d, Q, K, V, Out, Stats, causal_mask_, dropout_rate_,
                     scale_, functor::FlashAttentionRng{seed_val, offset_val}));
  }
};

template <typename Device, typename T>
class FlashAttentionGradOp : public OpKernel {
 private:
  float dropout_rate_;
  float scale_;
  bool causal_mask_;

 public:
  explicit FlashAttentionGradOp(OpKernelConstruction *context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dropout_rate", &dropout_rate_));
    OP_REQUIRES(context, dropout_rate_ >= 0.0 && dropout_rate_ < 1.0,
                errors::InvalidArgument("dropout_rate must be in [0, 1)"));
    OP_REQUIRES_OK(context, context->GetAttr("scale", &scale_));
    OP_REQUIRES_OK(context, context->GetAttr("is_causal", &causal_mask_));
  }

  void Compute(OpKernelContext *context) override {
    const Tensor &q_tensor = context->input(0);
    const Tensor &k_tensor = context->input(1);
    const Tensor &v_tensor = context->input(2);
    const Tensor &out_tensor = context->input(3);
    const Tensor &stats_tensor = context->input(4);
    const Tensor &do_tensor = context->input(5);
    const Tensor &seed_tensor = context->input(6);
    const Tensor &offset_tensor = context->input(7);

    OP_REQUIRES(context, TensorShapeUtils::IsScalar(seed_tensor.shape()),
                errors::InvalidArgument("seed must be scalar"));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(offset_tensor.shape()),
                errors::InvalidArgument("offset must be scalar"));

    uint64_t seed_val = seed_tensor.scalar<uint64_t>()();
    uint64_t offset_val = offset_tensor.scalar<uint64_t>()();

    for (const Tensor *t : {&q_tensor, &k_tensor, &v_tensor}) {
      OP_REQUIRES(context, t->dims() == 4,
                  errors::InvalidArgument(
                      "query, key, and value must be rank 4 ([batch, seq, "
                      "heads, dim]), got shape ",
                      t->shape().DebugString()));
    }

    const int64_t B = q_tensor.dim_size(0);
    const int64_t S_q = q_tensor.dim_size(1);
    const int64_t H = q_tensor.dim_size(2);
    const int64_t D_qk = q_tensor.dim_size(3);
    const int64_t S_kv = k_tensor.dim_size(1);
    const int64_t D_v = v_tensor.dim_size(3);
    OP_REQUIRES_OK(context,
                   functor::CheckDimsFitInt32(B, H, S_q, S_kv, D_qk, D_v));

    OP_REQUIRES_OK(context, CheckKV(k_tensor, v_tensor, B, H, D_qk));
    // Built with BuildTensorShape: see FlashAttentionOp.
    TensorShape out_shape, stats_shape;
    OP_REQUIRES_OK(context,
                   TensorShape::BuildTensorShape({B, S_q, H, D_v}, &out_shape));
    OP_REQUIRES_OK(context,
                   TensorShape::BuildTensorShape({B, H, S_q}, &stats_shape));
    OP_REQUIRES_OK(context, ExpectShape("output", out_tensor, out_shape));
    OP_REQUIRES_OK(context, ExpectShape("output_grad", do_tensor, out_shape));
    OP_REQUIRES_OK(context,
                   ExpectShape("softmax_lse", stats_tensor, stats_shape));

    if (causal_mask_) {
      OP_REQUIRES(
          context, S_q == S_kv,
          errors::InvalidArgument("seq size Q != K while is_causal = True"));
    }

    OP_REQUIRES_OK(
        context,
        functor::FlashAttentionGradFunctor<Device, T>::CheckFeatureSize(
            D_qk, D_v, dropout_rate_));

    Tensor *dq_tensor = nullptr;
    Tensor *dk_tensor = nullptr;
    Tensor *dv_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, q_tensor.shape(), &dq_tensor));
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, k_tensor.shape(), &dk_tensor));
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, v_tensor.shape(), &dv_tensor));

    auto Q = q_tensor.template tensor<T, 4>();
    auto K = k_tensor.template tensor<T, 4>();
    auto V = v_tensor.template tensor<T, 4>();
    auto Out = out_tensor.template tensor<T, 4>();
    auto dO = do_tensor.template tensor<T, 4>();
    auto Stats = stats_tensor.template tensor<float, 3>();

    auto dQ = dq_tensor->template tensor<T, 4>();
    auto dK = dk_tensor->template tensor<T, 4>();
    auto dV = dv_tensor->template tensor<T, 4>();

    const bool deterministic = OpDeterminismRequired();
    Tensor workspace;
    void *workspace_ptr = nullptr;
    const int64_t workspace_bytes =
        functor::FlashAttentionGradFunctor<Device, T>::WorkspaceBytes(
            B, H, S_q, S_kv, D_qk, D_v, deterministic);
    OP_REQUIRES_OK(
        context, context->allocate_temp(DT_INT8, TensorShape({workspace_bytes}),
                                        &workspace));
    workspace_ptr = workspace.flat<int8_t>().data();

    const Device &d = context->template eigen_device<Device>();
    OP_REQUIRES_OK(
        context,
        functor::FlashAttentionGradFunctor<Device, T>()(
            d, Q, K, V, Out, Stats, dO, dQ, dK, dV, causal_mask_, dropout_rate_,
            scale_, functor::FlashAttentionRng{seed_val, offset_val},
            deterministic, workspace_ptr));
  }
};

namespace functor {
// cpu implemetation
static float get_random_float_cpu(uint64_t seed, uint64_t offset, int64_t b,
                                  int i, int j, int stride_m, int stride_n) {
  uint64_t batch_stride = (uint64_t)stride_m * (uint64_t)stride_n;
  uint64_t idx = (uint64_t)b * batch_stride + (uint64_t)i * (uint64_t)stride_n +
                 (uint64_t)j;

  uint64_t global_idx = offset + idx;

  uint64_t state = seed + global_idx;
  state = state * 6364136223846793005ULL + 1442695040888963407ULL;

  uint32_t val = (state >> 32) & 0xFFFFFFFF;
  return static_cast<float>(val >> 8) * (1.0f / 16777216.0f);
}

template <typename T>
struct FlashAttentionFunctor<CPUDevice, T> {
  static absl::Status CheckFeatureSize(int D_qk, int D_v) {
    return std::max(D_qk, D_v) <= 128 ? absl::OkStatus()
                                      : FeatureSizeError(D_qk, D_v);
  }

  absl::Status operator()(const CPUDevice &d,
                          typename TTypes<T, 4>::ConstTensor Q,
                          typename TTypes<T, 4>::ConstTensor K,
                          typename TTypes<T, 4>::ConstTensor V,
                          typename TTypes<T, 4>::Tensor Out,
                          typename TTypes<float, 3>::Tensor stats,
                          bool causal_mask, float dropout_rate, float scale,
                          FlashAttentionRng rng) const {
    const uint64_t seed = rng.seed;
    const uint64_t offset = rng.offset;
    const int B = Q.dimension(0);
    const int S_q = Q.dimension(1);
    const int H = Q.dimension(2);
    const int S_kv = K.dimension(1);
    const int D_qk = Q.dimension(3);
    const int D_v = V.dimension(3);

    const float neg_inf = -1e9f;
    const float zero = 0.0f;
    const float one = 1.0f;
    float dp_inverse_scale = one;
    if (dropout_rate > 0.0f) {
      dp_inverse_scale = one / (one - dropout_rate);
    }

    // Shards over (batch, head, query) rows, row = (b * H + h) * S_q + i;
    // rows are independent. The dropout mask is indexed by bh = b * H + h,
    // matching the GPU kernels.
    auto shard = [&](int64_t start, int64_t end) {
      std::vector<float> O_accum(D_v);
      for (int64_t row = start; row < end; ++row) {
        const int64_t bh = row / S_q;
        const int i = static_cast<int>(row - bh * S_q);
        const int64_t b = bh / H;
        const int64_t h = bh % H;
        {
          float m_i = neg_inf;
          float l_i = zero;

          std::fill(O_accum.begin(), O_accum.end(), zero);

          // --- Loop Keys (j) ---
          for (int j = 0; j < S_kv; ++j) {
            if (causal_mask && (j > i)) {
              continue;
            }

            float score = zero;
            for (int k = 0; k < D_qk; ++k) {
              score += static_cast<float>(Q(b, i, h, k)) *
                       static_cast<float>(K(b, j, h, k));
            }
            score *= scale;

            // Online Softmax
            float m_prev = m_i;
            m_i = (score > m_prev) ? score : m_prev;

            float ratio_prev = exp(m_prev - m_i);
            float ratio_curr_l = exp(score - m_i);
            float ratio_curr_o = ratio_curr_l;

            // Dropout
            if (dropout_rate > 0.0f) {
              if (get_random_float_cpu(seed, offset, bh, i, j, S_q, S_kv) <
                  dropout_rate) {
                ratio_curr_o = zero;
              } else {
                ratio_curr_o *= dp_inverse_scale;
              }
            }

            l_i = l_i * ratio_prev + ratio_curr_l;

            // update Accumulator (O_accum)

            for (int k = 0; k < D_v; ++k) {
              O_accum[k] *= ratio_prev;

              float v_val = static_cast<float>(V(b, j, h, k));
              O_accum[k] += v_val * ratio_curr_o;
            }

          }  // End Loop j

          // Write Output
          // Exact LSE, as in the GPU kernels: l_i >= 1 once any key is
          // attended. A row without keys (S_kv == 0) gets -inf, the
          // log-sum-exp of the empty set.
          const bool has_keys = l_i > zero;
          stats(b, h, i) = has_keys ? m_i + Eigen::numext::log(l_i)
                                    : -std::numeric_limits<float>::infinity();

          float inv_l = has_keys ? (one / l_i) : zero;

          for (int k = 0; k < D_v; ++k) {
            Out(b, i, h, k) = static_cast<T>(O_accum[k] * inv_l);
          }
        }
      }  // End Loop row
    };

    // Cost of one query row.
    double compute_cycles =
        static_cast<double>(S_kv) * (2.0 * D_qk + 2.0 * D_v + 10.0);
    Eigen::TensorOpCost cost(0, 0, compute_cycles);
    d.parallelFor(static_cast<int64_t>(B) * H * S_q, cost, shard);
    return absl::OkStatus();
  }
};

template <typename T>
struct FlashAttentionGradFunctor<CPUDevice, T> {
  static int64_t WorkspaceBytes(int B, int H, int S_q, int S_kv, int D_qk,
                                int D_v, bool deterministic) {
    return 0;
  }

  static absl::Status CheckFeatureSize(int D_qk, int D_v, float dropout_rate) {
    return std::max(D_qk, D_v) <= 128 ? absl::OkStatus()
                                      : FeatureSizeError(D_qk, D_v);
  }

  absl::Status operator()(
      const CPUDevice &d, typename TTypes<T, 4>::ConstTensor Q,
      typename TTypes<T, 4>::ConstTensor K,
      typename TTypes<T, 4>::ConstTensor V,
      typename TTypes<T, 4>::ConstTensor Out,
      typename TTypes<float, 3>::ConstTensor Stats,
      typename TTypes<T, 4>::ConstTensor dO, typename TTypes<T, 4>::Tensor dQ,
      typename TTypes<T, 4>::Tensor dK, typename TTypes<T, 4>::Tensor dV,
      bool causal_mask, float dropout_rate, float scale, FlashAttentionRng rng,
      bool /*deterministic*/, void * /*workspace*/) const {
    // Always deterministic: every output element is summed by one thread in a
    // fixed order.
    const uint64_t seed = rng.seed;
    const uint64_t offset = rng.offset;
    const int B = Q.dimension(0);
    const int S_q = Q.dimension(1);
    const int H = Q.dimension(2);
    const int S_kv = K.dimension(1);
    const int D_k = Q.dimension(3);
    const int D_v = V.dimension(3);

    const float zero = 0.0f;
    const float one = 1.0f;

    float dp_inverse_scale = one;
    if (dropout_rate > 0.0f) {
      dp_inverse_scale = one / (one - dropout_rate);
    }

    // D_i = dO_i . Out_i, the softmax-backward row term.
    auto row_sum = [&](int64_t b, int64_t h, int i) {
      float D_i = zero;
      for (int k = 0; k < D_v; ++k) {
        D_i += static_cast<float>(dO(b, i, h, k)) *
               static_cast<float>(Out(b, i, h, k));
      }
      return D_i;
    };

    // The dropped-and-rescaled attention weight P_ij (returned in *prob) and
    // dS_ij for one (query i, key j) pair of the (b, h) sequence, bh = b * H +
    // h. Callers skip pairs masked by causal_mask.
    auto pair = [&](int64_t b, int64_t h, int64_t bh, int i, int j, float D_i,
                    float stats_val, float *prob) {
      float score = zero;
      for (int k = 0; k < D_k; ++k) {
        score += static_cast<float>(Q(b, i, h, k)) *
                 static_cast<float>(K(b, j, h, k));
      }
      score *= scale;
      const float prob_raw = std::exp(score - stats_val);
      if (dropout_rate > 0.0f &&
          get_random_float_cpu(seed, offset, bh, i, j, S_q, S_kv) <
              dropout_rate) {
        *prob = zero;
        return -prob_raw * D_i * scale;
      }
      *prob = prob_raw * dp_inverse_scale;
      // dP_ij = dO_i . V_j; dS = P_scaled * (dP - D / sigma) * scale
      float dP_val = zero;
      for (int k = 0; k < D_v; ++k) {
        dP_val += static_cast<float>(dO(b, i, h, k)) *
                  static_cast<float>(V(b, j, h, k));
      }
      return *prob * (dP_val - D_i / dp_inverse_scale) * scale;
    };

    const int64_t BH = static_cast<int64_t>(B) * H;
    // dK and dV sum over all queries, so accumulate them in float and round
    // to T once at the end; T = half/bfloat16 would lose most of the sum.
    if (BH >= d.numThreads()) {
      // Enough (batch, head) pairs to keep every thread busy: each shard owns
      // whole pairs, so it can compute dQ, dK and dV in one pass with private
      // dK/dV buffers.
      auto shard = [&](int64_t start, int64_t end) {
        std::vector<float> dQ_accum(D_k);
        std::vector<float> dK_accum(static_cast<size_t>(S_kv) * D_k);
        std::vector<float> dV_accum(static_cast<size_t>(S_kv) * D_v);
        for (int64_t bh = start; bh < end; ++bh) {
          const int64_t b = bh / H;
          const int64_t h = bh % H;
          std::fill(dK_accum.begin(), dK_accum.end(), zero);
          std::fill(dV_accum.begin(), dV_accum.end(), zero);
          for (int i = 0; i < S_q; ++i) {
            const float stats_val = Stats(b, h, i);
            const float D_i = row_sum(b, h, i);
            std::fill(dQ_accum.begin(), dQ_accum.end(), zero);
            for (int j = 0; j < S_kv; ++j) {
              if (causal_mask && (j > i)) continue;
              float prob;
              const float dS_val = pair(b, h, bh, i, j, D_i, stats_val, &prob);
              float *dV_j = &dV_accum[static_cast<size_t>(j) * D_v];
              for (int k = 0; k < D_v; ++k) {
                dV_j[k] += prob * static_cast<float>(dO(b, i, h, k));
              }
              float *dK_j = &dK_accum[static_cast<size_t>(j) * D_k];
              for (int k = 0; k < D_k; ++k) {
                dK_j[k] += dS_val * static_cast<float>(Q(b, i, h, k));
                dQ_accum[k] += dS_val * static_cast<float>(K(b, j, h, k));
              }
            }
            for (int k = 0; k < D_k; ++k) {
              dQ(b, i, h, k) = static_cast<T>(dQ_accum[k]);
            }
          }
          for (int j = 0; j < S_kv; ++j) {
            for (int k = 0; k < D_k; ++k) {
              dK(b, j, h, k) =
                  static_cast<T>(dK_accum[static_cast<size_t>(j) * D_k + k]);
            }
            for (int k = 0; k < D_v; ++k) {
              dV(b, j, h, k) =
                  static_cast<T>(dV_accum[static_cast<size_t>(j) * D_v + k]);
            }
          }
        }
      };
      const double compute_cycles =
          static_cast<double>(S_q) * S_kv * (4.0 * D_k + 4.0 * D_v + 30.0);
      d.parallelFor(BH, Eigen::TensorOpCost(0, 0, compute_cycles), shard);
      return absl::OkStatus();
    }

    // Too few (batch, head) pairs for the thread pool: parallelize over
    // sequence positions too. Shards writing dQ rows and shards writing dK/dV
    // rows must not share accumulators, so this takes two passes over the
    // pairs (recomputing P and dS), one over (bh, query) rows and one over
    // (bh, key) rows. Both are deterministic.
    std::vector<float> row_sums(static_cast<size_t>(BH) * S_q);
    auto q_shard = [&](int64_t start, int64_t end) {
      std::vector<float> dQ_accum(D_k);
      for (int64_t row = start; row < end; ++row) {
        const int64_t bh = row / S_q;
        const int i = static_cast<int>(row - bh * S_q);
        const int64_t b = bh / H;
        const int64_t h = bh % H;
        const float stats_val = Stats(b, h, i);
        const float D_i = row_sum(b, h, i);
        row_sums[row] = D_i;
        std::fill(dQ_accum.begin(), dQ_accum.end(), zero);
        for (int j = 0; j < S_kv; ++j) {
          if (causal_mask && (j > i)) continue;
          float prob;
          const float dS_val = pair(b, h, bh, i, j, D_i, stats_val, &prob);
          for (int k = 0; k < D_k; ++k) {
            dQ_accum[k] += dS_val * static_cast<float>(K(b, j, h, k));
          }
        }
        for (int k = 0; k < D_k; ++k) {
          dQ(b, i, h, k) = static_cast<T>(dQ_accum[k]);
        }
      }
    };
    auto kv_shard = [&](int64_t start, int64_t end) {
      std::vector<float> dK_accum(D_k);
      std::vector<float> dV_accum(D_v);
      for (int64_t row = start; row < end; ++row) {
        const int64_t bh = row / S_kv;
        const int j = static_cast<int>(row - bh * S_kv);
        const int64_t b = bh / H;
        const int64_t h = bh % H;
        std::fill(dK_accum.begin(), dK_accum.end(), zero);
        std::fill(dV_accum.begin(), dV_accum.end(), zero);
        for (int i = causal_mask ? j : 0; i < S_q; ++i) {
          float prob;
          const float dS_val = pair(b, h, bh, i, j, row_sums[bh * S_q + i],
                                    Stats(b, h, i), &prob);
          for (int k = 0; k < D_v; ++k) {
            dV_accum[k] += prob * static_cast<float>(dO(b, i, h, k));
          }
          for (int k = 0; k < D_k; ++k) {
            dK_accum[k] += dS_val * static_cast<float>(Q(b, i, h, k));
          }
        }
        for (int k = 0; k < D_k; ++k) {
          dK(b, j, h, k) = static_cast<T>(dK_accum[k]);
        }
        for (int k = 0; k < D_v; ++k) {
          dV(b, j, h, k) = static_cast<T>(dV_accum[k]);
        }
      }
    };
    // Cost of one dQ row (over all keys) and one dK/dV row (over all queries).
    d.parallelFor(
        BH * S_q,
        Eigen::TensorOpCost(
            0, 0, static_cast<double>(S_kv) * (3.0 * D_k + 2.0 * D_v + 25.0)),
        q_shard);
    d.parallelFor(
        BH * S_kv,
        Eigen::TensorOpCost(
            0, 0, static_cast<double>(S_q) * (3.0 * D_k + 3.0 * D_v + 25.0)),
        kv_shard);
    return absl::OkStatus();
  }
};
}  // namespace functor

#define REGISTER_CPU_KERNEL_F(type)                                            \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("FlashAttention").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      FlashAttentionOp<CPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("FlashAttentionGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      FlashAttentionGradOp<CPUDevice, type>);

REGISTER_CPU_KERNEL_F(float);
REGISTER_CPU_KERNEL_F(Eigen::half);
REGISTER_CPU_KERNEL_F(Eigen::bfloat16);
#undef REGISTER_CPU_KERNEL_F

#if GOOGLE_CUDA
#define REGISTER_GPU_KERNEL_F(type)                           \
  REGISTER_KERNEL_BUILDER(Name("FlashAttention")              \
                              .Device(DEVICE_GPU)             \
                              .TypeConstraint<type>("T")      \
                              .HostMemory("seed")             \
                              .HostMemory("offset"),          \
                          FlashAttentionOp<GPUDevice, type>); \
  REGISTER_KERNEL_BUILDER(Name("FlashAttentionGrad")          \
                              .Device(DEVICE_GPU)             \
                              .TypeConstraint<type>("T")      \
                              .HostMemory("seed")             \
                              .HostMemory("offset"),          \
                          FlashAttentionGradOp<GPUDevice, type>);

REGISTER_GPU_KERNEL_F(float);
REGISTER_GPU_KERNEL_F(Eigen::half);
REGISTER_GPU_KERNEL_F(Eigen::bfloat16);
#undef REGISTER_GPU_KERNEL_F
#endif

}  // namespace tensorflow