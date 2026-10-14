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
#include <type_traits>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

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
      OP_REQUIRES(context, t->dims() == 3,
                  errors::InvalidArgument(
                      "query, key, and value must be rank 3, got shape ",
                      t->shape().DebugString()));
    }

    const int64_t B = q_tensor.dim_size(0);
    const int64_t S_q = q_tensor.dim_size(1);
    const int64_t D_qk = q_tensor.dim_size(2);
    const int64_t S_kv = k_tensor.dim_size(1);
    const int64_t D_v = v_tensor.dim_size(2);
    OP_REQUIRES_OK(context,
                   functor::CheckDimsFitInt32(B, S_q, S_kv, D_qk, D_v));

    OP_REQUIRES(context, k_tensor.dim_size(0) == B,
                errors::InvalidArgument("K batch size mismatch"));
    OP_REQUIRES(context, k_tensor.dim_size(2) == D_qk,
                errors::InvalidArgument("K feature size mismatch"));
    OP_REQUIRES(context, v_tensor.dim_size(0) == B,
                errors::InvalidArgument("V batch size mismatch"));
    OP_REQUIRES(context, v_tensor.dim_size(1) == S_kv,
                errors::InvalidArgument("V seq size mismatch"));

    if (causal_mask_) {
      OP_REQUIRES(
          context, S_q == S_kv,
          errors::InvalidArgument("seq size Q != K while is_causal = True"));
    }

    OP_REQUIRES_OK(
        context,
        functor::FlashAttentionFunctor<Device, T>::CheckFeatureSize(D_qk, D_v));

    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({B, S_q, D_v}), &output_tensor));

    Tensor *stats_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({B, S_q}),
                                                     &stats_tensor));

    auto Q = q_tensor.template tensor<T, 3>();
    auto K = k_tensor.template tensor<T, 3>();
    auto V = v_tensor.template tensor<T, 3>();
    auto Out = output_tensor->template tensor<T, 3>();
    auto Stats = stats_tensor->template tensor<float, 2>();

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

    for (const Tensor *t :
         {&q_tensor, &k_tensor, &v_tensor, &out_tensor, &do_tensor}) {
      OP_REQUIRES(context, t->dims() == 3,
                  errors::InvalidArgument(
                      "query, key, value, output and output gradient "
                      "must be rank 3, got shape ",
                      t->shape().DebugString()));
    }
    OP_REQUIRES(
        context, stats_tensor.dims() == 2,
        errors::InvalidArgument("softmax_lse must be rank 2, got shape ",
                                stats_tensor.shape().DebugString()));

    const int64_t B = q_tensor.dim_size(0);
    const int64_t S_q = q_tensor.dim_size(1);
    const int64_t D_qk = q_tensor.dim_size(2);
    const int64_t S_kv = k_tensor.dim_size(1);
    const int64_t D_v = v_tensor.dim_size(2);
    OP_REQUIRES_OK(context,
                   functor::CheckDimsFitInt32(B, S_q, S_kv, D_qk, D_v));

    OP_REQUIRES(context, k_tensor.dim_size(0) == B,
                errors::InvalidArgument("K batch size mismatch"));
    OP_REQUIRES(context, k_tensor.dim_size(2) == D_qk,
                errors::InvalidArgument("K feature size mismatch"));

    OP_REQUIRES(context, v_tensor.dim_size(0) == B,
                errors::InvalidArgument("V batch size mismatch"));
    OP_REQUIRES(context, v_tensor.dim_size(1) == S_kv,
                errors::InvalidArgument("V seq size mismatch"));

    OP_REQUIRES(context, out_tensor.dim_size(0) == B,
                errors::InvalidArgument("O batch size mismatch"));
    OP_REQUIRES(context, out_tensor.dim_size(1) == S_q,
                errors::InvalidArgument("O seq size mismatch"));
    OP_REQUIRES(context, out_tensor.dim_size(2) == D_v,
                errors::InvalidArgument("O feature size mismatch"));

    OP_REQUIRES(context, do_tensor.dim_size(0) == B,
                errors::InvalidArgument("dO batch size mismatch"));
    OP_REQUIRES(context, do_tensor.dim_size(1) == S_q,
                errors::InvalidArgument("dO seq size mismatch"));
    OP_REQUIRES(context, do_tensor.dim_size(2) == D_v,
                errors::InvalidArgument("dO feature size mismatch"));

    OP_REQUIRES(context, stats_tensor.dim_size(0) == B,
                errors::InvalidArgument("S batch size mismatch"));
    OP_REQUIRES(context, stats_tensor.dim_size(1) == S_q,
                errors::InvalidArgument("S seq size mismatch"));

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

    auto Q = q_tensor.template tensor<T, 3>();
    auto K = k_tensor.template tensor<T, 3>();
    auto V = v_tensor.template tensor<T, 3>();
    auto Out = out_tensor.template tensor<T, 3>();
    auto dO = do_tensor.template tensor<T, 3>();
    auto Stats = stats_tensor.template tensor<float, 2>();

    auto dQ = dq_tensor->template tensor<T, 3>();
    auto dK = dk_tensor->template tensor<T, 3>();
    auto dV = dv_tensor->template tensor<T, 3>();

    Tensor workspace;
    void *workspace_ptr = nullptr;
    const int64_t workspace_bytes =
        functor::FlashAttentionGradFunctor<Device, T>::WorkspaceBytes(
            B, S_q, S_kv, D_qk, D_v);
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
            workspace_ptr));
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
  return static_cast<float>(val) / 4294967296.0f;
}

template <typename T>
struct FlashAttentionFunctor<CPUDevice, T> {
  static absl::Status CheckFeatureSize(int D_qk, int D_v) {
    return std::max(D_qk, D_v) <= 128 ? absl::OkStatus()
                                      : FeatureSizeError(D_qk, D_v);
  }

  absl::Status operator()(const CPUDevice &d,
                          typename TTypes<T, 3>::ConstTensor Q,
                          typename TTypes<T, 3>::ConstTensor K,
                          typename TTypes<T, 3>::ConstTensor V,
                          typename TTypes<T, 3>::Tensor Out,
                          typename TTypes<float, 2>::Tensor stats,
                          bool causal_mask, float dropout_rate, float scale,
                          FlashAttentionRng rng) const {
    const uint64_t seed = rng.seed;
    const uint64_t offset = rng.offset;
    const int B = Q.dimension(0);
    const int S_q = Q.dimension(1);
    const int S_kv = K.dimension(1);
    const int D_qk = Q.dimension(2);
    const int D_v = V.dimension(2);

    const float neg_inf = -1e9f;
    const float zero = 0.0f;
    const float one = 1.0f;
    const float eps = 1e-6f;
    float dp_inverse_scale = one;
    if (dropout_rate > 0.0f) {
      dp_inverse_scale = one / (one - dropout_rate);
    }

    auto shard = [&](int64_t start, int64_t end) {
      std::vector<float> O_accum(D_v);
      for (int64_t b = start; b < end; ++b) {
        // --- Loop Query (i) ---
        for (int i = 0; i < S_q; ++i) {
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
              score += static_cast<float>(Q(b, i, k) * K(b, j, k));
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
              if (get_random_float_cpu(seed, offset, b, i, j, S_q, S_kv) <
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

              float v_val = static_cast<float>(V(b, j, k));
              O_accum[k] += v_val * ratio_curr_o;
            }

          }  // End Loop j

          // Write Output
          stats(b, i) = m_i + Eigen::numext::log(l_i + eps);

          float inv_l = (l_i > eps) ? (one / l_i) : zero;

          for (int k = 0; k < D_v; ++k) {
            Out(b, i, k) = static_cast<T>(O_accum[k] * inv_l);
          }

        }  // End Loop i
      }  // End Loop b
    };

    // estimate cost
    double compute_cycles =
        static_cast<double>(S_q) * S_kv * (2.0 * D_qk + 2.0 * D_v + 10.0);
    Eigen::TensorOpCost cost(0, 0, compute_cycles);
    d.parallelFor(B, cost, shard);
    return absl::OkStatus();
  }
};

template <typename T>
struct FlashAttentionGradFunctor<CPUDevice, T> {
  static int64_t WorkspaceBytes(int B, int S_q, int S_kv, int D_qk, int D_v) {
    return 0;
  }

  static absl::Status CheckFeatureSize(int D_qk, int D_v, float dropout_rate) {
    return std::max(D_qk, D_v) <= 128 ? absl::OkStatus()
                                      : FeatureSizeError(D_qk, D_v);
  }

  absl::Status operator()(const CPUDevice &d,
                          typename TTypes<T, 3>::ConstTensor Q,
                          typename TTypes<T, 3>::ConstTensor K,
                          typename TTypes<T, 3>::ConstTensor V,
                          typename TTypes<T, 3>::ConstTensor Out,
                          typename TTypes<float, 2>::ConstTensor Stats,
                          typename TTypes<T, 3>::ConstTensor dO,
                          typename TTypes<T, 3>::Tensor dQ,
                          typename TTypes<T, 3>::Tensor dK,
                          typename TTypes<T, 3>::Tensor dV, bool causal_mask,
                          float dropout_rate, float scale,
                          FlashAttentionRng rng, void * /*workspace*/) const {
    const uint64_t seed = rng.seed;
    const uint64_t offset = rng.offset;
    const int B = Q.dimension(0);
    const int S_q = Q.dimension(1);
    const int S_kv = K.dimension(1);
    const int D_k = Q.dimension(2);
    const int D_v = V.dimension(2);

    const float zero = 0.0f;
    const float one = 1.0f;
    const float eps = 1e-6f;

    float dp_inverse_scale = one;
    if (dropout_rate > 0.0f) {
      dp_inverse_scale = one / (one - dropout_rate);
    }

    auto shard = [&](int64_t start, int64_t end) {
      // dK and dV sum over all queries, so accumulate them in float and round
      // to T once at the end; T = half/bfloat16 would lose most of the sum.
      // Shards own whole batch entries, so these buffers are private.
      std::vector<float> dQ_accum(D_k);
      std::vector<float> dK_accum(static_cast<size_t>(S_kv) * D_k);
      std::vector<float> dV_accum(static_cast<size_t>(S_kv) * D_v);
      for (int64_t b = start; b < end; ++b) {
        std::fill(dK_accum.begin(), dK_accum.end(), zero);
        std::fill(dV_accum.begin(), dV_accum.end(), zero);
        for (int i = 0; i < S_q; ++i) {
          float stats_val = Stats(b, i);
          float D_i = zero;
          for (int k = 0; k < D_v; ++k) {
            D_i += static_cast<float>(dO(b, i, k) * Out(b, i, k));
          }

          std::fill(dQ_accum.begin(), dQ_accum.end(), zero);
          for (int j = 0; j < S_kv; ++j) {
            if (causal_mask && (j > i)) {
              continue;
            }

            // (a) Q * K^T
            float score = zero;
            for (int k = 0; k < D_k; ++k) {
              score += static_cast<float>(Q(b, i, k) * K(b, j, k));
            }
            score *= scale;

            // (b) Softmax
            float prob = std::exp(score - stats_val);

            float prob_raw = prob;
            bool is_dropped(false);

            // (c) Dropout
            if (dropout_rate > 0.0f) {
              if (get_random_float_cpu(seed, offset, b, i, j, S_q, S_kv) <
                  dropout_rate) {
                is_dropped = true;
                prob = zero;
              } else {
                prob *= dp_inverse_scale;
              }
            }

            float dS_val = zero;
            if (is_dropped) {
              dS_val = -prob_raw * D_i * scale;
            } else {
              // 1. dV (Accumulate)
              float *dV_j = &dV_accum[static_cast<size_t>(j) * D_v];
              for (int k = 0; k < D_v; ++k) {
                dV_j[k] += prob * static_cast<float>(dO(b, i, k));
              }

              // 2. dP_val = dO_i . V_j
              float dP_val = zero;
              for (int k = 0; k < D_v; ++k) {
                dP_val += static_cast<float>(dO(b, i, k) * V(b, j, k));
              }

              // 3. dS = P_scaled * (dP - D / sigma) * scale
              dS_val = prob * (dP_val - D_i / dp_inverse_scale) * scale;
            }

            //  Update dK & Accumulate dQ (Merged) ---
            float *dK_j = &dK_accum[static_cast<size_t>(j) * D_k];
            for (int k = 0; k < D_k; ++k) {
              float val_Q = static_cast<float>(Q(b, i, k));
              float val_K = static_cast<float>(K(b, j, k));

              dK_j[k] += dS_val * val_Q;
              dQ_accum[k] += dS_val * val_K;
            }

          }  // End Loop j

          for (int k = 0; k < D_k; ++k) {
            dQ(b, i, k) = static_cast<T>(dQ_accum[k]);
          }
        }  // End Loop i

        for (int j = 0; j < S_kv; ++j) {
          for (int k = 0; k < D_k; ++k) {
            dK(b, j, k) =
                static_cast<T>(dK_accum[static_cast<size_t>(j) * D_k + k]);
          }
          for (int k = 0; k < D_v; ++k) {
            dV(b, j, k) =
                static_cast<T>(dV_accum[static_cast<size_t>(j) * D_v + k]);
          }
        }
      }  // End Loop b
    };

    double compute_cycles =
        static_cast<double>(S_q) * S_kv * (4.0 * D_k + 4.0 * D_v + 30.0);
    Eigen::TensorOpCost cost(0, 0, compute_cycles);
    d.parallelFor(B, cost, shard);
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