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

#ifndef TENSORFLOW_CORE_KERNELS_FLASH_ATTENTION_FLASH_ATTENTION_OP_H_
#define TENSORFLOW_CORE_KERNELS_FLASH_ATTENTION_FLASH_ATTENTION_OP_H_

#include <cstdint>
#include <limits>
#include <utility>

#include "tensorflow/core/framework/op_kernel.h"
#include "unsupported/Eigen/CXX11/Tensor"

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace tensorflow::functor {
// Dropout RNG seed and offset. Normally given by value; the XLA path only has
// them in device memory, in which case seed_dev/offset_dev point there and the
// GPU kernels read them on the device (seed/offset are then ignored).
struct FlashAttentionRng {
  uint64_t seed = 0;
  uint64_t offset = 0;
  const uint64_t *seed_dev = nullptr;
  const uint64_t *offset_dev = nullptr;
};

// Error for feature sizes the kernels can't handle. Up to 128 always works on
// GPU; up to 256 only on the FlashAttention-2 path (see fa2/fa2_api.h).
inline absl::Status FeatureSizeError(int D_qk, int D_v) {
  return errors::InvalidArgument(
      "Feature size D_qk=", D_qk, ", D_v=", D_v,
      " is not supported. The GPU kernels support up to 128; up to 256 "
      "requires "
      "float16/bfloat16 on an sm80+ GPU with D_qk == D_v and a multiple of 8 "
      "(and, for "
      "D > 192 with dropout, >= 144 KB shared memory per block, e.g. "
      "A100/H100). Set smaller "
      "feature size or use more heads");
}

// The functors and the FlashAttention-2 API take 32-bit dimensions, so larger
// ones would silently truncate. The kernels also index (batch, head) pairs
// with a single int.
inline absl::Status CheckDimsFitInt32(int64_t B, int64_t H, int64_t S_q,
                                      int64_t S_kv, int64_t D_qk, int64_t D_v) {
  constexpr int64_t kMax = std::numeric_limits<int32_t>::max();
  // B * H, saturated so it can't overflow int64.
  const int64_t BH = (H != 0 && B > kMax / H) ? kMax + 1 : B * H;
  const std::pair<const char *, int64_t> dims[] = {
      {"batch", B},   {"heads", H},   {"batch * heads", BH}, {"S_q", S_q},
      {"S_kv", S_kv}, {"D_qk", D_qk}, {"D_v", D_v}};
  for (const auto &[name, size] : dims) {
    if (size > kMax) {
      return errors::InvalidArgument("FlashAttention dimension ", name, "=",
                                     size, " exceeds the int32 limit (", kMax,
                                     ")");
    }
  }
  return absl::OkStatus();
}

template <typename Device, typename T>
struct FlashAttentionFunctor {
  // OK if the kernels for this device and dtype handle these feature sizes.
  static absl::Status CheckFeatureSize(int D_qk, int D_v);

  // Q/K/V/Out are [batch, seq, heads, dim]; stats is [batch, heads, seq_q].
  absl::Status operator()(const Device &d, typename TTypes<T, 4>::ConstTensor Q,
                          typename TTypes<T, 4>::ConstTensor K,
                          typename TTypes<T, 4>::ConstTensor V,
                          typename TTypes<T, 4>::Tensor Out,
                          typename TTypes<float, 3>::Tensor stats,
                          bool causal_mask, float dropout_rate, float scale,
                          FlashAttentionRng rng) const;
};

// With `deterministic` (normally OpDeterminismRequired()), the gradients are
// bitwise reproducible; otherwise dQ may use atomic adds, whose order varies.
// The caller decides once and passes the same value to WorkspaceBytes and
// operator(), since the workspace size depends on it.
template <typename Device, typename T>
struct FlashAttentionGradFunctor {
  // Bytes of device scratch memory operator() needs as `workspace` (may be 0),
  // for the same `deterministic`. Doesn't depend on which device runs it.
  static int64_t WorkspaceBytes(int B, int H, int S_q, int S_kv, int D_qk,
                                int D_v, bool deterministic);

  // OK if the kernels for this device and dtype handle these feature sizes.
  static absl::Status CheckFeatureSize(int D_qk, int D_v, float dropout_rate);

  absl::Status operator()(const Device &d, typename TTypes<T, 4>::ConstTensor Q,
                          typename TTypes<T, 4>::ConstTensor K,
                          typename TTypes<T, 4>::ConstTensor V,
                          typename TTypes<T, 4>::ConstTensor Out,
                          typename TTypes<float, 3>::ConstTensor Stats,
                          typename TTypes<T, 4>::ConstTensor dO,
                          typename TTypes<T, 4>::Tensor dQ,
                          typename TTypes<T, 4>::Tensor dK,
                          typename TTypes<T, 4>::Tensor dV, bool causal_mask,
                          float dropout_rate, float scale,
                          FlashAttentionRng rng, bool deterministic,
                          void *workspace) const;
};

};  // namespace tensorflow::functor

#endif  // TENSORFLOW_CORE_KERNELS_FLASH_ATTENTION_FLASH_ATTENTION_OP_H_
