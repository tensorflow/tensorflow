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

template <typename Device, typename T>
struct FlashAttentionFunctor {
  // OK if the kernels for this device and dtype handle these feature sizes.
  static absl::Status CheckFeatureSize(int D_qk, int D_v);

  absl::Status operator()(const Device &d, typename TTypes<T, 3>::ConstTensor Q,
                          typename TTypes<T, 3>::ConstTensor K,
                          typename TTypes<T, 3>::ConstTensor V,
                          typename TTypes<T, 3>::Tensor Out,
                          typename TTypes<float, 2>::Tensor stats,
                          bool causal_mask, float dropout_rate, float scale,
                          FlashAttentionRng rng) const;
};

template <typename Device, typename T>
struct FlashAttentionGradFunctor {
  // Bytes of device scratch memory operator() needs as `workspace` (may be 0).
  // Depends only on shapes, never on the current device.
  static int64_t WorkspaceBytes(int B, int S_q, int S_kv, int D_qk, int D_v);

  // OK if the kernels for this device and dtype handle these feature sizes.
  static absl::Status CheckFeatureSize(int D_qk, int D_v, float dropout_rate);

  absl::Status operator()(const Device &d, typename TTypes<T, 3>::ConstTensor Q,
                          typename TTypes<T, 3>::ConstTensor K,
                          typename TTypes<T, 3>::ConstTensor V,
                          typename TTypes<T, 3>::ConstTensor Out,
                          typename TTypes<float, 2>::ConstTensor Stats,
                          typename TTypes<T, 3>::ConstTensor dO,
                          typename TTypes<T, 3>::Tensor dQ,
                          typename TTypes<T, 3>::Tensor dK,
                          typename TTypes<T, 3>::Tensor dV, bool causal_mask,
                          float dropout_rate, float scale,
                          FlashAttentionRng rng, void *workspace) const;
};

};  // namespace tensorflow::functor

#endif  // TENSORFLOW_CORE_KERNELS_FLASH_ATTENTION_FLASH_ATTENTION_OP_H_
