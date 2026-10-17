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

// XLA custom-call targets for FlashAttention / FlashAttentionGrad (lowered in
// flash_attention_xla_op.cc).
// They wrap XLA's stream in an Eigen device and call the same functors as the
// TF kernels.
#define EIGEN_USE_GPU
#include "tensorflow/core/kernels/flash_attention/flash_attention_custom_call.h"

#include <cuda_runtime.h>

#include <cstring>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/flash_attention/flash_attention_op.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_target_registry.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

template <typename T>
absl::Status RunFlashAttn(cudaStream_t stream, void** buffers,
                          const FlashAttnDesc& d) {
  Eigen::GpuStreamDevice stream_device(&stream,
                                       /*device=*/-1);  // -1 = current device
  GPUDevice device(&stream_device);

  // buffers = [Q, K, V, seed, offset, Out, stats]  (operands, then tuple
  // outputs flattened)
  typename TTypes<T, 4>::ConstTensor Q(static_cast<const T*>(buffers[0]), d.B,
                                       d.Sq, d.H, d.Dqk);
  typename TTypes<T, 4>::ConstTensor K(static_cast<const T*>(buffers[1]), d.B,
                                       d.Sk, d.H, d.Dqk);
  typename TTypes<T, 4>::ConstTensor V(static_cast<const T*>(buffers[2]), d.B,
                                       d.Sk, d.H, d.Dv);
  functor::FlashAttentionRng rng;
  rng.seed_dev = static_cast<const uint64_t*>(buffers[3]);
  rng.offset_dev = static_cast<const uint64_t*>(buffers[4]);
  typename TTypes<T, 4>::Tensor Out(static_cast<T*>(buffers[5]), d.B, d.Sq, d.H,
                                    d.Dv);
  typename TTypes<float, 3>::Tensor stats(static_cast<float*>(buffers[6]), d.B,
                                          d.H, d.Sq);

  return functor::FlashAttentionFunctor<GPUDevice, T>()(
      device, Q, K, V, Out, stats, d.causal, d.dropout, d.scale, rng);
}

template <typename T>
absl::Status RunFlashAttnGrad(cudaStream_t stream, void** buffers,
                              const FlashAttnBwdDesc& d) {
  Eigen::GpuStreamDevice stream_device(&stream,
                                       /*device=*/-1);  // -1 = current device
  GPUDevice device(&stream_device);

  // buffers = [Q, K, V, Out, stats, dO, seed, offset, dQ, dK, dV, workspace]
  typename TTypes<T, 4>::ConstTensor Q(static_cast<const T*>(buffers[0]), d.B,
                                       d.Sq, d.H, d.Dqk);
  typename TTypes<T, 4>::ConstTensor K(static_cast<const T*>(buffers[1]), d.B,
                                       d.Sk, d.H, d.Dqk);
  typename TTypes<T, 4>::ConstTensor V(static_cast<const T*>(buffers[2]), d.B,
                                       d.Sk, d.H, d.Dv);
  typename TTypes<T, 4>::ConstTensor Out(static_cast<const T*>(buffers[3]), d.B,
                                         d.Sq, d.H, d.Dv);
  typename TTypes<float, 3>::ConstTensor stats(
      static_cast<const float*>(buffers[4]), d.B, d.H, d.Sq);
  typename TTypes<T, 4>::ConstTensor dO(static_cast<const T*>(buffers[5]), d.B,
                                        d.Sq, d.H, d.Dv);
  functor::FlashAttentionRng rng;
  rng.seed_dev = static_cast<const uint64_t*>(buffers[6]);
  rng.offset_dev = static_cast<const uint64_t*>(buffers[7]);
  typename TTypes<T, 4>::Tensor dQ(static_cast<T*>(buffers[8]), d.B, d.Sq, d.H,
                                   d.Dqk);
  typename TTypes<T, 4>::Tensor dK(static_cast<T*>(buffers[9]), d.B, d.Sk, d.H,
                                   d.Dqk);
  typename TTypes<T, 4>::Tensor dV(static_cast<T*>(buffers[10]), d.B, d.Sk, d.H,
                                   d.Dv);
  void* workspace = d.workspace_bytes > 0 ? buffers[11] : nullptr;

  return functor::FlashAttentionGradFunctor<GPUDevice, T>()(
      device, Q, K, V, Out, stats, dO, dQ, dK, dV, d.causal, d.dropout, d.scale,
      rng, d.deterministic, workspace);
}

// Decodes the descriptor, runs `Runner` for its dtype and reports failures
// through `status`.
template <typename Desc, template <typename> class Runner>
void RunCustomCall(cudaStream_t stream, void** buffers, const char* opaque,
                   size_t opaque_len, XlaCustomCallStatus* status) {
  absl::Status result;
  Desc d;
  if (opaque_len != sizeof(Desc)) {
    result = absl::InternalError(
        absl::StrCat("FlashAttention: bad descriptor size ", opaque_len));
  } else {
    std::memcpy(&d, opaque, sizeof(d));
    switch (d.dtype) {
      case kFlashAttnF16:
        result = Runner<Eigen::half>()(stream, buffers, d);
        break;
      case kFlashAttnBF16:
        result = Runner<Eigen::bfloat16>()(stream, buffers, d);
        break;
      case kFlashAttnF32:
        result = Runner<float>()(stream, buffers, d);
        break;
      default:
        result = absl::InternalError(
            absl::StrCat("FlashAttention: unsupported dtype ", d.dtype));
    }
  }
  if (result.ok()) {
    if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
      result = absl::InternalError(absl::StrCat(
          "FlashAttention: kernel launch failed: ", cudaGetErrorString(err)));
    }
  }
  if (!result.ok()) {
    XlaCustomCallStatusSetFailure(status, result.message().data(),
                                  result.message().size());
  }
}

template <typename T>
struct FwdRunner {
  absl::Status operator()(cudaStream_t stream, void** buffers,
                          const FlashAttnDesc& d) const {
    return RunFlashAttn<T>(stream, buffers, d);
  }
};

template <typename T>
struct BwdRunner {
  absl::Status operator()(cudaStream_t stream, void** buffers,
                          const FlashAttnBwdDesc& d) const {
    return RunFlashAttnGrad<T>(stream, buffers, d);
  }
};

void FlashAttnFwdCustomCall(cudaStream_t stream, void** buffers,
                            const char* opaque, size_t opaque_len,
                            XlaCustomCallStatus* status) {
  RunCustomCall<FlashAttnDesc, FwdRunner>(stream, buffers, opaque, opaque_len,
                                          status);
}

void FlashAttnBwdCustomCall(cudaStream_t stream, void** buffers,
                            const char* opaque, size_t opaque_len,
                            XlaCustomCallStatus* status) {
  RunCustomCall<FlashAttnBwdDesc, BwdRunner>(stream, buffers, opaque,
                                             opaque_len, status);
}

XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("tf_flash_attention_fwd",
                                         FlashAttnFwdCustomCall, "CUDA");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("tf_flash_attention_bwd",
                                         FlashAttnBwdCustomCall, "CUDA");

}  // namespace tensorflow
