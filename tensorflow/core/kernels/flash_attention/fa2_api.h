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

#ifndef TENSORFLOW_CORE_KERNELS_FLASH_ATTENTION_FA2_API_H_
#define TENSORFLOW_CORE_KERNELS_FLASH_ATTENTION_FA2_API_H_
// Thin wrapper around the vendored FlashAttention-2 kernels. Kept free of
// TensorFlow and PyTorch headers so it can be compiled with FA2's own flags.
//
// All tensors are contiguous and row-major: q/out are [B, S_q, H, D], k/v are
// [B, S_kv, H, D] (fp16, or bf16 when bf16 = true), lse is float [B, H, S_q],
// matching the FlashAttention op.
//
// Dropout RNG: seed/offset are used by value unless seed_dev/offset_dev are
// non-null, in which case they point to device memory holding the values
// (read by the kernels, so no host sync is needed; used by the XLA path).

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace flash_attention_fa2 {

// True if the current device and shapes can use FlashAttention-2: the library
// was built with FA2 code for this GPU (sm80+), D_qk == D_v, and the head dim
// is a multiple of 8 and <= 256.
bool Supported(int d_qk, int d_v);

// Supported, plus the backward-only restriction: FA2's hdim 256 backward kernel
// needs >= 144 KB of shared memory per block for dropout, which sm86/sm89 lack.
bool BackwardSupported(int d_qk, int d_v, float dropout);

// The shape part of Supported: true if FA2 could handle these head dims on a
// suitable GPU. Doesn't touch the device, so it's safe at XLA compile time.
bool ShapeSupported(int d_qk, int d_v);

// True if FA2 can launch with this batch size and head count: its kernels put
// them in gridDim.y and gridDim.z, which are limited to 65535.
bool GridSupported(int B, int H);

cudaError_t Forward(cudaStream_t stream, bool bf16, const void *q,
                    const void *k, const void *v, void *out, float *lse, int B,
                    int H, int S_q, int S_kv, int D, float scale, float dropout,
                    bool causal, uint64_t seed, uint64_t offset,
                    const uint64_t *seed_dev, const uint64_t *offset_dev);

// Scratch memory Backward needs (dQ accumulator, softmax row sums, RNG state).
// With `deterministic`, FA2 keeps one dQ accumulator per column block, and
// their number depends on the GPU's SM count; this sizes them for the largest
// visible GPU, so the result is valid on any of them.
size_t BackwardWorkspaceBytes(int B, int H, int S_q, int D, bool deterministic);

// With `deterministic`, uses FA2's deterministic mode: no two thread blocks
// accumulate into the same dQ element, so the gradients are bitwise
// reproducible (dK/dV always are).

cudaError_t Backward(cudaStream_t stream, bool bf16, const void *q,
                     const void *k, const void *v, const void *out,
                     const float *lse, const void *dout, void *dq, void *dk,
                     void *dv, void *workspace, int B, int H, int S_q, int S_kv,
                     int D, float scale, float dropout, bool causal,
                     bool deterministic, uint64_t seed, uint64_t offset,
                     const uint64_t *seed_dev, const uint64_t *offset_dev);

}  // namespace flash_attention_fa2

#endif  // TENSORFLOW_CORE_KERNELS_FLASH_ATTENTION_FA2_API_H_
