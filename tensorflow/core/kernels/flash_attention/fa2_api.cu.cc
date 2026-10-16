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

// Parameter setup mirrors mha_fwd / mha_bwd in
// @flash_attention//:flash_api.cpp (FlashAttention v2.8.3), restricted
// to fixed-length batches with as many key/value heads as query heads.

#include "tensorflow/core/kernels/flash_attention/fa2_api.h"

#if __has_include("flash.h")

#include <c10/cuda/CUDAException.h>
#include <cutlass/numeric_types.h>

#include <algorithm>
#include <atomic>
#include <cmath>

#include "flash.h"

namespace flash_attention_fa2 {
namespace {

using namespace flash_attention_fa2_impl;
// Both fp16 and bf16 are 2 bytes; used for zero-filling outputs.
constexpr size_t kElemBytes = 2;
static_assert(sizeof(cutlass::half_t) == kElemBytes &&
              sizeof(cutlass::bfloat16_t) == kElemBytes);

// The forward kernel stores (seed, offset) for the backward pass when dropout
// is on. We pass them to the backward pass ourselves, so this is a write-only
// sink.
__device__ uint64_t g_fwd_rng_sink[2];

__global__ void SetRngState(uint64_t *rng_state, uint64_t seed, uint64_t offset,
                            const uint64_t *seed_dev,
                            const uint64_t *offset_dev) {
  rng_state[0] = seed_dev != nullptr ? *seed_dev : seed;
  rng_state[1] = offset_dev != nullptr ? *offset_dev : offset;
}

__global__ void FillFloats(float *ptr, size_t n, float value) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
       i += size_t(blockDim.x) * gridDim.x) {
    ptr[i] = value;
  }
}

constexpr int RoundUp(int x, int m) { return (x + m - 1) / m * m; }
constexpr size_t AlignUp(size_t x) { return (x + 255) / 256 * 256; }

// True if this library has FA2 code (SASS, or PTX it can JIT) for the current
// device. SetRngState is compiled with the same arch flags as the FA2 kernels,
// so it serves as the probe. This rejects pre-sm80 GPUs as well as GPUs the
// build simply didn't target.
bool DeviceHasKernelImage() {
  constexpr int kMaxDevices = 64;
  static std::atomic<int> cached[kMaxDevices];  // 0 = unknown, 1 = yes, -1 = no
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= kMaxDevices)
    return false;
  int val = cached[dev].load(std::memory_order_relaxed);
  if (val == 0) {
    // Builds that also target pre-sm80 GPUs contain FA2 stubs for them (see
    // ARCH_SUPPORTS_FLASH), so a kernel image alone isn't enough.
    int major = 0;
    cudaFuncAttributes attr;
    const bool ok =
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                               dev) == cudaSuccess &&
        major >= 8 && cudaFuncGetAttributes(&attr, SetRngState) == cudaSuccess;
    if (!ok)
      cudaGetLastError();  // don't leave the error for the next launch check
    val = ok ? 1 : -1;
    cached[dev].store(val, std::memory_order_relaxed);
  }
  return val == 1;
}

// Batch and row (sequence position) strides, in elements, of a [B, S, H, D]
// tensor. The head stride is D.
struct Strides {
  int64_t batch, row;
};
Strides BSHDStrides(int S, int H, int D) {
  return {int64_t(S) * H * D, int64_t(H) * D};
}

void SetFwdParams(Flash_fwd_params &p, bool bf16, const void *q, const void *k,
                  const void *v, void *out, float *lse, int B, int H, int S_q,
                  int S_kv, int D, float scale, float dropout, bool causal,
                  uint64_t seed, uint64_t offset, const uint64_t *seed_dev,
                  const uint64_t *offset_dev) {
  p = {};
  p.is_bf16 = bf16;

  const Strides sq = BSHDStrides(S_q, H, D), skv = BSHDStrides(S_kv, H, D);
  p.q_ptr = const_cast<void *>(q);
  p.k_ptr = const_cast<void *>(k);
  p.v_ptr = const_cast<void *>(v);
  p.o_ptr = out;
  p.q_batch_stride = p.o_batch_stride = sq.batch;
  p.k_batch_stride = p.v_batch_stride = skv.batch;
  p.q_row_stride = p.o_row_stride = sq.row;
  p.k_row_stride = p.v_row_stride = skv.row;
  p.q_head_stride = p.k_head_stride = p.v_head_stride = p.o_head_stride = D;
  // [B, H, S_q], FA2's own layout.
  p.softmax_lse_ptr = lse;

  p.b = B;
  p.h = p.h_k = H;
  p.h_h_k_ratio = 1;
  p.seqlen_q = S_q;
  p.seqlen_k = S_kv;
  p.seqlen_q_rounded = RoundUp(S_q, 128);
  p.seqlen_k_rounded = RoundUp(S_kv, 128);
  p.d = D;
  p.d_rounded = RoundUp(D, 32);

  p.softcap = 0.0f;
  p.scale_softmax = scale;
  p.scale_softmax_log2 = scale * float(M_LOG2E);

  // FA2 works with the keep probability.
  p.p_dropout = 1.f - dropout;
  p.p_dropout_in_uint8_t = uint8_t(std::floor(p.p_dropout * 255.0));
  p.rp_dropout = 1.f / p.p_dropout;
  p.scale_softmax_rp_dropout = p.rp_dropout * p.scale_softmax;
  p.philox_args = seed_dev != nullptr
                      ? at::PhiloxCudaState(seed_dev, offset_dev)
                      : at::PhiloxCudaState(seed, offset);

  // Causal is window (left = S_kv, right = 0); S_q == S_kv is enforced by the
  // op, so FA2's bottom-right causal alignment matches the op's top-left one.
  p.is_causal = causal;
  p.window_size_left = causal ? S_kv : -1;
  p.window_size_right = causal ? 0 : -1;

  p.is_seqlens_k_cumulative = true;
  p.num_splits = 1;
  p.alibi_slopes_ptr = nullptr;
}

template <typename Elem, int kHeadDim>
void RunFwd(Flash_fwd_params &p, cudaStream_t stream) {
  if (p.is_causal)
    run_mha_fwd_<Elem, kHeadDim, true>(p, stream);
  else
    run_mha_fwd_<Elem, kHeadDim, false>(p, stream);
}

template <typename Elem, int kHeadDim>
void RunBwd(Flash_bwd_params &p, cudaStream_t stream) {
  if (p.is_causal)
    run_mha_bwd_<Elem, kHeadDim, true>(p, stream);
  else
    run_mha_bwd_<Elem, kHeadDim, false>(p, stream);
}

// Head dims and dtypes must match the kernels in
// //third_party/flash_attention:flash_attention.BUILD.
template <typename Elem>
void DispatchFwd(Flash_fwd_params &p, cudaStream_t stream) {
  if (p.d <= 32)
    RunFwd<Elem, 32>(p, stream);
  else if (p.d <= 64)
    RunFwd<Elem, 64>(p, stream);
  else if (p.d <= 96)
    RunFwd<Elem, 96>(p, stream);
  else if (p.d <= 128)
    RunFwd<Elem, 128>(p, stream);
  else if (p.d <= 192)
    RunFwd<Elem, 192>(p, stream);
  else
    RunFwd<Elem, 256>(p, stream);
}

template <typename Elem>
void DispatchBwd(Flash_bwd_params &p, cudaStream_t stream) {
  if (p.d <= 32)
    RunBwd<Elem, 32>(p, stream);
  else if (p.d <= 64)
    RunBwd<Elem, 64>(p, stream);
  else if (p.d <= 96)
    RunBwd<Elem, 96>(p, stream);
  else if (p.d <= 128)
    RunBwd<Elem, 128>(p, stream);
  else if (p.d <= 192)
    RunBwd<Elem, 192>(p, stream);
  else
    RunBwd<Elem, 256>(p, stream);
}

cudaError_t TakeError() {
  fa2_shim::record(cudaGetLastError());
  cudaError_t err = fa2_shim::last_error;
  fa2_shim::last_error = cudaSuccess;
  return err;
}

}  // namespace

bool ShapeSupported(int d_qk, int d_v) {
  return d_qk == d_v && d_qk > 0 && d_qk % 8 == 0 && d_qk <= 256;
}

bool GridSupported(int B, int H) {
  constexpr int kMaxGridYZ = 65535;
  return B <= kMaxGridYZ && H <= kMaxGridYZ;
}

bool Supported(int d_qk, int d_v) {
  return ShapeSupported(d_qk, d_v) && DeviceHasKernelImage();
}

bool BackwardSupported(int d_qk, int d_v, float dropout) {
  if (!Supported(d_qk, d_v)) return false;
  if (d_qk <= 192 || dropout <= 0.f) return true;
  // Mirrors the smem tiers in run_mha_bwd_hdim256
  // (flash_bwd_launch_template.h). Cached per device.
  constexpr int kMaxDevices = 64;
  static std::atomic<int> cached[kMaxDevices];  // 0 = unknown, 1 = yes, -1 = no
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= kMaxDevices) {
    cudaGetLastError();
    return false;
  }
  int val = cached[dev].load(std::memory_order_relaxed);
  if (val == 0) {
    int max_smem = 0;
    const bool ok = cudaDeviceGetAttribute(
                        &max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                        dev) == cudaSuccess;
    if (!ok) cudaGetLastError();  // don't leave the error for the next check
    val = (ok && max_smem >= 144 * 1024) ? 1 : -1;
    cached[dev].store(val, std::memory_order_relaxed);
  }
  return val == 1;
}

cudaError_t Forward(cudaStream_t stream, bool bf16, const void *q,
                    const void *k, const void *v, void *out, float *lse, int B,
                    int H, int S_q, int S_kv, int D, float scale, float dropout,
                    bool causal, uint64_t seed, uint64_t offset,
                    const uint64_t *seed_dev, const uint64_t *offset_dev) {
  if (B == 0 || H == 0 || S_q == 0) return cudaSuccess;
  const size_t rows_q = size_t(B) * H * S_q;
  if (S_kv == 0) {
    // No keys: the output is zero and the log-sum-exp of the empty set is -inf.
    constexpr int kThreads = 256;
    const int blocks = static_cast<int>(
        std::min<size_t>((rows_q + kThreads - 1) / kThreads, 4096));
    FillFloats<<<blocks, kThreads, 0, stream>>>(lse, rows_q, -INFINITY);
    if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) return err;
    return cudaMemsetAsync(out, 0, rows_q * D * kElemBytes, stream);
  }

  fa2_shim::last_error = cudaSuccess;
  Flash_fwd_params p;
  SetFwdParams(p, bf16, q, k, v, out, lse, B, H, S_q, S_kv, D, scale, dropout,
               causal, seed, offset, seed_dev, offset_dev);
  void *sink = nullptr;
  cudaGetSymbolAddress(&sink, g_fwd_rng_sink);
  p.rng_state = static_cast<uint64_t *>(sink);

  if (bf16)
    DispatchFwd<cutlass::bfloat16_t>(p, stream);
  else
    DispatchFwd<cutlass::half_t>(p, stream);
  return TakeError();
}

// dq_accum is [B, S_q_rounded, H, D_rounded] and dsoftmax_sum is
// [B, H, S_q_rounded], both float.
size_t BackwardWorkspaceBytes(int B, int H, int S_q, int D) {
  const size_t rows = size_t(B) * H * RoundUp(S_q, 128);
  const size_t dq_accum = AlignUp(rows * RoundUp(D, 32) * sizeof(float));
  const size_t softmax_d = AlignUp(rows * sizeof(float));
  return dq_accum + softmax_d + 2 * sizeof(uint64_t);
}

cudaError_t Backward(cudaStream_t stream, bool bf16, const void *q,
                     const void *k, const void *v, const void *out,
                     const float *lse, const void *dout, void *dq, void *dk,
                     void *dv, void *workspace, int B, int H, int S_q, int S_kv,
                     int D, float scale, float dropout, bool causal,
                     uint64_t seed, uint64_t offset, const uint64_t *seed_dev,
                     const uint64_t *offset_dev) {
  if (B == 0 || H == 0) return cudaSuccess;
  if (S_q == 0 || S_kv == 0) {
    // Every gradient is empty or zero.
    const size_t q_bytes = size_t(B) * H * S_q * D * kElemBytes;
    const size_t kv_bytes = size_t(B) * H * S_kv * D * kElemBytes;
    cudaMemsetAsync(dq, 0, q_bytes, stream);
    cudaMemsetAsync(dk, 0, kv_bytes, stream);
    return cudaMemsetAsync(dv, 0, kv_bytes, stream);
  }

  fa2_shim::last_error = cudaSuccess;
  Flash_bwd_params p;
  SetFwdParams(p, bf16, q, k, v, const_cast<void *>(out),
               const_cast<float *>(lse), B, H, S_q, S_kv, D, scale, dropout,
               causal, seed, offset, seed_dev, offset_dev);

  const Strides sq = BSHDStrides(S_q, H, D), skv = BSHDStrides(S_kv, H, D);
  p.do_ptr = const_cast<void *>(dout);
  p.dq_ptr = dq;
  p.dk_ptr = dk;
  p.dv_ptr = dv;
  p.do_batch_stride = p.dq_batch_stride = sq.batch;
  p.dk_batch_stride = p.dv_batch_stride = skv.batch;
  p.do_row_stride = p.dq_row_stride = sq.row;
  p.dk_row_stride = p.dv_row_stride = skv.row;
  p.do_head_stride = p.dq_head_stride = p.dk_head_stride = p.dv_head_stride = D;

  // Same layout as BackwardWorkspaceBytes.
  char *ws = static_cast<char *>(workspace);
  const size_t rows = size_t(B) * H * RoundUp(S_q, 128);
  p.dq_accum_ptr = ws;
  ws += AlignUp(rows * RoundUp(D, 32) * sizeof(float));
  p.dsoftmax_sum = ws;
  ws += AlignUp(rows * sizeof(float));
  p.rng_state = reinterpret_cast<uint64_t *>(ws);
  p.dk_accum_ptr = p.dv_accum_ptr = nullptr;
  p.deterministic = false;
  p.dq_accum_split_stride = 0;

  if (dropout > 0.f)
    SetRngState<<<1, 1, 0, stream>>>(p.rng_state, seed, offset, seed_dev,
                                     offset_dev);

  if (bf16)
    DispatchBwd<cutlass::bfloat16_t>(p, stream);
  else
    DispatchBwd<cutlass::half_t>(p, stream);
  return TakeError();
}

}  // namespace flash_attention_fa2

#else  // !__has_include("flash.h")

// Built without @flash_attention: report FA2 as unsupported everywhere, so the
// op always uses its own memory-efficient kernels.
namespace flash_attention_fa2 {

bool ShapeSupported(int d_qk, int d_v) { return false; }

bool GridSupported(int B, int H) { return false; }

bool Supported(int d_qk, int d_v) { return false; }

bool BackwardSupported(int d_qk, int d_v, float dropout) { return false; }

cudaError_t Forward(cudaStream_t stream, bool bf16, const void *q,
                    const void *k, const void *v, void *out, float *lse, int B,
                    int H, int S_q, int S_kv, int D, float scale, float dropout,
                    bool causal, uint64_t seed, uint64_t offset,
                    const uint64_t *seed_dev, const uint64_t *offset_dev) {
  return cudaErrorNotSupported;
}

size_t BackwardWorkspaceBytes(int B, int H, int S_q, int D) { return 0; }

cudaError_t Backward(cudaStream_t stream, bool bf16, const void *q,
                     const void *k, const void *v, const void *out,
                     const float *lse, const void *dout, void *dq, void *dk,
                     void *dv, void *workspace, int B, int H, int S_q, int S_kv,
                     int D, float scale, float dropout, bool causal,
                     uint64_t seed, uint64_t offset, const uint64_t *seed_dev,
                     const uint64_t *offset_dev) {
  return cudaErrorNotSupported;
}

}  // namespace flash_attention_fa2

#endif  // __has_include("flash.h")
