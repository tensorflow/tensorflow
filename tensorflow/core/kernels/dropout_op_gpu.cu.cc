#if TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/random_op.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/stream_executor/temporary_device_memory.h"
#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "dropout_op.h"

#if TENSORFLOW_USE_ROCM
#include "rocm/include/hip/hip_fp16.h"
#else
#include "third_party/gpus/cuda/include/cuda_fp16.hpp"
#endif

namespace tensorflow {

template <class T, class U>
__device__ void apply_dropout(T& out, half2 rng, half2 rate, half2 scale) {
  half2 mask = make_half2(__low2half(rng) > __low2half(rate),
                          __high2half(rng) > __high2half(rate));
  out = __hmul2(__hmul2(mask, scale), out);
}

template <class T, class U>
__device__ void apply_dropout(T& out, half2 rng, half2 rate, float2 scale) {
  out.x *= scale.x;
  out.y *= scale.y;

  out.x = (__low2half(rng) > __low2half(rate)) ? U(out.x) : 0.0f;
  out.y = (__high2half(rng) > __high2half(rate)) ? U(out.y) : 0.0f;
}

template <>
__device__ void apply_dropout<half2, half>(half2& out, half2 rng, half2 rate,
                                           float2 scale) {
  half2 mask = make_half2(__low2half(rng) > __low2half(rate),
                          __high2half(rng) > __high2half(rate));
  out = __hmul2(__hmul2(mask, __float22half2_rn(scale)), out);
}

__device__ void uint32_to_half4(uint32 x, uint32 y, half2& h1, half2& h2) {
  uint32 x1 = 0x3c003c00 | (x & 0x03ff03ff);
  x ^= y;
  x = (x >> 10) | (x << 22);
  uint32 x2 = 0x3c003c00 | (x & 0x03ff03ff);
  h1 = reinterpret_cast<half2&>(x1);
  h2 = reinterpret_cast<half2&>(x2);
}

template <typename T, typename U, typename V>
__global__ void RNGAndApplyDropoutKernel(random::PhiloxRandom gen, uint32 size,
                                         T* _out, const T* _in, U rate,
                                         V scale) {
  constexpr bool is_half2 = std::is_same<T, half2>::value;
  constexpr bool is_half = std::is_same<T, Eigen::half>::value;
  constexpr bool is_float = std::is_same<T, float>::value;
  // Cast inputs from Eigen::half to __half. TODO: is there a better way of
  // doing this?
  typedef typename std::conditional<is_half, half, T>::type TT;
  TT* out = reinterpret_cast<TT*>(_out);
  const TT* in = reinterpret_cast<const TT*>(_in);
  typedef typename std::conditional<
      is_half || is_half2, half2,
      typename std::conditional<is_float, float2, double2>::type>::type
      PackedType;

  rate += half2(1.0f, 1.0f);

  const uint32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32 total_thread_count = gridDim.x * blockDim.x;
  uint32 thread_step = total_thread_count;
  uint32 offset = thread_id;

  gen.Skip(thread_id);

  while (true) {
    if (offset >= size) break;

    const TT* p = in + offset;
    TT* pout = out + offset;

    static const int slen = is_half2 ? 2 : 4;
    union {
      TT s[slen];
      PackedType p[2];
    } preload;
    half2 rng_data[2];

    // Generate 128 bits of random data
    auto sample = gen();

    for (int j = 0; j < 4; j++) {
      // Collect either 2 or 4 values from the input buffer
      uint32 off = offset;
      for (int i = 0; i < slen; i++) {
        if (off < size) preload.s[i] = p[0];
        off += thread_step;
        p += thread_step;
      }

      // Generate 4 random float16 values in [1,2) range
      // Note: this requires some random bit reuse (since each float16 takes
      // 10 bits, and we're trying to generate 16 of them ot of 128 bit per
      // RNG run), but we're not designing NSA-proof encryption here ...
      uint32_to_half4(sample[j], sample[(j + 2) & 3] ^ (j >= 2 ? -1 : 0),
                      rng_data[0], rng_data[1]);

      // Apply dropout, in pairs
      apply_dropout<PackedType, TT>(preload.p[0], rng_data[0], rate, scale);
      apply_dropout<PackedType, TT>(preload.p[1], rng_data[1], rate, scale);

      // Write out the results
      for (int i = 0; i < slen; i++) {
        if (offset < size) pout[0] = preload.s[i];
        offset += thread_step;
        pout += thread_step;
      }
    }
    gen.Skip(total_thread_count - 1);
  }
}

template <typename T>
__global__ void ApplyDropoutGradKernel(T* outgrads, const T* grads,
                                       const T* ins, const T* outs, float rate,
                                       float scale, uint64 num_elements) {
  for (uint64 i = threadIdx.x + blockIdx.x * blockDim.x; i < num_elements;
       i += blockDim.x * gridDim.x)
    outgrads[i] = grads[i] * T((outs[i] == T(0)) ? 0.0f : scale);
}

template <>
__global__ void ApplyDropoutGradKernel(Eigen::half* _outgrads,
                                       const Eigen::half* _grads,
                                       const Eigen::half* _ins,
                                       const Eigen::half* _outs, float rate,
                                       float scale, uint64 num_elements) {
  __half* outgrads = reinterpret_cast<__half*>(_outgrads);
  const __half* grads = reinterpret_cast<const __half*>(_grads);
  const __half* outs = reinterpret_cast<const __half*>(_outs);
  for (uint64 i = threadIdx.x + blockIdx.x * blockDim.x; i < num_elements;
       i += blockDim.x * gridDim.x)
    outgrads[i] = __float2half(
        (outs[i] == __half(0.0f)) ? 0.0f : __half2float(grads[i]) * scale);
}

template <typename T>
void ApplyDropout<GPUDevice, T>::operator()(const GPUDevice& d, T* out,
                                            const T* in, const float* unused,
                                            float rate, uint64 num_elements,
                                            random::PhiloxRandom gen,
                                            bool seeded) {
  float scale = 1. / (1 - rate);
  // //tensorflow/python:auto_mixed_precision_test appears to want identical
  // RNG behavior between half and float for the same seed (otherwise, the
  // gradient test fails.) To keep it happy, we prohibit the use of half2
  // shortcut when seed is explicitly specified.
  bool do_half2 =
      std::is_same<T, Eigen::half>::value && !(num_elements & 1) && !seeded;
  if (do_half2) num_elements /= 2;
  int64 kThreadInBlock = 256;
  // int64 kMaxBlock = do_half2 ? 1024 : 128;  // experimental best
  // int64 kMaxBlock = 128;
  //(void)ReadInt64FromEnvVar("TF_DROPOUT_THREADS", 256, &kThreadInBlock);
  //(void)ReadInt64FromEnvVar("TF_DROPOUT_MAX_BLOCKS", 0, &kMaxBlock);
  // we process 4 half2 in half2 mode and 16 in other cases
  int group_size = do_half2 ? 8 : 16;
  uint64 num_groups = (num_elements + group_size - 1) / group_size;
  uint64 num_blocks = (num_groups + kThreadInBlock - 1) / kThreadInBlock;
  // num_blocks = min(kMaxBlock, num_blocks);
  // if(kMaxBlock>0)
  //  num_blocks = kMaxBlock;
  // else if(kMaxBlock==-1)
  //  num_blocks = (num_blocks+63) & ~63;

  // for FP32, it's optimal to run at 256x256
  if (std::is_same<T, float>::value && num_blocks > 256) num_blocks = 256;

  if (do_half2) {
    TF_CHECK_OK(GpuLaunchKernel(
        RNGAndApplyDropoutKernel<half2, half2, half2>, num_blocks,
        kThreadInBlock, 0, d.stream(), gen, num_elements,
        reinterpret_cast<half2*>(out), reinterpret_cast<const half2*>(in),
        __floats2half2_rn(rate, rate), __floats2half2_rn(scale, scale)));
  } else {
    TF_CHECK_OK(GpuLaunchKernel(
        RNGAndApplyDropoutKernel<T, half2, float2>, num_blocks, kThreadInBlock,
        0, d.stream(), gen, num_elements, out, in,
        __floats2half2_rn(rate, rate), make_float2(scale, scale)));
  }
}

template <typename T>
void ApplyDropoutGrad<GPUDevice, T>::operator()(const GPUDevice& d, T* outgrads,
                                                const T* grads, const T* ins,
                                                const T* outs, float rate,
                                                uint64 num_elements) {
  float scale = 1. / (1 - rate);
  int64 kThreadInBlock = 1024;
  int64 kMaxBlock = 512;
  TF_CHECK_OK(GpuLaunchKernel(
      ApplyDropoutGradKernel<T>,
      min(kMaxBlock, (num_elements + kThreadInBlock - 1) / kThreadInBlock),
      kThreadInBlock, 0, d.stream(), outgrads, grads, ins, outs, rate, scale,
      num_elements));
}

template void ApplyDropout<GPUDevice, Eigen::half>::operator()(
    const GPUDevice& d, Eigen::half* out, const Eigen::half* in,
    const float* rng_data, float rate, uint64 num_elements,
    random::PhiloxRandom gen, bool);
template void ApplyDropout<GPUDevice, float>::operator()(
    const GPUDevice& d, float* out, const float* in, const float* rng_data,
    float rate, uint64 num_elements, random::PhiloxRandom gen, bool);
template void ApplyDropout<GPUDevice, double>::operator()(
    const GPUDevice& d, double* out, const double* in, const float* rng_data,
    float rate, uint64 num_elements, random::PhiloxRandom gen, bool);

template void ApplyDropoutGrad<GPUDevice, Eigen::half>::operator()(
    const GPUDevice& d, Eigen::half* outgrads, const Eigen::half* grads,
    const Eigen::half* ins, const Eigen::half* outs, float rate,
    uint64 num_elements);
template void ApplyDropoutGrad<GPUDevice, float>::operator()(
    const GPUDevice& d, float* outgrads, const float* grads, const float* ins,
    const float* outs, float rate, uint64 num_elements);
template void ApplyDropoutGrad<GPUDevice, double>::operator()(
    const GPUDevice& d, double* outgrads, const double* grads,
    const double* ins, const double* outs, float rate, uint64 num_elements);

};  // namespace tensorflow

#endif
