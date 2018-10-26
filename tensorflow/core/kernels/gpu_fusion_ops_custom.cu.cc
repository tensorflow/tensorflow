/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#if TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/gpu_fusion_ops.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

#include "hip/hip_fp16.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace rocm_kernels {

//-------------------------------------------------------------------

// The "__hgt2" routine seems to return a '-1' for 'true' instead of '+1'
// Also there is no 'abs' routine for 'half2', which we can use to convert the
// '-1' to a '+1'. Since the RHS op in both instances of our usage of the __hgt2
// is 0, creating a custom "> 0" routine for half2 that will return '+1' for
// true

inline __device__ half2 greaterThanZero(half2 in) {
  return half2(static_cast<__half2_raw>(in).data.x > 0.0f16,
               static_cast<__half2_raw>(in).data.y > 0.0f16);
}

//-------------------------------------------------------------------

__global__ void AddReluKernel(int nthreads, const float* in0, const float* in1,
                              float* out) {
  GPU_1D_KERNEL_LOOP(index, nthreads) {
    out[index] = fmaxf(0.0f, in0[index] + in1[index]);
  }
}

void FusionAddRelu(OpKernelContext* ctx, const float* in0, const float* in1,
                   float* out, unsigned N) {
  GPUDevice d = ctx->eigen_device<GPUDevice>();
  GpuLaunchConfig config = GetGpuLaunchConfig(N, d);
  GPU_LAUNCH_KERNEL(AddReluKernel, dim3(config.block_count),
                    dim3(config.thread_per_block), 0, d.stream(),
                    config.virtual_thread_count, in0, in1, out);
}

__global__ void AddReluKernel(int nthreads, const half2* in0, const half2* in1,
                              half2* out) {
  GPU_1D_KERNEL_LOOP(index, nthreads) {
    half2 sum = __hadd2(in0[index], in1[index]);
    out[index] = __hmul2(greaterThanZero(sum), sum);
  }
}

void FusionAddRelu(OpKernelContext* ctx, const Eigen::half* in0,
                   const Eigen::half* in1, Eigen::half* out, unsigned N) {
  const half2* h_in0 = reinterpret_cast<const half2*>(in0);
  const half2* h_in1 = reinterpret_cast<const half2*>(in1);
  half2* h_out = reinterpret_cast<half2*>(out);
  unsigned h_N = (N + 1) / 2;
  
  GPUDevice d = ctx->eigen_device<GPUDevice>();
  GpuLaunchConfig config = GetGpuLaunchConfig(h_N, d);
  GPU_LAUNCH_KERNEL(AddReluKernel, dim3(config.block_count),
                    dim3(config.thread_per_block), 0, d.stream(),
                    config.virtual_thread_count, h_in0, h_in1, h_out);
}
  
//-------------------------------------------------------------------

__global__ void AddReluKernelBcast(int nthreads, unsigned M, const float* in0,
                                   const float* in1, float* out) {
  GPU_1D_KERNEL_LOOP(index, nthreads) {
    out[index] = fmaxf(0.0f, in0[index] + in1[index / M]);
  }
}

void FusionAddReluBcast(OpKernelContext* ctx, const float* in0,
                        const float* in1, float* out, unsigned N, unsigned M) {
  GPUDevice d = ctx->eigen_device<GPUDevice>();
  GpuLaunchConfig config = GetGpuLaunchConfig(N, d);
  GPU_LAUNCH_KERNEL(AddReluKernelBcast, dim3(config.block_count),
                    dim3(config.thread_per_block), 0, d.stream(),
                    config.virtual_thread_count, M, in0, in1, out);
}

__global__ void AddReluKernelBcast(int nthreads, unsigned M, const half2* in0,
                                   const half2* in1, half2* out) {
  GPU_1D_KERNEL_LOOP(index, nthreads) {
    half2 sum = __hadd2(in0[index], in1[index / M]);
    out[index] = __hmul2(greaterThanZero(sum), sum);
  }
}

void FusionAddReluBcast(OpKernelContext* ctx, const Eigen::half* in0,
                        const Eigen::half* in1, Eigen::half* out, unsigned N,
                        unsigned M) {
  const half2* h_in0 = reinterpret_cast<const half2*>(in0);
  const half2* h_in1 = reinterpret_cast<const half2*>(in1);
  half2* h_out = reinterpret_cast<half2*>(out);
  unsigned h_N = (N + 1) / 2;
  unsigned h_M = M / 2;

  GPUDevice d = ctx->eigen_device<GPUDevice>();
  GpuLaunchConfig config = GetGpuLaunchConfig(h_N, d);
  GPU_LAUNCH_KERNEL(AddReluKernelBcast, dim3(config.block_count),
                    dim3(config.thread_per_block), 0, d.stream(),
                    config.virtual_thread_count, h_M, h_in0, h_in1, h_out);
}

//-------------------------------------------------------------------

__global__ void AddNReluGradKernel(int nthreads, const float* in0,
                                   const float* in1, const float* in2,
                                   float* out) {
  GPU_1D_KERNEL_LOOP(index, nthreads) {
    out[index] = (in2[index] > 0.0f) ? (in0[index] + in1[index]) : 0.0f;
  }
}

void FusionAddNReluGrad(OpKernelContext* ctx, const float* in0,
                        const float* in1, const float* in2, float* out,
                        unsigned N) {
  GPUDevice d = ctx->eigen_device<GPUDevice>();
  GpuLaunchConfig config = GetGpuLaunchConfig(N, d);
  GPU_LAUNCH_KERNEL(AddNReluGradKernel, dim3(config.block_count),
                    dim3(config.thread_per_block), 0, d.stream(),
                    config.virtual_thread_count, in0, in1, in2, out);
}

__global__ void AddNReluGradKernel(int nthreads, const half2* in0,
                                   const half2* in1, const half2* in2,
                                   half2* out) {
  GPU_1D_KERNEL_LOOP(index, nthreads) {
    out[index] =
        __hmul2(greaterThanZero(in2[index]), __hadd2(in0[index], in1[index]));
  }
}

void FusionAddNReluGrad(OpKernelContext* ctx, const Eigen::half* in0,
                        const Eigen::half* in1, const Eigen::half* in2,
                        Eigen::half* out, unsigned N) {
  const half2* h_in0 = reinterpret_cast<const half2*>(in0);
  const half2* h_in1 = reinterpret_cast<const half2*>(in1);
  const half2* h_in2 = reinterpret_cast<const half2*>(in2);
  half2* h_out = reinterpret_cast<half2*>(out);
  unsigned h_N = (N + 1) / 2;

  GPUDevice d = ctx->eigen_device<GPUDevice>();
  GpuLaunchConfig config = GetGpuLaunchConfig(h_N, d);
  GPU_LAUNCH_KERNEL(AddNReluGradKernel, dim3(config.block_count),
                    dim3(config.thread_per_block), 0, d.stream(),
                    config.virtual_thread_count, h_in0, h_in1, h_in2, h_out);
}

//-------------------------------------------------------------------

}  // namespace rocm_kernels

}  // namespace tensorflow

#endif
