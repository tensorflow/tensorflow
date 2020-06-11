/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include <stdio.h>

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/relu_op_functor.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"

#if TENSORFLOW_USE_ROCM
#include "rocm/include/hip/hip_fp16.h"
typedef __half2 half2;
#endif

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

static constexpr int VectorSizeElements = 8;
namespace functor {

// This kernel computes ReluGrad by processing one half2, two fp16, at a time.
// It effectively does: backdrops = (feature > 0) ? gradient : 0
// It also tries to use native half2 primitives as much as possible.
__global__ void ReluGradHalfKernel(const Eigen::half* __restrict__ gradient,
                                   const Eigen::half* __restrict__ feature,
                                   Eigen::half* __restrict__ backprop,
                                   int32 count) {
  int32 half2_count = count >> 1;
  int32 index = blockIdx.x * blockDim.x + threadIdx.x;
  const int32 total_device_threads = gridDim.x * blockDim.x;

  while (index < half2_count) {
    // The fast branch.
    // One half2, two fp16, is fetched and processed at a time.
    half2 gradient_h2 = reinterpret_cast<const half2*>(gradient)[index];
    half2 feature_h2 = reinterpret_cast<const half2*>(feature)[index];
    half2* p_backprop_h2 = reinterpret_cast<half2*>(backprop) + index;

#if __CUDA_ARCH__ >= 530
    // Fast path, when half2 primitives are available.
    const half2 kZeroH2 = __float2half2_rn(0.f);
    // mask = (feature > 0)
    half2 mask_h2 = __hgt2(feature_h2, kZeroH2);
    // backprop = mask * gradient
    half2 backprop_h2 = __hmul2(mask_h2, gradient_h2);
#else
    // Fall back: convert half2 to float2 for processing.
    float2 feature_f2 = __half22float2(feature_h2);
    float2 gradient_f2 = __half22float2(gradient_h2);
    float2 backprop_f2 =
        make_float2((feature_f2.x > 0.0f) ? float(gradient_f2.x) : 0.0f,
                    (feature_f2.y > 0.0f) ? float(gradient_f2.y) : 0.0f);
    // Convert back to half2.
    half2 backprop_h2 = __float22half2_rn(backprop_f2);
#endif

    // Write back the result.
    *p_backprop_h2 = backprop_h2;

    index += total_device_threads;
  }

  if ((count & 0x1) == 1 && index == half2_count) {
    // If the total number of the elements is odd, process the last element.
    Eigen::half grad_h = gradient[count - 1];
    Eigen::half feature_h = feature[count - 1];

    float grad_f = static_cast<float>(grad_h);
    float feature_f = static_cast<float>(feature_h);
    float backprop_f = (feature_f > 0) ? grad_f : 0;

    Eigen::half backprop_h(backprop_f);
    backprop[count - 1] = backprop_h;
  }
}

__global__ void ReluGradHalfKernelVector(
    const Eigen::half* __restrict__ gradient,
    const Eigen::half* __restrict__ feature,
    Eigen::half* __restrict__ backprop, int32 count) {
  int32 half8_count = count / VectorSizeElements;
  int32 index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < half8_count) {
    // Cast to xx_h8 for vector load and store.
    float4 gradient_h8 = reinterpret_cast<const float4*>(gradient)[index];
    float4 feature_h8 = reinterpret_cast<const float4*>(feature)[index];
    float4* p_backprop_h8 = reinterpret_cast<float4*>(backprop) + index;

    half2 *gradient_h2 = reinterpret_cast<half2*>(&gradient_h8);
    half2 *feature_h2 = reinterpret_cast<half2*>(&feature_h8);
    float4 backprop_h8;
    half2* p_backprop_h2 = reinterpret_cast<half2*>(&backprop_h8);

    // Fast path, when half2 primitives are available.
#if __CUDA_ARCH__ >= 530
    const half2 kZeroH2 = __float2half2_rn(0.f);
#endif
    for (int i = 0; i < VectorSizeElements / 2; i++) {
#if __CUDA_ARCH__ >= 530
      // mask = (feature > 0)
      half2 mask_h2 = __hgt2(feature_h2[i], kZeroH2);
      // backprop = mask * gradient
      half2 backprop_h2 = __hmul2(mask_h2, gradient_h2[i]);
#else
      // Fall back: convert half2 to float2 for processing.
      float2 feature_f2 = __half22float2(feature_h2[i]);
      float2 gradient_f2 = __half22float2(gradient_h2[i]);
      float2 backprop_f2 = make_float2((feature_f2.x > 0) ? gradient_f2.x : 0,
                                       (feature_f2.y > 0) ? gradient_f2.y : 0);
      // Convert back to half2.
      half2 backprop_h2 = __float22half2_rn(backprop_f2);
#endif
      p_backprop_h2[i] = backprop_h2;
    }
    // Write back the result.
    *p_backprop_h8 = backprop_h8;
  }

  int remaining_count = (count % VectorSizeElements);

  if (index < remaining_count) {
    // Use first threads to process the remaining elements.
    Eigen::half grad_h = gradient[half8_count * VectorSizeElements + index];
    Eigen::half feature_h = feature[half8_count * VectorSizeElements + index];

    float grad_f = static_cast<float>(grad_h);
    float feature_f = static_cast<float>(feature_h);
    float backprop_f = (feature_f > 0) ? grad_f : 0;

    Eigen::half backprop_h(backprop_f);
    backprop[half8_count * VectorSizeElements + index] = backprop_h;
  }
}

template <typename Device>
struct ReluGrad<Device, Eigen::half> {
  // Computes ReluGrad backprop.
  //
  // gradient: gradient backpropagated to the Relu op.
  // feature: either the inputs that were passed to the Relu, or its outputs
  //           (using either one yields the same result here).
  // backprop: gradient to backpropagate to the Relu inputs.
  void operator()(const Device& d,
                  typename TTypes<Eigen::half>::ConstTensor gradient,
                  typename TTypes<Eigen::half>::ConstTensor feature,
                  typename TTypes<Eigen::half>::Tensor backprop) {
    // NOTE: When the activation is exactly zero, we do not propagate the
    // associated gradient value. This allows the output of the Relu to be used,
    // as well as its input.
    auto gradient_ptr = reinterpret_cast<uintptr_t>(gradient.data());
    auto feature_ptr = reinterpret_cast<uintptr_t>(feature.data());
    auto backprop_ptr = reinterpret_cast<uintptr_t>(backprop.data());
    bool aligned = gradient_ptr % 16 == 0 && feature_ptr % 16 == 0 &&
                   backprop_ptr % 16 == 0;
    int32 count = gradient.size();
    constexpr int32 kThreadInBlock = 512;
    if (count == 0) return;
    if (aligned) {
      int32 half8_count = Eigen::divup(count, VectorSizeElements);
      int32 kBlock = Eigen::divup(half8_count, kThreadInBlock);
      TF_CHECK_OK(GpuLaunchKernel(
          ReluGradHalfKernelVector, kBlock, kThreadInBlock,
          0, d.stream(), gradient.data(), feature.data(), backprop.data(),
          count));
    } else {
      int32 half2_count = Eigen::divup(count, 2);
      GpuLaunchConfig config = GetGpuLaunchConfigFixedBlockSize(
          half2_count, d, ReluGradHalfKernel, 0, kThreadInBlock);
      TF_CHECK_OK(GpuLaunchKernel(
          ReluGradHalfKernel, config.block_count, config.thread_per_block, 0,
          d.stream(), gradient.data(), feature.data(), backprop.data(), count));
    }
  }
};

__global__ void Relu_int8x4_kernel(int vect_count,
                                   const int32* __restrict__ input,
                                   int32* __restrict__ output) {
  CUDA_1D_KERNEL_LOOP(index, vect_count) {
#if GOOGLE_CUDA
    output[index] = __vmaxs4(input[index], 0);
#else
    uint32 signs = (~input[index]) & 0x80808080;
    signs = signs >> 7;
    signs |= signs << 1;
    signs |= signs << 2;
    signs |= signs << 4;
    signs &= 0x7f7f7f7f;
    output[index] = input[index] & signs;
#endif
  }
}

// Functor used by ReluOp to do the computations.
template <typename Device>
struct Relu<Device, qint8> {
  // Computes Relu activation of 'input' containing int8 elements, whose buffer
  // size should be a multiple of 4, and aligned to an int32* boundary.
  // (Alignment should be guaranteed by the GPU tensor allocator).
  // 'output' should have the same size as 'input'.
  void operator()(const Device& d, typename TTypes<qint8>::ConstTensor input,
                  typename TTypes<qint8>::Tensor output) {
    int32 count = input.size();
    if (count == 0) return;

    int32 vect_count = Eigen::divup(count, 4);
    constexpr int32 kThreadInBlock = 512;
    GpuLaunchConfig config = GetGpuLaunchConfigFixedBlockSize(
        vect_count, d, Relu_int8x4_kernel, 0, kThreadInBlock);
    TF_CHECK_OK(GpuLaunchKernel(
        Relu_int8x4_kernel, config.block_count, config.thread_per_block, 0,
        d.stream(), vect_count, reinterpret_cast<const int32*>(input.data()),
        reinterpret_cast<int32*>(output.data())));
  }
};

}  // namespace functor

// Definition of the GPU implementations declared in relu_op.cc.
#define DEFINE_GPU_KERNELS(T)                           \
  template struct functor::Relu<GPUDevice, T>;          \
  template struct functor::ReluGrad<GPUDevice, T>;      \
  template struct functor::Relu6<GPUDevice, T>;         \
  template struct functor::Relu6Grad<GPUDevice, T>;     \
  template struct functor::LeakyRelu<GPUDevice, T>;     \
  template struct functor::LeakyReluGrad<GPUDevice, T>; \
  template struct functor::Elu<GPUDevice, T>;           \
  template struct functor::EluGrad<GPUDevice, T>;       \
  template struct functor::Selu<GPUDevice, T>;          \
  template struct functor::SeluGrad<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_KERNELS);
template struct functor::Relu<GPUDevice, qint8>;

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
