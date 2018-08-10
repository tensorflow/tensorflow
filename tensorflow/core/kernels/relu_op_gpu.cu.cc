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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/relu_op_functor.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/util/cuda_launch_config.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {
// This kernel computes ReluGrad by processing one half2, two fp16, at a time.
// It effectively does: backdrops = (feature > 0) ? gradient : 0
// It also tries to use native half2 primitives as much as possible.
__global__ void ReluGradHalfKernel(const Eigen::half* gradient,
                                   const Eigen::half* feature,
                                   Eigen::half* backprop, int32 count) {
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
    float2 backprop_f2 = make_float2((feature_f2.x > 0) ? gradient_f2.x : 0,
                                     (feature_f2.y > 0) ? gradient_f2.y : 0);
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
    int32 count = gradient.size();
    if (count == 0) return;
    int32 half2_count = Eigen::divup(count, 2);
    const int32 kThreadInBlock = 512;
    CudaLaunchConfig config = GetCudaLaunchConfigFixedBlockSize(
        half2_count, d, ReluGradHalfKernel, 0, kThreadInBlock);
    ReluGradHalfKernel<<<config.block_count, config.thread_per_block, 0,
                         d.stream()>>>(gradient.data(), feature.data(),
                                       backprop.data(), count);
  }
};
}  // namespace functor

// Definition of the GPU implementations declared in relu_op.cc.
#define DEFINE_GPU_KERNELS(T)                       \
  template struct functor::Relu<GPUDevice, T>;      \
  template struct functor::ReluGrad<GPUDevice, T>;  \
  template struct functor::Relu6<GPUDevice, T>;     \
  template struct functor::Relu6Grad<GPUDevice, T>; \
  template struct functor::Elu<GPUDevice, T>;       \
  template struct functor::EluGrad<GPUDevice, T>;   \
  template struct functor::Selu<GPUDevice, T>;      \
  template struct functor::SeluGrad<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_KERNELS);

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
