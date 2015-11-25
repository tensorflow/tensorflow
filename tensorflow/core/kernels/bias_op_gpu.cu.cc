/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include <algorithm>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/bias_op.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

// Definition of the GPU implementations declared in bias_op.cc.

namespace functor {

template <typename T>
__global__ void BiasOpCustomKernel(int nthreads, const T* input, const T* bias,
                                   int bias_size, int replicate_count,
                                   T* output) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int bias_offset = index % bias_size;
    output[index] = ldg(input + index) + ldg(bias + bias_offset);
  }
}

template <typename T, int Dims>
struct Bias<GPUDevice, T, Dims> {
  typedef GPUDevice Device;
  // Add "bias" to "input", broadcasting it on all dimensions but the last one.
  void operator()(const Device& d, typename TTypes<T, Dims>::ConstTensor input,
                  typename TTypes<T>::ConstVec bias,
                  typename TTypes<T, Dims>::Tensor output) {
    const int bias_size = bias.dimension(0);
    const int rest_size = input.size() / bias_size;
    CudaLaunchConfig config = GetCudaLaunchConfig(output.size(), d);
    BiasOpCustomKernel<<<config.block_count, config.thread_per_block, 0,
                         d.stream()>>>(config.virtual_thread_count,
                                       input.data(), bias.data(), bias_size,
                                       rest_size, output.data());
  }
};

}  // namespace functor

#define DEFINE_GPU_SPECS(T)                       \
  template struct functor::Bias<GPUDevice, T, 2>; \
  template struct functor::Bias<GPUDevice, T, 3>; \
  template struct functor::Bias<GPUDevice, T, 4>; \
  template struct functor::Bias<GPUDevice, T, 5>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
