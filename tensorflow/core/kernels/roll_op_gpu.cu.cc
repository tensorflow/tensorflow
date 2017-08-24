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

#ifdef GOOGLE_CUDA

#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "roll_op.h"

using namespace tensorflow;

#define EIGEN_USE_GPU

// Define the CUDA kernel.
template <typename T>
__global__ void RollCudaKernel(int64 N, int D, int* dim_size, const T* input, T* output,\
                const int* shifts, const int64* strides) {
  for (int64 in_i = blockIdx.x * blockDim.x + threadIdx.x; in_i < N;\
                  i += blockDim.x * gridDim.x) {
    int64 out_i = in_i;
    // loop through dimensions
    for (int d = 0; d < D; d++) {
      // find indices input/output for current dimension
      const int ds = dim_size[d];
      const int in_dim_i = (in_i / strides[d]) % ds;
      const int out_dim_i = ((in_dim_i + shifts[d]) % ds + ds) % ds; // modulo that works with negatives
      // convert back to flat index
      out_i += (out_dim_i - in_dim_i) * strides[d];
    }

    output[out_i] = input[in_i];
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct RollFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int64 N, int D, int* dim_size, const T* input, T* output,\
                  const int* shifts, const int64* strides){
    CudaLaunchConfig config = GetCudaLaunchConfig(out_size, d);
    RollCudaKernel<T>
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
            N, D, dim_size, input, output, shifts, strides);
  }
};

// Instantiate functors for the types of OpKernels registered.
typedef Eigen::GpuDevice GPUDevice;

// Definition of the GPU implementations declared in pad_op.cc.
#define DEFINE_GPU_SPECS(T)                      \
  template struct RollFunctor<GPUDevice, T>; \
TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);

#endif  // GOOGLE_CUDA
