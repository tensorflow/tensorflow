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
__global__ void RollCudaKernel(int64 N, int D, int* dim_size, const T* input,
                T* output, const int* thresholds, const int64* dim_strides) {

  const int64 start = blockIdx.x * blockDim.x + threadIdx.x;
  int in_dim_i[D]; // array of indices for each dimension
  int delta_i = 0; // the difference between out_i and in_i
  // initialize in_dim_i and delta_i
  for (int d = 0; d < D; d++) {
    const int ds = dim_size[d];
    // stride is the number of indices over in the flattened tensor
    // you need to skip in order to make it over to an adjacent element
    // along a dimension.
    const int64 stride = dim_strides[d] / ds;
    // calculated this way will always be positive modulo of shift
    const int shift = ds - thresholds[d];
    const int indx = (start / stride) % ds;
    in_dim_i[d] = indx;
    // calculate dimension index after the shift
    const int out_dim_i = (indx + shift) % ds;
    delta_i += (out_dim_i - indx) * stride;
  }

  for (int64 in_i = start; in_i < N; i += blockDim.x * gridDim.x) {
    const int64 out_i = in_i + delta_i;
    output_flat[out_i] = input_flat[in_i];

    // create next combination of in_dim_i[d]
    // while at it adjust delta_i if needed
    for (int d = D-1; d >= 0; d--) {
      const int indx = (in_dim_i[d] + 1) % dim_size[d];
      in_dim_i[d] = indx;
      if (indx != 0) {
        if (indx == thresholds[d]) {
          delta_i -= dim_strides[d]; // now wraps around
        }
        break; // indx != 0 don't need to carry
      }else{
        delta_i += dim_strides[d]; // indx became 0 so reverse wrap around
      }
    }
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct RollFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int64 N, int D, int* dim_size, const T* input, T* output
                  const int* thresholds, const int64* dim_strides){
    CudaLaunchConfig config = GetCudaLaunchConfig(out_size, d);
    RollCudaKernel<T>
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
            N, D, dim_size, input, output, thresholds, dim_strides);
  }
};

// Instantiate functors for the types of OpKernels registered.
typedef Eigen::GpuDevice GPUDevice;

// Definition of the GPU implementations declared in pad_op.cc.
#define DEFINE_GPU_SPECS(T)                      \
  template struct RollFunctor<GPUDevice, T>; \
TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);

#endif  // GOOGLE_CUDA
