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

#include "roll_op.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

#define EIGEN_USE_GPU

// CUDA kernel.
template <typename T>
__global__ void RollCudaKernel(int64 N, int D, int* dim_size, const T* input,
                               T* output, const int* threshold,
                               const int64* dim_range) {
  const int64 start = blockIdx.x * blockDim.x + threadIdx.x;
  const int64 end = N;

  int indices[D];  // array of indices for each dimension
  int offset = 0;  // the shift along the flat tensor for current element
  // initialize indices and offset
  for (int d = 0; d < D; d++) {
    // stride is the number of indices over in the flattened tensor
    // you need to skip in order to make it over to an adjacent element
    // along a dimension.
    const int64 stride = dim_range[d] / dim_size[d];
    const int shift = dim_size[d] - threshold[d];
    const int indx = (start / stride) % dim_size[d];
    indices[d] = indx;
    // calculate dimension index after the shift
    const int shifted_indx = (indx + shift) % dim_size[d];
    offset += (shifted_indx - indx) * stride;
  }

  CUDA_1D_KERNEL_LOOP(start, end) {
    output[i + offset] = input[i];
    // create next combination of indices
    // while at it adjust offset if needed
    for (int d = D - 1; d >= 0; d--) {
      const int indx = (indices[d] + 1) % dim_size[d];
      indices[d] = indx;
      if (indx != 0) {
        if (indx == threshold[d]) {  // we've reached the threshold
          // dim_range[d] = threshold[d] + shift[d]
          // offset = shift[d] + ... other offsets
          // offset - dim_range[d] = -threshold[d] + ... other offsets
          // thus we undo our previous offset as well as add a new offset of
          // -threshold[d] in one opperation
          offset -= dim_range[d];  // now wraps around
        }
        break;                         // indx != 0 don't need to carry
      } else if (threshold[d] != 0) {  // if threshold is 0 shift is 0
        offset += dim_range[d];        // indx became 0 so reverse wrap around
      }
    }
  }
}

// GPU implementation that launches the CUDA kernel.
template <typename T>
struct RollFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, int64 N, int D, int* dim_size,
                  const T* input, T* output const int* threshold,
                  const int64* dim_range) {
    CudaLaunchConfig config = GetCudaLaunchConfig(out_size, d);
    RollCudaKernel<T>
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
            N, D, dim_size, input, output, threshold, dim_range);
  }
};

// Instantiate functors for the types of OpKernels registered.
typedef Eigen::GpuDevice GPUDevice;

// Definition of the GPU implementations declared in roll_op.h.
#define DEFINE_GPU_SPECS(T)                  \
  template struct RollFunctor<GPUDevice, T>; \
  TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);

#endif  // GOOGLE_CUDA
