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

#include "tensorflow/contrib/tensorrt/custom_plugin_examples/inc_op_kernel.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {

__global__ void VecInc(const float* vec, float inc, float* dest, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) dest[i] = vec[i] + inc;
}

void IncrementKernel(const float* d_input, float inc, float* d_output,
                     int count, cudaStream_t stream) {
  int threads_per_block = 256;
  int blocks_per_grid = (count + threads_per_block - 1) / threads_per_block;

  VecInc<<<threads_per_block, blocks_per_grid, 0, stream>>>(d_input, inc,
                                                            d_output, count);
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
#endif  // GOOGLE_TENSORRT
