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

#if GOOGLE_CUDA

#include "tensorflow/core/common_runtime/gpu/gpu_device_kernel_check.h"
#include "tensorflow/stream_executor/cuda/cuda_activation.h"

namespace {
__global__ void test_kernel(float* val) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    (*val) = 12345.;
  }
}
}  // namespace

namespace tensorflow {

void run_test_kernel(float* val, cudaStream_t cu_stream) {
  test_kernel<<<1, 1, 0, cu_stream>>>(val);
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
