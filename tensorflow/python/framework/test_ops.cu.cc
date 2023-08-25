/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

#include "tensorflow/core/util/gpu_kernel_helper.h"

typedef Eigen::GpuDevice GPUDevice;

namespace tensorflow {

namespace {

__global__ void sleep_kernel(int seconds) {
#if __CUDA_ARCH__ >= 700  // __nanosleep requires compute capability 7.0
  int64_t nanoseconds = int64_t{seconds} * 1'000'000'000;
  // Passing too high a number to __nanosleep makes it sleep for much less time
  // than the passed-in number. So only pass 1,000,000 and keep calling
  // __nanosleep in a loop.
  for (int64_t i = 0; i < nanoseconds; i += 1'000'000) {
    __nanosleep(1'000'000);
  }
#endif
}

}  // namespace

void GpuSleep(OpKernelContext* ctx, int seconds) {
  auto* cu_stream = ctx->eigen_device<GPUDevice>().stream();
  CHECK(cu_stream);  // Crash OK
  TF_CHECK_OK(GpuLaunchKernel(sleep_kernel, 1, 1, 0, cu_stream, seconds));
}

}  // namespace tensorflow
#endif  // GOOGLE_CUDA
