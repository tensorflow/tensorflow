/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#include "tensorflow/core/kernels/adjust_hsv_gpu.cu.h"
#include "tensorflow/core/kernels/adjust_saturation_op.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

namespace functor {

void AdjustSaturationGPU::operator()(GPUDevice* device,
                                     const int64 number_of_elements,
                                     const float* const input,
                                     const float* const scale,
                                     float* const output) {
  const auto stream = device->stream();
  const GpuLaunchConfig config =
      GetGpuLaunchConfig(number_of_elements, *device);
  const int threads_per_block = config.thread_per_block;
  const int block_count =
      (number_of_elements + threads_per_block - 1) / threads_per_block;
  GPU_LAUNCH_KERNEL((internal::adjust_hsv_nhwc<false, true, false>),
      dim3(block_count), dim3(threads_per_block), 0, stream,
          number_of_elements, input, output, nullptr, scale, nullptr);
}
}  // namespace functor
}  // namespace tensorflow
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
