/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/population_count_op.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename T>
__global__ void PopulationCountKernel(const int size,
                                      const T* __restrict__ input,
                                      uint8* __restrict__ output) {
  GPU_1D_KERNEL_LOOP(i, size) { output[i] = __popc(ldg(input + i)); }
}

template <>
__global__ void PopulationCountKernel(const int size,
                                      const int8* __restrict__ input,
                                      uint8* __restrict__ output) {
  // For some reason, __popc on a negative int8 gets confused.
  GPU_1D_KERNEL_LOOP(i, size) {
    output[i] = __popc(ldg(reinterpret_cast<const uint8*>(input + i)));
  }
}

template <>
__global__ void PopulationCountKernel(const int size,
                                      const int16* __restrict__ input,
                                      uint8* __restrict__ output) {
  // For some reason, __popc on a negative int16 gets confused.
  GPU_1D_KERNEL_LOOP(i, size) {
    output[i] = __popc(ldg(reinterpret_cast<const uint16*>(input + i)));
  }
}

template <>
__global__ void PopulationCountKernel<int64_t>(const int size,
                                               const int64* __restrict__ input,
                                               uint8* __restrict__ output) {
  GPU_1D_KERNEL_LOOP(i, size) { output[i] = __popcll(ldg(input + i)); }
}

#define DEFINE_GPU_SPECS(T)                                                   \
  template <>                                                                 \
  void PopulationCount<GPUDevice, T>::operator()(                             \
      OpKernelContext* c, typename TTypes<T>::ConstFlat input,                \
      TTypes<uint8>::Flat output) {                                           \
    const GPUDevice& d = c->eigen_device<GPUDevice>();                        \
    int64 total_count = input.size();                                         \
    if (total_count == 0) {                                                   \
      /* Return on empty input. Output is empty for empty inputs,             \
       similar to CPU kernel.   */                                            \
      return;                                                                 \
    }                                                                         \
    GpuLaunchConfig config = GetGpuLaunchConfig(total_count, d);              \
    TF_CHECK_OK(GpuLaunchKernel(PopulationCountKernel<T>, config.block_count, \
                                config.thread_per_block, 0, d.stream(),       \
                                total_count, input.data(), output.data()));   \
  }

TF_CALL_uint8(DEFINE_GPU_SPECS);
TF_CALL_int8(DEFINE_GPU_SPECS);
TF_CALL_uint16(DEFINE_GPU_SPECS);
TF_CALL_int16(DEFINE_GPU_SPECS);
TF_CALL_int32(DEFINE_GPU_SPECS);
TF_CALL_int64(DEFINE_GPU_SPECS);

#undef DEFINE_GPU_SPECS

}  // namespace functor

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
