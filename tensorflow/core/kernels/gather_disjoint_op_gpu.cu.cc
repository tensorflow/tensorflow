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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/cuda_device_array_gpu.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

namespace {

template <typename T>
__global__ void GatherDisjointKernel(const int64 slice_elems,
                                     const int64 output_size,
                                     CudaDeviceArrayStruct<int8> zero_indicator,
                                     T* output) {
  int8* zero_loc = GetCudaDeviceArrayOnDevice(&zero_indicator);
  CUDA_1D_KERNEL_LOOP(output_index, output_size) {
    const int64 slice_id = output_index / slice_elems;
    const int8 fill_zero = zero_loc[slice_id];
    if (fill_zero) output[output_index] = static_cast<T>(0);
  }
}

}  // namespace

template <typename T>
void GatherDisjointOpGPUImpl(const Eigen::GpuDevice& gpu_device,
                             const int64 first_dim_size,
                             const int64 slice_elems,
                             const CudaDeviceArrayStruct<int8>& zero_indicator,
                             T* output) {
  const int64 output_size = first_dim_size * slice_elems;
  auto config = GetCudaLaunchConfig(output_size, gpu_device);

  GatherDisjointKernel<T><<<config.block_count, config.thread_per_block, 0,
                            gpu_device.stream()>>>(slice_elems, output_size,
                                                   zero_indicator, output);
}

#define REGISTER_GPU(T)                                               \
  template void GatherDisjointOpGPUImpl(                              \
      const Eigen::GpuDevice& gpu_device, const int64 first_dim_size, \
      const int64 slice_elems,                                        \
      const CudaDeviceArrayStruct<int8>& zero_indicator, T* output);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
TF_CALL_complex64(REGISTER_GPU);
TF_CALL_complex128(REGISTER_GPU);
TF_CALL_int64(REGISTER_GPU);
TF_CALL_int32(REGISTER_GPU);

#undef REGISTER_GPU

}  // namespace tensorflow
#endif  // GOOGLE_CUDA
