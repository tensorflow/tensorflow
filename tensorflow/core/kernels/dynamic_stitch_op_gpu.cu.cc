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
__global__ void DynamicStitchKernel(const int32 slice_size,
                                    const int32 output_size,
                                    CudaDeviceArrayStruct<int32> input_indices,
                                    CudaDeviceArrayStruct<const T*> input_ptrs,
                                    T* output) {
  int32* data_indices = GetCudaDeviceArrayOnDevice(&input_indices);
  const T** data_ptrs = GetCudaDeviceArrayOnDevice(&input_ptrs);
  CUDA_1D_KERNEL_LOOP(output_index, output_size) {
    const int32 slice_id = output_index / slice_size;
    const int32 slice_offset = output_index % slice_size;
    const int32 input_index = data_indices[slice_id];
    if (input_index != -1) {
      output[output_index] = ldg(data_ptrs[input_index] + slice_offset);
    }
  }
}

}  // namespace

template <typename T>
void DynamicStitchGPUImpl(const Eigen::GpuDevice& gpu_device,
                          const int32 slice_size, const int32 first_dim_size,
                          const CudaDeviceArrayStruct<int>& input_indices,
                          const CudaDeviceArrayStruct<const T*>& input_ptrs,
                          T* output) {
  const int32 output_size = first_dim_size * slice_size;
  auto config = GetCudaLaunchConfig(output_size, gpu_device);

  DynamicStitchKernel<T>
      <<<config.block_count, config.thread_per_block, 0, gpu_device.stream()>>>(
          slice_size, output_size, input_indices, input_ptrs, output);
}

#define REGISTER_GPU(T)                                           \
  template void DynamicStitchGPUImpl(                             \
      const Eigen::GpuDevice& gpu_device, const int32 slice_size, \
      const int32 first_dim_size,                                 \
      const CudaDeviceArrayStruct<int32>& input_indices,          \
      const CudaDeviceArrayStruct<const T*>& input_ptrs, T* output);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
TF_CALL_complex64(REGISTER_GPU);
TF_CALL_complex128(REGISTER_GPU);
TF_CALL_int64(REGISTER_GPU);
TF_CALL_int32(REGISTER_GPU)

#undef REGISTER_GPU

}  // namespace tensorflow
#endif  // GOOGLE_CUDA
