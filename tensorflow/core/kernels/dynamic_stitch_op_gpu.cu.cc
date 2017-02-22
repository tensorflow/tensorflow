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
#include <math.h>
#include <stdio.h>
#include <algorithm>
#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/cuda_device_array_gpu.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;
namespace {
template <typename T>
__global__ void DynamicStitchGPUKernel(
    const int slice_size, const int output_size,
    CudaDeviceArrayStruct<int32> indices_flat,
    CudaDeviceArrayStruct<const T*> data_slice_ptrs, T* output) {
  const T** data_ptrs = GetCudaDeviceArrayOnDevice(&data_slice_ptrs);
  int32* output_indices = GetCudaDeviceArrayOnDevice(&indices_flat);
  CUDA_1D_KERNEL_LOOP(thread_id, output_size) {
    const int slice_id = thread_id / slice_size;
    const int slice_offset = thread_id % slice_size;
    int32 output_id = output_indices[slice_id] * slice_size + slice_offset;
    output[output_id] = ldg(data_ptrs[slice_id] + slice_offset);
  }
}
}  // namespace
template <typename T>
struct DynamicStitchGPULaunch {
  static void Run(const GPUDevice& d, const int slice_size,
                  const int first_dim_size,
                  const CudaDeviceArrayStruct<int32>& indices_flat,
                  const CudaDeviceArrayStruct<const T*>& data_slice_ptrs,
                  T* output) {
    // TODO(MycChiu): Currently these launch settings are hard-coded and
    // roughly optimized for my local system, is there a better way to
    // automatically determine the optimal launch settings?
    const int block_size = 256;
    const int grid_size = 60;
    const int output_size = slice_size * first_dim_size;
    DynamicStitchGPUKernel<T><<<grid_size, block_size, 0>>>(
        slice_size, output_size, indices_flat, data_slice_ptrs, output);
  }
};

template struct DynamicStitchGPULaunch<float>;
template struct DynamicStitchGPULaunch<double>;
}  // tensorflow

#endif  // GOOGLE_CUDA
