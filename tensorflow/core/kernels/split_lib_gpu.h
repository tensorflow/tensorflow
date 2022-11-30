/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_SPLIT_LIB_GPU_H_
#define TENSORFLOW_CORE_KERNELS_SPLIT_LIB_GPU_H_

#define EIGEN_USE_THREADS
#define EIGEN_USE_GPU

#include <memory>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/gpu_device_array_gpu.h"
#include "tensorflow/core/kernels/split_lib.h"

namespace tensorflow {

template <typename T>
struct SplitOpGPULaunch {
  void Run(const Eigen::GpuDevice& d, const T* input, int32_t prefix_dim_size,
           int32_t split_dim_size, int32_t suffix_dim_size,
           const GpuDeviceArrayStruct<T*>& output_ptr_data);
};

template <typename T, typename IntType>
struct SplitVOpGPULaunch {
  void Run(const Eigen::GpuDevice& d, bool fixed, const T* input,
           int total_cols, int total_rows,
           const GpuDeviceArrayStruct<IntType>& output_scan,
           const GpuDeviceArrayStruct<T*>& output_ptr_data);
};

// Explicit instantiations in split_lib_gpu.cu.cc.
#define REGISTER_GPU_KERNEL(T)                        \
  extern template struct SplitOpGPULaunch<T>;         \
  extern template struct SplitVOpGPULaunch<T, int8>;  \
  extern template struct SplitVOpGPULaunch<T, int32>; \
  extern template struct SplitVOpGPULaunch<T, int64_t>;

TF_CALL_bfloat16(REGISTER_GPU_KERNEL);
TF_CALL_uint8(REGISTER_GPU_KERNEL);
TF_CALL_GPU_ALL_TYPES(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SPLIT_LIB_GPU_H_
