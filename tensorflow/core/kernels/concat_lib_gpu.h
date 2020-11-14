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

#ifndef TENSORFLOW_CORE_KERNELS_CONCAT_LIB_GPU_H_
#define TENSORFLOW_CORE_KERNELS_CONCAT_LIB_GPU_H_

#define EIGEN_USE_THREADS
#define EIGEN_USE_GPU

#include <memory>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/concat_lib.h"
#include "tensorflow/core/kernels/gpu_device_array_gpu.h"

namespace tensorflow {

template <typename T, typename IntType>
void ConcatGPUSlice(
    const Eigen::GpuDevice& gpu_device,
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        inputs_flat,
    typename TTypes<T, 2>::Matrix* output);

template <typename T, typename IntType>
void ConcatGPUImpl(const Eigen::GpuDevice& d,
                   const GpuDeviceArrayStruct<const T*>& input_ptrs,
                   const GpuDeviceArrayStruct<IntType>& ptr_offsets,
                   bool same_size, int slice_size,
                   typename TTypes<T, 2>::Matrix* output);

// Explicit instantiations in concat_lib_gpu_impl.cu.cc.
#define REGISTER(T)                                                           \
  extern template void ConcatGPUSlice<T, int32>(                              \
      const Eigen::GpuDevice& gpu_device,                                     \
      const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>& \
          inputs_flat,                                                        \
      typename TTypes<T, 2>::Matrix* output);                                 \
  extern template void ConcatGPUSlice<T, int64>(                              \
      const Eigen::GpuDevice& gpu_device,                                     \
      const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>& \
          inputs_flat,                                                        \
      typename TTypes<T, 2>::Matrix* output);                                 \
  extern template void ConcatGPUImpl<T, int32>(                               \
      const Eigen::GpuDevice& d,                                              \
      const GpuDeviceArrayStruct<const T*>& input_ptrs,                       \
      const GpuDeviceArrayStruct<int32>& ptr_offsets, bool fixed_size,        \
      int split_size, typename TTypes<T, 2>::Matrix* output);                 \
  extern template void ConcatGPUImpl<T, int64>(                               \
      const Eigen::GpuDevice& d,                                              \
      const GpuDeviceArrayStruct<const T*>& input_ptrs,                       \
      const GpuDeviceArrayStruct<int64>& ptr_offsets, bool fixed_size,        \
      int split_size, typename TTypes<T, 2>::Matrix* output);

TF_CALL_INTEGRAL_TYPES(REGISTER);  // int32 Needed for TensorLists.
TF_CALL_bfloat16(REGISTER);
TF_CALL_GPU_ALL_TYPES(REGISTER);
#undef REGISTER

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CONCAT_LIB_GPU_H_
