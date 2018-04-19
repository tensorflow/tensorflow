/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

// Contains structs and functions to be included in device code.

#ifndef TENSORFLOW_CORE_KERNELS_CUDA_DEVICE_ARRAY_GPU_H_
#define TENSORFLOW_CORE_KERNELS_CUDA_DEVICE_ARRAY_GPU_H_

#if GOOGLE_CUDA

namespace tensorflow {

static constexpr int kMaxInlineCudaPointers = 8;
// To decode on the device side, use GetCudaDeviceArrayOnDevice.
// To encode on the host side, use CudaDeviceArrayOnHost.
template <typename ValueType, int MaxInlineValues = 8>
struct CudaDeviceArrayStruct {
  int32 size;
  // used if size <= MaxInlineValues;
  ValueType inline_values[MaxInlineValues];
  ValueType* out_of_line_values = nullptr;  // used if size > MaxInlineValues;
};

template <typename ValueType, int MaxInlineValues = 8>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE ValueType* GetCudaDeviceArrayOnDevice(
    CudaDeviceArrayStruct<ValueType, MaxInlineValues>* data) {
  if (data->size <= MaxInlineValues) {
    return data->inline_values;
  } else {
    return data->out_of_line_values;
  }
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CORE_KERNELS_CUDA_DEVICE_ARRAY_GPU_H_
