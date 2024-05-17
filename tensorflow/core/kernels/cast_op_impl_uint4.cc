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

#include <cstdint>

#define EIGEN_USE_THREADS

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/kernels/cast_op.h"
#include "tensorflow/core/kernels/cast_op_impl.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

CastFunctorType GetCpuCastFromUint4(DataType dst_dtype) {
  // Only allow casts to integer types.
  CAST_CASE(CPUDevice, uint4, int4);
  CAST_CASE(CPUDevice, uint4, int8);
  CAST_CASE(CPUDevice, uint4, int16);
  CAST_CASE(CPUDevice, uint4, int32);
  CAST_CASE(CPUDevice, uint4, int64_t);
  CAST_CASE(CPUDevice, uint4, uint4);
  CAST_CASE(CPUDevice, uint4, uint8);
  CAST_CASE(CPUDevice, uint4, uint16);
  CAST_CASE(CPUDevice, uint4, uint32);
  CAST_CASE(CPUDevice, uint4, uint64_t);
  return nullptr;
}

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
CastFunctorType GetGpuCastFromUint4(DataType dst_dtype) {
  // Only allow casts to integer types.
  CAST_CASE(GPUDevice, uint4, int4);
  CAST_CASE(GPUDevice, uint4, int8);
  CAST_CASE(GPUDevice, uint4, int16);
  CAST_CASE(GPUDevice, uint4, int32);
  CAST_CASE(GPUDevice, uint4, int64_t);
  CAST_CASE(GPUDevice, uint4, uint4);
  CAST_CASE(GPUDevice, uint4, uint8);
  CAST_CASE(GPUDevice, uint4, uint16);
  CAST_CASE(GPUDevice, uint4, uint32);
  CAST_CASE(GPUDevice, uint4, uint64_t);
  return nullptr;
}
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
