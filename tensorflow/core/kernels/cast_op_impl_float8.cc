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

#include "tensorflow/core/kernels/cast_op_impl.h"

#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

CastFunctorType GetCpuCastFromFloat8e5m2(DataType dst_dtype) {
  CURRY_TYPES3(CAST_CASE, CPUDevice, float8_e5m2);
  CAST_CASE(CPUDevice, float8_e5m2, float8_e5m2);
  CAST_CASE(CPUDevice, float8_e5m2, float8_e4m3fn);
  return nullptr;
}

CastFunctorType GetCpuCastFromFloat8e4m3fn(DataType dst_dtype) {
  CURRY_TYPES3(CAST_CASE, CPUDevice, float8_e4m3fn);
  CAST_CASE(CPUDevice, float8_e4m3fn, float8_e5m2);
  CAST_CASE(CPUDevice, float8_e4m3fn, float8_e4m3fn);
  return nullptr;
}

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
CastFunctorType GetGpuCastFromFloat8e5m2(DataType dst_dtype) {
  CAST_CASE(GPUDevice, float8_e5m2, float);
  CAST_CASE(GPUDevice, float8_e5m2, bfloat16);
  CAST_CASE(GPUDevice, float8_e5m2, Eigen::half);
  CAST_CASE(GPUDevice, float8_e5m2, float8_e5m2);
  CAST_CASE(GPUDevice, float8_e5m2, float8_e4m3fn);
  return nullptr;
}

CastFunctorType GetGpuCastFromFloat8e4m3fn(DataType dst_dtype) {
  CAST_CASE(GPUDevice, float8_e4m3fn, float);
  CAST_CASE(GPUDevice, float8_e4m3fn, bfloat16);
  CAST_CASE(GPUDevice, float8_e4m3fn, Eigen::half);
  CAST_CASE(GPUDevice, float8_e4m3fn, float8_e5m2);
  CAST_CASE(GPUDevice, float8_e4m3fn, float8_e4m3fn);
  return nullptr;
}

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
