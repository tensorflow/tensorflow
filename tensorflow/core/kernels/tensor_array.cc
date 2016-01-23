/* Copyright 2015 Google Inc. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/tensor_array.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/aggregate_ops_cpu.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace tensor_array {

#define TENSOR_ARRAY_WRITE_OR_ADD(Device, T)                                   \
  template <>                                                                  \
  Status TensorArrayWriteOrAdd<T, Device>(OpKernelContext * ctx, Tensor * sum, \
                                          const Tensor* current,               \
                                          const Tensor* add) {                 \
    functor::Add2Functor<Device, T> add_functor;                               \
    add_functor(ctx->template eigen_device<Device>(), sum->flat<T>(),          \
                current->flat<T>(), add->flat<T>());                           \
    return Status::OK();                                                       \
  }

#define TENSOR_ARRAY_WRITE_OR_ADD_CPU(T) TENSOR_ARRAY_WRITE_OR_ADD(CPUDevice, T)
TF_CALL_NUMBER_TYPES(TENSOR_ARRAY_WRITE_OR_ADD_CPU)
#undef TENSOR_ARRAY_WRITE_OR_ADD_CPU

#if GOOGLE_CUDA

#define TENSOR_ARRAY_WRITE_OR_ADD_GPU(T) TENSOR_ARRAY_WRITE_OR_ADD(GPUDevice, T)
TF_CALL_GPU_NUMBER_TYPES(TENSOR_ARRAY_WRITE_OR_ADD_GPU);
#undef TENSOR_ARRAY_WRITE_OR_ADD_GPU

#endif  // GOOGLE_CUDA

#undef TENSOR_ARRAY_WRITE_OR_ADD

}  // namespace tensor_array

}  // namespace tensorflow
