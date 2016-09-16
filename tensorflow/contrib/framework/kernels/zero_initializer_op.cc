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

#include "tensorflow/contrib/framework/kernels/zero_initializer_op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
#define REGISTER_KERNELS(D, T) \
  REGISTER_KERNEL_BUILDER(Name("ZeroInitializer") \
      .Device(DEVICE_##D) \
      .TypeConstraint<T>("T"), \
      ZeroInitializerOp<T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

#if GOOGLE_CUDA
#define REGISTER_GPU_KERNELS(T) REGISTER_KERNELS(GPU, T);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#endif // GOOGLE_CUDA

#undef REGISTER_KERNELS
} // namespace tensorflow
