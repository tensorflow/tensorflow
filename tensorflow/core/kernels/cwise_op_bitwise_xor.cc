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

#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER6(BinaryOp, CPU, "BitwiseXor", functor::bitwise_xor, int8, int16, int32,
          int64, uint8, uint16);

#if TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNEL(TYPE)                                      \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("BitwiseXor").Device(DEVICE_SYCL).TypeConstraint<TYPE>("T"), \
      BinaryOp<SYCLDevice, functor::bitwise_xor<TYPE>>);
REGISTER_SYCL_KERNEL(int8);
REGISTER_SYCL_KERNEL(int16);
REGISTER_SYCL_KERNEL(int32);
REGISTER_SYCL_KERNEL(int64);
REGISTER_SYCL_KERNEL(uint8);
REGISTER_SYCL_KERNEL(uint16);
#undef REGISTER_SYCL_KERNEL

#endif  // TENSORFLOW_USE_SYCL

#if GOOGLE_CUDA
REGISTER6(BinaryOp, GPU, "BitwiseXor", functor::bitwise_xor, int8, int16, int32,
          int64, uint8, uint16);
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
