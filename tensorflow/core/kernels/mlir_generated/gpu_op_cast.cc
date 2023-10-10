/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/kernels/mlir_generated/base_gpu_op.h"

namespace tensorflow {

#define CURRY_TYPES(FN, arg0) \
  FN(arg0, DT_BOOL);          \
  FN(arg0, DT_INT8);          \
  FN(arg0, DT_INT16);         \
  FN(arg0, DT_INT32);         \
  FN(arg0, DT_INT64);         \
  FN(arg0, DT_UINT8);         \
  FN(arg0, DT_UINT16);        \
  FN(arg0, DT_UINT32);        \
  FN(arg0, DT_UINT64);        \
  FN(arg0, DT_HALF);          \
  FN(arg0, DT_FLOAT);         \
  FN(arg0, DT_DOUBLE);        \
  FN(arg0, DT_COMPLEX64);     \
  FN(arg0, DT_COMPLEX128)

#define GENERATE_AND_REGISTER_CAST_GPU(input_type, output_type)               \
  GENERATE_UNARY_GPU_KERNEL2(Cast, input_type, output_type)                   \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("Cast")                                                            \
          .TypeConstraint<typename EnumToDataType<input_type>::Type>("SrcT")  \
          .TypeConstraint<typename EnumToDataType<output_type>::Type>("DstT") \
          .Device(DEVICE_GPU),                                                \
      MLIR_OP(Cast, GPU, input_type, output_type))

CURRY_TYPES(GENERATE_AND_REGISTER_CAST_GPU, DT_BOOL)
CURRY_TYPES(GENERATE_AND_REGISTER_CAST_GPU, DT_INT8)
CURRY_TYPES(GENERATE_AND_REGISTER_CAST_GPU, DT_INT16)
CURRY_TYPES(GENERATE_AND_REGISTER_CAST_GPU, DT_INT32)
CURRY_TYPES(GENERATE_AND_REGISTER_CAST_GPU, DT_INT64)
CURRY_TYPES(GENERATE_AND_REGISTER_CAST_GPU, DT_UINT8)
CURRY_TYPES(GENERATE_AND_REGISTER_CAST_GPU, DT_UINT16)
CURRY_TYPES(GENERATE_AND_REGISTER_CAST_GPU, DT_UINT32)
CURRY_TYPES(GENERATE_AND_REGISTER_CAST_GPU, DT_UINT64)
CURRY_TYPES(GENERATE_AND_REGISTER_CAST_GPU, DT_HALF)
CURRY_TYPES(GENERATE_AND_REGISTER_CAST_GPU, DT_FLOAT)
CURRY_TYPES(GENERATE_AND_REGISTER_CAST_GPU, DT_DOUBLE)
CURRY_TYPES(GENERATE_AND_REGISTER_CAST_GPU, DT_COMPLEX64)
CURRY_TYPES(GENERATE_AND_REGISTER_CAST_GPU, DT_COMPLEX128)

#undef REGISTER_CAST_GPU
#undef CURRY_TYPES

}  // namespace tensorflow
