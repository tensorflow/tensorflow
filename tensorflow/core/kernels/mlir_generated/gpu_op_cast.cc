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

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/mlir_generated/base_gpu_op.h"

namespace tensorflow {

#define CURRY_TYPES(FN, arg0, arg1) \
  FN(arg0, i1, bool, arg1);         \
  FN(arg0, i8, int8, arg1);         \
  FN(arg0, i16, int16, arg1);       \
  FN(arg0, i32, int32, arg1);       \
  FN(arg0, i64, int64, arg1);       \
  FN(arg0, f16, Eigen::half, arg1); \
  FN(arg0, f32, float, arg1);       \
  FN(arg0, f64, double, arg1)

#define GENERATE_AND_REGISTER_CAST_GPU(mlir_input_type, mlir_output_type, \
                                       result_data_type, input_data_type) \
  GENERATE_UNARY_GPU_KERNEL2(Cast, mlir_input_type, mlir_output_type,     \
                             result_data_type, input_data_type);          \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Cast")                                                        \
          .TypeConstraint<input_data_type>("SrcT")                        \
          .TypeConstraint<result_data_type>("DstT")                       \
          .Device(DEVICE_GPU),                                            \
      MLIR_OP(Cast, GPU, mlir_input_type, mlir_output_type))

CURRY_TYPES(GENERATE_AND_REGISTER_CAST_GPU, i1, bool)
CURRY_TYPES(GENERATE_AND_REGISTER_CAST_GPU, i8, int8)
CURRY_TYPES(GENERATE_AND_REGISTER_CAST_GPU, i16, int16)
CURRY_TYPES(GENERATE_AND_REGISTER_CAST_GPU, i32, int32)
CURRY_TYPES(GENERATE_AND_REGISTER_CAST_GPU, i64, int64)
CURRY_TYPES(GENERATE_AND_REGISTER_CAST_GPU, f16, Eigen::half)
CURRY_TYPES(GENERATE_AND_REGISTER_CAST_GPU, f32, float)
CURRY_TYPES(GENERATE_AND_REGISTER_CAST_GPU, f64, double)

#undef REGISTER_CAST_GPU
#undef CURRY_TYPES

}  // namespace tensorflow
