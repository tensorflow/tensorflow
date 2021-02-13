/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

GENERATE_AND_REGISTER_BINARY_GPU_KERNEL(BitwiseAnd, i8, int8);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL(BitwiseAnd, i16, int16);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL(BitwiseAnd, i32, int32);
GENERATE_AND_REGISTER_BINARY_GPU_KERNEL(BitwiseAnd, i64, int64);

// TODO(b/172804967): Enable once fixed.
// GENERATE_AND_REGISTER_BINARY_GPU_KERNEL(BitwiseAnd, ui8, uint8);
// GENERATE_AND_REGISTER_BINARY_GPU_KERNEL(BitwiseAnd, ui16, uint16);
// GENERATE_AND_REGISTER_BINARY_GPU_KERNEL(BitwiseAnd, ui32, uint32);
// GENERATE_AND_REGISTER_BINARY_GPU_KERNEL(BitwiseAnd, ui64, uint64);

}  // namespace tensorflow
