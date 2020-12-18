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
#include "tensorflow/core/kernels/mlir_generated/gpu_ops_base.h"

namespace tensorflow {

GENERATE_AND_REGISTER_BINARY_KERNEL(BitwiseOr, i8, DT_INT8, int8);
GENERATE_AND_REGISTER_BINARY_KERNEL(BitwiseOr, i16, DT_INT16, int16);
GENERATE_AND_REGISTER_BINARY_KERNEL(BitwiseOr, i32, DT_INT32, int32);
GENERATE_AND_REGISTER_BINARY_KERNEL(BitwiseOr, i64, DT_INT64, int64);

// TODO(b/172804967): Enable once fixed.
// GENERATE_AND_REGISTER_BINARY_KERNEL(BitwiseOr, ui8, DT_UINT8, uint8);
// GENERATE_AND_REGISTER_BINARY_KERNEL(BitwiseOr, ui16, DT_UINT16, uint16);
// GENERATE_AND_REGISTER_BINARY_KERNEL(BitwiseOr, ui32, DT_UINT32, uint32);
// GENERATE_AND_REGISTER_BINARY_KERNEL(BitwiseOr, ui64, DT_UINT64, uint64);

}  // namespace tensorflow
