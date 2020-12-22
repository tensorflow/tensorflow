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

GENERATE_AND_REGISTER_BINARY_KERNEL(Mul, f16, DT_HALF, Eigen::half);
GENERATE_AND_REGISTER_BINARY_KERNEL(Mul, f32, DT_FLOAT, float);
GENERATE_AND_REGISTER_BINARY_KERNEL(Mul, f64, DT_DOUBLE, double);
GENERATE_AND_REGISTER_BINARY_KERNEL(Mul, i8, DT_INT8, int8);
// TODO(b/25387198): We cannot use a regular GPU kernel for int32.
GENERATE_AND_REGISTER_BINARY_KERNEL(Mul, i16, DT_INT16, int16);
GENERATE_AND_REGISTER_BINARY_KERNEL(Mul, i64, DT_INT64, int64);

}  // namespace tensorflow
