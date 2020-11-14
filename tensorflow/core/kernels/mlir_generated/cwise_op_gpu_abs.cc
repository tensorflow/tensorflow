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

#include <string>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/mlir_generated/abs_f16_kernel.h"
#include "tensorflow/core/kernels/mlir_generated/abs_f32_kernel.h"
#include "tensorflow/core/kernels/mlir_generated/abs_f64_kernel.h"
#include "tensorflow/core/kernels/mlir_generated/abs_i32_kernel.h"
#include "tensorflow/core/kernels/mlir_generated/abs_i64_kernel.h"
#include "tensorflow/core/kernels/mlir_generated/cwise_op_gpu_base.h"

namespace tensorflow {
namespace {
GENERATE_OP_KERNEL_BASE(Abs);
}  // namespace

GENERATE_AND_REGISTER_UNARY_KERNEL(Abs, F16, Eigen::half);
GENERATE_AND_REGISTER_UNARY_KERNEL(Abs, F32, float);
GENERATE_AND_REGISTER_UNARY_KERNEL(Abs, F64, double);
GENERATE_AND_REGISTER_UNARY_KERNEL(Abs, I32, int32);
GENERATE_AND_REGISTER_UNARY_KERNEL(Abs, I64, int64);
}  // namespace tensorflow
